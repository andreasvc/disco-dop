"""Extract recurring tree fragments from constituency treebanks.

NB: there is a known bug in multiprocessing which makes it impossible to detect
Ctrl-C or fatal errors like segmentation faults in child processes which causes
the master program to wait forever for output from its children. Therefore if
you want to abort, kill the program manually (e.g., press Ctrl-Z and issue
'kill %1'). If the program seems stuck, re-run without multiprocessing
(pass --numproc 1) to see if there might be a bug."""

from __future__ import division, print_function
import io
import os
import re
import sys
import codecs
import logging
if sys.version[0] > '2':
	imap = map
else:
	from itertools import imap
from multiprocessing import Pool, cpu_count, log_to_stderr, SUBDEBUG
from collections import defaultdict
from itertools import count
from getopt import gnu_getopt, GetoptError
from discodop.tree import Tree
from discodop.treebank import READERS
from discodop.treetransforms import binarize, introducepreterminals, unbinarize
from discodop._fragments import readtreebank, getctrees, \
		extractfragments, exactcounts, \
		completebitsets, coverbitsets
from discodop.parser import workerfunc

USAGE = '''\
Usage: %(cmd)s <treebank1> [treebank2] [options]
  or: %(cmd)s --batch=<dir> <treebank1> <treebank2>... [options]
If only one treebank is given, fragments occurring at least twice are sought.
If two treebanks are given, finds common fragments between first & second.
Input is in Penn treebank format (S-expressions), one tree per line.
Output contains lines of the form "tree<TAB>frequency".
Frequencies refer to the first treebank by default.
Output is sent to stdout; to save the results, redirect to a file.
Options:
  --fmt=[%(fmts)s]
                when format is not 'bracket', work with discontinuous trees;
                output is in 'discbracket' format:
                tree<TAB>sentence<TAB>frequency
                where 'tree' has indices as leaves, referring to elements of
                'sentence', a space separated list of words.
  -o file       Write output to 'file' instead of stdout.
  --indices     report sets of indices instead of frequencies.
  --cover       include all depth-1 fragments of first treebank corresponding
                to single productions.
  --complete    find complete matches of fragments from treebank1 (needle) in
                treebank2 (haystack); frequencies are from haystack.
  --batch=dir   enable batch mode; any number of treebanks > 1 can be given;
                first treebank will be compared to all others.
                Results are written to filenames of the form dir/A_B.
  --numproc=n   use n independent processes, to enable multi-core usage
                (default: 1); use 0 to detect the number of CPUs.
  --numtrees=n  only read first n trees from first treebank
  --encoding=x  use x as treebank encoding, e.g. UTF-8, ISO-8859-1, etc.
  --nofreq      do not report frequencies.
  --approx      report counts of occurrence as maximal fragment (lower bound)
  --relfreq     report relative frequencies wrt. root node of fragments.
  --debin       debinarize fragments.
  --twoterms    only consider fragments with at least two lexical terminals.
  --adjacent    only consider pairs of adjacent fragments (n, n + 1).
  --alt         alternative output format: (NP (DT "a") NN)
                default: (NP (DT a) (NN ))
  --debug       extra debug information, ignored when numproc > 1.
  --quiet       disable all messages.\
''' % dict(cmd=sys.argv[0], fmts='|'.join(READERS))

FLAGS = ('approx', 'indices', 'nofreq', 'complete', 'complement', 'cover',
		'alt', 'relfreq', 'twoterms', 'adjacent', 'debin', 'debug', 'quiet')
OPTIONS = ('fmt=', 'numproc=', 'numtrees=', 'encoding=', 'batch=')
PARAMS = {}
FRONTIERRE = re.compile(r"\(([^ ()]+) \)")
TERMRE = re.compile(r"\(([^ ()]+) ([^ ()]+)\)")
APPLY = lambda x, _y: x()


def main(argv=None):
	"""Command line interface to fragment extraction."""
	if argv is None:
		argv = sys.argv
	try:
		opts, args = gnu_getopt(argv[1:], 'o:', FLAGS + OPTIONS)
	except GetoptError as err:
		print("%s\n%s" % (err, USAGE))
		return
	opts = dict(opts)

	for flag in FLAGS:
		PARAMS[flag] = '--' + flag in opts
	PARAMS['disc'] = opts.get('--fmt', 'bracket') != 'bracket'
	PARAMS['fmt'] = opts.get('--fmt', 'bracket')
	numproc = int(opts.get("--numproc", 1))
	if numproc == 0:
		numproc = cpu_count()
	limit = int(opts.get('--numtrees', 0)) or None
	encoding = opts.get("--encoding", "UTF-8")
	batchdir = opts.get("--batch")

	if len(args) < 1:
		print("missing treebank argument")
	if batchdir is None and len(args) not in (1, 2):
		print("incorrect number of arguments:", args)
		print(USAGE)
		return
	if batchdir:
		assert numproc == 1, ("Batch mode only supported in single-process "
				"mode. Use the xargs command for multi-processing.")
	if args[0] == '-':
		args[0] = '/dev/stdin'
	for a in args:
		assert os.path.exists(a), "not found: %r" % a
	if PARAMS['complete']:
		assert not PARAMS['twoterms'] and not PARAMS['adjacent'], (
				'--twoterms and --adjacent are incompatible with --complete.')
	if PARAMS['complete']:
		assert len(args) == 2 or batchdir, (
				"need at least two treebanks with --complete.")
		assert not PARAMS['approx'] and not PARAMS['nofreq'], (
				"--complete is incompatible with --nofreq and --approx")

	level = logging.WARNING if PARAMS['quiet'] else logging.DEBUG
	logging.basicConfig(level=level, format='%(message)s')
	if PARAMS['debug'] and numproc > 1:
		logger = log_to_stderr()
		logger.setLevel(SUBDEBUG)

	logging.info("Disco-DOP Fragment Extractor")

	assert numproc
	logging.info("parameters:\n%s", "\n".join("    %s:\t%r" % kv
		for kv in sorted(PARAMS.items())))
	logging.info("\n".join("treebank%d: %s" % (n + 1, a)
		for n, a in enumerate(args)))

	if numproc == 1 and batchdir:
		batch(batchdir, args, limit, encoding, '--debin' in opts)
	else:
		fragmentkeys, counts = regular(args, numproc, limit, encoding)
		out = (io.open(opts['-o'], 'w', encoding=encoding)
				if '-o' in opts else None)
		if '--debin' in opts:
			fragmentkeys = debinarize(fragmentkeys)
		printfragments(fragmentkeys, counts, out=out)


def regular(filenames, numproc, limit, encoding):
	"""non-batch processing. multiprocessing optional."""
	mult = 1
	if PARAMS['approx']:
		fragments = defaultdict(int)
	else:
		fragments = {}
	# detect corpus reading errors in this process (e.g., wrong encoding)
	initworker(filenames[0], filenames[1] if len(filenames) == 2 else None,
			limit, encoding)
	if numproc == 1:
		mymap = imap
		myapply = APPLY
	else:  # multiprocessing, start worker processes
		pool = Pool(processes=numproc, initializer=initworker,
			initargs=(filenames[0],
				filenames[1] if len(filenames) == 2 else None, limit, encoding))
		mymap = pool.imap
		myapply = pool.apply
	numtrees = (PARAMS['trees1'].len if limit is None
			else min(PARAMS['trees1'].len, limit))

	if PARAMS['complete']:
		trees1, trees2 = PARAMS['trees1'], PARAMS['trees2']
		fragments = completebitsets(trees1, PARAMS['sents1'], PARAMS['labels'],
				max(trees1.maxnodes, (trees2 or trees1).maxnodes),
				PARAMS['disc'])
	else:
		if len(filenames) == 1:
			work = workload(numtrees, mult, numproc)
		else:
			chunk = numtrees // (mult * numproc) + 1
			work = [(a, a + chunk) for a in range(0, numtrees, chunk)]
		if numproc != 1:
			logging.info("work division:\n%s", "\n".join("    %s:\t%r" % kv
				for kv in sorted(dict(numchunks=len(work), mult=mult).items())))
		dowork = mymap(worker, work)
		for n, results in enumerate(dowork):
			if PARAMS['approx']:
				for frag, x in results.items():
					fragments[frag] += x
			else:
				fragments.update(results)
	if PARAMS['cover']:
		cover = myapply(coverfragworker, ())
		if PARAMS['approx']:
			fragments.update(zip(cover,
					exactcounts(PARAMS['trees1'], PARAMS['trees1'],
					cover.values())))
		else:
			fragments.update(cover)
		logging.info("merged %d cover fragments", len(cover))
	fragmentkeys = list(fragments)
	if PARAMS['nofreq']:
		counts = None
	elif PARAMS['approx']:
		counts = [fragments[a] for a in fragmentkeys]
	else:
		task = "indices" if PARAMS['indices'] else "counts"
		logging.info("dividing work for exact %s", task)
		bitsets = [fragments[a] for a in fragmentkeys]
		countchunk = len(bitsets) // numproc + 1
		work = list(range(0, len(bitsets), countchunk))
		work = [(n, len(work), bitsets[a:a + countchunk])
				for n, a in enumerate(work)]
		counts = []
		logging.info("getting exact %s", task)
		for a in mymap(exactcountworker, work):
			counts.extend(a)
	if numproc != 1:
		pool.close()
		pool.join()
		del dowork, pool
	return fragmentkeys, counts


def batch(outputdir, filenames, limit, encoding, debin):
	"""batch processing: three or more treebanks specified.
	The use case for this is when you have one big treebank which you want to
	compare to lots of smaller sets of trees, and get the results for each
	comparison in a separate file."""
	initworker(filenames[0], None, limit, encoding)
	trees1 = PARAMS['trees1']
	sents1 = PARAMS['sents1']
	if PARAMS['approx']:
		fragments = defaultdict(int)
	else:
		fragments = {}
	for filename in filenames[1:]:
		PARAMS.update(read2ndtreebank(filename, PARAMS['labels'],
			PARAMS['prods'], PARAMS['fmt'], limit, encoding))
		trees2 = PARAMS['trees2']
		sents2 = PARAMS['sents2']
		if PARAMS['complete']:
			fragments = completebitsets(trees2, sents2, PARAMS['labels'],
					max(trees1.maxnodes, (trees2 or trees1).maxnodes),
					PARAMS['disc'])
		else:
			fragments = extractfragments(trees2, sents2, 0, 0,
					PARAMS['labels'], trees1, sents1,
					discontinuous=PARAMS['disc'], debug=PARAMS['debug'],
					approx=PARAMS['approx'],
					twoterms=PARAMS['twoterms'],
					adjacent=PARAMS['adjacent'])
		counts = None
		fragmentkeys = list(fragments)
		if PARAMS['approx'] or not fragments:
			counts = fragments.values()
		elif not PARAMS['nofreq']:
			bitsets = [fragments[a] for a in fragmentkeys]
			logging.info('getting %s for %d fragments',
					'indices of occurrence' if PARAMS['indices']
					else 'exact counts', len(bitsets))
			counts = exactcounts(trees2, trees1, bitsets,
					indices=PARAMS['indices'])
		outputfilename = '%s/%s_%s' % (outputdir,
				os.path.basename(filenames[0]), os.path.basename(filename))
		out = io.open(outputfilename, 'w', encoding=encoding)
		if debin:
			fragmentkeys = debinarize(fragmentkeys)
		printfragments(fragmentkeys, counts, out=out)
		logging.info("wrote to %s", outputfilename)


def readtreebanks(treebank1, treebank2=None, fmt='bracket',
		limit=None, encoding='utf-8'):
	"""Read one or two treebanks."""
	labels = []
	prods = {}
	trees1, sents1 = readtreebank(treebank1, labels, prods,
			fmt, limit, encoding)
	trees2, sents2 = readtreebank(treebank2, labels, prods,
			fmt, limit, encoding)
	trees1.indextrees(prods)
	if trees2:
		trees2.indextrees(prods)
	return dict(trees1=trees1, sents1=sents1, trees2=trees2, sents2=sents2,
			prods=prods, labels=labels)


def read2ndtreebank(treebank2, labels, prods, fmt='bracket',
		limit=None, encoding='utf-8'):
	"""Read a second treebank."""
	trees2, sents2 = readtreebank(treebank2, labels, prods,
			fmt, limit, encoding)
	logging.info("%r: %d trees; %d nodes (max %d). "
			"labels: %d, prods: %d",
			treebank2, len(trees2), trees2.numnodes, trees2.maxnodes,
			len(set(PARAMS['labels'])), len(PARAMS['prods']))
	return dict(trees2=trees2, sents2=sents2, prods=prods, labels=labels)


def initworker(treebank1, treebank2, limit, encoding):
	"""Read treebanks for this worker.

	We do this separately for each process under the assumption that this is
	advantageous with a NUMA architecture."""
	PARAMS.update(readtreebanks(treebank1, treebank2,
			limit=limit, fmt=PARAMS['fmt'], encoding=encoding))
	if PARAMS['debug']:
		print("\nproductions:")
		for a, b in sorted(PARAMS['prods'].items(), key=lambda x: x[1]):
			print(b, *a)
	trees1 = PARAMS['trees1']
	assert trees1
	m = "treebank1: %d trees; %d nodes (max: %d);" % (
			trees1.len, trees1.numnodes, trees1.maxnodes)
	if treebank2:
		trees2 = PARAMS['trees2']
		assert trees2
		m += " treebank2: %d trees; %d nodes (max %d);" % (
				trees2.len, trees2.numnodes, trees2.maxnodes)
	logging.info("%s labels: %d, prods: %d", m, len(set(PARAMS['labels'])),
			len(PARAMS['prods']))


def initworkersimple(trees, sents, disc, trees2=None, sents2=None):
	"""Initialization for a worker in which a treebank was already loaded."""
	PARAMS.update(getctrees(trees, sents, disc, trees2, sents2))
	assert PARAMS['trees1']


@workerfunc
def worker(interval):
	"""Worker function for fragment extraction."""
	offset, end = interval
	trees1 = PARAMS['trees1']
	trees2 = PARAMS['trees2']
	sents1 = PARAMS['sents1']
	sents2 = PARAMS['sents2']
	assert offset < trees1.len
	result = {}
	result = extractfragments(trees1, sents1, offset, end,
			PARAMS['labels'], trees2, sents2, approx=PARAMS['approx'],
			discontinuous=PARAMS['disc'], complement=PARAMS['complement'],
			debug=PARAMS['debug'], twoterms=PARAMS['twoterms'],
			adjacent=PARAMS['adjacent'])
	logging.debug("finished %d--%d", offset, end)
	return result


@workerfunc
def exactcountworker(args):
	"""Worker function for counting of fragments."""
	n, m, bitsets = args
	trees1 = PARAMS['trees1']
	if PARAMS['complete']:
		results = exactcounts(trees1, PARAMS['trees2'], bitsets,
				indices=PARAMS['indices'])
		logging.debug("complete matches %d of %d", n + 1, m)
		return results
	results = exactcounts(trees1, trees1, bitsets, indices=PARAMS['indices'])
	if PARAMS['indices']:
		logging.debug("exact indices %d of %d", n + 1, m)
	else:
		logging.debug("exact counts %d of %d", n + 1, m)
	return results


@workerfunc
def coverfragworker():
	"""Worker function that gets depth-1 fragments.

	Does not need multiprocessing but using it avoids reading the treebank
	again."""
	# TODO: expand this to extract all depth-n fragments for arbitrary n.
	trees1 = PARAMS['trees1']
	trees2 = PARAMS['trees2']
	return coverbitsets(trees1, PARAMS['sents1'], PARAMS['labels'],
			max(trees1.maxnodes, (trees2 or trees1).maxnodes),
			PARAMS['disc'])


def workload(numtrees, mult, numproc):
	"""Calculate an even workload.

	When *n* trees are compared against themselves, ``n * (n - 1)`` total
	comparisons are made. Each tree *m* has to be compared to all trees *x*
	such that ``m < x <= n``
	(meaning there are more comparisons for lower *n*).

	:returns: a sequence of ``(start, end)`` intervals such that
		the number of comparisons is approximately balanced."""
	# could base on number of nodes as well.
	if numproc == 1:
		return [(0, numtrees)]
	# here chunk is the number of tree pairs that will be compared
	goal = togo = total = 0.5 * numtrees * (numtrees - 1)
	chunk = total // (mult * numproc) + 1
	goal -= chunk
	result = []
	last = 0
	for n in range(1, numtrees):
		togo -= numtrees - n
		if togo <= goal:
			goal -= chunk
			result.append((last, n))
			last = n
	if last < numtrees:
		result.append((last, numtrees))
	return result


def getfragments(trees, sents, numproc=1, disc=True,
		iterate=False, complement=False, indices=True, cover=True):
	"""Get recurring fragments with exact counts in a single treebank.

	:returns: a dictionary whose keys are fragments as strings, and
		indices as values. When ``disc`` is ``True``, keys are of the form
		``(frag, sent)`` where ``frag`` is a unicode string, and ``sent``
		is a list of words as unicode strings; when ``disc`` is ``False``, keys
		are of the form ``frag`` where ``frag`` is a unicode string.
	:param trees: a sequence of binarized Tree objects.
	:param numproc: number of processes to use; pass 0 to use detected # CPUs.
	:param disc: when disc=True, assume trees with discontinuous constituents.
	:param iterate, complement: see :func:`_fragments.extractfragments`"""
	if numproc == 0:
		numproc = cpu_count()
	numtrees = len(trees)
	assert numtrees
	mult = 1  # 3 if numproc > 1 else 1
	fragments = {}
	trees = trees[:]
	work = workload(numtrees, mult, numproc)
	PARAMS.update(disc=disc, indices=indices, approx=False, complete=False,
			complement=complement, debug=False, adjacent=False, twoterms=False)
	initworkersimple(trees, list(sents), disc)
	if numproc == 1:
		mymap = map
		myapply = APPLY
	else:
		logging.info("work division:\n%s", "\n".join("    %s: %r" % kv
				for kv in sorted(dict(numchunks=len(work),
					numproc=numproc).items())))
		# start worker processes
		pool = Pool(processes=numproc, initializer=initworkersimple,
				initargs=(trees, list(sents), disc))
		mymap = pool.map
		myapply = pool.apply
	# collect recurring fragments
	logging.info("extracting recurring fragments")
	for a in mymap(worker, work):
		fragments.update(a)
	# add 'cover' fragments corresponding to single productions
	if cover:
		cover = myapply(coverfragworker, ())
		before = len(fragments)
		fragments.update(cover)
		logging.info("merged %d unseen cover fragments", len(fragments) - before)
	fragmentkeys = list(fragments)
	bitsets = [fragments[a] for a in fragmentkeys]
	countchunk = len(bitsets) // numproc + 1
	work = list(range(0, len(bitsets), countchunk))
	work = [(n, len(work), bitsets[a:a + countchunk])
			for n, a in enumerate(work)]
	logging.info("getting exact counts for %d fragments", len(bitsets))
	counts = []
	for a in mymap(exactcountworker, work):
		counts.extend(a)
	if numproc != 1:
		pool.close()
		pool.join()
		del pool
	if iterate:  # optionally collect fragments of fragments
		logging.info("extracting fragments of recurring fragments")
		PARAMS['complement'] = False  # needs to be turned off if it was on
		newfrags = fragments
		trees, sents = None, None
		ids = count()
		for _ in range(10):  # up to 10 iterations
			newtrees = [binarize(
					introducepreterminals(Tree.parse(tree, parse_leaf=int),
					ids=ids), childchar="}") for tree, _ in newfrags]
			newsents = [["#%d" % next(ids) if word is None else word
					for word in sent] for _, sent in newfrags]
			newfrags, newcounts = iteratefragments(
					fragments, newtrees, newsents, trees, sents, numproc)
			if len(newfrags) == 0:
				break
			if trees is None:
				trees = []
				sents = []
			trees.extend(newtrees)
			sents.extend(newsents)
			fragmentkeys.extend(newfrags)
			counts.extend(newcounts)
			fragments.update(zip(newfrags, newcounts))
	logging.info("found %d fragments", len(fragmentkeys))
	if not disc:
		return {a.decode('utf-8'): b for a, b in zip(fragmentkeys, counts)}
	return {(a.decode('utf-8'), b): c
			for (a, b), c in zip(fragmentkeys, counts)}


def iteratefragments(fragments, newtrees, newsents, trees, sents, numproc):
	"""Get fragments of fragments."""
	numtrees = len(newtrees)
	assert numtrees
	if numproc == 1:  # set fragments as input
		initworkersimple(newtrees, newsents, PARAMS['disc'], trees, sents)
		mymap = map
	else:
		# since the input trees change, we need a new pool each time
		pool = Pool(processes=numproc, initializer=initworkersimple,
				initargs=(newtrees, newsents, PARAMS['disc'], trees, sents))
		mymap = pool.imap
	newfragments = {}
	for a in mymap(worker, workload(numtrees, 1, numproc)):
		newfragments.update(a)
	logging.info("before: %d, after: %d, difference: %d",
		len(fragments), len(set(fragments) | set(newfragments)),
		len(set(newfragments) - set(fragments)))
	# we have to get counts for these separately because they're not coming
	# from the same set of trees
	newkeys = list(set(newfragments) - set(fragments))
	bitsets = [newfragments[a] for a in newkeys]
	countchunk = len(bitsets) // numproc + 1
	if countchunk == 0:
		return newkeys, []
	work = list(range(0, len(bitsets), countchunk))
	work = [(n, len(work), bitsets[a:a + countchunk])
			for n, a in enumerate(work)]
	logging.info("getting exact counts for %d fragments", len(bitsets))
	counts = []
	for a in mymap(exactcountworker, work):
		counts.extend(a)
	if numproc != 1:
		pool.close()
		pool.join()
		del pool
	return newkeys, counts


def altrepr(a):
	"""Rewrite bracketed tree to alternative format.

	Replace double quotes with double single quotes: " -> ''
	Quote terminals with double quotes terminal: -> "terminal"
	Remove parentheses around frontier nodes: (NN ) -> NN

	>>> altrepr("(NP (DT a) (NN ))")
	'(NP (DT "a") NN)'
	"""
	return FRONTIERRE.sub(r'\1', TERMRE.sub(r'(\1 "\2")', a.replace('"', "''")))


def debinarize(fragments):
	"""Debinarize fragments; fragments that fail to debinarize left as-is."""
	result = []
	for origfrag in fragments:
		if PARAMS['disc']:
			frag, sent = origfrag
		else:
			frag = origfrag
		try:
			frag = str(unbinarize(Tree(frag)))
		except:
			result.append(origfrag)
		else:
			result.append((frag, sent) if PARAMS['disc'] else frag)
	return result


def printfragments(fragments, counts, out=None):
	"""Dump fragments to standard output or some other file object."""
	if out is None:
		out = sys.stdout
		if sys.stdout.encoding is None:
			out = codecs.getwriter('utf-8')(out)
	if PARAMS['alt']:
		for n, a in enumerate(fragments):
			fragments[n] = altrepr(a)
	if PARAMS['complete']:
		logging.info("total number of matches: %d", sum(counts))
	else:
		logging.info("number of fragments: %d", len(fragments))
	if PARAMS['nofreq']:
		for a in fragments:
			out.write("%s\n" % (("%s\t%s" % (a[0].decode('ascii'),
					' '.join("%s" % x if x else '' for x in a[1])))
					if PARAMS['disc'] else a))
		return
	# a frequency of 0 is normal when counting occurrences of given fragments
	# in a second treebank
	if PARAMS['complete']:
		threshold = 0
		zeroinvalid = False
	# a frequency of 1 is normal when comparing two treebanks
	# or when non-recurring fragments are added
	elif (PARAMS.get('trees2') or PARAMS['cover']
			or PARAMS['complement'] or PARAMS['approx']):
		threshold = 0
		zeroinvalid = True
	else:  # otherwise, raise alarm.
		threshold = 1
		zeroinvalid = True
	if PARAMS['indices']:
		for a, theindices in zip(fragments, counts):
			if len(theindices) > threshold:
				out.write("%s\t%r\n" % (("%s\t%s" % (a[0].decode('ascii'),
					' '.join("%s" % x if x else '' for x in a[1])))
					if PARAMS['disc'] else a.decode('utf-8'),
					[n for n in sorted(theindices.elements())
						if n - 1 in theindices or n + 1 in theindices]
					if PARAMS['adjacent'] else
					list(sorted(theindices.elements()))))
			elif zeroinvalid:
				raise ValueError("invalid fragment--frequency=1: %r" % a)
	elif PARAMS['relfreq']:
		sums = defaultdict(int)
		for a, freq in zip(fragments, counts):
			if freq > threshold:
				sums[a[1:a.index(' ')]] += freq
			elif zeroinvalid:
				raise ValueError("invalid fragment--frequency=%d: %r" % (
					freq, a))
		for a, freq in zip(fragments, counts):
			out.write("%s\t%d/%d\n" % (("%s\t%s" % (a[0].decode('ascii'),
				' '.join("%s" % x if x else '' for x in a[1])))
				if PARAMS['disc'] else a.decode('utf-8'),
				freq, sums[a[1:a.index(' ')]]))
	else:
		for a, freq in zip(fragments, counts):
			if freq > threshold:
				out.write("%s\t%d\n" % (("%s\t%s" % (a[0].decode('ascii'),
					' '.join("%s" % x if x else '' for x in a[1])))
					if PARAMS['disc'] else a.decode('utf-8'), freq))
			elif zeroinvalid:
				raise ValueError("invalid fragment--frequency=1: %r" % a)


def test():
	"""Demonstration of fragment extractor."""
	main("fragments.py --disc alpinosample.export".split())


__all__ = ['altrepr', 'batch', 'coverfragworker', 'debinarize',
		'exactcountworker', 'getfragments', 'initworker', 'initworkersimple',
		'iteratefragments', 'printfragments', 'read2ndtreebank',
		'readtreebanks', 'regular', 'worker', 'workload']

if __name__ == '__main__':
	main()
