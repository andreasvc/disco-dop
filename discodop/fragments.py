"""Extract recurring tree fragments from constituency treebanks.

NB: there is a known bug in multiprocessing which makes it impossible to detect
Ctrl-C or fatal errors like segmentation faults in child processes which causes
the master program to wait forever for output from its children. Therefore if
you want to abort, kill the program manually (e.g., press Ctrl-Z and issue
'kill %1'). If the program seems stuck, re-run without multiprocessing
(pass --numproc 1) to see if there might be a bug."""

from __future__ import division, print_function, absolute_import, \
		unicode_literals
import io
import os
import re
import sys
import codecs
import logging
import tempfile
if sys.version_info[0] == 2:
	from itertools import imap as map  # pylint: disable=E0611,W0622
from itertools import count
import multiprocessing
from collections import defaultdict
from getopt import gnu_getopt, GetoptError
from .tree import brackettree
from .treebank import writetree
from .treetransforms import binarize, introducepreterminals, unbinarize
from . import _fragments
from .util import workerfunc
from .containers import Vocabulary

SHORTUSAGE = '''\
Usage: discodop fragments <treebank1> [treebank2] [options]
  or: discodop fragments --batch=<dir> <treebank1> <treebank2>... [options]'''
FLAGS = ('approx', 'indices', 'nofreq', 'complete', 'complement', 'alt',
		'relfreq', 'adjacent', 'debin', 'debug', 'quiet', 'help')
OPTIONS = ('fmt=', 'numproc=', 'numtrees=', 'encoding=', 'batch=', 'cover=',
		'twoterms=')
PARAMS = {}
FRONTIERRE = re.compile(r'\(([^ ()]+) \)')  # for altrepr()
TERMRE = re.compile(r'\(([^ ()]+) ([^ ()]+)\)')  # for altrepr()


def main(argv=None):
	"""Command line interface to fragment extraction."""
	if argv is None:
		argv = sys.argv[2:]
	try:
		opts, args = gnu_getopt(argv, 'ho:', FLAGS + OPTIONS)
	except GetoptError as err:
		print('error:', err, file=sys.stderr)
		print(SHORTUSAGE)
		sys.exit(2)
	opts = dict(opts)

	for flag in FLAGS:
		PARAMS[flag] = '--' + flag in opts
	PARAMS['disc'] = opts.get('--fmt', 'bracket') != 'bracket'
	PARAMS['fmt'] = opts.get('--fmt', 'bracket')
	numproc = int(opts.get('--numproc', 1))
	if numproc == 0:
		numproc = cpu_count()
	if not numproc:
		raise ValueError('numproc should be an integer > 0. got: %r' % numproc)
	limit = int(opts.get('--numtrees', 0)) or None
	PARAMS['cover'] = None
	if '--cover' in opts and ',' in opts['--cover']:
		a, b = opts['--cover'].split(',')
		PARAMS['cover'] = int(a), int(b)
	elif '--cover' in opts:
		PARAMS['cover'] = int(opts.get('--cover', 0)), 999
	PARAMS['twoterms'] = opts.get('--twoterms')
	encoding = opts.get('--encoding', 'utf8')
	batchdir = opts.get('--batch')

	if len(args) < 1:
		print('missing treebank argument')
	if batchdir is None and len(args) not in (1, 2):
		print('incorrect number of arguments:', args, file=sys.stderr)
		print(SHORTUSAGE)
		sys.exit(2)
	if batchdir:
		if numproc != 1:
			raise ValueError('Batch mode only supported in single-process '
				'mode. Use the xargs command for multi-processing.')
	tmp = None
	for n, fname in enumerate(args):
		if fname == '-':
			if numproc != 1:
				# write to temp file so that contents can be read
				# in multiple processes
				if tmp is not None:
					raise ValueError('can only read from stdin once.')
				tmp = tempfile.NamedTemporaryFile()
				tmp.write(open(sys.stdin.fileno(), 'rb').read())
				tmp.flush()
				args[n] = tmp.name
		elif not os.path.exists(fname):
			raise ValueError('not found: %r' % fname)
	if PARAMS['complete']:
		if len(args) < 2:
			raise ValueError('need at least two treebanks with --complete.')
		if PARAMS['twoterms'] or PARAMS['adjacent']:
			raise ValueError('--twoterms and --adjacent are incompatible '
					'with --complete.')
		if PARAMS['approx'] or PARAMS['nofreq']:
			raise ValueError('--complete is incompatible with --nofreq '
					'and --approx')

	level = logging.WARNING if PARAMS['quiet'] else logging.DEBUG
	logging.basicConfig(level=level, format='%(message)s')
	if PARAMS['debug'] and numproc > 1:
		logger = multiprocessing.log_to_stderr()
		logger.setLevel(multiprocessing.SUBDEBUG)

	logging.info('Disco-DOP Fragment Extractor')

	logging.info('parameters:\n%s', '\n'.join('    %s:\t%r' % kv
		for kv in sorted(PARAMS.items())))
	logging.info('\n'.join('treebank%d: %s' % (n + 1, a)
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
	if tmp is not None:
		del tmp


def regular(filenames, numproc, limit, encoding):
	"""non-batch processing. multiprocessing optional."""
	mult = 1
	if PARAMS['approx']:
		fragments = defaultdict(int)
	else:
		fragments = {}
	# detect corpus reading errors in this process (e.g., wrong encoding)
	initworker(
			filenames[0],
			filenames[1] if len(filenames) == 2 else None,
			limit, encoding)
	if numproc == 1:
		mymap, myworker = map, worker
	else:  # multiprocessing, start worker processes
		pool = multiprocessing.Pool(
				processes=numproc, initializer=initworker,
				initargs=(filenames[0], filenames[1] if len(filenames) == 2
					else None, limit, encoding))
		mymap, myworker = pool.imap, mpworker
	numtrees = (PARAMS['trees1'].len if limit is None
			else min(PARAMS['trees1'].len, limit))

	if PARAMS['complete']:
		trees1, trees2 = PARAMS['trees1'], PARAMS['trees2']
		fragmentkeys, bitsets = _fragments.completebitsets(
				trees1, PARAMS['vocab'],
				max(trees1.maxnodes, trees2.maxnodes), PARAMS['disc'])
	else:
		if len(filenames) == 1:
			work = workload(numtrees, mult, numproc)
		else:
			chunk = numtrees // (mult * numproc) + 1
			work = [(a, a + chunk) for a in range(0, numtrees, chunk)]
		if numproc != 1:
			logging.info('work division:\n%s', '\n'.join('    %s:\t%r' % kv
				for kv in sorted(dict(numchunks=len(work), mult=mult).items())))
		dowork = mymap(myworker, work)
		for results in dowork:
			if PARAMS['approx']:
				for frag, x in results.items():
					fragments[frag] += x
			else:
				fragments.update(results)
		fragmentkeys = list(fragments)
		bitsets = [fragments[a] for a in fragmentkeys]
	if PARAMS['nofreq']:
		counts = None
	elif PARAMS['approx']:
		counts = [fragments[a] for a in fragmentkeys]
	else:
		task = 'indices' if PARAMS['indices'] else 'counts'
		logging.info('dividing work for exact %s', task)
		countchunk = len(bitsets) // numproc + 1
		work = list(range(0, len(bitsets), countchunk))
		work = [(n, len(work), bitsets[a:a + countchunk])
				for n, a in enumerate(work)]
		counts = []
		logging.info('getting exact %s', task)
		for a in mymap(
				exactcountworker if numproc == 1 else mpexactcountworker, work):
			counts.extend(a)
	if PARAMS['cover']:
		maxdepth, maxfrontier = PARAMS['cover']
		before = len(fragmentkeys)
		cover = _fragments.allfragments(PARAMS['trees1'], PARAMS['vocab'],
				maxdepth, maxfrontier, PARAMS['disc'], PARAMS['indices'])
		for a in cover:
			if a not in fragments:
				fragmentkeys.append(a)
				counts.append(cover[a])
		logging.info('merged %d cover fragments '
				'up to depth %d with max %d frontier non-terminals.',
				len(fragmentkeys) - before, maxdepth, maxfrontier)
	if numproc != 1:
		pool.close()
		pool.join()
		del dowork, pool
	return fragmentkeys, counts


def batch(outputdir, filenames, limit, encoding, debin):
	"""batch processing: three or more treebanks specified.

	Compares the first treebank to all others, and writes the results
	to ``outputdir/A_B`` where ``A`` and ``B`` are the respective filenames.
	Counts/indices are from the other (B) treebanks.
	There are at least 2 use cases for this:

	1. Comparing one treebank to a series of others. The first treebank will
		only be loaded once.
	2. In combination with ``--complete``, the first treebank is a set of
		fragments used as queries on the other treebanks specified."""
	initworker(filenames[0], None, limit, encoding)
	trees1 = PARAMS['trees1']
	maxnodes = trees1.maxnodes
	if PARAMS['complete']:
		fragmentkeys, bitsets = _fragments.completebitsets(
				trees1, PARAMS['vocab'],
				maxnodes, PARAMS['disc'])
		fragments = True
	elif PARAMS['approx']:
		fragments = defaultdict(int)
	else:
		fragments = {}
	for filename in filenames[1:]:
		PARAMS.update(read2ndtreebank(filename, PARAMS['vocab'],
			PARAMS['fmt'], limit, encoding))
		trees2 = PARAMS['trees2']
		if not PARAMS['complete']:
			fragments = _fragments.extractfragments(trees1, 0, 0,
					PARAMS['vocab'], trees2, disc=PARAMS['disc'],
					debug=PARAMS['debug'], approx=PARAMS['approx'],
					twoterms=PARAMS['twoterms'], adjacent=PARAMS['adjacent'])
			fragmentkeys = list(fragments)
			bitsets = [fragments[a] for a in fragmentkeys]
			maxnodes = max(trees1.maxnodes, trees2.maxnodes)
		counts = None
		if PARAMS['approx'] or not fragments:
			counts = fragments.values()
		elif not PARAMS['nofreq']:
			logging.info('getting %s for %d fragments',
					'indices of occurrence' if PARAMS['indices']
					else 'exact counts', len(bitsets))
			counts = _fragments.exactcounts(trees1, trees2, bitsets,
					indices=PARAMS['indices'],
					maxnodes=maxnodes)
		outputfilename = '%s/%s_%s' % (outputdir,
				os.path.basename(filenames[0]), os.path.basename(filename))
		out = io.open(outputfilename, 'w', encoding=encoding)
		if debin:
			fragmentkeys = debinarize(fragmentkeys)
		printfragments(fragmentkeys, counts, out=out)
		logging.info('wrote to %s', outputfilename)


def readtreebanks(filename1, filename2=None, fmt='bracket',
		limit=None, encoding='utf8'):
	"""Read one or two treebanks."""
	vocab = Vocabulary()
	trees1 = _fragments.readtreebank(filename1, vocab,
			fmt, limit, encoding)
	trees2 = _fragments.readtreebank(filename2, vocab,
			fmt, limit, encoding)
	trees1.indextrees(vocab)
	if trees2:
		trees2.indextrees(vocab)
	return dict(trees1=trees1, trees2=trees2, vocab=vocab)


def read2ndtreebank(filename2, vocab, fmt='bracket',
		limit=None, encoding='utf8'):
	"""Read a second treebank."""
	trees2 = _fragments.readtreebank(filename2, vocab,
			fmt, limit, encoding)
	trees2.indextrees(vocab)
	logging.info('%r: %d trees; %d nodes (max %d); '
			'word tokens: %d\n%r',
			filename2, len(trees2), trees2.numnodes, trees2.maxnodes,
			trees2.numwords, PARAMS['vocab'])
	return dict(trees2=trees2, vocab=vocab)


def initworker(filename1, filename2, limit, encoding):
	"""Read treebanks for this worker.

	We do this separately for each process under the assumption that this is
	advantageous with a NUMA architecture."""
	PARAMS.update(readtreebanks(filename1, filename2,
			limit=limit, fmt=PARAMS['fmt'], encoding=encoding))
	trees1 = PARAMS['trees1']
	if PARAMS['debug']:
		print('\nproductions:')
		for a, b in sorted([(PARAMS['vocab'].prodrepr(n), n)
				for n in range(len(PARAMS['vocab'].prods))],
				key=lambda x: x[1]):
			print('%d. %s' % (b, a))
		print('treebank 1:')
		for n in range(trees1.len):
			trees1.printrepr(n, PARAMS['vocab'])
	if not trees1:
		raise ValueError('treebank1 empty.')
	m = 'treebank1: %d trees; %d nodes (max: %d); %d word tokens.\n' % (
			trees1.len, trees1.numnodes, trees1.maxnodes, trees1.numwords)
	if filename2:
		trees2 = PARAMS['trees2']
		if PARAMS['debug']:
			print('treebank 2:')
			for n in range(trees2.len):
				trees2.printrepr(n, PARAMS['vocab'])
		if not trees2:
			raise ValueError('treebank2 empty.')
		m += 'treebank2: %d trees; %d nodes (max %d); %d word tokens.\n' % (
				trees2.len, trees2.numnodes, trees2.maxnodes, trees2.numwords)
	logging.info('%s%r', m, PARAMS['vocab'])


def initworkersimple(trees, sents, trees2=None, sents2=None):
	"""Initialization for a worker in which a treebank was already loaded."""
	PARAMS.update(_fragments.getctrees(zip(trees, sents),
			None if trees2 is None else zip(trees2, sents2)))
	assert PARAMS['trees1'], PARAMS['trees1']


@workerfunc
def mpworker(interval):
	"""Worker function for fragment extraction (multiprocessing wrapper)."""
	return worker(interval)


def worker(interval):
	"""Worker function for fragment extraction."""
	offset, end = interval
	trees1 = PARAMS['trees1']
	trees2 = PARAMS['trees2']
	assert offset < trees1.len
	result = {}
	result = _fragments.extractfragments(trees1, offset, end,
			PARAMS['vocab'], trees2, approx=PARAMS['approx'],
			disc=PARAMS['disc'], complement=PARAMS['complement'],
			debug=PARAMS['debug'], twoterms=PARAMS['twoterms'],
			adjacent=PARAMS['adjacent'])
	logging.debug('finished %d--%d', offset, end)
	return result


@workerfunc
def mpexactcountworker(args):
	"""Worker function for counts (multiprocessing wrapper)."""
	return exactcountworker(args)


def exactcountworker(args):
	"""Worker function for counting of fragments."""
	n, m, bitsets = args
	trees1 = PARAMS['trees1']
	if PARAMS['complete']:
		results = _fragments.exactcounts(trees1, PARAMS['trees2'], bitsets,
				indices=PARAMS['indices'])
		logging.debug('complete matches chunk %d of %d', n + 1, m)
		return results
	results = _fragments.exactcounts(
			trees1, trees1, bitsets, indices=PARAMS['indices'])
	if PARAMS['indices']:
		logging.debug('exact indices chunk %d of %d', n + 1, m)
	else:
		logging.debug('exact counts chunk %d of %d', n + 1, m)
	return results


def workload(numtrees, mult, numproc):
	"""Calculate an even workload.

	When *n* trees are compared against themselves, ``n * (n - 1)`` total
	comparisons are made. Each tree ``m`` has to be compared to all trees ``x``
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


def recurringfragments(trees, sents, numproc=1, disc=True,
		iterate=False, complement=False, indices=True, maxdepth=1,
		maxfrontier=999):
	"""Get recurring fragments with exact counts in a single treebank.

	:returns: a dictionary whose keys are fragments as strings, and
		indices as values. When ``disc`` is ``True``, keys are of the form
		``(frag, sent)`` where ``frag`` is a unicode string, and ``sent``
		is a list of words as unicode strings; when ``disc`` is ``False``, keys
		are of the form ``frag`` where ``frag`` is a unicode string.
	:param trees: a sequence of binarized Tree objects, with indices as leaves.
	:param sents: the corresponding sentences (lists of strings).
	:param numproc: number of processes to use; pass 0 to use detected # CPUs.
	:param disc: when disc=True, assume trees with discontinuous constituents;
		resulting fragments will be of the form (frag, sent);
		otherwise fragments will be strings with words as leaves.
	:param iterate, complement: see :func:`_fragments.extractfragments`
	:param indices: when False, return integer counts instead of indices.
	:param maxdepth: when > 0, add 'cover' fragments to result, corresponding
		to all fragments up to given depth; pass 0 to disable.
	:param maxfrontier: maximum number of frontier non-terminals (substitution
		sites) in cover fragments; a limit of 0 only gives fragments that
		bottom out in terminals; the default 999 is unlimited for practical
		purposes."""
	if numproc == 0:
		numproc = cpu_count()
	numtrees = len(trees)
	if not numtrees:
		raise ValueError('no trees.')
	mult = 1  # 3 if numproc > 1 else 1
	fragments = {}
	trees = trees[:]
	work = workload(numtrees, mult, numproc)
	PARAMS.update(disc=disc, indices=indices, approx=False, complete=False,
			complement=complement, debug=False, adjacent=False, twoterms=None)
	initworkersimple(trees, list(sents))
	if numproc == 1:
		mymap, myworker = map, worker
	else:
		logging.info('work division:\n%s', '\n'.join('    %s: %r' % kv
				for kv in sorted(dict(numchunks=len(work),
					numproc=numproc).items())))
		# start worker processes
		pool = multiprocessing.Pool(
				processes=numproc, initializer=initworkersimple,
				initargs=(trees, list(sents)))
		mymap, myworker = pool.map, mpworker
	# collect recurring fragments
	logging.info('extracting recurring fragments')
	for a in mymap(myworker, work):
		fragments.update(a)
	fragmentkeys = list(fragments)
	bitsets = [fragments[a] for a in fragmentkeys]
	countchunk = len(bitsets) // numproc + 1
	work = list(range(0, len(bitsets), countchunk))
	work = [(n, len(work), bitsets[a:a + countchunk])
			for n, a in enumerate(work)]
	logging.info('getting exact counts for %d fragments', len(bitsets))
	counts = []
	for a in mymap(
			exactcountworker if numproc == 1 else mpexactcountworker, work):
		counts.extend(a)
	# add all fragments up to a given depth
	if maxdepth:
		cover = _fragments.allfragments(PARAMS['trees1'], PARAMS['vocab'],
				maxdepth, maxfrontier, disc, indices)
		before = len(fragmentkeys)
		for a in cover:
			if a not in fragments:
				fragmentkeys.append(a)
				counts.append(cover[a])
		logging.info('merged %d cover fragments '
				'up to depth %d with max %d frontier non-terminals.',
				len(fragmentkeys) - before, maxdepth, maxfrontier)
	if numproc != 1:
		pool.close()
		pool.join()
		del pool
	if iterate:  # optionally collect fragments of fragments
		logging.info('extracting fragments of recurring fragments')
		PARAMS['complement'] = False  # needs to be turned off if it was on
		newfrags = fragments
		trees, sents = None, None
		ids = count()
		for _ in range(10):  # up to 10 iterations
			newfrags = [brackettree(tree) for tree in newfrags]
			newtrees = [binarize(
					introducepreterminals(tree, sent, ids=ids),
					childchar='}') for tree, sent in newfrags]
			newsents = [['#%d' % next(ids) if word is None else word
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
	logging.info('found %d fragments', len(fragmentkeys))
	return dict(zip(fragmentkeys, counts))


def allfragments(trees, sents, maxdepth, maxfrontier=999):
	"""Return all fragments up to a certain depth, # frontiers."""
	PARAMS.update(disc=True, indices=True, approx=False, complete=False,
			complement=False, debug=False, adjacent=False, twoterms=None)
	initworkersimple(trees, list(sents))
	return _fragments.allfragments(PARAMS['trees1'],
			PARAMS['vocab'], maxdepth, maxfrontier,
			disc=PARAMS['disc'], indices=PARAMS['indices'])


def iteratefragments(fragments, newtrees, newsents, trees, sents, numproc):
	"""Get fragments of fragments."""
	numtrees = len(newtrees)
	if not numtrees:
		raise ValueError('no trees.')
	if numproc == 1:  # set fragments as input
		initworkersimple(newtrees, newsents, trees, sents)
		mymap, myworker = map, worker
	else:
		# since the input trees change, we need a new pool each time
		pool = multiprocessing.Pool(
				processes=numproc, initializer=initworkersimple,
				initargs=(newtrees, newsents, trees, sents))
		mymap, myworker = pool.imap, mpworker
	newfragments = {}
	for a in mymap(myworker, workload(numtrees, 1, numproc)):
		newfragments.update(a)
	logging.info('before: %d, after: %d, difference: %d',
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
	logging.info('getting exact counts for %d fragments', len(bitsets))
	counts = []
	for a in mymap(
			exactcountworker if numproc == 1 else mpexactcountworker, work):
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

	>>> print(altrepr('(NP (DT a) (NN ))'))
	(NP (DT "a") NN)
	"""
	return FRONTIERRE.sub(r'\1', TERMRE.sub(r'(\1 "\2")', a.replace('"', "''")))


def debinarize(fragments):
	"""Debinarize fragments; fragments that fail to debinarize left as-is."""
	result = []
	for origfrag in fragments:
		frag, sent = brackettree(origfrag)
		try:
			frag = writetree(unbinarize(frag), sent, 0,
					'discbracket' if PARAMS['disc'] else 'bracket').strip()
		except Exception:  # pylint: disable=broad-except
			result.append(origfrag)
		else:
			result.append(frag)
	return result


def printfragments(fragments, counts, out=None):
	"""Dump fragments to standard output or some other file object."""
	if out is None:
		out = sys.stdout
		if sys.stdout.encoding is None:
			out = codecs.getwriter('utf8')(out)
	if PARAMS['alt']:
		for n, a in enumerate(fragments):
			fragments[n] = altrepr(a)
	if PARAMS['complete']:
		logging.info('total number of matches: %d',
				sum(sum(a) for a in counts)
				if PARAMS['indices'] else sum(counts))
	else:
		logging.info('number of fragments: %d', len(fragments))
	if PARAMS['nofreq']:
		for a in fragments:
			out.write(a + '\n')
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
				out.write('%s\t%s\n' % (a,
					[n for n in theindices
						if n - 1 in theindices or n + 1 in theindices]
					if PARAMS['adjacent'] else
					str(theindices)[len("array('I', "):-len(')')]))
			elif zeroinvalid:
				raise ValueError('invalid fragment--frequency=1: %r' % a)
	elif PARAMS['relfreq']:
		sums = defaultdict(int)
		for a, freq in zip(fragments, counts):
			if freq > threshold:
				sums[a[1:a.index(' ')]] += freq
			elif zeroinvalid:
				raise ValueError('invalid fragment--frequency=%d: %r' % (
					freq, a))
		for a, freq in zip(fragments, counts):
			out.write('%s\t%d/%d\n' % (
				a, freq, sums[a[1:a.index(' ')]]))
	else:
		for a, freq in zip(fragments, counts):
			if freq > threshold:
				out.write('%s\t%d\n' % (a, freq))
			elif zeroinvalid:
				raise ValueError('invalid fragment--frequency=1: %r' % a)


def cpu_count():
	"""Return number of CPUs or 1."""
	try:
		return multiprocessing.cpu_count()
	except NotImplementedError:
		return 1


def test():
	"""Demonstration of fragment extractor."""
	main('--fmt=export alpinosample.export'.split())


__all__ = ['main', 'regular', 'batch', 'readtreebanks', 'read2ndtreebank',
		'initworker', 'initworkersimple', 'worker', 'exactcountworker',
		'workload', 'recurringfragments', 'iteratefragments', 'allfragments',
		'debinarize', 'printfragments', 'altrepr', 'cpu_count']
