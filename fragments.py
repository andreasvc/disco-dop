""" Fast Fragment Seeker
Extracts recurring fragments from constituency treebanks.

NB: there is a known bug in multiprocessing which makes it impossible to detect
ctrl-c or fatal errors like segmentation faults in children which causes the
master program to wait forever for output from its children. Therefore if
you want to abort, kill the program manually (e.g., press ctrl-z and run
'kill %1'). If the program seems stuck, re-run without multiprocessing
(pass --numproc 1) to see if there might be a bug. """

import os, re, sys, codecs, logging
from multiprocessing import Pool, cpu_count, log_to_stderr, SUBDEBUG
from collections import defaultdict
from itertools import count, imap
from getopt import gnu_getopt, GetoptError
from treetransforms import binarize, introducepreterminals
from _fragments import extractfragments, fastextractfragments, \
		exactcounts, exactindices, readtreebank, indextrees, getctrees, \
		completebitsets, coverbitsets

# this fixes utf-8 output when piped through e.g. less
# won't help if the locale is not actually utf-8, of course
sys.stdout = codecs.getwriter('utf8')(sys.stdout)
params = {}

def usage():
	print """Fast Fragment Seeker
usage: %s [options] treebank1 [treebank2]
If only one treebank is given, fragments occurring at least twice are sought.
If two treebanks are given, finds common fragments between first & second.
Input is in Penn treebank format (S-expressions), one tree per line.
Output contains lines of the form "tree<TAB>frequency".
Frequencies always refer to the first treebank.
Output is sent to stdout; to save the results, redirect to a file.
--complete    find complete matches of fragments from treebank2 in treebank1.
--indices     report sets of indices instead of frequencies.
--disc        work with discontinuous trees; input is in Negra export format.
              output: tree<TAB>sentence<TAB>frequency
              where "tree' has indices as leaves, referring to elements of
              "sentence", a space separated list of words.
--cover       include all `cover' fragments corresponding to single productions.
--numproc n   use n independent processes, to enable multi-core usage.
              The default is not to use multiprocessing; use 0 to use all CPUs.
--encoding x  use x as treebank encoding, e.g. UTF-8, ISO-8859-1, etc.
--numtrees n  only read first n trees from first treebank
--batch dir   enable batch mode; any number of treebanks > 1 can be given;
              first treebank will be compared to all others.
              results are written to filenames of the form dir/A_B.
--quadratic   use the slower, quadratic algorithm for finding fragments.
--approx      report approximate frequencies
--nofreq      do not report frequencies.
--alt         alternative output format: (NP (DT "a") NN)
              default: (NP (DT a) (NN ))
--debug       extra debug information, ignored when numproc > 1.
--quiet       disable all messages.""" % sys.argv[0]

def main(argv):
	flags = ("approx", "indices", "nofreq", "complete", "complement",
			"disc", "quiet", "debug", "quadratic", "cover", "alt")
	options = ("numproc=", "numtrees=", "encoding=", "batch=")
	try: opts, args = gnu_getopt(argv[1:], "", flags + options)
	except GetoptError as err:
		print err
		usage()
		exit(2)
	else: opts = dict(opts)

	for flag in flags: params[flag] = "--" + flag in opts
	numproc = int(opts.get("--numproc", 1))
	if numproc == 0: numproc = cpu_count()
	numtrees = int(opts.get("--numtrees", 0))
	encoding = opts.get("--encoding", "UTF-8")
	batch = opts.get("--batch")

	if len(args) < 1: print "missing treebank argument"
	if batch is None and len(args) not in (1, 2):
		print "incorrect number of arguments:", args
		usage()
		exit(2)
	if batch:
		assert numproc == 1, "batch mode only supported in single-process mode"
	if args[0] == "-": args[0] = "/dev/stdin"
	for a in args: assert os.path.exists(a), "not found: %s" % a
	if params['complete']: assert len(args) == 2 or batch, (
		"need at least two treebanks with --complete.")
	level = logging.WARNING if params['quiet'] else logging.INFO
	logging.basicConfig(level=level, format='%(message)s')

	if params['debug'] and numproc > 1:
		logger = log_to_stderr()
		logger.setLevel(SUBDEBUG)

	logging.info("Fast Fragment Seeker")

	assert numproc
	limit = numtrees
	if numtrees == 0:
		if params['disc']: numtrees = open(args[0]).read().count("#BOS")
		else: numtrees = sum(1 for _ in open(args[0]))
	assert numtrees
	if numproc == 1: mult = 1
	else: mult = 3 #max(1, int(log(numtrees, 10)))
	chunk = numtrees / (mult*numproc)
	if numtrees % (mult*numproc): chunk += 1
	numchunks = numtrees / chunk + (1 if numtrees % chunk else 0)
	if params['approx']: fragments = defaultdict(int)
	else: fragments = {}
	logging.info("parameters:\n%s", "\n".join("    %s:\t%r" % kv
		for kv in sorted(params.items())))
	logging.info("\n".join("treebank%d: %s" % (n + 1, a)
		for n, a in enumerate(args)))
	params['chunk'] = chunk

	if numproc == 1 and batch:
		initworker(args[0], None, limit, encoding)
		trees1 = params['trees1']; sents1 = params['sents1']
		if params['complete']:
			raise NotImplementedError
		for a in args[1:]:
			params.update(read2ndtreebank(a, params['labels'], params['prods'],
				params['disc'], limit, encoding))
			trees2 = params['trees2']; sents2 = params['sents2']
			labels = params['labels']; prods = params['prods']
			if params['quadratic']:
				fragments = extractfragments(trees1, sents1, 0, 0, labels,
					prods, params['revlabel'], trees2, sents2,
					discontinuous=params['disc'], debug=params['debug'],
					approx=params['approx'])
			else:
				fragments = fastextractfragments(trees1, sents1, 0, 0, labels,
					prods, params['revlabel'], trees2, sents2,
					discontinuous=params['disc'], debug=params['debug'],
					approx=params['approx'])
			if not params['approx'] and not params['nofreq']:
				fragmentkeys, bitsets = map(list, zip(*fragments.iteritems()))
				if params['indices']:
					logging.info("getting indices of occurrence")
					counts = exactindices(trees1, trees1, bitsets,
							params['treeswithprod'], fast=not params['quadratic'])
				else:
					logging.info("getting exact counts")
					counts = exactcounts(trees1, trees1, bitsets,
							params['treeswithprod'], fast=not params['quadratic'])
				fragments = zip(fragmentkeys, counts)
			filename="%s/%s_%s" % (batch, os.path.basename(args[0]),
				os.path.basename(a))
			out = codecs.open(filename, "w", encoding=encoding)
			printfragments(fragments, out=out)
			logging.info("wrote to %s", filename)
		return

	# multiprocessing part
	if numproc == 1:
		initworker(args[0], args[1] if len(args) == 2 else None, limit, encoding)
		mymap = imap
		myapply = apply
	else:
		# start worker processes
		pool = Pool(processes=numproc, initializer=initworker,
			initargs=(args[0], args[1] if len(args) == 2 else None,
				limit, encoding))
		mymap = pool.imap
		myapply = pool.apply
		# FIXME: detect corpus reading errors here (e.g. wrong encoding)
		# currently they will lead to an unclear backtrace due to multiprocessing

	if params['complete']:
		initworker(args[0], args[1] if len(args) == 2 else None,
			limit, encoding)
		fragments = completebitsets(params['trees2'],
				sents2 if params['disc'] else None, params['revlabel'])
	else:
		if numproc != 1:
			logging.info("work division:\n%s", "\n".join("    %s:\t%r" % kv
				for kv in sorted(dict(chunksize=chunk, numchunks=numchunks,
				mult=mult).items())))
		dowork = mymap(worker, range(0, numtrees, chunk))
		for n, a in enumerate(dowork):
			if params['approx']:
				for frag, x in a.items(): fragments[frag] += x
			else: fragments.update(a)
	if params['cover']:
		cover = myapply(coverfragworker, ())
		if params['approx']:
			fragments.update(zip(cover.keys(), exactcounts(trees1, trees1,
				cover.values(), params['treeswithprod'],
				fast=not params['quadratic'])))
		else: fragments.update(cover)
		logging.info("merged %d cover fragments", len(cover))
	if params['approx'] or params['nofreq']:
		fragments = fragments.items()
	else:
		task = "indices" if params['indices'] else "counts"
		logging.info("dividing work for exact %s", task)
		countchunk = 20000
		fragmentkeys, bitsets = map(list, zip(*fragments.iteritems()))
		work = [bitsets[a:a+countchunk] for a in range(0, len(bitsets), countchunk)]
		work = [(n, len(work), a) for n, a in enumerate(work)]
		logging.info("getting exact %s", task)
		counts = []
		for a in mymap(exactcountworker, work): counts.extend(a)
		fragments = zip(fragmentkeys, counts)
	if numproc != 1:
		pool.terminate()
		pool.join()
		del dowork, pool
	printfragments(fragments)

def readtreebanks(treebank1, treebank2=None, discontinuous=False,
	limit=0, encoding="utf-8"):
	labels = {}
	prods = {}
	trees1, sents1 = readtreebank(treebank1, labels, prods,
		not params['quadratic'], discontinuous, limit, encoding)
	trees2, sents2 = readtreebank(treebank2, labels, prods,
		not params['quadratic'], discontinuous, limit, encoding)
	revlabel = sorted(labels, key=labels.get)
	treeswithprod = indextrees(trees1, prods)
	return dict(trees1=trees1, sents1=sents1, trees2=trees2, sents2=sents2,
		labels=labels, prods=prods, revlabel=revlabel,
		treeswithprod=treeswithprod)

def read2ndtreebank(treebank2, labels, prods, discontinuous=False,
	limit=0, encoding="utf-8"):
	trees2, sents2 = readtreebank(treebank2, labels, prods,
		not params['quadratic'], discontinuous, limit, encoding)
	revlabel = sorted(labels, key=labels.get)
	return dict(trees2=trees2, sents2=sents2, labels=labels, prods=prods,
		revlabel=revlabel)

def initworker(treebank1, treebank2, limit, encoding):
	params.update(readtreebanks(treebank1, treebank2,
		limit=limit, discontinuous=params['disc'], encoding=encoding))
	trees1 = params['trees1']
	assert trees1
	m = "treebank1: %d trees; %d nodes (max: %d);" % (
		len(trees1), trees1.nodes, trees1.maxnodes)
	if treebank2:
		trees2 = params['trees2']
		assert trees2
		m += " treebank2: %d trees; %d nodes (max %d);" % (
			len(trees2), trees2.nodes, trees2.maxnodes)
	logging.info("%s labels: %d, prods: %d", m, len(params['labels']),
		len(params['prods']))

def worker(offset):
	trees1 = params['trees1']; sents1 = params['sents1']
	trees2 = params['trees2']; sents2 = params['sents2']
	labels = params['labels']; prods = params['prods']
	assert offset < len(trees1)
	end = min(offset + params['chunk'], len(trees1))
	result = {}
	if params['quadratic']:
		result = extractfragments(trees1, sents1, offset, end, labels,
			prods, params['revlabel'], trees2, sents2,
			approx=params['approx'],
			discontinuous=params['disc'], debug=False)
	else:
		result = fastextractfragments(trees1, sents1, offset, end, labels,
			prods, params['revlabel'], trees2, sents2,
			approx=params['approx'], discontinuous=params['disc'],
			complement=params['complement'], debug=False)
	logging.info("finished %d--%d", offset, end)
	return result

def exactcountworker(args):
	n, m, fragments = args
	trees1 = params['trees1']
	if params['indices']:
		results = exactindices(trees1, trees1, fragments,
				params['treeswithprod'], fast=not params['quadratic'])
		logging.info("exact indices %d of %d", n+1, m)
	elif params['complete']:
		results = exactcounts(trees1, params['trees2'], fragments,
				params['treeswithprod'], fast=not params['quadratic'])
		logging.info("complete fragments %d of %d", n+1, m)
	else:
		results = exactcounts(trees1, trees1, fragments,
				params['treeswithprod'], fast=not params['quadratic'])
		logging.info("exact counts %d of %d", n+1, m)
	return results

def coverfragworker():
	return coverbitsets(params['trees1'], params['sents1'],
			params['treeswithprod'], params['revlabel'], params['disc'])

def initworkersimple(trees, sents, trees2=None, sents2=None):
	params.update(getctrees(trees, sents, trees2, sents2))
	assert params['trees1']

def getfragments(trees, sents, numproc=1, iterate=False, complement=False):
	""" Get recurring fragments with exact counts in a single treebank. """
	if numproc == 0: numproc = cpu_count()
	numtrees = len(trees)
	assert numtrees
	mult = 1
	chunk = numtrees / (mult*numproc)
	if numtrees % (mult*numproc): chunk += 1
	numchunks = numtrees / chunk + (1 if numtrees % chunk else 0)
	fragments = {}
	trees = trees[:]
	params.update(chunk=chunk, disc=True, approx=False,
		complete=False, indices=False, quadratic=False, complement=complement)
	if numproc == 1:
		initworkersimple(trees, list(sents))
		mymap = imap
		myapply = apply
	else:
		logging.info("work division:\n%s", "\n".join("    %s: %r" % kv
			for kv in sorted(dict(chunksize=chunk, numchunks=numchunks,
				numproc=numproc).items())))
		# start worker processes
		pool = Pool(processes=numproc, initializer=initworkersimple,
			initargs=(trees, list(sents)))
		mymap = pool.imap
		myapply = pool.apply
	# collect recurring fragments
	logging.info("extracting recurring fragments")
	for a in mymap(worker, range(0, numtrees, chunk)):
		fragments.update(a)
	# add 'cover' fragments corresponding to single productions
	cover = myapply(coverfragworker, ())
	fragments.update(cover)
	logging.info("merged %d cover fragments", len(cover))
	countchunk = 20000
	fragmentkeys, bitsets = map(list, zip(*fragments.iteritems()))
	tmp = range(0, len(bitsets), countchunk)
	work = [(n, len(tmp), bitsets[a:a+countchunk]) for n, a in enumerate(tmp)]
	logging.info("getting exact counts")
	counts = []
	for a in mymap(exactcountworker, work): counts.extend(a)
	if numproc != 1:
		pool.terminate()
		pool.join()
		del pool
	if iterate: # optionally collect fragments of fragments
		from nltk import Tree
		logging.info("extracting fragments of recurring fragments")
		params['complement'] = False #needs to be turned off if it was on
		newfrags = fragments
		trees, sents = None, None
		ids = count()
		for _ in range(10): # up to 10 iterations
			newtrees, newsents = zip(*newfrags)
			newtrees = [binarize(
				introducepreterminals(Tree.parse(a, parse_leaf=int), ids=ids),
				childChar="}") for a in newtrees]
			newsents = [["#%d" % ids.next() if word is None else word
				for word in sent] for sent in newsents]

			newfrags, newcounts = iteratefragments(
					fragments, newtrees, newsents, trees, sents, numproc)
			if len(newfrags) == 0: break
			if trees is None: trees = []; sents = []
			trees.extend(newtrees)
			sents.extend(newsents)
			fragmentkeys.extend(newfrags)
			counts.extend(newcounts)
			fragments.update(zip(newfrags, newcounts))
	logging.info("found %d fragments", len(fragmentkeys))
	return dict(zip(fragmentkeys, counts))

def iteratefragments(fragments, newtrees, newsents, trees, sents, numproc):
	""" Get fragments of fragments. """
	numtrees = len(newtrees)
	assert numtrees
	chunk = numtrees / numproc
	if numtrees % numproc: chunk += 1
	if numproc == 1: # set fragments as input
		initworkersimple(newtrees, newsents, trees, sents)
		mymap = map
	else:
		# since the input trees change, we need a new pool each time
		pool = Pool(processes=numproc, initializer=initworkersimple,
			initargs=(newtrees, newsents, trees, sents))
		mymap = pool.imap
	newfragments = {}
	for a in mymap(worker, range(0, numtrees, chunk)):
		newfragments.update(a)
	logging.info("before: %d, after: %d, difference: %d",
		len(fragments), len(fragments.viewkeys() | newfragments.keys()),
		len(newfragments.viewkeys() - fragments.viewkeys()))
	# we have to get counts for these separately because they're not coming
	# from the same set of trees
	newkeys = list(newfragments.viewkeys() - fragments.viewkeys())
	bitsets = [newfragments[a] for a in newkeys]
	countchunk = 20000
	tmp = range(0, len(bitsets), countchunk)
	work = [(n, len(tmp), bitsets[n:n+countchunk]) for n, a in enumerate(tmp)]
	logging.info("getting exact counts")
	counts = []
	for a in mymap(exactcountworker, work): counts.extend(a)
	if numproc != 1:
		pool.terminate()
		pool.join()
		del pool
	return newkeys, counts

frontierre = re.compile(r"\(([^ ()]+) \)")
termre = re.compile(r"\(([^ ()]+) ([^ ()]+)\)")
def altrepr(a):
	""" Alternative format
	Replace double quotes with double single quotes: " -> ''
	Quote terminals with double quotes terminal: -> "terminal"
	Remove parentheses around frontier nodes: (NN ) -> NN
	>>> altrepr("(NP (DT a) (NN ))"
	(NP (DT "a") NN)
	"""
	return frontierre.sub(r'\1', termre.sub(r'(\1 "\2")', a.replace('"', "''")))

def printfragments(fragments, out=sys.stdout):
	logging.info("number of fragments: %d", len(fragments))
	if params['nofreq']:
		for a, _ in fragments:
			if params['alt']: a = altrepr(a)
			out.write("%s\n" % (("%s\t%s" % (a[0],
					" ".join("%s" % x if x else "" for x in a[1])))
					if params['disc'] else a))
		return
	# when comparing two treebanks, a frequency of 1 is normal;
	# otherwise, raise alarm.
	if params.get('trees2') or params['cover'] or params['complement']:
		threshold = 0
	else: threshold = 1
	if params['indices']:
		for a, theindices in fragments:
			if params['alt']: a = altrepr(a)
			if len(theindices) > threshold:
				out.write("%s\t%r\n" % ( ("%s\t%s" % (a[0],
					" ".join("%s" % x if x else "" for x in a[1])))
					if params['disc'] else a,
					list(sorted(theindices))))
			elif threshold:
				logging.warning("invalid fragment--frequency=1: %r", a)
		return
	for a, freq in fragments:
		if freq > threshold:
			if params['alt']: a = altrepr(a)
			out.write("%s\t%d\n" % (("%s\t%s" % (a[0],
				" ".join("%s" % x if x else "" for x in a[1])))
				if params['disc'] else a, freq))
		elif threshold: logging.warning("invalid fragment--frequency=1: %r", a)

if __name__ == '__main__': main(sys.argv)
