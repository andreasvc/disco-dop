import os, logging
from multiprocessing import Pool, log_to_stderr, SUBDEBUG
from collections import defaultdict
from itertools import count, islice
from math import log
from sys import argv
from nltk import Tree
from _fragmentseeker import extractfragments, fastextractfragments, tolist,\
	readtreebanks, exactcounts, indextrees, completebitsets
from grammar import canonicalize
from treebank import NegraCorpusReader
from containers import Ctrees

params = {}

def initworker(treebank1, treebank2, sort, limit, encoding):
	global params
	params.update(readtreebanks(treebank1, treebank2, sort=sort, limit=limit,
		discontinuous=not penn, encoding=encoding))
	trees1 = params['trees1']
	assert trees1
	m = "treebank1: %d trees; %d nodes (max: %d);" % (
		len(trees1), trees1.nodes, trees1.maxnodes)
	if treebank2:
		trees2 = params['trees2']
		assert trees2
		m += " treebank2: %d trees; %d nodes (max %d);" % (
			len(trees2), trees2.nodes, trees2.maxnodes)
	logging.info("%s labels: %d, prods: %d" % (m, len(params['labels']),
		len(params['prods'])))

def worker(offset):
	trees1 = params['trees1']; sents1 = params['sents1']
	trees2 = params['trees2']; sents2 = params['sents2']
	labels = params['labels']; prods = params['prods']
	revlabel = params['revlabel']
	assert offset < len(trees1)
	end = min(offset + chunk, len(trees1))
	result = {}
	try:
		if quadratic:
			result = extractfragments(trees1, sents1, offset, end, labels,
				prods, revlabel, trees2, sents2, approx=not exact,
				discontinuous=not penn, debug=False)
		else:
			result = fastextractfragments(trees1, sents1, offset, end, labels,
				prods, revlabel, trees2, sents2, approx=not exact,
				discontinuous=not penn, debug=False)
	except Exception as e: logging.error(e)
	logging.info("finished %d--%d" % (offset, end))
	return result

def exactcountworker((n, m, fragments)):
	trees1 = params['trees1']; sents1 = params['sents1']
	trees2 = params['trees2']; sents2 = params['sents2']
	penn = params['penn']; revlabel = params['revlabel']
	treeswithprod = params['treeswithprod']; quadratic = params['quadratic']
	counts = exactcounts(trees2 or trees1, sents2 or sents1, trees1,
		fragments, not penn, revlabel, treeswithprod,
		fast=not quadratic)
	logging.info("exact counts %d of %d" % (n+1, m))
	return counts

def main(argv):
	global params
	global trees1, sents1, trees2, sents2, labels, prods, revlabel
	global quadratic, exact, chunk, penn
	if "--numproc" in argv:
		pos = argv.index("--numproc")
		numproc = int(argv[pos+1])
		argv[pos:pos+2] = []
	else: numproc = 1
	if "--numtrees" in argv:
		pos = argv.index("--numtrees")
		numtrees = int(argv[pos+1])
		argv[pos:pos+2] = []
	else: numtrees = 0
	if "--encoding" in argv:
		pos = argv.index("--encoding")
		encoding = argv[pos+1]
		argv[pos:pos+2] = []
	else: encoding = "UTF-8"
	penn = not "--disc" in argv
	quiet = "--quiet" in argv
	debug = "--debug" in argv
	complete = "--complete" in argv
	exact = "--exact" in argv
	quadratic = "--quadratic" in argv
	nofreq = '--nofreq' in argv
	if nofreq: argv.remove('--nofreq')
	if not penn: argv.remove("--disc")
	if quiet: argv.remove("--quiet")
	if exact: argv.remove("--exact")
	if debug: argv.remove("--debug")
	if complete: argv.remove("--complete")
	if quadratic: argv.remove("--quadratic")
	if len(argv) not in (2, 3): print "missing treebank argument"
	if any(a.startswith("--") for a in argv):
		print "unrecognized options:", [a.startswith("--") for a in argv]
	if len(argv) not in (2, 3) or any(a.startswith("--") for a in argv):
		print """\
usage: %s [options] treebank1 [treebank2]
If only one treebank is given, fragments occurring at least twice are sought.
If two treebanks are given, common fragments to both are produced.
Input is in Penn treebank format (S-expressions), one tree per line.
Output is sent to stdout; to save the results, redirect to a file.
--numproc n     use n independent processes, to enable multi-core usage.
	            The default is not to use multiprocessing.
--encoding x	use x as treebank encoding, e.g. UTF-8, ISO-8859-1, etc.
--numtrees n	only read first n trees from treebank
--disc          work with discontinuous trees; input is in Negra export format.
--complete      look for complete matches of trees from treebank1 in treebank2.
--exact         find exact frequencies (complete implies exact).
--quadratic     use the slower, quadratic algorithm for finding fragments.
--nofreq        do not report frequencies.
--debug         extra debug information, ignored when numproc > 1.
--quiet         disable all log messages.""" % argv[0]
		exit()
	if argv[1] == "-": argv[1] = "/dev/stdin"
	assert os.path.exists(argv[1]), "not found: %s" % argv[1]
	if len(argv) == 3: assert os.path.exists(argv[2]), "not found: %s" % argv[2]
	if complete: assert len(argv) == 3, "need two treebanks with --complete."
	level = logging.WARNING if quiet else logging.INFO
	logging.basicConfig(level=level, format='%(message)s')
	
	if debug and numproc > 1:
		logger = log_to_stderr()
		logger.setLevel(SUBDEBUG)

	logging.info("fragment seeker")

	assert numproc
	limit = numtrees
	if numtrees == 0:
		if penn: numtrees = sum(1 for _ in open(argv[1]))
		else: numtrees = open(argv[1]).read().count("#BOS")
	assert numtrees
	mult = 3 #max(1, int(log(numtrees, 10)))
	chunk = numtrees / (mult*numproc)
	if numtrees % (mult*numproc): chunk += 1
	numchunks = numtrees / chunk + (1 if numtrees % chunk else 0)
	if exact: fragments = {}
	else: fragments = defaultdict(int)
	params.update(penn=penn, exact=exact, chunk=chunk, quadratic=quadratic)
	logging.info("parameters:\n%s" % "\n".join("    %s: %r" % kv
		for kv in sorted(dict(
		quiet=quiet, exact=exact, numproc=numproc, disc=not penn,
		complete=complete, quadratic=quadratic, treebank1=argv[1],
		treebank2=argv[2] if len(argv)==3 else None).items())))

	if numproc == 1:
		initworker(argv[1], argv[2] if len(argv) == 3 else None,
			not quadratic, limit, encoding)
		trees1 = params['trees1']; sents1 = params['sents1']
		trees2 = params['trees2']; sents2 = params['sents2']
		labels = params['labels']; prods = params['prods']
		revlabel = params['revlabel']; treeswithprod = params['treeswithprod']
		if complete:
			fragments = completebitsets(trees1, sents1, revlabel, not penn)
		elif quadratic:
			fragments = extractfragments(trees1, sents1, 0, 0, labels, prods,
				revlabel, trees2, sents2, approx=not exact,
				discontinuous=not penn, debug=debug)
		else:
			fragments = fastextractfragments(trees1, sents1, 0, 0, labels,
				prods, revlabel, trees2, sents2, approx=not exact,
				discontinuous=not penn, debug=debug)
		if exact or complete:
			logging.info("getting exact counts")
			counts = exactcounts(trees2 or trees1, sents2 or sents1, trees1,
				fragments.values(), not penn, revlabel, treeswithprod,
				fast=not quadratic)
		else: counts = fragments.values()
		printfragments(fragments, counts, nofreq, complete)
		return

	if complete:
		initworker(argv[1], argv[2] if len(argv) == 3 else None,
			not quadratic, limit, encoding)
		trees1 = params['trees1']; sents1 = params['sents1']
		revlabel = params['revlabel']
		fragments = completebitsets(trees1, sents1, revlabel, not penn)
	else:
		logging.info("work division:\n%s" % "\n".join("    %s: %r" % kv
			for kv in sorted(dict(
			chunksize=chunk,numchunks=numchunks,mult=mult).items())))
		# start worker processes
		pool = Pool(processes=numproc, initializer=initworker,
			initargs=(argv[1], argv[2] if len(argv) == 3 else None,
				not quadratic, limit, encoding))
		dowork = pool.imap_unordered(worker, range(0, numtrees, chunk))
		for n, a in enumerate(dowork):
			if exact: fragments.update(a)
			else:
				for frag, x in a.items(): fragments[frag] += x
	if exact or complete:
		logging.info("dividing work for exact counts")
		countchunk = 20000
		f = fragments.values()
		work = [f[n:n+countchunk] for n in range(0, len(f), countchunk)]
		work = [(n, len(work), a) for n, a in enumerate(work)]
		dowork = pool.imap(exactcountworker, work)
		logging.info("getting exact counts")
		counts = []
		for a in dowork: counts.extend(a)
	else: counts = fragments.values()

	pool.terminate()
	pool.join()
	del dowork, pool
	printfragments(fragments, counts, nofreq, complete)

def printfragments(fragments, counts, nofreq, complete):
	penn = params['penn']
	if nofreq:
		logging.info("number of fragments: %d" % len(fragments))
		for a, freq in zip(fragments.keys(), counts):
			print "%s" % (a if penn else ("%s\t%s" % a))
		return
	if complete:
		for a, freq in zip(fragments.keys(), counts):
			if freq: print "%s\t%d" % (a if penn else ("%s\t%s" % a), freq)
	else:
		logging.info("number of fragments: %d" % len(fragments))
		for a, freq in zip(fragments.keys(), counts):
			if freq > 1:
				print "%s\t%d" % (a if penn else ("%s\t%r" % a), freq)
			else: logging.info("invalid fragment--frequency=1: %s" % a)

if __name__ == '__main__': main(argv)
