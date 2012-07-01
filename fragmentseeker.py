import os, logging, codecs
from multiprocessing import Pool, log_to_stderr, SUBDEBUG
from collections import defaultdict
from sys import argv, stdout
from _fragmentseeker import extractfragments, fastextractfragments,\
	readtreebank, indextrees, exactcounts, exactindices, completebitsets

# this fixes utf-8 output when piped through e.g. less
# won't help if the locale is not actually utf-8, of course
stdout = codecs.getwriter('utf8')(stdout)
params = {}

def readtreebanks(treebank1, treebank2=None, sort=True, discontinuous=False,
	limit=0, encoding="utf-8"):
	labels = {}
	prods = {}
	trees1, sents1 = readtreebank(treebank1, labels, prods, sort,
		discontinuous, limit, encoding)
	trees2, sents2 = readtreebank(treebank2, labels, prods, sort,
		discontinuous, limit, encoding)
	revlabel = sorted(labels, key=labels.get)
	treeswithprod = indextrees(trees1, prods)
	return dict(trees1=trees1, sents1=sents1, trees2=trees2, sents2=sents2,
		labels=labels, prods=prods, revlabel=revlabel,
		treeswithprod=treeswithprod)

def read2ndtreebank(treebank2, labels, prods, sort=True, discontinuous=False,
	limit=0, encoding="utf-8"):
	trees2, sents2 = readtreebank(treebank2, labels, prods, sort,
		discontinuous, limit, encoding)
	revlabel = sorted(labels, key=labels.get)
	return dict(trees2=trees2, sents2=sents2, labels=labels, prods=prods,
		revlabel=revlabel)

def initworker(treebank1, treebank2, sort, limit, encoding):
	global params
	params.update(readtreebanks(treebank1, treebank2, sort=sort, limit=limit,
		discontinuous=not params['penn'], encoding=encoding))
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
	end = min(offset + params['chunk'], len(trees1))
	result = {}
	try:
		if params['quadratic']:
			result = extractfragments(trees1, sents1, offset, end, labels,
				prods, revlabel, trees2, sents2,
				approx=not (params['exact'] or params['indices']),
				discontinuous=not params['penn'], debug=False)
		else:
			result = fastextractfragments(trees1, sents1, offset, end, labels,
				prods, revlabel, trees2, sents2,
				approx=not (params['exact'] or params['indices']),
				discontinuous=not params['penn'], debug=False)
	except Exception as e: logging.error(e)
	logging.info("finished %d--%d" % (offset, end))
	return result

def exactcountworker((n, m, fragments)):
	trees1 = params['trees1']; sents1 = params['sents1']
	penn = params['penn']; revlabel = params['revlabel']
	treeswithprod = params['treeswithprod']; quadratic = params['quadratic']
	if params['indices']:
		theindices = exactindices(trees1, sents1, fragments, not penn, revlabel,
			treeswithprod, fast=not quadratic)
		logging.info("exact indices %d of %d" % (n+1, m))
		return theindices
	else:
		counts = exactcounts(trees1, sents1, fragments, not penn, revlabel,
			treeswithprod, fast=not quadratic)
		logging.info("exact counts %d of %d" % (n+1, m))
		return counts

def main(argv):
	global params
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
	if "--batch" in argv:
		pos = argv.index("--batch")
		batch = argv[pos+1]
		argv[pos:pos+2] = []
	else: batch = False
	penn = not "--disc" in argv
	quiet = "--quiet" in argv
	debug = "--debug" in argv
	complete = "--complete" in argv
	exact = "--exact" in argv
	indices = "--indices" in argv
	quadratic = "--quadratic" in argv
	nofreq = '--nofreq' in argv
	if nofreq: argv.remove('--nofreq')
	if not penn: argv.remove("--disc")
	if quiet: argv.remove("--quiet")
	if exact: argv.remove("--exact")
	if indices: argv.remove("--indices")
	if debug: argv.remove("--debug")
	if complete: argv.remove("--complete")
	if quadratic: argv.remove("--quadratic")
	if len(argv) < 2: print "missing treebank argument"
	if any(a.startswith("--") for a in argv):
		print "unrecognized options:", [a.startswith("--") for a in argv]
	if ((not batch and len(argv) not in (2, 3))
		or any(a.startswith("--") for a in argv)):
		print """\
usage: %s [options] treebank1 [treebank2]
If only one treebank is given, fragments occurring at least twice are sought.
If two treebanks are given, finds common fragments between first & second.
Input is in Penn treebank format (S-expressions), one tree per line.
Output contains lines of the form "tree<TAB>frequency".
Output is sent to stdout; to save the results, redirect to a file.
--complete    find complete matches of fragments from treebank1 in treebank2.
--exact       find exact frequencies (complete implies exact).
--indices     report sets of indices instead of frequencies.
--disc        work with discontinuous trees; input is in Negra export format.
              output: tree<TAB>sentence<TAB>frequency
              where "tree' has indices as leaves, referring to elements of
              "sentence", a space separated list of words.
--numproc n   use n independent processes, to enable multi-core usage.
              The default is not to use multiprocessing.
--encoding x  use x as treebank encoding, e.g. UTF-8, ISO-8859-1, etc.
--numtrees n  only read first n trees from treebank
--batch dir   enable batch mode; any number of treebanks > 1 can be given;
              first treebank will be compared to all others.
              results are written to filenames of the form dir/A_B.
--quadratic   use the slower, quadratic algorithm for finding fragments.
--nofreq      do not report frequencies.
--debug       extra debug information, ignored when numproc > 1.
--quiet       disable all log messages.""" % argv[0]
		exit()
	if batch:
		assert numproc == 1, "batch mode only supported in single-process mode"
	if argv[1] == "-": argv[1] = "/dev/stdin"
	assert os.path.exists(argv[1]), "not found: %s" % argv[1]
	for a in argv[2:]: assert os.path.exists(a), "not found: %s" % a
	if complete: assert len(argv) == 3 or batch, (
		"need at least two treebanks with --complete.")
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
	if exact or indices: fragments = {}
	else: fragments = defaultdict(int)
	params.update(penn=penn, exact=exact, indices=indices, chunk=chunk,
		quadratic=quadratic, complete=complete, nofreq=nofreq)
	logging.info("parameters:\n%s" % "\n".join("    %s: %r" % kv
		for kv in sorted(dict(
		quiet=quiet, exact=exact, indices=indices, numproc=numproc,
		disc=not penn, complete=complete, quadratic=quadratic,
		treebank1=argv[1], treebank2=argv[2] if len(argv)==3 else None
		).items())))

	if numproc == 1 and batch:
		initworker(argv[1], None, not quadratic, limit, encoding)
		trees1 = params['trees1']; sents1 = params['sents1']
		if complete:
			fragments = completebitsets(trees1, sents1, params['revlabel'],
				not penn)
		for a in argv[2:]:
			params.update(read2ndtreebank(a, params['labels'], params['prods'],
				not quadratic, not penn, limit, encoding))
			trees2 = params['trees2']; sents2 = params['sents2']
			labels = params['labels']; prods = params['prods']
			revlabel = params['revlabel']
			treeswithprod = params['treeswithprod']
			if complete: pass
			elif quadratic:
				fragments = extractfragments(trees1, sents1, 0, 0, labels,
					prods, revlabel, trees2, sents2,
					approx=not (exact or indices), discontinuous=not penn,
					debug=debug)
			else:
				fragments = fastextractfragments(trees1, sents1, 0, 0, labels,
					prods, revlabel, trees2, sents2,
					approx=not (exact or indices), discontinuous=not penn,
					debug=debug)
			if exact or complete:
				logging.info("getting exact counts")
				counts = exactcounts(trees1, sents1, fragments.values(),
					not penn, revlabel, treeswithprod, fast=not quadratic)
			elif indices:
				logging.info("getting indices of occurrence")
				counts = exactindices(trees1, sents1, fragments.values(),
					not penn, revlabel, treeswithprod, fast=not quadratic)
			else: counts = fragments.values()
			filename="%s/%s_%s" % (batch, os.path.basename(argv[1]),
				os.path.basename(a))
			out = codecs.open(filename, "w", encoding=encoding)
			printfragments(fragments, counts, out=out)
			logging.info("wrote to %s" % filename)
		return
	elif numproc == 1:
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
				revlabel, trees2, sents2, approx=not (exact or indices),
				discontinuous=not penn, debug=debug)
		else:
			fragments = fastextractfragments(trees1, sents1, 0, 0, labels,
				prods, revlabel, trees2, sents2, approx=not (exact or indices),
				discontinuous=not penn, debug=debug)
		if exact or complete:
			logging.info("getting exact counts")
			counts = exactcounts(trees1, sents1, fragments.values(),
				not penn, revlabel, treeswithprod, fast=not quadratic)
		elif indices:
			logging.info("getting indices of occurrence")
			counts = exactindices(trees1, sents1, fragments.values(),
				not penn, revlabel, treeswithprod, fast=not quadratic)
		else: counts = fragments.values()
		printfragments(fragments, counts)
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
		# FIXME: detect corpus reading errors here (e.g. wrong encoding)
		dowork = pool.imap_unordered(worker, range(0, numtrees, chunk))
		for n, a in enumerate(dowork):
			if exact or indices: fragments.update(a)
			else:
				for frag, x in a.items(): fragments[frag] += x
	if exact or complete or indices:
		task = "indices" if indices else "counts"
		logging.info("dividing work for exact %s" % task)
		countchunk = 20000
		f = fragments.values()
		work = [f[n:n+countchunk] for n in range(0, len(f), countchunk)]
		work = [(n, len(work), a) for n, a in enumerate(work)]
		dowork = pool.imap(exactcountworker, work)
		logging.info("getting exact %s" % task)
		counts = []
		for a in dowork: counts.extend(a)
	else: counts = fragments.values()

	pool.terminate()
	pool.join()
	del dowork, pool
	printfragments(fragments, counts)

def printfragments(fragments, counts, out=stdout):
	penn = params['penn']
	logging.info("number of fragments: %d" % len(fragments))
	if params['nofreq']:
		for a, freq in zip(fragments.keys(), counts):
			out.write("%s\n" % (a if penn
				else ("%s\t%s" % (a[0],
					" ".join("%s" % x if x else "" for x in a[1])))))
		return
	# with complete fragments or comparing two treebanks,
	# a frequency of 1 is normal; otherwise, raise alarm.
	if params['complete'] or len(argv) == 3: threshold = 1
	else: threshold = 0
	if params['indices']:
		for a, theindices in zip(fragments.keys(), counts):
			if len(theindices) > threshold:
				out.write("%s\t%r\n" % (a if penn else ("%s\t%s" % (a[0],
					" ".join("%s" % x if x else "" for x in a[1]))),
					list(sorted(theindices))))
			elif threshold:
				logging.warning("invalid fragment--frequency=1: %s" % a)
		return
	for a, freq in zip(fragments.keys(), counts):
		if freq > threshold:
			out.write("%s\t%d\n" % (a if penn else ("%s\t%s" % (a[0],
				" ".join("%s" % x if x else "" for x in a[1]))), freq))
		elif threshold: logging.warning("invalid fragment--frequency=1: %s" % a)

if __name__ == '__main__': main(argv)
