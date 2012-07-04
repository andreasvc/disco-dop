""" Fast Fragment Seeker
Extracts recurring fragments from constituency treebanks.

NB: there is a known bug in multiprocessing which makes it impossible to detect
ctrl-c or other problems like segmentation faults in children which causes the
master program to wait forever for output from its children. Therefore if
there appears to be a problem, re-run without multiprocessing (--numproc 1) to
see if there might be a bug. """

import os, logging, codecs, argparse
from multiprocessing import Pool, log_to_stderr, SUBDEBUG
from collections import defaultdict
from sys import argv, stdout
from getopt import gnu_getopt, GetoptError
from _fragmentseeker import extractfragments, fastextractfragments,\
	readtreebank, indextrees, exactcounts, exactindices, completebitsets

# this fixes utf-8 output when piped through e.g. less
# won't help if the locale is not actually utf-8, of course
stdout = codecs.getwriter('utf8')(stdout)
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
--exact       find exact frequencies
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
# disabled (broken):
# --complete    find complete matches of fragments from treebank1 in treebank2.


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
	global params
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
	logging.info("%s labels: %d, prods: %d" % (m, len(params['labels']),
		len(params['prods'])))

def worker(offset):
	trees1 = params['trees1']; sents1 = params['sents1']
	trees2 = params['trees2']; sents2 = params['sents2']
	labels = params['labels']; prods = params['prods']
	assert offset < len(trees1)
	end = min(offset + params['chunk'], len(trees1))
	result = {}
	try:
		if params['quadratic']:
			result = extractfragments(trees1, sents1, offset, end, labels,
				prods, params['revlabel'], trees2, sents2,
				approx=not (params['exact'] or params['indices']),
				discontinuous=params['disc'], debug=False)
		else:
			result = fastextractfragments(trees1, sents1, offset, end, labels,
				prods, params['revlabel'], trees2, sents2,
				approx=not (params['exact'] or params['indices']),
				discontinuous=params['disc'], debug=False)
	except Exception as e: logging.error(e)
	logging.info("finished %d--%d" % (offset, end))
	return result

def exactcountworker((n, m, fragments)):
	trees1 = params['trees1']; sents1 = params['sents1']
	if params['indices']:
		results = exactindices(trees1, trees1, fragments, params['disc'],
				params['revlabel'], params['treeswithprod'],
				fast=not params['quadratic'])
		logging.info("exact indices %d of %d" % (n+1, m))
	elif params['complete']:
		results = exactcounts(trees1, trees2, fragments, params['disc'],
				params['revlabel'], params['treeswithprod'],
				fast=not params['quadratic'])
	else:
		results = exactcounts(trees1, trees1, fragments, params['disc'],
				params['revlabel'], params['treeswithprod'],
				fast=not params['quadratic'])
		logging.info("exact counts %d of %d" % (n+1, m))
	return results

def main(argv):
	global params
	flags = ("exact", "indices", "nofreq", # "complete",
			"disc", "quiet", "debug", "quadratic")
	options = ("numproc=", "numtrees=", "encoding=", "batch=")
	try: opts, args = gnu_getopt(argv[1:], "", flags + options)
	except GetoptError, err:
		print err
		usage()
		exit(2)
	else: opts = dict(opts)

	for flag in flags: params[flag] = "--" + flag in opts
	numproc = int(opts.get("--numproc", 1))
	numtrees = int(opts.get("--numtrees", 0))
	encoding = opts.get("--encoding", "UTF-8")
	batch = opts.get("--batch")

	#argparser = argparse.ArgumentParser(description="Foo.")
	#argparser.add_argument("treebank1")
	#parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
	#parser.add_argument("--quiet", help="decrease output verbosity", action="store_true")
	#args = argparser.parse_args()

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

	logging.info("fragment seeker")

	assert numproc
	limit = numtrees
	if numtrees == 0:
		if params['disc']: numtrees = open(args[0]).read().count("#BOS")
		else: numtrees = sum(1 for _ in open(args[0]))
	assert numtrees
	mult = 3 #max(1, int(log(numtrees, 10)))
	chunk = numtrees / (mult*numproc)
	if numtrees % (mult*numproc): chunk += 1
	numchunks = numtrees / chunk + (1 if numtrees % chunk else 0)
	if params['exact'] or params['indices']: fragments = {}
	else: fragments = defaultdict(int)
	params.update(chunk=chunk)
	logging.info("parameters:\n%s" % "\n".join("    %s: %r" % kv
		for kv in sorted(params.items())))
	logging.info("\n".join("treebank%d: %s" % (n + 1, a)
		for n, a in enumerate(args)))

	if numproc == 1 and batch:
		initworker(args[0], None, limit, encoding)
		trees1 = params['trees1']; sents1 = params['sents1']
		if params['complete']:
			raise NotImplemented
		for a in args[1:]:
			params.update(read2ndtreebank(a, params['labels'], params['prods'],
				params['disc'], limit, encoding))
			trees2 = params['trees2']; sents2 = params['sents2']
			labels = params['labels']; prods = params['prods']
			if params['complete']: pass
			elif params['quadratic']:
				fragments = extractfragments(trees1, sents1, 0, 0, labels,
					prods, params['revlabel'], trees2, sents2,
					discontinuous=params['disc'], debug=params['debug'],
					approx=not (params['exact'] or params['indices']))
			else:
				fragments = fastextractfragments(trees1, sents1, 0, 0, labels,
					prods, params['revlabel'], trees2, sents2,
					discontinuous=params['disc'], debug=params['debug'],
					approx=not (params['exact'] or params['indices']))
			if params['indices']:
				logging.info("getting indices of occurrence")
				counts = exactindices(trees1, trees1, fragments.values(),
						params['disc'], params['revlabel'],
						params['treeswithprod'], fast=not params['quadratic'])
			elif params['exact']:
				logging.info("getting exact counts")
				counts = exactcounts(trees1, trees1, fragments.values(),
						params['disc'], params['revlabel'],
						params['treeswithprod'], fast=not params['quadratic'])
			else: counts = fragments.values()
			filename="%s/%s_%s" % (batch, os.path.basename(args[0]),
				os.path.basename(a))
			out = codecs.open(filename, "w", encoding=encoding)
			printfragments(fragments, counts, out=out)
			logging.info("wrote to %s" % filename)
		return
	elif numproc == 1:
		initworker(args[0], args[1] if len(args) == 2 else None,
			limit, encoding)
		trees1 = params['trees1']; sents1 = params['sents1']
		trees2 = params['trees2']; sents2 = params['sents2']
		labels = params['labels']; prods = params['prods']
		if params['complete']:
			#					        needle
			fragments = completebitsets(trees2, sents2, params['revlabel'],
					params['disc'])
			for a,b in fragments.items():
				print a, bin(b[0]), len(bin(b[0])) - 2, b[1], b[2]
		elif params['quadratic']:
			fragments = extractfragments(trees1, sents1, 0, 0, labels, prods,
				params['revlabel'], trees2, sents2,
				approx=not (params['exact'] or params['indices']),
				discontinuous=params['disc'], debug=params['debug'])
		else:
			fragments = fastextractfragments(trees1, sents1, 0, 0, labels,
				prods, params['revlabel'], trees2, sents2,
				approx=not (params['exact'] or params['indices']),
				discontinuous=params['disc'], debug=params['debug'])
		if params['nofreq']:
			counts = None
		elif params['complete']:
			logging.info("getting exact counts")
			#					haystack,needle  needle
			counts = exactcounts(trees1, trees2, fragments.values(),
				params['disc'], params['revlabel'], params['treeswithprod'],
				fast=not params['quadratic'])
		elif params['indices']:
			logging.info("getting indices of occurrence")
			counts = exactindices(trees1, trees1, fragments.values(),
				params['disc'], params['revlabel'], params['treeswithprod'],
				fast=not params['quadratic'])
		elif params['exact']:
			logging.info("getting exact counts")
			counts = exactcounts(trees1, trees1, fragments.values(),
				params['disc'], params['revlabel'], params['treeswithprod'],
				fast=not params['quadratic'])
		else: counts = fragments.values()
		printfragments(fragments, counts)
		return

	# multiprocessing part
	# start worker processes
	pool = Pool(processes=numproc, initializer=initworker,
		initargs=(args[0], args[1] if len(args) == 2 else None,
			limit, encoding))
	# FIXME: detect corpus reading errors here (e.g. wrong encoding)

	if params['complete']:
		initworker(args[0], args[1] if len(args) == 2 else None,
			limit, encoding)
		fragments = completebitsets(params['trees2'], params['sents2'],
			params['revlabel'], params['disc'])
	else:
		logging.info("work division:\n%s" % "\n".join("    %s: %r" % kv
			for kv in sorted(dict(
			chunksize=chunk,numchunks=numchunks,mult=mult).items())))
		dowork = pool.imap_unordered(worker, range(0, numtrees, chunk))
		for n, a in enumerate(dowork):
			if params['exact'] or params['indices']: fragments.update(a)
			else:
				for frag, x in a.items(): fragments[frag] += x
	if params['exact'] or params['complete'] or params['indices']:
		task = "indices" if params['indices'] else "counts"
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
	logging.info("number of fragments: %d" % len(fragments))
	if params['nofreq']:
		for a in fragments:
			out.write("%s\n" % (("%s\t%s" % (a[0],
					" ".join("%s" % x if x else "" for x in a[1])))
					if params['disc'] else a))
		return
	# when comparing two treebanks, a frequency of 1 is normal;
	# otherwise, raise alarm.
	if params.get('trees2'): threshold = 0
	else: threshold = 1
	if params['indices']:
		for a, theindices in zip(fragments.keys(), counts):
			if len(theindices) > threshold:
				out.write("%s\t%r\n" % ( ("%s\t%s" % (a[0],
					" ".join("%s" % x if x else "" for x in a[1])))
					if params['disc'] else a,
					list(sorted(theindices))))
			elif threshold:
				logging.warning("invalid fragment--frequency=1: %s" % a)
		return
	for a, freq in zip(fragments.keys(), counts):
		if freq > threshold:
			out.write("%s\t%d\n" % (("%s\t%s" % (a[0],
				" ".join("%s" % x if x else "" for x in a[1])))
				if params['disc'] else a, freq))
		elif threshold: logging.warning("invalid fragment--frequency=1: %s" % a)

if __name__ == '__main__': main(argv)
