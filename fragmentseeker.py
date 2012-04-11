import os, logging, multiprocessing
from math import log
from itertools import count, islice
from multiprocessing import Pool
from sys import argv, stderr
from collections import defaultdict
from pprint import pformat
from nltk import Tree
from _fragmentseeker import extractfragments, fastextractfragments, tolist,\
	add_cfg_rules, add_srcg_rules, getlabels, getprods, sorttrees
from grammar import canonicalize
from treebank import NegraCorpusReader
from bit import pyintnextset as nextset
from containers import Ctrees

trees1 = []; sents1 = []; trees2 = []; sents2 = []; labels = {}; prods = {}
chunk = 0; complete = False; fast = False

def worker(offset):
	assert offset < len(trees1)
	end = min(offset + chunk, len(trees1))
	logging.info("started %d--%d" % (offset, end))
	result = {}
	if fast:
		try: result = fastextractfragments(trees1, sents1, offset, end, labels, prods, trees2, sents2, complete=complete)
		except Exception as e: logging.error(e)
	else:
		try: result = extractfragments(trees1, sents1, offset, end, labels, prods, trees2, sents2, complete=complete)
		except Exception as e: logging.error(e)
	logging.info("finished %d--%d" % (offset, end))
	return result

def splittrees(treebank, limit=0):
	#todo: use incremental tree reading here. but: requires guessing #nodes
	if limit: treebank = map(Tree, islice(open(treebank), limit))
	else: treebank = map(Tree, open(treebank))
	sents = [tree.leaves() for tree in treebank]
	for m, tree, sent in zip(count(), treebank, sents):
		tmp = Tree("TMP", [tree])
		try: tmp.chomsky_normal_form()
		except:
			print tree, '\n', "line", m
			raise
		add_cfg_rules(tree)
		for n, a in enumerate(tree.subtrees(lambda x: len(x) == 0 or (len(x) == 1 and not isinstance(x[0], Tree)))):
			if len(a) == 0: sent.insert(n, None)
			a[:] = [n]
	return treebank, sents

def pathsplit(p):
	return p.rsplit("/", 1) if "/" in p else (".", p)

def settrees(t1, s1, t2, s2, l, p):
	global trees1, sents1, trees2, sents2, labels, prods
	trees1, sents1 = t1, s1
	trees2, sents2 = t2, s2
	labels, prods = l, p

def main(argv):
	global trees1, sents1, trees2, sents2, labels, prods
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
	penn = "--penn" in argv
	quiet = "--quiet" in argv
	debug = "--debug" in argv
	complete = "--complete" in argv
	fast = "--fast" in argv
	if penn: argv.remove("--penn")
	if quiet: argv.remove("--quiet")
	if debug: argv.remove("--debug")
	if complete: argv.remove("--complete")
	if fast: argv.remove("--fast")
	if len(argv) not in (2, 3):
		print """\
usage: %s [--penn] [--numproc n] [--quiet] treebank1 [[--complete] treebank2]
If only one treebank is given, fragments occurring at least twice are sought.
If two treebanks are given, common fragments to both are produced.
Output is sent to stdout.
--penn, enables Penn treebank format (S-expressions), one tree per line.
	Otherwise Negra export format (discontinuous trees) is assumed. 
--numproc, use specified number of processes, to enable multi-core usage.
	The default is not to use multiprocessing.
--complete: look for complete matches of trees from treebank1 in treebank2
--debug: extra debug information, ignored when numproc > 1.
--quiet: disable all log messages.""" % argv[0]
		exit()
	assert os.path.exists(argv[1]), "not found: %s" % argv[1]
	if len(argv) == 3: assert os.path.exists(argv[2]), "not found: %s" % argv[2]

	level = logging.WARNING if quiet else logging.INFO
	logging.basicConfig(level=level, format='%(message)s')
	
	#logger = multiprocessing.log_to_stderr()
	#logger.setLevel(multiprocessing.SUBDEBUG)

	logging.info("parameters:\n %s" % pformat(dict(
		quiet=quiet, numproc=numproc, penn=penn, complete=complete, fast=fast,
		treebank1=argv[1], treebank2=argv[2] if len(argv)==3 else None))[1:-1])
	logging.info("reading treebank")
	if penn:
		trees1, sents1 = splittrees(argv[1], limit=numtrees)
		trees1 = map(tolist, trees1)
	else:
		corpus = NegraCorpusReader(*pathsplit(argv[1]))
		trees1 = corpus.parsed_sents(); sents1 = corpus.sents()
		for tree in trees1: tree.chomsky_normal_form()
		trees1 = [tolist(add_srcg_rules(canonicalize(x), y))
						for x, y in zip(trees1, sents1)]
	assert trees1
	logging.info("treebank1 size: %d; max number of nodes: %d" % (
		len(trees1), max(map(len, trees1))))

	if len(argv) == 3:
		if penn:
			trees2, sents2 = splittrees(argv[2])
			trees2 = map(tolist, trees2)
		else:
			corpus = NegraCorpusReader(*pathsplit(argv[2]))
			trees2 = corpus.parsed_sents(); sents2 = corpus.sents()
			for tree in trees2: tree.chomsky_normal_form()
			trees2 = [tolist(add_srcg_rules(canonicalize(x), y))
						for x, y in zip(trees2, sents2)]
		assert trees2
		logging.info("treebank2 size: %d; max number of nodes: %d" % (
			len(trees2), max(map(len, trees2))))
	else: trees2 = None; sents2 = None
	labels = getlabels(trees1)
	prods = getprods(trees1)
	if fast:
		sorttrees(trees1, prods)
		if len(argv) == 3: sorttrees(trees2, prods)
	logging.info("labels: %d, productions: %d" % (len(labels), len(prods)))
	if debug:
		for a,b in zip(trees1, sents1): print max(a, key=lambda x: x.node=="TOP").pprint(margin=9999), " ".join(b)
		for a,b in zip(trees2 or (), sents2 or ()):
			print max(a, key=lambda x: x.node=="TOP").pprint(margin=9999), " ".join(b)
	trees1 = Ctrees(trees1, labels, prods)
	if trees2: trees2 = Ctrees(trees2, labels, prods)
	logging.info("starting")
	return numproc, penn, debug, complete, fast

def run(numproc, penn, debug, c, f):
	global complete, fast, chunk
	complete = c
	fast = f
	assert numproc >= 1
	numtrees = len(trees1)
	mult = max(1, int(log(numtrees, 10)))
	chunk = numtrees / (mult*numproc)
	if numtrees % (mult*numproc): chunk += 1
	numchunks = numtrees / chunk + (1 if numtrees % chunk else 0)
	results = defaultdict(set)

	if numproc == 1:
		if fast:
			return fastextractfragments(trees1, sents1, 0, 0, labels, prods,
				trees2, sents2, debug=debug, complete=complete)
		return extractfragments(trees1, sents1, 0, 0, labels, prods,
			trees2, sents2, debug=debug, complete=complete)
	
	# start worker processes
	pool = Pool(processes=numproc, maxtasksperchild=mult)
	dowork = pool.imap_unordered(worker, range(0, numtrees, chunk))
	#dowork = pool.map_async(worker, range(0, numtrees, chunk))
	#dowork = dowork.get(timeout=0xFFFFFF) # half a year
	#pool.close()
	for n, a in enumerate(dowork):
		for frag, x in a.items(): results[frag].update(x)
		logging.info("merged %d of %d" % (n+1, numchunks))
	pool.terminate()
	pool.join()
	del dowork, pool
	return results

if __name__ == '__main__':
	nofreq = '--nofreq' in argv
	if nofreq: argv.remove('--nofreq')
	numproc, penn, debug, complete, fast = main(argv)
	results = run(numproc, penn, debug, complete, fast)
	del trees1
	if trees2: del trees2
	#for a, b in sorted(results.items()):
	for a, b in results.items():
		if nofreq: print "%s" % (a if penn else ("%s\t%s" % a))
		else: print "%s\t%d" % (a if penn else ("%s\t%s" % a), len(b))

