import os, logging, multiprocessing
from itertools import count
from multiprocessing import Pool
from sys import argv
from collections import defaultdict
from pprint import pformat
from nltk import Tree
from _fragmentseeker import extractfragments, tolist, add_srcg_rules,\
		getlabels, getprods, getsent
from grammar import canonicalize
from treebank import NegraCorpusReader
from bit import pyintnextset as nextset
from containers import Ctrees

#def nextset(bs, _): return min(bs)

trees1 = []; sents1 = []; trees2 = []; sents2 = []; labels = {}; prods = {}
chunk = 0; complete = False

def f(offset):
	try: return extractfragments(trees1, sents1, offset, chunk, labels, prods, trees2, sents2, complete=complete)
	except Exception as e:
		logging.error(e)
		return {}

def splittrees(treebank, repeat=0):
	if repeat:
		treebank = map(Tree, "".join(open(treebank).read()
						for _ in range(repeat)).splitlines())
	else: treebank = map(Tree, open(treebank))
	sents = [tree.leaves() for tree in treebank]
	for m, tree, sent in zip(count(), treebank, sents):
		for n, a in enumerate(tree.subtrees(lambda x: len(x) == 0 or (len(x) == 1 and not isinstance(x[0], Tree)))):
			if len(a) == 0: sent.insert(n, None)
			a[:] = [n]
		tree = Tree("TMP", [tree])
		try: tree.chomsky_normal_form()
		except:
			print tree, '\n', "line", m
			raise
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
	if "--repeat" in argv:
		pos = argv.index("--repeat")
		repeat = int(argv[pos+1])
		argv[pos:pos+2] = []
	else: repeat = 0
	if "--penn" in argv: penn = True; argv.remove("--penn")
	else: penn = False
	if "--quiet" in argv: quiet = True; argv.remove("--quiet")
	else: quiet = False
	if "--debug" in argv: debug = True; argv.remove("--debug")
	else: debug = False
	if "--complete" in argv: complete = True; argv.remove("--complete")
	else: complete = False
	if len(argv) not in (2, 3):
		print """\
usage: %s [--penn] [--numproc n] [--quiet] treebank1 [treebank2]
If only one treebank is given, fragments occurring at least twice are sought.
If two treebanks are given, common fragments to both are produced.
Output is sent to stdout.
--penn, enables Penn treebank format (S-expressions), one tree per line.
	Otherwise Negra export format (discontinuous trees) is assumed. 
--numproc, use specified number of processes, to enable multi-core usage.
	The default is not to use multiprocessing.
--quiet: disable all log messages.""" % argv[0]
		exit()
	assert os.path.exists(argv[1]), argv[1]
	if len(argv) == 3: assert os.path.exists(argv[2]), argv[2]

	format = '%(message)s'
	if quiet: logging.basicConfig(level=logging.WARNING, format=format)
	else: logging.basicConfig(level=logging.INFO, format=format)
	
	#logger = multiprocessing.log_to_stderr()
	#logger.setLevel(multiprocessing.SUBDEBUG)

	logging.info("parameters:\n %s" % pformat(dict(
		quiet=quiet, numproc=numproc, penn=penn,
		treebank1=argv[1], treebank2=(argv[2] if len(argv)==3 else None)))[1:-1])
	logging.info("reading treebank")
	if penn:
		trees1, sents1 = splittrees(argv[1], repeat)
		trees1 = [tolist(add_srcg_rules(x, y))
					for x, y in zip(trees1, sents1)]
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
			trees2 = [tolist(add_srcg_rules(x, y))
					for x, y in zip(trees2, sents2)]
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
	logging.info("labels: %d, productions: %d" % (len(labels), len(prods)))
	if debug:
		for a,b in zip(trees1, sents1): print a[0].pprint(margin=9999), b
		if trees2:
			for a,b in zip(trees2, sents2): print a[0].pprint(margin=9999), b
	trees1 = Ctrees(trees1, labels, prods)
	if trees2: trees2 = Ctrees(trees2, labels, prods)
	logging.info("starting")
	return numproc, penn, debug, complete

def run(numproc, penn, debug, c):
	global complete
	complete = c
	assert numproc >= 1
	numtrees = len(trees1)
	mult = 2
	chunk = numtrees / (mult*numproc)
	if numtrees % (mult*numproc): chunk += 1
	results = defaultdict(set)

	if numproc == 1:
		chunk = numtrees
		return extractfragments(trees1, sents1, 0, chunk, labels, prods, trees2, sents2, debug=debug, complete=complete)
	
	# start worker processes
	pool = Pool(processes=numproc, maxtasksperchild=mult)
	dowork = pool.imap_unordered(f, range(0, numtrees, chunk))
	#pool.close()
	for n, a in enumerate(dowork):
		logging.info("result %d of %d" % (n+1,
			numtrees / chunk + (1 if numtrees % chunk else 0)))
		for frag, x in a.items(): results[frag].update(x)
	pool.terminate()
	pool.join()
	del dowork, pool
	return results

if __name__ == '__main__':
	nofreq = '--nofreq' in argv
	if nofreq: argv.remove('--nofreq')
	numproc, penn, debug, complete = main(argv)
	results = run(numproc, penn, debug, complete)
	del trees1
	if trees2: del trees2
	#for a, b in sorted(results.items()):
	for a, b in results.items():
		if nofreq: print "%s" % (a if penn else ("%s\t%s" % a))
		else: print "%s\t%d" % (a if penn else ("%s\t%s" % a), len(b))

