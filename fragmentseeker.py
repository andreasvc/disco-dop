import os, logging, multiprocessing
from multiprocessing import Pool
from sys import argv
from collections import defaultdict
from nltk import Tree
from _fragmentseeker import extractfragments, tolist, add_srcg_rules,\
	getlabels, getprods, getsent, getsubtree
#levelorder, 
from grammar import canonicalize
from treebank import NegraCorpusReader
from bit import bitcount, nextset

trees1 = []; sents1 = []; trees2 = []; sents2 = []; labels = {}; prods = {}
chunk = 0

def f(offset):
	return extractfragments(trees1, sents1, offset, chunk, labels, prods, trees2, sents2)

def splittrees(treebank, repeat=0):
	if repeat:
		treebank = map(Tree, "".join(open(treebank).read()
						for _ in range(repeat)).splitlines())
	else: treebank = map(Tree, open(treebank).readlines())
	sents = [tree.leaves() for tree in treebank]
	for tree in treebank:
		for n, idx in enumerate(tree.treepositions('leaves')): tree[idx] = n
		tree.chomsky_normal_form()
	return treebank, sents

def pathsplit(p):
	return p.rsplit("/", 1) if "/" in p else (".", p)

if __name__ == '__main__':
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
	assert os.path.exists(argv[1])
	if len(argv) == 3: assert os.path.exists(argv[2])

	format = '%(message)s'
	if quiet: logging.basicConfig(level=logging.WARNING, format=format)
	else: logging.basicConfig(level=logging.INFO, format=format)

	from pprint import pformat
	logging.info("parameters:\n %s" % pformat(dict(
		quiet=quiet, numproc=numproc, penn=penn,
		treebank1=argv[1], treebank2=(argv[2] if len(argv)==3 else None)))[1:-1])
	logging.info("reading treebank")
	levelorder = tolist
	if penn:
		trees1, sents1 = splittrees(argv[1], repeat)
		trees1 = [levelorder(add_srcg_rules(x, y))
					for x, y in zip(trees1, sents1)]
		if len(argv) == 3:
			trees2, sents2 = splittrees(argv[2])
			trees2 = [levelorder(add_srcg_rules(x, y))
					for x, y in zip(trees2, sents2)]
	else:
		corpus = NegraCorpusReader(*pathsplit(argv[1]))
		trees1 = corpus.parsed_sents(); sents1 = corpus.sents()
		for tree in trees1: tree.chomsky_normal_form()
		trees1 = [levelorder(add_srcg_rules(canonicalize(x), y))
						for x, y in zip(trees1, sents1)]
		if len(argv) == 3:
			corpus = NegraCorpusReader(*pathsplit(argv[2]))
			trees2 = corpus.parsed_sents(); sents2 = corpus.sents()
			for tree in trees2: tree.chomsky_normal_form()
			trees2 = [levelorder(add_srcg_rules(canonicalize(x), y))
						for x, y in zip(trees2, sents2)]
	logging.info("treebank1 size: %d" % len(trees1))
	logging.info("max number of nodes: %d" % max(map(len, trees1)))

	if len(argv) == 2:
		labels = getlabels(trees1)
		prods = getprods(trees1)
		trees2 = None; sents2 = None
	else:
		labels = getlabels(trees1 + trees2)
		prods = getprods(trees1 + trees2)
		logging.info("treebank2 size: %d" % len(trees2))
		logging.info("max number of nodes: %d" % max(map(len, trees2)))
	logging.info("starting")
		
	numtrees = len(trees1)
	chunk = numtrees / (10*numproc)
	if numtrees % numproc: chunk += 1
	results = defaultdict(set)

	# start worker processes
	if numproc == 1: dowork = [f(0)]
	else:
		pool = Pool(processes=numproc)
		dowork = pool.imap_unordered(f, range(0, numtrees, chunk))
	for n, a in enumerate(dowork):
		logging.info("result %d of %d" % (n+1, numproc))
		for (n, bs), x in a.items():
			frag = getsent(getsubtree(trees1[n][nextset(bs, 0)], bs),
							sents1[n], penn)
			frag = Tree(frag)
			frag.un_chomsky_normal_form()
			results[frag.pprint(margin=9999)].update(x)
	for a, b in sorted(results.items()):
		print "%s\t%d" % (a if penn else ("%s\t%s" % a), len(b))
