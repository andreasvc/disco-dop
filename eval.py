import sys, os.path
from getopt import gnu_getopt, GetoptError
from itertools import count, izip
from collections import defaultdict, Counter as multiset
from nltk import Tree, FreqDist
from nltk.metrics import accuracy, edit_distance
from treebank import NegraCorpusReader
from grammar import ranges
from treetransforms import disc
from treedist import treedist
#from treedist import newtreedist as treedist

def readparams(filename):
	""" read an EVALB-style parameter file and return a dictionary. """
	params = defaultdict(list)
	# NB: we ignore MAX_ERROR, we abort immediately on error.
	validkeysonce = "DEBUG MAX_ERROR CUTOFF_LEN LABELED DISC_ONLY".split()
	params = { "DEBUG" : 0, "MAX_ERROR": 10, "CUTOFF_LEN" : 40,
					"LABELED" : 1, "DISC_ONLY" : 0,
					"DELETE_LABEL" : [], "DELETE_LABEL_FOR_LENGTH" : [],
					"EQ_LABEL" : {}, "EQ_WORD" : {} }
	seen = set()
	for a in open(filename) if filename else ():
		line = a.strip()
		if line and not line.startswith("#"):
			key, val = line.split(None, 1)
			if key in validkeysonce:
				assert key not in seen, "cannot declare parameter %s twice" %key
				seen.add(key)
				params[key] = int(val)
			elif key in ("DELETE_LABEL", "DELETE_LABEL_FOR_LENGTH"):
				params[key].append(val)
			elif key in ("EQ_LABEL", "EQ_WORD"):
				hd = val.split()[0]
				assert not any(a in params[key] for a in val.split()), (
					"Values for EQ_LABEL and EQ_WORD should be disjoint.")
				params[key].update((a, hd) for a in val.split()[1:])
			else:
				raise ValueError("unrecognized parameter key: %s" % key)
	return params

def transform(tree, sent, delete, eqlabel, eqword):
	""" Apply the transformations according to the parameter file. """
	for a in reversed(list(tree.subtrees(lambda n: isinstance(n[0], Tree)))):
		for n, b in zip(count(), a)[::-1]:
			if not b: a.pop(n)	#remove empty nodes
			elif b.node in delete:
				# replace phrasal node with its children;
				# remove pre-terminal entirely
				if isinstance(b[0], Tree): a[n:n+1] = b
				else: del a[n]
			else: b.node = eqlabel.get(b.node, b.node)
	# removed POS tags cause the numbering to be off, restore.
	leafmap = dict((m, n) for n, m in enumerate(sorted(tree.leaves())))
	# retain words still in tree
	sent = [sent[n] for n in sorted(leafmap.itervalues())]
	for a in tree.treepositions('leaves'):
		tree[a] = leafmap[tree[a]]
		if sent[tree[a]] in eqword: sent[tree[a]] = eqword[sent[tree[a]]]

def bracketings(tree, labeled=True, delete=(), disconly=False):
	""" Return the labeled set of bracketings for a tree:
	for each nonterminal node, the set will contain a tuple with the label and
	the set of terminals which it dominates.
	>>> bracketings(Tree("(S (NP 1) (VP (VB 0) (JJ 2)))"))
	frozenset([('VP', frozenset(['0', '2'])),
				('S', frozenset(['1', '0', '2']))])
	>>> bracketings(Tree("(S (NP 1) (VP (VB 0) (JJ 2)))"), delete=["NP"])
	frozenset([('VP', frozenset(['0', '2'])),
				('S', frozenset(['0', '2']))])
	"""
	return multiset( (a.node if labeled else "", frozenset(a.leaves()) )
			for a in tree.subtrees()
				if isinstance(a[0], Tree)
					and a.node not in delete
					and (not disconly or disc(a)))

def printbrackets(brackets):
	return ", ".join("%s[%s]" % (a, ",".join(
		"-".join(str(y) for y in set((x[0], x[-1])))
		for x in ranges(sorted(b)))) for a, b in brackets)

def leafancestorpaths(tree):
	#uses [] to mark components, and () to mark constituent boundaries
	#deleted words/tags should not affect boundary detection
	paths = dict((a, []) for a in tree.leaves())
	# skip root node; skip POS tags
	for a in tree.subtrees(lambda n: n is not tree and isinstance(n[0], Tree)):
		leaves = a.leaves()
		first, last = min(leaves), max(leaves)
		for b in leaves:
			# mark beginning of constituents / components
			if len(leaves) > 1 and b - 1 not in leaves:
				if   b == first and "(" not in paths[b]: paths[b].append("(")
				elif b != first and "[" not in paths[b]: paths[b].append("[")
			# add this label to the lineage
			paths[b].append(a.node)
			# mark end of constituents / components
			if len(leaves) > 1 and b + 1 not in leaves:
				if   b == last and ")" not in paths[b]: paths[b].append(")")
				elif b != last and "]" not in paths[b]: paths[b].append("]")
	return paths

def pathscore(gold, cand):
	#catch the case of empty lineages
	#not sure about this normalization formula
	if gold == cand: return 1.0
	return max(0, (1.0 - (2.0 * edit_distance(cand, gold))
					/ (len(gold) + len(cand))))

def leafancestor(goldtree, candtree):
	""" Geoffrey Sampson, Anna Babarcz (2003):
	A test of the leaf-ancestor metric for parse accuracy """
	gold = leafancestorpaths(goldtree)
	cand = leafancestorpaths(candtree)
	return mean([pathscore(gold[leaf], cand[leaf]) for leaf in gold])

def treedisteval(a, b, includeroot=False, debug=False):
	ted = treedist(a, b, debug)
	# Dice denominator
	denom = len(list(a.subtrees()) + list(b.subtrees()))
	# optionally discount ROOT nodes and preterminals
	if not includeroot: denom -= 2
	#if not includepreterms: denom -= len(a.leaves() + b.leaves())
	return ted, denom

def recall(reference, test):
	return sum((reference & test).values()) / float(sum(reference.values()))

def precision(reference, test):
	return sum((reference & test).values()) / float(sum(test.values()))

def f_measure(reference, test, alpha=0.5):
	p = precision(reference, test)
	r = recall(reference, test)
	if p == 0 or r == 0: return 0
	return 1.0/(alpha/p + (1-alpha)/r)

def harmean(seq):
	try: return len([a for a in seq if a]) / sum(1./a for a in seq if a)
	except ZeroDivisionError: return "zerodiv"

def mean(seq):
	return sum(seq) / float(len(seq)) if seq else None #"zerodiv"

def splitpath(path):
	if "/" in path: return path.rsplit("/", 1)
	else: return ".", path

def nonetozero(a):
	try: result = a()
	except ZeroDivisionError: return 0
	return 0 if result is None else result

def main(goldfile, parsesfile, param, goldencoding, parsesencoding):
	assert os.path.exists(goldfile), "gold file not found"
	assert os.path.exists(parsesfile), "parses file not found"
	gold = NegraCorpusReader(*splitpath(goldfile), encoding=goldencoding)
	parses = NegraCorpusReader(*splitpath(parsesfile), encoding=parsesencoding)
	goldlen = len(gold.parsed_sents())
	parseslen = len(parses.parsed_sents())
	start = 0; end = goldlen
	if start < 0: start += goldlen
	if end < 0: end += goldlen
	assert goldlen == parseslen, ("unequal number of sentences "
		"in gold & candidates: %d vs %d" % (goldlen, parseslen))
	goldlen = end - start

	if param["DEBUG"]:
		print "Parameters:"
		for a in param: print "%s\t%s" % (a, param[a])
		print """
   Sentence                 Matched   Brackets            Corr      Tag
  ID Length  Recall  Precis Bracket   gold   test  Words  Tags Accuracy    LA
______________________________________________________________________________\
"""
	exact = 0.
	maxlenseen = sentcount = dicenoms = dicedenoms = 0
	goldpos = []
	candpos = []
	la = []
	goldb = multiset()
	candb = multiset()
	goldbcat = defaultdict(multiset)
	candbcat = defaultdict(multiset)
	for n, ctree, csent, gtree, gsent in izip(count(1), parses.parsed_sents(),
		parses.sents(), gold.parsed_sents(), gold.sents()):
		if n < start: continue
		elif n > end: break
		cpos = sorted(ctree.pos())
		gpos = sorted(gtree.pos())
		lencpos = sum(1 for _, b in cpos
			if b not in param["DELETE_LABEL_FOR_LENGTH"])
		lengpos = sum(1 for _, b in gpos
			if b not in param["DELETE_LABEL_FOR_LENGTH"])
		assert lencpos == lengpos, "sentence length mismatch"
		if lencpos > param["CUTOFF_LEN"]: continue
		sentcount += 1
		if maxlenseen < lencpos: maxlenseen = lencpos
		transform(ctree, csent,
			param["DELETE_LABEL"], param["EQ_LABEL"], param["EQ_WORD"])
		transform(gtree, gsent,
			param["DELETE_LABEL"], param["EQ_LABEL"], param["EQ_WORD"])
		assert csent == gsent, ("candidate & gold sentences do not match:\n"
			"%s\%s" % (" ".join(csent), " ".join(gsent)))
		cbrack = bracketings(ctree, param["LABELED"], set([gtree.node]
			).intersection(param["DELETE_LABEL"]), param["DISC_ONLY"])
		gbrack = bracketings(gtree, param["LABELED"], set([gtree.node]
			).intersection(param["DELETE_LABEL"]), param["DISC_ONLY"])
		if cbrack == gbrack: exact += 1
		candb.update((n, a) for a in cbrack.elements())
		goldb.update((n, a) for a in gbrack.elements())
		for a in gbrack: goldbcat[a[0]][(n, a)] += 1
		for a in cbrack: candbcat[a[0]][(n, a)] += 1
		goldpos.extend(gpos)
		candpos.extend(cpos)
		la.append(leafancestor(gtree, ctree))
		if la[-1] == 1 and gbrack != cbrack:
			print "leaf ancestor score 1.0 but no exact match: (bug?)"
			print gtree, '\n', ctree
			g = leafancestorpaths(gtree); c = leafancestorpaths(ctree)
			print g, '\n', c
			for leaf in g: print pathscore(g[leaf], c[leaf])
		ted, denom = treedisteval(gtree, ctree,
			includeroot=gtree.node not in param["DELETE_LABEL"])
		dicenoms += ted; dicedenoms += denom
		if param["DEBUG"] == 0: continue
		print "%4d  %5d  %6.2f  %6.2f   %5d  %5d  %5d  %5d  %4d  %6.2f %6.2f %2d" % (
			#" %(
			n,
			len(gpos),
			100 * nonetozero(lambda: recall(gbrack, cbrack)),
			100 * nonetozero(lambda: precision(gbrack, cbrack)),
			sum((gbrack & cbrack).values()),
			sum(gbrack.values()),
			sum(cbrack.values()),
			lengpos,
			sum(1 for a, b in zip(gpos, cpos) if a==b),
			100 * accuracy(gpos, cpos),
			100 * la[-1],
			ted, #1 if ted == 0 else 1 - ted / float(denom)
			)
		if param["DEBUG"] > 1:
			print "gold:", gbrack
			print "cand:", cbrack

	if param["LABELED"]:
		print """\n\
__________________ Category Statistics ___________________
     label      % gold   catRecall   catPrecis   catFScore
__________________________________________________________"""
		for a in sorted(set(goldbcat) | set(candbcat),
				key=lambda x: len(goldbcat[x]), reverse=True):
			print " %s      %6.2f      %6.2f      %6.2f      %6.2f" % (
				a.rjust(9),
				100 * sum(goldbcat[a].values()) / float(len(goldb)),
				100 * nonetozero(lambda: recall(goldbcat[a], candbcat[a])),
				100 * nonetozero(lambda: precision(goldbcat[a], candbcat[a])),
				100 * nonetozero(lambda: f_measure(goldbcat[a], candbcat[a])))

		print """\n\
Wrong Category Statistics (10 most frequent errors)
   test/gold   count
____________________"""
		gmismatch = dict(((n, indices), label)
					for n, (label, indices) in (goldb - candb).keys())
		wrong = FreqDist((label, gmismatch[n, indices])
					for n, (label, indices) in (candb - goldb).keys()
					if (n, indices) in gmismatch)
		for labels, freq in wrong.items()[:10]:
			print "%s %7d" % ("/".join(labels).rjust(12), freq)
	discbrackets = sum(len(bracketings(tree, disconly=True))
			for tree in parses.parsed_sents())
	gdiscbrackets = sum(len(bracketings(tree, disconly=True))
			for tree in gold.parsed_sents())

	print "\n____________ Summary <= %d _______" % param["CUTOFF_LEN"]
	print "number of sentences:       %6d" % (sentcount)
	print "maximum length:            %6d" % (maxlenseen)
	print "gold brackets (disc.):     %6d (%d)" % (len(goldb), gdiscbrackets)
	print "cand. brackets (disc.):    %6d (%d)" % (len(candb), discbrackets)
	print "labeled precision:         %6.2f" % (
		100 * nonetozero(lambda: precision(goldb, candb)))
	print "labeled recall:            %6.2f" % (
		100 * nonetozero(lambda: recall(goldb, candb)))
	print "labeled f-measure:         %6.2f" % (
		100 * nonetozero(lambda: f_measure(goldb, candb)))
	print "exact match:               %6.2f" % (100 * (exact / sentcount))
	print "leaf-ancestor:             %6.2f" % (100 * mean(la))
	print "tree-dist (Dice micro avg) %6.2f" % (
		100 * (1 - dicenoms / float(dicedenoms)))
	print "tagging accuracy:          %6.2f" % (100*accuracy(goldpos, candpos))

def test():
	main("sample2.export", "sample2.export", readparams(None),
		'iso-8859-1', 'iso-8859-1')
	exit(0)

def usage():
	print """\
wrong number of arguments. usage:
%s gold parses [params] [options]
(where gold and parses are files in export format, params is in EVALB format,
and options may consist of:

--goldenc enc
--parsesenc enc  To specify a different encoding than the default UTF-8.
--cutofflen n    Overrides the sentence length cutoff of the parameter file.
--debug          Enable printing of verbose information.
--disconly       Only evaluate on discontinuous constituents.

Example:	%s sample2.export parses.export TEST.prm --goldenc iso-8859-1\
""" % (sys.argv[0], sys.argv[0])

if __name__ == '__main__':
	flags = ("test", "debug", "disconly")
	options = ('inputenc=', 'outputenc=', 'cutofflen=')
	try:
		opts, args = gnu_getopt(sys.argv[1:], "", flags + options)
		opts = dict(opts)
		if '--test' in opts: test()
		assert 2 <= len(args) <= 3
		goldfile, parsesfile = args[:2]
	except (GetoptError, ValueError, AssertionError) as err:
		print "error:", err
		usage(); exit(2)
	param = readparams(args[2] if len(args) == 3 else None)
	if '--cutofflen' in opts: param['CUTOFF_LEN'] = int(opts['--cutofflen'])
	if '--disconly' in opts: param['DISC_ONLY'] = 1
	if '--debug' in opts: param['DEBUG'] = 1
	main(goldfile, parsesfile, param,
		opts.get('--inputenc', 'utf-8'),
		opts.get('--outputenc', 'utf-8'))
