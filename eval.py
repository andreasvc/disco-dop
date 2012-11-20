import sys, os.path
from getopt import gnu_getopt, GetoptError
from itertools import count, izip, izip_longest
from collections import defaultdict, Counter as multiset
from nltk import Tree, FreqDist
from nltk.metrics import accuracy, edit_distance
from treebank import NegraCorpusReader, DiscBracketCorpusReader, \
		BracketCorpusReader
from treedist import treedist, newtreedist
#from treedist import newtreedist as treedist

usage = """usage:
%s gold parses [params] [options]
(where gold and parses are files with parse trees, params is in EVALB format,
and options may consist of:

--cutofflen n    Overrides the sentence length cutoff of the parameter file.
--verbose        Print table with per sentence information.
--debug          Print debug information showing per sentence bracketings etc.
--disconly       Only evaluate bracketings of discontinuous constituents
                 (only affects Parseval measures).
--goldenc enc
--parsesenc enc  To specify a different encoding than the default UTF-8.
--goldfmt
--parsesfmt      Specify a corpus format. Options: export, bracket, discbracket

Example:
%s sample2.export parses.export TEST.prm --goldenc iso-8859-1\
""" % (sys.argv[0], sys.argv[0])

def main():
	flags = ("test", "verbose", "debug", "disconly", "ted")
	options = ('goldenc=', 'parsesenc=', 'goldfmt=', 'parsesfmt=', 'cutofflen=')
	try:
		opts, args = gnu_getopt(sys.argv[1:], "", flags + options)
		opts = dict(opts)
		if '--test' in opts: test()
		assert 2 <= len(args) <= 3, "Wrong number of arguments."
		goldfile, parsesfile = args[:2]
	except (GetoptError, ValueError, AssertionError) as err:
		print "error:", err, usage
		exit(2)
	param = readparams(args[2] if len(args) == 3 else None)
	if '--cutofflen' in opts: param['CUTOFF_LEN'] = int(opts['--cutofflen'])
	if '--disconly' in opts: param['DISC_ONLY'] = 1
	if '--verbose' in opts: param['DEBUG'] = 1
	if '--debug' in opts: param['DEBUG'] = 2
	if '--ted' in opts: param['TED'] = 1
	assert os.path.exists(goldfile), "gold file not found"
	assert os.path.exists(parsesfile), "parses file not found"
	Readers = {'export' : NegraCorpusReader,
		'bracket': BracketCorpusReader,
		'discbracket': DiscBracketCorpusReader}
	assert opts.get('--goldfmt', 'export') in Readers
	assert opts.get('--parsesfmt', 'export') in Readers
	goldencoding = opts.get('--goldenc', 'utf-8')
	parsesencoding = opts.get('--parsesenc', 'utf-8')
	goldreader = Readers[opts.get('--goldfmt', 'export')]
	parsesreader = Readers[opts.get('--parsesfmt', 'export')]
	gold = goldreader(*splitpath(goldfile), encoding=goldencoding)
	parses = parsesreader(*splitpath(parsesfile), encoding=parsesencoding)
	doeval(gold, parses, param)

def doeval(gold, parses, param):
	if param["DEBUG"]:
		print "Parameters:"
		for a in param: print "%s\t%s" % (a, param[a])
		print (
			"   Sentence                 Matched   Brackets            Corr      Tag\n"
			"  ID Length  Recall  Precis Bracket   gold   test  Words  Tags Accuracy    LA\n"
			"______________________________________________________________________________")
	exact = 0.
	maxlenseen = sentcount = dicenoms = dicedenoms = 0
	goldpos = []
	candpos = []
	la = []
	goldb = multiset()
	candb = multiset()
	goldbcat = defaultdict(multiset)
	candbcat = defaultdict(multiset)
	ted = denom = 0
	parses_sents = parses.sents()
	gold_sents = gold.sents()
	gold_parsed_sents = gold.parsed_sents()
	for n, ctree in parses.parsed_sents().iteritems():
		csent = parses_sents[n]
		gsent = gold_sents[n]
		gtree = gold_parsed_sents[n]
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
		transform(ctree, csent, cpos, dict(gpos),
			param["DELETE_LABEL"], param["EQ_LABEL"], param["EQ_WORD"])
		transform(gtree, gsent, gpos, dict(gpos),
			param["DELETE_LABEL"], param["EQ_LABEL"], param["EQ_WORD"])
		assert csent == gsent, ("candidate & gold sentences do not match:\n"
			"%s\%s" % (" ".join(csent), " ".join(gsent)))
		cbrack = bracketings(ctree, param["LABELED"], param["DELETE_LABEL"],
				param["DISC_ONLY"])
		gbrack = bracketings(gtree, param["LABELED"], param["DELETE_LABEL"],
				param["DISC_ONLY"])
		if cbrack == gbrack: exact += 1
		candb.update((n, a) for a in cbrack.elements())
		goldb.update((n, a) for a in gbrack.elements())
		for a in gbrack: goldbcat[a[0]][(n, a)] += 1
		for a in cbrack: candbcat[a[0]][(n, a)] += 1
		goldpos.extend(gpos)
		candpos.extend(cpos)
		la.append(leafancestor(gtree, ctree, param["DELETE_LABEL"]))
		if la[-1] == 1 and gbrack != cbrack:
			print "leaf ancestor score 1.0 but no exact match: (bug?)"
		elif la[-1] is None: del la[-1]
		if param["TED"]:
			ted, denom = treedisteval(gtree, ctree,
				includeroot=gtree.node not in param["DELETE_LABEL"])
		dicenoms += ted; dicedenoms += denom
		if param["DEBUG"] == 0: continue
		print "%4d  %5d  %6.2f  %6.2f   %5d  %5d  %5d  %5d  %4d  %6.2f %6.2f %s" % (
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
			str(ted).rjust(2) if param["TED"] else "",
			)
		if param["DEBUG"] > 1:
			print "Gold tree:      %s\nCandidate tree: %s" % (
				gtree.pprint(margin=999), ctree.pprint(margin=999))
			print "Gold brackets:      %s\nCandidate brackets: %s" % (
				printbrackets(gbrack), printbrackets(cbrack))
			g = leafancestorpaths(gtree, param["DELETE_LABEL"])
			c = leafancestorpaths(ctree, param["DELETE_LABEL"])
			for leaf in g:
				print "%6.3g  %s     %s : %s" % (pathscore(g[leaf], c[leaf]),
					gsent[leaf].ljust(15), " ".join(g[leaf][::-1]).rjust(20),
					" ".join(c[leaf][::-1]))
			print "%6.3g  average = leaf-ancestor score" % la[-1]
			if param["TED"]:
				print "Tree-dist: %g / %g = %g" % (
					ted, denom, 1 - ted / float(denom))
				newtreedist(gtree, ctree, True)

	summary(param, goldb, candb, goldpos, candpos, goldbcat, candbcat,
			sentcount, maxlenseen, exact, la, dicenoms, dicedenoms)

def summary(param, goldb, candb, goldpos, candpos, goldbcat, candbcat,
		sentcount, maxlenseen, exact, la, dicenoms, dicedenoms):
	if param["LABELED"]:
		print ("\n Category Statistics (10 most frequent categories / errors)\n"
			"  label  % gold  recall   prec.      F1           test/gold   count\n"
			"________________________________________       _____________________")
		gmismatch = dict(((n, indices), label)
					for n, (label, indices) in (goldb - candb).keys())
		wrong = FreqDist((label, gmismatch[n, indices])
					for n, (label, indices) in (candb - goldb).keys()
					if (n, indices) in gmismatch)
		freqcats = sorted(set(goldbcat) | set(candbcat),
				key=lambda x: len(goldbcat[x]), reverse=True)
		for cat, mismatch in izip_longest(freqcats[:10], wrong.keys()[:10]):
			if cat is None: print "                                       ",
			else: print "%s  %6.2f  %6.2f  %6.2f  %6.2f" % (
					cat.rjust(7),
					100 * sum(goldbcat[cat].values()) / float(len(goldb)),
					100 * nonetozero(lambda: recall(goldbcat[cat], candbcat[cat])),
					100 * nonetozero(lambda: precision(goldbcat[cat], candbcat[cat])),
					100 * nonetozero(lambda: f_measure(goldbcat[cat], candbcat[cat]))),
			if mismatch is not None:
				print "      %s %7d" % (
						"/".join(mismatch).rjust(12), wrong[mismatch]),
			print

		if accuracy(goldpos, candpos) != 1:
			print ("\n Tag Statistics (10 most frequent tags / errors)\n"
				"    tag  % gold  recall   prec.      F1           test/gold   count\n"
				"________________________________________       _____________________")
			tags = FreqDist(tag for _, tag in goldpos)
			wrong = FreqDist((g, c) for (_, g), (_, c) in zip(goldpos, candpos) if g != c)
			for tag, mismatch in izip_longest(tags.keys()[:10], wrong.keys()[:10]):
				goldtag = multiset(n for n, (w,t) in enumerate(goldpos)
						if t == tag)
				candtag = multiset(n for n, (w,t) in enumerate(candpos)
						if t == tag)
				if tag is None: print "".rjust(40),
				else:
					print "%s  %6.2f  %6.2f  %6.2f  %6.2f" % (
							tag.rjust(7),
							100 * len(goldtag) / float(len(goldpos)),
							100 * recall(goldtag, candtag),
							100 * precision(goldtag, candtag),
							100 * f_measure(goldtag, candtag)),
				if mismatch is not None:
					print "        %s %7d" % (
						"/".join(mismatch).rjust(12), wrong[mismatch]),
				print

	discbrackets = sum(1 for n, (a, b) in candb.elements()
			if b != set(range(min(b), max(b)+1)))
	gdiscbrackets = sum(1 for n, (a, b) in goldb.elements()
			if b != set(range(min(b), max(b)+1)))

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
	if param["TED"]:
		print "tree-dist (Dice micro avg) %6.2f" % (
			100 * (1 - dicenoms / float(dicedenoms)))
	print "tagging accuracy:          %6.2f" % (100*accuracy(goldpos, candpos))

def readparams(filename):
	""" read an EVALB-style parameter file and return a dictionary. """
	params = defaultdict(list)
	# NB: we ignore MAX_ERROR, we abort immediately on error.
	validkeysonce = "DEBUG MAX_ERROR CUTOFF_LEN LABELED DISC_ONLY".split()
	params = { "DEBUG" : 0, "MAX_ERROR": 10, "CUTOFF_LEN" : 40,
					"LABELED" : 1, "DISC_ONLY" : 0,
					"DELETE_LABEL" : [], "DELETE_LABEL_FOR_LENGTH" : [],
					"EQ_LABEL" : {}, "EQ_WORD" : {}, "TED": 0 }
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

def transform(tree, sent, pos, gpos, delete, eqlabel, eqword):
	""" Apply the transformations according to the parameter file,
	except for deleting the root node, which is a special case because if there
	is more than one child it cannot be deleted. """
	for a in reversed(list(tree.subtrees(lambda n: isinstance(n[0], Tree)))):
		for n, b in zip(count(), a)[::-1]:
			if not b: a.pop(n)  #remove empty nodes
			elif b.node in delete and isinstance(b[0], Tree):
				# replace phrasal node with its children;
				a[n:n+1] = b
			elif not isinstance(b[0], Tree) and gpos[b[0]] in delete:
				# remove pre-terminal entirely, but only look at gold tree,
				# to ensure the sentence lengths stay the same
				del a[n]
			else: b.node = eqlabel.get(b.node, b.node)
	# removed POS tags cause the numbering to be off, restore.
	leafmap = dict((m, n) for n, m in enumerate(sorted(tree.leaves())))
	# retain words still in tree
	sent[:] = [sent[n] for n in sorted(leafmap)]
	pos[:] = [pos[n] for n in sorted(leafmap)]
	for a in tree.treepositions('leaves'):
		tree[a] = leafmap[tree[a]]
		if sent[tree[a]] in eqword: sent[tree[a]] = eqword[sent[tree[a]]]

def bracketings(tree, labeled=True, delete=(), disconly=False):
	""" Return the labeled set of bracketings for a tree:
	for each nonterminal node, the set will contain a tuple with the label and
	the set of terminals which it dominates.
	>>> bracketings(Tree("(S (NP 1) (VP (VB 0) (JJ 2)))"))
	Counter({('VP', frozenset(['0', '2'])): 1,
				('S', frozenset(['1', '0', '2'])): 1})
	>>> bracketings(Tree("(S (NP 1) (VP (VB 0) (JJ 2)))"), delete=["S"])
	Counter({('VP', frozenset(['0', '2'])): 1})
	"""
	return multiset( (a.node if labeled else "", frozenset(a.leaves()) )
			for a in tree.subtrees()
				if a and isinstance(a[0], Tree)
					and a.node not in delete
					and (not disconly or disc(a)))

def printbrackets(brackets):
	return ", ".join("%s[%s]" % (a, ",".join(
		"-".join(str(y) for y in set(x))
		for x in intervals(sorted(b)))) for a, b in brackets)

def leafancestorpaths(tree, delete):
	#uses [] to mark components, and () to mark constituent boundaries
	#deleted words/tags should not affect boundary detection
	paths = dict((a, []) for a in tree.leaves())
	# do a top-down level-order traversal
	thislevel = [tree]
	while thislevel:
		nextlevel = []
		for n in thislevel:
			leaves = sorted(n.leaves())
			# skip POS tags and empty nodes
			if not leaves or not isinstance(n[0], Tree): continue
			first, last = min(leaves), max(leaves)
			# skip nodes to be deleted
			if n.node not in delete:
				for b in leaves:
					# mark end of constituents / components
					if b + 1 not in leaves:
						if b == last and ")" not in paths[b]:
							paths[b].append(")")
						elif b != last and "]" not in paths[b]:
							paths[b].append("]")
					# add this label to the lineage
					paths[b].append(n.node)
					# mark beginning of constituents / components
					if b - 1 not in leaves:
						if b == first and "(" not in paths[b]:
							paths[b].append("(")
						elif b != first and "[" not in paths[b]:
							paths[b].append("[")
			nextlevel.extend(n)
		thislevel = nextlevel
	return paths

def pathscore(gold, cand):
	#catch the case of empty lineages
	if gold == cand: return 1.0
	return max(0, (1.0 - edit_distance(cand, gold)
					/ float(len(gold) + len(cand))))

def leafancestor(goldtree, candtree, delete):
	""" Geoffrey Sampson, Anna Babarcz (2003):
	A test of the leaf-ancestor metric for parse accuracy """
	gold = leafancestorpaths(goldtree, delete)
	cand = leafancestorpaths(candtree, delete)
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
	if reference: return sum((reference & test).values()) / float(sum(reference.values()))
	return float('nan')

def precision(reference, test):
	if test: return sum((reference & test).values()) / float(sum(test.values()))
	return float('nan')

def f_measure(reference, test, alpha=0.5):
	p = precision(reference, test)
	r = recall(reference, test)
	if p == 0 or r == 0: return 0
	return 1.0/(alpha/p + (1-alpha)/r)

def harmean(seq):
	try: return len([a for a in seq if a]) / sum(1./a for a in seq if a)
	except ZeroDivisionError: return "zerodiv"

def mean(seq):
	return (sum(seq) / float(len(seq))) if seq else None #"zerodiv"

def splitpath(path):
	if "/" in path: return path.rsplit("/", 1)
	else: return ".", path

def intervals(s):
	""" partition s into a sequence of intervals corresponding to contiguous ranges
	An interval is a pair (a, b), with a <= b denoting terminals x s.t. a <= x <= b
	>>> list(intervals((0, 1, 3, 4, 6, 7, 8)))
	[(0, 1), (3, 4), (6, 8)]"""
	start = prev = None
	for a in s:
		if start is None: start = prev = a
		elif a == prev + 1: prev = a
		else:
			yield start, prev
			start = prev = a
	if start is not None: yield start, prev

def disc(node):
	""" This function evaluates whether a particular node is locally
	discontinuous.  The root node will, by definition, be continuous.
	Nodes can be continuous even if some of their children are discontinuous.
	"""
	if not isinstance(node, Tree): return False
	start = prev = None
	for a in sorted(node.leaves()):
		if start is None: start = prev = a
		elif a == prev + 1: prev = a
		else: return True
	return False

def nonetozero(a):
	try: result = a()
	except ZeroDivisionError: return 0
	return 0 if result is None else result

def test():
	doeval(
		NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1"),
		NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1"),
		readparams(None))

if __name__ == '__main__': main()
