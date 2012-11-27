import sys, os.path
from getopt import gnu_getopt, GetoptError
from itertools import count, izip_longest
from collections import defaultdict, Counter as multiset
from nltk import Tree, FreqDist
from nltk.metrics import accuracy, edit_distance
from treebank import NegraCorpusReader, DiscBracketCorpusReader, \
		BracketCorpusReader
from treedist import treedist, newtreedist
#from treedist import newtreedist as treedist

usage = """\
Evaluation of (discontinuous) parse trees, following EVALB as much as possible.
usage: %s gold parses [param] [options]
where gold and parses are files with parse trees, param is in EVALB format,
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

The parameter file supports these additional options (see EVALB README for the
others):

PRESERVE_LABELS  default 0 (false); when true, do not strip away everything
                 after '-' or '=' in non-terminal labels.
DISC_ONLY        only consider discontinuous constituents for F-scores.
TED              when enabled, give tree-edit distance scores; disabled by
                 default as these are slow to compute.
DEBUG            -1 only print summary table
                 0 additionally, print category / tag breakdowns (default)
                 these breakdowns are about the results up to the cutoff length
                 1 give per-sentence results ('--verbose')
                 2 give detailed information for each sentence ('--debug')
MAX_ERROR        this values is ignored, no errors are tolerated.
                 the parameter is accepted to support usage of unmodified
                 EVALB parameter files.
""" % (sys.argv[0], sys.argv[0])

header = """
   Sentence                 Matched   Brackets            Corr      Tag
  ID Length  Recall  Precis Bracket   gold   test  Words  Tags Accuracy    LA
______________________________________________________________________________\
"""

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
	param = readparam(args[2] if len(args) == 3 else None)
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
	doeval(gold.parsed_sents(),
			gold.tagged_sents(),
			parses.parsed_sents(),
			parses.tagged_sents(),
			param)

def doeval(gold_trees, gold_sents, cand_trees, cand_sents, param):
	assert gold_trees, "no trees in gold file"
	assert cand_trees, "no trees in parses file"
	if param["DEBUG"] > 0:
		print "Parameters:"
		for a in param: print "%s\t%s" % (a, param[a])
		print header
	# the suffix '40' is for the length restricted results
	maxlenseen = sentcount = maxlenseen40 = sentcount40 = 0
	goldb = multiset(); candb = multiset()
	goldb40 = multiset(); candb40 = multiset()
	goldbcat = defaultdict(multiset); candbcat = defaultdict(multiset)
	goldbcat40 = defaultdict(multiset); candbcat40 = defaultdict(multiset)
	exact = 0.0; exact40 = 0.0
	la = []; la40 = []
	dicenoms = dicedenoms = dicenoms40 = dicedenoms40 = 0
	goldpos = []; candpos = []
	goldpos40 = []; candpos40 = []
	for n, ctree in cand_trees.iteritems():
		gtree = gold_trees[n].copy(True)
		cpos = sorted(ctree.pos())
		gpos = sorted(gtree.pos())
		csent = [w for w, _ in cand_sents[n]]
		gsent = [w for w, _ in gold_sents[n]]
		lencpos = sum(1 for _, b in cpos
			if b not in param["DELETE_LABEL_FOR_LENGTH"])
		lengpos = sum(1 for _, b in gpos
			if b not in param["DELETE_LABEL_FOR_LENGTH"])
		assert lencpos == lengpos, "sentence length mismatch"
		# massage the data (in-place modifications)
		transform(ctree, csent, cpos, dict(gpos),
			param["DELETE_LABEL"], param["EQ_LABEL"], param["EQ_WORD"],
			not param["PRESERVE_FUNCTIONS"])
		transform(gtree, gsent, gpos, dict(gpos),
			param["DELETE_LABEL"], param["EQ_LABEL"], param["EQ_WORD"],
			not param["PRESERVE_FUNCTIONS"])
		#if not gtree or not ctree: continue
		assert csent == gsent, ("candidate & gold sentences do not match:\n"
			"%r // %r" % (" ".join(csent), " ".join(gsent)))
		cbrack = bracketings(ctree, param["LABELED"], param["DELETE_LABEL"],
				param["DISC_ONLY"])
		gbrack = bracketings(gtree, param["LABELED"], param["DELETE_LABEL"],
				param["DISC_ONLY"])
		sentcount += 1
		if maxlenseen < lencpos: maxlenseen = lencpos
		if cbrack == gbrack: exact += 1
		candb.update((n, a) for a in cbrack.elements())
		goldb.update((n, a) for a in gbrack.elements())
		for a in gbrack: goldbcat[a[0]][(n, a)] += 1
		for a in cbrack: candbcat[a[0]][(n, a)] += 1
		goldpos.extend(gpos)
		candpos.extend(cpos)
		la.append(leafancestor(gtree, ctree, param["DELETE_LABEL"]))
		if param["TED"]:
			ted, denom = treedisteval(gtree, ctree,
				includeroot=gtree.node not in param["DELETE_LABEL"])
			dicenoms += ted; dicedenoms += denom
		if lencpos <= param["CUTOFF_LEN"]:
			sentcount40 += 1
			if maxlenseen40 < lencpos: maxlenseen40 = lencpos
			candb40.update((n, a) for a in cbrack.elements())
			goldb40.update((n, a) for a in gbrack.elements())
			for a in gbrack: goldbcat40[a[0]][(n, a)] += 1
			for a in cbrack: candbcat40[a[0]][(n, a)] += 1
			if cbrack == gbrack: exact40 += 1
			goldpos40.extend(gpos)
			candpos40.extend(cpos)
			if la[-1] is not None: la40.append(la[-1])
			if param["TED"]:
				dicenoms40 += ted; dicedenoms40 += denom
		if la[-1] == 1 and gbrack != cbrack:
			print "leaf ancestor score 1.0 but no exact match: (bug?)"
		elif la[-1] is None: del la[-1]
		if param["DEBUG"] <= 0: continue
		print "%4d  %5d  %s  %s   %5d  %5d  %5d  %5d  %4d  %6.2f %6.2f %s" % (
			n,
			lengpos,
			nozerodiv(lambda: recall(gbrack, cbrack)),
			nozerodiv(lambda: precision(gbrack, cbrack)),
			sum((gbrack & cbrack).values()),
			sum(gbrack.values()),
			sum(cbrack.values()),
			len(gpos),
			sum(1 for a, b in zip(gpos, cpos) if a==b),
			100 * accuracy(gpos, cpos),
			100 * la[-1],
			str(ted).rjust(2) if param["TED"] else "",
			)
		if param["DEBUG"] > 1:
			print "Sentence:", " ".join(gsent)
			print "Gold tree:      %s\nCandidate tree: %s" % (
				gtree.pprint(margin=999), ctree.pprint(margin=999))
			print "Gold brackets:      %s\nCandidate brackets: %s" % (
				printbrackets(gbrack), printbrackets(cbrack))
			print "Matched brackets:      %s\nUnmatched brackets: %s" % (
				printbrackets(gbrack & cbrack),
				printbrackets((cbrack - gbrack) | (gbrack - cbrack)))
			g = leafancestorpaths(gtree, param["DELETE_LABEL"])
			c = leafancestorpaths(ctree, param["DELETE_LABEL"])
			for leaf in g:
				print "%6.3g  %s     %s : %s" % (pathscore(g[leaf], c[leaf]),
					str(gsent[leaf]).ljust(15), " ".join(g[leaf][::-1]).rjust(20),
					" ".join(c[leaf][::-1]))
			print "%6.3g  average = leaf-ancestor score" % la[-1]
			if param["TED"]:
				print "Tree-dist: %g / %g = %g" % (
					ted, denom, 1 - ted / float(denom))
				newtreedist(gtree, ctree, True)

	breakdowns(param, goldb40, candb40, goldpos40, candpos40, goldbcat40,
			candbcat40, maxlenseen)
	summary(param, goldb, candb, goldpos, candpos, sentcount, maxlenseen,
			exact, la, dicenoms, dicedenoms, goldb40, candb40, goldpos40,
			candpos40, sentcount40, maxlenseen40, exact40, la40, dicenoms40,
			dicedenoms40)

def breakdowns(param, goldb, candb, goldpos, candpos, goldbcat, candbcat,
		maxlenseen):
	if param["LABELED"] and param["DEBUG"] != -1:
		print
		print " Category Statistics (10 most frequent categories / errors)",
		if maxlenseen > param["CUTOFF_LEN"]:
			print "for length <= %d" % param["CUTOFF_LEN"],
		print
		print "  label  % gold   recall   prec.     F1           test/gold   count"
		print "_______________________________________        ____________________"
		gmismatch = dict(((n, indices), label)
					for n, (label, indices) in (goldb - candb).keys())
		wrong = FreqDist((label, gmismatch[n, indices])
					for n, (label, indices) in (candb - goldb).keys()
					if (n, indices) in gmismatch)
		freqcats = sorted(set(goldbcat) | set(candbcat),
				key=lambda x: len(goldbcat[x]), reverse=True)
		for cat, mismatch in izip_longest(freqcats[:10], wrong.keys()[:10]):
			if cat is None: print "                                       ",
			else: print "%s  %6.2f  %s  %s  %s" % (
					cat.rjust(7),
					100 * sum(goldbcat[cat].values()) / float(len(goldb)),
					nozerodiv(lambda: recall(goldbcat[cat], candbcat[cat])),
					nozerodiv(lambda: precision(goldbcat[cat], candbcat[cat])),
					nozerodiv(lambda: f_measure(goldbcat[cat], candbcat[cat]))),
			if mismatch is not None:
				print "       %s %7d" % (
						"/".join(mismatch).rjust(12), wrong[mismatch]),
			print

		if accuracy(goldpos, candpos) != 1:
			print
			print " Tag Statistics (10 most frequent tags / errors)",
			if maxlenseen > param["CUTOFF_LEN"]:
				print "for length <= %d" % param["CUTOFF_LEN"],
			print "\n    tag  % gold  recall   prec.      F1           test/gold   count"
			print "_______________________________________        ____________________"
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
					print "       %s %7d" % (
						"/".join(mismatch).rjust(12), wrong[mismatch]),
				print

def summary(param, goldb, candb, goldpos, candpos, sentcount, maxlenseen,
		exact, la, dicenoms, dicedenoms, goldb40, candb40, goldpos40,
		candpos40, sentcount40, maxlenseen40, exact40, la40, dicenoms40,
		dicedenoms40):
	discbrackets = sum(1 for n, (a, b) in candb.elements()
			if b != set(range(min(b), max(b)+1)))
	gdiscbrackets = sum(1 for n, (a, b) in goldb.elements()
			if b != set(range(min(b), max(b)+1)))

	if maxlenseen <= param["CUTOFF_LEN"]:
		print "\n%s" % " Summary (ALL) ".center(35, '_')
		print "number of sentences:       %6d" % (sentcount)
		print "longest sentence:          %6d" % (maxlenseen)
		print "gold brackets (disc.):     %6d (%d)" % (len(goldb), gdiscbrackets)
		print "cand. brackets (disc.):    %6d (%d)" % (len(candb), discbrackets)
		print "labeled recall:            %s" % (
				nozerodiv(lambda: recall(goldb, candb)))
		print "labeled precision:         %s" % (
				nozerodiv(lambda: precision(goldb, candb)))
		print "labeled f-measure:         %s" % (
				nozerodiv(lambda: f_measure(goldb, candb)))
		print "exact match:               %s" % (
				nozerodiv(lambda: exact / sentcount))
		print "leaf-ancestor:             %s" % (
				nozerodiv(lambda: mean(la)))
		if param["TED"]:
			print "tree-dist (Dice micro avg) %6.2f" % (
					nozerodiv(lambda: 1 - dicenoms / float(dicedenoms)))
		print "tagging accuracy:          %s" % (
				nozerodiv(lambda: accuracy(goldpos, candpos)))
		return

	discbrackets40 = sum(1 for n, (a, b) in candb40.elements()
			if b != set(range(min(b), max(b)+1)))
	gdiscbrackets40 = sum(1 for n, (a, b) in goldb40.elements()
			if b != set(range(min(b), max(b)+1)))
	print "\n%s ALL ____ <= %d" % (
			" Summary ".center(29, "_"), param["CUTOFF_LEN"])
	print "number of sentences:       %6d     %6d" % (sentcount, sentcount40)
	print "longest sentence:          %6d     %6d" % (maxlenseen, maxlenseen40)
	print "gold brackets:             %6d     %6d" % (len(goldb), len(goldb40))
	print "cand. brackets:            %6d     %6d" % (len(candb), len(candb40))
	if gdiscbrackets or discbrackets:
		print "disc. gold brackets:       %6d     %6d" % (
				gdiscbrackets, gdiscbrackets40)
		print "disc. cand. brackets:      %6d     %6d" % (
				discbrackets, discbrackets40)
	print "labeled recall:            %s     %s" % (
			nozerodiv(lambda: recall(goldb, candb)),
			nozerodiv(lambda: recall(goldb40, candb40)))
	print "labeled precision:         %s     %s" % (
			nozerodiv(lambda: precision(goldb, candb)),
			nozerodiv(lambda: precision(goldb40, candb40)))
	print "labeled f-measure:         %s     %s" % (
			nozerodiv(lambda: f_measure(goldb, candb)),
			nozerodiv(lambda: f_measure(goldb40, candb40)))
	print "exact match:               %s     %s" % (
			nozerodiv(lambda: exact / sentcount),
			nozerodiv(lambda: exact40 / sentcount40))
	print "leaf-ancestor:             %s     %s" % (
			nozerodiv(lambda: mean(la)),
			nozerodiv(lambda: mean(la40)))
	if param["TED"]:
		print "tree-dist (Dice micro avg) %s     %s" % (
			nozerodiv(lambda: (1 - dicenoms / float(dicedenoms))),
			nozerodiv(lambda: (1 - dicenoms40 / float(dicedenoms40))))
	print "tagging accuracy:          %s     %s" % (
			nozerodiv(lambda: accuracy(goldpos, candpos)),
			nozerodiv(lambda: accuracy(goldpos40, candpos40)))

def readparam(filename):
	""" read an EVALB-style parameter file and return a dictionary. """
	param = defaultdict(list)
	# NB: we ignore MAX_ERROR, we abort immediately on error.
	validkeysonce = "DEBUG MAX_ERROR CUTOFF_LEN LABELED DISC_ONLY".split()
	param = { "DEBUG" : 0, "MAX_ERROR": 10, "CUTOFF_LEN" : 40,
					"LABELED" : 1, "DISC_ONLY" : 0, "PRESERVE_FUNCTIONS": 0,
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
				param[key] = int(val)
			elif key in ("DELETE_LABEL", "DELETE_LABEL_FOR_LENGTH"):
				param[key].append(val)
			elif key in ("EQ_LABEL", "EQ_WORD"):
				hd = val.split()[0]
				assert not any(a in param[key] for a in val.split()), (
					"Values for %s should be disjoint." % key)
				param[key].update((a, hd) for a in val.split()[1:])
			else:
				raise ValueError("unrecognized parameter key: %s" % key)
	return param

def transform(tree, sent, pos, gpos, delete, eqlabel, eqword, stripfunctions):
	""" Apply the transformations according to the parameter file,
	except for deleting the root node, which is a special case because if there
	is more than one child it cannot be deleted. """
	leaves = range(len(sent))
	posnodes = []
	for a in reversed(list(tree.subtrees(lambda n: isinstance(n[0], Tree)))):
		for n, b in zip(count(), a)[::-1]:
			if stripfunctions:
				x = b.node.find("-"); y = b.node.find("=")
				if x >= 1: a.node = b.node[:x]
				if y >= 0: a.node = b.node[:y]
			b.node = eqlabel.get(b.node, b.node)
			if not b: a.pop(n)  #remove empty nodes
			elif isinstance(b[0], Tree):
				if b.node in delete:
					# replace phrasal node with its children;
					a[n:n+1] = b
			elif gpos[b[0]] in delete:
				# remove pre-terminal entirely, but only look at gold tree,
				# to ensure the sentence lengths stay the same
				leaves.remove(b[0])
				del a[n]
			else:
				posnodes.append(b)
				a.indices = b.indices = b[:]
	# retain words still in tree
	sent[:] = [sent[n] for n in leaves]
	pos[:] = [pos[n] for n in leaves]
	# removed POS tags cause the numbering to be off, restore.
	leafmap = dict((m, n) for n, m in enumerate(leaves))
	for a in posnodes:
		a[0] = leafmap[a[0]]
		a.indices = [a[0]]
		if sent[a[0]] in eqword: sent[a[0]] = eqword[sent[a[0]]]
	# cache spans
	for a in reversed(list(tree.subtrees())):
		if isinstance(a[0], Tree):
			a.indices = []
			for b in a: a.indices.extend(b.indices)
		else: a.indices = [a[0]]

def bracketings(tree, labeled=True, delete=(), disconly=False):
	""" Return the labeled set of bracketings for a tree:
	for each nonterminal node, the set will contain a tuple with the label and
	the set of terminals which it dominates.
	Tree must have been processed by transform().
	>>> tree = Tree.parse("(S (NP 1) (VP (VB 0) (JJ 2)))", parse_leaf=int)
	>>> transform(tree, tree.leaves(), tree.pos(), dict(tree.pos()), (), \
			{}, {}, False)
	>>> bracketings(tree)
	Counter({('S', frozenset([0, 1, 2])): 1, ('VP', frozenset([0, 2])): 1})
	>>> tree = Tree.parse("(S (NP 1) (VP (VB 0) (JJ 2)))", parse_leaf=int)
	>>> transform(tree, tree.leaves(), tree.pos(), dict(tree.pos()), ("VP",), \
			{}, {}, False)
	>>> bracketings(tree)
	Counter({('S', frozenset([0, 1, 2])): 1})
	>>> tree = Tree.parse("(S (NP 1) (VP (VB 0) (JJ 2)))", parse_leaf=int)
	>>> transform(tree, tree.leaves(), tree.pos(), dict(tree.pos()), ("S",), \
			{}, {}, False)
	>>> bracketings(tree, delete=("S",))
	Counter({('VP', frozenset([0, 2])): 1})
	"""
	return multiset((a.node if labeled else "", frozenset(a.indices))
			for a in tree.subtrees()
				if a and isinstance(a[0], Tree) # nonempty and not a preterminal
					and a.node not in delete
					and (not disconly or disc(a)))

def printbrackets(brackets):
	return ", ".join("%s[%s]" % (a, ",".join(
		"-".join(str(y) for y in sorted(set(x)))
		for x in intervals(sorted(b)))) for a, b in brackets)

def leafancestorpaths(tree, delete):
	#uses [] to mark components, and () to mark constituent boundaries
	#deleted words/tags should not affect boundary detection
	paths = dict((a, []) for a in tree.indices)
	# do a top-down level-order traversal
	thislevel = [tree]
	while thislevel:
		nextlevel = []
		for n in thislevel:
			leaves = sorted(n.indices)
			# skip empty nodes and POS tags
			if not leaves or not isinstance(n[0], Tree): continue
			first, last = min(leaves), max(leaves)
			# skip root node if it is to be deleted
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

def recall(reference, candidate):
	if reference: return sum((reference & candidate).values()) / float(sum(reference.values()))
	return float('nan')

def precision(reference, candidate):
	if candidate: return sum((reference & candidate).values()) / float(sum(candidate.values()))
	return float('nan')

def f_measure(reference, candidate, alpha=0.5):
	p = precision(reference, candidate)
	r = recall(reference, candidate)
	if p == 0 or r == 0: return 0
	return 1.0/(alpha/p + (1-alpha)/r)

def harmean(seq):
	try: return len([a for a in seq if a]) / sum(1./a for a in seq if a)
	except ZeroDivisionError: return "zerodiv"

def mean(seq):
	return (sum(seq) / float(len(seq)))

def splitpath(path):
	if "/" in path: return path.rsplit("/", 1)
	else: return ".", path

def intervals(s):
	""" Partition s into a sequence of intervals corresponding to contiguous
	ranges. An interval is a pair (a, b), with a <= b denoting terminals x
	such that a <= x <= b.
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
	for a in sorted(node.indices):
		if start is None: start = prev = a
		elif a == prev + 1: prev = a
		else: return True
	return False

def nozerodiv(a):
	""" Convenience function to catch zero division or None as a result,
	and otherwise format as a percentage with two decimals. """
	try: result = a()
	except ZeroDivisionError: return ' 0DIV!'
	return '  None' if result is None else "%6.2f" % (100 * result)

def test():
	gold = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1")
	parses = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1")
	doeval(gold.parsed_sents(),
			gold.tagged_sents(),
			parses.parsed_sents(),
			parses.tagged_sents(),
			readparam(None))

if __name__ == '__main__': main()
