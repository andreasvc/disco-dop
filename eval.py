""" Evaluation of (discontinuous) parse trees, following EVALB as much as
possible, as well as some alternative evaluation metrics.  """
import sys, os.path
from getopt import gnu_getopt, GetoptError
from itertools import count, izip_longest
from collections import defaultdict, Counter as multiset
from tree import Tree
from treebank import NegraCorpusReader, DiscBracketCorpusReader, \
		BracketCorpusReader, readheadrules, dependencies, export
from treedist import treedist, newtreedist
#from treedist import newtreedist as treedist

USAGE = """\
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
--ted            Enable tree-edit distance evaluation.
--headrules x    Specify rules for head assignment; this enables dependency
                 evaluation.

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

HEADER = """
   Sentence                 Matched   Brackets            Corr      Tag
  ID Length  Recall  Precis Bracket   gold   test  Words  Tags Accuracy    LA
______________________________________________________________________________\
"""

def main():
	""" Command line interface for evaluation. """
	flags = ("test", "verbose", "debug", "disconly", "ted")
	options = ('goldenc=', 'parsesenc=', 'goldfmt=', 'parsesfmt=', 'cutofflen=',
		'headrules=',)
	try:
		opts, args = gnu_getopt(sys.argv[1:], "", flags + options)
		opts = dict(opts)
		if '--test' in opts:
			test()
			return
		assert 2 <= len(args) <= 3, "Wrong number of arguments."
		goldfile, parsesfile = args[:2]
	except (GetoptError, ValueError, AssertionError) as err:
		print "error: %s\n%s" % (err, USAGE)
		exit(2)
	param = readparam(args[2] if len(args) == 3 else None)
	param['CUTOFF_LEN'] = int(opts.get('--cutofflen', param['CUTOFF_LEN']))
	param['DISC_ONLY'] = '--disconly' in opts
	param['DEBUG'] = max(param['DEBUG'],
			'--verbose' in opts, 2 * ('--debug' in opts))
	param['TED'] |= '--ted' in opts
	param['DEP'] = '--headrules' in opts
	if '--headrules' in opts:
		param['HEADRULES'] = readheadrules(opts['--headrules'])
	assert os.path.exists(goldfile), "gold file not found"
	assert os.path.exists(parsesfile), "parses file not found"
	readers = {'export' : NegraCorpusReader,
		'bracket': BracketCorpusReader,
		'discbracket': DiscBracketCorpusReader}
	assert opts.get('--goldfmt', 'export') in readers
	assert opts.get('--parsesfmt', 'export') in readers
	goldencoding = opts.get('--goldenc', 'utf-8')
	parsesencoding = opts.get('--parsesenc', 'utf-8')
	goldreader = readers[opts.get('--goldfmt', 'export')]
	parsesreader = readers[opts.get('--parsesfmt', 'export')]
	gold = goldreader(*splitpath(goldfile), encoding=goldencoding)
	parses = parsesreader(*splitpath(parsesfile), encoding=parsesencoding)
	print doeval(gold.parsed_sents(),
			gold.tagged_sents(),
			parses.parsed_sents(),
			parses.tagged_sents(),
			param)

def doeval(gold_trees, gold_sents, cand_trees, cand_sents, param):
	""" Do the actual evaluation on given parse trees and parameters.
	Results are printed to standard output. """
	assert gold_trees, "no trees in gold file"
	assert cand_trees, "no trees in parses file"
	if param["DEBUG"] > 0:
		print "Parameters:"
		for a in param:
			print "%s\t%s" % (a, param[a])
		print HEADER
	# the suffix '40' is for the length restricted results
	maxlenseen = sentcount = maxlenseen40 = sentcount40 = 0
	goldb = multiset()
	candb = multiset()
	goldb40 = multiset()
	candb40 = multiset()
	goldbcat = defaultdict(multiset)
	candbcat = defaultdict(multiset)
	goldbcat40 = defaultdict(multiset)
	candbcat40 = defaultdict(multiset)
	la = []
	la40 = []
	golddep = []
	canddep = []
	golddep40 = []
	canddep40 = []
	goldpos = []
	candpos = []
	goldpos40 = []
	candpos40 = []
	exact = exact40 = 0.0
	dicenoms = dicedenoms = dicenoms40 = dicedenoms40 = 0
	import codecs
	gdepfile = codecs.open("/tmp/gold.dep", "w", encoding='utf-8')
	cdepfile = codecs.open("/tmp/cand.dep", "w", encoding='utf-8')
	for n, ctree in cand_trees.iteritems():
		gtree = gold_trees[n]
		cpos = sorted(ctree.pos())
		gpos = sorted(gtree.pos())
		csent = [w for w, _ in cand_sents[n]]
		gsent = [w for w, _ in gold_sents[n]]
		lencpos = sum(1 for _, b in cpos
			if b not in param["DELETE_LABEL_FOR_LENGTH"])
		lengpos = sum(1 for _, b in gpos
			if b not in param["DELETE_LABEL_FOR_LENGTH"])
		assert lencpos == lengpos, ("sentence length mismatch. "
				"sents:\n%s\n%s" % (" ".join(csent), " ".join(gsent)))
		# massage the data (in-place modifications)
		transform(ctree, csent, cpos, dict(gpos),
			param["DELETE_LABEL"], param["EQ_LABEL"], param["EQ_WORD"],
			not param["PRESERVE_FUNCTIONS"])
		transform(gtree, gsent, gpos, dict(gpos),
			param["DELETE_LABEL"], param["EQ_LABEL"], param["EQ_WORD"],
			not param["PRESERVE_FUNCTIONS"])
		#if not gtree or not ctree:
		#	continue
		assert csent == gsent, ("candidate & gold sentences do not match:\n"
			"%r // %r" % (" ".join(csent), " ".join(gsent)))
		cbrack = bracketings(ctree, param["LABELED"], param["DELETE_LABEL"],
				param["DISC_ONLY"])
		gbrack = bracketings(gtree, param["LABELED"], param["DELETE_LABEL"],
				param["DISC_ONLY"])
		sentcount += 1
		# this is to deal with "sentences" with only a single punctuation mark.
		if not gpos:
			sentcount40 += 1
			continue
		if maxlenseen < lencpos:
			maxlenseen = lencpos
		if cbrack == gbrack:
			exact += 1
		candb.update((n, a) for a in cbrack.elements())
		goldb.update((n, a) for a in gbrack.elements())
		for a in gbrack:
			goldbcat[a[0]][(n, a)] += 1
		for a in cbrack:
			candbcat[a[0]][(n, a)] += 1
		goldpos.extend(gpos)
		candpos.extend(cpos)
		la.append(leafancestor(gtree, ctree, param["DELETE_LABEL"]))
		if param["TED"]:
			ted, denom = treedisteval(gtree, ctree,
				includeroot=gtree.node not in param["DELETE_LABEL"])
			dicenoms += ted
			dicedenoms += denom
		if param["DEP"]:
			cdep = dependencies(ctree, param['HEADRULES'])
			gdep = dependencies(gtree, param['HEADRULES'])
			canddep.extend(cdep)
			golddep.extend(gdep)
			gdepfile.write(export(gtree, gsent, n, "conll", param['HEADRULES']))
			cdepfile.write(export(ctree, csent, n, "conll", param['HEADRULES']))
		if lencpos <= param["CUTOFF_LEN"]:
			sentcount40 += 1
			if maxlenseen40 < lencpos:
				maxlenseen40 = lencpos
			candb40.update((n, a) for a in cbrack.elements())
			goldb40.update((n, a) for a in gbrack.elements())
			for a in gbrack:
				goldbcat40[a[0]][(n, a)] += 1
			for a in cbrack:
				candbcat40[a[0]][(n, a)] += 1
			if cbrack == gbrack:
				exact40 += 1
			goldpos40.extend(gpos)
			candpos40.extend(cpos)
			if la[-1] is not None:
				la40.append(la[-1])
			if param["TED"]:
				dicenoms40 += ted
				dicedenoms40 += denom
			if param["DEP"]:
				canddep40.extend(cdep)
				golddep40.extend(gdep)
		if la[-1] == 1 and gbrack != cbrack:
			print "leaf ancestor score 1.0 but no exact match: (bug?)"
		elif la[-1] is None:
			del la[-1]
		if param["DEBUG"] <= 0:
			continue
		print "%4s  %5d  %s  %s   %5d  %5d  %5d  %5d  %4d  %s %6.2f%s%s" % (
			n,
			lengpos,
			nozerodiv(lambda: recall(gbrack, cbrack)),
			nozerodiv(lambda: precision(gbrack, cbrack)),
			sum((gbrack & cbrack).values()),
			sum(gbrack.values()),
			sum(cbrack.values()),
			len(gpos),
			sum(1 for a, b in zip(gpos, cpos) if a==b),
			nozerodiv(lambda: accuracy(gpos, cpos)),
			100 * la[-1],
			str(ted).rjust(3) if param["TED"] else "",
			nozerodiv(lambda: accuracy(gdep, cdep)) if param["DEP"] else "",
			)
		if param["DEBUG"] > 1:
			print "Sentence:", " ".join(gsent)
			print "Gold tree:      %s\nCandidate tree: %s" % (
				gtree.pprint(margin=999), ctree.pprint(margin=999))
			print "Gold brackets:      %s\nCandidate brackets: %s" % (
				strbracketings(gbrack), strbracketings(cbrack))
			print "Matched brackets:      %s\nUnmatched brackets: %s" % (
				strbracketings(gbrack & cbrack),
				strbracketings((cbrack - gbrack) | (gbrack - cbrack)))
			g = leafancestorpaths(gtree, param["DELETE_LABEL"])
			c = leafancestorpaths(ctree, param["DELETE_LABEL"])
			for leaf in g:
				print "%6.3g  %s     %s : %s" % (pathscore(g[leaf], c[leaf]),
					unicode(gsent[leaf]).ljust(15),
					" ".join(g[leaf][::-1]).rjust(20),
					" ".join(c[leaf][::-1]))
			print "%6.3g  average = leaf-ancestor score" % la[-1]
			if param["TED"]:
				print "Tree-dist: %g / %g = %g" % (
					ted, denom, 1 - ted / float(denom))
				newtreedist(gtree, ctree, True)
			if param["DEP"]:
				print "Sentence:", " ".join(gsent)
				print "dependencies gold                                   cand"
				for (_, a, b), (_, c, d) in zip(gdep, cdep):
					# use original sentences because we don't delete
					# punctuation for dependency evaluation
					print "%15s -> %15s           %15s -> %15s" % (
						gold_sents[n][a-1][0], gold_sents[n][b-1][0],
						cand_sents[n][c-1][0], cand_sents[n][d-1][0])

	breakdowns(param, goldb40, candb40, goldpos40, candpos40, goldbcat40,
			candbcat40, maxlenseen)
	msg = summary(param, goldb, candb, goldpos, candpos, sentcount,
			maxlenseen, exact, la, dicenoms, dicedenoms, golddep, canddep,
			goldb40, candb40, goldpos40, candpos40, sentcount40, maxlenseen40,
			exact40, la40, dicenoms40, dicedenoms40, golddep40, canddep40)
	return msg

def breakdowns(param, goldb, candb, goldpos, candpos, goldbcat, candbcat,
		maxlenseen):
	""" Print breakdowns for the most frequent labels / tags. """
	if param["LABELED"] and param["DEBUG"] != -1:
		print
		print " Category Statistics (10 most frequent categories / errors)",
		if maxlenseen > param["CUTOFF_LEN"]:
			print "for length <= %d" % param["CUTOFF_LEN"],
		print
		print "  label  % gold  recall    prec.     F1",
		print "          test/gold   count"
		print "_______________________________________",
		print "       ____________________"
		gmismatch = {(n, indices): label
					for n, (label, indices) in (goldb - candb).keys()}
		wrong = multiset((label, gmismatch[n, indices])
					for n, (label, indices) in (candb - goldb).keys()
					if (n, indices) in gmismatch)
		freqcats = sorted(set(goldbcat) | set(candbcat),
				key=lambda x: len(goldbcat[x]), reverse=True)
		for cat, mismatch in izip_longest(freqcats[:10], wrong.most_common(10)):
			if cat is None:
				print "                                       ",
			else:
				print "%s  %6.2f  %s  %s  %s" % (
					cat.rjust(7),
					100 * sum(goldbcat[cat].values()) / float(len(goldb)),
					nozerodiv(lambda: recall(goldbcat[cat], candbcat[cat])),
					nozerodiv(lambda: precision(goldbcat[cat], candbcat[cat])),
					nozerodiv(lambda: f_measure(goldbcat[cat], candbcat[cat]))),
			if mismatch is not None:
				print "       %s %7d" % (
						"/".join(mismatch[0]).rjust(12), mismatch[1]),
			print

		if accuracy(goldpos, candpos) != 1:
			print
			print " Tag Statistics (10 most frequent tags / errors)",
			if maxlenseen > param["CUTOFF_LEN"]:
				print "for length <= %d" % param["CUTOFF_LEN"],
			print "\n    tag  % gold  recall   prec.      F1",
			print "          test/gold   count"
			print "_______________________________________",
			print "       ____________________"
			tags = multiset(tag for _, tag in goldpos)
			wrong = multiset((g, c)
					for (_, g), (_, c) in zip(goldpos, candpos) if g != c)
			for tag, mismatch in izip_longest(tags.most_common(10),
					wrong.most_common(10)):
				if tag is None:
					print "".rjust(40),
				else:
					goldtag = multiset(n for n, (w, t) in enumerate(goldpos)
							if t == tag[0])
					candtag = multiset(n for n, (w, t) in enumerate(candpos)
							if t == tag[0])
					print "%s  %6.2f  %6.2f  %6.2f  %6.2f" % (
							tag[0].rjust(7),
							100 * len(goldtag) / float(len(goldpos)),
							100 * recall(goldtag, candtag),
							100 * precision(goldtag, candtag),
							100 * f_measure(goldtag, candtag)),
				if mismatch is not None:
					print "       %s %7d" % (
						"/".join(mismatch[0]).rjust(12), mismatch[1]),
				print
		print

def summary(param, goldb, candb, goldpos, candpos, sentcount, maxlenseen,
		exact, la, dicenoms, dicedenoms, golddep, canddep,
		goldb40, candb40, goldpos40, candpos40, sentcount40, maxlenseen40,
		exact40, la40, dicenoms40, dicedenoms40, golddep40, canddep40):
	""" Return overview with scores for all sentences. """
	discbrackets = sum(1 for n, (a, b) in candb.elements()
			if b != set(range(min(b), max(b)+1)))
	gdiscbrackets = sum(1 for n, (a, b) in goldb.elements()
			if b != set(range(min(b), max(b)+1)))

	if maxlenseen <= param["CUTOFF_LEN"]:
		msg = ["%s" % " Summary (ALL) ".center(35, '_'),
			"number of sentences:       %6d" % (sentcount),
			"longest sentence:          %6d" % (maxlenseen)]
		if gdiscbrackets or discbrackets:
			msg.extend(["gold brackets (disc.):     %6d (%d)" % (
					len(goldb), gdiscbrackets),
				"cand. brackets (disc.):    %6d (%d)" % (
					len(candb), discbrackets)])
		else:
			msg.extend(["gold brackets:             %6d" % len(goldb),
				"cand. brackets:            %6d" % len(candb)])
		msg.extend([
				"labeled recall:            %s" % (
					nozerodiv(lambda: recall(goldb, candb))),
				"labeled precision:         %s" % (
					nozerodiv(lambda: precision(goldb, candb))),
				"labeled f-measure:         %s" % (
					nozerodiv(lambda: f_measure(goldb, candb))),
				"exact match:               %s" % (
					nozerodiv(lambda: exact / sentcount)),
				"leaf-ancestor:             %s" % (
					nozerodiv(lambda: mean(la)))])
		if param["TED"]:
			msg.append("tree-dist (Dice micro avg) %s" % (
					nozerodiv(lambda: 1 - dicenoms / float(dicedenoms))))
		if param["DEP"]:
			msg.append("unlabeled dependencies:    %s" % (
					nozerodiv(lambda: accuracy(golddep, canddep))))
		msg.append("tagging accuracy:          %s" % (
				nozerodiv(lambda: accuracy(goldpos, candpos))))
		return "\n".join(msg)

	discbrackets40 = sum(1 for n, (a, b) in candb40.elements()
			if b != set(range(min(b), max(b)+1)))
	gdiscbrackets40 = sum(1 for n, (a, b) in goldb40.elements()
			if b != set(range(min(b), max(b)+1)))
	msg = ["%s <= %d ______ ALL" % (
			" Summary ".center(27, "_"), param["CUTOFF_LEN"]),
		"number of sentences:       %6d     %6d" % (sentcount40, sentcount),
		"longest sentence:          %6d     %6d" % (maxlenseen40, maxlenseen),
		"gold brackets:             %6d     %6d" % (len(goldb40), len(goldb)),
		"cand. brackets:            %6d     %6d" % (len(candb40), len(candb))]
	if gdiscbrackets or discbrackets:
		msg.extend(["disc. gold brackets:       %6d     %6d" % (
				gdiscbrackets40, gdiscbrackets),
				"disc. cand. brackets:      %6d     %6d" % (
				discbrackets40, discbrackets)])
	msg.extend(["labeled recall:            %s     %s" % (
			nozerodiv(lambda: recall(goldb40, candb40)),
			nozerodiv(lambda: recall(goldb, candb))),
		"labeled precision:         %s     %s" % (
			nozerodiv(lambda: precision(goldb40, candb40)),
			nozerodiv(lambda: precision(goldb, candb))),
		"labeled f-measure:         %s     %s" % (
			nozerodiv(lambda: f_measure(goldb40, candb40)),
			nozerodiv(lambda: f_measure(goldb, candb))),
		"exact match:               %s     %s" % (
			nozerodiv(lambda: exact40 / sentcount40),
			nozerodiv(lambda: exact / sentcount)),
		"leaf-ancestor:             %s     %s" % (
			nozerodiv(lambda: mean(la40)),
			nozerodiv(lambda: mean(la)))])
	if param["TED"]:
		msg.append("tree-dist (Dice micro avg) %s     %s" % (
			nozerodiv(lambda: (1 - dicenoms40 / float(dicedenoms40))),
			nozerodiv(lambda: (1 - dicenoms / float(dicedenoms)))))
	if param["DEP"]:
		msg.append("unlabeled dependencies:    %s     %s  (%d / %d)" % (
				nozerodiv(lambda: accuracy(golddep40, canddep40)),
				nozerodiv(lambda: accuracy(golddep, canddep)),
				len(filter(lambda (a, b): a ==b, zip(golddep, canddep))),
				len(golddep)))
	msg.append("tagging accuracy:          %s     %s" % (
			nozerodiv(lambda: accuracy(goldpos40, candpos40)),
			nozerodiv(lambda: accuracy(goldpos, candpos))))
	return "\n".join(msg)

def readparam(filename):
	""" read an EVALB-style parameter file and return a dictionary. """
	param = defaultdict(list)
	# NB: we ignore MAX_ERROR, we abort immediately on error.
	validkeysonce = "DEBUG MAX_ERROR CUTOFF_LEN LABELED DISC_ONLY".split()
	param = { "DEBUG" : 0, "MAX_ERROR": 10, "CUTOFF_LEN" : 40,
				"LABELED" : 1, "DELETE_LABEL_FOR_LENGTH" : [],
				"DELETE_LABEL" : [], "EQ_LABEL" : set(), "EQ_WORD" : set(),
				"DISC_ONLY" : 0, "PRESERVE_FUNCTIONS": 0, "TED": 0, "DEP": 0}
	seen = set()
	for a in open(filename) if filename else ():
		line = a.strip()
		if line and not line.startswith("#"):
			key, val = line.split(None, 1)
			if key in validkeysonce:
				assert key not in seen, "cannot declare %s twice" % key
				seen.add(key)
				param[key] = int(val)
			elif key in ("DELETE_LABEL", "DELETE_LABEL_FOR_LENGTH"):
				param[key].append(val)
			elif key in ("EQ_LABEL", "EQ_WORD"):
				# these are given as undirected pairs
				# will be represented as equivalence classes A => {A, B, C, D}
				try:
					b, c = val.split()
				except ValueError:
					raise ValueError("%s requires two values" % key)
				param[key].add((b, c))
			else:
				raise ValueError("unrecognized parameter key: %s" % key)
	# transitive closure of (undirected) EQ relations with DFS
	for key in ("EQ_LABEL", "EQ_WORD"):
		connectedcomponents = {}
		seen = set()
		for k, v in param[key]:
			if k in seen or v in seen:
				continue
			connectedcomponents[k] = {k, v}
			agenda = [x for x in param[key] if k in x or v in x]
			while agenda:
				a, b = agenda.pop()
				connectedcomponents[k].update((a, b))
				if a not in seen:
					agenda.extend(x for x in param[key] if a in x)
				if b not in seen:
					agenda.extend(x for x in param[key] if b in x)
				seen.update((a, b))
		# reverse mappping {'A': {'A', 'B'}} => {'A': 'A', 'B': 'A'}
		param[key] = {x: k for k in connectedcomponents
				for x in connectedcomponents[k]}
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
				x = b.node.find("-")
				y = b.node.find("=")
				if x >= 1:
					a.node = b.node[:x]
				if y >= 0:
					a.node = b.node[:y]
			b.node = eqlabel.get(b.node, b.node)
			if not b:
				a.pop(n)  #remove empty nodes
			elif isinstance(b[0], Tree):
				if b.node in delete:
					# replace phrasal node with its children
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
	leafmap = {m: n for n, m in enumerate(leaves)}
	for a in posnodes:
		a[0] = leafmap[a[0]]
		a.indices = [a[0]]
		if sent[a[0]] in eqword:
			sent[a[0]] = eqword[sent[a[0]]]
	# cache spans
	for a in reversed(list(tree.subtrees())):
		if not a:
			a.indices = []
		elif isinstance(a[0], Tree):
			a.indices = []
			for b in a:
				a.indices.extend(b.indices)
		else:
			a.indices = [a[0]]

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

def strbracketings(brackets):
	""" Return a string with a concise representation of a bracketing.

	>>> strbracketings(set([('S', frozenset([0, 1, 2])), \
			('VP', frozenset([0, 2]))]))
	'S[0-2], VP[0,2]'
	"""
	if not brackets:
		return "{}"
	return ", ".join("%s[%s]" % (a, ",".join(
		"-".join(str(y) for y in sorted(set(x)))
		for x in intervals(sorted(b)))) for a, b in brackets)

def leafancestorpaths(tree, delete):
	""" Generate a list of ancestors for each leaf node in a tree. """
	#uses [] to mark components, and () to mark constituent boundaries
	#deleted words/tags should not affect boundary detection
	paths = {a: [] for a in tree.indices}
	# do a top-down level-order traversal
	thislevel = [tree]
	while thislevel:
		nextlevel = []
		for n in thislevel:
			leaves = sorted(n.indices)
			# skip empty nodes and POS tags
			if not leaves or not isinstance(n[0], Tree):
				continue
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
	""" Get edit distance for two leaf-ancestor paths. """
	#catch the case of empty lineages
	if gold == cand:
		return 1.0
	return max(0, (1.0 - edit_distance(cand, gold)
					/ float(len(gold) + len(cand))))

def leafancestor(goldtree, candtree, delete):
	""" Geoffrey Sampson, Anna Babarcz (2003):
	A test of the leaf-ancestor metric for parse accuracy """
	gold = leafancestorpaths(goldtree, delete)
	cand = leafancestorpaths(candtree, delete)
	return mean([pathscore(gold[leaf], cand[leaf]) for leaf in gold])

def treedisteval(a, b, includeroot=False, debug=False):
	""" Get tree-distance for two trees and compute the Dice normalization. """
	ted = treedist(a, b, debug)
	# Dice denominator
	denom = len(list(a.subtrees()) + list(b.subtrees()))
	# optionally discount ROOT nodes and preterminals
	if not includeroot:
		denom -= 2
	#if not includepreterms:
	#	denom -= len(a.leaves() + b.leaves())
	return ted, denom

# If the goldfile contains n constituents for the same span, and the parsed
# file contains m constituents with that nonterminal, the scorer works as
# follows:
#
# i) If m>n, then the precision is n/m, recall is 100%
#
# ii) If n>m, then the precision is 100%, recall is m/n.
#
# iii) If n==m, recall and precision are both 100%.
def recall(reference, candidate):
	""" Get recall score for two multisets. """
	if not reference:
		return float('nan')
	return sum(min(reference[a], candidate[a])
			for a in reference & candidate) / float(
			sum(reference.values()))

def precision(reference, candidate):
	""" Get precision score for two multisets. """
	if not candidate:
		return float('nan')
	return sum(min(reference[a], candidate[a])
			for a in reference & candidate) / float(
			sum(candidate.values()))

def f_measure(reference, candidate, alpha=0.5):
	""" Get F1-measure for two multisets. """
	p = precision(reference, candidate)
	r = recall(reference, candidate)
	if p == 0 or r == 0:
		return 0
	return 1.0/(alpha/p + (1-alpha)/r)

def accuracy(reference, candidate):
	""" Given a sequence of reference values and a corresponding sequence of
	test values, return the fraction of corresponding values that are equal.
	In particular, return the fraction of indices
	0<i<=len(test) such that test[i] == reference[i]. """
	assert len(reference) == len(candidate), (
		"Sequences must have the same length.")
	return sum(1 for a, b in zip(reference, candidate) if a == b) / float(
		len(reference))

def harmean(seq):
	""" Compute harmonic mean of a sequence. """
	try:
		return len([a for a in seq if a]) / sum(1./a for a in seq if a)
	except ZeroDivisionError:
		return float('nan')

def mean(seq):
	""" Compute arithmetic mean of a sequence. """
	return sum(seq) / float(len(seq))

def splitpath(path):
	""" Split path into a pair of (directory, filename). """
	if "/" in path:
		return path.rsplit("/", 1)
	else:
		return ".", path

def intervals(s):
	""" Partition s into a sequence of intervals corresponding to contiguous
	ranges. An interval is a pair (a, b), with a <= b denoting terminals x
	such that a <= x <= b.

	>>> list(intervals((0, 1, 3, 4, 6, 7, 8)))
	[(0, 1), (3, 4), (6, 8)]"""
	start = prev = None
	for a in s:
		if start is None:
			start = prev = a
		elif a == prev + 1:
			prev = a
		else:
			yield start, prev
			start = prev = a
	if start is not None:
		yield start, prev

def disc(node):
	""" This function evaluates whether a particular node is locally
	discontinuous.  The root node will, by definition, be continuous.
	Nodes can be continuous even if some of their children are discontinuous.
	"""
	if not isinstance(node, Tree):
		return False
	start = prev = None
	for a in sorted(node.indices):
		if start is None:
			start = prev = a
		elif a == prev + 1:
			prev = a
		else:
			return True
	return False

def nozerodiv(a):
	""" Convenience function to catch zero division or None as a result,
	and otherwise format as a percentage with two decimals. """
	try:
		result = a()
	except ZeroDivisionError:
		return ' 0DIV!'
	return '  None' if result is None else "%6.2f" % (100 * result)


def edit_distance(s1, s2):
	""" Calculate the Levenshtein edit-distance between two strings. The edit
	distance is the number of characters that need to be substituted, inserted,
	or deleted, to transform s1 into s2.  For example, transforming "rain" to
	"shine" requires three steps, consisting of two substitutions and one
	insertion: "rain" -> "sain" -> "shin" -> "shine".  These operations could
	have been done in other orders, but at least three steps are needed.
	"""
	# set up a 2-D array
	len1 = len(s1)
	len2 = len(s2)
	# initialize 2-D array to zero
	lev = [[0] * (len2 + 1) for _ in range(len1 + 1)]
	for i in range(len1 + 1):
		lev[i][0] = i           # column 0: 0,1,2,3,4,...
	for j in range(len2 + 1):
		lev[0][j] = j           # row 0: 0,1,2,3,4,...

	# iterate over the array
	for i in range(len1):
		for j in range (len2):
			a = lev[i][j + 1] + 1               # skipping s1[i]
			b = lev[i][j] + (s1[i] != s2[j])    # matching s1[i] with s2[j]
			c = lev[i + 1][j] + 1               # skipping s2[j]
			lev[i + 1][j + 1] = min(a, b, c)    # pick the cheapest
	return lev[len1][len2]

def test():
	""" Simple sanity check; should give 100% score on all metrics. """
	gold = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1")
	parses = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1")
	doeval(gold.parsed_sents(),
			gold.tagged_sents(),
			parses.parsed_sents(),
			parses.tagged_sents(),
			readparam(None))

if __name__ == '__main__':
	main()
