import sys, os.path
from itertools import count, izip, islice
from collections import defaultdict
from collections import Counter as multiset
from nltk import Tree, FreqDist
from nltk.metrics import accuracy, edit_distance
from treebank import NegraCorpusReader
from grammar import ranges
from treetransforms import disc

def readparams(file):
	""" read an EVALB-style parameter file and return a dictionary. """
	params = defaultdict(list)
	# NB: we ignore MAX_ERROR, we abort immediately on error.
	validkeysonce = "DEBUG MAX_ERROR CUTOFF_LEN LABELED DISC_ONLY".split()
	params = { "DEBUG" : 1, "MAX_ERROR": 10, "CUTOFF_LEN" : 40,
					"LABELED" : 1, "DISC_ONLY" : 0,
					"DELETE_LABEL" : [], "DELETE_LABEL_FOR_LENGTH" : [],
					"EQ_LABEL" : [], "EQ_WORD" : [] }
	seen = set()
	for a in open(file) if file else ():
		line = a.strip()
		if line and not line.startswith("#"):
			key, val = line.split(None, 1)
			if key in validkeysonce:
				assert key not in seen, "cannot declare parameter %s twice" % key
				seen.add(key)
				params[key] = int(val)
			elif key in ("DELETE_LABEL", "DELETE_LABEL_FOR_LENGTH"):
				params[key].append(val)
			elif key in ("EQ_LABEL", "EQ_WORD"):
				hd = val.split()[0]
				params[key].append(dict((a, hd) for a in val.split()[1:]))
			else:
				raise ValueError("unrecognized parameter key: %s" % key)
	return params

def transform(tree, sent, delete, eqlabel, eqword):
	""" Apply the transformations according to the parameter file. """
	for a in reversed(list(tree.subtrees(lambda n: isinstance(n[0], Tree)))):
		for n, b in zip(count(), a)[::-1]:
			if not b: a.pop(n)	#remove empty nodes
			elif b.node in delete:
				# replace phrasal node with its children; remove pre-terminal entirely
				if isinstance(b[0], Tree): a[n:n+1] = b
				else: a.pop(n)
			else: b.node = getlabel(b.node, eqlabel)
	if eqword:
		for a in tree.treepositions('leaves'):
			if tree[a] in eqword: tree[a] = eqword[tree[a]]

def getlabel(label, eqlabel):
	for a in eqlabel:
		if label in a: return a[label]
	return label

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

def treedist(a, b, includepreterms=True, includeroot=False):
	from tdist import Node, treedistance
	def treetonode(tree):
		""" Convert NLTK tree object to a different format used by tdist.
		Ignores terminals and optionally pre-terminals."""
		#options: delete preterms but keep terms, discount them
		result = Node(tree.node)
		for a in tree:
			if isinstance(a[0], Tree): result.addkid(treetonode(a))
			#elif includepreterms: result.addkid(Node(a.node+str(a[0])))
			elif includepreterms: result.addkid(Node(a.node).addkid(Node(a[0])))
			else: result.addkid(Node(a[0]))
		return result
	ted = treedistance(treetonode(a), treetonode(b))
	# Dice denominator
	denom = len(list(a.subtrees())) + len(list(b.subtrees()))
	# optionally discount ROOT nodes and preterminals
	if not includeroot: denom -= 2
	if not includepreterms: denom -= 2 * len(a.leaves())
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

def printbrackets(brackets):
	return ", ".join("%s[%s]" % (a,
		",".join(map(lambda x: "%s-%s" % (x[0], x[-1])
		if len(x) > 1 else str(x[0]), ranges(sorted(b)))))
		for a,b in brackets)

def leafancestorpaths(tree):
	paths = dict((a, []) for a in tree.leaves())
	# skip root node; skip POS tags
	for a in islice(tree.subtrees(lambda n: isinstance(n[0], Tree)), 1, None):
		leaves = a.leaves()
		for b in a.leaves():
			# mark beginning of components
			if len(leaves) > 1:
				if b - 1 not in leaves and "(" not in paths[b]:
					paths[b].append("(")
			# add this label to the lineage
			paths[b].append(a.node)
			# mark end of components
			if len(leaves) > 1:
				if b + 1 not in leaves and ")" not in paths[b]:
					paths[b].append(")")
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

def nonetozero(a):
	try: result = a()
	except ZeroDivisionError: return 0
	return 0 if result is None else result

def harmean(seq):
	try: return len([a for a in seq if a]) / sum(1./a for a in seq if a)
	except: return "zerodiv"

def mean(seq):
	return sum(seq) / float(len(seq)) if seq else None #"zerodiv"

def splitpath(path):
	if "/" in path: return path.rsplit("/", 1)
	else: return ".", path

def main(goldfile, parsesfile, goldencoding='utf-8', parsesencoding='utf-8'):
	assert os.path.exists(goldfile), "gold file not found"
	assert os.path.exists(parsesfile), "parses file not found"
	gold = NegraCorpusReader(*splitpath(goldfile), encoding=goldencoding)
	parses = NegraCorpusReader(*splitpath(parsesfile), encoding=parsesencoding)
	param = readparams(sys.argv[3] if len(sys.argv) >= 4 else None)
	if len(sys.argv) == 5: param['CUTOFF_LEN'] = int(sys.argv[4])
	
	goldlen = len(gold.parsed_sents())
	parseslen = len(parses.parsed_sents())
	assert goldlen == parseslen, "unequal number of sentences in gold & candidates: %d vs %d" % (goldlen, parseslen)

	if param["DEBUG"]:
		print "Parameters:"
		for a in param: print "%s\t%s" % (a, param[a])
		print """
   Sentence                 Matched   Brackets            Corr      Tag
  ID Length  Recall  Precis Bracket   gold   test  Words  Tags Accuracy    LA
______________________________________________________________________________"""
	exact = 0.
	maxlenseen = sentcount = dicenoms = dicedenoms = 0
	goldpos = []
	candpos = []
	la = []
	goldb = multiset()
	candb = multiset()
	goldbcat = defaultdict(multiset)
	candbcat = defaultdict(multiset)
	for n, ctree, csent, gtree, gsent in izip(count(1), parses.parsed_sents(), parses.sents(), gold.parsed_sents(), gold.sents()):
		cpos = sorted(ctree.pos())
		gpos = sorted(gtree.pos())
		lencpos = sum(1 for a,b in cpos if b not in param["DELETE_LABEL_FOR_LENGTH"])
		lengpos = sum(1 for a,b in gpos if b not in param["DELETE_LABEL_FOR_LENGTH"])
		assert lencpos == lengpos, "sentence length mismatch"
		if lencpos > param["CUTOFF_LEN"]: continue
		sentcount += 1
		if maxlenseen < lencpos: maxlenseen = lencpos
		transform(ctree, csent, param["DELETE_LABEL"], param["EQ_LABEL"], param["EQ_WORD"])
		transform(gtree, gsent, param["DELETE_LABEL"], param["EQ_LABEL"], param["EQ_WORD"])
		assert csent == gsent, "candidate & gold sentences do not match:\n%s\%s" % (
				" ".join(csent), " ".join(gsent))
		cbrack = bracketings(ctree, param["LABELED"], set([gtree.node]).intersection(param["DELETE_LABEL"]), param["DISC_ONLY"])
		gbrack = bracketings(gtree, param["LABELED"], set([gtree.node]).intersection(param["DELETE_LABEL"]), param["DISC_ONLY"])
		if cbrack == gbrack: exact += 1
		candb.update((n,a) for a in cbrack.elements())
		goldb.update((n,a) for a in gbrack.elements())
		for a in gbrack: goldbcat[a[0]][(n, a)] += 1
		for a in cbrack: candbcat[a[0]][(n, a)] += 1
		goldpos.extend(gpos)
		candpos.extend(cpos)
		la.append(leafancestor(gtree, ctree))
		if la[-1] == 1 and gtree != ctree:
			print gtree
			print ctree
			g = leafancestorpaths(gtree)
			c = leafancestorpaths(ctree)
			print g
			print c
			print [pathscore(g[leaf], c[leaf]) for leaf in g]
			assert False
		#ted, denom = treedist(gtree, ctree, includeroot=gtree.node not in param["DELETE_LABEL"])
		#dicenoms += ted
		#dicedenoms += denom
		if param["DEBUG"] == 0: continue
		print "%4d  %5d  %6.2f  %6.2f   %5d  %5d  %5d  %5d  %4d  %6.2f %6.2f" % ( # %2d %g" % (
			n,
			len(gpos),
			100 * nonetozero(lambda: recall(gbrack, cbrack)),
			100 * nonetozero(lambda: precision(gbrack, cbrack)),
			sum((gbrack & cbrack).values()),
			sum(gbrack.values()),
			sum(cbrack.values()),
			lengpos,
			sum(1 for a,b in zip(gpos, cpos) if a==b),
			100 * accuracy(gpos, cpos),
			100 * la[-1],
			#ted,
			#1 if ted == 0 else 1 - ted / float(denom)
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
					for n,(label,indices) in (goldb - candb).keys())
		wrong = FreqDist((label, gmismatch[n, indices])
					for n,(label,indices) in (candb - goldb).keys()
					if (n, indices) in gmismatch)
		for labels, freq in wrong.items()[:10]:
			print "%s %7d" % ("/".join(labels).rjust(12), freq)
	discbrackets = sum(len(bracketings(tree, disconly=True))
			for tree in parses.parsed_sents())
	gdiscbrackets = sum(len(bracketings(tree, disconly=True))
			for tree in gold.parsed_sents())

	print "\n____________ Summary <= %d _______" % param["CUTOFF_LEN"]
	print "number of sentences:       %6d" % (n)
	print "maximum length:            %6d" % (maxlenseen)
	print "gold brackets (disc.):     %6d (%d)" % (len(goldb), gdiscbrackets)
	print "cand. brackets (disc.):    %6d (%d)" % (len(candb), discbrackets)
	print "labeled precision:         %6.2f" % (100 * nonetozero(lambda: precision(goldb, candb)))
	print "labeled recall:            %6.2f" % (100 * nonetozero(lambda: recall(goldb, candb)))
	print "labeled f-measure:         %6.2f" % (100 * nonetozero(lambda: f_measure(goldb, candb)))
	print "exact match:               %6.2f" % (100 * (exact / sentcount))
	print "leaf-ancestor:             %6.2f" % (100 * mean(la))
	#print "tree-dist (Dice micro avg) %6.2f" % (100 * (1 - dicenoms / float(dicedenoms)))
	print "tagging accuracy:          %6.2f" % (100 * accuracy(goldpos, candpos))

if __name__ == '__main__':
	if len(sys.argv) == 2 and sys.argv[1] == "--test":
		main("sample2.export", "sample2.export", 'iso-8859-1', 'iso-8859-1')
	elif 3 <= len(sys.argv) <= 5:
		goldfile, parsesfile = sys.argv[1:3]
		if "/" in goldfile and not os.path.exists(goldfile):
			goldfile, goldencoding = goldfile.rsplit("/", 1)
		if "/" in parsesfile and not os.path.exists(parsesfile):
			goldfile, goldencoding = goldfile.rsplit("/", 1)
		main(goldfile, parsesfile)
	else:
		print """\
wrong number of arguments. usage: %s gold[/encoding] parses[/encoding] [param [cutoff]]
(where gold and parses are files in export format, param is in EVALB format,
and cutoff is an integer for the length cutoff which overrides the parameter
file. Encoding defaults to UTF-8, other encodings need to be specified.)

Example:	%s sample2.export/iso-8859-1 myparses.export TEST.prm\
""" % (sys.argv[0], sys.argv[0])
