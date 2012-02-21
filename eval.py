# -*- coding: UTF-8 -*-
import sys
from itertools import count, izip
from operator import itemgetter
from collections import defaultdict
from collections import Counter as multiset
from nltk import Tree, FreqDist
from nltk.metrics import accuracy, edit_distance
from negra import NegraCorpusReader
from grammar import ranges
from treetransforms import disc

def recall(reference, test):
	return float(sum((reference & test).values()))/sum(reference.values())

def precision(reference, test):
	return float(sum((reference & test).values()))/sum(test.values())

def f_measure(reference, test, alpha=0.5):
	p = precision(reference, test)
	r = recall(reference, test)
	if p == 0 or r == 0: return 0
	return 1.0/(alpha/p + (1-alpha)/r)

def readparams(file):
	""" read a EVALB-style parameter file and return a dictionary. """
	params = defaultdict(list)
	# NB: we ignore MAX_ERROR, we abort immediately on error.
	# not yet implemented: DELETE_LABEL_FOR_LENGTH, EQ_WORD
	validkeysonce = "DEBUG MAX_ERROR CUTOFF_LEN LABELED DISC_ONLY".split()
	params = { "DEBUG" : 1, "MAX_ERROR": 10, "CUTOFF_LEN" : 40,
					"LABELED" : 1, "DISC_ONLY" : 0,
					"DELETE_LABEL" : [], "DELETE_LABEL_FOR_LENGTH" : [],
					"EQ_LABEL" : [], "EQ_WORDS" : [] }
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
				params[key].append(dict((a, hd) for a in val.split()))
			else:
				raise ValueError("unrecognized parameter key: %s" % key)
	return params

def bracketings(tree, labeled=True, delete=(), eqlabel={}, disconly=False):
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
	# collect all leaves dominated by nodes to be deleted:
	deleted = frozenset(leaf for subtree in tree.subtrees(lambda x: x.node in delete and not isinstance(x[0], Tree)) for leaf in subtree.leaves())
	return multiset( (getlabel(a.node, eqlabel) if labeled else "",
				frozenset(a.leaves()) - deleted)
				for a in tree.subtrees()
					if isinstance(a[0], Tree)
					and a.node not in delete
					and (not disconly or disc(a)))

def getlabel(label, eqlabel):
	for a in eqlabel:
		if label in a: return a[label]
	return label

def printbrackets(brackets):
	return ", ".join("%s[%s]" % (a,
					",".join(map(lambda x: "%s-%s" % (x[0], x[-1])
					if len(x) > 1 else str(x[0]), ranges(sorted(b)))))
					for a,b in brackets)

def leafancestorpaths(tree):
	paths = dict((a, []) for a in tree.leaves())
	# skip root label; skip POS tags
	for a in tree.subtrees(lambda n: n != tree and isinstance(n[0], Tree)):
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
	return max(0, (1.0 if gold == cand else
			1.0 - (2.0 * edit_distance(cand, gold))
					/ (len(gold) + len(cand))))

def leafancestor(goldtree, candtree):
	""" Geoffrey Sampson, Anna Babarcz (2003):
	A test of the leaf-ancestor metric for parse accuracy """
	gold = leafancestorpaths(goldtree)
	cand = leafancestorpaths(candtree)
	return mean([pathscore(gold[leaf], cand[leaf]) for leaf in gold])

def nonetozero(a):
	try:
		result = a()
		return 0 if result is None else result
	except ZeroDivisionError:
		return 0

def harmean(seq):
	try: return len([a for a in seq if a]) / sum(1./a if a else 0. for a in seq)
	except: return "zerodiv"

def mean(seq):
	return sum(seq) / float(len(seq)) if seq else None #"zerodiv"

def export(tree, sent, n):
	""" Convert a tree with indices as leafs and a sentence with the
	corresponding non-terminals to a single string in Negra's export format.
	NB: IDs do not follow the convention that IDs of children are all lower. """
	result = ["#BOS %d" % n]
	wordsandpreterminals = tree.treepositions('leaves') + [a[:-1] for a in tree.treepositions('leaves')]
	nonpreterminals = list(sorted([a for a in tree.treepositions() if a not in wordsandpreterminals and a != ()], key=len, reverse=True))
	wordids = dict((tree[a], a) for a in tree.treepositions('leaves'))
	for i, word in enumerate(sent):
		idx = wordids[i]
		result.append("\t".join((word[0],
				tree[idx[:-1]].node.replace("$[","$("),
				"--", "--",
				str(500+nonpreterminals.index(idx[:-2]) if len(idx) > 2 else 0))))
	for idx in nonpreterminals:
		result.append("\t".join(("#%d" % (500 + nonpreterminals.index(idx)),
				tree[idx].node,
				"--", "--",
				str(500+nonpreterminals.index(idx[:-1]) if len(idx) > 1 else 0))))
	result.append("#EOS %d" % n)
	return "\n".join(result) #.encode("utf-8")

def splitpath(path):
	if "/" in path: return path.rsplit("/", 1)
	else: return ".", path

def main():
	if len(sys.argv) not in (3, 4):
		print "wrong number of arguments. usage: %s gold parses [param]" % sys.argv[0]
		print "(where gold and parses are files in export format, param is in EVALB format)" 
		return
	gold = NegraCorpusReader(*splitpath(sys.argv[1]))
	parses = NegraCorpusReader(*splitpath(sys.argv[2]))
	param = readparams(sys.argv[3] if len(sys.argv) == 4 else None)
	goldlen = len(gold.parsed_sents())
	parseslen = len(parses.parsed_sents())
	assert goldlen == parseslen

	if param["DEBUG"]:
		print "param =", param
		print """\
   Sentence                 Matched   Brackets            Corr      Tag
  ID Length  Recall  Precis Bracket   gold   test  Words  Tags Accuracy    LA
______________________________________________________________________________"""
	exact = 0.
	maxlenseen = 0
	sentcount = 0
	goldpos = []
	candpos = []
	la = []
	goldb = multiset()
	candb = multiset()
	goldbcat = defaultdict(multiset)
	candbcat = defaultdict(multiset)
	for n, csent, gsent in izip(count(1), parses.parsed_sents(), gold.parsed_sents()):
		cpos = sorted(csent.pos())
		gpos = sorted(gsent.pos())
		lencpos = sum(1 for a,b in cpos if b not in param["DELETE_LABEL_FOR_LENGTH"])
		lengpos = sum(1 for a,b in gpos if b not in param["DELETE_LABEL_FOR_LENGTH"])
		assert lencpos == lengpos, "sentence length mismatch"
		if lencpos > param["CUTOFF_LEN"]: continue
		sentcount += 1
		if maxlenseen < lencpos: maxlenseen = lencpos
		cbrack = bracketings(csent, param["LABELED"], param["DELETE_LABEL"],
									param["EQ_LABEL"], param["DISC_ONLY"])
		gbrack = bracketings(gsent, param["LABELED"], param["DELETE_LABEL"],
									param["EQ_LABEL"], param["DISC_ONLY"])
		if cbrack == gbrack: exact += 1
		candb.update((n,a) for a in cbrack.elements())
		goldb.update((n,a) for a in gbrack.elements())
		for a in gbrack: goldbcat[a[0]][(n, a)] += 1
		for a in cbrack: candbcat[a[0]][(n, a)] += 1
		goldpos.extend(gpos)
		candpos.extend(cpos)
		la.append(leafancestor(gsent, csent))
		if param["DEBUG"] == 0: continue
		print "%4d  %5d  %6.2f  %6.2f   %5d  %5d  %5d  %5d  %4d  %6.2f %6.2f" % (
			n,
			len(gpos),
			100 * nonetozero(lambda: recall(gbrack, cbrack)),
			100 * nonetozero(lambda: precision(gbrack, cbrack)),
			sum((gbrack & cbrack).values()),
			sum(gbrack.values()),
			sum(cbrack.values()),
			lengpos, # how is words supposed to be different from len?? should we leave out punctuation or something?
			sum(1 for a,b in zip(gpos, cpos) if a==b),
			100 * accuracy(gpos, cpos),
			100 * la[-1]
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
	print "tagging accuracy:          %6.2f" % (100 * accuracy(goldpos, candpos))

if __name__ == '__main__': main()
