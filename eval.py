# -*- coding: UTF-8 -*-
import sys
from itertools import count, izip
from operator import itemgetter
from collections import defaultdict
from nltk import Tree, FreqDist
from nltk.metrics import precision, recall, f_measure, accuracy, edit_distance
from negra import NegraCorpusReader
from grammar import ranges
#import plac

def bracketings(tree):
	""" Return the labeled set of bracketings for a tree: 
	for each nonterminal node, the set will contain a tuple with the label and
	the set of terminals which it dominates.
	>>> bracketings(Tree("(S (NP 1) (VP (VB 0) (JJ 2)))"))
	frozenset([('VP', frozenset(['0', '2'])),
				('S', frozenset(['1', '0', '2']))])
	"""
	return frozenset( (a.node, frozenset(a.leaves()) )
				for a in tree.subtrees()
				if isinstance(a[0], Tree))

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
			1.0 - (2.0 * edit_distance(gold, cand))
					/ (len(gold) + len(cand))))

def leafancestor(goldtree, candtree):
	""" Geoffrey Sampson (2000):
	A proposal for improving the measurement of parse accuracy """
	gold = leafancestorpaths(goldtree)
	cand = leafancestorpaths(candtree)
	return mean([pathscore(gold[leaf], cand[leaf]) for leaf in gold])

def nonetozero(a):
	return 0 if a is None else a

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


def main():
	if len(sys.argv) not in (3, 4):
		print "wrong number of arguments. usage: %s gold parses [maxsent]" % sys.argv[0]
		print "(where gold and parses are files in export format)" 
		return
	gold = NegraCorpusReader(".", sys.argv[1])
	parses = NegraCorpusReader(".", sys.argv[2])
	goldlen = len(gold.parsed_sents())
	parseslen = len(parses.parsed_sents())
	if len(sys.argv) == 4:
		maxsent = min(int(sys.argv[3]), goldlen)
	else: maxsent = goldlen
	assert maxsent
	assert goldlen == parseslen

	#print "#. precision\trecall\t\tf-measure\tPOS accuracy"
	print """\
   Sentence                 Matched   Brackets            Corr      Tag
  ID Length  Recall  Precis Bracket   gold   test  Words  Tags Accuracy    LA
______________________________________________________________________________"""
	exact = 0.
	maxlenseen = 0
	goldpos = []
	candpos = []
	la = []
	goldb = set()
	candb = set()
	goldbcat = defaultdict(set)
	candbcat = defaultdict(set)
	for n, csent, gsent in izip(count(), parses.parsed_sents(), gold.parsed_sents()):
		if n and n == maxsent: break
		cpos = sorted(csent.pos())
		gpos = sorted(gsent.pos())
		cbrack = bracketings(csent)
		gbrack = bracketings(gsent)
		if cbrack == gbrack: exact += 1
		candb.update((n,a) for a in cbrack)
		goldb.update((n,a) for a in gbrack)
		for a in gbrack: goldbcat[a[0]].add((n, a))
		for a in cbrack: candbcat[a[0]].add((n, a))
		goldpos.extend(gpos)
		candpos.extend(cpos)
		la.append(leafancestor(gsent, csent))
		if maxlenseen < len(cpos): maxlenseen = len(cpos)
		print "%4d  %5d  %6.2f  %6.2f   %5d  %5d  %5d  %5d  %4d  %6.2f %6.2f" % (
			n+1,
			len(gpos),
			100 * nonetozero(recall(gbrack, cbrack)),
			100 * nonetozero(precision(gbrack, cbrack)),
			len(gbrack & cbrack),
			len(gbrack),
			len(cbrack),
			len(gpos), # how is words supposed to be different from len?? should we leave out punctuation or something?
			sum(1 for a,b in zip(gpos, cpos) if a==b),
			100 * accuracy(gpos, cpos),
			100 * la[-1]
			)
	# what about multiple unaries w/same label??

	print """\n\
__________________ Category Statistics ___________________
     label      % gold   catRecall   catPrecis   catFScore
__________________________________________________________"""
	for a in sorted(set(goldbcat) | set(candbcat),
			key=lambda x: -len(goldbcat[x])):
		print " %s      %6.2f      %6.2f      %6.2f      %6.2f" % (
			a.rjust(9),
			100 * len(goldbcat[a]) / float(len(goldb)),
			100 * nonetozero(recall(goldbcat[a], candbcat[a])),
			100 * nonetozero(precision(goldbcat[a], candbcat[a])),
			100 * nonetozero(f_measure(goldbcat[a], candbcat[a])))
	
	print """\n\
Wrong Category Statistics
   test/gold   count
____________________"""
	gmismatch = dict(((n, indices), label)
				for n,(label,indices) in goldb - candb)
	wrong = FreqDist((label, gmismatch[n, indices])
				for n,(label,indices) in candb - goldb
				if (n, indices) in gmismatch)
	for labels, freq in wrong.items():
		print "%s %7d" % ("/".join(labels).rjust(12), freq)

	print "\n____________ Summary ____________"
	print "number of sentences:       %6d" % (maxsent)
	print "maximum length:            %6d" % (maxlenseen)
	print "labeled precision:         %6.2f" % (100 * nonetozero(precision(goldb, candb)))
	print "labeled recall:            %6.2f" % (100 * nonetozero(recall(goldb, candb)))
	print "labeled f-measure:         %6.2f" % (100 * nonetozero(f_measure(goldb, candb)))
	print "leaf-ancestor:             %6.2f" % (100 * mean(la))
	print "exact match:               %6.2f" % (100 * (exact / maxsent))
	print "tagging accuracy:          %6.2f" % (100 * accuracy(goldpos, candpos))

if __name__ == '__main__': main()
