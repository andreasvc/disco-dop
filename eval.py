# -*- coding: UTF-8 -*-
from nltk import Tree
from nltk.metrics import precision, recall, f_measure, accuracy
from negra import NegraCorpusReader
import sys
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
				for a in tree.subtrees() if isinstance(a[0], Tree))

def printbrackets(brackets):
	return ", ".join("%s[%s]" % (a,
					",".join(map(lambda x: "%s-%s" % (x[0], x[-1])
					if len(x) > 1 else str(x[0]), ranges(sorted(b)))))
					for a,b in brackets)

def harmean(seq):
	try: return len([a for a in seq if a]) / sum(1./a if a else 0. for a in seq)
	except: return "zerodiv"

def mean(seq):
	return sum(seq) / float(len(seq)) if seq else "zerodiv"

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
	if len(sys.argv) != 3:
		print "wrong number of arguments. usage: %s gold parses" % sys.argv[0]
		print sys.argv
		return
	gold = NegraCorpusReader(".", sys.argv[1])
	parses = NegraCorpusReader(".", sys.argv[2])
	assert len(gold.sents())
	assert len(gold.sents()) == len(parses.sents())

	exact = sum(1.0 for a,b
				in zip(gold.parsed_sents(), parses.parsed_sents())
				if bracketings(a) == bracketings(b))
	goldbrackets = frozenset((n, a)
			for n, sent in enumerate(gold.parsed_sents())
				for a in bracketings(sent))
	candb = frozenset((n, a)
			for n, sent in enumerate(parses.parsed_sents())
				for a in bracketings(sent))

	print "exact match:\t\t%5.2f" % (100 * (exact / len(gold.sents())))
	print "labeled precision:\t%5.2f" % (100 * precision(goldbrackets, candb))
	print "labeled recall:\t\t%5.2f" % (100 * recall(goldbrackets, candb))
	print "labeled f-measure:\t%5.2f" % (100 * f_measure(goldbrackets, candb))

if __name__ == '__main__': main()
