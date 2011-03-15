from negra import NegraCorpusReader
from rcgrules import srcg_productions, dop_srcg_rules, induce_srcg
from plcfrs import parse, enumchart, mostprobableparse, fs
from nltk import FreqDist, Tree
from nltk.metrics import precision
from itertools import islice, chain
from math import log, e
from pprint import pprint

def rem_marks(tree):
	for a in tree.subtrees(lambda x: "_" in x.node):
		a.node = a.node.rsplit("_", 1)[0]
	for a in tree.treepositions('leaves'):
		tree[a] = int(tree[a])
	return tree
def s(a):
	return (a.lhs(), tuple(sorted(a.rhs())))

# Negra treebank version 2 sample:
# http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/annotation/sample2.export
corpus = NegraCorpusReader(".", "sample2.export")
grammar = []
trees, sents = corpus.parsed_sents(), corpus.sents()

n = 1
grammar = induce_srcg(list(trees), sents)
dopgrammar = dop_srcg_rules(chain(*(list(trees) for a in range(n))), n*list(sents))

for tree, sent in zip(trees, sents):
	print len(sent), " ".join(sent)
	print "SRCG:",
	chart, start = parse(sent, grammar, start='ROOT', viterbi=True)
	if not chart: print "no parse"
	for result, prob in enumchart(chart, start):
		result.un_chomsky_normal_form()
		print "p =", e**prob,
		if rem_marks(result) == tree: print "exact match"
		else: 
			print "labeled precision", precision(set(map(s, result.productions())), set(map(s,tree.productions())))

	print "DOP:",
	chart, start = parse(sent, dopgrammar, start='ROOT', viterbi=False)
	if not chart: print "no parse"
	for result, prob in mostprobableparse(chart, start, 100).items():
		print "p =", prob,
		result = rem_marks(Tree(result))
		result.un_chomsky_normal_form()
		if result == tree: print "exact match"
		else: 
			print "labeled precision", precision(set(map(s, result.productions())), set(map(s,tree.productions())))
			print result
	print
