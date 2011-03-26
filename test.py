# -*- coding: UTF-8 -*-
from negra import NegraCorpusReader
from rcgrules import srcg_productions, dop_srcg_rules, induce_srcg, enumchart, extractfragments
from nltk import FreqDist, Tree
from nltk.metrics import precision
from itertools import islice, chain
from math import log, e
from pprint import pprint
import cPickle
try: import pyximport
except:
	from plcfrs import parse, mostprobableparse
else:
	#yximport.install()
	#from plcfrs_cython import parse, mostprobableparse
	from plcfrs import parse, mostprobableparse

def rem_marks(tree):
	for a in tree.subtrees(lambda x: "_" in x.node):
		a.node = a.node.rsplit("_", 1)[0]
	for a in tree.treepositions('leaves'):
		tree[a] = int(tree[a])
	return tree
def s(a):
	return (a.lhs(), tuple(sorted(a.rhs())))

# Tiger treebank version 2 sample:
# http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/annotation/sample2.export
corpus = NegraCorpusReader(".", "sample2.export")
grammar = []
trees, sents = corpus.parsed_sents(), corpus.sents()

n = 9
grammar = induce_srcg(list(trees), sents)
dopgrammar = dop_srcg_rules(chain(*(list(trees) for a in range(n))), n*list(sents))

#cPickle.dump(dopgrammar, open("dopgrammar.pickle", "wb"))
#import cPickle
#dopgrammar = cPickle.load(open("dopgrammar.pickle","rb"))
#from plcfrs import parse, mostprobableparse
nodes = n * sum(len(list(a.subtrees())) for a in trees)
print "DOP model based on", n*3, "sentences,", nodes, "nodes, max", nodes*8, "nonterminals"
#sents = ["","","Wie am Samstag berichtet , mu\xdf das Institut seine Aktivit\xe4ten in den Vereinigten Staaten einstellen und eventuell Geldstrafen von mehr als einer Milliarde Dollar zahlen .".split()]
#for a,b in extractfragments(trees).items():
#	print a,b
#exit()
for tree, sent in zip(trees, sents)[2:]:
	print len(sent), " ".join(sent)
	"""print "SRCG:",
	chart, start = parse(sent, grammar, start='ROOT', viterbi=True)
	if not chart: print "no parse"
	for result, prob in enumchart(chart, start):
		result = Tree(result)
		result.un_chomsky_normal_form()
		print "p =", e**prob,
		if rem_marks(result) == tree: print "exact match"
		else: 
			print "labeled precision", precision(set(map(s, result.productions())), set(map(s,tree.productions())))
	"""
	print "DOP:",
	viterbi = True
	sample = False
	n = 1
	chart, start = parse(sent, dopgrammar, start='ROOT', viterbi=viterbi, n=n)
	if not chart: print "no parse"
	print "viterbi =", viterbi, "n=%d" % n if viterbi else '',
	for result, prob in mostprobableparse(chart, start,n=100,sample=sample).items():
		print "p =", prob,
		result = rem_marks(Tree(result))
		result.un_chomsky_normal_form()
		if result == tree: print "exact match"
		else: 
			print "labeled precision", precision(set(map(s, result.productions())), set(map(s,tree.productions())))
			print result
	print
