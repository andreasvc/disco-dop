from math import exp
from nltk import Tree
import grammar, containers, treetransforms, plcfrs, kbest

trees = [Tree.parse(a, parse_leaf=int) for a in """\
(ROOT (S (NP (NNP (John 0) (John 7))) (VP (VB (misses 1) (manque 5)) (PP (IN (a` 6)) (NP (NNP (Mary 2) (Mary 4)))))) (SEP (| 3)))
(ROOT (S (NP (NNP (Mary 0) (Mary 4))) (VP (VB (likes 1) (aimes 5)) (NP (DT (la 6)) (NN (pizza 2) (pizza 7))))) (SEP (| 3)))""".splitlines()]
sents = [["0"] * len(a.leaves()) for a in trees]

def main():
	map(treetransforms.binarize, trees)
	scfg = grammar.induce_srcg(trees, sents)
	compiled_scfg = grammar.Grammar(scfg)
	print "sentences:"
	for t in trees: print " ".join(w for _, w in sorted(t.pos()))
	print "treebank:"
	for t in trees: print t
	print compiled_scfg, "\n"
	do("John likes Mary | John aimes Mary".split(), compiled_scfg)
	print "incorrect translation:"
	do("John likes Mary | Mary aimes John".split(), compiled_scfg)
	do(u"John misses pizza | la pizza manque a` John".split(), compiled_scfg)
	print "incorrect translation:"
	do(u"John misses pizza | John manque a` la pizza".split(), compiled_scfg)

def do(testsent, compiled_scfg):
	chart, start = plcfrs.parse(["0"] * len(testsent),
		compiled_scfg,
		tags=testsent, start=compiled_scfg.toid["ROOT"],
		exhaustive=True)
	print "input:", " ".join("%d:%s" % a for a in enumerate(testsent))
	if start:
		results = kbest.lazykbest(chart, start, 10, compiled_scfg.tolabel)
		for tree, prob in results:
			tree = Tree(tree)
			treetransforms.unbinarize(tree)
			print exp(-prob), tree
	else: print "no parse!"
	print

main()
