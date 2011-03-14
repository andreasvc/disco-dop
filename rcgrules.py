from plcfrs import fs, parse, enumchart
from dopg import nodefreq, frequencies, decorate_with_ids
from nltk import Tree, Nonterminal, FreqDist
from math import log, e
from itertools import chain, count, product
from pprint import pprint
import re

def rangeheads(s):
	""" iterate over a sequence of numbers and yield first element of each contiguous range """
	return [a[0] for a in ranges(s)]

def ranges(s):
	""" partition s into a sequence of lists corresponding to contiguous ranges """
	rng = []
	for a in s:
		if not rng or a == rng[-1]+1:
			rng.append(a)
		else:
			yield rng
			rng = [a]
	if rng: yield rng

def subst(s):
	""" substitute variables for indices in a sequence """
	return ["?X%d" % a for a in s]

def node_arity(n, vars, inplace=False):
	""" mark node with arity if necessary """
	if len(vars) > 1:
		if inplace: n.node = "%s_%d" % (n.node, len(vars))
		else: return "%s_%d" % (n.node, len(vars))
		return n.node
	else: return n.node

def srcg_productions(tree, sent, arity_marks=True):
	""" given a tree with indices as terminals, and a sentence
	with the corresponding words for these indices, produce a set
	of simple RCG rules. has the side-effect of adding arity
	markers to node labels (so don't run twice with the same tree) """
	rules = []
	for st in tree.subtrees():
		if st.height() == 2:
			lhs = "['%s', ['%s']]" % (st.node, sent[int(st[0])])
			rhs = "[Epsilon]"
		else:
			vars = [rangeheads(sorted(map(int, a.leaves()))) for a in st]
			lvars = list(ranges(sorted(chain(*(map(int, a.leaves()) for a in st)))))
			lvars = [[x for x in a if any(x in c for c in vars)] for a in lvars]
			lvars = map(subst, lvars)
			lhs = "['%s', %s]" % (node_arity(st, lvars, True) if arity_marks else st.node, repr(lvars)[1:-1].replace("'",""))
			rhs = ", ".join("['%s', %s]" % (node_arity(a, b) if arity_marks else a.node, repr(subst(b))[1:-1].replace("'","")) for a,b in zip(st, vars))
		rules.append("[%s, %s]" % (lhs, rhs))
	return rules

def dop_srcg_rules(trees, sents):
	ids, rules = count(1), []
	fd,ntfd = FreqDist(), FreqDist()
	for t, sent in zip(trees, sents):
		t.chomsky_normal_form()
		prods = map(fs, srcg_productions(t, sent))
		ut = decorate_with_ids(t, ids)
		ut.chomsky_normal_form()
		uprods = map(fs, srcg_productions(ut, sent, False))
		nodefreq(t, ut, fd, ntfd)
		rules.extend(set(chain(*(product(*zip(*a)) for a in zip(prods, uprods)))))
	rules = FreqDist(rules)
	return dict((fs(rule), log(freq * reduce((lambda x,y: x*y),
		map((lambda z: '@' in z[0] and fd[z[0]] or 1),
		rule[1:])) / float(fd[rule[0][0]])))
		for rule, freq in rules.items())

def do(sent, grammar):
        print "sentence", sent
        p, start = parse(sent, grammar)
        if p:
                l = FreqDist()
                for n,(a,prob) in enumerate(enumchart(p, start)):
                        #print n, prob, a
                        l.inc(re.sub(r"@[0-9]+", "", Tree.convert(a).pprint(margin=99999)), e**prob)
                for a in l: print l[a], Tree(a)
        else: print "no parse"
        print

def main():
	tree = Tree("(S (VP (VP (PROAV 0) (VVPP 2)) (VAINF 3)) (VMFIN 1))")
	sent = "Daruber muss nachgedacht werden".split()
	tree.chomsky_normal_form()
	pprint(srcg_productions(tree.copy(True), sent))
	pprint(dop_srcg_rules([tree.copy(True)], [sent]))
	do(sent, dop_srcg_rules([tree], [sent]))
if __name__ == '__main__': main()
