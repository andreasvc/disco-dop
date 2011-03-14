from nltk import Tree, Nonterminal
from math import log
from itertools import chain
from pprint import pprint

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

def node_arity(node, vars):
	""" mark node with arity if necessary """
	if len(vars) > 1:
		return "%s_%s" % (node, str(len(vars)) if len(vars) > 1 else '')
	else: return node

def srcg_productions(tree, sent):
	""" given a tree with indices as terminals, and a sentence with the
	corresponding words for these indices, produce a set of simple RCG rules """
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
			lhs = "['%s', %s]" % (node_arity(st.node, lvars), repr(lvars)[1:-1].replace("'",""))
			rhs = ", ".join("['%s', %s]" % (node_arity(a.node, b), repr(subst(b))[1:-1].replace("'","")) for a,b in zip(st, vars))
		rules.append("[%s, %s]" % (lhs, rhs))
	return rules

def main():
	tree = Tree("(S (VP (VP (PROAV 0) (VVPP 2)) (VAINF 3)) (VMFIN 1))")
	sent = "Daruber muss nachgedacht werden".split()
	tree.chomsky_normal_form()
	pprint(srcg_productions(tree, sent))
if __name__ == '__main__': main()
