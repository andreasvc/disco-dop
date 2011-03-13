from nltk import Tree, Nonterminal
from math import log
from itertools import chain
from pprint import pprint

def concat(s):
	""" iterate over a sequence of numbers and yield first element of each continguous range """
	last = None
	for a in s:
		if last == None or a > last + 1:
			yield a
		last = a

def ranges(s):
	""" partition s into a sequence of lists corresponding to contiguous ranges """
	range = []
	for a in s:
		if not range or a==range[-1]+1:
			range.append(a)
		else:
			yield range
			range = [a]
	if range: yield range

def subst(s):
	""" substitute variables for indices in a sequence """
	return ["?X%d" % a for a in s]

def srcg_productions(tree, sent):
	""" given a tree with indices as terminals, and a sentence with the
	corresponding words for these indices, produce a set of simple RCG rules """
	rules = []
	for st in tree.subtrees():
		if st.height() == 2:
			lhs = "['%s', [%s]]" % (st.node, sent[int(st[0])])
			rhs = "[Epsilon]"
		else:
			vars = [list(concat(sorted(map(int, a.leaves())))) for a in st]	
			lvars = map(subst, list(ranges(sorted(chain(*vars)))))
			lhs = "['%s', %s]" % (st.node, repr(lvars)[1:-1].replace("'",""))
			rhs = ", ".join("['%s', %s]" % (a.node, repr(subst(b))[1:-1].replace("'","")) for a,b in zip(st, vars))
		rules.append("[%s, %s]" % (lhs, rhs))
	return rules

def main():
	tree = Tree("(S (VP2 (VP2 (PROAV 0) (VVPP 2)) (VAINF 3)) (VMFIN 1))")
	sent = "Daruber muss nachgedacht werden".split()
	tree.chomsky_normal_form()
	pprint(srcg_productions(tree, sent))
if __name__ == '__main__': main()
