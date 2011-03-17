from dopg import nodefreq, frequencies, decorate_with_ids
from nltk import Tree, Nonterminal, FreqDist, SExprTokenizer
from math import log, e
from itertools import chain, count, product 
from pprint import pprint
import re
sexp=SExprTokenizer("[]")

def fs(rule):
	vars = []
	Epsilon = "Epsilon"
	for a in re.findall("\?[XYZ][0-9]*", rule):
		if a not in vars: vars.append(a)
		exec("%s = []" % a[1:])	
	return eval(re.sub(r"\?([XYZ][0-9]*)", r"\1", rule))

def fs1(rule):
	return [a.strip() for a in sexp.tokenize(rule[1:-1]) if a != ',']

def rangeheads(s):
	""" iterate over a sequence of numbers and yield first element of each
	contiguous range """
	return [a[0] for a in ranges(s)]

def ranges(s):
	""" partition s into a sequence of lists corresponding to contiguous ranges
	""" 
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

def alpha_normalize(s):
	""" In a string containing n variables, variables are renamed to
		X1 .. Xn, in order of first appearance """
	vars = []
	for a in re.findall("\?X[0-9]+", s):
		if a not in vars: vars.append(a)
	return re.sub("\?X[0-9]+", lambda v: "?X%d" % (vars.index(v.group())+1), s)

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
		rules.append(alpha_normalize("[%s, %s]" % (lhs, rhs)))
	return rules

def dop_srcg_rules(trees, sents):
	""" Induce a reduction of DOP to an SRCG, similar to how Goodman (1996)
	reduces DOP1 to a PCFG """
	ids, rules = count(1), []
	fd,ntfd = FreqDist(), FreqDist()
	for tree, sent in zip(trees, sents):
		t = tree.copy(True)
		t.chomsky_normal_form()
		prods = map(fs1, srcg_productions(t, sent))
		ut = decorate_with_ids(t, ids)
		ut.chomsky_normal_form()
		uprods = map(fs1, srcg_productions(ut, sent, False))
		nodefreq(t, ut, fd, ntfd)
		rules.extend(chain(*(product(*((x,) if x==y else (x,y) for x,y in zip(a,b))) for a,b in zip(prods, uprods))))
	rules = FreqDist("[%s]" % ", ".join(a) for a in rules)
	return [(fs(rule), log(freq * reduce((lambda x,y: x*y),
		map((lambda z: '@' in z[0] and fd[z[0]] or 1),
		fs(rule)[1:])) / float(fd[fs(rule)[0][0]])))
		for rule, freq in rules.items()]

def induce_srcg(trees, sents):
	""" Induce an SRCG, similar to how a PCFG is read off from a treebank """
	grammar = []
	for tree, sent in zip(trees, sents):
		t = tree.copy(True)
		t.chomsky_normal_form()
		grammar.extend(srcg_productions(t, sent))
	grammar = FreqDist(grammar)
	fd = FreqDist(fs(a)[0][0] for a in grammar)
	return [(fs(rule), log(freq*1./fd[fs(rule)[0][0]])) for rule,freq in grammar.items()]

def cartpi(seq):
	""" itertools.product doesn't support infinite sequences!"""
	if seq: return ((a,) + b for b in cartpi(seq[1:]) for a in seq[0])
	return ((), )

def bfcartpi(seq):
	"""breadth-first (diagonal) cartesian product
	>>> list(bfcartpi([[0,1,2], [0,1,2]]))
	[(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1), (0, 2), (1, 2), (2, 2)]"""
	#wrap items of seq in iterators
	seqit = [(x for x in a) for a in seq]
	#fetch initial values
	try: seqlist = [[a.next()] for a in seqit]	
	except StopIteration: return
	yield tuple(a[0] for a in seqlist)
	left = len(seqlist) * [True]
	while any(left):
		for n,a,b in ((n,seqlist[n], seqit[n]) for n,y in enumerate(left) if y):
			try: a.append(b.next())
			except: left[n] = False; continue
			for result in cartpi(seqlist[:n] + [a[-1:]] + seqlist[n+1:]): yield result

def enumchart(chart, start):
	"""exhaustively enumerate trees in chart headed by start in top down 
		fashion. chart is a dictionary with lhs -> (rhs, logprob) """
	for a,p in chart[start]:
		if len(a) == 1 and isinstance(a[0], int):	#terminal
			#yield Tree(start[0], a), p
			yield "(%s %d)" % (start[0], a[0]), p
			continue
		for x in bfcartpi(map(lambda y: enumchart(chart, y), a)):
			#yield Tree(start[0], [z[0] for z in x]), p+x[0][1]+x[1][1]
			tree = "(%s %s)" % (start[0], " ".join(z[0] for z in x))
			yield tree, p+sum(z[1] for z in x)

def do(sent, grammar):
	from plcfrs import parse
	print "sentence", sent
	p, start = parse(sent, grammar)
	if p:
		l = FreqDist()
		for n,(a,prob) in enumerate(enumchart(p, start)):
			#print n, prob, a
			l.inc(re.sub(r"@[0-9]+", "", a), e**prob)
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
