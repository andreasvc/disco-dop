from dopg import nodefreq, frequencies, decorate_with_ids
from nltk import ImmutableTree, Tree, Nonterminal, FreqDist, SExprTokenizer
from math import log, e
from itertools import chain, count, islice
from pprint import pprint
import re
sexp=SExprTokenizer("()")

def fs(rule):
	vars = []
	Epsilon = "Epsilon"
	for a in re.findall("\?[XYZ][0-9]*", rule):
		if a not in vars: vars.append(a)
		exec("%s = []" % a[1:])
	return tuple(zip(*eval(re.sub(r"\?([XYZ][0-9]*)", r"\1", rule))))

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
	return tuple("?X%d" % a for a in s)

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
			rhs = "[Epsilon, []]"
		else:
			vars = [rangeheads(sorted(map(int, a.leaves()))) for a in st]
			lvars = list(ranges(sorted(chain(*(map(int, a.leaves()) for a in st)))))
			lvars = [[x for x in a if any(x in c for c in vars)] for a in lvars]
			lvars = tuple(map(subst, lvars))
			lhs = intern(node_arity(st, lvars, True) if arity_marks else st.node)
			lhs = "('%s', %s)" % (lhs, repr(lvars).replace("'",""))
			rhs = ", ".join("('%s', %s)" % (node_arity(a, b) if arity_marks else a.node, repr(subst(b)).replace("'","")) for a,b in zip(st, vars))
		rules.append(alpha_normalize("(%s, %s)" % (lhs, rhs)))
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
		rules.extend(chain(*(cartpi(list((x,) if x==y else (x,y) for x,y in zip(a,b))) for a,b in zip(prods, uprods))))
	rules = FreqDist("(%s)" % ", ".join(a) for a in rules)
	return [(fs(rule), log(freq * reduce((lambda x,y: x*y),
		map((lambda z: '@' in z and fd[z] or 1),
		fs(rule)[0][1:])) / float(fd[fs(rule)[0][0]])))
		for rule, freq in rules.items()]

def extractfragments(trees):
	""" Seeks the largest fragments occurring at least twice in the corpus.
	Algorithm from: Sangati et al., Efficiently extract recurring tree fragments from large treebanks"""
	fraglist = FreqDist()
	partfraglist = set()
	for n,a in enumerate(trees):
		for b in trees[n+1:]:
			l = set()
			for i in a.treepositions():
				for j in b.treepositions():
					x = extractmaxfragments(a,b,i,j)
					if x in l: continue
					for y in l:
						if x < y: break
						if x > y:
							l.remove(y)
							l.add(frozenset(x))
							break
					else: l.add(frozenset(x))
					#partfraglist.update(extractmaxpartialfragments(a[i], b[j]))
			fraglist.update(filter(None, (fragmentfromindices(a,x) for x in l)))
			#fraglist.update(chain(*(filter(None, fragmentfromindices(a,x)) for x in l if not any(x < y for y in l))))
	return fraglist #| partfraglist

def allmax(seq, key):
	if not seq: return []
	m = max(map(key, seq))
	return [a for a in seq if key(a) == m]

def same((a,b)):
	if type(a) != type(b): return False
	elif isinstance(a, Tree): return a.node == b.node
	else: return a == b

def fragmentfromindices(tree, indices):
	if not indices: return
	froot = min(indices, key=len)
	if not isinstance(tree[froot], Tree): return
	tree = Tree.convert(tree.copy(True))
	remind = set()
	for a in reversed(tree.treepositions()):
		# iterate over indices from bottom to top, right to left,
		# so that other indices remain valid after deleting each subtree
		if a not in indices and len(a) > len(froot):
			del tree[a]
	return tree[froot].freeze() if tree[froot].height() > 1 else None

def extractmaxfragments(a, b, i, j):
	""" a fragment is a connected subset of nodes where each node either has
	zero children, or as much as in the original tree.""" 
	if not same((a[i],b[j])): return set()
	nodeset = set([i])
	if not isinstance(a[i], Tree) or not isinstance(b[j], Tree): return nodeset
	if len(a[i])==len(b[j]) and all(map(same, zip(a[i],b[j]))):
		for n,x in enumerate(a[i]):
			nodeset.update(extractmaxfragments(a,b,i+(n,), j+(n,)))
	return nodeset

def extractmaxpartialfragments(a, b):
	""" partial fragments allow difference in number of children
	not tested. """
	if not same((a,b)): return set()
	mappingsset = maxsetmappings(a,b,0,0,True)
	if not mappingsset: return [ImmutableTree(a.node, [])]
	partfragset = set()
	for mapping in mappingsset:
		maxpartialfragmentpairs = [set() for x in mapping]
		for i, (n1, n2) in enumerate(mapping):
			maxpartialfragments[i] = extractmaxpartialfragments(n1, n2)
		for nodeset in cartpi(maxpartialfragmentpairs):
			nodeset.union(a)
			partfragset.add(nodeset)
	return partfragset

def maxsetmappings(a,b,x,y,firstCall=False):
	mappings = []
	startx = x if firstCall else x + 1
	starty = y if firstCall else y + 1
	endx = len(a) - 1
	endy = len(b) - 1
	startxexists = startx < len(a)
	startyexists = starty < len(b)
	while startxexists or startyexists:
		if startxexists:
			for celly in range(endy, starty + 1, -1):
				if a[startx] == b[celly]:
					endy = celly
					submappings = maxsetmappings(a,b,startx,cely)
					if not firstCall:
						for mapping in submappings:
							mapping.add((a[startx],b[celly]))
					mappings.extend(submappings)
		if startyexists:
			for cellx in range(endx, startx + 1, -1):
				if a[startx] == b[starty]:
					endx = cellx
					submappings = maxsetmappings(a,b,cellx,starty)
					if not firstCall:
						for mapping in submappings:
							mapping.add((a[cellx],b[starty]))
					mappings.extend(submappings)
		if startxexists and startyexists and a[startx] == b[starty]:
			submappings = maxsetmappings(a,b,startx,starty,False)
			if not firstCall:
				for mapping in submappings:
					mapping.add((a[startx],b[starty]))
			mappings.extend(submappings)
			break
		if startx+1 <= endx: startx += 1
		else: startxexists = False
		if starty+1 <= endy: starty += 1
		else: startyexists = False
	return set(mappings)

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
	""" itertools.product doesn't support infinite sequences!
	>>> list(islice(cartpi([count(), count(0)]), 9))
	[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8)]"""
	if seq: return (b + (a,) for b in cartpi(seq[:-1]) for a in seq[-1])
	return ((), )

def bfcartpi(seq):
	"""breadth-first (diagonal) cartesian product
	>>> list(islice(bfcartpi([count(), count(0)]), 9))
	[(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]"""
	#wrap items of seq in generators
	seqit = [(x for x in a) for a in seq]
	#fetch initial values
	try: seqlist = [[a.next()] for a in seqit]	
	except StopIteration: return
	yield tuple(a[0] for a in seqlist)
	#bookkeeping of which iterators still have values
	stopped = len(seqit) * [False]
	n = len(seqit)
	while not all(stopped):
		if n == 0: n = len(seqit) - 1
		else: n -= 1
		if stopped[n]: continue
		try: seqlist[n].append(seqit[n].next())
		except StopIteration: stopped[n] = True; continue
		for result in cartpi(seqlist[:n] + [seqlist[n][-1:]] + seqlist[n+1:]):
			yield result

def enumchart(chart, start):
	"""exhaustively enumerate trees in chart headed by start in top down 
		fashion. chart is a dictionary with lhs -> (rhs, logprob) """
	for a,p in chart[start]:
		if len(a) == 1 and a[0][0] == "Epsilon":
			yield "(%s %d)" % (start[0], a[0][1][0]), p
			continue
		for x in bfcartpi(map(lambda y: enumchart(chart, y), a)):
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
	#"""
	tree = Tree("(S (VP (VP (PROAV 0) (VVPP 2)) (VAINF 3)) (VMFIN 1))")
	sent = "Daruber muss nachgedacht werden".split()
	tree.chomsky_normal_form()
	pprint(srcg_productions(tree.copy(True), sent))
	pprint(dop_srcg_rules([tree.copy(True)], [sent]))
	do(sent, dop_srcg_rules([tree], [sent]))
	exit()
	#"""
	treebank = """(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (JJ hungry) (NN dog))))
(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (JJ yellow) (NN cat))))
(S (NP (DT The) (JJ little) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN cat)) (VP (VBP ate) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP ate) (NP (DT the) (NN cat))))"""
	treebank = map(Tree, treebank.splitlines())
	i,j = (), ()
	#fragments = extractmaxfragments(a,b,i,j)
	fragments = extractfragments(treebank)
	for a,b in sorted(fragments.items()):
		print a,b
if __name__ == '__main__': main()
