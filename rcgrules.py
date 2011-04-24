from dopg import nodefreq, decorate_with_ids
from nltk import ImmutableTree, Tree, FreqDist, memoize
from math import log, exp
from heapq import nsmallest, heappush, heappop
from itertools import chain, count, islice
from pprint import pprint
from collections import defaultdict, namedtuple

def rangeheads(s):
	""" iterate over a sequence of numbers and yield first element of each
	contiguous range
	>>> rangeheads( (0, 1, 3, 4, 6) )
	[0, 3, 6]
	"""
	return [a[0] for a in ranges(s)]

def ranges(s):
	""" partition s into a sequence of lists corresponding to contiguous ranges
	>>> list(ranges( (0, 1, 3, 4, 6) ))
	[[0, 1], [3, 4], [6]]
	""" 
	rng = []
	for a in s:
		if not rng or a == rng[-1]+1:
			rng.append(a)
		else:
			yield rng
			rng = [a]
	if rng: yield rng

def node_arity(n, vars, inplace=False):
	""" mark node with arity if necessary """
	if len(vars) > 1 and not n.node.endswith("_%d" % len(vars)):
		if inplace: n.node = "%s_%d" % (n.node, len(vars))
		else: return "%s_%d" % (n.node, len(vars))
	return n.node

def alpha_normalize(s):
	""" Substitute variables so that the variables on the left-hand side appear consecutively.
		E.g. [0,1], [2,3] instead of [0,1], [3,4]. Modifies s in-place."""
	# flatten left hand side variables into a single list
	lvars = list(chain(*s[1][0]))
	for a,b in zip(lvars, range(len(lvars))):
		if a==b: continue
		for x in s[1][0]:  # left hand side
			if a in x: x[x.index(a)] = b
		for x in s[1][1:]: # right hand side
			if a in x: x[x.index(a)] = b
	return s

def freeze(l):
	return tuple(map(freeze, l)) if isinstance(l, (list, tuple)) else l

def unfreeze(l):
	return list(map(unfreeze, l)) if isinstance(l, (list, tuple)) else l

def srcg_productions(tree, sent, arity_marks=True):
	""" given a tree with indices as terminals, and a sentence
	with the corresponding words for these indices, produce a set
	of simple RCG rules. has the side-effect of adding arity
	markers to node labels """
	rules = []
	for st in tree.subtrees():
		if st.height() == 2:
			nonterminals = (intern(st.node), 'Epsilon')
			vars = ((sent[int(st[0])],),())
			rule = zip(nonterminals, vars)
		else:
			rvars = [rangeheads(sorted(map(int, a.leaves()))) for a in st]
			lvars = list(ranges(sorted(chain(*(map(int, a.leaves()) for a in st)))))
			lvars = [[x for x in a if any(x in c for c in rvars)] for a in lvars]
			lhs = intern(node_arity(st, lvars, True) if arity_marks else st.node)
			nonterminals = (lhs,) + tuple(node_arity(a, b) if arity_marks else a.node for a,b in zip(st, rvars))
			vars = (lvars,) + tuple(rvars)
			if vars[0][0][0] != vars[1][0]:
				# sort the right hand side so that the first argument comes from the first nonterminal
				# A[0,1] -> B[1] C[0]  becomes A[0,1] -> C[0] B[1]
				# although this boils down to a simple swap in a binarized grammar, for generality we do a sort instead
				vars, nonterminals = zip((vars[0], nonterminals[0]), *sorted(zip(vars[1:], nonterminals[1:]), key=lambda x: vars[0][0][0] != x[0][0]))
			rule = zip(nonterminals, vars)
		rules.append(rule)
	return rules

def varstoindices(rule):
	nonterminals, vars = zip(*unfreeze(rule))
	if rule[1][0] != 'Epsilon':
		# replace the variable numbers by indices pointing to the
		# nonterminal on the right hand side from which they take 
		# their value.
		# A[0,1,2] -> A[0,2] B[1]  becomes  A[0, 1, 0] -> B C
		for x in vars[0]:
			for n,y in enumerate(x):
				for m,z in enumerate(vars[1:]):
					if y in z: x[n] = m
	return nonterminals, freeze(vars[0])

def induce_srcg(trees, sents):
	""" Induce an SRCG, similar to how a PCFG is read off from a treebank """
	grammar = []
	for tree, sent in zip(trees, sents):
		t = tree.copy(True)
		grammar.extend(map(varstoindices, srcg_productions(t, sent)))
	grammar = FreqDist(grammar)
	fd = FreqDist()
	for rule,freq in grammar.items(): fd.inc(rule[0][0], freq)
	return [(rule, log(float(freq)/fd[rule[0][0]])) for rule,freq in grammar.items()]

def dop_srcg_rules(trees, sents, normalize=False, shortestderiv=False):
	""" Induce a reduction of DOP to an SRCG, similar to how Goodman (1996)
	reduces DOP1 to a PCFG.
	Normalize means the application of the equal weights estimate """
	ids, rules = count(1), []
	fd,ntfd = FreqDist(), FreqDist()
	for tree, sent in zip(trees, sents):
		#t = canonicalize(tree.copy(True))
		t = tree.copy(True)
		prods = map(varstoindices, srcg_productions(t, sent))
		ut = decorate_with_ids(t, ids)
		uprods = map(varstoindices, srcg_productions(ut, sent, False))
		nodefreq(t, ut, fd, ntfd)
		rules.extend(chain(*([(c,avar) for c in cartpi(list((x,) if x==y else (x,y) for x,y in zip(a,b)))] for (a,avar),(b,bvar) in zip(prods, uprods))))
	rules = FreqDist(rules)
	if shortestderiv:
		return [(rule, log(1 if '@' in rule[0][0] else 0.5)) for rule in rules]
	# should we distinguish what kind of arguments a node takes in fd?
	return [(rule, log(freq * reduce((lambda x,y: x*y),
		map((lambda z: fd[z] if '@' in z else 1), rule[0][1:])) / 
		(float(fd[rule[0][0]]) * (ntfd[rule[0][0]] 
		if '@' not in rule[0][0] and normalize else 1))))
		for rule, freq in rules.items()]

def splitgrammar(grammar):
	""" split the grammar into various lookup tables, mapping nonterminal labels to numeric identifiers"""
	Grammar = namedtuple("Grammar", "unary lbinary rbinary lexical bylhs toid tolabel".split())
	unary, lbinary, rbinary, lexical, bylhs = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
	#unary, lbinary, rbinary, lexical, bylhs = {}, {}, {}, {}, {}
	# get a list of all nonterminals; make sure Epsilon and ROOT are first, and assign them unique IDs
	#nonterminals = list(enumerate(["Epsilon", "ROOT"] + sorted(set(a for (rule,yf),weight in grammar for a in rule) - set(["Epsilon", "ROOT"]))))
	nonterminals = list(enumerate(["Epsilon", "ROOT"] + sorted(set(chain(*(rule for (rule,yf),weight in grammar))) - set(["Epsilon", "ROOT"]))))
	toid, tolabel = dict((lhs, n) for n, lhs in nonterminals), dict((n, lhs) for n, lhs in nonterminals)
	# negate the log probabilities because the heap we use is a min-heap
	for (rule,yf),w in grammar:
		r = tuple(toid[a] for a in rule), yf
		if len(rule) == 2:
			if r[0][1] == 0: #Epsilon
				# lexical productions (mis)use the field for the yield function to store the word
				lexical.setdefault(yf[0], []).append((r, -w))
				bylhs.setdefault(r[0][0], []).append((r, -w))
			else:
				unary.setdefault(r[0][1], []).append((r, -w))
				bylhs.setdefault(r[0][0], []).append((r, -w))
		elif len(rule) == 3:
			lbinary.setdefault(r[0][1], []).append((r, -w))
			rbinary.setdefault(r[0][2], []).append((r, -w))
			bylhs.setdefault(r[0][0], []).append((r, -w))
		else: raise ValueError("grammar not binarized: %s" % repr(r))
	return Grammar(unary, lbinary, rbinary, lexical, bylhs, toid, tolabel)

def canonicalize(tree):
	for a in tree.subtrees():
		a.sort()
	return tree

def allmax(seq, key):
	""" return all x s.t. key(x)==max(seq, key)"""
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

def extractfragments(trees):
	""" Seeks the largest fragments occurring at least twice in the corpus.
	Algorithm from: Sangati et al., Efficiently extract recurring tree fragments from large treebanks"""
	fraglist = FreqDist()
	partfraglist = set()
	trees = [a.freeze() for a in trees]
	for n,a in enumerate(trees):
		for b in trees[n+1:]:
			l = set()
			mem = {}
			for i in a.treepositions():
				for j in b.treepositions():
					x = extractmaxfragments(a,b,i,j, mem)
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
			del mem
	return fraglist #| partfraglist

def extractmaxfragments(a, b, i, j, mem):
	""" a fragment is a connected subset of nodes where each node either has
	zero children, or as much as in the original tree.""" 
	if (a,b,i,j) in mem: return mem[(a,b,i,j)]
	if not same((a[i],b[j])): return set()
	nodeset = set([i])
	if not isinstance(a[i], Tree) or not isinstance(b[j], Tree): return nodeset
	if len(a[i])==len(b[j]) and all(map(same, zip(a[i],b[j]))):
		for n,x in enumerate(a[i]):
			nodeset.update(extractmaxfragments(a,b,i+(n,), j+(n,), mem))
	mem[(a,b,i,j)] = nodeset
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

@memoize
def fanout(tree):
	return len(rangeheads(sorted(tree.leaves()))) if isinstance(tree, Tree) else 1

def complexityfanout(tree):
	return (fanout(tree) + sum(map(fanout, tree)), fanout(tree))

def minimalbinarization(tree, score):
	""" Gildea (2009): Optimal parsing strategies for linear context-free rewriting systems
	Expects an immutable tree where the terminals are integers corresponding to indices.

	>>> minimalbinarization(ImmutableTree("NP", [ImmutableTree("ART", [0]), ImmutableTree("ADJ", [1]), ImmutableTree("NN", [2])]), complexityfanout)
	ImmutableTree('NP', [ImmutableTree('NP|<ART-ADJ>', [ImmutableTree('ART', [0]), ImmutableTree('ADJ', [1])]), ImmutableTree('NN', [2])])
	"""
	def newproduction(a, b):
		if min(nonterms[a]) > min(nonterms[b]): a, b = b, a
		newlabel = "%s|<%s>" % (tree.node, "-".join(x.node for x,y in sorted(nonterms[a] | nonterms[b], key=lambda z: z[1])))
		return ImmutableTree(newlabel, [a, b])
	if len(tree) <= 2: return tree
	workingset = set()
	agenda = []
	nonterms = {}
	goal = set((a, tuple(a.leaves())) for a in tree)
	for a in tree:
		workingset.add((score(a), a))
		heappush(agenda, (score(a), a))
		nonterms[a] = set([(a, tuple(a.leaves()))])
	while agenda:
		x, px = heappop(agenda)
		if (x, px) not in workingset: continue
		if nonterms[px] == goal:
			px = ImmutableTree(tree.node, px[:])
			return px
		for y, p1 in list(workingset):
			if (y, p1) not in workingset or nonterms[px] & nonterms[p1]: continue
			p2 = newproduction(px, p1)
			p2nonterms = nonterms[px] | nonterms[p1]
			x2 = score(p2)
			inferior = [(y, p2x) for y, p2x in workingset if nonterms[p2x] == p2nonterms and x2 < y]
			if inferior or p2nonterms not in nonterms.values():
				workingset.add((x2, p2))
				heappush(agenda, (x2, p2))
			for a in inferior:
				workingset.discard(a)
				del nonterms[a[1]]
			nonterms[p2] = p2nonterms

def binarizetree(tree):
	""" Recursively binarize a tree. Tree needs to be immutable."""
	if not isinstance(tree, Tree): return tree
	# bypass algorithm when there are no discontinuities:
	elif len(rangeheads(tree.leaves())) == 1:
		newtree = Tree(tree.node, map(binarizetree, tree))
		newtree.chomsky_normal_form()
		return newtree
	return Tree(tree.node, map(binarizetree, minimalbinarization(tree, complexityfanout)))

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

def enumchart(chart, start, tolabel, n=1, depth=0):
	"""exhaustively enumerate trees in chart headed by start in top down 
		fashion. chart is a dictionary with lhs -> [(rhs, logprob), (rhs, logprob) ... ]
		this function doesn't really belong in this file but Cython doesn't
		support generators so this function is "in exile" over here.  """
	if depth >= 100: return
	for iscore,p,rhs in chart[start]:
		if rhs[0].label == 0: #Epsilon
			yield "(%s %d)" % (tolabel[start.label], rhs[0].vec), p
		else:
			# this doesn't seem to be a good idea:
			#for x in nsmallest(n, cartpi(map(lambda y: enumchart(chart, y, tolabel, n, depth+1), rhs)), key=lambda x: sum(p for z,p in x)):
			#for x in sorted(islice(bfcartpi(map(lambda y: enumchart(chart, y, tolabel, n, depth+1), rhs)), n), key=lambda x: sum(p for z,p in x)):
			for x in islice(bfcartpi(map(lambda y: enumchart(chart, y, tolabel, depth+1), rhs)), n):
				tree = "(%s %s)" % (tolabel[start.label], " ".join(z for z,px in x))
				yield tree, p+sum(px for z,px in x)

def do(sent, grammar):
	from plcfrs import parse, mostprobableparse
	print "sentence", sent
	p, start = parse(sent, grammar, start=grammar.toid['S'], viterbi=False, n=100)
	if p:
		mpp = mostprobableparse(p, start, grammar.tolabel)
		for t in mpp:
			print exp(mpp[t]), t
	else: print "no parse"
	print

def main():
	from treetransforms import un_collinize
	from negra import NegraCorpusReader
	n = NegraCorpusReader(".", "sample2\.export")
	for tree, sent in zip(n.parsed_sents(), n.sents()):
		print tree
		a = binarizetree(tree.freeze())
		print a; print
		un_collinize(a)
		print a, a == tree
	tree = Tree("(S (VP (VP (PROAV 0) (VVPP 2)) (VAINF 3)) (VMFIN 1))")
	sent = "Daruber muss nachgedacht werden".split()
	tree.chomsky_normal_form()
	pprint(srcg_productions(tree.copy(True), sent))
	pprint(induce_srcg([tree.copy(True)], [sent]))
	print splitgrammar(induce_srcg([tree.copy(True)], [sent]))
	pprint(dop_srcg_rules([tree.copy(True)], [sent]))
	do(sent, splitgrammar(dop_srcg_rules([tree], [sent])))

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
if __name__ == '__main__':
	import doctest
	# do doctests, but don't be pedantic about whitespace (I suspect it is the
	# militant anti-tab faction who are behind this obnoxious default)
	fail, attempted = doctest.testmod(verbose=False,
	optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
	if attempted and not fail:
		print "%d doctests succeeded!" % attempted 
	main()
