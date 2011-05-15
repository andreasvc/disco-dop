from dopg import nodefreq, decorate_with_ids
from nltk import ImmutableTree, Tree, FreqDist, memoize
from math import log, exp
from heapq import nsmallest, heappush, heappop
from itertools import chain, count, islice, repeat
from pprint import pprint
from collections import defaultdict, namedtuple
from operator import mul

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
	return n.node if isinstance(n, Tree) else n

def alpha_normalize(s):
	""" Substitute variables so that the variables on the left-hand side appear consecutively.
		E.g. [0,1], [2,3] instead of [0,1], [3,4]. Modifies s in-place."""
	# flatten left hand side variables into a single list
	if s[0][1] == 'Epsilon': return s
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
		if len(st) == 1 and not isinstance(st[0], Tree):
			if sent[int(st[0])] is None: continue
			#nonterminals = (intern(str(st.node)), 'Epsilon')
			nonterminals = (st.node, 'Epsilon')
			vars = ((sent[int(st[0])],),())
			rule = zip(nonterminals, vars)
		elif isinstance(st[0], Tree):
			leaves = map(int, st.leaves())
			cnt = count(len(leaves))
			rleaves = [a.leaves() if isinstance(a, Tree) else [a] for a in st]
			rvars = [rangeheads(sorted(map(int, l))) for a,l in zip(st, rleaves)]
			lvars = list(ranges(sorted(chain(*(map(int, l) for l in rleaves)))))
			lvars = [[x for x in a if any(x in c for c in rvars)] for a in lvars]
			lhs = intern(str(node_arity(st, lvars, True) if arity_marks else st.node))
			nonterminals = (lhs,) + tuple(node_arity(a, b) if arity_marks else a.node for a,b in zip(st, rvars))
			vars = (lvars,) + tuple(rvars)
			if vars[0][0][0] != vars[1][0]:
				# sort the right hand side so that the first argument comes
				# from the first nonterminal
				# A[0,1] -> B[1] C[0]  becomes A[0,1] -> C[0] B[1]
				# although this boils down to a simple swap in a binarized
				# grammar, for generality we do a sort instead
				vars, nonterminals = zip((vars[0], nonterminals[0]), *sorted(zip(vars[1:], nonterminals[1:]), key=lambda x: vars[0][0][0] != x[0][0]))
			rule = zip(nonterminals, vars)
		else: continue
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

def induce_srcg(trees, sents, arity_marks=True):
	""" Induce an SRCG, similar to how a PCFG is read off from a treebank """
	grammar = []
	for tree, sent in zip(trees, sents):
		t = tree.copy(True)
		grammar.extend(map(varstoindices, srcg_productions(t, sent, arity_marks)))
	grammar = FreqDist(grammar)
	fd = FreqDist()
	for rule,freq in grammar.items(): fd.inc(rule[0][0], freq)
	return [(rule, log(float(freq)/fd[rule[0][0]])) for rule,freq in grammar.items()]

def dop_srcg_rules(trees, sents, normalize=False, shortestderiv=False, interpolate=1.0, wrong_interpolate=False, arity_marks=True):
	""" Induce a reduction of DOP to an SRCG, similar to how Goodman (1996)
	reduces DOP1 to a PCFG.
	Normalize means the application of the equal weights estimate
	interpolate 0.5 means 50% dop probabilities, 50% srcg (depth 1) probabilities"""
	ids, rules = count(1), []
	fd,ntfd = FreqDist(), FreqDist()
	for tree, sent in zip(trees, sents):
		t = canonicalize(tree.copy(True))
		t = tree.copy(True)
		prods = map(varstoindices, srcg_productions(t, sent, arity_marks))
		ut = decorate_with_ids(t, ids)
		uprods = map(varstoindices, srcg_productions(ut, sent, False))
		nodefreq(t, ut, fd, ntfd)
		# fd: how many subtrees are headed by node X (e.g. NP or NP@12), 
		# 	counts of NP@... should sum to count of NP
		# ntfd: frequency of a node in corpus (only makes sense for NP, not NP@12, the latter is always one)
		rules.extend(chain(*([(c,avar) for c in cartpi(list((x,)
								if x==y else (x,y)
								for x,y in zip(a,b)))]
							for (a,avar),(b,bvar) in zip(prods, uprods))))
	rules = FreqDist(rules)
	# should we distinguish what kind of arguments a node takes in fd?
	probmodel = [((rule, yf), log(freq * reduce(mul, (fd[z] for z in rule[1:] if '@' in z), 1)
		/ (float(fd[rule[0]]) * (ntfd[rule[0]] if normalize and '@' not in rule[0] else 1.0))
		* (interpolate if '@' not in rule[0] else 1.0)
		+ ((1 - interpolate) * (float(freq) / ntfd[rule[0]])
				if not any('@' in z for z in rule) and not wrong_interpolate else 0)))
		for (rule, yf), freq in rules.items()]
	if shortestderiv:
		nonprobmodel = [(rule, log(1 if '@' in rule[0][0] else 0.5)) for rule in rules]
		return (nonprobmodel, dict(probmodel))
	return probmodel
	
#freq * reduce(mul, (fd[z] for z in rule[1:] if '@' in z), 1)		/ (float(fd[rule[0]]) * (ntfd[rule[0]] if normalize and '@' not in rule[0] else 1.0))		* interpolate if '@' not in rule[0] else 1.0		+ (1 - interpolate) * (freq / ntfd[rule[0]]) if not any('@' in z for z in rule) else 0

def splitgrammar(grammar):
	""" split the grammar into various lookup tables, mapping nonterminal labels to numeric identifiers"""
	Grammar = namedtuple("Grammar", "unary lbinary rbinary lexical bylhs toid tolabel".split())
	#unary, lbinary, rbinary, lexical, bylhs = {}, {}, {}, {}, {}
	# get a list of all nonterminals; make sure Epsilon and ROOT are first, and assign them unique IDs
	nonterminals = list(enumerate(["Epsilon", "ROOT"] + sorted(set(chain(*(rule for (rule,yf),weight in grammar))) - set(["Epsilon", "ROOT"]))))
	toid, tolabel = dict((lhs, n) for n, lhs in nonterminals), dict((n, lhs) for n, lhs in nonterminals)
	unary, lbinary, rbinary, bylhs = ([[] for a in nonterminals] for b in range(4))
	lexical = defaultdict(list)
	# negate the log probabilities because the heap we use is a min-heap
	for (rule,yf),w in grammar:
		r = tuple(toid[a] for a in rule), yf
		if len(rule) == 2:
			if r[0][1] == 0: #Epsilon
				# lexical productions (mis)use the field for the yield function to store the word
				lexical.setdefault(yf[0], []).append((r, -w))
			else:
				unary[r[0][1]].append((r, -w))
			bylhs[r[0][0]].append((r, -w))
		elif len(rule) == 3:
			lbinary[r[0][1]].append((r, -w))
			rbinary[r[0][2]].append((r, -w))
			bylhs[r[0][0]].append((r, -w))
		else: raise ValueError("grammar not binarized: %s" % repr(r))
	return Grammar(unary, lbinary, rbinary, lexical, bylhs, toid, tolabel)

def canonicalize(tree):
	for a in tree.subtrees(lambda n: len(n) > 1):
		a.sort(key=lambda n: n.leaves())
	return tree

def allmax(seq, key):
	""" return all x s.t. key(x)==max(seq, key)"""
	if not seq: return []
	m = max(map(key, seq))
	return [a for a in seq if key(a) == m]

def same(a, b, asent, bsent):
	if type(a) != type(b): return False
	elif isinstance(a, Tree): return a.node == b.node
	else: return asent[a] == bsent[b]

def fragmentfromindices(tree, sent, indices):
	""" Given a tree and a set of connected indices, return the fragment
	described by these indices.
	>>> np = Tree('NP', [Tree('DET', [2]), Tree('NN', [3])])
	>>> tree = Tree('S', [Tree('NP', [0]), Tree('VP', [Tree('V', [1]), np])])
	>>> sent = "Mary sees a unicorn".split()
	>>> indices = set([(1,), (1, 0), (1, 0, 0), (1, 1)])
	>>> print "%s, %s" % fragmentfromindices(tree, sent, indices)
	(VP (V 0) (NP 1)), ('sees', None)
	"""
	if not indices: return
	froot = min(indices, key=len)
	if not isinstance(tree[froot], Tree): return
	result = Tree(tree[froot].node, [])
	sent = list(sent[:])
	agenda = [(result, a, froot+(n,)) for n,a in enumerate(tree[froot])]
	while agenda:
		current, node, idx = agenda.pop(0)
		if idx in indices:
			if isinstance(node, Tree):
				child = Tree(node.node, [])
				current.append(child)
				new = [(child, a, idx+(n,)) for n,a in enumerate(node)
							if idx+(n,) in indices]
				if new: agenda += new
				else:
					child[:] = rangeheads(sorted(node.leaves()))
					for a in child: sent[a] = None
			else:
				current.append(node)
	# frontier nodes and leaves should both receive index
	# with frontier nodes pointing to None in sent
	# discontinous frontier nodes get multiple indices (ranges)
	# frontier nodes should get rangeheads(a.leaves())
	# renumber
	leaves = result.leaves()
	sent = tuple(a for n,a in enumerate(sent) if n in leaves)
	leafmap = dict(zip(sorted(leaves), count()))
	for n, a in enumerate(result.treepositions('leaves')):
		result[a] = leafmap[result[a]]
	return (result.freeze(), sent) if len(result) else None

def hapaxproductions(trees, sents):
	""" Return a set of productions p, such that p + fragments
	covers the given treebank. Extracts all productions not part of any 
	fragment"""
	return FreqDist(varstoindices(rule) for tree, sent in zip(trees, sents)
					for rule in srcg_productions(tree, sent, False)).hapaxes()

def leaves_and_frontier_nodes(tree, sent):
	return chain(*(leaves_and_frontier_nodes(child, sent)
			if isinstance(child, Tree) and len(child)
			else ([tree] if sent[child] is None else [child])
			for child in tree))

def flatten((tree, sent)):
	return ImmutableTree(tree.node,
		[(a if isinstance(a, Tree) or sent[a] is None
		else ImmutableTree("#&" + sent[a], [a]))
		for a in leaves_and_frontier_nodes(tree, sent)]
		if len(tree) > 1 else tree[:])

def doubledop(trees, sents):
	backtransform = {}
	newprods = []
	# to assign IDs to ambiguous fragments (same yield)
	cnt = count()
	trees = list(trees)
	# adds arity markers
	for t,s in zip(trees, sents): srcg_productions(t, s)
	# most work happens here
	fragments = extractfragments(trees, sents)
	# everything below here is wrong
	# (doesn't deal with discontinuties & frontier nodes properly)
	productions = map(flatten, fragments)
	for prod, (frag, terminals) in zip(productions, fragments):
		if prod == frag: continue
		if prod not in backtransform:
			backtransform[prod] = frag
		else:
			if backtransform[prod]:
				newprods.append((ImmutableTree("#%d" % cnt.next(),
										prod[:]), ()))
				newprods.append((ImmutableTree(prod.node,
										[newprods[-1][0]]), terminals))
				backtransform[newprods[-1][0]] = backtransform[prod, terminals]
				backtransform[prod] = None
			newprods.append((ImmutableTree("#%d" % cnt.next(), prod[:]), ()))
			newprods.append((ImmutableTree(prod.node, [newprods[-1][0]]),
																terminals))
			backtransform[newprods[-1][0]] = frag
	ntfd = defaultdict(float)
	for (a,asent),b in fragments.items(): ntfd[a.node] += b
	# collect rules which occur once to complement the recurring fragments
	hapax = hapaxproductions(trees, sents)
	for a in hapax: ntfd[a[0][0]] += 1
	#ntfd = FreqDist(a.node for tree in trees for a in tree.subtrees())
	# frontier nodes and variables?
	# binarize productions here.
	# 	binarization should not affect earlier binarization --> different chars
	grammar = [rule
		for a, ((f, fsent), b) in zip(productions, fragments.items())
		if backtransform.get(a, False)
		for rule in zip(map(varstoindices,
				srcg_productions(binarizetree(a, "{}"), fsent)),
				chain((log(float(b) / ntfd[f.node]),), repeat(0.0)))]
	grammar += [rule for a, b in newprods
		for rule in zip(map(varstoindices,
			srcg_productions(binarizetree(a, "{}"), b)),
			chain((log((fragments[backtransform[a], b]
				if a in backtransform else 1)
			/ (ntfd[a.node] if a.node in ntfd else 1.0)),), repeat(0.0)))]
	grammar += [(a, log(1.0/ntfd[a[0][0]])) for a in hapax]
	return set(grammar), backtransform

def top_production(tree):
	return ImmutableTree(tree.node, [ImmutableTree(a.node, [])
				if isinstance(a, Tree) else a for a in tree])

def recoverfromfragments(derivation, sent, backtransform):
	prod = top_production(derivation)
	def renumber(tree):
		result = Tree.convert(tree)
		leaves = result.leaves()
		leafmap = dict(zip(leaves, count()))
		for a in result.treepositions('leaves'):
			result[a] = leafmap[result[a]]
		return result.freeze()
	result = Tree.convert(backtransform.get(renumber(prod), prod))
	if len(derivation) == 1 and derivation[0].node[0] == "#":
		derivation = derivation[0]
	for r, t in zip(leaves_and_frontier_nodes(result), derivation):
		if isinstance(r, Tree):
			new = recoverfromfragments(t, backtransform)
			assert r.node == new.node and len(new)
			r[:] = new[:]
		else:
			# terminals should already match.
			assert r == t
	return result

def add_srcg_rules(tree, sent):
	for a, b in zip(tree.subtrees(), srcg_productions(tree, sent, False)):
		a.prod = alpha_normalize(zip(*b))
		a.len = len(a)
	return tree

def extractfragments(trees, sents):
	""" Seeks the largest fragments occurring at least twice in the corpus.
	Algorithm from: Sangati et al., Efficiently extract recurring tree
	fragments from large treebanks. """
	fraglist = defaultdict(set) #FreqDist()
	#partfraglist = set()
	trees = [add_srcg_rules(canonicalize(a).freeze(), b)
				for a, b in zip(trees, sents)]
	trees = [dict((idx, a[idx]) for idx in a.treepositions()) for a in trees]
	
	mem = {}; l = set()
	for n,a,asent in zip(count(), trees, sents):
		for b,bsent in zip(trees, sents)[n+1:]:
			for i in a:
				for j in b:
					try: x = frozenset(mem[i, j])
					except KeyError:
						x = frozenset(extractmaxfragments(a, b, i, j, asent, bsent, mem))
					if x in l or len(x) < 2: continue
					# disjoint-set datastructure here?
					for y in l:
						if x < y: break
						elif x > y:
							l.remove(y)
							l.add(x)
							break
					else: l.add(x)
					#partfraglist.update(extractmaxpartialfragments(a[i], b[j]))
			fragments = (fragmentfromindices(a, asent, x) for x in l)
			#fraglist.update(set(fragments))
			for x in set(fragments):
				if x: fraglist[x].update((a[()], b[()]))
			mem.clear(); l.clear()
	#fraglist.pop(None, None)
	fragcounts = defaultdict(int)
	fragcounts.update((a, len(b)) for a, b in fraglist.items())
	return fragcounts #| partfraglist

def extractmaxfragments(a,b, i,j, asent,bsent, mem):
	""" a fragment is a connected subset of nodes where each node either has
	zero children, or as much as in the original tree.""" 
	if (i, j) in mem: return mem[i, j]
	# compare label & arity / terminal
	if type(a[i]) != type(b[j]):
		mem[i, j] = set()
		return set()
	elif isinstance(a[i], Tree):
		#alhs = a[i].prod[0]
		#blhs = b[j].prod[0]
		#if alhs[0] != blhs[0] or len(alhs[1]) != len(blhs[1]):
		# assume presence of arity markers
		if a[i].node != b[j].node:
			mem[i, j] = set()
			return set()
	elif asent[a[i]] != bsent[b[j]]:
		mem[i, j] = set()
		return set()

	nodeset = set([i])
	if not isinstance(a[i], Tree):
		mem[i, j] = nodeset
		return nodeset
	if a[i].prod == b[j].prod:
		for n, x in enumerate(a[i]):
			nodeset.update(extractmaxfragments(a, b, i+(n,), j+(n,),
													asent, bsent, mem))
	mem[i, j] = nodeset
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

def minimalbinarization(tree, score, sep="|"):
	""" Gildea (2009): Optimal parsing strategies for linear context-free rewriting systems
	Expects an immutable tree where the terminals are integers corresponding to indices.

	>>> minimalbinarization(ImmutableTree("NP", [ImmutableTree("ART", [0]), ImmutableTree("ADJ", [1]), ImmutableTree("NN", [2])]), complexityfanout)
	ImmutableTree('NP', [ImmutableTree('ART', [0]), ImmutableTree('NP|<ADJ-NN>', [ImmutableTree('ADJ', [1]), ImmutableTree('NN', [2])])])
	"""
	def newproduction(a, b):
		#if min(a.leaves()) > min(b.leaves()): a, b = b, a
		if (min(chain(*(y for x,y in nonterms[a]))) >
				min(chain(*(y for x,y in nonterms[b])))): a, b = b, a
		newlabel = "%s%s<%s>" % (tree.node, sep, "-".join(x.node for x,y
				in sorted(nonterms[a] | nonterms[b], key=lambda z: z[1])))
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
			if (y, p1) not in workingset or nonterms[px] & nonterms[p1]:
				continue
			p2 = newproduction(px, p1)
			p2nonterms = nonterms[px] | nonterms[p1]
			x2 = score(p2)
			inferior = [(y, p2x) for y, p2x in workingset
							if nonterms[p2x] == p2nonterms and x2 < y]
			if inferior or p2nonterms not in nonterms.values():
				workingset.add((x2, p2))
				heappush(agenda, (x2, p2))
			for a in inferior:
				workingset.discard(a)
				del nonterms[a[1]]
			nonterms[p2] = p2nonterms

def binarizetree(tree, sep="|"):
	""" Recursively binarize a tree. Tree needs to be immutable."""
	if not isinstance(tree, Tree): return tree
	# bypass algorithm when there are no discontinuities:
	elif len(rangeheads(tree.leaves())) == 1:
		newtree = Tree(tree.node, map(lambda t: binarizetree(t, sep), tree))
		newtree.chomsky_normal_form(childChar=sep)
		return newtree
	return Tree(tree.node, map(lambda t: binarizetree(t, sep),
				minimalbinarization(tree, complexityfanout, sep)))

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

def enumchart(chart, start, tolabel, n=1):
	"""exhaustively enumerate trees in chart headed by start in top down 
		fashion. chart is a dictionary with 
			lhs -> [(insideprob, ruleprob, rhs), (insideprob, ruleprob, rhs) ... ]
		this function doesn't really belong in this file but Cython doesn't
		support generators so this function is "in exile" over here.  """
	for iscore,p,rhs in chart[start]:
		if rhs[0].label == 0: #Epsilon
			yield "(%s %d)" % (tolabel[start.label], rhs[0].vec), p
		else:
			# this doesn't seem to be a good idea:
			#for x in nsmallest(n, cartpi(map(lambda y: enumchart(chart, y, tolabel, n, depth+1), rhs)), key=lambda x: sum(p for z,p in x)):
			#for x in sorted(islice(bfcartpi(map(lambda y: enumchart(chart, y, tolabel, n, depth+1), rhs)), n), key=lambda x: sum(p for z,p in x)):
			for x in islice(bfcartpi(map(lambda y: enumchart(chart, y, tolabel), rhs)), n):
				tree = "(%s %s)" % (tolabel[start.label], " ".join(z for z,px in x))
				yield tree, p+sum(px for z,px in x)

def exportrparse(grammar):
	""" Export an unsplitted grammar to rparse format. All frequencies are 1,
	but probabilities are exported.  """
	def repryf(yf):
		return "[" + ", ".join("[" + ", ".join("true" if a == 1 else "false" for a in b) + "]" for b in yf) + "]"
	def rewritelabel(a):
		a = a.replace("ROOT", "VROOT")
		if "|" in a:
			arity = a.rsplit("_", 1)[-1] if "_" in a[a.rindex(">"):] else "1"
			parent = a[a.index("^")+2:a.index(">",a.index("^"))] if "^" in a else ""
			parent = "^"+"-".join(x.replace("_","") if "_" in x else x+"1" for x in parent.split("-"))
			children = a[a.index("<")+1:a.index(">")].split("-")
			children = "-".join(x.replace("_","") if "_" in x else x+"1" for x in children)
			current = a.split("|")[0]
			current = "".join(current.split("_")) if "_" in current else current+"1"
			return "@^%s%s-%sX%s" % (current, parent, children, arity)
		return "".join(a.split("_")) if "_" in a else a+"1"
	for (r,yf),w in grammar:
		if r[1] != 'Epsilon':
			yield "1 %s:%s --> %s [%s]" % (repr(exp(w)), rewritelabel(r[0]), " ".join(map(rewritelabel, r[1:])), repryf(yf))

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
	from treetransforms import un_collinize, collinize
	from negra import NegraCorpusReader
	import sys, codecs
	# this fixes utf-8 output when piped through e.g. less
	# won't help if the locale is not actually utf-8, of course
	sys.stdout = codecs.getwriter('utf8')(sys.stdout)
	corpus = NegraCorpusReader(".", "sample2\.export", encoding="iso-8859-1", headorder=True, headfinal=True, headreverse=False)
	corpus = NegraCorpusReader("../rparse", "tigerproc\.export", headorder=True, headfinal=True, headreverse=False)
	for tree, sent in zip(corpus.parsed_sents()[:3], corpus.sents()):
		print tree.pprint(margin=999)
		a = binarizetree(tree.freeze())
		print a.pprint(margin=999); print
		un_collinize(a)
		print a.pprint(margin=999), a == tree
	print
	tree = Tree("(S (VP (VP (PROAV 0) (VVPP 2)) (VAINF 3)) (VMFIN 1))")
	sent = "Daruber muss nachgedacht werden".split()
	tree = Tree("(S (NP 1) (VP (V 0) (ADV 2) (ADJ 3)))")
	sent = "is Mary very sad".split()
	tree = Tree("(S (NP 1) (VP (V 0) (ADJ 2)))")
	sent = "is Mary sad".split()
	tree2 = Tree("(S (NP 0) (VP (V 1) (ADV 2) (ADJ 3)))")
	sent2 = "Mary is really happy".split()
	#tree, sent = corpus.parsed_sents()[0], corpus.sents()[0]
	#pprint(srcg_productions(tree, sent))
	#tree.chomsky_normal_form(horzMarkov=1, vertMarkov=0)
	#tree2.chomsky_normal_form(horzMarkov=1, vertMarkov=0)
	collinize(tree, factor="right", horzMarkov=0, vertMarkov=0, tailMarker='', minMarkov=1)
	collinize(tree2, factor="right", horzMarkov=0, vertMarkov=0, tailMarker='', minMarkov=1)
	for (r, yf), w in sorted(induce_srcg([tree.copy(True), tree2], [sent, sent2]),
								key=lambda x: (x[0][0][1] == 'Epsilon', x)):
		print "%.2f %s --> %s\t%r" % (exp(w), r[0], " ".join(r[1:]), list(yf))
	print
	for ((r, yf), w1), (r2, w2) in zip(dop_srcg_rules([tree.copy(True), tree2], [sent, sent2], interpolate=1.0),
									dop_srcg_rules([tree.copy(True), tree2], [sent, sent2], interpolate=0.5)):
		assert (r, yf) == r2
		print "%.2f %.2f %s --> %s\t%r" % (exp(w1), exp(w2), r[0], " ".join(r[1:]), list(yf))

	for a in sorted(exportrparse(induce_srcg([tree.copy(True)], [sent]))): print a

	pprint(splitgrammar(induce_srcg([tree.copy(True)], [sent])))
	#pprint(dop_srcg_rules([tree.copy(True)], [sent], interpolate=0.5))
	do(sent, splitgrammar(dop_srcg_rules([tree], [sent])))

	treebank = """(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (JJ hungry) (NN dog))))
(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (JJ yellow) (NN cat))))
(S (NP (DT The) (JJ little) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN cat)) (VP (VBP ate) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP ate) (NP (DT the) (NN cat))))"""
	treebank = map(Tree, treebank.splitlines())
	sents = [tree.leaves() for tree in treebank]
	for tree, sent in zip(treebank, sents):
		for n, idx in enumerate(tree.treepositions('leaves')): tree[idx] = n
	for (a,aa),b in sorted(extractfragments(treebank, sents).items()):
		print a.pprint(margin=999), aa,b
	print
	trees = list(corpus.parsed_sents()[:100])
	[a.chomsky_normal_form(horzMarkov=1) for a in trees]
	print "fragments",
	for (a,aa),b in sorted(extractfragments(trees, corpus.sents()).items()):
		print a.pprint(margin=999),aa,b
	print
	grammar, backtransform = doubledop(trees, corpus.sents())
	print 'grammar',
	pprint(grammar)
	print "backtransform: {",
	for a,b in backtransform.items():
		print repr(a), ":", b
	print "}"

if __name__ == '__main__':
	from doctest import testmod, NORMALIZE_WHITESPACE, ELLIPSIS
	# do doctests, but don't be pedantic about whitespace (I suspect it is the
	# militant anti-tab faction who are behind this obnoxious default)
	fail, attempted = testmod(verbose=False, optionflags=NORMALIZE_WHITESPACE | ELLIPSIS)
	if attempted and not fail: print "%d doctests succeeded!" % attempted
	main()
