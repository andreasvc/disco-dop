import codecs, re
from operator import mul, itemgetter
from array import array
from math import log, exp
from collections import defaultdict
from itertools import chain, count, islice, imap, repeat
from nltk import ImmutableTree, Tree, FreqDist, memoize
from dopg import nodefreq, decorate_with_ids
from containers import Grammar

def srcg_productions(tree, sent, arity_marks=True, side_effect=True):
	""" given a tree with integer indices as terminals, and a sentence
	with the corresponding words for these indices, produce a sequence
	of simple RCG rules. has the side-effect of adding arity
	markers to node labels.
	>>> tree = Tree.parse("(S (NP 1) (VP (V 0) (ADJ 2)))", parse_leaf=int)
	>>> sent = "is Mary happy".split()
	>>> srcg_productions(tree, sent)
	[[('S', [[0, 1, 2]]), ('VP_2', [0, 2]), ('NP', [1])],
	[('NP', ('Mary',)), ('Epsilon', ())],
	[('VP_2', [[0], [2]]), ('V', [0]), ('ADJ', [2])],
	[('V', ('is',)), ('Epsilon', ())],
	[('ADJ', ('happy',)), ('Epsilon', ())]]
	>>> srcg_productions(Tree.parse("(NN 0 1)", parse_leaf=int),[None, None])
	[[('NN', [[0], [1]])]]
	"""
	rules = []
	assert set(tree.leaves()) == set(range(len(sent))), (
		"mismatch between indices and sentence: "
		"each word in sentence should have "
		"exactly one corresponding index in leaves of tree")
	for st in tree.subtrees():
		if not st: raise ValueError(("Empty node. Frontier nodes should "
			"designate which part(s) of the sentence they contribute to."))
		elif all(isinstance(a, int) for a in st): #isinstance(st[0], int):
			if len(st) == 1 and sent[st[0]] is not None:
				rule = [(st.node, (sent[st[0]],)), ('Epsilon', ())]
			elif all(sent[a] is None for a in st): # frontier node
				rule = [(st.node, [[a] for a in st])]
			else: raise ValueError(("Preterminals should dominate a single "
				"terminal; frontier nodes should dominate a sequence of "
				"indices that are None in the sentence.\n"
				"subtree: %s\nsent: %r" % (st, sent)))
		elif all(isinstance(a, Tree) for a in st): #isinstance(st[0], Tree):
			rleaves = [a.leaves() if isinstance(a, Tree) else [a] for a in st]
			rvars = [rangeheads(sorted(l)) for a,l in zip(st, rleaves)]
			lvars = ranges(sorted(a for rng in rleaves for a in rng))
			lvars = [[x for x in a if any(x in c for c in rvars)]
				for a in lvars]
			lhs = node_arity(st, lvars, side_effect) if arity_marks else st.node
			nonterminals = (lhs,) + tuple(node_arity(a, b) if arity_marks
										else a.node for a,b in zip(st, rvars))
			vars = (lvars,) + tuple(rvars)
			if vars[0][0][0] != vars[1][0]:
				# sort the right hand side so that the first argument comes
				# from the first nonterminal
				# A[0,1] -> B[1] C[0]  becomes A[0,1] -> C[0] B[1]
				# although this boils down to a simple swap in a binarized
				# grammar, for generality we do a sort instead
				vars, nonterminals = zip((vars[0], nonterminals[0]),
					*sorted(zip(vars[1:], nonterminals[1:]),
					key=lambda x: vars[0][0][0] != x[0][0]))
			rule = zip(nonterminals, vars)
		else: raise ValueError("Neither Tree node nor integer index:\n"
			"%r, %r" % (st[0], type(st[0])))
		rules.append(rule)
	return rules

def varstoindices(rule):
	""" replace the variable numbers by indices pointing to the nonterminal on
	the right hand side from which they take their value. restricted to ordered
	srcg rules.  A[0,1,2] -> A[0,2] B[1]  becomes  A[0, 1, 0] -> B C

	>>> varstoindices([['S', [[0, 1, 2]]], ['NP', [1]], ['VP', [0, 2]]])
	(('S', 'NP', 'VP'), ((1, 0, 1),))
	"""
	nonterminals, vars = zip(*unfreeze(rule))
	if rule[1][0] != 'Epsilon':
		for x in vars[0]:
			for n,y in enumerate(x):
				for m,z in enumerate(vars[1:]):
					if y in z: x[n] = m
	return nonterminals, freeze(vars[0])

def coarse_grammar(trees, sents, arity_marks=True, level=0):
	""" collapse all labels to X except ROOT and POS tags. """
	if level == 0: repl = lambda x: "X"
	label = re.compile("[^^|<>-]+")
	for tree in trees:
		for subtree in tree.subtrees():
			if subtree.node != "ROOT" and isinstance(subtree[0], Tree):
				subtree.node = label.sub(repl, subtree.node)
	return induce_srcg(trees, sents, arity_marks)

def induce_srcg(trees, sents, arity_marks=True, freqs=False):
	""" Induce a probabilistic SRCG, similar to how a PCFG is read off
	from a treebank """
	rules = []
	for tree, sent in zip(trees, sents):
		rules.extend(map(varstoindices,
				srcg_productions(tree, sent, arity_marks, side_effect=False)))

	grammar = FreqDist(rules)
	fd = FreqDist()
	for rule, freq in grammar.iteritems(): fd.inc(rule[0][0], freq)
	return [(rule, freq if freqs else log(float(freq)/fd[rule[0][0]]))
				for rule, freq in grammar.iteritems()]

def dop_srcg_rules(trees, sents, normalize=False, shortestderiv=False,
					arity_marks=True, freqs=False):
	""" Induce a reduction of DOP to an SRCG, similar to how Goodman (1996)
	reduces DOP1 to a PCFG.
	Normalize means the application of the equal weights estimate.
	arity_marks enables or disables arity markers
	freqs gives frequencies when enabled; default is probabilities. """
	ids, rules = count(), FreqDist()
	fd, ntfd = FreqDist(), FreqDist()
	for tree, sent in zip(trees, sents):
		t = canonicalize(tree.copy(True))
		prods = map(varstoindices, srcg_productions(t, sent, arity_marks))
		ut = decorate_with_ids(t, ids)
		uprods = map(varstoindices, srcg_productions(ut, sent, False))
		# replace addressed root node with unaddressed node
		uprods[0] = ((prods[0][0][0],) + uprods[0][0][1:], uprods[0][1])
		nodefreq(t, ut, fd, ntfd)
		# fd: how many subtrees are headed by node X (e.g. NP or NP@12), 
		# 	counts of NP@... should sum to count of NP
		# ntfd: frequency of a node in corpus (only makes sense for exterior
		#   nodes (NP), not interior nodes (NP@12), the latter are always one)
		rules.update(chain(*([(c,avar) for c in cartpi(list(
								(x,) if x==y else (x,y)
								for x,y in zip(a,b)))]
							for (a,avar),(b,bvar) in zip(prods, uprods))))

	@memoize
	def sumfracs(nom, denoms):
		return sum(nom / denom for denom in denoms)

	def rfe(((r, yf), freq)):
		# relative frequency estimate, aka DOP1 (Bod 1992; Goodman 1996)
		return (r, yf), (freq *
			reduce(mul, (fd[z] for z in r[1:] if '@' in z), 1)
			/ (1 if freqs else (float(fd[r[0]]))))

	def bodewe(((r, yf), freq)):
		# Bod (2003, figure 3) 
		return (r, yf), (freq *
			reduce(mul, (fd[z] for z in r[1:] if '@' in z), 1)
			/ ((1 if freqs else float(fd[r[0]]))
				* (ntfd[r[0]] if '@' not in r[0] else 1.)))

	# map of exterior (unaddressed) nodes to normalized denominators:
	ewedenoms = {}
	def goodmanewe(((r, yf), freq)):
		# Goodman (2003, eq. 1.5). Probably a typographic mistake.
		if '@' in r[0]: return rfe(((r, yf), freq))
		nom = reduce(mul, (fd[z] for z in r[1:] if '@' in z), 1)
		return (r, yf), sumfracs(nom, ewedenoms[r[0]])

	if normalize and False: #goodmanewe
		for aj in fd:
			a = aj.split("@")[0]
			if a == aj: continue	# exterior / unaddressed node
			ewedenoms.setdefault(a, []).append(float(fd[aj] * ntfd[a]))
		for a in ewedenoms: ewedenoms[a] = tuple(ewedenoms[a])
		probmodel = [(rule, log(p))
			for rule, p in imap(goodmanewe, rules.iteritems())]
	elif normalize: #bodewe
		probmodel = [(rule, p if freqs else log(p))
			for rule, p in imap(bodewe, rules.iteritems())]
	else:
		probmodel = [(rule, p if freqs else log(p))
			for rule, p in imap(rfe, rules.iteritems())]
	if shortestderiv:
		nonprobmodel = [(rule, log(1. if '@' in rule[0][0] else 0.5))
							for rule in rules]
		return (nonprobmodel, dict(probmodel))
	return probmodel

def doubledop(trees, sents):
	from _fragmentseeker import extractfragments1
	from treetransforms import minimalbinarization, complexityfanout, addbitsets
	backtransform = {}
	newprods = []
	# to assign IDs to ambiguous fragments (same yield)
	ids = ("#%d" % n for n in count())
	trees = list(trees)
	# adds arity markers
	srcg = FreqDist(rule for tree, sent in zip(trees, sents)
			for rule in map(varstoindices, srcg_productions(tree, sent)))
	# most work happens here
	fragments = extractfragments1(trees, sents)
	fragments = dict(((ImmutableTree.parse(a[0], parse_leaf=int), a[1]), b)
			for a,b in fragments.items())
	productions = map(flatten, fragments)
	# construct a mapping of productions to fragments
	for prod, (frag, terminals) in zip(productions, fragments):
		if prod == frag: continue
		if prod not in backtransform:
			backtransform[prod] = frag
		else:
			if backtransform[prod]:
				newprods.append((ImmutableTree(ids.next(), prod[:]), ()))
				newprods.append((ImmutableTree(prod.node,
						[ImmutableTree(newprods[-1][0].node, [])]), terminals))
				backtransform[newprods[-1][0]] = backtransform[prod, terminals]
				backtransform[prod] = None
			newprods.append((ImmutableTree(ids.next(), prod[:]), ()))
			newprods.append((ImmutableTree(prod.node,
				[ImmutableTree(newprods[-1][0].node, [])]), terminals))
			backtransform[newprods[-1][0]] = frag
	# collect rules
	grammar = dict(rule
		for a, ((f, fsent), b) in zip(productions, fragments.iteritems())
		if backtransform.get(a, False)
		for rule in zip(map(varstoindices, srcg_productions(Tree.convert(
				minimalbinarization(addbitsets(a), complexityfanout, sep="}")),
				fsent, arity_marks=True, side_effect=True)),
			chain((b, ), repeat(1))))
	# ambiguous fragments
	grammar.update(rule for a, b in newprods
		for rule in zip(map(varstoindices, srcg_productions(Tree.convert(
					minimalbinarization(a, complexityfanout, sep="}")),
				b, arity_marks=a in backtransform, side_effect=True)),
			chain((fragments.get((backtransform.get(a), b), 1),), repeat(1))))
	# unseen srcg rules
	grammar.update((a, srcg[a]) for a in set(srcg.keys()) - set(grammar.keys()))
	# relative frequences as probabilities
	ntfd = defaultdict(float)
	for a, b in grammar.iteritems():
		ntfd[a[0][0]] += b
	for a, b in grammar.iteritems():
		grammar[a] = log(b / ntfd.get(a[0][0], b))
	return grammar.items(), backtransform

def postorder(tree, f=None):
	""" Do a postorder traversal of tree; similar to Tree.subtrees(),
	but Tree.subtrees() does a preorder traversal. """
	for child in tree:
		if isinstance(child, Tree):
			for a in postorder(child):
				if not f or f(a): yield a
	if not f or f(tree): yield tree

def canonicalize(tree):
	""" canonical linear precedence (of first component of each node) order """
	for a in postorder(tree, lambda n: len(n) > 1):
		a.sort(key=lambda n: n.leaves())
	return tree

def rangeheads(s):
	""" iterate over a sequence of numbers and yield first element of each
	contiguous range
	>>> rangeheads( (0, 1, 3, 4, 6) )
	[0, 3, 6]
	"""
	return [a for a in s if a - 1 not in s]

def ranges(s):
	""" partition s into a sequence of lists corresponding to contiguous ranges
	>>> list(ranges( (0, 1, 3, 4, 6) ))
	[[0, 1], [3, 4], [6]]"""
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

def baseline(wordstags):
	""" a right branching baseline parse
	>>> baseline([('like','X'), ('this','X'), ('example','X'), ('here','X')])
	'(NP (X like) (NP (X this) (NP (X example) (NP (X here) ))))' """
	if wordstags == []: return ''
	return "(%s (%s %s) %s)" % ("NP", wordstags[0][1],
			wordstags[0][0], baseline(wordstags[1:]))

def printrule(r,yf,w):
	return "%.2f %s --> %s\t %r" % (exp(w), r[0], "  ".join(r[1:]), list(yf))

def printrulelatex(r):
	""" Print a rule in latex format (before it went through varstoindices)
	"""
	lhs = r[0]; rhs = r[1:]
	print "$",
	if isinstance(lhs[1][0], tuple) or isinstance(lhs[1][0], list):
		r = alpha_normalize(r)
		lhs = r[0]; rhs = r[1:]
	if not isinstance(lhs[1][0], tuple) and not isinstance(lhs[1][0], list):
		print r"\textrm{%s}(\textrm{%s})" % (
			lhs[0].replace("$",r"\$"), lhs[1][0]),
	else:
		print "\\textrm{%s}(%s)" % (lhs[0].replace("$","\\$").replace("_","\_"),
			",".join(" ".join("x_{%r}" % a for a in x)  for x in lhs[1])),
	print r"\rightarrow",
	for x in rhs:
		if x[0] == 'Epsilon':
			print r'\epsilon',
		else:
			print "\\textrm{%s}(%s)" % (
				x[0].replace("$","\\$").replace("_","\_"),
				",".join("x_{%r}" % a for a in x[1])),
			if x != rhs[-1]: print '\\:',
	print r' $ \\'

def printrulelatex2(((r, yf), w), doexp=True):
	r""" Print a rule in latex format (after it went through varstoindices)
	>>> r = ((('VP_2@1', 'NP@2', 'VVINF@5'), ((0,), (1,))), -0.916290731874155)
	>>> printrulelatex2(r)
	0.4 &  $ \textrm{VP\_2@1}(x_{0},x_{1}) \rightarrow \textrm{NP@2}(x_{0}) \: \textrm{VVINF@5}(x_{1})  $ \\
	"""
	c = count()
	newrhs = []
	vars = []
	if r[1] == "Epsilon":
		newrhs = [("Epsilon", [])]
		lhs = (r[0], yf)
	else:
		# NB: not working correctly ... variables get mixed up
		for n,a in enumerate(r[1:]):
			z = sum(1 for comp in yf for y in comp if y == n)
			newrhs.append((a, [c.next() for x in range(z)]))
			vars.append(list(newrhs[-1][1]))
		lhs = (r[0], [[vars[x].pop(0) for x in comp] for comp in yf])
	print (exp(w) if doexp else w),"& ",
	printrulelatex(tuple([lhs]+newrhs))

def alpha_normalize(prod):
	""" Substitute variables so that the variables on the left-hand side appear
	consecutively; e.g. [0,1], [2,3] instead of [0,1], [3,4].
	Modifies prod in-place.
	>>> alpha_normalize([('S', [[2, 4, 7]]), ('VP_2', [2, 7]), ('NP', [4])])
	[('S', [[0, 1, 2]]), ('VP_2', [0, 2]), ('NP', [1])]
	>>> alpha_normalize([('NN', [[11, 15]])])
	[('NN', [[0, 1]])]
	"""
	if len(prod) > 1 and prod[1][0] == 'Epsilon': return prod
	lvars = [b for a in prod[0][1] for b in a]
	for b, a in enumerate(lvars):
		if a==b: continue
		for x in prod[0][1]:  # left hand side
			if a in x: x[x.index(a)] = b
		for _, x in prod[1:]: # right hand side
			if a in x: x[x.index(a)] = b
	return prod

def freeze(l):
	return tuple(map(freeze, l)) if isinstance(l, (list, tuple)) else l

def unfreeze(l):
	return list(map(unfreeze, l)) if isinstance(l, (list, tuple)) else l

def leaves_and_frontier_nodes(tree, sent):
	"""Terminals must be integers; frontier nodes must have indices as well.
	>>> tree = Tree("ROOT", [Tree("S_2", [0, 2]), Tree("ROOT|<$,>_2",
		[Tree("$,", [1]), Tree("$.", [3])])]).freeze()
	>>> sent = [None, ',', None, '.']
	>>> print list(leaves_and_frontier_nodes(tree, sent))
	[ImmutableTree('S_2', [0, 2]), 1, 3]
	"""
	if (not any(isinstance(child, Tree) for child in tree)
		and all(sent[child] is None for child in tree)):
		return [tree]
	return [a for b in list(leaves_and_frontier_nodes(child, sent)
			if isinstance(child, Tree) and len(child)
			else ([tree] if sent[child] is None else [child])
			for child in tree) for a in b]

def flatten((tree, sent)):
	"""
	>>> sent = [None, ',', None, '.']
	>>> tree = Tree("ROOT", [Tree("S_2", [0, 2]), Tree("ROOT|<$,>_2",
		[Tree("$,", [1]), Tree("$.", [3])])]).freeze()
	>>> print flatten((tree, sent))
	(ROOT (S_2 0 2) (#&, 1) (#&. 3))
	"""
	assert isinstance(tree, Tree), (tree,sent)
	return ImmutableTree(tree.node,
		[(a if isinstance(a, Tree) or sent[a] is None
		else ImmutableTree("#&" + sent[a], [a]))
		for a in leaves_and_frontier_nodes(tree, sent)]
		if all(isinstance(a, Tree) for a in tree) else tree[:])

def top_production(tree):
	return Tree(tree.node,
		[Tree(a.node, rangeheads(sorted(a.leaves())))
				if isinstance(a, Tree) else a for a in tree])

def recoverfromfragments(derivation, backtransform):
	""" Reconstruct a DOP derivation from a double DOP derivation with
	flattened fragments. """
	prod = top_production(derivation)
	def renumber(tree):
		leaves = tree.leaves()
		leafmap = dict(zip(sorted(leaves), count()))
		for a in tree.treepositions('leaves'):
			tree[a] = leafmap[tree[a]]
		return tree.freeze(), dict(zip(count(), leaves))
	rprod, leafmap = renumber(prod)
	result = Tree.convert(backtransform.get(rprod, prod))
	# revert renumbering
	for a in result.treepositions('leaves'):
		result[a] = leafmap.get(result[a], result[a])
	if (len(derivation) == 1 and isinstance(derivation[0], Tree)
		and derivation[0].node[0] == "#"):
		derivation = derivation[0]
	for r, t in zip(result.subtrees(lambda t: t.height() == 2), derivation):
		if isinstance(r, Tree):
			if isinstance(t, Tree):
				new = recoverfromfragments(t, backtransform)
				#assert r.node == new.node and len(new)
				r[:] = new[:]
			#else: print "?", r, t
			# terminals should already match.
			#assert r == t
	return result

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
		this function doesn't really belong in this file but Cython didn't
		support generators so this function is "in exile" over here.  """
	for edge in chart[start]:
		if edge.left.label == 0: #Epsilon
			yield "(%s %d)" % (tolabel[start.label], edge.left.vec), edge.prob
		else:
			if edge.right: rhs = (edge.left, edge.right)
			else: rhs = (edge.left, )
			for x in islice(bfcartpi(map(lambda y: enumchart(chart, y, tolabel), rhs)), n):
				tree = "(%s %s)" % (tolabel[start.label], " ".join(z for z,px in x))
				yield tree, edge.prob+sum(px for z,px in x)

def exportrparse(grammar):
	""" Export an unsplitted grammar to rparse format. All frequencies are 1,
	but probabilities are exported.  """
	def repryf(yf):
		return "[" + ", ".join("[" + ", ".join("true" if a == 1 else "false"
			for a in b) + "]" for b in yf) + "]"
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
			yield ("1 %s:%s --> %s [%s]" % (repr(exp(w)), rewritelabel(r[0]),
				" ".join(map(rewritelabel, r[1:])), repryf(yf)))

def read_bitpar_grammar(rules, lexicon, encoding='utf-8', ewe=False):
	grammar = []
	ntfd = defaultdict(float)
	ntfd1 = defaultdict(set)
	for a in codecs.open(rules, encoding=encoding):
		a = a.split()
		p, rule = float(a[0]), a[1:]
		if rule[0] == "VROOT": rule[0] = "ROOT"
		ntfd[rule[0]] += p
		ntfd1[rule[0].split("@")[0]].add(rule[0])
		grammar.append(([(rule[0], [range(len(rule) - 1)])]
					+ [(a, [n]) for n, a in enumerate(rule[1:])], p))
	for a in codecs.open(lexicon, encoding=encoding):
		a = a.split()
		word, tags = a[0], a[1:]
		tags = zip(tags[::2], map(float, tags[1::2]))
		for t,p in tags:
			ntfd[t] += p
			ntfd1[t.split("@")[0]].add(t)
		grammar.extend((((t, (word,)), ('Epsilon', ())), p) for t, p in tags)
	if ewe:
		return Grammar([(varstoindices(rule),
			log(p / (ntfd[rule[0][0]] * len(ntfd1[rule[0][0].split("@")[0]]))))
			for rule, p in grammar])
	return Grammar([(varstoindices(rule), log(p / ntfd[rule[0][0]]))
							for rule, p in grammar])

def write_bitpar_grammar(grammar, rules, lexicon, encoding='utf-8'):
	""" write grammar as a bitpar grammar to files specified by rules and
	lexicon."""
	rules = codecs.open(rules, "w", encoding=encoding)
	for a in grammar.bylhs:
		for rule in a:
			assert len(rule.args == 1)		#CFG rule?
			rules.write("%f\t%s\t%s\n" % (
					exp(-rule.prob), grammar.tolabel[rule.lhs],
					"".join(grammar.tolabel[rule.rhs1],
					"\t" + grammar.tolabel[rule.rhs2] if rule.rhs2 else '')))
	lexicon = codecs.open(lexicon, "w", encoding=encoding)
	for word in grammar.lexical:
		lexicon.write("%s\t" % word)
		for term in grammar.lexical[word]:
			lexicon.write("%f\t%s" % (exp(-term.prob),
								grammar.tolabel[term.lhs]))
		lexicon.write("\n")
	rules.close(); lexicon.close()

def write_lncky_grammar(rules, lexicon, out, encoding='utf-8'):
	""" Takes a bitpar grammar and converts it to the format of
	Mark Jonhson's cky parser. """
	grammar = []
	for a in codecs.open(rules, encoding=encoding):
		a = a.split()
		p, rule = a[0], a[1:]
		grammar.append("%s %s --> %s\n" % (p, rule[0], " ".join(rule[1:])))
	for a in codecs.open(lexicon, encoding=encoding):
		a = a.split()
		word, tags = a[0], a[1:]
		tags = zip(tags[::2], tags[1::2])
		grammar.extend("%s %s --> %s\n" % (p, t, word) for t, p in tags)
	assert "VROOT" in grammar[0]
	codecs.open(out, "w", encoding=encoding).writelines(grammar)

def write_srcg_grammar(grammar, rules, lexicon, encoding='utf-8'):
	""" Writes a grammar as produced by induce_srcg or dop_srcg_rules (so
	before it goes through Grammar()) into a simple text file format.
	Fields are separated by tabs. Components of the yield function are
	comma-separated. E.g.
	rules: S	NP	VP	010	0.5
		VP_2	VB	NP	0,1	0.4
	lexicon: NN	Epsilon	Haus	0.3
	"""
	rules = codecs.open(rules, "w", encoding=encoding)
	lexicon = codecs.open(lexicon, "w", encoding=encoding)
	for (r,yf),w in grammar:
		if len(r) == 2 and r[1] == "Epsilon":
			lexicon.write("%s\t%s\t%g\n" % ("\t".join(r), yf[0], w))
		else:
			yfstr = ",".join("".join(map(str, a)) for a in yf)
			rules.write("%s\t%s\t%g\n" % ("\t".join(r), yfstr, w))
	rules.close(); lexicon.close()

def read_srcg_grammar(rules, lexicon, encoding='utf-8'):
	""" Reads a grammar as produced by write_srcg_grammar. """
	rules = (a[:-1].split('\t') for a in codecs.open(rules, encoding=encoding))
	lexicon = (a[:-1].split('\t') for a in codecs.open(lexicon,
															encoding=encoding))
	rules = [((tuple(a[:-2]), tuple(tuple(map(int, b))
			for b in a[-2].split(","))), float(a[-1])) for a in rules]
	lexicon = [((tuple(a[:-2]), (a[-2])), float(a[-1])) for a in lexicon]
	return rules, lexicon

def read_penn_format(corpus, maxlen=15, n=7200):
	trees = [a for a in islice((Tree(a)
		for a in codecs.open(corpus, encoding='iso-8859-1')), n)
		if len(a.leaves()) <= maxlen]
	sents = [a.pos() for a in trees]
	for tree in trees:
		for n, a in enumerate(tree.treepositions('leaves')):
			tree[a] = n
	return trees, sents

def terminals(tree, sent):
	""" Replaces indices with words for a CF-tree. """
	tree = Tree(tree)
	tree.node = "VROOT"
	for a, (w, t) in zip(tree.treepositions('leaves'), sent):
		tree[a] = w
	return tree.pprint(margin=999)

def rem_marks(tree):
	""" Remove arity marks, make sure indices at leaves are integers."""
	for a in tree.subtrees(lambda x: "_" in x.node):
		a.node = a.node.rsplit("_", 1)[0]
	for a in tree.treepositions('leaves'):
		tree[a] = int(tree[a])
	return tree

def alterbinarization(tree):
	"""converts the binarization of rparse to the format that NLTK expects
	S1 is the constituent, CS1 the parent, CARD1 the current sibling/child
	@^S1^CS1-CARD1X1   -->  S1|<CARD1>^CS1 """
	#how to optionally add \2 if nonempty?
	tree = re.sub(
		"@\^([A-Z.,()$]+)\d+(\^[A-Z.,()$]+\d+)*(?:-([A-Z.,()$]+)\d+)*X\d+",
		r"\1|<\3>", tree)
	# remove arity markers
	tree = re.sub(r"([A-Z.,()$]+)\d+", r"\1", tree)
	tree = re.sub("VROOT", r"ROOT", tree)
	assert "@" not in tree
	return tree

def subsetgrammar(a, b):
	""" test whether grammar a is a subset of b. """
	difference = set(imap(itemgetter(0), a)) - set(imap(itemgetter(0), b))
	if not difference: return True
	print "missing productions:"
	for r, yf in difference:
		print printrule(r, yf, 0.0)
	return False

def mean(seq):
	return sum(seq) / float(len(seq)) if seq else None #"zerodiv"

def grammarinfo(grammar, dump=None):
	""" print some statistics on a grammar, before it goes through Grammar().
	dump: if given a filename, will dump distribution of parsing complexity
	to a file (i.e., p.c. 3 occurs 234 times, 4 occurs 120 times, etc."""
	lhs = set(rule[0] for (rule,yf),w in grammar)
	l = len(grammar)
	result = "labels: %d" % len(set(rule[a] for (rule,yf),w in grammar
							for a in range(3) if len(rule) > a))
	result += " of which preterminals: %d\n" % (
		len(set(rule[0] for (rule,yf),w in grammar
		if rule[1] == "Epsilon")) or len(set(rule[a] for (rule,yf),w in grammar
				for a in range(1,3) if len(rule) > a and rule[a] not in lhs)))
	ll = sum(1 for (rule,yf),w in grammar if rule[1] == "Epsilon")
	result += "clauses: %d  lexical clauses: %d" % (l, ll)
	result += " non-lexical clauses: %d\n" % (l - ll)
	n, r, yf, w = max((len(yf), rule, yf, w) for (rule, yf), w in grammar)
	result += "max fan-out: %d in " % n
	result += printrule(r, yf, w)
	result += " average: %g\n" % mean([len(yf) for (rule, yf), w, in grammar])
	n, r, yf, w = max((sum(map(len, yf)), rule, yf, w)
				for (rule, yf), w in grammar if rule[1] != "Epsilon")
	result += "max vars: %d in %s\n" % (n, printrule(r, yf, w))
	def fanout(sym):
		if "_" not in sym: return 1
		return int(sym.split("_")[1].split("@")[0])
	pc = [sum(map(fanout, rule)) for (rule, yf), w in grammar]
	n, r, yf, w = max((sum(map(fanout, rule)), rule, yf, w)
							for (rule, yf), w in grammar)
	result += "max parsing complexity: %d in %s" % (n, printrule(r, yf, w))
	result += " average %g" % mean(pc)
	if dump:
		pcdist = FreqDist(pc)
		open(dump, "w").writelines("%d\t%d\n" % x for x in pcdist.items())
	return result

def read_rparse_grammar(file):
	result = []
	for line in open(file):
		yf = eval(line[line.index("[[["):].replace("false","0").replace("true","1"))[0]
		line = line[:line.index("[[[")].split()
		line.pop(0) #freq?
		prob, lhs = line.pop(0).split(":")
		line.pop(0) # -->
		result.append(((tuple([lhs] + line), tuple(map(tuple, yf))),
			log(float(prob))))
	return result

def do(sent, grammar):
	from plcfrs import parse, pprint_chart
	from disambiguation import marginalize
	print "sentence", sent
	p, start = parse(sent, grammar, start=grammar.toid['S'])
	if start:
		mpp = marginalize(p, start, grammar.tolabel)
		for t in mpp:
			print exp(mpp[t]), t
	else:
		print "no parse"
		pprint_chart(p, sent, grammar.tolabel)
	print

def main():
	from treetransforms import unbinarize, binarize, optimalbinarize
	from treebank import NegraCorpusReader
	import sys, codecs
	# this fixes utf-8 output when piped through e.g. less
	# won't help if the locale is not actually utf-8, of course
	sys.stdout = codecs.getwriter('utf8')(sys.stdout)
	corpus = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1",
		headorder=True, headfinal=True, headreverse=False)
	#corpus = NegraCorpusReader("../rparse", "tigerproc.export", headorder=True,
	#	headfinal=True, headreverse=False)
	for tree, sent in zip(corpus.parsed_sents()[:3], corpus.sents()):
		print tree.pprint(margin=999)
		a = Tree.convert(optimalbinarize(tree))
		print a.pprint(margin=999); print
		unbinarize(a)
		print a.pprint(margin=999), a == tree
	print
	tree = Tree.parse("(S (VP (VP (PROAV 0) (VVPP 2)) (VAINF 3)) (VMFIN 1))",
		parse_leaf=int)
	sent = "Daruber muss nachgedacht werden".split()
	tree = Tree.parse("(S (NP 1) (VP (VB 0) (JJ 2)))", parse_leaf=int)
	sent = "is Gatsby rich".split()
	tree2 = Tree.parse("(S (NP 0) (VP (VB 1) (NP 2)))", parse_leaf=int)
	sent2 = "Daisy loved Gatsby".split()
	#tree, sent = corpus.parsed_sents()[0], corpus.sents()[0]
	#pprint(srcg_productions(tree, sent))
	#tree.chomsky_normal_form(horzMarkov=1, vertMarkov=0)
	#tree2.chomsky_normal_form(horzMarkov=1, vertMarkov=0)
	binarize(tree, factor="right", horzMarkov=0, vertMarkov=0, tailMarker='')
	binarize(tree2, factor="right", horzMarkov=0, vertMarkov=0, tailMarker='')
	for (r, yf), w in sorted(induce_srcg([tree.copy(True), tree2],
		[sent, sent2]), key=lambda x: (x[0][0][1] == 'Epsilon', x)):
		print printrule(r,yf,w)
	print
	for a in sorted(exportrparse(induce_srcg([tree.copy(True)], [sent]))):
		print a

	print "print grammar"
	for a in induce_srcg([tree.copy(True)], [sent]): print a
	print "print Grammar()"
	print Grammar(induce_srcg([tree.copy(True)], [sent]))
	return


	do(sent, Grammar(dop_srcg_rules([tree,tree2], [sent,sent2])))
	grammar = dop_srcg_rules([tree,tree2], [sent,sent2])
	print 'dop reduction'
	for (r, yf), w in sorted(grammar):
		print printrule(r,yf,w)
	trees = list(corpus.parsed_sents()[:10])
	sents = corpus.sents()[:100]
	[a.chomsky_normal_form(horzMarkov=1) for a in trees]
	grammar, backtransform = doubledop(trees, sents)
	print '\ndouble dop grammar',
	for (r, yf), w in sorted(grammar):
		print printrule(r,yf,w)
	print "backtransform: {",
	for a,b in backtransform.items():
		try: print a,
		except: print a.pprint(),
		print ":", b
	print "}"
	grammar = Grammar(grammar)
	grammar.testgrammar()
	from plcfrs import parse
	from disambiguation import marginalize
	for tree, sent in zip(corpus.parsed_sents(), sents[:10]):
		print "sentence", sent
		p, start = parse(sent, grammar, start=grammar.toid['ROOT'],
			exhaustive=True)
		print "gold", canonicalize(tree)
		print "parsetrees:"
		if start:
			mpp = marginalize(p, start, grammar.tolabel)
			for t in mpp:
				t2 = Tree(t)
				for idx in t2.treepositions('leaves'): t2[idx] = int(t2[idx])
				unbinarize(t2, childChar="}")
				r = recoverfromfragments(canonicalize(t2), backtransform)
				unbinarize(r)
				r = rem_marks(r)
				print mpp[t], rem_marks(tree) == r, r
		else: print "no parse"
		print

if __name__ == '__main__':
	from doctest import testmod, NORMALIZE_WHITESPACE, ELLIPSIS
	# do doctests, but don't be pedantic about whitespace (I suspect it is the
	# militant anti-tab faction who are behind this obnoxious default)
	fail, attempted = testmod(verbose=False,
		optionflags=NORMALIZE_WHITESPACE | ELLIPSIS)
	main()
	if attempted and not fail:
		print "%s: %d doctests succeeded!" % (__file__, attempted)
