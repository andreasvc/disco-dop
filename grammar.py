import logging, codecs, re
from operator import mul, itemgetter
from array import array
from math import log, exp
from collections import defaultdict
from itertools import chain, count, islice, imap, repeat
from nltk import ImmutableTree, Tree, FreqDist, memoize
from containers import Grammar

frontierorterm = re.compile(r"(\(([^ ]+)( [0-9]+)(?: [0-9])*\))")
rootnode = re.compile(r"\([^ ]+\b")
fanoutre = re.compile("_([0-9]+)(?:@[-0-9]+)?$")

def lcfrs_productions(tree, sent):
	""" Given a tree with integer indices as terminals, and a sentence
	with the corresponding words for these indices, produce a sequence
	of LCFRS productions. Always produces monotone LCFRS rules.
	For best results, tree should be canonicalized.
	>>> tree = Tree.parse("(S (VP_2 (V 0) (ADJ 2)) (NP 1))", parse_leaf=int)
	>>> sent = "is Mary happy".split()
	>>> lcfrs_productions(tree, sent)
	[(('S', 'VP_2', 'NP'), ((0, 1, 0),)),
	(('VP_2', 'V', 'ADJ'), ((0,), (1,))),
	(('V', 'Epsilon'), ('is',)),
	(('ADJ', 'Epsilon'), ('happy',)),
	(('NP', 'Epsilon'), ('Mary',))]
	"""
	leaves = tree.leaves()
	assert len(set(leaves)) == len(leaves), (
		"indices should be unique. indices: %r\ntree: %s" % (leaves, tree))
	assert sent, ("no sentence.\n"
		"tree: %s\nindices: %r\nsent: %r" % (tree.pprint(), leaves, sent))
	assert all(isinstance(a, int) for a in leaves), (
		"indices should be integers.\ntree: %s\nindices: %r\nsent: %r" % (
		tree.pprint(), leaves, sent))
	assert all(0 <= a < len(sent) for a in leaves), (
		"indices should point to a word in the sentence.\n"
		"tree: %s\nindices: %r\nsent: %r" % (tree.pprint(), leaves, sent))
	#tree = canonicalized(tree)
	rules = []
	for st in tree.subtrees():
		if not st: raise ValueError(("Empty node. Frontier nodes should "
			"designate which part(s) of the sentence they contribute to."
			"tree: %s\nindices: %r\nsent: %r" % (tree.pprint(), leaves, sent)))
		elif all(isinstance(a, int) for a in st): #isinstance(st[0], int):
			if len(st) == 1 and sent[st[0]] is not None: # terminal node
				rule = ((st.node, 'Epsilon'), (sent[st[0]],))
			elif all(sent[a] is None for a in st): # frontier node
				continue
			else: raise ValueError(("Preterminals should dominate a single "
				"terminal; frontier nodes should dominate a sequence of "
				"indices that are None in the sentence.\n"
				"subtree: %s\nsent: %r" % (st, sent)))
		elif all(isinstance(a, Tree) for a in st): #isinstance(st[0], Tree):
			# convert leaves() to bitsets
			childleaves = [a.leaves() if isinstance(a, Tree) else [a] for a in st]
			leaves = [(idx, n) for n, child in enumerate(childleaves)
					for idx in child]
			leaves.sort(key=itemgetter(0), reverse=True)
			tmpleaves = leaves[:]
			previdx, prevparent = leaves.pop()
			yf = [[prevparent]]
			while leaves:
				idx, parent = leaves.pop()
				if idx != previdx + 1:	# a discontinuity
					yf.append([parent])
				elif parent != prevparent:	# switch to a different non-terminal
					yf[-1].append(parent)
				# otherwise terminal is part of current range
				previdx, prevparent = idx, parent
			nonterminals = (st.node,) + tuple(a.node for a in st)
			rule = (nonterminals, tuple(map(tuple, yf)))
			#assert len(yf) == len(rangeheads(st.leaves())) == (
			#	int(fanoutre.search(st.node).group(1))
			#		if fanoutre.search(st.node) else len(yf)), (
			#	"rangeheads: %r\nyf: %r\nleaves: %r\n\t%r\n"
			#	"childleaves: %r\ntree:\n%s\nsent: %r" % (
			#		rangeheads(st.leaves()), yf,
			#	st.leaves(), tmpleaves, childleaves, st.pprint(margin=9999), sent))
		else: raise ValueError("Neither Tree node nor integer index:\n"
			"%r, %r" % (st[0], type(st[0])))
		rules.append(rule)
	return rules

def induce_plcfrs(trees, sents, freqs=False):
	""" Induce a probabilistic LCFRS, similar to how a PCFG is read off
	from a treebank """
	grammar = FreqDist(rule for tree, sent in zip(trees, sents)
			for rule in lcfrs_productions(tree, sent))
	if freqs: return grammar
	lhsfd = FreqDist()
	for rule, freq in grammar.iteritems(): lhsfd[rule[0][0]] += freq
	for rule, freq in grammar.iteritems():
		grammar[rule] = log(float(freq) / lhsfd[rule[0][0]])
	return list(grammar.items())

def dop_lcfrs_rules(trees, sents, normalize=False, shortestderiv=False,
		freqs=False):
	""" Induce a reduction of DOP to an LCFRS, similar to how Goodman (1996)
	reduces DOP1 to a PCFG.
	Normalize means the application of the equal weights estimate.
	freqs gives frequencies when enabled; default is probabilities. """
	ids, rules = count(), FreqDist()
	fd, ntfd = FreqDist(), FreqDist()
	for n, tree, sent in zip(count(), trees, sents):
		t = canonicalize(tree.copy(True))
		prods = lcfrs_productions(t, sent)
		ut = decorate_with_ids(n, t)
		uprods = lcfrs_productions(ut, sent)
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

def doubledop(trees, sents, numproc=1, stroutput=True, debug=False):
	""" Extract a Double-DOP grammar from a treebank. That is, a fragment
	grammar containing all fragments that occur at least twice, plus all
	individual productions needed to obtain full coverage.
	Input trees need to be binarized. A second level of binarization (a normal
	form) is needed when fragments are converted to individual grammar rules,
	which occurs through the removal of internal nodes. When the remaining
	nodes do not uniquely identify the fragment, an extra node with an
	identifier is inserted: #n where n in an integer. In fragments with
	terminals, we replace their POS tags with a tag uniquely identifying that
	terminal and tag: tag@word.
	"""
	from fragments import getfragments
	#from treetransforms import minimalbinarization, complexityfanout, addbitsets
	from treetransforms import defaultrightbin, addbitsets
	grammar = {}
	newprods = {}
	backtransform = {}
	# to assign IDs to ambiguous fragments (same yield)
	ids = ("(#%d" % n for n in count())
	# find recurring fragments in treebank, as well as depth-1 'cover' fragments
	fragments = getfragments(trees, sents, numproc)

	# fragments are given back as strings; we could work with trees as strings
	# all the way, but to do binarization and rule extraction, NLTK Tree objects
	# are better.
	if stroutput:
		productions = map(flattenstr, fragments)
	else:
		fragments = dict(((ImmutableTree.parse(a[0], parse_leaf=int), a[1]), b)
			for a, b in fragments.items())
		productions = map(flatten, fragments)
	# construct a mapping of productions to fragments
	for prod, (frag, terminals) in zip(productions, fragments):
		if ((stroutput and frontierorterm.match(prod)) or
			(not stroutput and isinstance(prod[0], int))):
			lexprod = lcfrs_productions(addbitsets(prod), terminals)[0]
			prob = fragments[frag, terminals]
			assert lexprod not in grammar
			grammar[lexprod] = prob
			continue
		if prod in backtransform:
			if backtransform[prod] is not None:
				newlabel = ids.next()
				if stroutput:
					label = prod[1:prod.index(" ")]
					prod1 = newlabel + prod[prod.index(" "):]
					newprod = "(%s %s)" % (label, prod1)
				else:
					label = prod.node
					prod1 = ImmutableTree(newlabel[1:], prod[:])
					newprod = ImmutableTree(prod.node, [prod1])
				newprod = ImmutableTree(label,
					[defaultrightbin(addbitsets(newprod)[0], "}", fanout=True)])
				newprods[newprod] = backtransform[prod][1]
				backtransform[prod1] = backtransform[prod]
				backtransform[prod] = None # disable this production
			newlabel = ids.next()
			if stroutput:
				label = prod[1:prod.index(" ")]
				prod1 = newlabel + prod[prod.index(" "):]
				newprod = "(%s %s)" % (label, prod1)
			else:
				label = prod.node
				prod1 = ImmutableTree(newlabel[1:], prod[:])
				newprod = ImmutableTree(prod.node, [prod1])
			newprod = ImmutableTree(label,
				[defaultrightbin(addbitsets(newprod)[0], "}", fanout=True)])
			newprods[newprod] = terminals
			backtransform[prod1] = frag, terminals
		else:
			backtransform[prod] = frag, terminals
	if debug:
		print "training data:"
		for a, b in zip(trees, sents):
			print a[0].pprint(margin=9999)
			print " ".join('_' if x is None else quotelabel(x) for x in b)
			print
		print "recurring fragments:"
		for a, b in zip(productions, fragments):
			print "fragment: %s\nprod:     %s\nfreq: %2d  sent: %s\n" % (
					b[0], a, fragments[b],
					" ".join('_' if x is None else quotelabel(x) for x in b[1]))
		print "ambiguous fragments:"
		if len(newprods) == 0: print "None"
		for a, b in newprods.iteritems():
			print "prod: %s\nsent: %s" % (a,
					" ".join('_' if x is None else quotelabel(x) for x in b))
			if backtransform.get(a,''): print "frag:", backtransform[a]
			print
		print "backtransform:"
		for a, b in backtransform.items():
			if b: print a, ":\n\t", b[0], " ".join(
					'_' if x is None else quotelabel(x) for x in b[1])

	# collect rules
	grammar.update(rule
		for a, ((_, fsent), b) in zip(productions, fragments.iteritems())
		if backtransform.get(a) is not None
		for rule in zip(lcfrs_productions(defaultrightbin(
			addbitsets(a), "}", fanout=True), fsent), chain((b,), repeat(1))))
	# ambiguous fragments (fragments that map to the same flattened production)
	grammar.update(rule for a, b in newprods.iteritems()
		for rule in zip(lcfrs_productions(a, b),
			chain((fragments.get(backtransform.get(a), 1),), repeat(1))))
	#ensure ascii strings, drop terminals, drop sentinels. drop no-op transforms?
	backtransform = dict((a, str(b[0]) if stroutput else b[0])
			for a, b in backtransform.iteritems() if b is not None) #and a != b)
	# relative frequences as probabilities
	ntfd = defaultdict(float)
	for a, b in grammar.iteritems(): ntfd[a[0][0]] += b
	grammar = [(a, log(b/ntfd[a[0][0]])) for a, b in grammar.iteritems()]
	return grammar, backtransform

def coarse_grammar(trees, sents, level=0):
	""" collapse all labels to X except ROOT and POS tags. """
	if level == 0: repl = lambda x: "X"
	label = re.compile("[^^|<>-]+")
	for tree in trees:
		for subtree in tree.subtrees():
			if subtree.node != "ROOT" and isinstance(subtree[0], Tree):
				subtree.node = label.sub(repl, subtree.node)
	return induce_plcfrs(trees, sents)

def nodefreq(tree, utree, subtreefd, nonterminalfd):
	"""count frequencies of nodes and calculate the number of
	subtrees headed by each node. updates "subtreefd" and "nonterminalfd"
	as a side effect. Expects a normal tree and a tree with IDs.
		@param subtreefd: the FreqDist to store the counts of subtrees
		@param nonterminalfd: the FreqDist to store the counts of non-terminals

	>>> fd = FreqDist()
	>>> tree = Tree("(S (NP mary) (VP walks))")
	>>> utree = decorate_with_ids(1, tree)
	>>> nodefreq(tree, utree, fd, FreqDist())
	4
	>>> fd.items()
	[('S', 4), ('NP', 1), ('NP@1-1', 1), ('VP', 1), ('VP@1-2', 1)]
	"""
	assert isinstance(tree, Tree)
	assert len(tree), ("node with zero children.\n"
		"this error occurs when a node has zero children,"
		"e.g., (TOP (wrong))")
	nonterminalfd[tree.node] += 1.0
	if any(isinstance(a, Tree) for a in tree):
		n = reduce(mul, (nodefreq(x, ux, subtreefd, nonterminalfd) + 1
			for x, ux in zip(tree, utree)))
	else: # lexical production
		n = 1
	subtreefd[tree.node] += n
	# only add counts when utree.node is actually an interior node,
	# e.g., root node receives no ID so shouldn't be counted twice
	if utree.node != tree.node:  #if subtreefd[utree.node] == 0:
		subtreefd[utree.node] += n
	return n

def decorate_with_ids(n, tree):
	""" add unique identifiers to each internal non-terminal of a tree.
	n should be an identifier of the sentence
	>>> tree = Tree("(S (NP (DT the) (N dog)) (VP walks))")
	>>> decorate_with_ids(1, tree)
	Tree('S', [Tree('NP@1-1', [Tree('DT@1-2', ['the']), Tree('N@1-3', ['dog'])]),
			Tree('VP@1-4', ['walks'])])
	"""
	utree = Tree.convert(tree)
	ids = 0
	for child in utree: #top node should not get an ID
		for a in child.subtrees():
			if not isinstance(a, Tree): continue
			#if any(isinstance(b, Tree) for b in a):
			ids += 1
			a.node = "%s@%d-%d" % (a.node, n, ids)
	return utree

packed_graph_ids = 0
packedgraph = {}
def decorate_with_ids_mem(n, tree):
	""" add unique identifiers to each internal non-terminal of a tree.
	this version does memoization, which means that equivalent subtrees
	(including the yield) will get the same IDs. Experimental. """
	def recursive_decorate(tree):
		global packed_graph_ids
		if tree in packedgraphs: return tree
		if isinstance(tree, Tree):
			packed_graph_ids += 1
			packedgraphs[tree] = ImmutableTree("%s@%d-%d" %
				(tree.node, n, packed_graph_ids), map(recursive_decorate, tree))
		return packedgraphs[tree]
	global packed_graph_ids
	packed_graph_ids = 0
	return ImmutableTree(tree.node, map(recursive_decorate, tree))

def quotelabel(label):
	""" Escapes two things: parentheses and non-ascii characters."""
	# this hack escapes non-ascii characters, so that phrasal labels
	# can remain ascii-only.
	return label.replace('(', '[').replace(')', ']').encode('unicode-escape')

def flattenstr((tree, sent)): #(tree, sent)):
	""" This version accepts and returns strings instead of Tree objects
	>>> sent = [None, ',', None, '.']
	>>> tree = "(ROOT (S_2 0 2) (ROOT|<$,>_2 ($, 1) ($. 3)))"
	>>> print flattenstr((tree, sent))
	(ROOT (S_2 0 2) ($,@, 1) ($.@. 3))
	>>> print flattenstr(("(NN 0)", ["foo"]))
	(NN 0)
	"""
	assert isinstance(tree, basestring), (tree, sent)
	def repl(x):
		n = x.group(3) # index w/leading space
		nn = int(n)
		if sent[nn] is None:
			return x.group(0)	# (tag index)
		# (tag@word idx)
		return "(%s@%s%s)" % (x.group(2), quotelabel(sent[nn]), n)
	if tree.count(" ") == 1: return tree
	# give terminals unique POS tags
	newtree = frontierorterm.sub(repl, tree)
	# remove internal nodes
	return "%s %s)" % (newtree[:newtree.index(" ")],
		" ".join(x[0] for x in sorted(frontierorterm.findall(newtree),
			key=lambda y: int(y[2]))))

def flatten((tree, sent)):
	"""
	>>> sent = [None, ',', None, '.']
	>>> tree = Tree("ROOT", [Tree("S_2", [0, 2]), Tree("ROOT|<$,>_2", [Tree("$,", [1]), Tree("$.", [3])])]).freeze()
	>>> print flatten((tree, sent))
	(ROOT (S_2 0 2) ($,@, 1) ($.@. 3))
	>>> tree = ImmutableTree.parse("(S (NP (DT 0) (NN 1)) (VP (VBP 2) (NP 3)))", parse_leaf=int)
	>>> sent = ['The', None, 'saw', None]
	>>> print flatten((tree, sent))
	(S (DT@The 0) (NN 1) (VBP@saw 2) (NP 3))
	"""
	assert isinstance(tree, Tree), (tree,sent)
	if all(isinstance(a, Tree) for a in tree):
		children = [b if isinstance(b, Tree) and sent[b[0]] is None else
			ImmutableTree("%s@%s" % (b.node, quotelabel(sent[b[0]])), b[:])
			for b in preterminals_and_frontier_nodes(tree, sent)]
		children.sort(key=itemgetter(0))
	else:
		children = tree[:]
	return ImmutableTree(tree.node, children)

def preterminals_and_frontier_nodes(tree, sent):
	"""Terminals must be integers; frontier nodes must have indices as well.
	>>> tree = Tree("ROOT", [Tree("S_2", [0, 2]), Tree("ROOT|<$,>_2", [Tree("$,", [1]), Tree("$.", [3])])]).freeze()
	>>> sent = [None, ',', None, '.']
	>>> print list(preterminals_and_frontier_nodes(tree, sent))
	[ImmutableTree('S_2', [0, 2]), ImmutableTree('$,', [1]), ImmutableTree('$.', [3])]
	"""
	if any(isinstance(child, Tree) for child in tree):
		return [b for a in [preterminals_and_frontier_nodes(child, sent)
					for child in tree] for b in a]
	return [tree]

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

def canonicalized(tree):
	""" canonical linear precedence (of first component of each node) order.
	returns a new tree. """
	if not isinstance(tree, Tree): return tree
	children = map(canonicalized, tree)
	if len(children) > 1: children.sort(key=lambda n: n.leaves())
	return Tree(tree.node, children)

def rangeheads(s):
	""" iterate over a sequence of numbers and yield first element of each
	contiguous range. input should be shorted.
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

def defaultparse(wordstags):
	""" a right branching default parse
	>>> defaultparse([('like','X'), ('this','X'), ('example', 'NN'), ('here','X')])
	'(NP (X like) (NP (X this) (NP (NN example) (NP (X here) ))))' """
	if wordstags == []: return ''
	return "(%s (%s %s) %s)" % ("NP", wordstags[0][1],
			wordstags[0][0], defaultparse(wordstags[1:]))

def printrule(r,yf,w):
	return "%.2f %s --> %s\t %r" % (exp(w), r[0], "  ".join(r[1:]), list(yf))

def printrulelatex(((r, yf), w), doexp=True):
	r""" Print a rule in latex format.
	>>> r = ((('VP_2@1', 'NP@2', 'VVINF@5'), ((0,), (1,))), -0.916290731874155)
	>>> printrulelatex(r)
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
	r = tuple([lhs]+newrhs)
	lhs = r[0]; rhs = r[1:]
	print "$",
	if isinstance(lhs[1][0], tuple) or isinstance(lhs[1][0], list):
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

def freeze(l):
	return tuple(map(freeze, l)) if isinstance(l, (list, tuple)) else l

def unfreeze(l):
	return list(map(unfreeze, l)) if isinstance(l, (list, tuple)) else l

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
	""" Export a grammar to rparse format. All frequencies are 1,
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

def read_bitpar_grammar(rules, lexicon, encoding='utf-8', dop=False, ewe=False):
	""" Read a bitpar grammar. Must be a binarized grammar.
	Note that the VROOT symbol will be read as `ROOT' instead. Frequencies will
	be converted to exact relative frequencies (unless ewe is specified)."""
	grammar = []
	ntfd = defaultdict(float)
	ntfd1 = defaultdict(set)
	for a in open(rules):
		a = a.split()
		p, rule = float(a[0]), a[1:]
		if rule[0] == "VROOT": rule[0] = "ROOT"
		ntfd[rule[0]] += p
		if dop: ntfd1[rule[0].split("@")[0]].add(rule[0])
		if len(rule) == 2:
			grammar.append(((tuple(rule), ((0,),)), p))
		elif len(rule) == 3:
			grammar.append(((tuple(rule), ((0,1),)), p))
		else: raise ValueError
		#grammar.append(([(rule[0], [range(len(rule) - 1)])]
		#			+ [(a, [n]) for n, a in enumerate(rule[1:])], p))
	for a in codecs.open(lexicon, encoding=encoding):
		a = a.split()
		word, tags = a[0], a[1:]
		tags = zip(tags[::2], map(float, tags[1::2]))
		for t, p in tags:
			ntfd[t] += p
			if dop: ntfd1[t.split("@")[0]].add(t)
		grammar.extend((((t, 'Epsilon'), (word,)), p) for t, p in tags)
	if ewe:
		return Grammar([(rule,
			log(p / (ntfd[rule[0][0]] * len(ntfd1[rule[0][0].split("@")[0]]))))
			for rule, p in grammar])
	return Grammar([(rule, log(p / ntfd[rule[0][0]]))
							for rule, p in grammar])

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

def write_lcfrs_grammar(grammar, rules, lexicon, encoding='utf-8'):
	""" Writes a grammar as produced by induce_plcfrs or dop_lcfrs_rules
	(so before it goes through Grammar()) into a simple text file format.
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

def read_lcfrs_grammar(rules, lexicon, encoding='utf-8'):
	""" Reads a grammar as produced by write_lcfrs_grammar. """
	rules = (a.strip().split('\t') for a in open(rules))
	lexicon = (a.strip().split('\t') for a in codecs.open(lexicon,
			encoding=encoding))
	grammar = [((a[:-2], tuple(tuple(map(int, b)) for b in a[-2].split(","))),
			float.fromhex(a[-1])) for a in rules]
	grammar += [(((t, 'Epsilon'), (w,)), float.fromhex(p)) for t,w,p in lexicon]
	return Grammar(grammar)

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
		result = fanoutre.search(sym)
		return 1 if result is None else int(result.group(1))
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
		yf = eval(line[line.index("[[["):].replace("false","0").replace(
			"true","1"))[0]
		line = line[:line.index("[[[")].split()
		line.pop(0) #freq?
		prob, lhs = line.pop(0).split(":")
		line.pop(0) # -->
		result.append(((tuple([lhs] + line), tuple(map(tuple, yf))),
			log(float(prob))))
	return result

def do(sent, grammar):
	from parser import parse, pprint_chart
	from disambiguation import marginalize
	print "sentence", sent
	p, start, _ = parse(sent, grammar, start=grammar.toid['S'])
	if start:
		mpp, _ = marginalize(p, start, grammar.tolabel)
		for t in mpp:
			print exp(mpp[t]), t
	else:
		print "no parse"
		pprint_chart(p, sent, grammar.tolabel)
	print

def main():
	from treetransforms import unbinarize, binarize, optimalbinarize
	from treebank import NegraCorpusReader, BracketCorpusReader
	from parser import parse, pprint_chart
	from treetransforms import addfanoutmarkers
	from disambiguation import marginalize, recoverfragments, \
			recoverfragments_str, recoverfragments_strstr
	from kbest import lazykbest
	import sys, codecs
	# this fixes utf-8 output when piped through e.g. less
	# won't help if the locale is not actually utf-8, of course
	stdout = codecs.getwriter('utf8')(sys.stdout)
	#filename = "negraproc.export"
	filename = "sample2.export"
	corpus = NegraCorpusReader(".", filename, encoding="iso-8859-1",
		headorder=True, headfinal=True, headreverse=False, movepunct=True)
	#corpus = BracketCorpusReader(".", "treebankExample.mrg")
	sents = corpus.sents()
	trees = [a.copy(True) for a in corpus.parsed_sents()[:10]]
	for a in trees:
		a.chomsky_normal_form(horzMarkov=1)
		addfanoutmarkers(a)

	print 'plcfrs'
	lcfrs = induce_plcfrs(trees, sents)
	print Grammar(lcfrs)

	print 'dop reduction'
	grammar = Grammar(dop_lcfrs_rules(trees[:2], sents[:2]))
	print grammar
	grammar.testgrammar(logging)

	debug = True
	for stroutput in (True,): #(False, True):
		trees = [a.copy(True) for a in trees]
		grammarx, backtransform = doubledop(trees, sents,
			stroutput=stroutput, debug=debug)
		print '\ndouble dop grammar (stroutput=%r)' % stroutput
		grammar = Grammar(grammarx)
		print unicode(grammar)
		assert grammar.testgrammar(logging) #DOP1 should sum to 1.
		for tree, sent in zip(corpus.parsed_sents(), sents):
			print "sentence:",
			for w in sent: stdout.write(' ' + w)
			root = tree.node
			chart, start, msg = parse(sent, grammar, start=grammar.toid[root],
				exhaustive=True)
			print '\n', msg,
			print "\ngold ", tree.pprint(margin=9999)
			print "double dop",
			if start:
				#if stroutput: mpp, _ = marginalize(chart, start, grammar.tolabel)
				mpp = {}; parsetrees = {}
				for t, p in lazykbest(chart, start, 1000, grammar.tolabel, "}<"):
					t2 = Tree.parse(t, parse_leaf=int)
					if stroutput:
						#r = Tree(recoverfragments_str(canonicalize(t2),
						#		backtransform))
						try:
							r = Tree(recoverfragments_strstr(t,	backtransform))
						except KeyError:
							print 'derivation', t
							raise
					else:
						r = recoverfragments(canonicalize(t2), backtransform)
					unbinarize(r)
					r = rem_marks(r).pprint(margin=9999)
					mpp[r] = mpp.get(r, 0.0) + exp(-p)
					parsetrees.setdefault(r, []).append((t, p))
				print len(mpp), 'parsetrees',
				print sum(map(len, parsetrees.values())), 'derivations'
				for t, tp in sorted(mpp.items(), key=itemgetter(1)):
					print tp, '\n', t,
					print "match:", t == tree.pprint(margin=9999)
					assert len(set(parsetrees[t])) == len(parsetrees[t])
					if not debug: continue
					for deriv, p in sorted(parsetrees[t], key=itemgetter(1)):
						print ' <= %6g %s' % (exp(-p), deriv)
			else:
				print "no parse"
				pprint_chart(chart, sent, grammar.tolabel)
			print
	tree = Tree.parse("(ROOT (S (F (E (S (C (B (A 0))))))))", parse_leaf=int)
	g = Grammar(induce_plcfrs([tree],[range(10)]))
	print "tree: %s\nunary closure:" % tree
	g.printclosure()

if __name__ == '__main__': main()
