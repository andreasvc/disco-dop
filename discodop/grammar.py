"""Assorted functions to read off grammars from treebanks."""
from __future__ import division, print_function, absolute_import, \
		unicode_literals
import io
import re
import gzip
import codecs
from operator import mul, itemgetter
from collections import defaultdict, Counter
from itertools import count, islice, repeat
try:
	from cyordereddict import OrderedDict
except ImportError:
	from collections import OrderedDict
from .tree import Tree, ImmutableTree, DiscTree, escape, unescape
from .treebank import LEAVESRE
from .util import openread
from functools import reduce  # pylint: disable=redefined-builtin

RULERE = re.compile(
		r'(?P<RULE1>(?P<LHS1>[^ \t]+).*)\t'
		r'(?P<WEIGHT1>(?P<FREQ1>[-.e0-9]+)(?:\/[0-9]+)?)$'
		r'|(?P<FREQ2>[-.e0-9]+)\t(?P<RULE2>(?P<LHS2>[^ \t]+).*)$')
FRONTIERORTERM = re.compile(r"\(([^ ]+) (([0-9]+)=([^ ()]*)(?: [0-9]+=)*)\)")


def lcfrsproductions(tree, sent, frontiers=False):
	r"""Read off LCFRS productions from a tree with indices and a sentence.

	Tree should contain integer indices as terminals, and a sentence with the
	corresponding words for these indices. Always produces monotone LCFRS
	rules. For best results, tree should be canonicalized. When ``frontiers``
	is ``True``, frontier nodes will generate empty productions, by default
	they are ignored.

	>>> tree = Tree("(S (VP_2 (V 0) (ADJ 2)) (NP 1))")
	>>> sent = "is Mary happy".split()
	>>> print('\n'.join(printrule(r, yf)  # doctest: +NORMALIZE_WHITESPACE
	...		for r, yf in lcfrsproductions(tree, sent)))
	010	S VP_2 NP
	0,1	VP_2 V ADJ
	is	V Epsilon
	happy	ADJ Epsilon
	Mary	NP Epsilon"""
	leaves = tree.leaves()
	if len(set(leaves)) != len(leaves):
		raise ValueError('indices should be unique. indices: %r\ntree: %s'
				% (leaves, tree))
	if not sent:
		raise ValueError('no sentence.\ntree: %s\nindices: %r\nsent: %r'
				% (tree.pprint(), leaves, sent))
	if not all(isinstance(a, int) for a in leaves):
		raise ValueError('indices should be integers.\ntree: %s\nindices: %r\n'
				'sent: %r' % (tree.pprint(), leaves, sent))
	if not all(0 <= a < len(sent) for a in leaves):
		raise ValueError('indices should point to a word in the sentence.\n'
			'tree: %s\nindices: %r\nsent: %r' % (tree.pprint(), leaves, sent))
	rules = []
	for st in tree.subtrees():
		if not st:
			raise ValueError(("Empty node. Frontier nodes should designate "
				"which part(s) of the sentence they contribute to.\ntree:"
				"%s\nindices: %r\nsent: %r" % (tree.pprint(), leaves, sent)))
		# elif all(isinstance(a, int) for a in st):
		elif isinstance(st[0], int):
			if len(st) == 1 and sent[st[0]] is not None:  # terminal node
				rule = ((st.label, 'Epsilon'), (escape(sent[st[0]]), ))
			elif frontiers:
				rule = ((st.label, ), ())
			else:
				continue
			# else:
			# 	raise ValueError(("Preterminals should dominate a single "
			# 		"terminal; frontier nodes should dominate a sequence of "
			# 		"indices that are None in the sentence.\n"
			# 		"subtree: %s\nsent: %r" % (st, sent)))
		elif all(isinstance(a, Tree) for a in st):  # isinstance(st[0], Tree):
			# convert leaves() to bitsets
			childleaves = [a.leaves() if isinstance(a, Tree) else [a]
					for a in st]
			leaves = [(idx, n) for n, child in enumerate(childleaves)
					for idx in child]
			leaves.sort(key=itemgetter(0), reverse=True)
			previdx, prevparent = leaves.pop()
			yf = [[prevparent]]
			while leaves:
				idx, parent = leaves.pop()
				if idx != previdx + 1:  # a discontinuity
					yf.append([parent])
				elif parent != prevparent:  # switch to a different non-terminal
					yf[-1].append(parent)
				# otherwise terminal is part of current range
				previdx, prevparent = idx, parent
			nonterminals = (st.label, ) + tuple(a.label for a in st)
			rule = (nonterminals, tuple(map(tuple, yf)))
		else:
			raise ValueError("Neither Tree node nor integer index:\n"
				"%r, %r" % (st[0], type(st[0])))
		rules.append(rule)
	return rules


def treebankgrammar(trees, sents, extrarules=None):
	"""Induce a probabilistic LCFRS with relative frequencies of productions.

	When trees contain no discontinuities, the result is equivalent to a
	treebank PCFG.

	:param extarules: A dictionary of productions that will be merged with the
		grammar, with (pseudo)frequencies as values."""
	grammar = Counter(rule for tree, sent in zip(trees, sents)
			for rule in lcfrsproductions(tree, sent))
	if extrarules is not None:
		for rule in extrarules:
			grammar[rule] += extrarules[rule]
	lhsfd = Counter()
	for rule, freq in grammar.items():
		lhsfd[rule[0][0]] += freq
	return sortgrammar((rule, (freq, lhsfd[rule[0][0]]))
			for rule, freq in grammar.items())


def dopreduction(trees, sents, packedgraph=False, decorater=None,
		extrarules=None):
	"""Induce a reduction of DOP to an LCFRS.

	Similar to how Goodman (1996, 2003) reduces DOP to a PCFG.
	http://aclweb.org/anthology/W96-0214

	:param packedgraph: packed graph encoding (Bansal & Klein 2010, sec 4.2).
		http://aclweb.org/anthology/P10-1112
	:param decorator: a TreeDecorator instance (packedgraph is ignored if this
		is passed) .
	:returns: a set of rules with the relative frequency estimate as
		probilities, and a dictionary with alternate weights."""
	# fd: how many subtrees are headed by node X (e.g. NP or NP@1-2),
	# 	counts of NP@... should sum to count of NP
	# ntfd: frequency of a node in treebank
	fd = defaultdict(int)
	ntfd = defaultdict(int)
	rules = defaultdict(int)
	if decorater is None:
		decorater = TreeDecorator(memoize=packedgraph)

	# collect rules
	for tree, sent in zip(trees, sents):
		prods = lcfrsproductions(tree, sent)
		dectree = decorater.decorate(tree, sent)
		uprods = lcfrsproductions(dectree, sent)
		nodefreq(tree, dectree, fd, ntfd)
		for (a, avar), (b, bvar) in zip(prods, uprods):
			assert avar == bvar
			for c in cartpi([(x, ) if x == y else (x, y)
					for x, y in zip(a, b)]):
				rules[c, avar] += 1
	if extrarules is not None:
		for rule in extrarules:
			rules[rule] += extrarules[rule]
			fd[rule[0][0]] += extrarules[rule]

	def weights(rule):
		""":returns: rule with RFE and EWE probability."""
		# relative frequency estimate, aka DOP1 (Bod 1992; Goodman 1996, 2003)
		# http://aclweb.org/anthology/C92-3126
		# http://aclweb.org/anthology/W96-0214
		(r, yf), freq = rule
		rfe = ((1 if '@' in r[0] else freq) * reduce(mul,
				(fd[z] for z in r[1:] if '@' in z), 1), fd[r[0]])
		# Bod (2003, figure 3): correction factor for number of subtrees.
		# Caveat: the original formula (Goodman 2003, eq. 8.23) has a_j in the
		# denominator of all rules; this is probably a misprint.
		ewe = (float((1 if '@' in r[0] else freq) *
				reduce(mul, (fd[z] for z in r[1:] if '@' in z), 1))
				/ (fd[r[0]] * (ntfd[r[0]] if '@' not in r[0] else 1)))
		# Goodman (2003, p 135). any rule corresponding to the introduction of
		# a fragment has a probability of 1/2, else 1.
		shortest = 1 if '@' in r[0] else 1 / 2
		# Goodman (2003, eq. 8.22). Prob. of fragment is reduced by factor of 2
		# for each non-root non-terminal it contains.
		bon = 1 / 4 if '@' in r[0] else 1 / (4 * ntfd[r[0]])
		return ((r, yf), rfe), ewe, shortest, bon

	rules = sortgrammar(rules.items())
	rules, ewe, shortest, bon = zip(*(weights(r) for r in rules))
	return list(rules), dict(
			ewe=list(ewe), shortest=list(shortest), bon=list(bon))


def doubledop(trees, sents, debug=False, binarized=True, maxdepth=1,
		maxfrontier=999, complement=False, iterate=False, numproc=None,
		extrarules=None):
	"""Extract a Double-DOP grammar from a treebank.

	That is, a fragment grammar containing fragments that occur at least twice,
	plus all individual productions needed to obtain full coverage.
	Input trees need to be binarized.

	:param binarized: Whether the resulting grammar should be binarized;
		this may be False when bitpar is used which applies its own
		binarization.
	:param maxdepth: add non-maximal/non-recurring fragments with depth
		`1 < depth < maxdepth`.
	:param maxfrontier: limit number of frontier non-terminals; not yet
		implemented.
	:param iterate, complement, numproc: cf. fragments.recurringfragments()
	:returns: a tuple (grammar, altweights, backtransform, fragments)
		:grammar: a sequence of productions.
		:altweights: a dictionary containing alternate weights.
		:backtransform: needed to recover trees from compressed derivations.
		:fragments: a sequence of the fragments used to build the grammar,
		in the same order as they appear in ``grammar``."""
	from .fragments import recurringfragments
	fragments = recurringfragments(trees, sents, numproc, disc=True,
			indices=True, maxdepth=maxdepth, maxfrontier=maxfrontier,
			iterate=iterate, complement=complement)
	return dopgrammar(trees, fragments, debug=debug, binarized=binarized,
			extrarules=extrarules)


def dop1(trees, sents, maxdepth=4, maxfrontier=999, binarized=True,
		extrarules=None):
	"""Return an all-fragments DOP1 model with relative frequencies.

	:param maxdepth: restrict fragments to `1 < depth < maxdepth`.
	:param maxfrontier: limit number of frontier non-terminals; not yet
		implemented.
	:returns: a tuple (grammar, altweights, backtransform, fragments)
		:grammar: a sequence of productions.
		:altweights: a dictionary containing alternate weights.
		:backtransform: needed to recover trees from compressed derivations.
		:fragments: a sequence of the fragments used to build the grammar,
		in the same order as they appear in ``grammar``."""
	from .fragments import allfragments
	fragments = allfragments(trees, sents, maxdepth, maxfrontier)
	return dopgrammar(trees, fragments, binarized=binarized,
			extrarules=extrarules)


def dopgrammar(trees, fragments, binarized=True, extrarules=None, debug=False):
	"""Create a DOP grammar from a set of fragments and occurrences.

	A second level of binarization (a normal form) is needed when fragments are
	converted to individual grammar rules, which occurs through the removal of
	internal nodes. The binarization adds unique identifiers so that each
	grammar rule can be mapped back to its fragment. In fragments with
	terminals, we replace their POS tags with a tag uniquely identifying that
	terminal and tag: ``tag@word``.

	:param fragments: a dictionary of fragments from binarized trees, with
		occurrences as values (a mapping of sentence numbers to counts).
	:param binarized: Whether the resulting grammar should be binarized;
		this may be False when bitpar is used which applies its own
		binarization.
	:param extrarules: Additional rules to add to the grammar.
	:returns: a tuple (grammar, altweights, backtransform, fragments)
		altweights is a dictionary containing alternate weights."""
	def getweight(frag):
		""":returns: frequency, EWE, and other weights for fragment."""
		freq = len(fragments[frag])
		root = frag[1:frag.index(' ')]
		nonterms = frag.count('(') - 1
		# Sangati & Zuidema (2011, eq. 5)
		# FIXME: verify that this formula is equivalent to Bod (2003).
		ewe = sum(1 / fragmentcount[idx]
				for idx in fragments[frag])
		# Bonnema (2003, p. 34)
		bon = 2 ** -nonterms * (freq / ntfd[root])
		short = 0.5
		return freq, ewe, bon, short

	uniformweight = (1, 1, 1, 1)
	grammar = {}
	backtransform = {}
	ids = UniqueIDs()
	# build index of the number of fragments extracted from a tree for ewe
	fragmentcount = defaultdict(int)
	for indices in fragments.values():
		for idx in indices:
			fragmentcount[idx] += 1
	# ntfd: frequency of a non-terminal node in treebank
	ntfd = Counter(node.label for tree in trees for node in tree.subtrees())

	# binarize, turn into LCFRS productions
	# use artificial markers of binarization as disambiguation,
	# construct a mapping of productions to fragments
	for frag in fragments:
		prods, newfrag = flatten(frag, ids, backtransform, binarized)
		prod = prods[0]
		if prod[0][1] == 'Epsilon':  # lexical production
			grammar[prod] = getweight(frag)
			continue

		# first binarized production gets prob. mass
		grammar[prod] = getweight(frag)
		grammar.update(zip(prods[1:], repeat(uniformweight)))
		# & becomes key in backtransform
		backtransform[prod] = frag, newfrag
	if debug:
		ids = UniqueIDs()
		flatfrags = [flatten(frag, ids, {}, binarized)
				for frag in fragments]
		print("recurring fragments:")
		for a, b in zip(flatfrags, fragments):
			print("fragment: %s\nprod:     %s" % (b, "\n\t".join(
				printrule(r, yf, 0) for r, yf in a[0])))
			print("template: %s\nfreq: %2d\n" % (a[1], len(fragments[b])))
		print("backtransform:")
		for a, b in backtransform.items():
			print(a, b)

	if extrarules is not None:
		for rule in extrarules:
			x = extrarules[rule]
			a = b = c = 0
			if rule in grammar:
				a, b, c, _ = grammar[rule]
			grammar[rule] = (a + x, b + x, c + x, 0.5)
	# fix order of grammar rules
	grammar = sortgrammar(grammar.items())
	# align fragments and backtransform with corresponding grammar rules
	fragments = [(frag, fragments[frag]) for frag in
			(backtransform[rule][0] for rule, _ in grammar if rule in
				backtransform)]
	backtransform = [backtransform[rule][1] for rule, _ in grammar
			if rule in backtransform]
	# relative frequences as probabilities (don't normalize shortest & bon)
	ntsums = defaultdict(int)
	ntsumsewe = defaultdict(int)
	for rule, (freq, ewe, _, _) in grammar:
		ntsums[rule[0][0]] += freq
		ntsumsewe[rule[0][0]] += ewe
	eweweights = [float(ewe) / ntsumsewe[rule[0][0]]
			for rule, (_, ewe, _, _) in grammar]
	bonweights = [bon for rule, (_, _, bon, _) in grammar]
	shortest = [s for rule, (_, _, _, s) in grammar]
	grammar = [(rule, (freq, ntsums[rule[0][0]]))
			for rule, (freq, _, _, _) in grammar]
	return grammar, backtransform, dict(
			ewe=eweweights, bon=bonweights, shortest=shortest), fragments


def compiletsg(fragments, binarized=True):
	"""Compile a set of weighted fragments (i.e., a TSG) into a grammar.

	Similar to dopgrammar(), only the values are weights instead of counts.

	:param fragments: a dictionary of fragments mapped to weights. The
		fragments must consist of strings in discbracket format.
	:param binarized: Whether the resulting grammar should be binarized.
	:returns: a ``(grammar, backtransform, altweights)`` tuple similar to what
		``doubledop()`` returns; altweights will be empty."""
	grammar = {}
	backtransform = {}
	ids = UniqueIDs()
	for frag, weight in fragments.items():
		prods, newfrag = flatten(frag, ids, backtransform, binarized)
		if prods[0][0][1] == 'Epsilon':  # lexical production
			grammar[prods[0]] = weight
			continue
		grammar[prods[0]] = weight
		grammar.update(zip(prods[1:], repeat(1)))
		backtransform[prods[0]] = newfrag
	grammar = sortgrammar(grammar.items())
	backtransform = [backtransform[rule] for rule, _ in grammar
			if rule in backtransform]
	return grammar, backtransform, {}


def sortgrammar(grammar, altweights=None):
	"""Sort grammar productions in three clusters: phrasal, binarized, lexical.

	1. normal phrasal rules, ordered by lhs symbol
	2. non-initial binarized 2dop rules (to align the 2dop backtransform with
		the rules in cluster 1 which introduce a new fragment)
	3. lexical rules sorted by word"""
	def sortkey(rule):
		"""Sort key ``(word or '', 2dop binarized rule?, lhs)``."""
		(nts, yf), _p = rule
		word = yf[0] if nts[1] == 'Epsilon' else ''
		return word, '}<' in nts[0], nts[0]

	if altweights is None:
		return sorted(grammar, key=sortkey)

	idx = sorted(range(len(grammar)), key=lambda n: sortkey(grammar[n]))
	altweights = {name: [weights[n] for n in idx]
			for name, weights in altweights.items()}
	grammar = [grammar[n] for n in idx]
	return grammar, altweights


def flatten(frag, ids, backtransform, binarized):
	r"""Auxiliary function for Double-DOP.

	Remove internal nodes from a fragment and read off the (binarized)
	productions of the resulting flattened fragment. Aside from returning
	productions, also return fragment with lexical and frontier nodes replaced
	by a templating symbol '{n}' where n is an index.
	Trees are in the form of strings.

	:param frag: a tree fragment
	:param ids: an iterator which yields unique IDs for non-terminals
		introduced by the binarization
	:returns: a tuple (prods, template).

	>>> ids = UniqueIDs()
	>>> frag = "(ROOT (S_2 0= 2=) (ROOT|<$,>_2 ($, 1=,) ($. 3=.)))"
	>>> prods, template = flatten(frag, ids, {}, True)
	>>> print('\n'.join(printrule(r, yf) for r, yf in prods))
	... # doctest: +NORMALIZE_WHITESPACE
	01	ROOT ROOT}<0> $.@.
	010	ROOT}<0> S_2 $,@,
	,	$,@, Epsilon
	.	$.@. Epsilon
	>>> print(template)
	(ROOT {0} (ROOT|<$,>_2 {1} {2}))
	>>> prods, template = flatten(frag, ids, {}, False)
	>>> print('\n'.join(printrule(r, yf) for r, yf in prods))
	... # doctest: +NORMALIZE_WHITESPACE
	0102	ROOT S_2 $,@, $.@.
	,	$,@, Epsilon
	.	$.@. Epsilon
	>>> print(template)
	(ROOT {0} (ROOT|<$,>_2 {1} {2}))"""
	from .treetransforms import factorconstituent, addbitsets
	sent = {}

	def repl(x):
		"""Add information to a frontier or terminal node.

		:frontiers: ``(label indices)``
		:terminals: ``(label@word idx)``"""
		label = x.group(1)  # label
		word = x.group(4)
		if not word:
			# indices = x.group(2).replace('=', '')
			# return '(%s %s)' % (label, indices)
			return x.group(0)  # (label indices)
		idx = x.group(2)  # index
		# (label@word idx)
		return "(%s@%s %s)" % (label, word, idx)

	def substleaf(x):
		"""Collect word and return index."""
		idx, word = x.split('=', 1)
		idx = int(idx)
		sent[idx] = word or None
		return int(idx)

	if frag.count(' ') == 1:
		frag_ = Tree.parse(frag, parse_leaf=substleaf)
		sent = [sent.get(n, None) for n in range(max(sent) + 1)]
		frag = frag[:frag.index(' ')] + ' 0)'
		return lcfrsproductions(addbitsets(frag_), sent), frag
	# give terminals unique POS tags
	prod = FRONTIERORTERM.sub(repl, frag)
	# remove internal nodes, reorder
	prod = "%s %s)" % (
			prod[:prod.index(' ')],
			' '.join(x.group(0) for x
				in sorted(FRONTIERORTERM.finditer(prod),
					key=lambda x: int(x.group(3)))))
	prod_ = Tree.parse(prod, parse_leaf=substleaf)
	sent = [sent.get(n, None) for n in range(max(sent) + 1)]
	tmp = addbitsets(prod_)
	if binarized:
		tmp = factorconstituent(tmp, "}", factor='left', markfanout=True,
				markyf=True, ids=ids, threshold=2)
	prods = lcfrsproductions(tmp, sent)
	# remember original order of frontiers / terminals for template
	order = {x.group(2): "{%d}" % n
			for n, x in enumerate(FRONTIERORTERM.finditer(prod))}
	# mark substitution sites and ensure string.
	newtree = FRONTIERORTERM.sub(lambda x: order[x.group(2)], frag)
	prod = prods[0]
	if prod in backtransform:
		# normally, rules of fragments are disambiguated by binarization IDs.
		# In case there's a fragment with only one or two frontier nodes,
		# we add an artificial node.
		newlabel = "%s}<%s>%s" % (prod[0][0], next(ids),
				'' if len(prod[1]) == 1 else '_%d' % len(prod[1]))
		prod1 = ((prod[0][0], newlabel) + prod[0][2:], prod[1])
		# we have to determine fanout of the first nonterminal
		# on the right hand side
		prod2 = ((newlabel, prod[0][1]),
			tuple((0,) for component in prod[1]
			for a in component if a == 0))
		prods[:1] = [prod1, prod2]
	return prods, newtree


def nodefreq(tree, dectree, subtreefd, nonterminalfd):
	"""Auxiliary function for DOP reduction.

	Counts frequencies of nodes and calculate the number of
	subtrees headed by each node. updates ``subtreefd`` and ``nonterminalfd``
	as a side effect. Expects a normal tree and a tree with IDs.

	:param subtreefd: the Counter to store the counts of subtrees
	:param nonterminalfd: the Counter to store the counts of non-terminals

	>>> fd = Counter()
	>>> d = TreeDecorator()
	>>> tree = Tree("(S (NP 0) (VP 1))")
	>>> dectree = d.decorate(tree, ['mary', 'walks'])
	>>> nodefreq(tree, dectree, fd, Counter())
	4
	>>> fd == Counter({'S': 4, 'NP': 1, 'VP': 1, 'NP@1-0': 1, 'VP@1-1': 1})
	True"""
	nonterminalfd[tree.label] += 1
	nonterminalfd[dectree.label] += 1
	if isinstance(tree[0], Tree):
		n = reduce(mul, (nodefreq(x, ux, subtreefd, nonterminalfd) + 1
			for x, ux in zip(tree, dectree)))
	else:  # lexical production
		n = 1
	subtreefd[tree.label] += n
	# only add counts when dectree.label is actually an interior node,
	# e.g., root node receives no ID so shouldn't be counted twice
	if dectree.label != tree.label:  # if subtreefd[dectree.label] == 0:
		# NB: assignment, not addition; addressed nodes should be unique, and
		# with packed graph encoding we don't want duplicate counts.
		subtreefd[dectree.label] = n
	return n


class TreeDecorator(object):
	"""Auxiliary class for DOP reduction.

	Adds unique identifiers to each internal non-terminal of a tree.

	:param memoize: if ``True``, identifiers will be reused for equivalent
		subtrees (including all terminals).
	:param n: the initial sentence number.
	"""

	def __init__(self, memoize=False, n=1):
		self.n = n  # sentence number
		self.ids = 0  # node number
		self.memoize = memoize
		self.packedgraphs = {}

	def decorate(self, tree, sent):
		"""Return a copy of tree with labels decorated with IDs.

		>>> d = TreeDecorator()
		>>> tree = Tree('(S (NP (DT 0) (N 1)) (VP 2))')
		>>> print(d.decorate(tree, ['the', 'dog', 'walks']))
		(S (NP@1-0 (DT@1-1 0) (N@1-2 1)) (VP@1-3 2))
		>>> d = TreeDecorator(memoize=True)
		>>> print(d.decorate(Tree('(S (NP (DT 0) (N 1)) (VP 2))'),
		...		['the', 'dog', 'walks']))
		(S (NP@1-1 (DT@1-2 0) (N@1-3 1)) (VP@1-4 2))
		>>> print(d.decorate(Tree('(S (NP (DT 0) (N 1)) (VP 2))'),
		...		['the', 'dog', 'barks']))
		(S (NP@1-1 (DT@1-2 0) (N@1-3 1)) (VP@2-4 2))"""
		if self.memoize:
			self.ids = 0
			# wrap tree to get equality wrt sent
			tree = DiscTree(tree.freeze(), sent)
			dectree = ImmutableTree(tree.label, map(self._recdecorate, tree))
		else:
			dectree = Tree.convert(tree.copy(True))
			# skip top node, should not get an ID
			for m, a in enumerate(islice(dectree.subtrees(), 1, None)):
				a.label = '%s@%d-%d' % (a.label, self.n, m)
		self.n += 1
		return dectree

	def _recdecorate(self, tree):
		"""Traverse subtrees not yet seen."""
		if isinstance(tree, int):
			return tree
		elif tree not in self.packedgraphs:
			self.ids += 1
			self.packedgraphs[tree] = ImmutableTree(("%s@%d-%d" % (
					tree.label, self.n, self.ids)),
					[self._recdecorate(child) for child in tree])
			return self.packedgraphs[tree]
		return self._copyexceptindices(tree, self.packedgraphs[tree])

	def _copyexceptindices(self, tree1, tree2):
		"""Copy the nonterminals from tree2, but take indices from tree1."""
		if not isinstance(tree1, Tree):
			return tree1
		self.ids += 1
		return ImmutableTree(tree2.label,
			[self._copyexceptindices(a, b) for a, b in zip(tree1, tree2)])


class UniqueIDs(object):
	"""Produce strings with numeric IDs.

	Can be used as iterator (IDs will never be re-used) and dictionary (IDs
	will be re-used for same key).

	>>> ids = UniqueIDs()
	>>> print(next(ids))
	0
	>>> print(ids['foo'], ids['bar'], ids['foo'])
	1 2 1"""

	def __init__(self):
		self.cnt = 0  # next available ID
		self.ids = {}  # IDs for labels seen

	def __getitem__(self, key):
		val = self.ids.get(key)
		if val is None:
			val = self.ids[key] = '%d' % self.cnt
			self.cnt += 1
		return val

	def __next__(self):
		self.cnt += 1
		return '%d' % (self.cnt - 1)

	def __iter__(self):
		return self

	next = __next__


def rangeheads(s):
	"""Return first element of each range in a sorted sequence of numbers.

	>>> rangeheads( (0, 1, 3, 4, 6) )
	[0, 3, 6]"""
	sset = set(s)
	return [a for a in s if a - 1 not in sset]


def ranges(s):
	"""Partition s into a sequence of lists corresponding to contiguous ranges.

	>>> list(ranges( (0, 1, 3, 4, 6) ))
	[[0, 1], [3, 4], [6]]"""
	rng = []
	for a in s:
		if not rng or a == rng[-1] + 1:
			rng.append(a)
		else:
			yield rng
			rng = [a]
	if rng:
		yield rng


def defaultparse(wordstags, rightbranching=False):
	"""A default parse to generate when parsing fails.

	:param rightbranching:
		when True, return a right branching tree with NPs,
		otherwise return all words under a single constituent 'NOPARSE'.

	>>> print(defaultparse([('like','X'), ('this','X'), ('example', 'NN'),
	... ('here','X')]))
	(NOPARSE (X like) (X this) (NN example) (X here))
	>>> print(defaultparse([('like','X'), ('this','X'), ('example', 'NN'),
	... ('here','X')], True))
	(NP (X like) (NP (X this) (NP (NN example) (NP (X here)))))"""
	if rightbranching:
		if wordstags[1:]:
			return "(NP (%s %s) %s)" % (wordstags[0][1],
					wordstags[0][0], defaultparse(wordstags[1:], rightbranching))
		return "(NP (%s %s))" % wordstags[0][::-1]
	return "(NOPARSE %s)" % ' '.join("(%s %s)" % a[::-1] for a in wordstags)


def printrule(r, yf, w=None):
	""":returns: a string representation of a rule."""
	yfstr = ','.join(''.join('%s' % b for b in a) for a in yf)
	result = '%s\t%s %s' % (
			yfstr, r[0], ' '.join(x for x in r[1:]))
	if w is None:
		return result
	elif isinstance(w, tuple):
		return '%g/%d %s' % (w[0], w[1], result)
	return '%s %s' % (w, result)


def cartpi(seq):
	"""The cartesian product of a sequence of iterables.

	itertools.product doesn't support infinite sequences!

	>>> list(islice(cartpi([count(), count(0)]), 9))
	[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8)]"""
	if seq:
		return (b + (a, ) for b in cartpi(seq[:-1]) for a in seq[-1])
	return ((), )


def writegrammar(grammar, bitpar=False):
	"""Write a grammar in a simple text file format.

	Rules are written in the order as they appear in the sequence `grammar`,
	except that the lexicon file lists words in sorted order (with tags for
	each word in the order of `grammar`). For a description of the file format,
	see ``docs/fileformats.rst``.

	:param grammar:  a sequence of rule tuples, as produced by
		``treebankgrammar()``, ``dopreduction()``, or ``doubledop()``.
	:param bitpar: when ``True``, use bitpar format: for rules, put weight
		first and leave out the yield function. By default, a format that
		supports LCFRS is used.
	:returns: tuple of strings``(rules, lexicon)``

	Weights are written in the following format:

	- if ``bitpar`` is ``False``, write rational fractions; e.g., ``2/3``.
	- if ``bitpar`` is ``True``, write frequencies (e.g., ``2``)
		if probabilities sum to 1, i.e., in that case probabilities can be
		re-computed as relative frequencies. Otherwise, resort to floating
		point numbers (e.g., ``0.666``, imprecise)."""
	rules, lexicon = [], []
	lexical = OrderedDict()
	freqs = bitpar
	for (r, yf), w in grammar:
		if isinstance(w, tuple):
			if freqs:
				w = '%g' % w[0]
			else:
				w1, w2 = w
				if bitpar:
					w = '%s' % (w1 / w2)  # .hex()
				else:
					w = '%s/%s' % (w1, w2)
		elif isinstance(w, float):
			w = w.hex()
		if len(r) == 2 and r[1] == 'Epsilon':
			lexical.setdefault(yf[0], []).append((r[0], w))
			continue
		elif bitpar:
			rules.append(('%s\t%s\n' % (w, '\t'.join(x for x in r))))
		else:
			yfstr = ','.join(''.join(map(str, a)) for a in yf)
			rules.append(('%s\t%s\t%s\n' % (
					'\t'.join(x for x in r), yfstr, w)))
	for word in lexical:
		lexicon.append(unescape(word))
		for tag, w in lexical[word]:
			lexicon.append('\t%s %s' % (tag, w))
		lexicon.append('\n')
	return ''.join(rules), ''.join(lexicon)


def subsetgrammar(a, b):
	"""Test whether grammar a is a subset of b."""
	difference = set(map(itemgetter(0), a)) - set(map(itemgetter(0), b))
	if not difference:
		return True
	print("missing productions:")
	for r, yf in difference:
		print(printrule(r, yf, 0.0))
	return False


def mean(seq):
	"""Arithmetic mean."""
	return sum(seq) / len(seq)


def grammarinfo(grammar, dump=None):
	"""Print some statistics on a grammar, before it goes through Grammar().

	:param dump: if given a filename, will dump distribution of parsing
		complexity to a file (i.e., p.c. 3 occurs 234 times, 4 occurs 120
		times, etc.)"""
	lhs = {rule[0] for (rule, yf), w in grammar}
	l = len(grammar)
	result = "labels: %d" % len({rule[a] for (rule, yf), w in grammar
							for a in range(3) if len(rule) > a})
	result += " of which preterminals: %d\n" % (
		len({rule[0] for (rule, yf), w in grammar if rule[1] == 'Epsilon'})
		or len({rule[a] for (rule, yf), w in grammar
				for a in range(1, 3) if len(rule) > a and rule[a] not in lhs}))
	ll = sum(1 for (rule, yf), w in grammar if rule[1] == 'Epsilon')
	result += "clauses: %d  lexical clauses: %d" % (l, ll)
	result += " non-lexical clauses: %d\n" % (l - ll)
	n, r, yf, w = max((len(yf), rule, yf, w) for (rule, yf), w in grammar)
	result += "max fan-out: %d in " % n
	result += printrule(r, yf, w)
	result += " mean: %g\n" % mean([len(yf) for (_, yf), _, in grammar])
	n, r, yf, w = max((sum(map(len, yf)), rule, yf, w)
				for (rule, yf), w in grammar if rule[1] != 'Epsilon')
	result += "max variables: %d in %s\n" % (n, printrule(r, yf, w))

	def parsingcomplexity(yf):
		"""Sum the fanouts of LHS & RHS."""
		if isinstance(yf[0], tuple):
			return len(yf) + sum(map(len, yf))
		return 1  # NB: a lexical production has complexity 1

	pc = {(rule, yf, w): parsingcomplexity(yf) for (rule, yf), w in grammar}
	r, yf, w = max(pc, key=pc.get)
	result += "max parsing complexity: %d in %s" % (
			pc[r, yf, w], printrule(r, yf, w))
	result += " mean %g" % mean(pc.values())
	if dump:
		pcdist = Counter(pc.values())
		with io.open(dump, 'w', encoding='utf8') as out:
			out.writelines('%d\t%d\n' % x for x in pcdist.items())
	return result


def grammarstats(filename):
	"""Print statistics for PLCFRS/bitpar grammar (sorted by LHS)."""
	print('LHS\t# rules\tfreq. mass')
	label = cnt = freq = None
	for line in codecs.getreader('utf8')((gzip.open if filename.endswith('.gz')
			else open)(filename)):
		match = RULERE.match(line)
		if (match.group('LHS1') or match.group('LHS2')) != label:
			if label is not None:
				print('%s\t%d\t%d' % (label, cnt, freq))
			cnt = freq = 0
			label = (match.group('LHS1') or match.group('LHS2'))
		cnt += 1
		freq += float((match.group('FREQ1') or match.group('FREQ2')))
	if label is not None:
		print('%s\t%d\t%g' % (label, cnt, freq))


def splitweight(weight):
	"""Convert a weight / fraction in a string to a float / tuple.

	>>> [splitweight(a) for a in ('0.5', '0x1.0000000000000p-1', '1/2')]
	[0.5, 0.5, (1.0, 2.0)]"""
	if '/' in weight:
		a, b = weight.split('/')
		return (float(a), float(b))
	elif weight.startswith('0x'):
		return float.fromhex(weight)
	return float(weight)


def convertweight(weight):
	"""Convert a weight in a string to a float.

	>>> [convertweight(a) for a in ('0.5', '0x1.0000000000000p-1', '1/2')]
	[0.5, 0.5, 0.5]"""
	if '/' in weight:
		a, b = weight.split('/')
		return float(a) / float(b)
	elif weight.startswith('0x'):
		return float.fromhex(weight)
	return float(weight)


def stripweight(line):
	"""Extract rule without weight."""
	match = RULERE.match(line.strip())
	if match is None:
		raise ValueError('Malformed rule:\n%s' % line)
	return match.group('RULE1') or match.group('RULE2')


def sumrules(iterable, n):
	"""Given a sorted iterable of rules, sum weights of identical rules."""
	prev = None
	w1 = 0.0
	for line in iterable:
		match = RULERE.match(line)
		rule = match.group('RULE1') or match.group('RULE2')
		if rule != prev:
			if prev is not None:
				if match.group('RULE1') is None:
					yield '%g\t%s\n' % (w1 / n, rule)
				else:
					yield '%s\t%g\n' % (rule, w1 / n)
			prev = rule
			w1 = 0.0
		if match.group('RULE1') is None:
			w1 += convertweight(match.group('FREQ2'))
		else:
			w1 += convertweight(match.group('WEIGHT1'))
	if match.group('RULE1') is None:
		yield '%g\t%s\n' % (w1 / n, rule)
	else:
		yield '%s\t%g\n' % (rule, w1 / n)


def sumlex(iterable, n):
	"""Given a sorted lexicon iterable, sum weights of word/tag pairs."""
	prev = tags = None
	for line in iterable:
		word, rest = line.split(None, 1)
		rest = rest.split()
		if word != prev:
			if tags:
				yield '%s\t%s\n' % (prev, '\t'.join(
						'%s %g' % (a, w1 / n) for a, w1 in tags.items()))
			prev = word
			tags = {}
		for tag, w2 in zip(rest[::2], rest[1::2]):
			if tag not in tags:
				tags[tag] = 0.0
			tags[tag] += convertweight(w2)
	if tags:
		yield '%s\t%s\n' % (prev, '\t'.join(
				'%s %g' % (a, w1 / n) for a, w1 in tags.items()))


def sumfrags(iterable, n):
	"""Sum weights for runs of identical fragments."""
	prev = None
	w1 = 0.0
	for line in iterable:
		frag, w2 = line.rsplit('\t', 1)
		if frag != prev:
			if prev is not None:
				yield '%s\t%g\n' % (prev, w1 / n)
			prev = frag
			w1 = 0.0
		w1 += convertweight(w2)
	if prev is not None:
		yield '%s\t%g\n' % (prev, w1 / n)


def merge(filenames, outfilename, sumfunc, key):
	"""Interpolate weights of given files."""
	from . import plcfrs
	openfiles = [iter(openread(filename)) for filename in filenames]
	with codecs.getwriter('utf8')((gzip.open if outfilename.endswith('.gz')
			else open)(outfilename, 'wb')) as out:
		out.writelines(sumfunc(
				plcfrs.merge(*openfiles, key=key), len(openfiles)))


def addindices(frag):
	"""Convert fragment in bracket to discbracket format."""
	cnt = count()
	return LEAVESRE.sub(lambda m: ' %d=%s)' % (next(cnt), m.group(1)), frag)


__all__ = ['lcfrsproductions', 'treebankgrammar', 'dopreduction', 'doubledop',
		'dop1', 'dopgrammar', 'compiletsg', 'sortgrammar', 'flatten',
		'nodefreq', 'TreeDecorator', 'UniqueIDs', 'mean', 'addindices',
		'rangeheads', 'ranges', 'defaultparse', 'printrule', 'cartpi',
		'writegrammar', 'subsetgrammar', 'grammarinfo', 'grammarstats',
		'splitweight', 'convertweight', 'stripweight', 'sumrules', 'sumlex',
		'sumfrags', 'merge']
