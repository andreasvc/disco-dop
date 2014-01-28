"""Assorted functions to read off grammars from treebanks."""
from __future__ import division, print_function
import io
import re
import sys
import codecs
import logging
from math import exp
from operator import mul, itemgetter
from collections import defaultdict, OrderedDict, Counter as multiset
from itertools import count, islice, repeat
from discodop.tree import ImmutableTree, Tree
from discodop.treebank import READERS
if sys.version[0] >= '3':
	from functools import reduce  # pylint: disable=W0622
	unicode = str  # pylint: disable=W0622,C0103

FORMAT = '''\
The PLCFRS format is as follows. Rules are delimited by newlines.
Fields are separated by tabs. The fields are:

LHS	RHS1	[RHS2]	yield-function	weight

The yield function defines how the spans of the RHS nonterminals
are combined to form the spans of the LHS nonterminal. Components of the yield
function are comma-separated, 0 refers to a component of the first RHS
nonterminal, and 1 from the second. Weights are expressed as rational
fractions.
The lexicon is defined in a separate file. Lines start with a single word,
followed by pairs of possible tags and their probabilities:

WORD	TAG1	PROB1	[TAG2	PROB2 ...]

Example:
rules:   S	NP	VP	010	1/2
         VP_2	VB	NP	0,1	2/3
         NP	NN	0	1/4
lexicon: Haus	NN	3/10	JJ	1/9\
'''

USAGE = '''\
Read off grammars from treebanks.
Usage: %(cmd)s <type> <input> <output> [options]

type is one of:
   pcfg
   plcfrs
   ptsg
   dopreduction
   doubledop

input is a binarized treebank, or in the ptsg case, weighted fragments
in the same format as the output of the discodop fragments command;
output is the base name for the filenames to write the grammar to.

Options (* marks default option):
  --inputfmt=[*%(corpusfmts)s]
  --inputenc=[*utf-8|iso-8859-1|...]
  --dopestimator=[*rfe|ewe|shortest|...]
  --numproc=[*1|2|...]  only relevant for double dop fragment extraction
  --gzip                compress output with gzip, view with zless &c.
  --packed              use packed graph encoding for DOP reduction
  --unbinarized         produce an unbinarized grammar for use with bitpar
  -s X                  start symbol to use for PTSG.

When a PCFG is requested, or the input format is 'bracket' (Penn format), the
output will be in bitpar format. Otherwise the grammar is written as a PLCFRS.
The encoding of the input treebank may be specified. Output encoding will be
ASCII for the rules, and utf-8 for the lexicon.
\n%(grammarformat)s
''' % dict(cmd=sys.argv[0], grammarformat=FORMAT,
		corpusfmts='|'.join(READERS.keys()))


def lcfrsproductions(tree, sent, frontiers=False):
	"""Read off LCFRS productions from a tree with indices and a sentence.

	Tree should contain integer indices as terminals, and a sentence with the
	corresponding words for these indices. Always produces monotone LCFRS
	rules. For best results, tree should be canonicalized. When ``frontiers``
	is ``True``, frontier nodes will generate empty productions, by default
	they are ignored.

	>>> tree = Tree.parse("(S (VP_2 (V 0) (ADJ 2)) (NP 1))", parse_leaf=int)
	>>> sent = "is Mary happy".split()
	>>> for p in lcfrsproductions(tree, sent): print(p)
	(('S', 'VP_2', 'NP'), ((0, 1, 0),))
	(('VP_2', 'V', 'ADJ'), ((0,), (1,)))
	(('V', 'Epsilon'), ('is',))
	(('ADJ', 'Epsilon'), ('happy',))
	(('NP', 'Epsilon'), ('Mary',))"""
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
	rules = []
	for st in tree.subtrees():
		if not st:
			raise ValueError(("Empty node. Frontier nodes should designate "
				"which part(s) of the sentence they contribute to.\ntree:"
				"%s\nindices: %r\nsent: %r" % (tree.pprint(), leaves, sent)))
		#elif all(isinstance(a, int) for a in st):
		elif isinstance(st[0], int):
			if len(st) == 1 and sent[st[0]] is not None:  # terminal node
				rule = ((st.label, 'Epsilon'), (sent[st[0]], ))
			#elif all(sent[a] is None for a in st): # frontier node
			elif frontiers:
				rule = ((st.label, ), ())
			else:
				continue
			#else:
			#	raise ValueError(("Preterminals should dominate a single "
			#		"terminal; frontier nodes should dominate a sequence of "
			#		"indices that are None in the sentence.\n"
			#		"subtree: %s\nsent: %r" % (st, sent)))
		elif all(isinstance(a, Tree) for a in st):  # isinstance(st[0], Tree):
			# convert leaves() to bitsets
			childleaves = [a.leaves() if isinstance(a, Tree) else [a]
					for a in st]
			leaves = [(idx, n) for n, child in enumerate(childleaves)
					for idx in child]
			leaves.sort(key=itemgetter(0), reverse=True)
			#tmpleaves = leaves[:]
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


def treebankgrammar(trees, sents):
	"""Induce a probabilistic LCFRS with relative frequencies of productions.

	When trees contain no discontinuities, the result is equivalent to a
	treebank PCFG."""
	grammar = multiset(rule for tree, sent in zip(trees, sents)
			for rule in lcfrsproductions(tree, sent))
	lhsfd = multiset()
	for rule, freq in grammar.items():
		lhsfd[rule[0][0]] += freq
	return sortgrammar((rule, (freq, lhsfd[rule[0][0]]))
			for rule, freq in grammar.items())


def dopreduction(trees, sents, packedgraph=False):
	"""Induce a reduction of DOP to an LCFRS.

	Similar to how Goodman (1996, 2003) reduces DOP to a PCFG.

	:param packedgraph: packed graph encoding (Bansal & Klein 2010).
	:returns: a set of rules with the relative frequency estimate as
		probilities, and a dictionary with alternate weights."""
	# fd: how many subtrees are headed by node X (e.g. NP or NP@1-2),
	# 	counts of NP@... should sum to count of NP
	# ntfd: frequency of a node in treebank
	fd = defaultdict(int)
	ntfd = defaultdict(int)
	rules = defaultdict(int)
	decorater = TreeDecorator(memoize=packedgraph)

	# collect rules
	for tree, sent in zip(trees, sents):
		prods = lcfrsproductions(tree, sent)
		dectree = decorater.decorate(tree, sent)
		uprods = lcfrsproductions(dectree, sent)
		nodefreq(tree, dectree, fd, ntfd)
		for (a, avar), (b, bvar) in zip(prods, uprods):
			assert avar == bvar
			for c in cartpi([(x, ) if x == y else (x, y) for x, y in zip(a, b)]):
				rules[c, avar] += 1

	def weights(rule):
		""":returns: rule with RFE and EWE probability."""
		# relative frequency estimate, aka DOP1 (Bod 1992; Goodman 1996, 2003)
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


def doubledop(trees, sents, debug=False, binarized=True,
		complement=False, iterate=False, numproc=None):
	"""Extract a Double-DOP grammar from a treebank

	That is, a fragment grammar containing fragments that occur at least twice,
	plus all individual productions needed to obtain full coverage.
	Input trees need to be binarized. A second level of binarization (a normal
	form) is needed when fragments are converted to individual grammar rules,
	which occurs through the removal of internal nodes. The binarization adds
	unique identifiers so that each grammar rule can be mapped back to its
	fragment. In fragments with terminals, we replace their POS tags with a tag
	uniquely identifying that terminal and tag: ``tag@word``.

	:param binarized: Whether the resulting grammar should be binarized.
	:param iterate, complement, numproc: cf. fragments.getfragments()
	:returns: a tuple (grammar, altweights, backtransform)
		altweights is a dictionary containing alternate weights."""
	def getweight(frag, terminals):
		""":returns: frequency, EWE, and other weights for fragment."""
		freq = sum(fragments[frag, terminals].values())
		root = frag[1:frag.index(' ')]
		nonterms = frag.count('(') - 1
		# Sangati & Zuidema (2011, eq. 5)
		# FIXME: verify that this formula is equivalent to Bod (2003).
		ewe = sum(v / fragmentcount[k]
				for k, v in fragments[frag, terminals].items())
		# Bonnema (2003, p. 34)
		bon = 2 ** -nonterms * (freq / ntfd[root])
		short = 0.5
		return freq, ewe, bon, short

	from discodop.fragments import getfragments
	uniformweight = (1, 1, 1, 1)
	grammar = {}
	backtransform = {}
	ids = UniqueIDs()
	fragments = getfragments(trees, sents, numproc,
			iterate=iterate, complement=complement)
	# build index of the number of fragments extracted from a tree for ewe
	fragmentcount = defaultdict(int)
	for indices in fragments.values():
		for index, cnt in indices.items():
			fragmentcount[index] += cnt
	# ntfd: frequency of a non-terminal node in treebank
	ntfd = multiset(node.label for tree in trees for node in tree.subtrees())

	# binarize, turn into LCFRS productions
	# use artificial markers of binarization as disambiguation,
	# construct a mapping of productions to fragments
	for frag, terminals in fragments:
		prods, newfrag = flatten(frag, terminals, ids, backtransform, binarized)
		prod = prods[0]
		if prod[0][1] == 'Epsilon':  # lexical production
			grammar[prod] = getweight(frag, terminals)
			continue

		# first binarized production gets prob. mass
		grammar[prod] = getweight(frag, terminals)
		grammar.update(zip(prods[1:], repeat(uniformweight)))
		# & becomes key in backtransform
		backtransform[prod] = newfrag
	if debug:
		ids = count()
		flatfrags = [flatten(frag, terminals, ids, {}, binarized)
				for frag, terminals in fragments]
		print("recurring fragments:")
		for a, b in zip(flatfrags, fragments):
			print("fragment: %s\nprod:     %s" % (b[0], "\n\t".join(
				printrule(r, yf, 0) for r, yf in a[0])))
			print("template: %s\nfreq: %2d  sent: %s\n" % (
					a[1], len(fragments[b]), ' '.join('_' if x is None
					else quotelabel(x) for x in b[1])))
		print("backtransform:")
		for a, b in backtransform.items():
			print(a, b)

	# fix order of grammar rules; backtransform will mirror this order
	grammar = sortgrammar(grammar.items())

	# replace keys with numeric ids of rules, drop terminals.
	backtransform = [backtransform[rule] for rule, _ in grammar
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
			ewe=eweweights, bon=bonweights, shortest=shortest)


def compiletsg(fragments, binarized=True):
	"""Compile a set of weighted fragments (i.e., a TSG) into a grammar.

	:param fragments: a dictionary of fragments mapped to weights. The
		fragments may either consist of bracketed strings, or discontinuous
		bracketed strings as tuples of the form ``(frag, terminals)``.
	:param binarized: Whether the resulting grammar should be binarized.
	:returns: a ``(grammar, backtransform, altweights)`` tuple similar to what
		``doubledop()`` returns; altweights will be empty."""
	from discodop.treebank import LEAVESRE
	grammar = {}
	backtransform = {}
	ids = UniqueIDs()
	for frag, weight in fragments.items():
		if isinstance(frag, tuple):
			frag, terminals = frag
		else:  # convert to frag, terminal notation on the fly.
			cnt = count()
			terminals = [a or None for a in LEAVESRE.findall(frag)]
			frag = LEAVESRE.sub(lambda _: ' %d)' % next(cnt), frag)
		prods, newfrag = flatten(frag, terminals, ids, backtransform, binarized)
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

FRONTIERORTERM = re.compile(r"\(([^ ]+)( [0-9]+)(?: [0-9]+)*\)")


def flatten(tree, sent, ids, backtransform, binarized):
	"""Auxiliary function for Double-DOP.

	Remove internal nodes from a tree and read off the (binarized)
	productions of the resulting flattened tree. Aside from returning
	productions, also return tree with lexical and frontier nodes replaced by a
	templating symbol '{n}' where n is an index.
	Input is a tree and sentence, as well as an iterator which yields
	unique IDs for non-terminals introdudced by the binarization;
	output is a tuple (prods, frag). Trees are in the form of strings.

	>>> ids = UniqueIDs()
	>>> sent = [None, ',', None, '.']
	>>> tree = "(ROOT (S_2 0 2) (ROOT|<$,>_2 ($, 1) ($. 3)))"
	>>> flatten(tree, sent, ids, {}, True)  # doctest: +NORMALIZE_WHITESPACE
	([(('ROOT', 'ROOT}<0>', '$.@.'), ((0, 1),)),
	(('ROOT}<0>', 'S_2', '$,@,'), ((0, 1, 0),)),
	(('$,@,', 'Epsilon'), (',',)), (('$.@.', 'Epsilon'), ('.',))],
	'(ROOT {0} (ROOT|<$,>_2 {1} {2}))')
	>>> flatten(tree, sent, ids, {}, False)  # doctest: +NORMALIZE_WHITESPACE
	([(('ROOT', 'S_2', '$,@,', '$.@.'), ((0, 1, 0, 2),)),
		 (('$,@,', 'Epsilon'), (',',)), (('$.@.', 'Epsilon'), ('.',))],
	'(ROOT {0} (ROOT|<$,>_2 {1} {2}))')"""
	from discodop.treetransforms import factorconstituent, addbitsets

	def repl(x):
		"""Add information to a frontier or terminal node.

		:frontiers: ``(label indices)``
		:terminals: ``(tag@word idx)``"""
		n = x.group(2)  # index w/leading space
		nn = int(n)
		if sent[nn] is None:
			return x.group(0)  # (label indices)
		word = quotelabel(sent[nn])
		# (tag@word idx)
		return "(%s@%s%s)" % (x.group(1), word, n)

	if tree.count(' ') == 1:
		return lcfrsproductions(addbitsets(tree), sent), str(tree)
	# give terminals unique POS tags
	prod = FRONTIERORTERM.sub(repl, tree)
	# remove internal nodes, reorder
	prod = "%s %s)" % (prod[:prod.index(' ')],
			' '.join(x.group(0) for x in sorted(FRONTIERORTERM.finditer(prod),
			key=lambda x: int(x.group(2)))))
	tmp = addbitsets(prod)
	if binarized:
		tmp = factorconstituent(tmp, "}", factor='left', markfanout=True,
				markyf=True, ids=ids, threshold=2)
	prods = lcfrsproductions(tmp, sent)
	# remember original order of frontiers / terminals for template
	order = {x.group(2): "{%d}" % n
			for n, x in enumerate(FRONTIERORTERM.finditer(prod))}
	# mark substitution sites and ensure string.
	newtree = FRONTIERORTERM.sub(lambda x: order[x.group(2)], tree)
	prod = prods[0]
	if prod in backtransform:
		# normally, rules of fragments are disambiguated by binarization IDs.
		# In case there's a fragment with only one or two frontier nodes,
		# we add an artficial node.
		newlabel = "%s}<%d>%s" % (prod[0][0], next(ids),
				'' if len(prod[1]) == 1 else '_%d' % len(prod[1]))
		prod1 = ((prod[0][0], newlabel) + prod[0][2:], prod[1])
		# we have to determine fanout of the first nonterminal
		# on the right hand side
		prod2 = ((newlabel, prod[0][1]),
			tuple((0,) for component in prod[1]
			for a in component if a == 0))
		prods[:1] = [prod1, prod2]
	return prods, str(newtree)


def nodefreq(tree, dectree, subtreefd, nonterminalfd):
	"""Auxiliary function for DOP reduction.

	Counts frequencies of nodes and calculate the number of
	subtrees headed by each node. updates ``subtreefd`` and ``nonterminalfd``
	as a side effect. Expects a normal tree and a tree with IDs.

	:param subtreefd: the multiset to store the counts of subtrees
	:param nonterminalfd: the multiset to store the counts of non-terminals

	>>> fd = multiset()
	>>> d = TreeDecorator()
	>>> tree = Tree("(S (NP mary) (VP walks))")
	>>> dectree = d.decorate(tree, ['mary', 'walks'])
	>>> nodefreq(tree, dectree, fd, multiset())
	4
	>>> fd == multiset({'S': 4, 'NP': 1, 'VP': 1, 'NP@1-0': 1, 'VP@1-1': 1})
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
	If initialized with ``memoize=True``, equivalent subtrees will get the
	same identifiers."""
	def __init__(self, memoize=False):
		self.ids = self.n = 0
		self.packedgraphs = {}
		self.memoize = memoize

	def decorate(self, tree, sent):
		"""Return a copy of tree with labels decorated with IDs.

		>>> d = TreeDecorator()
		>>> tree = Tree.parse("(S (NP (DT 0) (N 1)) (VP 2))", parse_leaf=int)
		>>> d.decorate(tree, ['the', 'dog', 'walks'])
		... # doctest: +NORMALIZE_WHITESPACE
		Tree('S', [Tree('NP@1-0', [Tree('DT@1-1', [0]),
			Tree('N@1-2', [1])]), Tree('VP@1-3', [2])])
		>>> d = TreeDecorator(memoize=True)
		>>> print(d.decorate(Tree.parse("(S (NP (DT 0) (N 1)) (VP 2))",
		...		parse_leaf=int), ['the', 'dog', 'walks']))
		(S (NP@1-1 (DT@1-2 0) (N@1-3 1)) (VP@1-4 2))
		>>> print(d.decorate(Tree.parse("(S (NP (DT 0) (N 1)) (VP 2))",
		...		parse_leaf=int), ['the', 'dog', 'barks']))
		(S (NP@1-1 (DT@1-2 0) (N@1-3 1)) (VP@2-4 2))"""
		self.n += 1  # sentence number
		if self.memoize:
			self.ids = 0  # node number
			# wrap tree to get equality wrt sent
			tree = DiscTree(tree.freeze(), sent)
			dectree = ImmutableTree(tree.label, map(self._recdecorate, tree))
		else:
			dectree = Tree.convert(tree.copy(True))
			#skip top node, should not get an ID
			for m, a in enumerate(islice(dectree.subtrees(), 1, None)):
				a.label = "%s@%d-%d" % (a.label, self.n, m)
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


class DiscTree(ImmutableTree):
	"""Wrap an immutable tree with indices as leaves
	and a sentence. Provides hash & equality."""
	def __init__(self, tree, sent):
		self.sent = tuple(sent)
		super(DiscTree, self).__init__(tree.label, tuple(
				DiscTree(child, sent) if isinstance(child, Tree) else child
				for child in tree))

	def __eq__(self, other):
		return isinstance(other, Tree) and eqtree(self, self.sent,
				other, other.sent)

	def __hash__(self):
		return hash((self.label, ) + tuple(child.__hash__()
				if isinstance(child, Tree) else self.sent[child]
				for child in self))

	def __repr__(self):
		return "DisctTree(%r, %r)" % (
				super(DiscTree, self).__repr__(), self.sent)


def eqtree(tree1, sent1, tree2, sent2):
	"""Test whether two discontinuous trees are equivalent;
	assumes canonicalized() ordering."""
	if tree1.label != tree2.label or len(tree1) != len(tree2):
		return False
	for a, b in zip(tree1, tree2):
		istree = isinstance(a, Tree)
		if istree != isinstance(b, Tree):
			return False
		elif not istree:
			return sent1[a] == sent2[b]
		elif not a.__eq__(b):
			return False
	return True


def quotelabel(label):
	"""Escapes two things: parentheses and non-ascii characters.

	Parentheses are replaced by square brackets. Also escapes non-ascii
	characters, so that phrasal labels can remain ascii-only."""
	newlabel = label.replace('(', '[').replace(')', ']')
	# juggling to get str in both Python 2 and Python 3.
	return str(newlabel.encode('unicode-escape').decode('ascii'))


class UniqueIDs(object):
	"""Produce numeric IDs.

	Can be used as iterator (ID will not be re-used) and dictionary (ID will be
	re-used for same key).

	>>> ids = UniqueIDs()
	>>> next(ids)
	0
	>>> ids['foo'], ids['bar'], ids['foo']
	(1, 2, 1)"""
	def __init__(self):
		self.cnt = 0  # next available ID
		self.ids = {}  # IDs for labels seen

	def __getitem__(self, key):
		val = self.ids.get(key)
		if val is None:
			val = self.ids[key] = self.cnt
			self.cnt += 1
		return val

	def __next__(self):
		self.cnt += 1
		return self.cnt - 1

	def __iter__(self):
		return self

	next = __next__


def rangeheads(s):
	"""Iterate over a sequence of numbers and return first element of each
	contiguous range. Input should be shorted.

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
	"""A default parse, either right branching NPs, or all words under a single
	constituent 'NOPARSE'.

	>>> defaultparse([('like','X'), ('this','X'), ('example', 'NN'),
	... ('here','X')])
	'(NOPARSE (X like) (X this) (NN example) (X here))'
	>>> defaultparse([('like','X'), ('this','X'), ('example', 'NN'),
	... ('here','X')], True)
	'(NP (X like) (NP (X this) (NP (NN example) (NP (X here)))))'"""
	if rightbranching:
		if wordstags[1:]:
			return "(NP (%s %s) %s)" % (wordstags[0][1],
					wordstags[0][0], defaultparse(wordstags[1:], rightbranching))
		return "(NP (%s %s))" % wordstags[0][::-1]
	return "(NOPARSE %s)" % ' '.join("(%s %s)" % a[::-1] for a in wordstags)


def printrule(r, yf, w):
	""":returns: a string with a representation of a rule."""
	return "%s %s --> %s\t %r" % (w, r[0], ' '.join(x for x in r[1:]), list(yf))


def cartpi(seq):
	"""The cartesian product of a sequence of iterables.

	itertools.product doesn't support infinite sequences!

	>>> list(islice(cartpi([count(), count(0)]), 9))
	[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8)]"""
	if seq:
		return (b + (a, ) for b in cartpi(seq[:-1]) for a in seq[-1])
	return ((), )


def write_lcfrs_grammar(grammar, bitpar=False):
	"""Write a grammar in a simple text file format.

	Rules are written in the order as they appear in the sequence `grammar`,
	except that the lexicon file lists words in sorted order (with tags for
	each word in the order of `grammar`). For a description of the file format,
	see ``grammar.FORMAT``.

	:param grammar:  a sequence of rule tuples, as produced by
		``treebankgrammar()``, ``dopreduction()``, or ``doubledop()``.
	:param bitpar: when ``True``, use bitpar format: for rules, put weight
		first and leave out the yield function.
	:returns: tuple ``(rules, lexicon)``; bytes object & a unicode string,
		respectively.

	Weights are written in the following format:

	- if ``bitpar`` is ``False``, write rational fractions; e.g., ``2/3``.
	- if ``bitpar`` is ``True``, write frequencies (e.g., ``2``)
		if probabilities sum to 1, i.e., in that case probabilities can be
		re-computed as relative frequencies. Otherwise, resort to floating
		point numbers (e.g., ``0.666``, imprecise)."""
	rules, lexicon = [], []
	lexical = OrderedDict()
	freqs = False
	if bitpar:
		freqmass, denom = defaultdict(int), defaultdict(int)
		for (r, _), w in grammar:
			if isinstance(w, tuple):
				freqmass[r[0]] += w[0]
				denom[r[0]] = w[1]
			elif w != 1:
				break
		else:
			freqs = all(-1 <= denom[nt] - mass <= 1
					for nt, mass in freqmass.items())
	for (r, yf), w in grammar:
		if isinstance(w, tuple):
			if freqs:
				w = '%g' % w[0]
			else:
				w1, w2 = w
				if w1 == w2:
					w = '1'
				elif bitpar:
					w = '%g' % (w1 / w2)  # .hex()
				else:
					w = '%g/%d' % (w1, w2)
		elif isinstance(w, float):
			w = w.hex()
		if len(r) == 2 and r[1] == 'Epsilon':
			lexical.setdefault(unicode(yf[0]), []).append((r[0], w))
			continue
		elif bitpar:
			rules.append(('%s\t%s\n' % (w, '\t'.join(x for x in r))))
		else:
			yfstr = ','.join(''.join(map(str, a)) for a in yf)
			rules.append(('%s\t%s\t%s\n' % (
					'\t'.join(x for x in r), yfstr, w)))
	for word in lexical:
		lexicon.append(word)
		for tag, w in lexical[word]:
			lexicon.append(unicode('\t%s %s' % (tag, w)))
		lexicon.append(unicode('\n'))
	return ''.join(rules).encode('ascii'), u''.join(lexicon)


def write_lncky_grammar(rules, lexicon, out, encoding='utf-8'):
	"""Takes a bitpar grammar and converts it to the format of
	Mark Jonhson's cky parser."""
	grammar = []
	for a in io.open(rules, encoding=encoding):
		a = a.split()
		p, rule = a[0], a[1:]
		grammar.append('%s %s --> %s\n' % (p, rule[0], ' '.join(rule[1:])))
	for a in io.open(lexicon, encoding=encoding):
		a = a.split()
		word, tags = a[0], a[1:]
		tags = zip(tags[::2], tags[1::2])
		grammar.extend('%s %s --> %s\n' % (p, t, word) for t, p in tags)
	assert 'VROOT' in grammar[0]
	io.open(out, 'w', encoding=encoding).writelines(grammar)


def subsetgrammar(a, b):
	"""Test whether grammar a is a subset of b."""
	difference = set(map(itemgetter(0), a)) - set(map(itemgetter(0), b))
	if not difference:
		return True
	print("missing productions:")
	for r, yf in difference:
		print(printrule(r, yf, 0.0))
	return False


def grammarinfo(grammar, dump=None):
	"""Print some statistics on a grammar, before it goes through Grammar().

	:param dump: if given a filename, will dump distribution of parsing
		complexity to a file (i.e., p.c. 3 occurs 234 times, 4 occurs 120
		times, etc.)"""
	from discodop.eval import mean
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
	result += " average: %g\n" % mean([len(yf) for (_, yf), _, in grammar])
	n, r, yf, w = max((sum(map(len, yf)), rule, yf, w)
				for (rule, yf), w in grammar if rule[1] != 'Epsilon')
	result += "max variables: %d in %s\n" % (n, printrule(r, yf, w))

	def parsingcomplexity(yf):
		"""Sum the fanouts of LHS & RHS."""
		if isinstance(yf[0], tuple):
			return len(yf) + sum(map(len, yf))
		return 1  # NB: a lexical production has complexity 1

	pc = {(rule, yf, w): parsingcomplexity(yf)
							for (rule, yf), w in grammar}
	r, yf, w = max(pc, key=pc.get)
	result += "max parsing complexity: %d in %s" % (
			pc[r, yf, w], printrule(r, yf, w))
	result += " average %g" % mean(pc.values())
	if dump:
		pcdist = multiset(pc.values())
		open(dump, "w").writelines("%d\t%d\n" % x for x in pcdist.items())
	return result


def convertweight(weight):
	"""Convert a weight in a string to a float.

	>>> [convertweight(a) for a in ('0.5', '0x1.0000000000000p-1', '1/2')]
	[0.5, 0.5, (1.0, 2.0)]"""
	if '/' in weight:
		a, b = weight.split('/')
		return (float(a), float(b))
	elif weight.startswith('0x'):
		return float.fromhex(weight)
	return float(weight)


def test():
	"""Demonstrate grammar extraction."""
	from discodop import plcfrs
	from discodop.containers import Grammar
	from discodop.treebank import NegraCorpusReader
	from discodop.treetransforms import binarize, unbinarize, \
			addfanoutmarkers, removefanoutmarkers
	from discodop.disambiguation import recoverfragments
	from discodop.kbest import lazykbest
	logging.basicConfig(level=logging.DEBUG, format='%(message)s')
	filename = "alpinosample.export"
	corpus = NegraCorpusReader(filename, punct='move')
	sents = list(corpus.sents().values())
	trees = [addfanoutmarkers(binarize(a.copy(True), horzmarkov=1))
			for a in list(corpus.trees().values())[:10]]
	print('plcfrs\n', Grammar(treebankgrammar(trees, sents)))

	print('dop reduction')
	grammar = Grammar(dopreduction(trees[:2], sents[:2])[0],
			start=trees[0].label)
	print(grammar)
	grammar.testgrammar()

	debug = '--debug' in sys.argv
	grammarx, backtransform, _ = doubledop(trees, sents, debug=debug, numproc=1)
	print('\ndouble dop grammar')
	grammar = Grammar(grammarx, start=trees[0].label)
	grammar.getmapping(grammar, striplabelre=None,
			neverblockre=re.compile(b'^#[0-9]+|.+}<'),
			splitprune=False, markorigin=False)
	print(grammar)
	assert grammar.testgrammar(), "RFE should sum to 1."
	for tree, sent in zip(corpus.trees().values(), sents):
		print("sentence:", ' '.join(a.encode('unicode-escape').decode()
				for a in sent))
		chart, msg = plcfrs.parse(sent, grammar, exhaustive=True)
		print('\n', msg, '\ngold ', tree, '\n', 'double dop', end='')
		if chart:
			mpp = {}
			parsetrees = {}
			derivations, _ = lazykbest(chart, 1000, b'}<')
			for d, (t, p) in zip(chart.rankededges[chart.root()], derivations):
				r = Tree(recoverfragments(d.getkey(), chart, backtransform))
				r = str(removefanoutmarkers(unbinarize(r)))
				mpp[r] = mpp.get(r, 0.0) + exp(-p)
				parsetrees.setdefault(r, []).append((t, p))
			print(len(mpp), 'parsetrees', end='')
			print(sum(map(len, parsetrees.values())), 'derivations')
			for t, tp in sorted(mpp.items(), key=itemgetter(1)):
				print(tp, '\n', t, end='')
				print("match:", t == str(tree))
				assert len(set(parsetrees[t])) == len(parsetrees[t])
				if not debug:
					continue
				for deriv, p in sorted(parsetrees[t], key=itemgetter(1)):
					print(' <= %6g %s' % (exp(-p), deriv))
		else:
			print("no parse")
			print(chart)
		print()
	tree = Tree.parse("(ROOT (S (F (E (S (C (B (A 0))))))))", parse_leaf=int)
	Grammar(treebankgrammar([tree], [[str(a) for a in range(10)]]))


def main():
	"""Command line interface to create grammars from treebanks."""
	import gzip
	from getopt import gnu_getopt, GetoptError
	from discodop.treetransforms import addfanoutmarkers, canonicalize
	logging.basicConfig(level=logging.DEBUG, format='%(message)s')
	shortoptions = 's:'
	flags = ('gzip', 'packed', 'unbinarized')
	options = ('inputfmt=', 'inputenc=', 'dopestimator=', 'numproc=')
	try:
		opts, args = gnu_getopt(sys.argv[1:], shortoptions, flags + options)
		model, treebankfile, grammarfile = args
	except (GetoptError, ValueError) as err:
		print('error: %r\n%s' % (err, USAGE))
		sys.exit(2)
	opts = dict(opts)
	assert model in ('pcfg', 'plcfrs', 'dopreduction', 'doubledop', 'ptsg'), (
		'unrecognized model: %r' % model)
	assert opts.get('dopestimator', 'rfe') in ('rfe', 'ewe', 'shortest'), (
		'unrecognized estimator: %r' % opts['dopestimator'])

	if model == 'ptsg':  # read fragments
		splittedlines = (line.split('\t') for line in io.open(treebankfile,
					encoding=opts.get('--inputenc', 'utf8')))
		fragments = {(fields[0] if len(fields) == 2 else
				(fields[0], [a or None for a in fields[1].split(' ')])):
					convertweight(fields[-1]) for fields in splittedlines}
	else:  # read treebank
		corpus = READERS[opts.get('--inputfmt', 'export')](
				treebankfile,
				encoding=opts.get('--inputenc', 'utf8'))
		trees = list(corpus.trees().values())
		sents = list(corpus.sents().values())
		if not trees:
			raise ValueError('no trees; is --inputfmt correct?')
		for a in trees:
			canonicalize(a)
			addfanoutmarkers(a)

	# read off grammar
	if model in ('pcfg', 'plcfrs'):
		grammar = treebankgrammar(trees, sents)
	elif model == 'dopreduction':
		grammar, altweights = dopreduction(trees, sents,
				packedgraph='--packed' in opts)
	elif model == 'doubledop':
		grammar, backtransform, altweights = doubledop(trees, sents,
				numproc=int(opts.get('--numproc', 1)),
				binarized='--unbinarized' not in opts)
	elif model == 'ptsg':
		grammar, backtransform, altweights = compiletsg(fragments,
				binarized='--unbinarized' not in opts)
	if opts.get('--dopestimator', 'rfe') != 'rfe':
		grammar = [(rule, w) for (rule, _), w in
				zip(grammar, altweights[opts['--dopestimator']])]

	rulesname = grammarfile + '.rules'
	lexiconname = grammarfile + '.lex'
	myopen = open
	if '--gzip' in opts:
		myopen = gzip.open
		rulesname += '.gz'
		lexiconname += '.gz'
	bitpar = model == 'pcfg' or opts.get('--inputfmt') == 'bracket'
	if model == 'ptsg':
		bitpar = not isinstance(next(iter(fragments)), tuple)
	if '--unbinarized' not in opts:
		assert bitpar

	rules, lexicon = write_lcfrs_grammar(grammar, bitpar=bitpar)
	# write output
	with myopen(rulesname, 'w') as rulesfile:
		rulesfile.write(rules)
	with codecs.getwriter('utf-8')(myopen(lexiconname, 'w')) as lexiconfile:
		lexiconfile.write(lexicon)
	if model in ('doubledop', 'ptsg'):
		backtransformfile = '%s.backtransform%s' % (grammarfile,
			'.gz' if '--gzip' in opts else '')
		myopen(backtransformfile, 'w').writelines(
				'%s\n' % a for a in backtransform)
		print('wrote backtransform to', backtransformfile)
	print('wrote grammar to %s and %s.' % (rulesname, lexiconname))
	if len(grammar) < 10000:  # this is very slow so skip with large grammars
		print(grammarinfo(grammar))
	try:
		from discodop.containers import Grammar
		cgrammar = Grammar(rules, lexicon, bitpar=bitpar,
				start=opts.get('-s', next(iter(grammar))[0][0][0])
				if model == 'ptsg' else trees[0].label,
				binarized='--unbinarized' not in opts)
		cgrammar.testgrammar()
	except (ImportError, AssertionError) as err:
		print(err)

if __name__ == '__main__':
	if '--test' in sys.argv:
		test()
	else:
		main()
