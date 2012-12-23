""" Assorted functions to read grammars of treebanks. """
import codecs, logging, sys, re
from operator import mul, itemgetter
from math import log, exp
from collections import defaultdict, Counter as multiset
from itertools import chain, count, islice, imap, repeat
from tree import ImmutableTree, Tree
from containers import Grammar

usage = """Read off grammars from treebanks.
usage: %s [options] model input output

model is one of:
    pcfg
    plcfrs
    dopreduction
    doubledop
input is a binarized treebank,
output is the base for the filenames to write the grammar to.

options may consist of (* marks default option):
    --inputfmt [*export|discbracket|bracket]
    --inputenc [*UTF-8|ISO-8859-1|...]
    --dopestimator [dop1|ewe|shortest|...]
    --freqs               produce frequencies instead of probabilities
    --numproc [1|2|...]   only relevant for double dop fragment extraction
    --gzip                compress output with gzip, view with zless &c.
    --packed              use packed graph encoding for DOP reduction

When a PCFG is requested, or the input format is `bracket' (Penn format), the
output will be in bitpar format. Otherwise the grammar is written as an LCFRS.
The encoding of the input treebank may be specified. Output encoding will be
ASCII for the rules, and UTF-8 for the lexicon.

The LCFRS format is as follows. Rules are delimited by newlines.
Fields are separated by tabs. The fields are:

LHS	RHS1	[RHS2]	yield-function	weight

The yield function defines how the spans of the RHS nonterminals
are combined to form the spans of the LHS nonterminal. Components of the yield
function are comma-separated, 0 refers to a component of the first RHS
nonterminal, and 1 from the second. Weights are expressed as hexadecimal
negative logprobs. E.g.:

rules:   S    NP  VP  010 0x1.9c041f7ed8d33p+1
         VP_2    VB  NP  0,1 0x1.434b1382efeb8p+1
         NP      NN      0       0.3
lexicon: NN Epsilon Haus    0.3
""" % sys.argv[0]

frontierorterm = re.compile(r"(\(([^ ]+)( [0-9]+)(?: [0-9]+)*\))")
rootnode = re.compile(r"\([^ ]+\b")
fanoutre = re.compile("_([0-9]+)(?:@[-0-9]+)?$")

def lcfrs_productions(tree, sent, frontiers=False):
	""" Given a tree with integer indices as terminals, and a sentence
	with the corresponding words for these indices, produce a sequence
	of LCFRS productions. Always produces monotone LCFRS rules.
	For best results, tree should be canonicalized.
	When frontiers is true, frontier nodes will generate empty productions,
	by default they are ignored.

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
	rules = []
	for st in tree.subtrees():
		if not st:
			raise ValueError(("Empty node. Frontier nodes should designate "
				"which part(s) of the sentence they contribute to.\ntree:"
				"%s\nindices: %r\nsent: %r" % (tree.pprint(), leaves, sent)))
		#elif all(isinstance(a, int) for a in st):
		elif isinstance(st[0], int):
			if len(st) == 1 and sent[st[0]] is not None: # terminal node
				rule = ((st.node, 'Epsilon'), (sent[st[0]],))
			#elif all(sent[a] is None for a in st): # frontier node
			elif frontiers:
				rule = ((st.node, ), ())
			else:
				continue
			#else:
			#	raise ValueError(("Preterminals should dominate a single "
			#		"terminal; frontier nodes should dominate a sequence of "
			#		"indices that are None in the sentence.\n"
			#		"subtree: %s\nsent: %r" % (st, sent)))
		elif all(isinstance(a, Tree) for a in st): #isinstance(st[0], Tree):
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
			#	st.leaves(), tmpleaves, childleaves,
			#	st.pprint(margin=9999), sent))
		else:
			raise ValueError("Neither Tree node nor integer index:\n"
				"%r, %r" % (st[0], type(st[0])))
		rules.append(rule)
	return rules

def induce_plcfrs(trees, sents, freqs=False):
	""" Induce a probabilistic LCFRS, similar to how a PCFG is read off
	from a treebank """
	grammar = multiset(rule for tree, sent in zip(trees, sents)
			for rule in lcfrs_productions(tree, sent))
	if freqs:
		return list(grammar.items())
	lhsfd = multiset()
	for rule, freq in grammar.iteritems():
		lhsfd[rule[0][0]] += freq
	for rule, freq in grammar.iteritems():
		grammar[rule] = log(float(freq) / lhsfd[rule[0][0]])
	return list(grammar.items())

def dopreduction(trees, sents, ewe=False, shortestderiv=False,
		freqs=False, packedgraph=False):
	""" Induce a reduction of DOP to an LCFRS, similar to how Goodman (1996)
	reduces DOP1 to a PCFG.
		ewe: apply the equal weights estimate.
		freqs: return frequencies instead of probabilities.
	TODO: verify packed graph encoding (Bansal & Klein 2010). """
	global packed_graph_ids, packedgraphs
	# fd: how many subtrees are headed by node X (e.g. NP or NP@12),
	# 	counts of NP@... should sum to count of NP
	# ntfd: frequency of a node in corpus
	fd = defaultdict(float)
	ntfd = defaultdict(float)
	rules = defaultdict(float)
	if packedgraph:
		trees = [tree.freeze() for tree in trees]
		decoratefun = decorate_with_ids_mem
	else:
		decoratefun = decorate_with_ids

	# collect rules
	for n, t, sent in zip(count(), trees, sents):
		prods = lcfrs_productions(t, sent)
		ut = decoratefun(n, t, sent)
		uprods = lcfrs_productions(ut, sent)
		nodefreq(t, ut, fd, ntfd)
		for (a, avar), (b, bvar) in zip(prods, uprods):
			assert avar == bvar
			for c in cartpi([(x,) if x==y else (x, y) for x, y in zip(a, b)]):
				rules[c, avar] += 1

	if packedgraph:
		packedgraphs.clear()

	# define probabilities
	def rfe(rule):
		""" relative frequency estimate, aka DOP1 (Bod 1992; Goodman 1996) """
		(r, yf), freq = rule
		return (r, yf), ((1 if any('@' in z for z in r) else freq) *
			reduce(mul, (fd[z] for z in r[1:] if '@' in z), 1)
			/ (1 if freqs else fd[r[0]]))

	def bodewe(rule):
		""" Bod (2003, figure 3) """
		(r, yf), freq = rule
		return (r, yf), ((1 if '@' in r[0] else freq) *
			reduce(mul, (fd[z] for z in r[1:] if '@' in z), 1)
			/ ((1 if freqs else fd[r[0]])
				* (ntfd[r[0]] if '@' not in r[0] else 1.)))

	#@memoize
	#def sumfracs(nom, denoms):
	#	return sum(nom / denom for denom in denoms)
	#
	# map of exterior (unaddressed) nodes to normalized denominators:
	#ewedenoms = {}
	#def goodmanewe(rule):
	#	""" Goodman (2003, eq. 1.5). Probably a typographic mistake. """
	#	(r, yf), freq = rule
	#	if '@' in r[0]:
	#		return rfe(((r, yf), freq))
	#	nom = reduce(mul, (fd[z] for z in r[1:] if '@' in z), 1)
	#	return (r, yf), sumfracs(nom, ewedenoms[r[0]])
	#
	#if ewe: #goodmanewe
	#	for aj in fd:
	#		a = aj.split("@")[0]
	#		if a == aj:
	#			continue	# exterior / unaddressed node
	#		ewedenoms.setdefault(a, []).append(fd[aj] * ntfd[a])
	#	for a in ewedenoms:
	#		ewedenoms[a] = tuple(ewedenoms[a])
	#	probmodel = [(rule, log(p))
	#		for rule, p in imap(goodmanewe, rules.iteritems())]
	if ewe: #bodewe
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

def doubledop(fragments, debug=False, ewe=False, freqs=False):
	""" Extract a Double-DOP grammar from a treebank. That is, a fragment
	grammar containing all fragments that occur at least twice, plus all
	individual productions needed to obtain full coverage.
	Input trees need to be binarized. A second level of binarization (a normal
	form) is needed when fragments are converted to individual grammar rules,
	which occurs through the removal of internal nodes. When the remaining
	nodes do not uniquely identify the fragment, an extra node with an
	identifier is inserted: #n where n is an integer. In fragments with
	terminals, we replace their POS tags with a tag uniquely identifying that
	terminal and tag: tag@word.
	When ewe is true, the equal weights estimate is applied. This requires that
	the fragments are accompanied by indices instead of frequencies. """
	grammar = {}
	backtransform = {}
	ntfd = defaultdict(float)
	ids = count()
	if ewe:
		# build an index to get the number of fragments extracted from a tree
		fragmentcount = defaultdict(float)
		for indices in fragments.itervalues():
			for index, cnt in indices.iteritems():
				fragmentcount[index] += cnt

	# binarize, turn to lcfrs productions
	# use artificial markers of binarization as disambiguation,
	# construct a mapping of productions to fragments
	for frag, terminals in fragments:
		prods, newfrag = flatten(frag, terminals, ids)
		if prods[0][0][1] == 'Epsilon': #lexical production
			lexprod = prods[0]
			if ewe:
				# Sangati & Zuidema (2011, eq. 5)
				grammar[lexprod] = sum(v / fragmentcount[k]
						for k, v in fragments[frag, terminals].iteritems())
			else:
				grammar[lexprod] = fragments[frag, terminals]
			continue
		elif prods[0] in backtransform:
			# normally, rules of fragments are disambiguated by binarization IDs
			# in case there's a fragment with only one or two frontier nodes,
			# we add an artficial node.
			newlabel = "%s}<%d>%s" % (prods[0][0][0], ids.next(),
					'' if len(prods[0][1]) == 1 else '_%d' % len(prods[0][1]))
			prod1 = ((prods[0][0][0], newlabel) + prods[0][0][2:], prods[0][1])
			# we have to determine fanout of the first nonterminal
			# on the right hand side
			prod2 = ((newlabel, prods[0][0][1]),
				tuple((0,) for component in prods[0][1]
				for a in component if a == 0))
			prods[:1] = [prod1, prod2]
		# first binarized production gets prob. mass
		if ewe:
			# Sangati & Zuidema (2011, eq. 5)
			grammar[prods[0]] = sum(v / fragmentcount[k]
					for k, v in fragments[frag, terminals].iteritems())
		else:
			grammar[prods[0]] = fragments[frag, terminals]
		grammar.update(zip(prods[1:], repeat(1)))
		# & becomes key in backtransform
		backtransform[prods[0]] = newfrag
	if debug:
		ids = count()
		flatfrags = [flatten(frag, terminals, ids)
				for frag, terminals in fragments]
		doubledopdump(flatfrags, fragments, {}, backtransform)
	#sort grammar such that we have these clusters:
	# 1. non-binarized rules or initial rules of a binarized constituent
	# 2: non-initial binarized rules.
	# 3: lexical productions
	# this is so that the backtransform aligns with the first part of the rules
	grammar = sorted(grammar.items(), key=lambda (rule, _): (
				rule[0][1] == 'Epsilon',
				"}<" in rule[0][0],
				rule))
	# replace keys with numeric ids of rules, drop terminals.
	backtransform = {n: backtransform[r]
		for n, (r, _) in enumerate(grammar) if r in backtransform}
	if freqs:
		return grammar, backtransform
	# relative frequences as probabilities
	for rule, freq in grammar:
		ntfd[rule[0][0]] += freq
	grammar = [(rule, log(freq / ntfd[rule[0][0]])) for rule, freq in grammar]
	return grammar, backtransform

def doubledopdump(flatfrags, fragments, newprods, backtransform):
	""" Print some diagnostic information on a Double-DOP grammar. """
	print "recurring fragments:"
	for a, b in zip(flatfrags, fragments):
		if isinstance(a, tuple):
			print "fragment: %s\nprod:     %s" % (b[0], "\n\t".join(
				printrule(r, yf, 0) for r, yf in a[0]))
			print "template: %s\nfreq: %2d  sent: %s\n" % (
					a[1], fragments[b], " ".join('_' if x is None
					else quotelabel(x) for x in b[1]))
		else:
			print "fragment: %s\nprod:     %s\nfreq: %2d  sent: %s\n" % (
					b[0], a, fragments[b],
					" ".join('_' if x is None else quotelabel(x) for x in b[1]))
	print "ambiguous fragments:"
	if len(newprods) == 0:
		print "None"
	for a, b in newprods.iteritems():
		b = b[1]
		print "prod: %s\nsent: %s" % (a,
				" ".join('_' if x is None else quotelabel(x) for x in b))
		if backtransform.get(a,''):
			print "frag:", backtransform[a]
		print
	print "backtransform:"
	for a, b in backtransform.items():
		if not isinstance(b, tuple):
			print b
		elif b:
			print a, ":\n\t", b[0], " ".join(
				'_' if x is None else quotelabel(x) for x in b[1])

def coarse_grammar(trees, sents, level=0):
	""" collapse all labels to X except ROOT and POS tags. """
	if level == 0:
		repl = lambda x: "X"
	label = re.compile("[^^|<>-]+")
	for tree in trees:
		for subtree in tree.subtrees():
			if subtree.node != "ROOT" and isinstance(subtree[0], Tree):
				subtree.node = label.sub(repl, subtree.node)
	return induce_plcfrs(trees, sents)

def nodefreq(tree, utree, subtreefd, nonterminalfd):
	""" Auxiliary function for DOP reduction.
	Counts frequencies of nodes and calculate the number of
	subtrees headed by each node. updates "subtreefd" and "nonterminalfd"
	as a side effect. Expects a normal tree and a tree with IDs.
		@param subtreefd: the multiset to store the counts of subtrees
		@param nonterminalfd: the multiset to store the counts of non-terminals

	>>> fd = multiset()
	>>> tree = Tree("(S (NP mary) (VP walks))")
	>>> utree = decorate_with_ids(1, tree, ['mary', 'walks'])
	>>> nodefreq(tree, utree, fd, multiset())
	4
	>>> fd
	Counter({'S': 4, 'NP': 1, 'VP': 1, 'NP@1-0': 1, 'VP@1-1': 1})
	"""
	nonterminalfd[tree.node] += 1.0
	nonterminalfd[utree.node] += 1.0
	if isinstance(tree[0], Tree):
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

def decorate_with_ids(n, tree, sent):
	""" Auxiliary function for DOP reduction.
	Adds unique identifiers to each internal non-terminal of a tree.
	n should be an identifier of the sentence.

	>>> tree = Tree("(S (NP (DT the) (N dog)) (VP walks))")
	>>> decorate_with_ids(1, tree, ['the', 'dog', 'walks'])
	Tree('S', [Tree('NP@1-0', [Tree('DT@1-1', ['the']),
			Tree('N@1-2', ['dog'])]), Tree('VP@1-3', ['walks'])])
	"""
	utree = Tree.convert(tree.copy(True))
	ids = 0
	#skip top node, should not get an ID
	for a in islice(utree.subtrees(), 1, None):
		a.node = "%s@%d-%d" % (a.node, n, ids)
		ids += 1
	return utree

def eqtree(tree1, sent1, tree2, sent2):
	""" Test whether two discontinuous trees are equivalent;
	assumes canonicalized() ordering. """
	if tree1.node != tree2.node or len(tree1) != len(tree2):
		return False
	for a, b in zip(tree1, tree2):
		istree = isinstance(a, Tree)
		if istree != isinstance(b, Tree):
			return False
		elif istree:
			if not a.__eq__(b):
				return False
		else:
			return sent1[a] == sent2[b]
	return True

class DiscTree(ImmutableTree):
	""" Wrap an immutable tree with indices as leaves
	and a sentence. """
	def __init__(self, tree, sent):
		super(DiscTree, self).__init__(tree.node,
				tuple(DiscTree(a, sent) if isinstance(a, Tree) else a
				for a in tree))
		self.sent = sent
	def __eq__(self, other):
		return isinstance(other, Tree) and eqtree(self, self.sent,
				other, other.sent)
	def __hash__(self):
		return hash((self.node, ) + tuple(a.__hash__()
				if isinstance(a, Tree) else self.sent[a] for a in self))
	def __repr__(self):
		return "DisctTree(%r, %r)" % (
				super(DiscTree, self).__repr__(), self.sent)

packed_graph_ids = 0
packedgraphs = {}
def decorate_with_ids_mem(n, tree, sent):
	""" Auxiliary function for DOP reduction.
	Adds unique identifiers to each internal non-terminal of a tree.
	This version does memoization, which means that equivalent subtrees
	(including the yield) will get the same IDs. Experimental. """
	def recursive_decorate(tree):
		global packed_graph_ids
		if isinstance(tree, int):
			return tree
		# this is wrong, should take sent into account.
		# use (tree, sent) as key,
		# but translate indices to start at 0, gaps to have length 1.
		elif tree not in packedgraphs:
			packed_graph_ids += 1
			packedgraphs[tree] = ImmutableTree("%s@%d-%d" %
				(tree.node, n, packed_graph_ids),
				map(recursive_decorate, tree))
			return packedgraphs[tree]
		else:
			return copyexceptindices(tree, packedgraphs[tree])
	def copyexceptindices(tree1, tree2):
		""" """
		if not isinstance(tree1, Tree):
			return tree1
		return ImmutableTree(tree2.node,
			[copyexceptindices(a, b) for a, b in zip(tree1, tree2)])
	global packed_graph_ids
	packed_graph_ids = 0
	# wrap tree to get equality wrt sent
	tree = DiscTree(tree.freeze(), sent)
	#skip top node, should not get an ID
	return ImmutableTree(tree.node, map(recursive_decorate, tree))

def quotelabel(label):
	""" Escapes two things: parentheses and non-ascii characters.
	Parentheses are replaced by square brackets. Also escapes non-ascii
	characters, so that phrasal labels can remain ascii-only. """
	return label.replace('(', '[').replace(')', ']').encode('unicode-escape')

def flatten(tree, sent, ids):
	""" Auxiliary function for Double-DOP.
	Remove internal nodes from a tree and read off its binarized
	productions. Aside from returning productions, also return tree with
	lexical and frontier nodes replaced by a templating symbol '%s'.
	Input is a tree and sentence, as well as an iterator which yields
	unique IDs for non-terminals introdudced by the binarization;
	output is a tuple (prods, frag). Trees are in the form of strings.

	>>> ids = count()
	>>> sent = [None, ',', None, '.']
	>>> tree = "(ROOT (S_2 0 2) (ROOT|<$,>_2 ($, 1) ($. 3)))"
	>>> print flatten(tree, sent, ids)
	([(('ROOT', 'ROOT}<0>', '$.@.'), ((0, 1),)),
	(('ROOT}<0>', 'S_2', '$,@,'), ((0, 1, 0),)),
	(('$,@,', 'Epsilon'), (',',)), (('$.@.', 'Epsilon'), ('.',))],
	'(ROOT {0} (ROOT|<$,>_2 {1} {2}))')
	>>> print flatten("(NN 0)", ["foo"], ids)
	([(('NN', 'Epsilon'), ('foo',))], '(NN 0)')
	>>> flatten(r"(S (S|<VP> (S|<NP> (NP (ART 0) (CNP (CNP|<TRUNC> "
	... "(TRUNC 1) (CNP|<KON> (KON 2) (CNP|<NN> (NN 3)))))) (S|<VAFIN> "
	... "(VAFIN 4))) (VP (VP|<ADV> (ADV 5) (VP|<NP> (NP (ART 6) (NN 7)) "
	... "(VP|<NP> (NP_2 8 10) (VP|<VVPP> (VVPP 9))))))))",
	... (u'Das', u'Garten-', u'und', u'Friedhofsamt', u'hatte', u'kuerzlich',
	... u'dem', u'Ortsbeirat', None, None, None), ids)
	([(('S', 'S}<8>_2', 'VVPP'), ((0, 1, 0),)),
	(('S}<8>_2', 'S}<7>', 'NP_2'), ((0, 1), (1,))),
	(('S}<7>', 'S}<6>', 'NN@Ortsbeirat'), ((0, 1),)),
	(('S}<6>', 'S}<5>', 'ART@dem'), ((0, 1),)),
	(('S}<5>', 'S}<4>', 'ADV@kuerzlich'), ((0, 1),)),
	(('S}<4>', 'S}<3>', 'VAFIN@hatte'), ((0, 1),)),
	(('S}<3>', 'S}<2>', 'NN@Friedhofsamt'), ((0, 1),)),
	(('S}<2>', 'S}<1>', 'KON@und'), ((0, 1),)), (('S}<1>', 'ART@Das', \
	'TRUNC@Garten-'), ((0, 1),)),
	(('ART@Das', 'Epsilon'), (u'Das',)),
	(('TRUNC@Garten-', 'Epsilon'), (u'Garten-',)),
	(('KON@und', 'Epsilon'), (u'und',)),
	(('NN@Friedhofsamt', 'Epsilon'), (u'Friedhofsamt',)),
	(('VAFIN@hatte', 'Epsilon'), (u'hatte',)),
	(('ADV@kuerzlich', 'Epsilon'), (u'kuerzlich',)),
	(('ART@dem', 'Epsilon'), (u'dem',)),
	(('NN@Ortsbeirat', 'Epsilon'), (u'Ortsbeirat',))],
	'(S (S|<VP> (S|<NP> (NP {0} (CNP (CNP|<TRUNC> {1} (CNP|<KON> {2} \
	(CNP|<NN> {3}))))) (S|<VAFIN> {4})) (VP (VP|<ADV> {5} (VP|<NP> \
	(NP {6} {7}) (VP|<NP> {8} (VP|<VVPP> {9})))))))')
	>>> flatten("(S|<VP>_2 (VP_3 (VP|<NP>_3 (NP 0) (VP|<ADV>_2 (ADV 2) "
	... "(VP|<VVPP> (VVPP 4))))) (S|<VAFIN> (VAFIN 1)))",
	... (None, None, None, None, None), ids)
	([(('S|<VP>_2', 'S|<VP>_2}<10>', 'VVPP'), ((0,), (1,))),
	(('S|<VP>_2}<10>', 'S|<VP>_2}<9>', 'ADV'), ((0, 1),)),
	(('S|<VP>_2}<9>', 'NP', 'VAFIN'), ((0, 1),))],
	'(S|<VP>_2 (VP_3 (VP|<NP>_3 {0} (VP|<ADV>_2 {2} (VP|<VVPP> {3})))) \
	(S|<VAFIN> {1}))') """
	from treetransforms import defaultleftbin, addbitsets
	assert isinstance(tree, basestring), (tree, sent)
	def repl(x):
		n = x.group(3) # index w/leading space
		nn = int(n)
		if sent[nn] is None:
			return x.group(0)	# (tag indices)
		# (tag@word idx)
		return "(%s@%s%s)" % (x.group(2), quotelabel(sent[nn]), n)
	if tree.count(" ") == 1:
		return lcfrs_productions(addbitsets(tree), sent), tree
	# give terminals unique POS tags
	prod = frontierorterm.sub(repl, tree)
	# remove internal nodes, reorder
	prod = "%s %s)" % (prod[:prod.index(" ")],
		" ".join(x[0] for x in sorted(frontierorterm.findall(prod),
		key=lambda x: int(x[2]))))
	prods = lcfrs_productions(defaultleftbin(addbitsets(prod), "}",
		markfanout=True, ids=ids, threshold=2), sent)
	# remember original order of frontiers / terminals for template
	order = {x[2]: "{%d}" % n
			for n, x in enumerate(frontierorterm.findall(prod))}
	# mark substitution sites and ensure ascii string.
	newtree = str(frontierorterm.sub(lambda x: order[x.group(3)], tree))
	return prods, newtree

def rangeheads(s):
	""" Iterate over a sequence of numbers and return first element of each
	contiguous range. Input should be shorted.

	>>> rangeheads( (0, 1, 3, 4, 6) )
	[0, 3, 6]
	"""
	sset = set(s)
	return [a for a in s if a - 1 not in sset]

def ranges(s):
	""" Partition s into a sequence of lists corresponding to contiguous ranges

	>>> list(ranges( (0, 1, 3, 4, 6) ))
	[[0, 1], [3, 4], [6]]"""
	rng = []
	for a in s:
		if not rng or a == rng[-1]+1:
			rng.append(a)
		else:
			yield rng
			rng = [a]
	if rng:
		yield rng

def printrule(r, yf, w):
	""" Return a string with a representation of a rule. """
	return "%.2f %s --> %s\t %r" % (exp(w), r[0], "  ".join(r[1:]), list(yf))

def printrulelatex(rule, doexp=True):
	r""" Return a string with a representation of a rule in latex format.

	>>> r = ((('VP_2@1', 'NP@2', 'VVINF@5'), ((0,), (1,))), -0.916290731874155)
	>>> printrulelatex(r)
	0.4 &  $ \textrm{VP\_2@1}(x_{0},x_{1}) \rightarrow \textrm{NP@2}(x_{0}) \: \textrm{VVINF@5}(x_{1})  $ \\
	"""
	(r, yf), w = rule
	c = count()
	newrhs = []
	variables = []
	if r[1] == "Epsilon":
		newrhs = [("Epsilon", [])]
		lhs = (r[0], yf)
	else:
		# NB: not working correctly ... variables get mixed up
		for n, a in enumerate(r[1:]):
			z = sum(1 for comp in yf for y in comp if y == n)
			newrhs.append((a, [c.next() for x in range(z)]))
			variables.append(list(newrhs[-1][1]))
		lhs = (r[0], [[variables[x].pop(0) for x in comp] for comp in yf])
	print (exp(w) if doexp else w), "& ",
	r = tuple([lhs]+newrhs)
	lhs = r[0]
	rhs = r[1:]
	print "$",
	if isinstance(lhs[1][0], tuple) or isinstance(lhs[1][0], list):
		lhs = r[0]
		rhs = r[1:]
	if not isinstance(lhs[1][0], tuple) and not isinstance(lhs[1][0], list):
		print r"\textrm{%s}(\textrm{%s})" % (
			lhs[0].replace("$", r"\$"), lhs[1][0]),
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
			if x != rhs[-1]:
				print '\\:',
	print r' $ \\'

def freeze(l):
	""" Recursively walk through a structure converting lists to tuples.

	>>> freeze([[0], [1]])
	((0,), (1,)) """
	return tuple(map(freeze, l)) if isinstance(l, (list, tuple)) else l

def unfreeze(l):
	""" Recursively walk through a structure converting tuples to lists.

	>>> unfreeze(((0,), (1,)))
	[[0], [1]] """
	return list(map(unfreeze, l)) if isinstance(l, (list, tuple)) else l

def cartpi(seq):
	""" itertools.product doesn't support infinite sequences!

	>>> list(islice(cartpi([count(), count(0)]), 9))
	[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8)] """
	if seq:
		return (b + (a,) for b in cartpi(seq[:-1]) for a in seq[-1])
	return ((), )

def bfcartpi(seq):
	""" breadth-first (diagonal) cartesian product

	>>> list(islice(bfcartpi([count(), count(0)]), 9))
	[(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]"""
	#wrap items of seq in generators
	seqit = [(x for x in a) for a in seq]
	#fetch initial values
	try:
		seqlist = [[a.next()] for a in seqit]
	except StopIteration:
		return
	yield tuple(a[0] for a in seqlist)
	#bookkeeping of which iterators still have values
	stopped = len(seqit) * [False]
	n = len(seqit)
	while not all(stopped):
		if n == 0:
			n = len(seqit) - 1
		else:
			n -= 1
		if stopped[n]:
			continue
		try:
			seqlist[n].append(seqit[n].next())
		except StopIteration:
			stopped[n] = True
			continue
		for result in cartpi(seqlist[:n] + [seqlist[n][-1:]] + seqlist[n+1:]):
			yield result

def read_rparse_grammar(filename):
	""" Read a grammar in the format as produced by rparse. """
	result = []
	for line in open(filename):
		yf = eval(line[line.index("[[["):].replace("false","0").replace(
			"true","1"))[0]
		line = line[:line.index("[[[")].split()
		line.pop(0) #freq?
		prob, lhs = line.pop(0).split(":")
		line.pop(0) # -->
		result.append(((tuple([lhs] + line), tuple(map(tuple, yf))),
			log(float(prob))))
	return result

def exportrparsegrammar(grammar):
	""" Export a grammar to rparse format. All frequencies are 1,
	but probabilities are exported.  """
	def repryf(yf):
		return "[" + ", ".join("[" + ", ".join("true" if a == 1 else "false"
			for a in b) + "]" for b in yf) + "]"
	def rewritelabel(a):
		a = a.replace("ROOT", "VROOT")
		if "|" in a:
			fanout = a.rsplit("_", 1)[-1] if "_" in a[a.rindex(">"):] else "1"
			parent = (a[a.index("^")+2:a.index(">", a.index("^"))]
					if "^" in a else "")
			parent = "^"+"-".join(x.replace("_","") if "_" in x else x+"1"
					for x in parent.split("-"))
			children = a[a.index("<")+1:a.index(">")].split("-")
			children = "-".join(x.replace("_","") if "_" in x else x+"1"
					for x in children)
			current = a.split("|")[0]
			current = ("".join(current.split("_")) if "_" in current
					else current + "1")
			return "@^%s%s-%sX%s" % (current, parent, children, fanout)
		return "".join(a.split("_")) if "_" in a else a+"1"
	for (r, yf), w in grammar:
		if r[1] != 'Epsilon':
			yield ("1 %s:%s --> %s [%s]" % (repr(exp(w)), rewritelabel(r[0]),
				" ".join(map(rewritelabel, r[1:])), repryf(yf)))

def read_bitpar_grammar(rules, lexicon, dop=False, ewe=False):
	""" Read a bitpar grammar given two file objects. Must be a binarized grammar.
	Note that the VROOT symbol will be read as `ROOT' instead. Frequencies will
	be converted to exact relative frequencies (unless ewe is specified)."""
	grammar = []
	ntfd = defaultdict(float)
	ntfd1 = defaultdict(set)
	for a in rules:
		a = a.split()
		p, rule = float(a[0]), a[1:]
		if rule[0] == "VROOT":
			rule[0] = "ROOT"
		ntfd[rule[0]] += p
		if dop:
			ntfd1[rule[0].split("@")[0]].add(rule[0])
		if len(rule) == 2:
			grammar.append(((tuple(rule), ((0,),)), p))
		elif len(rule) == 3:
			grammar.append(((tuple(rule), ((0, 1),)), p))
		else:
			raise ValueError
		#grammar.append(([(rule[0], [range(len(rule) - 1)])]
		#			+ [(a, [n]) for n, a in enumerate(rule[1:])], p))
	for a in lexicon:
		a = a.split()
		word, tags = a[0], a[1:]
		tags = zip(tags[::2], map(float, tags[1::2]))
		for t, p in tags:
			ntfd[t] += p
			if dop:
				ntfd1[t.split("@")[0]].add(t)
		grammar.extend((((t, 'Epsilon'), (word,)), p) for t, p in tags)
	if ewe:
		return Grammar([(rule,
			log(p / (ntfd[rule[0][0]] * len(ntfd1[rule[0][0].split("@")[0]]))))
			for rule, p in grammar])
	return [(rule, log(p / ntfd[rule[0][0]])) for rule, p in grammar]

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

def write_lcfrs_grammar(grammar, rules, lexicon, bitpar=False):
	""" Writes a grammar as produced by induce_plcfrs() or dopreduction()
	(so before it goes through Grammar()) into a simple text file format.
	Expects file objects with write() methods. Fields are separated by tabs.
	Components of the yield function are comma-separated; e.g.:
	rules: S	NP	VP	010	0.5
		VP_2	VB	NP	0,1	0.4
	lexicon: NN	Epsilon	Haus	0.3
	When bitpar is True, use bitpar format: for rules, put weight first and
	leave out the yield function. """
	for (r, yf), w in grammar:
		if len(r) == 2 and r[1] == "Epsilon":
			lexicon.write("%s\t%s\t%g\n" % ("\t".join(r), yf[0], float(w)))
		elif bitpar:
			rules.write("%g\t%s\n" % (w, "\t".join(r)))
		else:
			yfstr = ",".join("".join(map(str, a)) for a in yf)
			rules.write("%s\t%s\t%g\n" % ("\t".join(r), yfstr, float(w)))

def read_lcfrs_grammar(rules, lexicon, encoding='utf-8'):
	""" Reads a grammar as produced by write_lcfrs_grammar from two file
	objects. """
	rules = (a.strip().split('\t') for a in rules)
	grammar = [((tuple(a[:-2]), tuple(tuple(map(int, b))
			for b in a[-2].split(","))), float.fromhex(a[-1])) for a in rules]
	lexicon = (a.strip().split('\t') for a in lexicon)
	grammar += [(((t, 'Epsilon'), (w,)), float.fromhex(p))
			for t, w, p in lexicon]
	return grammar

def alterbinarization(tree):
	"""converts the binarization of rparse to the format that NLTK expects
	S1 is the constituent, CS1 the parent, CARD1 the current sibling/child
	@^S1^CS1-CARD1X1   -->  S1|<CARD1>^CS1 """
	#how to optionally add \2 if nonempty?
	tree = re.sub(
		"@\^([A-Z.,()$]+)\d+(\^[A-Z.,()$]+\d+)*(?:-([A-Z.,()$]+)\d+)*X\d+",
		r"\1|<\3>", tree)
	# remove fanout markers
	tree = re.sub(r"([A-Z.,()$]+)\d+", r"\1", tree)
	tree = re.sub("VROOT", r"ROOT", tree)
	assert "@" not in tree
	return tree

def subsetgrammar(a, b):
	""" test whether grammar a is a subset of b. """
	difference = set(imap(itemgetter(0), a)) - set(imap(itemgetter(0), b))
	if not difference:
		return True
	print "missing productions:"
	for r, yf in difference:
		print printrule(r, yf, 0.0)
	return False

def mean(seq):
	""" Return arithmetic mean. """
	return sum(seq) / float(len(seq)) if seq else None #"zerodiv"

def grammarinfo(grammar, dump=None):
	""" print some statistics on a grammar, before it goes through Grammar().
	dump: if given a filename, will dump distribution of parsing complexity
	to a file (i.e., p.c. 3 occurs 234 times, 4 occurs 120 times, etc. """
	lhs = set(rule[0] for (rule, yf), w in grammar)
	l = len(grammar)
	result = "labels: %d" % len(set(rule[a] for (rule, yf), w in grammar
							for a in range(3) if len(rule) > a))
	result += " of which preterminals: %d\n" % (
		len(set(rule[0] for (rule, yf), w in grammar if rule[1] == "Epsilon"))
		or len(set(rule[a] for (rule, yf), w in grammar
				for a in range(1,3) if len(rule) > a and rule[a] not in lhs)))
	ll = sum(1 for (rule, yf), w in grammar if rule[1] == "Epsilon")
	result += "clauses: %d  lexical clauses: %d" % (l, ll)
	result += " non-lexical clauses: %d\n" % (l - ll)
	n, r, yf, w = max((len(yf), rule, yf, w) for (rule, yf), w in grammar)
	result += "max fan-out: %d in " % n
	result += printrule(r, yf, w)
	result += " average: %g\n" % mean([len(yf) for (_, yf), _, in grammar])
	n, r, yf, w = max((sum(map(len, yf)), rule, yf, w)
				for (rule, yf), w in grammar if rule[1] != "Epsilon")
	result += "max variables: %d in %s\n" % (n, printrule(r, yf, w))
	def parsingcomplexity(yf):
		""" this sums the fanouts of LHS & RHS """
		if isinstance(yf[0], basestring):
			return 1 #NB: a lexical production has complexity 1
		else:
			return len(yf) + sum(map(len, yf))
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

def do(sent, grammar):
	from parser import parse, pprint_chart
	from disambiguation import marginalize
	print "sentence", sent
	p, start, _ = parse(sent, grammar, start=grammar.toid['S'])
	if start:
		mpp, _ = marginalize("mpp", p, start, grammar, 10)
		for t in mpp:
			print exp(mpp[t]), t
	else:
		print "no parse"
		pprint_chart(p, sent, grammar.tolabel)
	print

def test():
	from treetransforms import unbinarize, removefanoutmarkers
	from treebank import NegraCorpusReader
	from parser import parse, pprint_chart
	from treetransforms import addfanoutmarkers
	from disambiguation import recoverfragments
	from kbest import lazykbest
	from agenda import getkey
	from fragments import getfragments
	logging.basicConfig(level=logging.DEBUG, format='%(message)s')
	# this fixes utf-8 output when piped through e.g. less
	# won't help if the locale is not actually utf-8, of course
	#sys.stdout = codecs.getwriter('utf8')(sys.stdout)
	#filename = "negraproc.export"
	filename = "sample2.export"
	corpus = NegraCorpusReader(".", filename, encoding="iso-8859-1",
		headrules="negra.headrules", headfinal=True, headreverse=False,
		movepunct=True)
	#corpus = BracketCorpusReader(".", "treebankExample.mrg")
	sents = corpus.sents().values()
	trees = [a.copy(True) for a in corpus.parsed_sents().values()[:10]]
	for a in trees:
		a.chomsky_normal_form(horzMarkov=1)
		addfanoutmarkers(a)

	print 'plcfrs'
	lcfrs = Grammar(induce_plcfrs(trees, sents))
	print lcfrs

	print 'dop reduction'
	grammar = Grammar(dopreduction(trees[:2], sents[:2]))
	print grammar
	grammar.testgrammar(logging)

	fragments = getfragments(trees, sents, 1)
	debug = '--debug' in sys.argv
	grammarx, backtransform = doubledop(fragments, debug=debug)
	print '\ndouble dop grammar'
	grammar = Grammar(grammarx)
	grammar.getmapping(grammar, striplabelre=None,
		neverblockre=re.compile(r'^#[0-9]+|.+}<'),
		splitprune=False, markorigin=False)
	print unicode(grammar)
	assert grammar.testgrammar(logging) #DOP1 should sum to 1.
	for tree, sent in zip(corpus.parsed_sents().values(), sents):
		print "sentence:", " ".join(sent)
		root = tree.node
		chart, start, msg = parse(sent, grammar, start=grammar.toid[root],
			exhaustive=True)
		print '\n', msg,
		print "\ngold ", tree.pprint(margin=9999)
		print "double dop",
		if start:
			mpp = {}
			parsetrees = {}
			derivations, D = lazykbest(chart, start, 1000,
				grammar.tolabel, "}<")
			for d, (t, p) in zip(D[start], derivations):
				r = Tree(recoverfragments(getkey(d), D,
					grammar, backtransform))
				unbinarize(r)
				r = removefanoutmarkers(r).pprint(margin=9999)
				mpp[r] = mpp.get(r, 0.0) + exp(-p)
				parsetrees.setdefault(r, []).append((t, p))
			print len(mpp), 'parsetrees',
			print sum(map(len, parsetrees.values())), 'derivations'
			for t, tp in sorted(mpp.items(), key=itemgetter(1)):
				print tp, '\n', t,
				print "match:", t == tree.pprint(margin=9999)
				assert len(set(parsetrees[t])) == len(parsetrees[t])
				if not debug:
					continue
				for deriv, p in sorted(parsetrees[t], key=itemgetter(1)):
					print ' <= %6g %s' % (exp(-p), deriv)
		else:
			print "no parse"
			pprint_chart(chart, sent, grammar.tolabel)
		print
	tree = Tree.parse("(ROOT (S (F (E (S (C (B (A 0))))))))", parse_leaf=int)
	g = Grammar(induce_plcfrs([tree], [range(10)]))
	#print "tree: %s\nunary closure:" % tree
	#g.getunaryclosure()
	#g.printclosure()

def main():
	import gzip
	from getopt import gnu_getopt, GetoptError
	from treetransforms import addfanoutmarkers, canonicalize
	from treebank import NegraCorpusReader, DiscBracketCorpusReader, \
			BracketCorpusReader
	from fragments import getfragments
	logging.basicConfig(level=logging.DEBUG, format='%(message)s')
	shortoptions = ''
	flags = ("gzip", "freqs", "packed")
	options = ('inputfmt=', 'inputenc=', 'dopestimator=', 'numproc=')
	try:
		opts, args = gnu_getopt(sys.argv[1:], shortoptions, flags + options)
		model, treebankfile, grammarfile = args
	except (GetoptError, ValueError) as err:
		print "error: %r\n%s" % (err, usage)
		exit(2)
	opts = dict(opts)
	assert model in ("pcfg", "plcfrs", "dopreduction", "doubledop"), (
		"unrecognized model: %r" % model)
	freqs = opts.get('--freqs', False)

	# read treebank
	if opts.get('--inputfmt', 'export') == 'export':
		Reader = NegraCorpusReader
	elif opts.get('--inputfmt') == 'discbracket':
		Reader = DiscBracketCorpusReader
	elif opts.get('--inputfmt') == 'bracket':
		Reader = BracketCorpusReader
	else:
		raise ValueError("unrecognized format: %r" % opts.get('--inputfmt'))

	corpus = Reader(".", treebankfile)
	trees, sents = corpus.parsed_sents().values(), corpus.sents().values()
	for a in trees:
		canonicalize(a)
		addfanoutmarkers(a)

	# read off grammar
	if model in ("pcfg", "plcfrs"):
		grammar = induce_plcfrs(trees, sents, freqs=freqs)
	elif model == "dopreduction":
		estimator = opts.get('--dopestimator', 'dop1')
		grammar = dopreduction(trees, sents, ewe=estimator=='ewe',
				shortestderiv=estimator=='shortest', freqs=freqs,
				packedgraph="--packed" in opts)
	elif model == "doubledop":
		assert opts.get('--dopestimator', 'dop1') == 'dop1'
		numproc = int(opts.get('--numproc', 1))
		fragments = getfragments(trees, sents, numproc)
		grammar, backtransform = doubledop(fragments, freqs=freqs)

	print grammarinfo(grammar)
	if not freqs:
		cgrammar = Grammar(grammar)
		cgrammar.testgrammar(logging)
	rules = grammarfile + ".rules"
	lexicon = grammarfile + ".lex"
	if '--gzip' in opts:
		myopen = gzip.open
		rules += ".gz"
		lexicon += ".gz"
	else:
		myopen = open
	with myopen(rules, "w") as rulesfile:
		with codecs.getwriter('utf-8')(myopen(lexicon, "w")) as lexiconfile:
			# write output
			if model == "pcfg" or opts.get('--inputfmt') == 'bracket':
				if freqs:
					write_lcfrs_grammar(grammar, rulesfile, lexiconfile,
							bitpar=True)
				else:
					cgrammar.write_bitpar_grammar(rulesfile, lexiconfile)
			else:
				if freqs:
					write_lcfrs_grammar(grammar, rulesfile, lexiconfile)
				else:
					cgrammar.write_lcfrs_grammar(rulesfile, lexiconfile)
	if model == "doubledop":
		backtransformfile = "%s.backtransform%s" % (grammarfile,
			".gz" if '--gzip' in opts else "")
		myopen(backtransformfile, "w").writelines(
				"%s\n" % a for a in backtransform.itervalues())
		print "wrote backtransform to", backtransformfile
	print "wrote grammar to %s and %s." % (rules, lexicon)

if __name__ == '__main__':
	if '--test' in sys.argv:
		test()
	else:
		main()
