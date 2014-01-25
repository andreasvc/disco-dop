"""Disambiguate parse forests with various methods for parse selection."""

from __future__ import print_function
import re
from heapq import nlargest
from math import exp, log, isinf, fsum
from random import random
from bisect import bisect_right
from operator import itemgetter, attrgetter
from itertools import count
from collections import defaultdict, OrderedDict
from discodop import plcfrs, _fragments
from discodop.tree import Tree
from discodop.kbest import lazykbest, getderiv
from discodop.grammar import lcfrsproductions
from discodop.treetransforms import addbitsets, unbinarize, canonicalize
from discodop.plcfrs cimport Entry, new_Entry
from discodop.containers cimport Grammar, Rule, LexicalRule, Chart, Edges, \
		SmallChartItem, FatChartItem, Edge, RankedEdge, yieldranges, \
		new_RankedEdge, UChar, UInt, ULong, ULLong, logprobadd, logprobsum
cimport cython

from libc.string cimport memset
cdef extern from "macros.h":
	void SETBIT(ULong a[], int b)

REMOVEIDS = re.compile('@[-0-9]+')
BREMOVEIDS = re.compile(b'@[-0-9]+')
REMOVEWORDTAGS = re.compile('@[^ )]+')


cpdef getderivations(Chart chart, int k, bint kbest=True, bint sample=False,
		derivstrings=True):
	"""Get *k*-best and/or sampled derivations from chart.

	:param k: number of derivations to extract from chart
	:param sample: whether to *k* derivations sample from chart
	:param kbest: whether to extract *k*-best derivations from chart
	:param derivstrings: whether to create derivations as strings
	:returns: tuple ``(derivations, entries)``; two lists of equal length:

		:derivations: list of tuples ``(deriv, logprob)`` where deriv is a
			string; or list of None if ``derivstrings==False``
		:entries: list of RankedEdge objects.
	"""
	cdef list derivations = [], entries = []
	assert kbest or sample
	if kbest:
		derivations, _ = lazykbest(chart, k, derivs=derivstrings)
		entries = chart.rankededges[chart.root()]
	if sample:
		derivations.extend(
				getsamples(chart, k, None))
		# filter out duplicate derivations
		entries = chart.rankededges[chart.root()]
		filteredderivations = dict(zip(derivations, entries))
		entries[:] = filteredderivations.values()
		derivations[:] = filteredderivations.keys()
	return derivations, entries


cpdef marginalize(method, list derivations, list entries, Chart chart,
		list backtransform=None, list sent=None, list tags=None,
		int k=1000, int sldop_n=7, bint bitpar=False):
	"""Take a list of derivations and optimize a given objective function.

	Produces a dictionary of parse trees from derivations, scored by a given
	objective function.

	:param method:
		Available objective functions:

		:'mpp': Most Probable Parse
		:'mpd': Most Probable Derivation
		:'shortest': Most Probable Shortest Derivation
		:'sl-dop': Simplicity-Likelihood DOP; select most likely parse from the
			``sldop_n`` parse trees with the shortest derivations.
		:'sl-dop-simple': Approximation of Simplicity-Likelihood DOP
	:param backtransform:
		Dependending on the value of this parameter, one of two ways is
		employed to turn derivations into parse trees:

		:list of strings: assume Double-DOP and recover derivations by
			substituting fragments in this list for flattened rules in
			derivations.
		:``None``: assume derivations are from DOP reduction and strip
			annotations of the form ``@123`` from labels.
	:param k: when ``method='sl-dop``, number of derivations to consider.
	:param bitpar: whether bitpar was used in nbest mode.
	:returns: a dictionary mapping parsetrees (as strings) to probabilities
		(0 < p <= 1).
	"""
	cdef bint mpd = method == 'mpd'
	cdef bint shortest = method == 'shortest'
	cdef Entry entry
	cdef LexicalRule lexrule
	cdef dict parsetrees = {}, derivs = {}
	cdef str treestr, deriv
	cdef double prob, maxprob
	cdef int m

	if method == 'sl-dop':
		return sldop(dict(derivations), chart, sent, tags, k, sldop_n,
				backtransform, entries, bitpar)
	elif method == 'sl-dop-simple':
		return sldop_simple(dict(derivations), entries, sldop_n, chart,
				backtransform, bitpar)
	elif method == 'shortest':
		# filter out all derivations which are not shortest
		if backtransform is not None and not bitpar:
			maxprob = min([entry.value for entry in entries])
			entries = [entry for entry in entries if entry.value == maxprob]
		elif derivations:
			_, maxprob = min(derivations, key=itemgetter(1))
			derivations = [(deriv, prob) for deriv, prob in derivations
					if prob == maxprob]

	if backtransform is not None and not bitpar:  # Double-DOP
		for entry in entries:
			prob = entry.value
			try:
				treestr = recoverfragments(entry.key, chart, backtransform)
			except:
				continue
			if shortest:
				newprob = exp(-getderivprob(entry.key, chart, sent))
				score = (abs(int(prob / log(0.5))), newprob)
				if treestr not in parsetrees or score > parsetrees[treestr]:
					parsetrees[treestr] = score
					derivs[treestr] = entry.key
			elif not mpd and treestr in parsetrees:
				parsetrees[treestr].append(-prob)
			elif not mpd or (treestr not in parsetrees
						or -prob > parsetrees[treestr][0]):
				parsetrees[treestr] = [-prob]
				derivs[treestr] = entry.key
	else:  # DOP reduction / bitpar
		for (deriv, prob), entry in zip(derivations, entries):
			if backtransform is None:
				treestr = REMOVEIDS.sub('', deriv)
			else:
				try:
					treestr = recoverfragments(deriv, chart, backtransform)
				except:
					continue
			if shortest:
				# for purposes of tie breaking, calculate the derivation
				# probability in a different model.
				if bitpar:
					# because with bitpar we don't know which rules have been
					# used, read off the rules from the derivation ...
					tree = canonicalize(Tree.parse(deriv, parse_leaf=int))
					newprob = 0.0
					for t in tree.subtrees():
						if isinstance(t[0], Tree):
							assert 1 <= len(t) <= 2
							r = ((b'0', t.label.encode('ascii'),
								t[0].label.encode('ascii')) if len(t) == 1 else
								(b'01', t.label.encode('ascii'),
									t[0].label.encode('ascii'),
									t[1].label.encode('ascii')))
							m = chart.grammar.rulenos[r]
							newprob += chart.grammar.bylhs[0][m].prob
						else:
							m = chart.grammar.toid[t.label]
							try:
								lexrule = chart.grammar.lexicalbylhs[m][sent[t[0]]]
							except KeyError:
								newprob += 30.0
							else:
								newprob += lexrule.prob
				else:
					newprob = getderivprob(entry.key, chart, sent)
				score = (abs(int(prob / log(0.5))), exp(-newprob))
				if treestr not in parsetrees or score > parsetrees[treestr]:
					parsetrees[treestr] = score
					derivs[treestr] = deriv
			elif not mpd and treestr in parsetrees:
				# simple way of adding probabilities (too easy):
				parsetrees[treestr].append(-prob)
			elif not mpd or (treestr not in parsetrees
						or -prob > parsetrees[treestr][0]):
				parsetrees[treestr] = [-prob]
				derivs[treestr] = deriv

	for treestr, probs in parsetrees.items() if not shortest else ():
		parsetrees[treestr] = logprobsum(probs)
	for treestr, deriv_ in derivs.items():
		derivs[treestr] = extractfragments(deriv_, chart, backtransform)
	msg = '%d derivations, %d parsetrees' % (
			len(derivations if backtransform is None else entries),
			len(parsetrees))
	return parsetrees, derivs, msg


cdef sldop(dict derivations, Chart chart, list sent, list tags,
		int m, int sldop_n, list backtransform, entries, bint bitpar):
	"""'Proper' method for sl-dop.

	Parses sentence once more to find shortest derivations, pruning away any
	chart item not occurring in the *n* most probable parse trees; we need to
	parse again because we have to consider all derivations for the *n* most
	likely trees.

	:returns: the intersection of the most probable parse trees and their
		shortest derivations, with probabilities of the form (subtrees, prob).

	NB: doesn't seem to work so well, so may contain a subtle bug.
		Does not support PCFG charts."""
	cdef dict derivs = {}
	# collect derivations for each parse tree
	derivsfortree = defaultdict(set)
	if backtransform is None:
		for deriv in derivations:
			derivsfortree[REMOVEIDS.sub('', deriv)].add(deriv)
	elif bitpar:
		for deriv in derivations:
			derivsfortree[recoverfragments(deriv, chart,
					backtransform)].add(deriv)
	else:
		for entry in entries:
			deriv = getderiv(chart.root(), (<Entry>entry).key, chart, None)
			derivations[deriv] = (<Entry>entry).value
			derivsfortree[recoverfragments((<Entry>entry).key, chart,
					backtransform)].add(deriv)
	# sum over probs of derivations to get probs of parse trees
	parsetreeprob = {tree: logprobsum([-derivations[d] for d in ds])
			for tree, ds in derivsfortree.items()}

	nmostlikelytrees = set(nlargest(sldop_n, parsetreeprob,
			key=parsetreeprob.get))
	chart.grammar.switch(u'shortest', True)
	shortestderivations, _explored, chart2 = treeparsing(
			nmostlikelytrees, sent, chart.grammar, m, backtransform, tags)
	if not chart2.rankededges.get(chart2.root()):
		return {}, {}, 'SL-DOP couldn\'t find parse for tree'
	result = {}
	for (deriv, s), entry in zip(shortestderivations,
			chart2.rankededges[chart2.root()]):
		if backtransform is None:
			treestr = REMOVEIDS.sub('', deriv)
		else:
			treestr = recoverfragments(deriv if bitpar else (<Entry>entry).key,
					chart2, backtransform)
		if treestr in nmostlikelytrees and treestr not in result:
			result[treestr] = (abs(int(s / log(0.5))), parsetreeprob[treestr])
			derivs[treestr] = extractfragments(
					deriv if bitpar or backtransform is None
					else (<Entry>entry).key,
					chart2, backtransform)
			if len(result) > sldop_n:
				break
	if not len(result):
		return {}, {}, 'no matching derivation found'
	msg = '(%d derivations, %d of %d parsetrees)' % (
		len(derivations), min(sldop_n, len(parsetreeprob)), len(parsetreeprob))
	return result, derivs, msg


cdef sldop_simple(dict derivations, list entries, int sldop_n,
		Chart chart, list backtransform, bint bitpar):
	"""Simple sl-dop method.

	Estimates shortest derivation directly from number of addressed nodes in
	the *k*-best derivations. After selecting the *n* best parse trees, the one
	with the shortest derivation is returned. In other words, selects shortest
	derivation among the list of available derivations, instead of finding the
	shortest among all possible derivations using Viterbi."""
	cdef Entry entry
	cdef dict derivs = {}, keys = {}
	derivsfortree = defaultdict(set)
	# collect derivations for each parse tree
	if backtransform is None:
		for deriv in derivations:
			tree = REMOVEIDS.sub('', deriv)
			derivsfortree[tree].add(deriv)
	elif bitpar:
		for deriv in derivations:
			tree = recoverfragments(deriv, chart, backtransform)
			derivsfortree[tree].add(deriv)
	else:
		for entry in entries:
			deriv = getderiv(chart.root(), entry.key, chart, '}<')
			tree = recoverfragments(entry.key, chart, backtransform)
			keys[deriv] = entry.key
			derivations[deriv] = entry.value
			derivsfortree[tree].add(deriv)

	# sum over derivations to get parse trees
	parsetreeprob = {tree: logprobsum([-derivations[d] for d in ds])
			for tree, ds in derivsfortree.items()}
	selectedtrees = nlargest(sldop_n, parsetreeprob, key=parsetreeprob.get)

	# the number of fragments used is the number of
	# nodes (open parens), minus the number of interior
	# (addressed) nodes.
	result = {}
	for tree in selectedtrees:
		score, deriv = min([(deriv.count('(') -
				len([a for a in deriv.split() if '@' in a or '}<' in a]),
				deriv)
				for deriv in derivsfortree[tree]])
		result[tree] = (-score, parsetreeprob[tree])
		derivs[tree] = extractfragments(
				deriv if bitpar or backtransform is None
				else keys[deriv], chart, backtransform)
	msg = '(%d derivations, %d of %d parsetrees)' % (
			len(derivations), len(result), len(parsetreeprob))
	return result, derivs, msg


cpdef str recoverfragments(deriv, Chart chart, list backtransform):
	"""Reconstruct a DOP derivation from a derivation with flattened fragments.

	:param deriv: a RankedEdge or a string representing a derivation.
	:param backtransform: a list with fragments (as string templates)
		corresponding to grammar rules.
	:returns: expanded derivation as a string.

	The flattened fragments in the derivation should be left-binarized, expect
	when ``deriv`` is a string in which case the derivation does not have to be
	binarized.

	Does on-the-fly debinarization following labels that are not mapped to a
	label in the coarse grammar, i.e., it assumes that neverblockre is only
	used to avoid blocking nonterminals from the double-dop binarization
	(containing the string '}<'). Note that this means getmapping() has to have
	been called on `chart.grammar`, even when not doing coarse-to-fine
	parsing."""
	if isinstance(deriv, RankedEdge):
		result = recoverfragments_(deriv, chart, backtransform)
	elif isinstance(deriv, basestring):
		deriv = Tree.parse(deriv, parse_leaf=int)
		result = recoverfragments_str(deriv, chart, backtransform)
	else:
		raise ValueError('derivation has unexpected type %r.' % type(deriv))
	return REMOVEWORDTAGS.sub('', result)


cdef str recoverfragments_(RankedEdge deriv, Chart chart,
		list backtransform):
	cdef RankedEdge child
	cdef list children = []
	cdef str frag = backtransform[deriv.edge.rule.no]  # template
	# NB: this is the only code that uses the .head field of RankedEdge

	# collect all children w/on the fly left-factored debinarization
	if deriv.edge.rule.rhs2:  # is there a right child?
		# keep going while left child is part of same binarized constituent
		# instead of looking for a binarization marker in the label string, we
		# use the fact that such labels do not have a mapping as proxy.
		while chart.grammar.mapping[deriv.edge.rule.rhs1] == 0:
			# one of the right children
			children.append((<Entry>chart.rankededges[
					chart.right(deriv)][deriv.right]).key)
			# move on to next node in this binarized constituent
			deriv = (<Entry>chart.rankededges[
					chart.left(deriv)][deriv.left]).key
		# last right child
		if deriv.edge.rule.rhs2:  # is there a right child?
			children.append((<Entry>chart.rankededges[
					chart.right(deriv)][deriv.right]).key)
	elif chart.grammar.mapping[deriv.edge.rule.rhs1] == 0:
		deriv = (<Entry>chart.rankededges[
				chart.left(deriv)][deriv.left]).key
	# left-most child
	children.append((<Entry>chart.rankededges[
			chart.left(deriv)][deriv.left]).key)

	# recursively expand all substitution sites
	# FIXME: to avoid using str + decoding, we could use
	# PyObject* PyBytes_FromFormat(const char *format, ...)
	# PyBytes_FromFormat('(%s %d)', <char *>..., ...)
	children = [('(%s %d)' % (
		str(chart.grammar.tolabel[chart.label(child.head)].decode('ascii')),
		chart.lexidx(child.head, child.edge)))
		if child.edge.rule is NULL
		else recoverfragments_(child, chart, backtransform)
				for child in reversed(children)]

	# substitute results in template
	return frag.format(*children)

	# even better: build result incrementally; use bytearray,
	# extended in recursive calls w/strings from backtransform.
	# step 1: collect RankedEdges in a list (children);
	#		i.e., exctract nodes from binarized constituent.
	# step 2: iterate over parts of template, alternately adding string from it
	#		and making a recursive call to insert the relevant child RankedEdge
	# new backtransform format:
	#backtransform[prod] = (list_of_strs, list_of_idx)
	#backtransform[34] = ([b'(NP (DT ', b') (NN ', b'))'], [0, 1])
	#alternatively: (better locality?)
	#frag = backtransform[34] = [b'(NP (DT ', 0, b') (NN ', 1, b'))']
	#result += frag[0]
	#for n in range(1, len(result), 2):
	#	foo(result, children[frag[n]])
	#	result += frag[n + 1]


cdef str recoverfragments_str(deriv, Chart chart, list backtransform):
	cdef list children = []
	cdef str frag
	# e.g.: ('0123', 'X', 'A', 'B', 'C', 'D')
	prod = (''.join(map(str, range(len(deriv)))).encode('ascii'),
			deriv.label) + tuple([a.label.encode('ascii') for a in deriv])
	frag = backtransform[chart.grammar.rulenos[prod]]  # template
	# collect children w/on the fly left-factored debinarization
	if len(deriv) >= 2:  # is there a right child?
		# keep going while left child is part of same binarized constituent
		# this shortcut assumes that neverblockre is only used to avoid
		# blocking nonterminals from the double-dop binarization.
		while '}<' in deriv[0].label:
			# one of the right children
			children.extend(reversed(deriv[1:]))
			# move on to next node in this binarized constituent
			deriv = deriv[0]
		# last right child
		children.extend(reversed(deriv[1:]))
	elif '}<' in deriv[0].label:
		deriv = deriv[0]
	# left-most child
	children.append(deriv[0])

	# recursively expand all substitution sites
	children = [recoverfragments_str(child, chart, backtransform)
			if isinstance(child[0], Tree)
			else ('(%s %d)' % (child.label, child[0]))
			for child in reversed(children)]

	# substitute results in template
	return frag.format(*children)


def extractfragments(deriv, chart, list backtransform):
	"""Extract the list of fragments that were used in a given derivation.

	:returns: a list of fragments of the form ``(frag, sent)`` where frag is a
		string and sent is a list of tokens; in ``sent``, ``None`` indicates a
		frontier non-terminal, and a string indicates a token. """
	result = []
	if isinstance(deriv, RankedEdge):
		extractfragments_(deriv, chart, backtransform, result)
	elif isinstance(deriv, basestring) and backtransform is None:
		deriv = Tree.parse(deriv, parse_leaf=int)
		result = [REMOVEIDS.sub('', str(splitfrag(node)))
				for node in deriv.subtrees(frontiernt)]
	elif isinstance(deriv, basestring):
		deriv = Tree.parse(deriv, parse_leaf=int)
		extractfragments_str(deriv, chart, backtransform, result)
	else:
		raise ValueError
	return [_fragments.pygetsent(frag, chart.sent) for frag in result]


def frontiernt(node):
	"""Test whether node from a DOP derivation is a frontier nonterminal."""
	return '@' not in node.label


def splitfrag(node):
	"""Return a copy of a tree with subtrees labeled without '@' removed."""
	children = []
	for child in node:
		if not isinstance(child, Tree):
			children.append(child)
		elif '@' in child.label:
			children.append(child if isinstance(child[0], int)
					else splitfrag(child))
		else:
			children.append(Tree(child.label, ['%d:%d' % (
					min(child.leaves()), max(child.leaves()))]))
	return Tree(node.label, children)


cdef extractfragments_(RankedEdge deriv, Chart chart,
		list backtransform, list result):
	cdef RankedEdge child
	cdef list children = []
	cdef str frag = backtransform[deriv.edge.rule.no]  # template

	# collect all children w/on the fly left-factored debinarization
	if deriv.edge.rule.rhs2:  # is there a right child?
		# keep going while left child is part of same binarized constituent
		# instead of looking for a binarization marker in the label string, we
		# use the fact that such labels do not have a mapping as proxy.
		while chart.grammar.mapping[deriv.edge.rule.rhs1] == 0:
			# one of the right children
			children.append((<Entry>chart.rankededges[
					chart.right(deriv)][deriv.right]).key)
			# move on to next node in this binarized constituent
			deriv = (<Entry>chart.rankededges[
					chart.left(deriv)][deriv.left]).key
		# last right child
		if deriv.edge.rule.rhs2:  # is there a right child?
			children.append((<Entry>chart.rankededges[
					chart.right(deriv)][deriv.right]).key)
	elif chart.grammar.mapping[deriv.edge.rule.rhs1] == 0:
		deriv = (<Entry>chart.rankededges[
				chart.left(deriv)][deriv.left]).key
	# left-most child
	children.append((<Entry>chart.rankededges[
			chart.left(deriv)][deriv.left]).key)

	result.append(frag.format(*['(%s %s)' % (
			chart.grammar.tolabel[chart.label(deriv.head)].split('@')[0],
			(chart.lexidx(deriv.head, deriv.edge)
				if '@' in chart.grammar.tolabel[chart.label(deriv.head)]
				else yieldranges(chart.indices(deriv.head))))
			for deriv in reversed(children)]))
	# recursively visit all substitution sites
	for child in reversed(children):
		if not child.edge.rule is NULL:
			extractfragments_(child, chart, backtransform, result)
		elif '@' not in chart.grammar.tolabel[chart.label(deriv.head)]:
			result.append('(%s %d)' % (
					chart.grammar.tolabel[chart.label(deriv.head)],
					chart.lexidx(deriv.head, deriv.edge)))


cdef extractfragments_str(deriv, Chart chart,
		list backtransform, list result):
	cdef list children = []
	cdef str frag
	assert 1 <= len(deriv) <= 2, deriv
	prod = (b'0', deriv.label.encode('ascii'), deriv[0].label.encode('ascii')
			) if len(deriv) == 1 else (b'01', deriv.label.encode('ascii'),
			deriv[0].label.encode('ascii'),
			deriv[1].label.encode('ascii'))
	frag = backtransform[chart.grammar.rulenos[prod]]  # template
	# collect children w/on the fly left-factored debinarization
	if len(deriv) == 2:  # is there a right child?
		# keep going while left child is part of same binarized constituent
		# this shortcut assumes that neverblockre is only used to avoid
		# blocking nonterminals from the double-dop binarization.
		while '}<' in deriv[0].label:
			# one of the right children
			children.append(deriv[1])
			# move on to next node in this binarized constituent
			deriv = deriv[0]
		# last right child
		if len(deriv) == 2:  # is there a right child?
			children.append(deriv[1])
	elif '}<' in deriv[0].label:
		deriv = deriv[0]
	# left-most child
	children.append(deriv[0])

	result.append(frag.format(*['(%s %s)' % (
			child.label.split('@')[0],
			(child[0] if '@' in child.label
				else _fragments.yieldranges(sorted(child.leaves()))))
			for child in reversed(children)]))
	# recursively visit all substitution sites
	for child in reversed(children):
		if isinstance(child[0], Tree):
			extractfragments_str(child, chart, backtransform, result)
		elif '@' not in chart.label:
			result.append('(%s %d)' % (chart.label, chart[0]))


def treeparsing(trees, sent, Grammar grammar, int m, backtransform, tags=None):
	"""Assign probabilities to a sequence of trees with a DOP grammar.

	Given a sequence of trees (as strings), parse them with a DOP grammar
	to get parse tree probabilities; will consider multiple derivations."""
	# Parsing & pruning inside the disambiguation module is rather kludgy,
	# but the problem is that we need to get probabilities of trees,
	# not just of derivations. Therefore the coarse-to-fine methods
	# do not apply directly.
	cdef FatChartItem fitem
	cdef int x, lensent
	whitelist = [{} for _ in grammar.toid]
	for treestr in trees:
		tree = Tree.parse(treestr, parse_leaf=int)
		lensent = len(tree.leaves())
		for n in tree.subtrees():
			leaves = n.leaves()
			if lensent < sizeof(ULLong) * 8:
				item = SmallChartItem(0, sum([1L << x for x in leaves]))
			else:
				fitem = item = FatChartItem(0)
				for x in leaves:
					SETBIT(fitem.vec, x)
			try:
				whitelist[grammar.toid[n.label.encode('ascii')]][item] = 0.0
			except KeyError:
				return [], "'%s' not in grammar" % n.label, None

	# Project labels to all possible labels that generate that label. For DOP
	# reduction, all possible ids; for Double DOP, ignore artificial labels.
	for label, n in grammar.toid.items():
		if backtransform is None:
			whitelist[n] = whitelist[grammar.toid[BREMOVEIDS.sub(b'', label)]]
		elif b'@' in label or b'}<' in label:
			whitelist[n] = None  # do not prune item

	# finally, we parse with the small set of allowed labeled spans.
	# we do not parse with PCFG even if possible, because that requires a
	# different way of pruning.
	chart, _ = plcfrs.parse(sent, grammar, tags=tags, whitelist=whitelist)
	if not chart:
		return [], "tree parsing failed", None
	return lazykbest(chart, m) + (chart, )


cdef double getderivprob(RankedEdge deriv, Chart chart, list sent):
	"""Recursively calculate probability of a derivation.

	Useful to obtain probability of derivation under different probability
	model of the same grammar."""
	cdef double result
	if deriv.edge.rule is NULL:  # is terminal
		label = chart.label(deriv.head)
		word = sent[chart.lexidx(deriv.head, deriv.edge)]
		return (<LexicalRule>chart.grammar.lexicalbylhs[label][word]).prob
	result = chart.grammar.bylhs[0][deriv.edge.rule.no].prob
	result += getderivprob((<Entry>chart.rankededges[
			chart.left(deriv)][deriv.left]).key,
			chart, sent)
	if deriv.edge.rule.rhs2:
		result += getderivprob((<Entry>chart.rankededges[
				chart.right(deriv)][deriv.right]).key,
				chart, sent)
	return result

cpdef viterbiderivation(Chart chart):
	"""Wrapper to get Viterbi derivation from chart."""
	# Ask for at least 10 derivations because unary cycles.
	derivations = lazykbest(chart, 10)[0]
	return derivations[0]


def getsamples(Chart chart, k, debin=None):
	"""Samples *k* derivations from a chart."""
	cdef dict tables = {}
	cdef Edges edges
	cdef Edge *edge
	cdef double prob, prev
	cdef size_t n, m
	chartidx = {}
	for item in chart.getitems():
		chartidx[item] = []
		for n, edges in enumerate(chart.parseforest[item]):
			for m in range(edges.len):
				edge = &(edges.data[m])
				if edge.rule is NULL:
					chartidx[item].append((chart.subtreeprob(item), n, m))
				else:
					prob = edge.rule.prob
					# FIXME: work w/inside prob?
					#prob += chart.subtreeprob(chart._left(item, edge))
					#if edge.rule.rhs2:
					#	prob += chart.subtreeprob(chart._right(item, edge))
					chartidx[item].append((prob, n, m))
		# sort edges so that highest prob (=lowest neglogprob) comes first
		chartidx[item].sort()
	for item in chartidx:
		tables[item] = []
		prev = 0.0
		for prob, _, _ in chartidx[item]:
			prev += exp(-prob)
			tables[item].append(prev)
	result = []
	for _ in range(k):
		treestr, p = samplechart(chart.root(), chart, chartidx, tables, debin)
		result.append((str(treestr), p))
	return result


cdef samplechart(item, Chart chart,
		dict chartidx, dict tables, bytes debin):
	"""Samples a derivation from a chart."""
	cdef Edge *edge
	cdef double prob
	cdef RankedEdge rankededge
	cdef list lst = tables[item]
	rnd = random() * lst[len(lst) - 1]
	idx = bisect_right(lst, rnd)
	_, n, m = chartidx[item][idx]
	edge = &((<Edges>chart.parseforest[item][n]).data[m])
	label = chart.label(item)
	if edge.rule is NULL:  # terminal
		idx = chart.lexidx(item, edge)
		rankededge = new_RankedEdge(item, edge, 0, -1)
		chart.rankededges.setdefault(item, []).append(
				new_Entry(rankededge, chart.subtreeprob(item), 0))
		deriv = "(%s %d)" % (chart.grammar.tolabel[label].decode('ascii'), idx)
		return deriv, chart.subtreeprob(item)
	children = [samplechart(chart.copy(child), chart, chartidx, tables, debin)
			for child in (chart._left(item, edge), chart._right(item, edge))
				if child is not None]
	if debin is not None and debin in chart.grammar.tolabel[label]:
		tree = ' '.join([a for a, _ in children])
	else:
		tree = '(%s %s)' % (chart.grammar.tolabel[label].decode('ascii'),
				' '.join([a for a, _ in children]))
	# create an edge that has as children the edges that were just created
	# by our recursive calls
	rankededge = new_RankedEdge(item, edge,
			len(chart.rankededges[chart._left(item, edge)]) - 1,
			(len(chart.rankededges[chart._right(item, edge)]) - 1)
				if edge.rule.rhs2 else -1)
	prob = edge.rule.prob + fsum([b for _, b in children])
	# NB: this is actually 'samplededges', not 'rankededges'
	chart.rankededges.setdefault(item, []).append(
			new_Entry(rankededge, prob, 0))
	return tree, prob


def doprerank(chart, sent, k, Grammar coarse, Grammar fine):
	"""Rerank *k*-best coarse trees w/parse probabilities of DOP reduction.

	cf. ``dopparseprob()``."""
	cdef dict results = {}
	derivations, _ = lazykbest(chart, k, derivs=True)
	for derivstr, _ in derivations:
		deriv = addbitsets(derivstr)
		results[derivstr] = exp(dopparseprob(deriv, sent, coarse, fine))
	return results


def dopparseprob(tree, sent, Grammar coarse, Grammar fine):
	"""Compute the exact DOP parse probability of a Tree in a DOP reduction.

	This follows up on a suggestion made by Goodman (2003, p. 143) of
	calculating DOP probabilities of given parse trees, although I'm not sure
	it has complexity *O(nP)* as he suggests (with *n* as number of nodes in
	input, and *P* as max number of rules consistent with a node in the input).
	Furthermore, the idea of sampling trees "long enough" until we have the MPP
	is no faster than sampling without applying this procedure, because to
	determine that some probability *p* is the maximal probability, we need to
	collect the probability mass *p_seen* of enough parse trees such that we
	have some parsetree with probability *p* > (1 - *p_seen*), which requires
	first seeing almost all parse trees, unless *p* is exceptionally high.
	Hence, this method is mostly useful in a reranking framework where it is
	known in advance that a small set of trees is of interest.

	Expects a mapping which gives a list of consistent rules from the reduction
	as produced by ``fine.getrulemapping(coarse)``.

	NB: this algorithm could also be used to determine the probability of
	derivations, but then the input would have to distinguish whether nodes are
	internal nodes of fragments, or whether they join two fragments."""
	neginf = float('-inf')
	cdef dict chart = {}  # chart[label, left, right] = prob
	cdef tuple a, b, c
	cdef Rule *rule
	cdef LexicalRule lexrule
	cdef object n  # pyint
	assert fine.logprob, 'Grammar should have log probabilities.'
	# Log probabilities are not ideal here because we do lots of additions,
	# but the probabilities are very small.
	# A possible alternative is to scale them somehow.

	# add all matching POS tags
	for n, pos in tree.pos():
		pos = pos.encode('ascii')
		word = sent[n]
		for lexrule in fine.lexicalbyword[word]:
			if (fine.tolabel[lexrule.lhs] == pos
					or fine.tolabel[lexrule.lhs].startswith(pos + b'@')):
				chart[lexrule.lhs, 1 << n] = -lexrule.prob

	# do post-order traversal (bottom-up)
	for node, (prod, yf) in list(zip(tree.subtrees(),
			lcfrsproductions(tree, sent)))[::-1]:
		if not isinstance(node[0], Tree):
			continue
		yf = ','.join([''.join(map(str, x)) for x in yf]).encode('ascii')
		prod = coarse.rulenos[(yf, ) + tuple([x.encode('ascii') for x in prod])]
		if len(node) == 1:  # unary node
			for ruleno in fine.rulemapping[prod]:
				rule = &(fine.bylhs[0][ruleno])
				b = (rule.rhs1, node.bitset)
				if b in chart:
					a = (rule.lhs, node.bitset)
					if a in chart:
						chart[a] = logprobadd(chart[a], -rule.prob + chart[b])
					else:
						chart[a] = (-rule.prob + chart[b])
		elif len(node) == 2:  # binary node
			for ruleno in fine.rulemapping[prod]:
				rule = &(fine.bylhs[0][ruleno])
				b = (rule.rhs1, node[0].bitset)
				c = (rule.rhs2, node[1].bitset)
				if b in chart and c in chart:
					a = (rule.lhs, node.bitset)
					if a in chart:
						chart[a] = logprobadd(chart[a],
							(-rule.prob + chart[b] + chart[c]))
					else:
						chart[a] = -rule.prob + chart[b] + chart[c]
		else:
			raise ValueError('expected binary tree without empty nodes.')
	return chart.get((fine.toid[tree.label], tree.bitset), neginf)


def test():
	from discodop.grammar import dopreduction
	from discodop.containers import Grammar
	from discodop import plcfrs

	def e(x):
		a, b = x
		if isinstance(b, tuple):
			return (a.replace("@", ''), (int(abs(b[0])), b[1]))
		return a.replace("@", ''), b

	def maxitem(d):
		return max(d.items(), key=itemgetter(1))

	trees = [Tree.parse(t, parse_leaf=int) for t in
		'''(ROOT (A (A 0) (B 1)) (C 2))
		(ROOT (C 0) (A (A 1) (B 2)))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (A 0) (C (B 1) (C 2)))
		(ROOT (A 0) (C (B 1) (C 2)))'''.splitlines()]
	sents = [a.split() for a in
		'''d b c\n c a b\n a e f\n a e f\n a e f\n a e f\n d b f\n d b f
		d b f\n d b g\n e f c\n e f c\n e f c\n e f c\n e f c\n e f c\n f b c
		a d e'''.splitlines()]
	xgrammar, altweights = dopreduction(trees, sents)
	grammar = Grammar(xgrammar)
	grammar.register(u'shortest', altweights['shortest'])
	print(grammar)
	sent = "a b c".split()
	chart, _ = plcfrs.parse(sent, grammar, None, True)
	assert chart
	vitderiv, vitprob = viterbiderivation(chart)
	derivations, entries = getderivations(chart, 1000, True, False, True)
	mpd, _, _ = marginalize("mpd", derivations, entries, chart)
	mpp, _, _ = marginalize("mpp", derivations, entries, chart)
	sldop_, _, _ = marginalize("sl-dop", derivations, entries, chart, k=1000,
			sldop_n=7, sent=sent)
	sldopsimple, _, _ = marginalize("sl-dop-simple", derivations, entries,
			chart, k=1000, sldop_n=7, sent=sent)
	short, _, _ = marginalize("shortest", derivations, entries, chart,
			sent=sent)
	derivations, entries = getderivations(chart, 1000, False, True, True)
	mppsampled, _, _ = marginalize("mpp", derivations, entries, chart)
	print("\nvit:\t\t%s %r" % e((REMOVEIDS.sub('', vitderiv), exp(-vitprob))),
		"MPD:\t\t%s %r" % e(maxitem(mpd)),
		"MPP:\t\t%s %r" % e(maxitem(mpp)),
		"MPP sampled:\t%s %r" % e(maxitem(mppsampled)),
		"SL-DOP n=7:\t%s %r" % e(maxitem(sldop_)),
		"simple SL-DOP:\t%s %r" % e(maxitem(sldopsimple)),
		"shortest:\t%s %r" % e(maxitem(short)), sep='\n')
