""" Extract parse tree(s) from a chart following a particular objective
function. """

from __future__ import print_function
import re
import logging
from heapq import nlargest
from math import exp, log, isinf, fsum
from random import random
from bisect import bisect_right
from operator import itemgetter, attrgetter
from itertools import count
from collections import defaultdict, OrderedDict
from tree import Tree
from kbest import lazykbest, getderiv
from grammar import lcfrs_productions
from treetransforms import addbitsets
import plcfrs
from agenda cimport Entry, new_Entry
from treetransforms import unbinarize, canonicalize
from containers cimport Grammar, ChartItem, SmallChartItem, FatChartItem, \
		CFGChartItem, Edge, LCFRSEdge, CFGEdge, RankedEdge, RankedCFGEdge, \
		LexicalRule, Rule, UChar, UInt, ULong, ULLong, logprobadd, logprobsum
cimport cython

from libc.string cimport memset
cdef extern from "macros.h":
	void SETBIT(ULong a[], int b)

REMOVEIDS = re.compile('@[-0-9]+')
BREMOVEIDS = re.compile(b'@[-0-9]+')
REMOVEWORDTAGS = re.compile('@[^ )]+')


cpdef marginalize(method, chart, ChartItem start, Grammar grammar, int n,
		bint sample=False, bint kbest=True, list sent=None, list tags=None,
		int sldop_n=7, list backtransform=None,
		bint bitpar=False):
	""" Approximate MPP or MPD by summing over n random/best derivations from
	chart, return a dictionary mapping parsetrees to probabilities """
	cdef bint mpd = method == 'mpd'
	cdef bint shortest = method == 'shortest'
	cdef Entry entry
	cdef LexicalRule lexrule
	cdef dict parsetrees = {}, derivs = {}
	cdef list derivations = [], entries = []
	cdef str treestr, deriv
	cdef double prob, maxprob
	cdef int m

	assert kbest or sample
	if kbest and not bitpar:
		derivations, D, _ = lazykbest(chart, start, n, grammar.tolabel,
				None, derivs=backtransform is None)
		if isinstance(start, CFGChartItem):
			entries = D[(<CFGChartItem>start).start][
					(<CFGChartItem>start).end][start.label]
		else:
			entries = D[start]
	elif bitpar:
		derivations = sorted(chart.items(), key=itemgetter(1))
		entries = [None] * len(derivations)
		D = {}
	elif isinstance(start, CFGChartItem):
		D = [[{} for right in left] for left in chart]
	else:
		D = {}
	if sample:
		if bitpar:
			return parsetrees, derivs, 'sampling not possible with bitpar.'
		derivations.extend(
				getsamples(D, chart, start, n, grammar.tolabel, None))
		# filter out duplicate derivations
		if isinstance(start, CFGChartItem):
			entries = D[(<CFGChartItem>start).start][
					(<CFGChartItem>start).end][start.label]
		else:
			entries = D[start]
		filteredderivations = dict(zip(derivations, entries))
		entries[:] = filteredderivations.values()
		derivations[:] = filteredderivations.keys()
	if method == 'sl-dop':
		return sldop(dict(derivations), chart, sent, tags, grammar,
				n, sldop_n, backtransform, D, entries, bitpar)
	elif method == 'sl-dop-simple':
		return sldop_simple(dict(derivations), entries, n, sldop_n,
				D, chart, start, grammar, backtransform, bitpar)
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
			treestr = recoverfragments(entry.key, D,
					grammar, backtransform)
			if shortest:
				newprob = exp(-getderivprob(entry.key, D, sent, grammar))
				score = (prob / log(0.5), newprob)
				if treestr not in parsetrees or score > parsetrees[treestr]:
					parsetrees[treestr] = score
					derivs[treestr] = extractfragments(
							entry.key, D, grammar, backtransform)
			elif not mpd and treestr in parsetrees:
				parsetrees[treestr].append(-prob)
			elif not mpd or (treestr not in parsetrees
						or -prob > parsetrees[treestr][0]):
				parsetrees[treestr] = [-prob]
				derivs[treestr] = extractfragments(
						entry.key, D, grammar, backtransform)
	else:  # DOP reduction / bitpar
		for (deriv, prob), entry in zip(derivations, entries):
			if backtransform is None:
				treestr = REMOVEIDS.sub('', deriv)
			else:
				treestr = recoverfragments(deriv, D, grammar, backtransform)
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
							if len(t) == 1:
								r = (b'0', t.label, t[0].label)
							elif len(t) == 2:
								r = (b'01', t.label, t[0].label, t[1].label)
							m = grammar.rulenos[r]
							newprob += grammar.bylhs[0][m].prob
						else:
							m = grammar.toid[t.label]
							try:  # FIXME: bitpar smooths tags w/weight < 0.1!
								lexrule = grammar.lexicalbylhs[m][sent[t[0]]]
							except KeyError:
								newprob += 30.0
							else:
								newprob += lexrule.prob
				else:
					newprob = getderivprob(entry.key, D, sent, grammar)
				score = (prob / log(0.5), exp(-newprob))
				if treestr not in parsetrees or score > parsetrees[treestr]:
					parsetrees[treestr] = score
			elif not mpd and treestr in parsetrees:
				# simple way of adding probabilities (too easy):
				parsetrees[treestr].append(-prob)
			elif not mpd or (treestr not in parsetrees
						or -prob > parsetrees[treestr][0]):
				parsetrees[treestr] = [-prob]

	for treestr, probs in parsetrees.items() if not shortest else ():
		parsetrees[treestr] = logprobsum(probs)
	msg = '%d derivations, %d parsetrees' % (
			len(derivations if backtransform is None else entries),
			len(parsetrees))
	return parsetrees, derivs, msg


cdef sldop(dict derivations, chart, list sent, list tags, Grammar grammar,
		int m, int sldop_n, list backtransform, D, entries, bint bitpar):
	""" 'Proper' method for sl-dop. Parses sentence once more to find shortest
	derivations, pruning away any chart item not occurring in the n most
	probable parse trees; we need to parse again because we have to consider
	all derivations for the n most likely trees.

	:returns: the intersection of the most probable parse trees and their
		shortest derivations, with probabilities of the form (subtrees, prob).

	NB: doesn't seem to work so well, so may contain a subtle bug.
		does not support PCFG charts. """
	cdef dict derivs = {}
	# collect derivations for each parse tree
	derivsfortree = defaultdict(set)
	if backtransform is None:
		for deriv in derivations:
			derivsfortree[REMOVEIDS.sub('', deriv)].add(deriv)
	elif bitpar:
		for deriv in derivations:
			derivsfortree[recoverfragments(deriv, D,
					grammar, backtransform)].add(deriv)
	else:
		for entry in entries:
			deriv = getderiv((<Entry>entry).key, D, chart,
					grammar.tolabel, None)
			derivations[deriv] = (<Entry>entry).value
			derivsfortree[recoverfragments((<Entry>entry).key, D,
					grammar, backtransform)].add(deriv)
	# sum over probs of derivations to get probs of parse trees
	parsetreeprob = {tree: logprobsum([-derivations[d] for d in ds])
			for tree, ds in derivsfortree.items()}

	nmostlikelytrees = set(nlargest(sldop_n, parsetreeprob,
			key=parsetreeprob.get))
	grammar.switch(u'shortest', True)
	shortestderivations, DD, msg, start = treeparsing(
			nmostlikelytrees, sent, grammar, m, backtransform, tags)
	if not DD.get(start):
		return {}, {}, msg
	result = {}
	for (deriv, s), entry in zip(shortestderivations, DD[start]):
		if backtransform is None:
			treestr = REMOVEIDS.sub('', deriv)
		else:
			treestr = recoverfragments(deriv if bitpar else (<Entry>entry).key,
					DD, grammar, backtransform)
		if treestr in nmostlikelytrees and treestr not in result:
			result[treestr] = (s / log(0.5), parsetreeprob[treestr])
			if backtransform is not None:
				derivs[treestr] = extractfragments(
						deriv if bitpar else (<Entry>entry).key,
						DD, grammar, backtransform)
			if len(result) > sldop_n:
				break
	if not len(result):
		return {}, {}, 'no matching derivation found'
	msg = '(%d derivations, %d of %d parsetrees)' % (
		len(derivations), min(sldop_n, len(parsetreeprob)), len(parsetreeprob))
	return result, derivs, msg


cdef sldop_simple(dict derivations, list entries, int m, int sldop_n,
		D, chart, start, Grammar grammar, list backtransform, bint bitpar):
	""" simple sl-dop method; estimates shortest derivation directly from
	number of addressed nodes in the k-best derivations. After selecting the n
	best parse trees, the one with the shortest derivation is returned.
	In other words, selects shortest derivation among the list of available
	derivations, instead of finding the shortest among all possible derivations
	using Viterbi. """
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
			tree = recoverfragments(deriv, D, grammar, backtransform)
			derivsfortree[tree].add(deriv)
	else:
		for entry in entries:
			deriv = getderiv(entry.key, D, chart, grammar.tolabel, '}<')
			tree = recoverfragments(entry.key, D, grammar, backtransform)
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
		if backtransform is not None:
			derivs[tree] = extractfragments(deriv if bitpar else keys[deriv],
					D, grammar, backtransform)
	msg = '(%d derivations, %d of %d parsetrees)' % (
			len(derivations), len(result), len(parsetreeprob))
	return result, derivs, msg


def getsamples(D, chart, start, n, tolabel, debin=None):
	""" Samples n derivations from a chart. """
	if isinstance(start, CFGChartItem):
		return getsamples_cfg(D, chart, start, n, tolabel, debin)
	return getsamples_lcfrs(D, chart, start, n, tolabel, debin)


def getsamples_lcfrs(D, chart, start, n, tolabel, debin=None):
	cdef Edge edge
	cdef dict tables = {}
	chartcopy = {}
	for item in chart:
		#FIXME: work w/inside prob right?
		# sort so that highest probability edges come first
		chartcopy[item] = sorted(chart[item])
	for item in chartcopy:
		tables[item] = []
		prev = 0.0
		for edge in chartcopy[item]:
			prev += exp(-edge.inside)
			tables[item].append(prev)
	result = []
	for _ in range(n):
		s, p = samplechart_lcfrs(
				D, chartcopy, start, tolabel, tables, debin)
		result.append((str(s), p))
	return result


cdef samplechart_lcfrs(dict D, dict chart, ChartItem start, list tolabel,
		dict tables, bytes debin):
	""" Samples a derivation from a chart. """
	cdef LCFRSEdge edge
	cdef ChartItem child
	cdef double prob
	lst = tables[start]
	rnd = random() * lst[len(lst) - 1]
	idx = bisect_right(lst, rnd)
	edge = chart[start][idx]
	if edge.left.label == 0:  # == "Epsilon":
		idx = edge.left.lexidx()
		newedge = RankedEdge(start, edge, 0, -1)
		D.setdefault(start, []).append(new_Entry(newedge, edge.inside, 0))
		deriv = "(%s %d)" % (tolabel[start.label].decode('ascii'), idx)
		return deriv, edge.inside
	children = [samplechart_lcfrs(D, chart, child, tolabel, tables, debin)
			for child in (edge.left, edge.right) if child.label]
	if debin is not None and debin in tolabel[start.label]:
		tree = ' '.join([a for a, _ in children])
	else:
		tree = '(%s %s)' % (tolabel[start.label].decode('ascii'),
				' '.join([a for a, _ in children]))
	# create an edge that has as children the edges that were just created
	# by our recursive calls
	newedge = RankedEdge(start, edge, len(D[edge.left]) - 1,
			(len(D[edge.right]) - 1) if edge.right.label else -1)
	prob = edge.rule.prob + fsum([b for _, b in children])
	D.setdefault(start, []).append(new_Entry(newedge, prob, 0))
	return tree, prob


def getsamples_cfg(list D, list chart, CFGChartItem start, int n, list tolabel,
		debin=None):
	cdef Edge edge
	cdef dict tables = {}
	chartcopy = [[{} for r in l] for l in chart]
	for l in range(start.end):
		for r in range(l + 1, start.end + 1):
			for x in chart[l][r]:
				chartcopy[l][r][x] = sorted(chart[l][r][x])
				item = l, r, x
				tables[item] = []
				prev = 0.0
				for edge in chartcopy[l][r][x]:
					prev += exp(-edge.inside)
					tables[item].append(prev)
	result = []
	for _ in range(n):
		s, p = samplechart_cfg(
				D, chartcopy, (start.start, start.end, start.label),
						tolabel, tables, debin)
		result.append((str(s), p))
	return result


cdef samplechart_cfg(list D, list chart, tuple start, list tolabel,
		dict tables, bytes debin):
	""" Samples a derivation from a chart. """
	cdef CFGEdge edge
	cdef double prob
	left, right, label = start
	lst = tables[left, right, label]
	rnd = random() * lst[len(lst) - 1]
	idx = bisect_right(lst, rnd)
	edge = chart[left][right][label][idx]
	if edge.rule is NULL:  # == "Epsilon":
		newedge = RankedCFGEdge(label, left, right, edge, 0, -1)
		D[left][right].setdefault(label, []).append(
				new_Entry(newedge, edge.inside, 0))
		deriv = "(%s %d)" % (tolabel[label].decode('ascii'), left)
		return deriv, edge.inside
	children = [samplechart_cfg(D, chart, child, tolabel, tables, debin)
			for child in ((left, edge.mid, edge.rule.rhs1),
				(edge.mid, right, edge.rule.rhs2)) if child[2]]
	if debin is not None and debin in tolabel[label]:
		tree = ' '.join([a for a, _ in children])
	else:
		tree = '(%s %s)' % (tolabel[label].decode('ascii'),
				' '.join([a for a, _ in children]))
	# create an edge that has as children the edges that were just created
	# by our recursive calls
	newedge = RankedCFGEdge(label, left, right, edge,
			len(D[left][edge.mid][edge.rule.rhs1]) - 1,
			(len(D[edge.mid][right][edge.rule.rhs2]) - 1)
				if edge.rule.rhs2 else -1)
	prob = edge.rule.prob + fsum([b for _, b in children])
	D[left][right].setdefault(label, []).append(new_Entry(newedge, prob, 0))
	return tree, prob


def treeparsing(trees, sent, Grammar grammar, int m, backtransform, tags=None):
	""" Given a sequence of trees (as strings), parse them with a DOP grammar
	to get parse tree probabilities; i.e., will consider multiple derivations.
	"""
	# Parsing & pruning inside the disambiguation module is rather kludgy,
	# but the problem is that we need to get probabilities of trees,
	# not just of derivations. Therefore the coarse-to-fine methods
	# do not apply directly.
	cdef ChartItem item
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
			whitelist[grammar.toid[n.label.encode('ascii')]][item] = 0.0

	# Project labels to all possible labels that generate that label. For DOP
	# reduction, all possible ids; for Double DOP, ignore artificial labels.
	for label, n in grammar.toid.items():
		if backtransform is None:
			whitelist[n] = whitelist[grammar.toid[BREMOVEIDS.sub(b'', label)]]
		elif b'@' in label or b'}<' in label:
			whitelist[n] = None  # do not prune item
		else:
			whitelist[n] = whitelist[grammar.toid[label]]

	# finally, we parse with the small set of allowed labeled spans.
	# we do parse with PCFG even if possible, because that requires a different
	# way of pruning.
	chart, start, _ = plcfrs.parse(sent, grammar, tags=tags,
			whitelist=whitelist)
	if not start:
		return [], {}, "tree parsing failed", None
	return lazykbest(chart, start, m, grammar.tolabel) + (start, )


cdef double getderivprob(deriv, D, sent, Grammar grammar):
	""" Given a derivation as a ranked edge, recursively calculate its
	probability according to a grammar, which has to have matching rules & rule
	numbers. """
	if isinstance(deriv, RankedEdge):
		return getderivprob_lcfrs(deriv, D, sent, grammar)
	elif isinstance(deriv, RankedCFGEdge):
		return getderivprob_cfg(deriv, D, sent, grammar)


cdef double getderivprob_lcfrs(RankedEdge deriv, dict D,
		list sent, Grammar grammar):
	cdef double result
	if deriv.edge.rule is NULL:  # is terminal
		word = sent[deriv.edge.left.lexidx()]
		return (<LexicalRule>grammar.lexicalbylhs[deriv.head.label][word]).prob
	result = grammar.bylhs[0][deriv.edge.rule.no].prob
	result += getderivprob((<Entry>D[deriv.edge.left][deriv.left]).key,
			D, sent, grammar)
	if deriv.edge.right:
		result += getderivprob((<Entry>D[deriv.edge.right][deriv.right]).key,
				D, sent, grammar)
	return result


cdef double getderivprob_cfg(RankedCFGEdge deriv, list D,
		list sent, Grammar grammar):
	cdef double result
	if deriv.edge.rule is NULL:  # is terminal
		word = sent[deriv.start]
		return (<LexicalRule>grammar.lexicalbylhs[deriv.label][word]).prob
	result = grammar.bylhs[0][deriv.edge.rule.no].prob
	result += getderivprob((<Entry>D[deriv.start][deriv.edge.mid][
			deriv.edge.rule.rhs1][deriv.left]).key, D, sent, grammar)
	if deriv.edge.rule.rhs2:
		result += getderivprob((<Entry>D[deriv.edge.mid][deriv.end][
				deriv.edge.rule.rhs2][deriv.right]).key, D, sent, grammar)
	return result


cpdef viterbiderivation(chart, ChartItem start, list tolabel):
	# Ask for at least 10 derivations because unary cycles.
	derivations = lazykbest(chart, start, 10, tolabel)[0]
	return derivations[0]


cpdef str recoverfragments(deriv, D, Grammar grammar, list backtransform):
	""" Reconstruct a DOP derivation from a DOP derivation with
	flattened fragments which are left-binarized. `derivation` should be
	a RankedEdge representing a derivation, and backtransform should contain
	rule numbers as keys and strings as values. Uses the first binarized
	production as key, which map to string templates as values.

	:returns: expanded derivation as a string.

	Does on-the-fly debinarization following labels that are not mapped to a
	label in the coarse grammar, i.e., it assumes that neverblockre is only
	used to avoid blocking nonterminals from the double-dop binarization
	(containing the string '}<'). Note that this means getmapping() has to have
	been called on `grammar`, even when not doing coarse-to-fine parsing. """
	if isinstance(deriv, RankedEdge):
		result = recoverfragments_lcfrs(deriv, D, grammar, backtransform)
	elif isinstance(deriv, RankedCFGEdge):
		result = recoverfragments_cfg(deriv, D, grammar, backtransform)
	elif isinstance(deriv, basestring):
		deriv = Tree.parse(deriv, parse_leaf=int)
		result = recoverfragments_str(deriv, grammar, backtransform)
	else:
		raise ValueError
	return REMOVEWORDTAGS.sub('', result)


cdef str recoverfragments_lcfrs(RankedEdge deriv, dict D,
		Grammar grammar, list backtransform):
	cdef RankedEdge child
	cdef list children = []
	cdef str frag = backtransform[deriv.edge.rule.no]  # template

	# collect all children w/on the fly left-factored debinarization
	if deriv.edge.right.label:  # is there a right child?
		# keep going while left child is part of same binarized constituent
		# instead of looking for a binarization marker in the label string, we
		# use the fact that such labels do not have a mapping as proxy.
		while grammar.mapping[deriv.edge.left.label] == 0:
			# one of the right children
			children.append((<Entry>D[deriv.edge.right][deriv.right]).key)
			# move on to next node in this binarized constituent
			deriv = (<Entry>D[deriv.edge.left][deriv.left]).key
		# last right child
		if deriv.edge.right.label:  # is there a right child?
			children.append((<Entry>D[deriv.edge.right][deriv.right]).key)
	elif grammar.mapping[deriv.edge.left.label] == 0:
		deriv = (<Entry>D[deriv.edge.left][deriv.left]).key
	# left-most child
	children.append((<Entry>D[deriv.edge.left][deriv.left]).key)

	# recursively expand all substitution sites
	# FIXME: to avoid using str + decoding, we could use
	# PyObject* PyBytes_FromFormat(const char *format, ...)
	# PyBytes_FromFormat('(%s %d)', <char *>..., ...)
	children = [('(%s %d)' % (
		str(grammar.tolabel[child.head.label].decode('ascii')),
		child.edge.left.lexidx()))
		if child.edge.rule is NULL else recoverfragments_lcfrs(
				child, D, grammar, backtransform)
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


cdef str recoverfragments_cfg(RankedCFGEdge deriv, list D,
		Grammar grammar, list backtransform):
	cdef RankedCFGEdge child
	cdef list children = []
	cdef str frag = backtransform[deriv.edge.rule.no]  # template

	# collect children w/on the fly left-factored debinarization
	if deriv.edge.rule.rhs2:  # is there a right child?
		# keep going while left child is part of same binarized constituent
		# this shortcut assumes that neverblockre is only used to avoid
		# blocking nonterminals from the double-dop binarization.
		while grammar.mapping[deriv.edge.rule.rhs1] == 0:
			# one of the right children
			children.append((<Entry>D[deriv.edge.mid][deriv.end][
					deriv.edge.rule.rhs2][deriv.right]).key)
			# move on to next node in this binarized constituent
			deriv = (<Entry>D[deriv.start][deriv.edge.mid][
					deriv.edge.rule.rhs1][deriv.left]).key
		# last right child
		if deriv.edge.rule.rhs2:  # is there a right child?
			children.append((<Entry>D[deriv.edge.mid][deriv.end][
					deriv.edge.rule.rhs2][deriv.right]).key)
	elif grammar.mapping[deriv.edge.rule.rhs1] == 0:
		deriv = (<Entry>D[deriv.start][deriv.edge.mid][
				deriv.edge.rule.rhs1][deriv.left]).key
	# left-most child
	children.append((<Entry>D[deriv.start][deriv.edge.mid][
			deriv.edge.rule.rhs1][deriv.left]).key)

	# recursively expand all substitution sites
	children = [('(%s %d)' % (
			str(grammar.tolabel[child.label].decode('ascii')), child.start))
			if child.edge.rule is NULL
			else recoverfragments_cfg(child, D, grammar, backtransform)
			for child in reversed(children)]

	# substitute results in template
	return frag.format(*children)


cdef str recoverfragments_str(deriv, Grammar grammar, list backtransform):
	cdef list children = []
	cdef str frag
	if len(deriv) == 1:
		prod = (b'0', deriv.label, deriv[0].label)
	elif len(deriv) == 2:
		prod = (b'01', deriv.label, deriv[0].label, deriv[1].label)
	frag = backtransform[grammar.rulenos[prod]]  # template
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

	# recursively expand all substitution sites
	children = [recoverfragments_str(child, grammar, backtransform)
			if isinstance(child[0], Tree)
			else ('(%s %d)' % (child.label, child[0]))
			for child in reversed(children)]

	# substitute results in template
	return frag.format(*children)


def extractfragments(deriv, D, Grammar grammar, list backtransform):
	result = []
	if isinstance(deriv, RankedEdge):
		extractfragments_lcfrs(deriv, D, grammar, backtransform, result)
	elif isinstance(deriv, RankedCFGEdge):
		extractfragments_cfg(deriv, D, grammar, backtransform, result)
	elif isinstance(deriv, basestring):
		extractfragments_str(deriv, grammar, backtransform, result)
	else:
		raise ValueError
	return result


cdef extractfragments_lcfrs(RankedEdge deriv, dict D,
		Grammar grammar, list backtransform, list result):
	cdef RankedEdge child
	cdef list children = [], labels = []
	cdef str frag = backtransform[deriv.edge.rule.no]  # template

	# collect all children w/on the fly left-factored debinarization
	if deriv.edge.right.label:  # is there a right child?
		# keep going while left child is part of same binarized constituent
		# instead of looking for a binarization marker in the label string, we
		# use the fact that such labels do not have a mapping as proxy.
		while grammar.mapping[deriv.edge.left.label] == 0:
			# one of the right children
			children.append((<Entry>D[deriv.edge.right][deriv.right]).key)
			labels.append(grammar.tolabel[deriv.edge.right.label])
			# move on to next node in this binarized constituent
			deriv = (<Entry>D[deriv.edge.left][deriv.left]).key
		# last right child
		if deriv.edge.right.label:  # is there a right child?
			children.append((<Entry>D[deriv.edge.right][deriv.right]).key)
			labels.append(grammar.tolabel[deriv.edge.right.label])
	elif grammar.mapping[deriv.edge.left.label] == 0:
		deriv = (<Entry>D[deriv.edge.left][deriv.left]).key
	# left-most child
	children.append((<Entry>D[deriv.edge.left][deriv.left]).key)
	labels.append(grammar.tolabel[deriv.edge.left.label])

	frag = frag.format(*['(%s %d)' % (a.split('@')[0], n)
			for n, a in enumerate(reversed(labels))])
	sent = [a[a.index('@') + 1:].decode('unicode-escape') if '@' in a else None
			for a in reversed(labels)]
	result.append((frag, sent))
	# recursively visit all substitution sites
	for child in reversed(children):
		if not child.edge.rule is NULL:
			extractfragments_lcfrs(child, D, grammar, backtransform, result)


cdef extractfragments_cfg(RankedCFGEdge deriv, list D,
		Grammar grammar, list backtransform, list result):
	cdef RankedCFGEdge child
	cdef list children = [], labels = []
	cdef str frag = backtransform[deriv.edge.rule.no]  # template

	# collect all children w/on the fly left-factored debinarization
	if deriv.edge.rule.rhs2:  # is there a right child?
		# keep going while left child is part of same binarized constituent
		# instead of looking for a binarization marker in the label string, we
		# use the fact that such labels do not have a mapping as proxy.
		while grammar.mapping[deriv.edge.rule.rhs1] == 0:
			# one of the right children
			children.append((<Entry>D[deriv.edge.mid][deriv.end][
					deriv.edge.rule.rhs2][deriv.right]).key)
			labels.append(grammar.tolabel[deriv.edge.rule.rhs2])
			# move on to next node in this binarized constituent
			deriv = (<Entry>D[deriv.start][deriv.edge.mid][
					deriv.edge.rule.rhs1][deriv.left]).key
		# last right child
		if deriv.edge.rule.rhs2:  # is there a right child?
			children.append((<Entry>D[deriv.edge.mid][deriv.end][
					deriv.edge.rule.rhs2][deriv.right]).key)
			labels.append(grammar.tolabel[deriv.edge.rule.rhs2])
	elif grammar.mapping[deriv.edge.rule.rhs1] == 0:
		deriv = (<Entry>D[deriv.start][deriv.edge.mid][
				deriv.edge.rule.rhs1][deriv.left]).key
	# left-most child
	children.append((<Entry>D[deriv.start][deriv.edge.mid][
			deriv.edge.rule.rhs1][deriv.left]).key)
	labels.append(grammar.tolabel[deriv.edge.rule.rhs1])

	frag = frag.format(*['(%s %d)' % (a.split('@')[0], n)
			for n, a in enumerate(reversed(labels))])
	sent = [a[a.index('@') + 1:].decode('unicode-escape') if '@' in a else None
			for a in reversed(labels)]
	result.append((frag, sent))
	# recursively visit all substitution sites
	for child in reversed(children):
		if not child.edge.rule is NULL:
			extractfragments_cfg(child, D, grammar, backtransform, result)


cdef str extractfragments_str(deriv, Grammar grammar,
		list backtransform, list result):
	cdef list children = [], labels = []
	cdef str frag
	if len(deriv) == 1:
		prod = (b'0', deriv.label, deriv[0].label)
	elif len(deriv) == 2:
		prod = (b'01', deriv.label, deriv[0].label, deriv[1].label)
	frag = backtransform[grammar.rulenos[prod]]  # template
	# collect children w/on the fly left-factored debinarization
	if len(deriv) == 2:  # is there a right child?
		# keep going while left child is part of same binarized constituent
		# this shortcut assumes that neverblockre is only used to avoid
		# blocking nonterminals from the double-dop binarization.
		while '}<' in deriv[0].label:
			# one of the right children
			children.append(deriv[1])
			labels.append(deriv[1].label)
			# move on to next node in this binarized constituent
			deriv = deriv[0]
		# last right child
		if len(deriv) == 2:  # is there a right child?
			children.append(deriv[1])
			labels.append(deriv[1].label)
	elif '}<' in deriv[0].label:
		deriv = deriv[0]
	# left-most child
	children.append(deriv[0])
	labels.append(deriv[0].label)

	frag = frag.format(*['(%s %d)' % (a.split('@')[0], n)
			for n, a in enumerate(reversed(labels))])
	sent = [a[a.index('@') + 1:].decode('unicode-escape') if '@' in a else None
			for a in reversed(labels)]
	result.append((frag, sent))
	# recursively visit all substitution sites
	for child in reversed(children):
		if isinstance(child[0], Tree):
			extractfragments_str(child, grammar, backtransform, result)


def doprerank(chart, start, sent, n, Grammar coarse, Grammar fine):
	""" Given a chart from a coarse stage, re-rank its n-best derivations with
	DOP parse probabilities of a DOP reduction (cf. ``dopparseprob()``). """
	cdef dict results = {}
	derivations, _, _ = lazykbest(chart, start, n, coarse.tolabel,
			None, derivs=True)
	for derivstr, _ in derivations:
		deriv = addbitsets(derivstr)
		results[derivstr] = exp(dopparseprob(deriv, sent, coarse, fine))
	return results


def dopparseprob(tree, sent, Grammar coarse, Grammar fine):
	""" Given a Tree and a DOP reduction, compute the exact DOP parse
	probability.

	This follows up on a suggestion made by Goodman (2003, p. 143)
	of calculating DOP probabilities of given parse trees, although I'm not
	sure it has complexity O(nP) as he suggests (with n as number of nodes in
	input, and P as max number of rules consistent with a node in the input).
	Furthermore, the idea of sampling trees "long enough" until we have the MPP
	is no faster than sampling without applying this procedure, because to
	determine that some probability p is the maximal probability, we need to
	collect the probability mass p_seen of enough parse trees such that we have
	some parsetree with probability p > (1 - p_seen), which requires first
	seeing almost all parse trees, unless p is exceptionally high. Hence, this
	method is mostly useful in a reranking framework where it is known in
	advance that a small set of trees is of interest.

	Expects a mapping which gives a list of consistent rules from the reduction
	as produced by ``fine.getrulemapping(coarse)``.

	NB: this algorithm could also be used to determine the probability of
	derivations, but then the input would have to distinguish whether nodes are
	internal nodes of fragments, or whether they join two fragments. """
	neginf = float('-inf')
	cdef dict chart = {}  # chart[label, left, right] = prob
	cdef tuple a, b, c
	cdef Rule *rule
	cdef LexicalRule lexrule
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
				chart[lexrule.lhs, 1 << n] = logprobadd(
					chart.get((lexrule.lhs, 1 << n), neginf), -lexrule.prob)

	# do post-order traversal (bottom-up)
	for node, (prod, yf) in list(zip(tree.subtrees(),
			lcfrs_productions(tree, sent)))[::-1]:
		if not isinstance(node[0], Tree):
			continue
		yf = ','.join(''.join(map(str, a)) for a in yf)
		prod = coarse.rulenos[(yf, ) + prod]
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


def main():
	from grammar import dopreduction
	from containers import Grammar
	import plcfrs

	def e(x):
		a, b = x
		if isinstance(b, tuple):
			return (a.replace("@", ''), (int(abs(b[0])), b[1]))
		return a.replace("@", ''), b

	def maxitem(d):
		return max(d.items(), key=itemgetter(1))

	trees = [Tree.parse(t, parse_leaf=int) for t in
		"""(ROOT (A (A 0) (B 1)) (C 2))
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
		(ROOT (A 0) (C (B 1) (C 2)))""".splitlines()]
	sents = [a.split() for a in
		"""d b c\n c a b\n a e f\n a e f\n a e f\n a e f\n d b f\n d b f
		d b f\n d b g\n e f c\n e f c\n e f c\n e f c\n e f c\n e f c\n f b c
		a d e""".splitlines()]
	xgrammar, altweights = dopreduction(trees, sents)
	grammar = Grammar(xgrammar)
	grammar.register(u'shortest', altweights['shortest'])
	print(grammar)
	sent = "a b c".split()
	chart, start, _ = plcfrs.parse(sent, grammar, None, True)
	assert start
	vitderiv, vitprob = viterbiderivation(chart, start, grammar.tolabel)
	mpd, _, _ = marginalize("mpd", chart, start, grammar, 1000)
	mpp, _, _ = marginalize("mpp", chart, start, grammar, 1000)
	mppsampled, _, _ = marginalize("mpp", chart, start, grammar, 1000,
			sample=True, kbest=False)
	sldop1, _, _ = marginalize("sl-dop", chart, start, grammar, 1000,
			sldop_n=7, sent=sent)
	sldopsimple, _, _ = marginalize("sl-dop-simple", chart, start, grammar,
			1000, sldop_n=7, sent=sent)
	short, _, _ = marginalize("shortest", chart, start, grammar,
		1000, sent=sent)
	print("\nvit:\t\t%s %r" % e((REMOVEIDS.sub('', vitderiv), exp(-vitprob))),
		"MPD:\t\t%s %r" % e(maxitem(mpd)),
		"MPP:\t\t%s %r" % e(maxitem(mpp)),
		"MPP sampled:\t%s %r" % e(maxitem(mppsampled)),
		"SL-DOP n=7:\t%s %r" % e(maxitem(sldop1)),
		"simple SL-DOP:\t%s %r" % e(maxitem(sldopsimple)),
		"shortest:\t%s %r" % e(maxitem(short)), sep='\n')

if __name__ == '__main__':
	main()
