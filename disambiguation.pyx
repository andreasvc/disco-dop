""" Extract parse tree(s) from a chart following a particular objective
function. """

from __future__ import print_function
import re, logging
from heapq import nlargest
from math import fsum, exp, log
from random import random
from bisect import bisect_right
from operator import itemgetter, attrgetter
from itertools import count
from collections import defaultdict, OrderedDict
from tree import Tree
from kbest import lazykbest, getderiv
import plcfrs
from agenda cimport Entry, new_Entry
from grammar import induce_plcfrs, rangeheads
from treetransforms import unbinarize #, canonicalize
from containers cimport ChartItem, SmallChartItem, FatChartItem, CFGChartItem, \
		Edge, LCFRSEdge, CFGEdge, RankedEdge, RankedCFGEdge, Grammar, \
		UChar, UInt, ULong, ULLong
cimport cython

from libc.string cimport memset
cdef extern from "macros.h":
	void SETBIT(ULong a[], int b)

infinity = float('infinity')
removeids = re.compile("@[-0-9]+")
removewordtags = re.compile("@[^ )]+")
termsre = re.compile(" ([0-9]+)\\b")

cpdef marginalize(method, chart, ChartItem start, Grammar grammar, int n,
		bint sample=False, bint kbest=True, list sent=None, list tags=None,
		secondarymodel=None, int sldop_n=7, dict backtransform=None):
	""" approximate MPP or MPD by summing over n random/best derivations from
	chart, return a dictionary mapping parsetrees to probabilities """
	cdef bint mpd = method == "mpd"
	cdef bint shortest = method == "shortest"
	cdef Entry entry
	cdef dict parsetrees = <dict>defaultdict(float)
	cdef list derivations = [], entries = []
	cdef bytes treestr, deriv
	cdef double prob, maxprob
	cdef int m

	assert kbest or sample
	if kbest:
		derivations, D, _ = lazykbest(chart, start, n, grammar.tolabel,
				None, derivs=backtransform is None)
		if isinstance(start, CFGChartItem):
			entries = D[(<CFGChartItem>start).start][
					(<CFGChartItem>start).end][start.label]
		else:
			entries = D[start]
	else:
		if isinstance(start, CFGChartItem):
			D = [[{} for right in left] for left in chart]
		else:
			D = {}
	if sample:
		assert not isinstance(start, CFGChartItem), (
				"sampling not implemented for PCFG charts.")
		derivations.extend(
				getsamples(D, chart, start, n, grammar.tolabel, None))
		# filter out duplicate derivations
		filteredderivations = dict(zip(derivations, D[start]))
		entries[:] = filteredderivations.values()
		derivations[:] = filteredderivations.keys()

	if method == "sl-dop":
		assert not isinstance(start, CFGChartItem), (
				"sl-dop not implemented for PCFG charts.")
		return sldop(dict(derivations), chart, start, sent, tags, grammar,
				secondarymodel, n, sldop_n, backtransform, D, entries)
	elif method == "sl-dop-simple":
		return sldop_simple(dict(derivations), entries, n, sldop_n,
				D, chart, grammar, backtransform)
	elif method == "shortest":
		# filter out all derivations which are not shortest
		if backtransform is not None:
			maxprob = min([entry.value for entry in entries])
			entries = [entry for entry in entries if entry.value == maxprob]
		elif derivations:
			_, maxprob = min(derivations, key=itemgetter(1))
			derivations = [(a, b) for a, b in derivations if b == maxprob]

	if backtransform is not None:
		for entry in entries:
			prob = entry.value
			treestr = recoverfragments(entry.key, D,
					grammar, backtransform)
			if mpd:
				if exp(-prob) > parsetrees[treestr]:
					parsetrees[treestr] = exp(-prob)
			elif shortest:
				deriv = getderiv(entry.key, D, chart,
						grammar.tolabel, None)
				tree = Tree.parse(deriv, parse_leaf=int)
				newprob = exp(sum([secondarymodel.get(r, 0.0) for r, _
					in induce_plcfrs([tree], [sent])]))
				score = (prob / log(0.5), newprob)
				if score > parsetrees[treestr]:
					parsetrees[treestr] = score
			else:
				# simple way of adding probabilities (too easy):
				parsetrees[treestr] += exp(-prob)
				#if treestr in parsetrees:
				#	parsetrees[treestr].append(-prob)
				#else:
				#	parsetrees[treestr] = [-prob]
	else: #DOP reduction
		for deriv, prob in derivations:
			if shortest:
				# for purposes of tie breaking, calculate the derivation
				# probability in a different model. because we don't keep track
				# of which rules have been used, read off the rules from the
				# derivation ...
				tree = Tree.parse(deriv, parse_leaf=int)
				newprob = exp(sum([secondarymodel.get(r, 0.0) for r, _
					in induce_plcfrs([tree], [sent])]))
				if backtransform is not None:
					# tie breaking relies on binarized productions,
					# to recover derivation we need to unbinarize
					deriv = unbinarize(tree, childchar="}")
			treestr = removeids.sub("@" if mpd else "", deriv)
			if shortest:
				score = (prob / log(0.5), newprob)
				if score > parsetrees[treestr]:
					parsetrees[treestr] = score
			elif mpd and backtransform is not None:
				if exp(-prob) > parsetrees[treestr]:
					parsetrees[treestr] = exp(-prob)
			else:
				# simple way of adding probabilities (too easy):
				parsetrees[treestr] += exp(-prob)
				#if treestr in parsetrees:
				#	parsetrees[treestr].append(-prob)
				#else:
				#	parsetrees[treestr] = [-prob]

	# Adding probabilities in log space
	# http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point-numbers
	# https://facwiki.cs.byu.edu/nlp/index.php/Log_Domain_Computations
	#for parsetree in parsetrees:
	#	maxprob = max(parsetrees[parsetree])
	#	parsetrees[parsetree] = exp(fsum([maxprob, log(fsum([exp(prob - maxprob)
	#								for prob in parsetrees[parsetree]]))]))
	msg = "%d derivations, %d parsetrees" % (
			len(derivations if backtransform is None else entries),
			len(parsetrees))
	return parsetrees, msg

cdef sldop(dict derivations, chart, ChartItem start, list sent, list tags,
		Grammar dopgrammar, Grammar secondarymodel, int m, int sldop_n,
		dict backtransform, D, entries):
	""" `proper' method for sl-dop. parses sentence once more to find shortest
	derivations, pruning away any chart item not occurring in the n most
	probable parse trees. Returns the first result of the intersection of the
	most probable parse trees and shortest derivations.
	NB: doesn't seem to work so well, so may contain a subtle bug.
		should be rewritten to support PCFG charts, double-dop, etc.
	NB2: assumes ROOT is the top node."""
	cdef ChartItem item
	cdef FatChartItem fitem
	# collect derivations for each parse tree
	derivsfortree = defaultdict(set)
	if backtransform is None:
		for deriv in derivations:
			derivsfortree[removeids.sub("", deriv)].add(deriv)
	else:
		for entry in entries:
			deriv = getderiv(entry.key, D, chart, dopgrammar.tolabel, None)
			derivations[deriv] = entry.value
			derivsfortree[recoverfragments(entry.key, D,
					dopgrammar.tolabel, backtransform)].add(deriv)
	# sum over derivations to get parse trees
	parsetreeprob = {}
	for tree, derivs in derivsfortree.items():
		parsetreeprob[tree] = sum([exp(-derivations[d]) for d in derivs])

	# use getmapping and prunechart here instead of manually built whitelist
	# prunechart(chart, n, ...)
	# parse ...
	# OR: use two coarse-to-fine stages, the second collects SL-DOP parse
	whitelist = [{} for a in secondarymodel.toid]
	for a in chart:
		whitelist[(<ChartItem>a).label][a] = infinity
	for tt in nlargest(sldop_n, parsetreeprob, key=parsetreeprob.get):
		for n in Tree(tt).subtrees():
			if len(sent) < sizeof(ULLong) * 8:
				item = SmallChartItem(0, sum([1L << int(x)
						for x in n.leaves()]))
			else:
				fitem = item = FatChartItem(0)
				memset(<char *>fitem.vec, 0, sizeof(fitem.vec))
				for x in n.leaves():
					SETBIT(fitem.vec, x)
			whitelist[secondarymodel.toid[n.label]][item] = 0.0
	for label, n in secondarymodel.toid.items():
		whitelist[n] = whitelist[secondarymodel.toid[label.split("@")[0]]]
	mpp2 = {}
	for tt in nlargest(sldop_n, parsetreeprob, key=parsetreeprob.get):
		mpp2[tt] = parsetreeprob[tt]

	chart2, start2, _ = plcfrs.parse(sent, secondarymodel, tags=tags,
					exhaustive=True, estimates=None, whitelist=whitelist)
	if start2:
		shortestderivations, DD, _ = lazykbest(chart2, start2, m,
			secondarymodel.tolabel)
	else:
		shortestderivations = []
		logging.warning("shortest derivation parsing failed") # error?
	result = dict([max(mpp2.items(), key=itemgetter(1))])

	for (deriv, s), entry in zip(shortestderivations, DD[start]):
		if backtransform is None:
			tt = removeids.sub("", deriv)
		else:
			tt = recoverfragments(entry.key, D,
					dopgrammar.tolabel, backtransform)
		if tt in mpp2:
			result = dict([(tt, (s / log(0.5), mpp2[tt]))])
			break
	else:
		logging.warning("no matching derivation found") # error?
	msg = "(%d derivations, %d of %d parsetrees)" % (
		len(derivations), min(sldop_n, len(parsetreeprob)), len(parsetreeprob))
	return result, msg

cdef sldop_simple(dict derivations, list entries, int m, int sldop_n,
		D, chart, Grammar grammar, dict backtransform):
	""" simple sl-dop method; estimates shortest derivation directly from
	number of addressed nodes in the k-best derivations. After selecting the n
	best parse trees, the one with the shortest derivation is returned."""
	cdef Entry entry
	derivsfortree = defaultdict(set)
	# collect derivations for each parse tree
	if backtransform is None:
		for deriv in derivations:
			tree = removeids.sub("", deriv)
			derivsfortree[tree].add(deriv)
	else:
		for entry in entries:
			deriv = getderiv(entry.key, D, chart, grammar.tolabel, "}<")
			tree = recoverfragments(entry.key, D, grammar, backtransform)
			derivations[deriv] = entry.value
			derivsfortree[tree].add(deriv)

	# sum over derivations to get parse trees
	parsetreeprob = {}
	for tree, derivs in derivsfortree.items():
		parsetreeprob[tree] = sum([exp(-derivations[d]) for d in derivs])
	selectedtrees = nlargest(sldop_n, parsetreeprob, key=parsetreeprob.get)

	# the number of fragments used is the number of
	# nodes (open parens), minus the number of interior
	# (addressed) nodes.
	result = {tree: (-min([deriv.count("(") - (
			deriv.count("@") + deriv.count("(#"))
		for deriv in derivsfortree[tree]]), parsetreeprob[tree])
				for tree in selectedtrees}
	msg = "(%d derivations, %d of %d parsetrees)" % (
			len(derivations), len(result), len(parsetreeprob))
	return result, msg

cdef samplechart(dict D, dict chart, ChartItem start, list tolabel, dict tables,
		bytes debin):
	""" Samples a derivation from a chart. """
	cdef LCFRSEdge edge
	cdef ChartItem child
	cdef double prob
	lst = tables[start]
	rnd = random() * lst[len(lst) - 1]
	idx = bisect_right(lst, rnd)
	edge = chart[start][idx]
	if edge.left.label == 0: # == "Epsilon":
		idx = edge.left.lexidx()
		newedge = RankedEdge(start, edge, 0, -1)
		D.setdefault(start, []).append(new_Entry(newedge, edge.inside, 0))
		return "(%s %d)" % (tolabel[start.label], idx), edge.inside
	children = [samplechart(D, chart, child, tolabel, tables, debin)
		for child in (edge.left, edge.right) if child.label]
	if debin is not None and debin in tolabel[start.label]:
		tree = " ".join([a for a, _ in children])
	else:
		tree = "(%s %s)" % (tolabel[start.label],
				" ".join([a for a, _ in children]))
	# create an edge that has as children the edges that were just created
	# by our recursive call
	newedge = RankedEdge(start, edge, len(D[edge.left]) - 1,
			(len(D[edge.right]) - 1) if edge.right.label else -1)
	prob = edge.rule.prob + sum([b for _, b in children])
	D.setdefault(start, []).append(new_Entry(newedge, prob, 0))
	return tree, prob

def getsamples(D, chart, start, n, tolabel, debin=None):
	""" Samples n derivations from a chart. """
	cdef Edge edge
	cdef dict tables = {}, chartcopy = {}
	for item in chart:
		#FIXME: work w/inside prob right?
		#chart[item].sort(key=attrgetter('prob'))
		# sort so that highest probability edges come first
		chartcopy[item] = sorted(chart[item])
		tables[item] = []
		prev = 0.0
		for edge in chartcopy[item]:
			prev += exp(-edge.inside)
			tables[item].append(prev)
	return [samplechart(<dict>D, chartcopy, start, tolabel, tables, debin)
						for _ in range(n)]

cpdef viterbiderivation(chart, ChartItem start, list tolabel):
	# Ask for at least 10 derivations because unary cycles.
	derivations = lazykbest(chart, start, 10, tolabel)[0]
	return derivations[0]

cpdef bytes recoverfragments(derivation, D, Grammar grammar,
		dict backtransform):
	""" Reconstruct a DOP derivation from a DOP derivation with
	flattened fragments which are left-binarized. `derivation' should be
	a RankedEdge representing a derivation, and backtransform should contain
	rule numbers as keys and strings as values. Uses the first binarized
	production as key, which map to string templates as values. Returns
	expanded derivation as a string.

	Does on-the-fly debinarization following labels that are not mapped to a
	label in the coarse grammar, i.e., it assumes that neverblockre is only
	used to avoid blocking nonterminals from the double-dop binarization
	(containing the string '}<'). Note that this means getmapping() has to have
	been called on `grammar', even when not doing coarse-to-fine parsing. """
	if isinstance(derivation, RankedEdge):
		return removewordtags.sub("", recoverfragments_lcfrs(
				derivation, D, grammar, backtransform))
	elif isinstance(derivation, RankedCFGEdge):
		return removewordtags.sub("", recoverfragments_cfg(
				derivation, D, grammar, backtransform))
	raise ValueError("derivation should be RankedEdge or RankedCFGEdge.")

cdef bytes recoverfragments_lcfrs(RankedEdge derivation, dict D,
		Grammar grammar, dict backtransform):
	cdef RankedEdge child
	cdef LCFRSEdge childedge, derivedge = derivation.edge
	cdef list children = []

	# get fragment
	result = backtransform[(<LCFRSEdge>derivation.edge).rule.no]

	# recursively expand all substitution sites,
	# w/on the fly left-factored debinarization
	if derivedge.right.label: # is there a right child?
		# keep going while left child is part of same binarized constituent
		# instead of looking for a binarization marker in the label string, we
		# use the fact that such labels do not have a mapping as proxy.
		while grammar.mapping[derivedge.left.label] == 0:
			# one of the right children
			child = (<Entry>D[derivedge.right][derivation.right]).key
			childedge = child.edge
			children.append(('(%s %d)' % (
				grammar.tolabel[child.head.label], childedge.left.lexidx()))
				if childedge.rule is NULL else recoverfragments_lcfrs(
						child, D, grammar, backtransform))
			# move on to next node in this binarized constituent
			derivation = (<Entry>D[derivedge.left][derivation.left]).key
			derivedge = derivation.edge
		# last right child
		if derivedge.right.label: # is there a right child?
			child = (<Entry>D[derivedge.right][derivation.right]).key
			childedge = child.edge
			children.append('(%s %d)' % (
				grammar.tolabel[child.head.label], childedge.left.lexidx())
				if childedge.rule is NULL else recoverfragments_lcfrs(
						child, D, grammar, backtransform))
	elif grammar.mapping[derivedge.left.label] == 0:
		derivation = (<Entry>D[derivedge.left][derivation.left]).key
		derivedge = derivation.edge
	# left-most child
	child = (<Entry>D[derivedge.left][derivation.left]).key
	childedge = child.edge
	children.append(('(%s %d)' % (
		grammar.tolabel[child.head.label], childedge.left.lexidx()))
		if childedge.rule is NULL else recoverfragments_lcfrs(
				child, D, grammar, backtransform))
	children.reverse()
	return result.format(*children)

cdef bytes recoverfragments_cfg(RankedCFGEdge derivation, list D,
		Grammar grammar, dict backtransform):
	cdef RankedCFGEdge child
	cdef CFGEdge childedge, derivedge = derivation.edge
	cdef list children = []

	# get fragment
	result = backtransform[(<CFGEdge>derivation.edge).rule.no]

	# recursively expand all substitution sites,
	# w/on the fly left-factored debinarization
	if derivedge.rule.rhs2: # is there a right child?
		# keep going while left child is part of same binarized constituent
		# this shortcut assumes that neverblockre is only used to avoid
		# blocking nonterminals from the double-dop binarization.
		while grammar.mapping[derivedge.rule.rhs1] == 0:
			# one of the right children
			child = (<Entry>D[derivedge.mid][derivation.end][
					derivedge.rule.rhs2][derivation.right]).key
			childedge = child.edge
			children.append(('(%s %d)' % (grammar.tolabel[child.label],
				child.start)) if childedge.rule is NULL else
				recoverfragments_cfg(child, D, grammar, backtransform))
			# move on to next node in this binarized constituent
			derivation = (<Entry>D[derivation.start][derivedge.mid][
					derivedge.rule.rhs1][derivation.left]).key
			derivedge = derivation.edge
		# last right child
		if derivedge.rule.rhs2: # is there a right child?
			child = (<Entry>D[derivedge.mid][derivation.end][
					derivedge.rule.rhs2][derivation.right]).key
			childedge = child.edge
			children.append(('(%s %d)' % (grammar.tolabel[child.label],
				child.start)) if childedge.rule is NULL else
				recoverfragments_cfg(child, D, grammar, backtransform))
	elif grammar.mapping[derivedge.rule.rhs1] == 0:
		derivation = (<Entry>D[derivation.start][derivedge.mid][
				derivedge.rule.rhs1][derivation.left]).key
		derivedge = derivation.edge
	# left-most child
	child = (<Entry>D[derivation.start][derivedge.mid][
			derivedge.rule.rhs1][derivation.left]).key
	childedge = child.edge
	children.append(('(%s %d)' % (grammar.tolabel[child.label], child.start))
		if childedge.rule is NULL else recoverfragments_cfg(
				child, D, grammar, backtransform))
	children.reverse()
	return result.format(*children)

def main():
	from grammar import dopreduction
	from containers import Grammar
	import plcfrs
	def e(x):
		if isinstance(x[1], tuple):
			return x[0].replace("@", ""), (int(abs(x[1][0])), x[1][1])
		return x[0].replace("@", ""), x[1]
	def f(x):
		print("foo", x)
		return x[0], exp(-x[1])
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
	grammar = Grammar(dopreduction(trees, sents))
	shortest, secondarymodel = dopreduction(trees, sents, shortestderiv=True)
	shortest = Grammar(shortest)
	sent = "a b c".split()
	chart, start, _ = plcfrs.parse(sent, grammar, None, True)
	assert start
	vit = viterbiderivation(chart, start, grammar.tolabel)
	mpd, _ = marginalize("mpd", chart, start, grammar, 1000)
	mpp, _ = marginalize("mpp", chart, start, grammar, 1000)
	mppsampled, _ = marginalize("mpp", chart, start, grammar, 1000,
			sample=True, kbest=False)
	sldop1, _ = marginalize("sl-dop", chart, start, grammar, 1000,
			sldop_n=7, sent=sent, secondarymodel=shortest)
	sldopsimple, _ = marginalize("sl-dop-simple", chart, start, grammar, 1000,
			sldop_n=7, sent=sent)
	short, _ = marginalize("shortest", chart, start, shortest,
		1000, sent=sent, secondarymodel=secondarymodel)
	print("\nvit:\t\t%s %r" % e((removeids.sub("", vit[0]), exp(-vit[1]))),
		"MPD:\t\t%s %r" % e(maxitem(mpd)),
		"MPP:\t\t%s %r" % e(maxitem(mpp)),
		"MPP sampled:\t%s %r" % e(maxitem(mppsampled)),
		"SL-DOP n=7:\t%s %r" % e(maxitem(sldop1)),
		"simple SL-DOP:\t%s %r" % e(maxitem(sldopsimple)),
		#"shortest:\t%s %r" % f(min(short.items())),
		"shortest:\t%s %r" % e(maxitem(short)), sep='\n')

if __name__ == '__main__':
	main()
