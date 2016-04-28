"""Project selected items from a chart to corresponding items in next grammar.
"""
from __future__ import print_function
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from .tree import Tree
from .treetransforms import mergediscnodes, unbinarize, fanout, addbitsets
from .containers cimport Grammar, Chart, ChartItem, Edge, Edges, MoreEdges, \
		LexicalRule, RankedEdge, cellidx, compactcellidx, \
		CFGtoSmallChartItem, CFGtoFatChartItem
from .kbest import lazykbest
import numpy as np

include "constants.pxi"

# alternative: take coarse chart, return fine chart w/whitelist.
# cpdef Chart prunechart(coarsechart, Grammar fine, int k,
# 		bint splitprune, bint markorigin, bint finecfg, bint bitpar):


def prunechart(Chart coarsechart, Grammar fine, k,
		bint splitprune, bint markorigin, bint finecfg, bint bitpar):
	"""Produce a white list of selected chart items.

	The criterion is that they occur in the `k`-best derivations of ``chart``,
	or with posterior probability > `k`. Labels ``X`` in ``coarse.toid`` are
	projected to the labels in the mapping of the fine grammar, e.g., to ``X``
	and ``X@n-m`` for a DOP reduction.

	:param coarsechart: a Chart object produced by the PCFG or PLCFRS parser,
		or derivations from bitpar.
	:param fine: the grammar to map labels to after pruning. must have a
		mapping to the coarse grammar established by ``fine.getmapping()``.
	:param k: when ``k >= 1``: number of `k`-best derivations to consider;
		when ``k==0``, the chart is not pruned but filtered to contain only
		items that contribute to a complete derivation;
		when ``0 < k < 1``, inside-outside probabilities are computed and items
		with a posterior probabilities < `k` are pruned.
	:param splitprune: coarse stage used a split-PCFG where discontinuous node
		appear as multiple CFG nodes. Every discontinuous node will result
		in multiple lookups into whitelist to see whether it should be
		allowed on the agenda.
	:param markorigin: in combination with splitprune, coarse labels include an
		integer to distinguish components; e.g., CFG nodes NP*0 and NP*1
		map to the discontinuous node NP_2.
	:param bitpar: prune from bitpar derivations instead of actual chart
	:returns: ``(whitelist, items, msg)``

	For LCFRS, the white list is indexed as follows:
			:whitelisted: ``whitelist[label][item] == None``
			:blocked: ``item not in whitelist[label]``

	For a CFG, indexing is as follows:
		:whitelisted: ``whitelist[span][label] == None``
		:blocked: ``label not in whitelist[cell]``
	"""
	cdef set items
	cdef list whitelist
	cdef ChartItem chartitem
	if fine.mapping is NULL:
		raise ValueError('need to call fine.getmapping(coarse, ...).')
	if splitprune and markorigin:
		if fine.splitmapping is NULL or fine.splitmapping[0] is NULL:
			raise ValueError('need to call fine.getmapping(coarse, ...).')
	# prune coarse chart and collect items
	if 0 < k < 1:  # threshold on posterior probabilities
		items, msg = posteriorthreshold(coarsechart, k)
	else:  # construct a list of the k-best nonterminals to prune with
		if bitpar:
			items = bitparkbestitems(coarsechart, k, finecfg)
		elif k == 0:
			coarsechart.filter()
			items = set(coarsechart.getitems())
		else:
			_ = lazykbest(coarsechart, k, derivs=False)
			items = set(coarsechart.rankededges)
		msg = ('coarse items before pruning: %d; after: %d, '
				'based on %d/%d derivations' % (
				len(coarsechart.getitems()), len(items),
				len(coarsechart.rankededges[coarsechart.root()][:k]), k))
	# project items to fine grammar
	if finecfg:
		cfgwhitelist = [set() for _ in range(compactcellidx(
				coarsechart.lensent - 1, coarsechart.lensent,
				coarsechart.lensent, 1) + 1)]
		for item in items:
			label = coarsechart.label(item)
			finespan = coarsechart.asCFGspan(item, fine.nonterminals)
			for finelabel in range(1, fine.nonterminals):
				if (fine.mapping[finelabel] == 0
						or fine.mapping[finelabel] == label):
					cfgwhitelist[finespan].add(finelabel)
		return cfgwhitelist, None, msg
	else:
		whitelist = [None] * fine.nonterminals
		kbestspans = [set() for _ in coarsechart.grammar.toid]
		kbestspans[0] = None
		# uses ids of labels in coarse chart
		for item in items:
			# we can use coarsechart here because we only use the item to
			# define a span, which is the same for the fine chart.
			chartitem = coarsechart.asChartItem(item)
			label = chartitem.label
			chartitem.label = 0
			kbestspans[label].add(chartitem)
		# now construct a list which references these coarse items:
		for label in range(fine.nonterminals):
			if splitprune and markorigin and fine.fanout[label] != 1:
				if fine.splitmapping[label] is not NULL:
					whitelist[label] = [kbestspans[fine.splitmapping[label][n]]
							for n in range(fine.fanout[label])]
			else:
				if fine.mapping[label] != 0:
					whitelist[label] = kbestspans[fine.mapping[label]]
		return whitelist, items, msg


def bitparkbestitems(Chart chart, int k, bint finecfg):
	"""Produce ChartItems occurring in a dictionary of derivations.

	:param chart: a chart where rankededges is a dictionary of CFG derivations.
	:returns: a dictionary of ChartItems (mapping to None) occurring in the
		derivations."""
	cdef bint fatitems = chart.lensent >= (sizeof(uint64_t) * 8)
	cdef list derivs = chart.rankededges[chart.root()]
	cdef set items = set()
	for deriv, _ in derivs[:k]:
		t = Tree(deriv)
		for n in t.subtrees():
			label = chart.grammar.toid[n.label]
			leaves = n.leaves()
			start = min(leaves)
			end = max(leaves) + 1
			if finecfg:
				# create a cellidx to the coarse chart here,
				# only to convert it to a cellidx for the fine chart later...
				item = cellidx(start, end, chart.lensent,
						chart.grammar.nonterminals) + label
			elif fatitems:
				item = CFGtoFatChartItem(label, start, end)
			else:
				item = CFGtoSmallChartItem(label, start, end)
			items.add(item)
	return items


def posteriorthreshold(Chart chart, double threshold):
	"""Prune labeled spans from chart below given posterior threshold.

	:returns: dictionary of remaining items."""
	if not 0 < threshold < 1:
		raise ValueError('probability threshold should be between 0 and 1.')
	if not chart.itemsinorder:
		raise ValueError('need list of chart items in topological order.')
	origlogprob = chart.grammar.logprob
	chart.grammar.switch(
			chart.grammar.modelnames[chart.grammar.currentmodel],
			logprob=False)
	getinside(chart)
	getoutside(chart)
	chart.grammar.switch(
			chart.grammar.modelnames[chart.grammar.currentmodel],
			logprob=origlogprob)
	sentprob = chart.inside[chart.root()]
	threshold *= sentprob
	if not sentprob:
		raise ValueError('sentence has zero posterior prob.: %g' % sentprob)
	posterior = {item for item in chart.getitems()
			if chart.inside[item] * chart.outside[item] > threshold}

	unfiltered = len(chart.getitems())
	numitems = len(chart.outside)
	numremain = len(posterior)
	msg = ('coarse items before pruning=%d; filtered: %d;'
			' pruned: %d; sentprob=%g' % (
			unfiltered, numitems, numremain, sentprob))
	return posterior, msg


def getinside(Chart chart):
	"""Compute inside probabilities for a chart given its parse forest."""
	cdef size_t n
	cdef Edge *edge
	cdef Edges edges
	cdef MoreEdges *edgelist
	# this needs to be bottom up, so need order in which items were added
	# currently separate list, chart.itemsinorder
	# NB: sorting items by length is not enough,
	# unaries have to be in the right order...

	# choices for probs:
	# - normal => underflow (current)
	# - logprobs => loss of precision w/addition
	# - normal, scaled => how?

	# packing parse forest:
	# revitems = {item: n for n, item in enumerate(self.itemsinorder)}
	# now self.inside[n] and self.outside[n] can be double arrays.
	chart.inside = dict.fromkeys(chart.getitems(), 0.0)

	# traverse items in bottom-up order
	for item in chart.itemsinorder:
		edges = chart.getedges(item)
		edgelist = edges.head if edges is not None else NULL
		while edgelist is not NULL:
			for n in range(edges.len if edgelist is edges.head
					else EDGES_SIZE):
				edge = &(edgelist.data[n])
				if edge.rule is NULL:
					label = chart.label(item)
					word = chart.sent[chart.lexidx(edge)]
					prob = (<LexicalRule>chart.grammar.lexicalbylhs[
							label].get(word, 1.0)).prob
				elif edge.rule.rhs2 == 0:
					leftitem = chart._left(item, edge)
					prob = (edge.rule.prob
							* chart.inside[leftitem])
				else:
					leftitem = chart._left(item, edge)
					rightitem = chart._right(item, edge)
					prob = (edge.rule.prob
							* chart.inside[leftitem]
							* chart.inside[rightitem])
				# chart.addprob(item, prob)
				chart.inside[item] += prob
			edgelist = edgelist.prev


def getoutside(Chart chart):
	"""Compute outside probabilities for a chart given its parse forest."""
	cdef size_t n
	cdef Edge *edge
	cdef Edges edges
	cdef MoreEdges *edgelist
	cdef double outsideprob
	# cdef double sentprob = chart.inside[chart.root()]
	# traverse items in top-down order
	# could use list with idx of item in itemsinorder
	chart.outside = dict.fromkeys(chart.getitems(), 0.0)
	chart.outside[chart.root()] = 1.0
	for item in reversed(chart.itemsinorder):
		# can we define outside[item] simply as sentprob - inside[item] ?
		# chart.outside[item] = sentprob - chart.inside[item]
		edges = chart.getedges(item)
		edgelist = edges.head if edges is not None else NULL
		while edgelist is not NULL:
			for n in range(edges.len if edgelist is edges.head
					else EDGES_SIZE):
				edge = &(edgelist.data[n])
				if edge.rule is NULL:
					pass
				elif edge.rule.rhs2 == 0:
					leftitem = chart._left(item, edge)
					chart.outside[leftitem] += (edge.rule.prob
							* chart.outside[item])
				else:
					leftitem = chart._left(item, edge)
					rightitem = chart._right(item, edge)
					outsideprob = chart.outside[item]
					chart.outside[leftitem] += (edge.rule.prob
							* chart.inside[rightitem]
							* outsideprob)
					chart.outside[rightitem] += (edge.rule.prob
							* chart.inside[leftitem]
							* outsideprob)
			edgelist = edgelist.prev


def doctftest(coarse, fine, sent, tree, k, split, verbose=False):
	"""Test coarse-to-fine methods on a sentence."""
	from . import plcfrs
	from .disambiguation import getderivations, marginalize
	from .treetransforms import canonicalize, removefanoutmarkers
	from math import exp as pyexp
	sent, tags = zip(*sent)
	print(' C O A R S E ')
	chart, _ = plcfrs.parse(sent, coarse, tags=tags)
	if chart:
		derivations, entries = getderivations(chart, 10, True, False, True)
		mpp, _ = marginalize("mpp", derivations, entries, chart)
		for t, p, _ in mpp:
			print(pyexp(-p), end=' ')
			t = Tree.parse(t)
			if split:
				unbinarize(t, childchar=":", parentchar="!")
				mergediscnodes(t)
			unbinarize(t)
			t = canonicalize(removefanoutmarkers(t))
			print("exact match" if t == canonicalize(tree) else "no match")
	else:
		print("no parse")
		return
		# print(chart)
	l, _, _ = prunechart(chart, fine, k, split, True, False, False)
	if verbose:
		print("\nitems in 50-best of coarse chart")
		_ = lazykbest(chart, k, derivs=False)
		for a in chart.rankededges:
			print(coarse.tolabel[(<ChartItem>a).label],
					bin((<ChartItem>a).vec))
		print("\nwhitelist:")
		for n, x in enumerate(l):
			if isinstance(x, dict):
				print(fine.tolabel[n], map(bin, x))
			elif x:
				for m, y in enumerate(x):
					print(fine.tolabel[n], m, map(bin, y))
	print(' F I N E ')
	chart2, _ = plcfrs.parse(sent, fine, tags=tags, whitelist=l,
			splitprune=split, markorigin=True)
	if chart2:
		derivations, entries = getderivations(chart2, 10, True, False, True)
		mpp, _ = marginalize("mpp", derivations, entries, chart2)
		for t, p, _ in mpp:
			print(pyexp(-p), end=' ')
			t = Tree.parse(t)
			unbinarize(t)
			t = canonicalize(removefanoutmarkers(t))
			# print(t)
			print("exact match" if t == canonicalize(tree) else "no match", t)
	else:
		print("no parse. problem.")
		return
		# if verbose:
		# xp = set((coarse.tolabel[a.label], a.vec) for a in chart.keys()
		# 		if chart[a])
		# xpp = set((fine.tolabel[a.label], a.vec) for a in chart2.keys()
		# 		if chart2[a])
		# print("difference:")
		# for a in xp - xpp:
		# 	if "*" not in a[0]:
		# 		print(a[0], bin(a[1]))
		# print("\nfine chart:")
		# for a in xpp:
		# 	if "@" not in a[0]:
		# 		print(a[0], bin(a[1]))
		# print(chart2)


def test():
	import re
	from time import clock
	from .treetransforms import splitdiscnodes, binarize, addfanoutmarkers
	from .treebank import NegraCorpusReader
	from .grammar import treebankgrammar, dopreduction, subsetgrammar
	k = 50
	# corpus = NegraCorpusReader("toytb.export", encoding="iso-8859-1")
	# corpus = NegraCorpusReader("negraproc.export",
	# 	encoding="utf-8", headrules="negra.headrules", headfinal=True,
	#    headreverse=False)
	# train = 400
	# test = 40
	# testmaxlen = 999
	corpus = NegraCorpusReader('alpinosample.export')
	train = 0
	test = 3
	testmaxlen = 999
	# trees = corpus.trees().values()[:train]
	# sents = corpus.sents().values()[:train]
	trees = list(corpus.trees().values())
	sents = list(corpus.sents().values())

	dtrees = [t.copy(True) for t in trees]
	parenttrees = [t.copy(True) for t in trees]
	for t in trees:
		binarize(t, vertmarkov=1, horzmarkov=1)
		addfanoutmarkers(t)
	cftrees = [splitdiscnodes(t.copy(True), markorigin=True) for t in trees]
	for t in cftrees:
		binarize(t, horzmarkov=2, tailmarker='', leftmostunary=True,
			childchar=":")  # NB leftmostunary is important
		addfanoutmarkers(t)
	for t in parenttrees:
		binarize(t, vertmarkov=3, horzmarkov=1)
		addfanoutmarkers(t)
	for t in dtrees:
		binarize(t, vertmarkov=1, horzmarkov=1)
		addfanoutmarkers(t)
	normallcfrs = treebankgrammar(trees, sents)
	normal = Grammar(normallcfrs)
	parent = Grammar(treebankgrammar(parenttrees, sents))
	splitg = Grammar(treebankgrammar(cftrees, sents))
	for t, s in zip(cftrees, sents):
		for (_, yf), _ in treebankgrammar([t], [s]):
			assert len(yf) == 1
	fine999x = dopreduction(trees, sents)[0]
	fine999 = Grammar(fine999x)
	fine1 = Grammar(dopreduction(dtrees, sents)[0])
	trees = list(corpus.trees().values())[train:train + test]
	sents = list(corpus.tagged_sents().values())[train:train + test]
	if subsetgrammar(normallcfrs, fine999x):
		print("DOP grammar is a superset")
	else:
		print("DOP grammar is NOT a superset!")
	for msg, coarse, fine, split, enabled in zip(
			('normal', 'parentannot', 'cf-split'),
			(normal, parent, splitg),
			(fine999, fine1, fine1),
			(False, False, True),
			(True, False, True)):
		if not enabled:
			continue
		print("coarse grammar:", msg)
		fine.getmapping(coarse, re.compile('@[-0-9]+$'), None, split, True)
		begin = clock()
		for n, (sent, tree) in enumerate(zip(sents, trees)):
			if len(sent) > testmaxlen:
				continue
			print(n, end=' ')
			doctftest(coarse, fine, sent, tree, k, split, verbose=False)
		print("time elapsed", clock() - begin, "s")

__all__ = ['prunechart', 'bitparkbestitems', 'posteriorthreshold',
		'getinside', 'getoutside']
