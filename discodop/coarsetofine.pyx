"""Select suitably probable items from a chart and produce whitelist."""
from __future__ import print_function
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.algorithm cimport sort
import re
from .tree import Tree
from .treetransforms import mergediscnodes, unbinarize, fanout, addbitsets
from .containers cimport (Grammar, Chart, Edge, RankedEdge, LexicalRule,
		Label, ItemNo, cellidx, CFGtoSmallChartItem,
		CFGtoFatChartItem, SmallChartItem, FatChartItem, Whitelist)
from .bit cimport nextset, nextunset, anextset, anextunset
from .pcfg cimport CFGChart, DenseCFGChart, SparseCFGChart, CFGItem
from .plcfrs cimport SmallLCFRSChart, FatLCFRSChart
from .kbest import lazykbest
from .kbest cimport collectitems, getderiv
from roaringbitmap import RoaringBitmap
import numpy as np
from libc.math cimport exp

include "constants.pxi"

ctypedef fused ChartItem_fused:
	SmallChartItem
	FatChartItem
	size_t

# alternative: take coarse chart, return fine chart prepopulated with items
# from whitelist. However, this may involve unnecessary work on labels of fine
# chart that will not be used.
# cpdef Chart prunechart(coarsechart, Grammar fine, int k,
# 		bint splitprune, bint markorigin, bint finecfg):


def prunechart(Chart coarsechart, Grammar fine, k,
		bint splitprune, bint markorigin, bint finecfg,
		set require=None, set block=None):
	"""Produce a white list of selected chart items.

	The criterion is that they occur in the `k`-best derivations of ``chart``,
	or with posterior probability > `k`. Labels ``X`` in ``coarse.toid`` are
	projected to the labels in the mapping of the fine grammar, e.g., to ``X``
	and ``X@n-m`` for a DOP reduction.

	:param coarsechart: a Chart object produced by the PCFG or PLCFRS parser.
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
	:param require: optionally, a list of tuples ``(label, indices)``; only
		k-best derivations containing these labeled spans will be selected.
		For example, ``('NP', [0, 1, 2])``; expects ``k > 1``.
	:param block: optionally, a list of tuples ``(label, indices)``;
		these labeled spans will be pruned.
	:returns: ``(whitelist, msg)``

	For LCFRS, the white list is indexed as follows:
		:whitelisted: ``item in whitelist[label]``, ``item`` is a
			SmallChartItem or FatChartItem depending on sent. len.
		:blocked: ``item not in whitelist[label]``

	For a CFG, indexing is as follows:
		:whitelisted: ``label in whitelist[span]``,
			``span`` is an integer encoding both begin and end;
			different from a cell because does not include no. of nonterminals.
		:blocked: ``label not in whitelist[span]``
	"""
	cdef Whitelist whitelist = Whitelist()
	cdef vector[ItemNo] items
	cdef SmallChartItem sitem
	cdef FatChartItem fitem
	cdef Label label
	cdef ItemNo item
	cdef size_t span
	if (fine.mapping.size() == 0 or (splitprune and markorigin
			and fine.splitmapping.size() == 0)):
		raise ValueError('need to call fine.getmapping(coarse, ...).')
	# prune coarse chart and collect items
	if require and k >= 1:
		lazykbest(coarsechart, k, derivs=False)
		root = coarsechart.root()
		require1 = []
		for strlabel, indices in require:
			matchingitems = getmatchingitems(coarsechart, strlabel, indices)
			if not matchingitems:
				raise ValueError('could not fulfill constraint '
						'(item not in chart): %r %r' % (strlabel, indices))
			require1.append(set(matchingitems))
		derivs = RoaringBitmap()
		for n in range(coarsechart.rankededges[root].size()):
			itemset = set()
			collectitems(
					root, coarsechart.rankededges[root][n].first,
					coarsechart, itemset)
			if not any(itemset.isdisjoint(matchingitems)
					for matchingitems in require1):
				derivs.add(n)
		itemset = RoaringBitmap()
		for n in derivs:
			collectitems(
					root, coarsechart.rankededges[root][n].first,
					coarsechart, itemset)
		items = [n for n in itemset]
		msg = 'applied \'required\' constraints; %d of %d derivations left' % (
				len(derivs), coarsechart.rankededges[root].size())
	elif 0 < k < 1:  # threshold on posterior probabilities
		items, msg = posteriorthreshold(coarsechart, k)
	elif k == 0:  # only drop items not part of a full derivation
		coarsechart.filter()
		items = [n for n in range(coarsechart.parseforest.size())
				if coarsechart.parseforest[n].size() != 0]
		msg = ('coarse items before pruning: %d; after filter: %d'
				% (coarsechart.numitems(), len(items)))
	elif k >= 1:  # construct a list of the k-best chart items to prune with
		lazykbest(coarsechart, k, derivs=False)
		items = [n for n in range(coarsechart.rankededges.size())
				if coarsechart.rankededges[n].size() != 0]
		msg = ('coarse items before pruning: %d; after: %d, '
				'based on %d/%d derivations' % (
				coarsechart.numitems(), len(items),
				min(coarsechart.rankededges[coarsechart.root()].size(), k), k))
	else:
		raise ValueError('invalid value for k parameter.')
	if block:
		itemset = set(items).difference([item
				for strlabel, indices in block
					for item in getmatchingitems(coarsechart, strlabel, indices)
				])
		msg += '; applied \'block\' constraints; %d of %d items left' % (
				len(itemset), items.size())
		items = [n for n in itemset]
	if finecfg:  # index items by cell
		whitelist.cfg.clear()
		whitelist.cfg.resize(cellidx(
				coarsechart.lensent - 1, coarsechart.lensent,
				coarsechart.lensent, 1) + 1)
		for item in items:
			span = coarsechart.asCFGspan(item)
			label = coarsechart.label(item)
			whitelist.cfg[span].insert(label)
		# for span in range(whitelist.cfg.size()):
		# 	sort(whitelist.cfg[span].begin(), whitelist.cfg[span].end())
	else:  # index items by label
		if <unsigned>coarsechart.lensent >= sizeof(sitem.vec) * 8:
			whitelist.fat.clear()
			whitelist.fat.resize(coarsechart.grammar.nonterminals)
			for item in items:
				label = coarsechart.label(item)
				fitem = coarsechart.asFatChartItem(item)
				whitelist.fat[label].insert(fitem)
		else:
			whitelist.small.clear()
			whitelist.small.resize(coarsechart.grammar.nonterminals)
			for item in items:
				label = coarsechart.label(item)
				sitem = coarsechart.asSmallChartItem(item)
				whitelist.small[label].insert(sitem)
	whitelist.mapping = &(fine.mapping[0])
	whitelist.splitmapping = &(fine.splitmapping[0])
	return whitelist, msg


def getmatchingitems(Chart chart, str strlabel, indices):
	matchingitems = []
	for n in chart.grammar.tblabelmapping.get(strlabel, []):
		item = chart.itemid1(n, indices)
		if item != 0:
			matchingitems.append(item)
	return matchingitems


def posteriorthreshold(Chart chart, double threshold):
	"""Prune labeled spans from chart below given posterior threshold.

	:returns: dictionary of remaining items."""
	cdef ItemNo itemidx
	if not 0 < threshold < 1:
		raise ValueError('expected posterior threshold k with 0 < k < 1.')
	if not chart.inside.size():
		origlogprob = chart.grammar.logprob
		chart.grammar.switch(chart.grammar.currentmodel, logprob=False)
		getinside(chart)
		getoutside(chart)
		chart.grammar.switch(chart.grammar.currentmodel, logprob=origlogprob)
	sentprob = chart.inside[chart.root()]
	threshold *= sentprob
	if not sentprob:
		raise ValueError('sentence has zero posterior prob.: %g' % sentprob)
	posterior = [itemidx for itemidx
			in range(chart.inside.size())
			if chart.inside[itemidx] * chart.outside[itemidx] > threshold]

	unfiltered = chart.numitems()
	numitems = 0
	for itemidx in range(1, chart.numitems() + 1):
		numitems += chart.outside[itemidx] != 0.0
	numremain = len(posterior)
	msg = ('coarse items before pruning=%d; filtered: %d;'
			' pruned: %d; sentprob=%g' % (
			unfiltered, numitems, numremain, sentprob))
	return posterior, msg


def getinside(Chart chart):
	"""Compute inside probabilities for a chart given its parse forest."""
	# this needs to be bottom up, so need order in which items were added
	# currently separate list, chart.itemsinorder
	# NB: sorting items by length is not enough,
	# unaries have to be in the right order...

	# choices for probs:
	# - normal => underflow (current)
	# - logprobs => loss of precision w/addition
	# - normal, scaled => how?
	cdef ItemNo n, item, leftitem, rightitem
	cdef Edge edge
	cdef double prob
	chart.inside.resize(chart.probs.size(), 0.0)

	# traverse items in bottom-up order
	for n in range(1, chart.numitems() + 1):
		item = chart.getitemidx(n)
		for edge in chart.parseforest[item]:
			if edge.rule is NULL:
				try:
					prob = chart.lexprob(item, edge)
				except ValueError:
					# fall back to Viterbi score from chart
					# if there is a single incoming edge this is correct
					assert chart.parseforest[item].size() == 1
					if chart.logprob:
						prob = exp(-chart.probs[item])
					else:
						prob = chart.probs[item]
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


def getoutside(Chart chart):
	"""Compute outside probabilities for a chart given its parse forest."""
	cdef ItemNo n, item, leftitem, rightitem
	cdef Edge edge
	cdef double outsideprob
	# cdef double sentprob = chart.inside[chart.root()]
	# traverse items in top-down order
	chart.outside.resize(chart.probs.size(), 0.0)
	chart.outside[chart.root()] = 1.0
	for n in range(chart.numitems(), 0, -1):
		item = chart.getitemidx(n)
		# can we define outside[item] simply as sentprob - inside[item] ?
		# chart.outside[item] = sentprob - chart.inside[item]
		for edge in chart.parseforest[item]:
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
		getderivations(chart, 10, True)
		mpp, _ = marginalize("mpp", chart)
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
	l, msg = prunechart(chart, fine, k, split, True, False)
	if verbose:
		print(msg)
		print("\nitems in 50-best of coarse chart")
		lazykbest(chart, k, derivs=False)
		for a in chart.rankededges:
			print(coarse.tolabel[a.label], bin(a.vec))
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
		getderivations(chart2, 10, True)
		mpp, _ = marginalize("mpp", chart2)
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

__all__ = ['prunechart', 'posteriorthreshold', 'getinside', 'getoutside']
