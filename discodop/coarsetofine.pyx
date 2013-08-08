""" Assorted functions to project items from a coarse chart to corresponding
items for a fine grammar. """
from __future__ import print_function
from collections import defaultdict
from tree import Tree
from treetransforms import mergediscnodes, unbinarize, fanout, addbitsets
from containers cimport ChartItem, Edge, RankedEdge, RankedCFGEdge, Grammar, \
		CFGChartItem, CFGEdge, LCFRSEdge, new_CFGChartItem, ULLong, UInt, \
		CFGtoSmallChartItem, CFGtoFatChartItem
from kbest import lazykbest
from agenda cimport Entry
import numpy as np


cpdef prunechart(chart, ChartItem goal, Grammar coarse, Grammar fine,
	int k, bint splitprune, bint markorigin, bint finecfg, bint bitpar):
	""" Produce a white list of chart items occurring in the k-best derivations
	of chart, where labels X in coarse.toid are projected to the labels
	X and X@n-m in fine.toid, for possible values of n and m.
	Modifies chart destructively.

	- k: number of k-best derivations to consider. When k==0, the chart is
		not pruned but filtered to contain only items that contribute to a
		complete derivation.
	- splitprune: coarse stage used a split-PCFG where discontinuous node
		appear as multiple CFG nodes. Every discontinuous node will result
		in multiple lookups into whitelist to see whether it should be
		allowed on the agenda.
	- markorigin: in combination with splitprune, coarse labels include an
		integer to distinguish components; e.g., CFG nodes NP*0 and NP*1
		map to the discontinuous node NP_2. """
	cdef dict d
	cdef list whitelist
	cdef ChartItem Ih
	assert fine.mapping is not NULL
	if splitprune and markorigin:
		assert fine.splitmapping is not NULL
		assert fine.splitmapping[0] is not NULL
	d = <dict>defaultdict(dict)
	# construct a list of the k-best nonterminals to prune with
	if bitpar:
		kbest = bitparkbestitems(chart, coarse, finecfg)
	else:
		kbest = kbest_items(chart, goal, k, finecfg, coarse.tolabel)
	if finecfg:
		whitelist = [[{} for _ in range((<CFGChartItem>goal).end + 1)]
				for _ in range((<CFGChartItem>goal).end)]
		for left in range((<CFGChartItem>goal).end):
			for right in range(left + 1, (<CFGChartItem>goal).end + 1):
				span = left, right
				cell = whitelist[left][right]
				for label in range(1, fine.nonterminals):
					if (fine.mapping[label] == 0
							or span in kbest[fine.mapping[label]]):
						cell[label] = None
	else:
		whitelist = [None] * fine.nonterminals
		kbestspans = [{} for a in coarse.toid]
		kbestspans[0] = None
		# uses ids of labels in coarse chart
		for Ih in kbest:
			label = Ih.label
			Ih.label = 0
			kbestspans[label][Ih] = 0.0
		# now construct a list which references these coarse items:
		for label in range(fine.nonterminals):
			if splitprune and markorigin and fine.fanout[label] != 1:
				if fine.splitmapping[label] is not NULL:
					whitelist[label] = [kbestspans[fine.splitmapping[label][n]]
							for n in range(fine.fanout[label])]
			else:
				if fine.mapping[label] != 0:
					whitelist[label] = kbestspans[fine.mapping[label]]
	return whitelist, len(kbest)


cpdef kbest_items(chart, ChartItem start, int k, bint finecfg, coarsetolabel):
	""" produce a dictionary with ChartItems as keys (value is currently unused)
	according to the k-best derivations in a chart. """
	cdef Entry entry
	cdef CFGChartItem ci
	if k == 0:
		items = filterchart(chart, start)
		for a in items:
			# use if probabilities matter
			#e = min(items[a])
			#items[a] = e.inside
			items[a] = 0.0
	else:
		_, D, agenda = lazykbest(chart, start, k, derivs=False)
		if isinstance(start, CFGChartItem):
			ci = start
			if ci.end >= (sizeof(ULLong) * 8):
				ei = CFGtoFatChartItem(ci.label, ci.start, ci.end)
			else:
				ei = CFGtoSmallChartItem(ci.label, ci.start, ci.end)
			if finecfg:
				items = [{} for _ in coarsetolabel]
				items[ci.label][ci.start, ci.end] = 0.0
			else:
				items = {ei: 0.0}
			for entry in D[(<CFGChartItem>start).start][
					(<CFGChartItem>start).end][start.label][:k]:
				traversekbestcfg(entry.key, entry.value, D, chart, items,
						(<CFGChartItem>start).end >= (sizeof(ULLong) * 8),
						finecfg, agenda)
				#agenda.difference_update(items)
		else:
			items = {start: 0.0}
			for entry in D[start][:k]:
				traversekbest(entry.key, entry.value, D, chart, items, agenda)
				#agenda.difference_update(items)
	return items


cdef traversekbest(RankedEdge ej, double rootprob, dict D, dict chart,
		dict items, set agenda):
	""" Traverse a derivation (e, j), collecting all items belonging to it, and
	noting (Viterbi) outside costs relative to its root edge (these costs are
	currently unused; only presence or absence in this list is exploited)"""
	cdef LCFRSEdge e = ej.edge
	cdef RankedEdge eejj
	cdef Entry entry
	cdef double prob
	if e.left in chart:
		entry = D[e.left][ej.left]
		eejj = entry.key
		prob = entry.value
		if e.left not in items:
			items[e.left] = rootprob - prob
		if True or eejj in agenda:
			traversekbest(eejj, rootprob, D, chart, items, agenda)
	if e.right.label:
		entry = D[e.right][ej.right]
		eejj = entry.key
		prob = entry.value
		if e.right not in items:
			items[e.right] = rootprob - prob
		if True or eejj in agenda:
			traversekbest(eejj, rootprob, D, chart, items, agenda)


cdef traversekbestcfg(RankedCFGEdge ej, double rootprob, list D, list chart,
		items, bint fatitems, bint finecfg, agenda):
	""" Traverse a derivation (e, j), collecting all items belonging to it, and
	noting (Viterbi) outside costs relative to its root edge (these costs are
	currently unused; only presence or absence in this list is exploited)"""
	cdef ChartItem ei
	cdef CFGEdge e = <CFGEdge>ej.edge
	cdef RankedCFGEdge eejj
	cdef Entry entry
	cdef double prob
	# TODO: efficiently track already-visited ranked edges.
	# maintaining a set only slows things down.
	if e.rule is NULL:
		return
	label = e.rule.rhs1
	start = ej.start
	end = e.mid
	if label in chart[start][end]:
		entry = D[start][end][label][ej.left]
		eejj = <RankedEdge>entry.key
		prob = entry.value
		if finecfg:
			if (start, end) not in items[label]:
				items[label][start, end] = rootprob - prob
		else:
			if fatitems:
				ei = CFGtoFatChartItem(label, start, end)
			else:
				ei = CFGtoSmallChartItem(label, start, end)
			if ei not in items:
				items[ei] = rootprob - prob
		if True or eejj in agenda:
			traversekbestcfg(eejj, rootprob, D, chart, items, fatitems,
					finecfg, agenda)
	if e.rule.rhs2:
		label = e.rule.rhs2
		start = e.mid
		end = ej.end
		entry = D[start][end][label][ej.right]
		eejj = <RankedEdge>entry.key
		prob = entry.value
		if finecfg:
			if (start, end) not in items[label]:
				items[label][start, end] = rootprob - prob
		else:
			if fatitems:
				ei = CFGtoFatChartItem(label, start, end)
			else:
				ei = CFGtoSmallChartItem(label, start, end)
			if ei not in items:
				items[ei] = rootprob - prob
		if True or eejj in agenda:
			traversekbestcfg(eejj, rootprob, D, chart, items, fatitems,
					finecfg, agenda)


cpdef filterchart(chart, ChartItem start):
	""" Remove all entries that do not contribute to a complete derivation
	headed by 'start'. """
	chart2 = {}
	if isinstance(start, CFGChartItem):
		start1 = (<CFGChartItem>start).start
		end = (<CFGChartItem>start).end
		fatitems = end >= (sizeof(ULLong) * 8)
		chart = [[b.copy() for b in a] for a in chart]
		filter_subtreecfg(start.label, start1, end, chart, chart2, fatitems)
	else:
		filter_subtree(start, chart, chart2)
	return chart2


cdef void filter_subtree(ChartItem start, dict chart, dict chart2):
	""" Recursively filter an LCFRS chart. """
	cdef LCFRSEdge edge
	cdef ChartItem item
	chart2[start] = chart[start]
	for edge in chart[start]:
		item = edge.left
		if item.label and item not in chart2:
			filter_subtree(edge.left, chart, chart2)
		item = edge.right
		if item.label and item not in chart2:
			filter_subtree(edge.right, chart, chart2)


cdef void filter_subtreecfg(label, start, end, list chart, dict chart2,
		bint fatitems):
	""" Recursively filter a PCFG chart. """
	cdef CFGEdge edge
	cdef ChartItem newitem
	if fatitems:
		newitem = CFGtoFatChartItem(label, start, end)
	else:
		newitem = CFGtoSmallChartItem(label, start, end)
	# should convert CFGEdge to LCFRSEdge, but currently not used anyway.
	chart2[newitem] = chart[start][end].pop(label)
	for edge in chart2[newitem]:
		if edge.rule is NULL:
			continue
		if edge.rule.rhs1 and edge.rule.rhs1 in chart[start][edge.mid]:
			filter_subtreecfg(edge.rule.rhs1, start, edge.mid,
					chart, chart2, fatitems)
		if edge.rule.rhs2 and edge.rule.rhs2 in chart[edge.mid][end]:
			filter_subtreecfg(edge.rule.rhs2, edge.mid, end,
					chart, chart2, fatitems)


def whitelistfromposteriors(inside, outside, CFGChartItem start,
		Grammar coarse, Grammar fine, double threshold,
		bint splitprune, bint markorigin, bint finecfg):
	""" Compute posterior probabilities and prune away cells below some
	threshold. this version is for use with parse_sparse(). """
	cdef UInt label
	cdef short lensent = start.end
	assert 0 < threshold < 1, (
			"threshold should be a cutoff for probabilities between 0 and 1.")
	sentprob = inside[0, lensent, start.label]
	posterior = (inside[:lensent, :lensent + 1]
		* outside[:lensent, :lensent + 1]) / sentprob

	finechart = [[{} for _ in range(lensent + 1)] for _ in range(lensent)]
	leftidx, rightidx, labels = (posterior[:lensent, :lensent + 1]
		> threshold).nonzero()

	kbestspans = [{} for _ in coarse.toid]
	fatitems = lensent >= (sizeof(ULLong) * 8)
	for label, left, right in zip(labels, leftidx, rightidx):
		if finecfg:
			ei = left, right
		elif fatitems:
			ei = CFGtoFatChartItem(0, left, right)
		else:
			ei = CFGtoSmallChartItem(0, left, right)
		kbestspans[label][ei] = 0.0

	if finecfg:
		whitelist = [[{} for _ in range(start.end + 1)]
				for _ in range(start.end)]
		for left in range(start.end):
			for right in range(left + 1, start.end + 1):
				span = left, right
				cell = whitelist[left][right]
				for label in range(1, fine.nonterminals):
					if (fine.mapping[label] == 0
							or span in kbestspans[fine.mapping[label]]):
						cell[label] = None
	else:
		whitelist = [None] * fine.nonterminals
		for label in range(fine.nonterminals):
			if splitprune and markorigin and fine.fanout[label] != 1:
				if fine.splitmapping[label] is not NULL:
					whitelist[label] = [kbestspans[fine.splitmapping[label][n]]
						for n in range(fine.fanout[label])]
			else:
				if fine.mapping[label] != 0:
					whitelist[label] = kbestspans[fine.mapping[label]]
	unfiltered = (outside != 0.0).sum()
	numitems = (posterior != 0.0).sum()
	numremain = (posterior > threshold).sum()
	return whitelist, sentprob, unfiltered, numitems, numremain


def whitelistfromposteriors_matrix(inside, outside, ChartItem goal,
		Grammar coarse, Grammar fine, finechart, short maxlen,
		double threshold):
	""" compute posterior probabilities and prune away cells below some
	threshold. this version produces a matrix with pruned spans having NaN as
	value. """
	cdef long label
	cdef short lensent = goal.right
	sentprob = inside[0, lensent, goal.label]
	#print >>stderr, "sentprob=%g" % sentprob
	posterior = (inside[:lensent, :lensent + 1, :]
			* outside[:lensent, :lensent + 1, :]) / sentprob
	inside[:lensent, :lensent + 1, :] = np.NAN
	inside[posterior > threshold] = np.inf
	#print >>stderr, ' ', (posterior > threshold).sum(),
	#print >>stderr, "of", (posterior != 0.0).sum(),
	#print >>stderr, "nonzero coarse items left",
	#labels, leftidx, rightidx = (posterior[:lensent, :lensent+1, :]
	#	> threshold).nonzero()
	#for left, right, label in zip(leftidx, rightidx, labels):
	#	for x in mapping[label]:
	#		finechart[left, right, x] = inside[left, right, label]
	for label in range(len(fine.toid)):
		finechart[:lensent, :lensent + 1, label] = inside[
				:lensent, :lensent + 1, fine.mapping[label]]


cpdef merged_kbest(dict chart, ChartItem start, int k, Grammar grammar):
	""" Like kbest_items, but apply the reverse of the Boyd (2007)
	transformation to the k-best derivations."""
	cdef dict newchart = <dict>defaultdict(dict)
	cdef list derivs = [Tree.parse(a, parse_leaf=int) for a, _
				in lazykbest(chart, start, k, grammar.tolabel)[0]]
	for tree in derivs:
		addbitsets(mergediscnodes(unbinarize(
				tree, childchar=":", parentchar="!")))
		for node in tree.subtrees():
			f = fanout(node)
			if f > 1:
				label = "%s_%d" % (node.label, f)
			else:
				label = node.label
			newchart[label][sum([1L << n for n in node.leaves()])] = 0.0
	return newchart


def bitparkbestitems(dict derivs, Grammar coarse, bint finecfg):
	""" Take a dictionary of CFG derivations as strings, and produce a list of
	ChartItems occurring in those derivations. """
	items = [{} for _ in coarse.tolabel] if finecfg else {}
	fatitems = None
	for deriv in derivs:
		t = Tree.parse(deriv, parse_leaf=int)
		if fatitems is None:
			fatitems = len(t.leaves()) >= (sizeof(ULLong) * 8)
		for n in t.subtrees():
			label = coarse.toid[n.label]
			leaves = n.leaves()
			start = min(leaves)
			end = max(leaves) + 1
			if finecfg:
				items[label][start, end] = 0.0
			elif fatitems:
				item = CFGtoFatChartItem(label, start, end)
				items[item] = 0.0
			else:
				item = CFGtoSmallChartItem(label, start, end)
				items[item] = 0.0
	return items


def doctf(coarse, fine, sent, tree, k, split, verbose=False):
	import plcfrs
	from disambiguation import marginalize
	from treetransforms import canonicalize, removefanoutmarkers
	from math import exp as pyexp
	sent, tags = zip(*sent)
	print(" C O A R S E ", end='')
	p, start, _ = plcfrs.parse(sent, coarse, tags=tags)
	if start:
		mpp, _, _ = marginalize("mpp", p, start, coarse, 10)
		for t in mpp:
			print(pyexp(-mpp[t]), end='')
			t = Tree.parse(t, parse_leaf=int)
			if split:
				unbinarize(t, childchar=":", parentchar="!")
				mergediscnodes(t)
			unbinarize(t)
			t = canonicalize(removefanoutmarkers(t))
			print("exact match" if t == canonicalize(tree) else "no match")
	else:
		print("no parse")
		return
		#pprint_chart(p, sent, coarse.tolabel)
	l, _ = prunechart(p, start, coarse, fine, k, split, True, False, False)
	if verbose:
		print("\nitems in 50-best of coarse chart")
		if split:
			d = merged_kbest(p, start, k, coarse)
			for label in d:
				print(label, map(bin, d[label].keys()))
		else:
			kbest = kbest_items(p, start, k, False, ())
			for a, b in kbest.items():
				print(coarse.tolabel[(<ChartItem>a).label],
						bin((<ChartItem>a).vec), b)
		print("\nwhitelist:")
		for n, x in enumerate(l):
			if isinstance(x, dict):
				print(fine.tolabel[n], map(bin, x))
			elif x:
				for m, y in enumerate(x):
					print(fine.tolabel[n], m, map(bin, y))
	print(" F I N E ", end='')
	pp, start, _ = plcfrs.parse(sent, fine, tags=tags, whitelist=l,
			splitprune=split, markorigin=True)
	if start:
		mpp, _, _ = marginalize("mpp", pp, start, fine, 10)
		for t in mpp:
			print(pyexp(-mpp[t]), end='')
			t = Tree.parse(t, parse_leaf=int)
			unbinarize(t)
			t = canonicalize(removefanoutmarkers(t))
			#print(t)
			print("exact match" if t == canonicalize(tree) else "no match", t)
	else:
		print("no parse. problem.")
		return
		#if verbose:
		#xp = set((coarse.tolabel[a.label], a.vec) for a in p.keys() if p[a])
		#xpp = set((fine.tolabel[a.label], a.vec) for a in pp.keys() if pp[a])
		#print("difference:")
		#for a in xp - xpp:
		#	if "*" not in a[0]:
		#		print(a[0], bin(a[1]))
		#print("\nfine chart:")
		#for a in xpp:
		#	if "@" not in a[0]:
		#		print(a[0], bin(a[1]))
		#pprint_chart(pp, sent, fine.tolabel)


def main():
	import re
	from treetransforms import splitdiscnodes, binarize, addfanoutmarkers
	from treebank import NegraCorpusReader
	from grammar import treebankgrammar, dopreduction, subsetgrammar
	from time import clock
	k = 50
	#corpus = NegraCorpusReader(".", "toytb.export", encoding="iso-8859-1")
	#corpus = NegraCorpusReader("../rparse", "negraproc.export",
	#	encoding="utf-8", headrules="negra.headrules", headfinal=True,
	#   headreverse=False)
	#train = 400
	#test = 40
	#testmaxlen = 999
	corpus = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1",
		headrules=None, headfinal=True, headreverse=False)
	train = 0
	test = 3
	testmaxlen = 999
	#trees = corpus.parsed_sents().values()[:train]
	#sents = corpus.sents().values()[:train]
	trees = list(corpus.parsed_sents().values())
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
		for (r, yf), w in treebankgrammar([t], [s]):
			assert len(yf) == 1
	fine999x = dopreduction(trees, sents)[0]
	fine999 = Grammar(fine999x)
	fine1 = Grammar(dopreduction(dtrees, sents)[0])
	trees = list(corpus.parsed_sents().values())[train:train + test]
	sents = list(corpus.tagged_sents().values())[train:train + test]
	if subsetgrammar(normallcfrs, fine999x):
		print("DOP grammar is a superset")
	else:
		print("DOP grammar is NOT a superset!")
	for msg, coarse, fine, split, enabled in zip(
		"normal parentannot cf-split".split(),
		(normal, parent, splitg),
		(fine999, fine1, fine1),
		(False, False, True),
		(True, False, True)):
		if not enabled:
			continue
		print("coarse grammar:", msg)
		fine.getmapping(coarse, re.compile(b'@[-0-9]+$'), None, split, True)
		begin = clock()
		for n, (sent, tree) in enumerate(zip(sents, trees)):
			if len(sent) > testmaxlen:
				continue
			print(n, end='')
			doctf(coarse, fine, sent, tree, k, split, verbose=False)
		print("time elapsed", clock() - begin, "s")

if __name__ == '__main__':
	main()
