"""Assorted functions to project items from a coarse chart to corresponding
items for a fine grammar.
"""
from collections import defaultdict
from nltk import Tree
from treetransforms import mergediscnodes, unbinarize, slowfanout
from containers cimport ChartItem, Edge, RankedEdge, RankedCFGEdge, Grammar, \
		CFGChartItem, CFGEdge, LCFRSEdge, new_CFGChartItem, ULLong, \
		CFGtoSmallChartItem, CFGtoFatChartItem
from kbest import lazykbest, lazykthbest, lazykbestcfg
from agenda cimport Entry

infinity = float('infinity')

cpdef prunechart(chart, ChartItem goal, Grammar coarse, Grammar fine,
	int k, bint splitprune, bint markorigin):
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
			map to the discontinuous node NP_2
	"""
	cdef dict d, kbest
	cdef list whitelist
	cdef ChartItem Ih
	assert fine.mapping is not NULL
	if splitprune and markorigin:
		assert fine.splitmapping is not NULL
		assert fine.splitmapping[0] is not NULL
	whitelist = [None] * len(fine.toid)
	d = <dict>defaultdict(dict)
	# construct a list of the k-best nonterminals to prune with
	kbest = kbest_items(chart, goal, k)
	kbestspans = [{} for a in coarse.toid]
	kbestspans[0] = None
	# uses ids of labels in coarse chart
	for Ih in kbest:
		label = Ih.label; Ih.label = 0
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

cpdef merged_kbest(dict chart, ChartItem start, int k, Grammar grammar):
	""" Like kbest_items, but apply the reverse of the Boyd (2007)
	transformation to the k-best derivations."""
	cdef dict newchart = <dict>defaultdict(dict)
	cdef list derivs = [Tree.parse(a, parse_leaf=int) for a, _
				in lazykbest(chart, start, k, grammar.tolabel)[0]]
	for a in derivs:
		unbinarize(a, childChar=":", parentChar="!")
		mergediscnodes(a)
	for tree in derivs:
		for node in tree.subtrees():
			arity = slowfanout(node)
			if arity > 1: label = "%s_%d" % (node.node, arity)
			else: label = node.node
			newchart[label][sum([1L << n for n in node.leaves()])] = 0.0
	return newchart

cpdef dict kbest_items(chart, ChartItem start, int k):
	""" produce a dictionary with ChartItems as keys (value is currently unused)
	according to the k-best derivations in a chart. """
	cdef dict items
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
		if isinstance(start, CFGChartItem):
			ci = start
			D = lazykbestcfg(chart, start, k)
			if ci.end >= (sizeof(ULLong) * 8):
				ei = CFGtoFatChartItem(ci.label, ci.start, ci.end)
			else: ei = CFGtoSmallChartItem(ci.label, ci.start, ci.end)
			items = { ei : 0.0 }
			for entry in D[(<CFGChartItem>start).start][
				(<CFGChartItem>start).end][start.label]:
				traversekbestcfg(entry.key, entry.value, D, chart, items,
					(<CFGChartItem>start).end >= (sizeof(ULLong) * 8))
		else:
			D = {}
			items = { start : 0.0 }
			lazykthbest(start, k, k, D, {}, chart, set())
			for entry in D[start]:
				traversekbest(entry.key, entry.value, D, chart, items)
	return items

cdef traversekbest(RankedEdge ej, double rootprob, dict D, dict chart, dict items):
	""" Traverse a derivation e,j, collecting all items belonging to it, and
	noting (Viterbi) outside costs relative to its root edge (these costs are
	currently unused; only presence or absence in this list is exploited)"""
	cdef LCFRSEdge e = ej.edge
	cdef RankedEdge eejj
	cdef Entry entry
	cdef double prob
	# TODO: efficiently track already-visited ranked edges.
	# maintaining a set only slows things down.
	if e.left in chart:
		if e.left in D:
			entry = D[e.left][ej.left]
			eejj = entry.key; prob = entry.value
		elif ej.left == 0:
			eejj = RankedEdge(e.left, min(chart[e.left]), 0, 0)
			prob = eejj.edge.inside
		else: raise ValueError
		if e.left not in items:
			items[e.left] = rootprob - prob
		traversekbest(eejj, rootprob, D, chart, items)
	if e.right.label:
		if e.right in D:
			entry = D[e.right][ej.right]
			eejj = entry.key; prob = entry.value
		elif ej.right == 0:
			eejj = RankedEdge(e.right, min(chart[e.right]), 0, 0)
			prob = eejj.edge.inside
		else: raise ValueError
		if e.right not in items:
			items[e.right] = rootprob - prob
		traversekbest(eejj, rootprob, D, chart, items)

cdef traversekbestcfg(RankedCFGEdge ej, double rootprob, list D, list chart,
		dict items, bint fatitems):
	""" Traverse a derivation e,j, collecting all items belonging to it, and
	noting (Viterbi) outside costs relative to its root edge (these costs are
	currently unused; only presence or absence in this list is exploited)"""
	cdef ChartItem ei
	cdef CFGEdge e = <CFGEdge>ej.edge
	cdef RankedCFGEdge eejj
	cdef Entry entry
	cdef double prob
	# TODO: efficiently track already-visited ranked edges.
	# maintaining a set only slows things down.
	label = 0 if e.rule is NULL else e.rule.rhs1
	start = ej.start; end = e.mid
	if label in chart[start][end]:
		if label in D[start][end]:
			entry = D[start][end][label][ej.left]
			eejj = <RankedEdge>entry.key
			prob = entry.value
		elif ej.left == 0:
			eejj = RankedCFGEdge(label, start, end,
				min(chart[start][end][label]), 0, 0)
			prob = eejj.edge.inside
		else: raise ValueError
		if fatitems: ei = CFGtoFatChartItem(label, start, end)
		else: ei = CFGtoSmallChartItem(label, start, end)
		if ei not in items:
			items[ei] = rootprob - prob
		traversekbestcfg(eejj, rootprob, D, chart, items, fatitems)
	if e.rule is not NULL and e.rule.rhs2:
		label = e.rule.rhs2
		start = e.mid; end = ej.end
		if label in D[start][end]:
			entry = D[start][end][label][ej.right]
			eejj = <RankedEdge>entry.key
			prob = entry.value
		elif ej.right == 0:
			eejj = RankedCFGEdge(label, start, end,
				min(chart[start][end][label]), 0, 0)
			prob = eejj.edge.inside
		else: raise ValueError
		if fatitems: ei = CFGtoFatChartItem(label, start, end)
		else: ei = CFGtoSmallChartItem(label, start, end)
		if ei not in items:
			items[ei] = rootprob - prob
		traversekbestcfg(eejj, rootprob, D, chart, items, fatitems)

cpdef filterchart(dict chart, ChartItem start):
	""" remove all entries that do not contribute to a complete derivation
	headed by "start" """
	chart2 = {}
	if isinstance(start, CFGChartItem):
		filter_subtreecfg(start, chart, chart2)
	else: filter_subtree(start, chart, chart2)
	return chart2

cdef void filter_subtree(ChartItem start, dict chart, dict chart2):
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

cdef CFGChartItem tmpitem = CFGChartItem(0, 0, 0)
cdef void filter_subtreecfg(CFGChartItem item, list chart, dict chart2):
	cdef CFGEdge edge
	# FIXME!
	chart2[item] = chart[item]
	for edge in chart[item]:
		tmpitem.label = edge.rule.rhs1
		tmpitem.start = item.start
		tmpitem.end = edge.mid
		if tmpitem.label and tmpitem not in chart2:
			filter_subtree(tmpitem, chart, chart2)
		tmpitem.label = edge.rule.rhs2
		tmpitem.start = edge.mid
		tmpitem.end = item.end
		if tmpitem.label and tmpitem not in chart2:
			filter_subtree(tmpitem, chart, chart2)

def doctf(coarse, fine, sent, tree, k, split, verbose=False):
	from parser import parse #, pprint_chart
	from disambiguation import marginalize
	from grammar import canonicalize, rem_marks
	from math import exp
	sent, tags = zip(*sent)
	print " C O A R S E ",
	p, start, _ = parse(sent, coarse, start=coarse.toid['ROOT'], tags=tags)
	if start:
		mpp, _ = marginalize(p, start, coarse.tolabel)
		for t in mpp:
			print exp(-mpp[t]),
			t = Tree.parse(t, parse_leaf=int)
			if split:
				unbinarize(t, childChar=":", parentChar="!")
				mergediscnodes(t)
			unbinarize(t)
			t = canonicalize(rem_marks(t))
			print "exact match" if t == canonicalize(tree) else "no match" #, t
	else:
		print "no parse"
		return
		#pprint_chart(p, sent, coarse.tolabel)
	l, _ = prunechart(p, start, coarse, fine, k, split, True)
	if verbose:
		print "\nitems in 50-best of coarse chart"
		if split:
			d = merged_kbest(p, start, k, coarse)
			for label in d:
				print label, map(bin, d[label].keys())
		else:
			kbest = kbest_items(p, start, k)
			for a,b in kbest.items():
				print coarse.tolabel[(<ChartItem>a).label], bin((<ChartItem>a).vec), b
		print "\nwhitelist:"
		for n,x in enumerate(l):
			if isinstance(x, dict):
				print fine.tolabel[n], map(bin, x)
			elif x:
				for m, y in enumerate(x):
					print fine.tolabel[n], m, map(bin, y)
	print " F I N E ",
	pp, start, _ = parse(sent, fine, start=fine.toid['ROOT'], tags=tags,
		whitelist=l, splitprune=split, markorigin=True)
	if start:
		mpp, _ = marginalize(pp, start, fine.tolabel)
		for t in mpp:
			print exp(-mpp[t]),
			t = Tree.parse(t, parse_leaf=int)
			unbinarize(t)
			t = canonicalize(rem_marks(t))
			#print t.pprint(margin=999)
			print "exact match" if t == canonicalize(tree) else "no match", t
	else:
		print "no parse. problem."; return
		#if verbose:
		#xp = set((coarse.tolabel[a.label], a.vec) for a in p.keys() if p[a])
		#xpp = set((fine.tolabel[a.label], a.vec) for a in pp.keys() if pp[a])
		#print "difference:"
		#for a in xp - xpp:
		#	if "*" not in a[0]: print a[0], bin(a[1])
		#print "\nfine chart:"
		#for a in xpp:
		#	if "@" not in a[0]: print a[0], bin(a[1])

		#pprint_chart(pp, sent, fine.tolabel)

def main():
	import re
	from treetransforms import splitdiscnodes, binarize, addfanoutmarkers
	from treebank import NegraCorpusReader
	from grammar import induce_plcfrs, dop_lcfrs_rules, subsetgrammar
	from time import clock
	k = 50
	#corpus = NegraCorpusReader(".", "toytb.export", encoding="iso-8859-1")
	#corpus = NegraCorpusReader("../rparse", "negraproc.export",
	#	encoding="utf-8", headorder=True, headfinal=True, headreverse=False)
	#train = 400; test = 40; testmaxlen = 999;
	corpus = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1",
		headorder=False, headfinal=True, headreverse=False)
	train = 0; test = 3; testmaxlen = 999;
	#trees = corpus.parsed_sents()[:train]
	#sents = corpus.sents()[:train]
	trees = corpus.parsed_sents()
	sents = corpus.sents()

	dtrees = [t.copy(True) for t in trees]
	parenttrees = [t.copy(True) for t in trees]
	for t in trees:
		binarize(t, vertMarkov=0, horzMarkov=1)
		addfanoutmarkers(t)
	cftrees = [splitdiscnodes(t.copy(True), markorigin=True) for t in trees]
	for t in cftrees:
		#t.chomsky_normal_form(childChar=":")
		binarize(t, horzMarkov=1, tailMarker='', leftMostUnary=True,
			childChar=":") #NB leftMostUnary is important
		addfanoutmarkers(t)
	for t in parenttrees:
		binarize(t, vertMarkov=2, horzMarkov=1)
		addfanoutmarkers(t)
	for t in dtrees:
		binarize(t, vertMarkov=0, horzMarkov=1)
		addfanoutmarkers(t)
	# mark heads, canonicalize, binarize head outward
	normallcfrs = induce_plcfrs(trees, sents)
	normal = Grammar(normallcfrs)
	parent = Grammar(induce_plcfrs(parenttrees, sents))
	splitg = Grammar(induce_plcfrs(cftrees, sents))
	for t,s in zip(cftrees, sents):
		for (r,yf),w in induce_plcfrs([t], [s]): assert len(yf) == 1
	fine999x = dop_lcfrs_rules(trees, sents)
	fine999 = Grammar(fine999x)
	fine1 = Grammar(dop_lcfrs_rules(dtrees, sents))
	trees = list(corpus.parsed_sents()[train:train+test])
	sents = corpus.tagged_sents()[train:train+test]
	if subsetgrammar(normallcfrs, fine999x):
		print "DOP grammar is a superset"
	else: print "DOP grammar is NOT a superset!"
	for msg, coarse, fine, split, enabled in zip(
		"normal parentannot cf-split".split(),
		(normal, parent, splitg),
		(fine999, fine1, fine1),
		(False, False, True),
		(True, False, True)):
		if not enabled: continue
		print "coarse grammar:", msg
		fine.getmapping(re.compile("@[-0-9]+$"), None, coarse,
				split, True)
		#??: fine.getdonotprune(re.compile(r"\|<"))
		begin = clock()
		for n, (sent, tree) in enumerate(zip(sents, trees)):
			if len(sent) > testmaxlen: continue
			print n,
			doctf(coarse, fine, sent, tree, k, split, verbose=False)
		print "time elapsed", clock() - begin, "s"

if __name__ == '__main__': main()
