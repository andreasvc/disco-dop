""" Assorted functions to project items from a coarse chart to corresponding
items for a fine grammar. """
from __future__ import print_function
from discodop.tree import Tree
from discodop.treetransforms import mergediscnodes, unbinarize, fanout, \
		addbitsets
from discodop.containers cimport Grammar, Chart, ChartItem, Edges, Edge, \
		Rule, LexicalRule, RankedEdge, ULLong, UInt, \
		cellidx, CFGtoSmallChartItem, CFGtoFatChartItem
from discodop.kbest import lazykbest
from discodop.plcfrs cimport Entry
import numpy as np

# alternative: take coarse chart, return fine chart w/whitelist.
#cpdef Chart prunechart(coarsechart, Grammar fine, int k,
#		bint splitprune, bint markorigin, bint finecfg, bint bitpar):


def prunechart(Chart coarsechart, Grammar fine, k,
		bint splitprune, bint markorigin, bint finecfg, bint bitpar):
	""" Produce a white list of chart items occurring in the k-best derivations
	of ``chart``, or with posterior probability > k. Labels ``X`` in
	``coarse.toid`` are projected to the labels in the mapping of the fine
	grammar, e.g., to ``X`` and ``X@n-m`` for a DOP reduction.

	:param coarsechart: a Chart object produced by the PCFG or PLCFRS parser,
		or derivations from bitpar.
	:param coarse: the grammar with which ``chart`` was produced.
	:param fine: the grammar to map labels to after pruning. must have a
		mapping to the coarse grammar established by ``fine.getmapping()``.
	:param k: when k >= 1: number of k-best derivations to consider; when k==0,
		the chart is not pruned but filtered to contain only items that
		contribute to a complete derivation;
		when 0 < k < 1, inside-outside probabilities are computed and items
		with a posterior probabilities < k are pruned.
	:param splitprune: coarse stage used a split-PCFG where discontinuous node
		appear as multiple CFG nodes. Every discontinuous node will result
		in multiple lookups into whitelist to see whether it should be
		allowed on the agenda.
	:param markorigin: in combination with splitprune, coarse labels include an
		integer to distinguish components; e.g., CFG nodes NP*0 and NP*1
		map to the discontinuous node NP_2.
	:param bitpar: prune from bitpar derivations instead of actual chart

	For LCFRS, the white list is indexed as follows:
			:whitelisted: ``whitelist[label][item] == None``
			:blocked: ``item not in whitelist[label]``

	For a CFG, indexing is as follows:
		:whitelisted: ``whitelist[span][label] == None``
		:blocked: ``label not in whitelist[cell]``
	"""
	cdef list whitelist
	cdef ChartItem chartitem
	assert fine.mapping is not NULL
	if splitprune and markorigin:
		assert fine.splitmapping is not NULL
		assert fine.splitmapping[0] is not NULL
	if 0 < k < 1:  # threshold on posterior probabilities
		items, msg = posteriorthreshold(coarsechart, k)
	else:  # construct a list of the k-best nonterminals to prune with
		if bitpar:
			items = bitparkbestitems(coarsechart, k, finecfg)
		else:
			if k == 0:
				coarsechart.filter()
				items = dict.fromkeys(coarsechart.getitems(), None)
			else:
				_ = lazykbest(coarsechart, k, derivs=False)
				items = dict.fromkeys(coarsechart.rankededges, None)
		msg = ('coarse items before pruning: %d; after: %d, '
				'based on %d/%d derivations' % (
				len(coarsechart.getitems()), len(items),
				len(coarsechart.rankededges[coarsechart.root()][:k]), k))
	if finecfg:
		cfgwhitelist = {}
		for item in items:
			label = coarsechart.label(item)
			finespan = coarsechart.asCFGspan(item, fine.nonterminals)
			if finespan not in cfgwhitelist:
				cfgwhitelist[finespan] = {}
			for finelabel in range(1, fine.nonterminals):
				if (fine.mapping[finelabel] == 0
						or fine.mapping[finelabel] == label):
					cfgwhitelist[finespan][finelabel] = None
		return cfgwhitelist, len(items)
	else:
		whitelist = [None] * fine.nonterminals
		kbestspans = [{} for _ in coarsechart.grammar.toid]
		kbestspans[0] = None
		# uses ids of labels in coarse chart
		for item in items:
			# we can use coarsechart here because we only use the item to
			# define a span, which is the same for the fine chart.
			chartitem = coarsechart.asChartItem(item)
			label = chartitem.label
			chartitem.label = 0
			kbestspans[label][chartitem] = None
		# now construct a list which references these coarse items:
		for label in range(fine.nonterminals):
			if splitprune and markorigin and fine.fanout[label] != 1:
				if fine.splitmapping[label] is not NULL:
					whitelist[label] = [kbestspans[fine.splitmapping[label][n]]
							for n in range(fine.fanout[label])]
			else:
				if fine.mapping[label] != 0:
					whitelist[label] = kbestspans[fine.mapping[label]]
		return whitelist, msg


def bitparkbestitems(Chart chart, int k, bint finecfg):
	""" Take a dictionary of CFG derivations as strings, and produce a list of
	ChartItems occurring in those derivations. """
	cdef bint fatitems = chart.lensent >= (sizeof(ULLong) * 8)
	cdef list derivs = chart.rankededges[chart.root()]
	cdef dict items = {}
	for deriv, _ in derivs[:k]:
		t = Tree.parse(deriv, parse_leaf=int)
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
			items[item] = None
	return items


def whitelistfromposteriors(inside, outside, start,
		Grammar coarse, Grammar fine, double threshold,
		bint splitprune, bint markorigin, bint finecfg):
	""" Compute posterior probabilities and prune away cells below some
	threshold. """
	cdef UInt label
	cdef short lensent = start[2]
	assert 0 < threshold < 1, (
			"threshold should be a cutoff for probabilities between 0 and 1.")
	sentprob = inside[0, lensent, start[0]]
	posterior = (inside[:lensent, :lensent + 1]
		* outside[:lensent, :lensent + 1]) / sentprob

	leftidx, rightidx, labels = (posterior[:lensent, :lensent + 1]
		> threshold).nonzero()

	kbestspans = [{} for _ in coarse.toid]
	fatitems = lensent >= (sizeof(ULLong) * 8)
	for label, left, right in zip(labels, leftidx, rightidx):
		if finecfg:
			ei = cellidx(left, right, lensent, fine.nonterminals)
		elif fatitems:
			ei = CFGtoFatChartItem(0, left, right)
		else:
			ei = CFGtoSmallChartItem(0, left, right)
		kbestspans[label][ei] = None

	if finecfg:
		whitelist = {}
		for left in range(start[2]):
			for right in range(left + 1, start[2] + 1):
				span = cellidx(left, right, lensent, fine.nonterminals)
				cell = whitelist[span] = {}
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
	msg = ('coarse items before pruning=%d; filtered: %d;'
			' pruned: %d; sentprob=%g' % (
			unfiltered, numitems, numremain, sentprob))
	return whitelist, msg


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


def posteriorthreshold(Chart chart, double threshold):
	""" Given a chart containing a parse forest, compute inside and outside
	probabilities. Results are stored in the chart.
	Compute posterior probabilities and prune away cells below some threshold.

	:returns: dictionary of remaining items
	"""
	assert 0 < threshold < 1, (
			'threshold should be a cutoff for probabilities between 0 and 1.')
	assert chart.itemsinorder, 'need list of chart items in topological order.'
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
	assert sentprob, sentprob
	posterior = {item: None for item in chart.getitems()
			if chart.inside[item] * chart.outside[item] / sentprob
			> threshold}

	unfiltered = len(chart.getitems())
	numitems = len(chart.outside)
	numremain = len(posterior)
	msg = ('coarse items before pruning=%d; filtered: %d;'
			' pruned: %d; sentprob=%g' % (
			unfiltered, numitems, numremain, sentprob))
	return posterior, msg


def getinside(Chart chart):
	""" Compute inside probabilities for a chart given its parse forest. """
	cdef size_t n
	cdef Edges edges
	cdef Edge *edge
	# this needs to be bottom up, so need order in which items were added
	# currently separate list, chart.itemsinorder
	# NB: sorting items by length is not enough,
	# unaries have to be in the right order...

	# choices for probs:
	# - normal => underflow (current)
	# - logprobs => loss of precision
	# - normal, scaled => how?

	# packing parse forest:
	# revitems = {item: n for n, item in self.itemsinorder}
	# now self.inside[n] and self.outside[n] can be double arrays.
	chart.inside = dict.fromkeys(chart.getitems(), 0.0)

	# traverse items in bottom-up order
	for item in chart.itemsinorder:
		for edges in chart.getedges(item):
			for n in range(edges.len):
				edge = &(edges.data[n])
				if edge.rule is NULL:
					label = chart.label(item)
					word = chart.sent[chart.lexidx(item, edge)]
					prob = (<LexicalRule>chart.grammar.lexicalbylhs[
							label][word]).prob
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
				#chart.addprob(item, prob)
				chart.inside[item] += prob


def getoutside(Chart chart):
	""" Compute outside probabilities for a chart given its parse forest. """
	cdef size_t n
	cdef Edges edges
	cdef Edge *edge
	cdef double outsideprob
	#cdef double sentprob = chart.inside[chart.root()]
	# traverse items in top-down order
	# could use list with idx of item in itemsinorder
	chart.outside = dict.fromkeys(chart.getitems(), 0.0)
	chart.outside[chart.root()] = 1.0
	for item in reversed(chart.itemsinorder):
		# can we define outside[item] simply as sentprob - inside[item] ?
		# chart.outside[item] = sentprob - chart.inside[item]
		for edges in chart.getedges(item):
			for n in range(edges.len):
				edge = &(edges.data[n])
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


def doctf(coarse, fine, sent, tree, k, split, verbose=False):
	""" Test coarse-to-fine methods on a sentence. """
	import plcfrs
	from disambiguation import marginalize
	from treetransforms import canonicalize, removefanoutmarkers
	from math import exp as pyexp
	sent, tags = zip(*sent)
	print(" C O A R S E ", end='')
	chart, _ = plcfrs.parse(sent, coarse, tags=tags)
	if chart:
		mpp, _, _ = marginalize("mpp", chart, coarse, 10)
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
		#print(chart)
	l, _ = prunechart(chart, fine, k, split, True, False, False)
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
	print(" F I N E ", end='')
	chart2, _ = plcfrs.parse(sent, fine, tags=tags, whitelist=l,
			splitprune=split, markorigin=True)
	if chart2:
		mpp, _, _ = marginalize("mpp", chart2, fine, 10)
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
		#xp = set((coarse.tolabel[a.label], a.vec) for a in chart.keys()
		#		if chart[a])
		#xpp = set((fine.tolabel[a.label], a.vec) for a in chart2.keys()
		#		if chart2[a])
		#print("difference:")
		#for a in xp - xpp:
		#	if "*" not in a[0]:
		#		print(a[0], bin(a[1]))
		#print("\nfine chart:")
		#for a in xpp:
		#	if "@" not in a[0]:
		#		print(a[0], bin(a[1]))
		#print(chart2)


def test():
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
	corpus = NegraCorpusReader('.', 'alpinosample.export')
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
		for (_, yf), _ in treebankgrammar([t], [s]):
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
