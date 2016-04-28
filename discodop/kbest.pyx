"""Extract the k-best derivations from a probabilistic parse forest.

Implementation of Huang & Chiang (2005): Better k-best parsing."""
from __future__ import print_function
from operator import itemgetter
from .plcfrs import DoubleAgenda
from .containers import ChartItem, RankedEdge, Grammar

cimport cython
from libc.stdint cimport uint32_t
from .containers cimport ChartItem, SmallChartItem, FatChartItem, \
		Grammar, ProbRule, Chart, Edge, Edges, MoreEdges, RankedEdge, \
		new_RankedEdge, CFGtoSmallChartItem, CFGtoFatChartItem
from .pcfg cimport CFGChart, DenseCFGChart, SparseCFGChart
from .plcfrs cimport DoubleEntry, DoubleAgenda, nsmallest, \
		LCFRSChart, SmallLCFRSChart, FatLCFRSChart, new_DoubleEntry
include "constants.pxi"

cdef DoubleAgenda getcandidates(Chart chart, v, int k):
	""":returns: a heap with up to k candidate arcs starting from vertex v."""
	# NB: the priority queue should either do a stable sort, or should
	# sort on rank vector as well to have ties resolved in FIFO order;
	# otherwise the sequence (0, 0) -> (1, 0) -> (1, 1) -> (0, 1) -> (1, 1)
	# can occur (given that the first two have probability x and the latter
	# three probability y), in which case insertion order should count.
	# Otherwise (1, 1) ends up in chart.rankededges[v] after which (0, 1)
	# generates it as a neighbor and puts it in cand[v] for a second time.
	cdef Edge *e
	cdef Edges edges
	cdef MoreEdges *edgelist
	cdef double prob
	cdef DoubleAgenda agenda = DoubleAgenda()
	cdef size_t n
	entries = []
	# loop over blocks of edges
	# compute viterbi prob from rule.prob + viterbi probs of children
	edges = chart.getedges(v)
	edgelist = edges.head if edges is not None else NULL
	while edgelist is not NULL:
		for n in range(edges.len if edgelist is edges.head
				else EDGES_SIZE):
			e = &(edgelist.data[n])
			if e.rule is NULL:
				# there can only be one lexical edge for this combination of
				# POS tag and terminal, use viterbi probability directly
				prob = chart.subtreeprob(v)
				left = right = -1
			else:
				left = right = 0
				prob = e.rule.prob
				prob += chart.subtreeprob(chart._left(v, e))
				if e.rule.rhs2:  # unary rule?
					prob += chart.subtreeprob(chart._right(v, e))
				else:
					right = -1
			re = new_RankedEdge(v, e, left, right)
			entries.append(new_DoubleEntry(re, prob, n))
		edgelist = edgelist.prev
	agenda.update_entries(nsmallest(k, entries))
	return agenda


cdef lazykthbest(v, int k, int k1, dict cand, Chart chart, set explored,
		int depthlimit):
	""""Explore up to *k*-best derivations headed by vertex *v*.

	:param k1: the global k, with ``k <= k1``
	:param cand: contains a queue of edges to consider for each vertex
	:param explored: a set of all edges already visited."""
	cdef DoubleEntry entry
	cdef RankedEdge ej, ej1
	cdef int ji
	cdef bint inrankededges = v in chart.rankededges
	# first visit of vertex v?
	if v not in cand:
		# initialize the heap
		cand[v] = getcandidates(chart, v, k1)
	while not inrankededges or len(chart.rankededges[v]) < k:
		inrankededges = inrankededges or v in chart.rankededges
		if inrankededges:
			with cython.wraparound(True):
				# last derivation
				entry = chart.rankededges[v][-1]
			ej = entry.key
			# update the heap, adding the successors of last derivation
			# start of inlined lazynext(v, ej, k1, cand, chart, explored)
			for i in range(2):  # add the |e| neighbors
				if i == 0 and ej.left >= 0:
					ei = chart.left(ej)
					ej1 = new_RankedEdge(v, ej.edge, ej.left + 1, ej.right)
				elif i == 1 and ej.right >= 0:
					ei = chart.right(ej)
					ej1 = new_RankedEdge(v, ej.edge, ej.left, ej.right + 1)
				else:
					break
				ji = ej1.left if i == 0 else ej1.right
				if depthlimit > 0:
					# recursively solve a subproblem
					# NB: increment j[i] again, j is zero-based and k is not
					lazykthbest(ei, ji + 1, k1, cand, chart, explored,
							depthlimit - 1)
					# if it exists and is not in heap yet
					if (ei in chart.rankededges
							and ji < len(chart.rankededges[ei])
							and ej1 not in explored):
						# 	and cand[ej1.head]): gives duplicates
						# add it to the heap
						cand[v][ej1] = getprob(chart, v, ej1)
						explored.add(ej1)
			# end of lazynext
		if not cand[v]:
			break
		# get the next best derivation and delete it from the heap
		entry = (<DoubleAgenda>cand[v]).popentry()
		if inrankededges:
			chart.rankededges[v].append(entry)
		else:
			chart.rankededges[v] = [entry]


cdef inline double getprob(Chart chart, v, RankedEdge ej) except -1.0:
	"""Get subtree probability of ``ej``.

	Try looking in ``chart.rankededges``, or else use viterbi probability."""
	cdef double prob = ej.edge.rule.prob
	ei = chart.left(ej)
	if ej.left == 0:
		prob += chart.subtreeprob(ei)
	elif ei in chart.rankededges:
		prob += (<DoubleEntry>(<list>chart.rankededges[ei])[ej.left]).value
	else:
		raise ValueError('non-zero rank vector %d not in explored '
				'derivations for %s' % (ej.right, chart.itemstr(v)))
	if ej.right == -1:
		return prob
	ei = chart.right(ej)
	if ej.right == 0:
		prob += chart.subtreeprob(ei)
	elif ei in chart.rankededges:
		prob += (<DoubleEntry>(<list>chart.rankededges[ei])[ej.right]).value
	else:
		raise ValueError('non-zero rank vector %d not in explored '
				'derivations for %s' % (ej.right, chart.itemstr(v)))
	return prob


cdef int explorederivation(v, RankedEdge ej, Chart chart, set explored,
		int depthlimit):
	"""Traverse derivation to ensure all 1-best RankedEdges are present.

	:returns: True when ``ej`` is a valid, complete derivation."""
	cdef DoubleEntry entry
	if depthlimit <= 0:  # to prevent cycles
		return False
	if ej.edge.rule is NULL:
		return True
	if ej.left != -1:
		leftitem = chart.left(ej)
		if leftitem not in chart.rankededges:
			assert ej.left == 0, '%d-best edge for %s of left item missing' % (
						ej.left, chart.itemstr(v))
			entry = getcandidates(chart, leftitem, 1).popentry()
			chart.rankededges[leftitem] = [entry]
			explored.add(entry.key)
		if not explorederivation(leftitem,
				<RankedEdge>(<DoubleEntry>chart.rankededges[
					leftitem][ej.left]).key,
				chart, explored, depthlimit - 1):
			return False
	if ej.right != -1:
		rightitem = chart.right(ej)
		if rightitem not in chart.rankededges:
			assert ej.right == 0, (('%d-best edge for right child '
					'of %s missing') % (ej.right, chart.itemstr(v)))
			entry = getcandidates(chart, rightitem, 1).popentry()
			chart.rankededges[rightitem] = [entry]
			explored.add(entry.key)
		return explorederivation(rightitem,
				<RankedEdge>(<DoubleEntry>chart.rankededges[
					rightitem][ej.right]).key,
				chart, explored, depthlimit - 1)
	return True


cdef inline _getderiv(list result, v, RankedEdge ej, Chart chart, str debin):
	"""Auxiliary function for ``getderiv()``.

	:param result: provide an empty list for the initial call."""
	cdef RankedEdge rankededge
	cdef uint32_t label = chart.label(v)
	if debin is None or debin not in chart.grammar.tolabel[label]:
		result.append('(')
		result.append(chart.grammar.tolabel[label])
		result.append(' ')
	if ej.edge.rule is NULL:  # lexical rule, left child is terminal
		result.append(str(chart.lexidx(ej.edge)))
	else:
		item = chart.left(ej)
		rankededge = (<DoubleEntry>chart.rankededges[item][ej.left]).key
		_getderiv(result, item, rankededge, chart, debin)
		if ej.right != -1:
			item = chart.right(ej)
			result.append(' ')
			rankededge = (<DoubleEntry>chart.rankededges[item][ej.right]).key
			_getderiv(result, item, rankededge, chart, debin)
	if debin is None or debin not in chart.grammar.tolabel[label]:
		result.append(')')


def getderiv(v, RankedEdge ej, Chart chart, str debin):
	"""Convert a RankedEdge to a string with a tree in bracket notation.

	A RankedEdge consists of an edge and a rank tuple: ``(e, j)`` notation
	('derivation with backpointers'). For example, given an edge based on the
	rule ``S => NP VP`` and the tuple ``(2, 1)``, this identifies a derivation
	headed by S and having the 2nd best NP and the 1st best VP as children.

	:param debin: perform on-the-fly debinarization, identify intermediate
		nodes using the substring ``debin``."""
	cdef list result = []
	_getderiv(result, v, ej, chart, debin)
	return ''.join(result)


def lazykbest(Chart chart, int k, str debin=None, bint derivs=True):
	"""Wrapper function to run ``lazykthbest``.

	Produces the ranked chart, as well as derivations as strings (when
	``derivs`` is True). chart is a monotone hypergraph; should be acyclic
	unless probabilities resolve the cycles (maybe nonzero weights for unary
	productions are sufficient?).

	:param k: the number of derivations to enumerate.
	:param debin: debinarize derivations."""
	cdef DoubleEntry entry
	cdef set explored = set()
	# assert not chart.rankededges, 'kbest derivations already extracted?'
	chart.rankededges = {}
	derivations = []
	cand = {}
	root = chart.root()
	lazykthbest(root, k, k, cand, chart, explored, MAX_DEPTH)
	chart.rankededges[root] = [entry for entry
			in chart.rankededges[root][:k]
			if explorederivation(root, entry.key, chart, explored, MAX_DEPTH)]
	if derivs:
		root = chart.root()
		derivations = [(getderiv(root, entry.key, chart, debin), entry.value)
				for entry in chart.rankededges[root]]
	return derivations, explored


def test():
	"""Demonstration of k-best algorithm."""
	from math import log, exp
	cdef DenseCFGChart dcchart
	cdef SparseCFGChart scchart
	cdef SmallLCFRSChart slchart
	cdef FatLCFRSChart flchart
	cdef Grammar gr
	k = 10
	gr = Grammar([
			((('NP', 'NP', 'PP'), ((0, 1), )), 0.4),
			((('PP', 'P', 'NP'), ((0, 1), )), 1),
			((('S', 'NP', 'VP'), ((0, 1), )), 1),
			((('VP', 'V', 'NP'), ((0, 1), )), 0.7),
			((('VP', 'VP', 'PP'), ((0, 1), )), 0.3),
			((('NP', 'Epsilon'), ('astronomers', )), 0.1),
			((('NP', 'Epsilon'), ('ears', )), 0.18),
			((('V', 'Epsilon'), ('saw', )), 1),
			((('NP', 'Epsilon'), ('saw', )), 0.04),
			((('NP', 'Epsilon'), ('stars', )), 0.18),
			((('NP', 'Epsilon'), ('telescopes', )), 0.1),
			((('P', 'Epsilon'), ('with', )), 1)], start='S')
	sent = "astronomers saw stars with telescopes".split()

	print('\ndense pcfg chart')
	dcchart = DenseCFGChart(gr, sent)
	dcchart.addedge(3, 0, 3, 1, &(gr.bylhs[0][2]))
	dcchart.addedge(3, 0, 5, 1, &(gr.bylhs[0][2]))
	dcchart.addedge(4, 1, 3, 2, &(gr.bylhs[0][3]))
	dcchart.addedge(4, 1, 5, 2, &(gr.bylhs[0][3]))
	dcchart.addedge(4, 1, 5, 3, &(gr.bylhs[0][4]))
	dcchart.addedge(1, 2, 5, 3, &(gr.bylhs[0][0]))
	dcchart.addedge(2, 3, 5, 4, &(gr.bylhs[0][1]))
	dcchart.addedge(1, 0, 1, 1, NULL)
	dcchart.addedge(1, 1, 2, 2, NULL)
	dcchart.addedge(5, 1, 2, 2, NULL)
	dcchart.addedge(1, 2, 3, 3, NULL)
	dcchart.addedge(6, 3, 4, 4, NULL)
	dcchart.addedge(1, 4, 5, 5, NULL)
	dcchart.updateprob(3, 0, 3, -log(0.0126000), 0.0)
	dcchart.updateprob(3, 0, 5, -log(0.0005040), 0.0)
	dcchart.updateprob(4, 1, 3, -log(0.1260000), 0.0)
	dcchart.updateprob(4, 1, 5, -log(0.0050400), 0.0)
	dcchart.updateprob(4, 1, 5, -log(0.0037800), 0.0)
	dcchart.updateprob(1, 2, 5, -log(0.0072000), 0.0)
	dcchart.updateprob(2, 3, 5, -log(0.1000000), 0.0)
	dcchart.updateprob(1, 0, 1, -log(0.1), 0.0)   # 'astronomers'
	dcchart.updateprob(1, 1, 2, -log(0.04), 0.0)  # NP => 'saw'
	dcchart.updateprob(5, 1, 2, -log(1), 0.0)     # V => 'saw'
	dcchart.updateprob(1, 2, 3, -log(0.18), 0.0)  # 'stars'
	dcchart.updateprob(6, 3, 4, -log(1), 0.0)     # 'with'
	dcchart.updateprob(1, 4, 5, -log(0.1), 0.0)   # 'telescopes'
	derivations = lazykbest(dcchart, k)[0]
	print(dcchart)
	for a, p in derivations:
		print('%f %s' % (exp(-p), a))

	print('\nsparse pcfg chart')
	scchart = SparseCFGChart(gr, sent)
	scchart.addedge(3, 0, 3, 1, &(gr.bylhs[0][2]))
	scchart.addedge(3, 0, 5, 1, &(gr.bylhs[0][2]))
	scchart.addedge(4, 1, 3, 2, &(gr.bylhs[0][3]))
	scchart.addedge(4, 1, 5, 2, &(gr.bylhs[0][3]))
	scchart.addedge(4, 1, 5, 3, &(gr.bylhs[0][4]))
	scchart.addedge(1, 2, 5, 3, &(gr.bylhs[0][0]))
	scchart.addedge(2, 3, 5, 4, &(gr.bylhs[0][1]))
	scchart.addedge(1, 0, 1, 1, NULL)
	scchart.addedge(1, 1, 2, 2, NULL)
	scchart.addedge(5, 1, 2, 2, NULL)
	scchart.addedge(1, 2, 3, 3, NULL)
	scchart.addedge(6, 3, 4, 4, NULL)
	scchart.addedge(1, 4, 5, 5, NULL)
	scchart.updateprob(3, 0, 3, -log(0.0126000), 0.0)
	scchart.updateprob(3, 0, 5, -log(0.0005040), 0.0)
	scchart.updateprob(4, 1, 3, -log(0.1260000), 0.0)
	scchart.updateprob(4, 1, 5, -log(0.0050400), 0.0)
	scchart.updateprob(4, 1, 5, -log(0.0037800), 0.0)
	scchart.updateprob(1, 2, 5, -log(0.0072000), 0.0)
	scchart.updateprob(2, 3, 5, -log(0.1000000), 0.0)
	scchart.updateprob(1, 0, 1, -log(0.1), 0.0)   # 'astronomers'
	scchart.updateprob(1, 1, 2, -log(0.04), 0.0)  # NP => 'saw'
	scchart.updateprob(5, 1, 2, -log(1), 0.0)     # V => 'saw'
	scchart.updateprob(1, 2, 3, -log(0.18), 0.0)  # 'stars'
	scchart.updateprob(6, 3, 4, -log(1), 0.0)     # 'with'
	scchart.updateprob(1, 4, 5, -log(0.1), 0.0)   # 'telescopes'
	derivations = lazykbest(scchart, k)[0]
	# print(scchart)
	for a, p in derivations:
		print('%f %s' % (exp(-p), a))

	print('\nsmall lcfrs chart')
	slchart = SmallLCFRSChart(gr, sent)
	slchart.addedge(CFGtoSmallChartItem(3, 0, 3), CFGtoSmallChartItem(0, 0, 1),
			&(gr.bylhs[0][2]))
	slchart.addedge(CFGtoSmallChartItem(3, 0, 5), CFGtoSmallChartItem(0, 0, 1),
			&(gr.bylhs[0][2]))
	slchart.addedge(CFGtoSmallChartItem(4, 1, 3), CFGtoSmallChartItem(0, 1, 2),
			&(gr.bylhs[0][3]))
	slchart.addedge(CFGtoSmallChartItem(4, 1, 5), CFGtoSmallChartItem(0, 1, 2),
			&(gr.bylhs[0][3]))
	slchart.addedge(CFGtoSmallChartItem(4, 1, 5), CFGtoSmallChartItem(0, 1, 3),
			&(gr.bylhs[0][4]))
	slchart.addedge(CFGtoSmallChartItem(1, 2, 5), CFGtoSmallChartItem(0, 2, 3),
			&(gr.bylhs[0][0]))
	slchart.addedge(CFGtoSmallChartItem(2, 3, 5), CFGtoSmallChartItem(0, 3, 4),
			&(gr.bylhs[0][1]))
	slchart.addlexedge(CFGtoSmallChartItem(1, 0, 1), 0)
	slchart.addlexedge(CFGtoSmallChartItem(1, 1, 2), 1)
	slchart.addlexedge(CFGtoSmallChartItem(5, 1, 2), 1)
	slchart.addlexedge(CFGtoSmallChartItem(1, 2, 3), 2)
	slchart.addlexedge(CFGtoSmallChartItem(6, 3, 4), 3)
	slchart.addlexedge(CFGtoSmallChartItem(1, 4, 5), 4)
	slchart.updateprob(CFGtoSmallChartItem(3, 0, 3), -log(0.0126000))
	slchart.updateprob(CFGtoSmallChartItem(3, 0, 5), -log(0.0005040))
	slchart.updateprob(CFGtoSmallChartItem(4, 1, 3), -log(0.1260000))
	slchart.updateprob(CFGtoSmallChartItem(4, 1, 5), -log(0.0050400))
	slchart.updateprob(CFGtoSmallChartItem(4, 1, 5), -log(0.0037800))
	slchart.updateprob(CFGtoSmallChartItem(1, 2, 5), -log(0.0072000))
	slchart.updateprob(CFGtoSmallChartItem(2, 3, 5), -log(0.1000000))
	slchart.updateprob(CFGtoSmallChartItem(1, 0, 1), -log(0.1))   # 'astronomers'
	slchart.updateprob(CFGtoSmallChartItem(1, 1, 2), -log(0.04))  # NP => 'saw'
	slchart.updateprob(CFGtoSmallChartItem(5, 1, 2), -log(1))     # V => 'saw'
	slchart.updateprob(CFGtoSmallChartItem(1, 2, 3), -log(0.18))  # 'stars'
	slchart.updateprob(CFGtoSmallChartItem(6, 3, 4), -log(1))     # 'with'
	slchart.updateprob(CFGtoSmallChartItem(1, 4, 5), -log(0.1))   # 'telescopes'
	derivations = lazykbest(slchart, k)[0]
	# print(slchart)
	for a, p in derivations:
		print('%f %s' % (exp(-p), a))

	print('\nfat lcfrs chart')
	# this hack is needed because fatchart keeps pointers to spans of items,
	# so garbage collection needs to be prevented.
	items = {a: CFGtoFatChartItem(0, a[0], a[1]) for a in
			((1, 2), (0, 1), (3, 4), (2, 3), (1, 2), (1, 3), (0, 1))}
	flchart = FatLCFRSChart(gr, sent)
	flchart.addedge(CFGtoFatChartItem(3, 0, 3), items[0, 1], &(gr.bylhs[0][2]))
	flchart.addedge(CFGtoFatChartItem(3, 0, 5), items[0, 1], &(gr.bylhs[0][2]))
	flchart.addedge(CFGtoFatChartItem(4, 1, 3), items[1, 2], &(gr.bylhs[0][3]))
	flchart.addedge(CFGtoFatChartItem(4, 1, 5), items[1, 2], &(gr.bylhs[0][3]))
	flchart.addedge(CFGtoFatChartItem(4, 1, 5), items[1, 3], &(gr.bylhs[0][4]))
	flchart.addedge(CFGtoFatChartItem(1, 2, 5), items[2, 3], &(gr.bylhs[0][0]))
	flchart.addedge(CFGtoFatChartItem(2, 3, 5), items[3, 4], &(gr.bylhs[0][1]))
	flchart.addlexedge(CFGtoFatChartItem(1, 0, 1), 0)
	flchart.addlexedge(CFGtoFatChartItem(1, 1, 2), 1)
	flchart.addlexedge(CFGtoFatChartItem(5, 1, 2), 1)
	flchart.addlexedge(CFGtoFatChartItem(1, 2, 3), 2)
	flchart.addlexedge(CFGtoFatChartItem(6, 3, 4), 3)
	flchart.addlexedge(CFGtoFatChartItem(1, 4, 5), 4)
	flchart.updateprob(CFGtoFatChartItem(3, 0, 3), -log(0.0126000))
	flchart.updateprob(CFGtoFatChartItem(3, 0, 5), -log(0.0005040))
	flchart.updateprob(CFGtoFatChartItem(4, 1, 3), -log(0.1260000))
	flchart.updateprob(CFGtoFatChartItem(4, 1, 5), -log(0.0050400))
	flchart.updateprob(CFGtoFatChartItem(4, 1, 5), -log(0.0037800))
	flchart.updateprob(CFGtoFatChartItem(1, 2, 5), -log(0.0072000))
	flchart.updateprob(CFGtoFatChartItem(2, 3, 5), -log(0.1000000))
	flchart.updateprob(CFGtoFatChartItem(1, 0, 1), -log(0.1))   # 'astronomers'
	flchart.updateprob(CFGtoFatChartItem(1, 1, 2), -log(0.04))  # NP => 'saw'
	flchart.updateprob(CFGtoFatChartItem(5, 1, 2), -log(1))     # V => 'saw'
	flchart.updateprob(CFGtoFatChartItem(1, 2, 3), -log(0.18))  # 'stars'
	flchart.updateprob(CFGtoFatChartItem(6, 3, 4), -log(1))     # 'with'
	flchart.updateprob(CFGtoFatChartItem(1, 4, 5), -log(0.1))   # 'telescopes'
	derivations = lazykbest(flchart, k)[0]
	# print(lchart)
	for a, p in derivations:
		print('%f %s' % (exp(-p), a))
	assert (len(flchart.rankededges[flchart.root()])
			== len(set(flchart.rankededges[flchart.root()])))
	assert len(derivations) == len(set(derivations))

__all__ = ['getderiv', 'lazykbest']
