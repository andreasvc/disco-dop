"""Extract the k-best derivations from a probabilistic parse forest.

Implementation of Huang & Chiang (2005): Better k-best parsing."""
from __future__ import print_function
from operator import itemgetter
from discodop.plcfrs import Agenda
from discodop.containers import ChartItem, RankedEdge, Grammar

cimport cython
from cpython.float cimport PyFloat_AS_DOUBLE
from discodop.containers cimport ChartItem, SmallChartItem, FatChartItem, \
		Grammar, Rule, Chart, Edges, Edge, RankedEdge, \
		new_RankedEdge, UInt, UChar, \
		CFGtoSmallChartItem, CFGtoFatChartItem
from discodop.pcfg cimport CFGChart, DenseCFGChart, SparseCFGChart
from discodop.plcfrs cimport Entry, Agenda, nsmallest, \
		LCFRSChart, SmallLCFRSChart, FatLCFRSChart


cdef getcandidates(Chart chart, v, int k):
	""":returns: a heap with up to k candidate arcs starting from vertex v."""
	# NB: the priority queue should either do a stable sort, or should
	# sort on rank vector as well to have ties resolved in FIFO order;
	# otherwise the sequence (0, 0) -> (1, 0) -> (1, 1) -> (0, 1) -> (1, 1)
	# can occur (given that the first two have probability x and the latter
	# three probability y), in which case insertion order should count.
	# Otherwise (1, 1) ends up in chart.rankededges[v] after which (0. 1)
	# generates it as a neighbor and puts it in cand[v] for a second time.
	cdef Edges edges
	cdef Edge *e
	cdef double prob
	results = []
	# loop over blocks of edges
	# compute viterbi prob from rule.prob + viterbi probs of children
	for pyedges in chart.getedges(v):
		edges = <Edges>pyedges
		for n in range(edges.len):
			e = &(edges.data[n])
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
			results.append((re, prob))
	return Agenda(nsmallest(k, results, key=itemgetter(1)))


@cython.wraparound(True)
cpdef inline lazykthbest(v, int k, int k1, dict cand, Chart chart,
		set explored):
	cdef Entry entry
	cdef RankedEdge ej
	# k1 is the global k
	# first visit of vertex v?
	if v not in cand:
		# initialize the heap
		cand[v] = getcandidates(chart, v, k1)
	while v not in chart.rankededges or len(chart.rankededges[v]) < k:
		if v in chart.rankededges:
			# last derivation
			entry = chart.rankededges[v][-1]
			ej = entry.key
			# update the heap, adding the successors of last derivation
			lazynext(v, ej, k1, cand, chart, explored)
		# get the next best derivation and delete it from the heap
		if cand[v]:
			chart.rankededges.setdefault(v, []).append(
					(<Agenda>cand[v]).popentry())
		else:
			break


cdef inline lazynext(v, RankedEdge ej, int k1, dict cand, Chart chart,
		set explored):
	cdef RankedEdge ej1
	# add the |e| neighbors
	for i in range(2):
		if i == 0 and ej.left >= 0:
			ei = chart.copy(chart.left(ej))
			ej1 = new_RankedEdge(v, ej.edge, ej.left + 1, ej.right)
		elif i == 1 and ej.right >= 0:
			ei = chart.copy(chart.right(ej))
			ej1 = new_RankedEdge(v, ej.edge, ej.left, ej.right + 1)
		else:
			break
		# recursively solve a subproblem
		# NB: increment j1[i] again because j is zero-based and k is not
		lazykthbest(ei, (ej1.right if i else ej1.left) + 1, k1,
							cand, chart, explored)
		# if it exists and is not in heap yet
		if ((ei in chart.rankededges and
				(ej1.right if i else ej1.left) < len(chart.rankededges[ei]))
				and ej1 not in explored):  # cand[ej1.head]): gives duplicates
			# add it to the heap
			cand[v][ej1] = getprob(chart, v, ej1)
			explored.add(ej1)

cdef inline double getprob(Chart chart, v, RankedEdge ej) except -1.0:
	"""Get subtree probability of ``ej``.

	Try looking in ``chart.rankededges``, or else use viterbi probability."""
	cdef double prob = ej.edge.rule.prob
	ei = chart.left(ej)
	if ej.left == 0:
		prob += chart.subtreeprob(ei)
	elif ei in chart.rankededges:
		prob += PyFloat_AS_DOUBLE(
				(<Entry>(<list>chart.rankededges[ei])[ej.left]).value)
	else:
		raise ValueError('non-zero rank vector %d not in explored '
				'derivations for %s' % (ej.right, chart.itemstr(v)))
	if ej.right == -1:
		return prob
	ei = chart.right(ej)
	if ej.right == 0:
		prob += chart.subtreeprob(ei)
	elif ei in chart.rankededges:
		prob += PyFloat_AS_DOUBLE(
				(<Entry>(<list>chart.rankededges[ei])[ej.right]).value)
	else:
		raise ValueError('non-zero rank vector %d not in explored '
				'derivations for %s' % (ej.right, chart.itemstr(v)))
	return prob


cdef int explorederivation(v, RankedEdge ej, Chart chart, set explored,
		int depthlimit):
	"""Traverse derivation to ensure all 1-best RankedEdges are present.

	:returns: True when ``ej`` is a valid, complete derivation."""
	cdef Entry entry
	if depthlimit <= 0:  # to prevent cycles
		return False
	if ej.edge.rule is NULL:
		return True
	leftitem = chart.left(ej)
	if ej.left != -1:
		leftitem = chart.copy(leftitem)
		if leftitem not in chart.rankededges:
			assert ej.left == 0, '%d-best edge for %s of left item missing' % (
						ej.left, chart.itemstr(v))
			entry = (<Agenda>getcandidates(chart, leftitem, 1)).popentry()
			chart.rankededges[leftitem] = [entry]
			explored.add(entry.key)
		if not explorederivation(leftitem,
				<RankedEdge>(<Entry>chart.rankededges[leftitem][ej.left]).key,
				chart, explored, depthlimit - 1):
			return False
	rightitem = chart.right(ej)
	if ej.right != -1:
		rightitem = chart.copy(rightitem)
		if rightitem not in chart.rankededges:
			assert ej.right == 0, '%d-best edge for %s of right item missing' % (
						ej.right, chart.itemstr(v))
			entry = (<Agenda>getcandidates(chart, rightitem, 1)).popentry()
			chart.rankededges[rightitem] = [entry]
			explored.add(entry.key)
		return explorederivation(rightitem,
				<RankedEdge>(<Entry>chart.rankededges[rightitem][ej.right]).key,
				chart, explored, depthlimit - 1)
	return True


cpdef inline _getderiv(bytearray result, v, RankedEdge ej, Chart chart,
		bytes debin):
	"""Auxiliary function for ``getderiv()``.

	:param result: provide an empty ``bytearray()`` for the initial call."""
	cdef RankedEdge rankededge
	cdef UInt label = chart.label(v)
	if debin is None or debin not in chart.grammar.tolabel[label]:
		result += b'('
		result += chart.grammar.tolabel[label]
		result += b' '
	if ej.edge.rule is NULL:  # lexical rule, left child is terminal
		result += str(chart.lexidx(ej.edge)).encode('ascii')
	else:
		item = chart.left(ej)
		rankededge = (<Entry>chart.rankededges[item][ej.left]).key
		_getderiv(result, item, rankededge, chart, debin)
		if ej.right != -1:
			item = chart.right(ej)
			result += b' '
			rankededge = (<Entry>chart.rankededges[item][ej.right]).key
			_getderiv(result, item, rankededge, chart, debin)
	if debin is None or debin not in chart.grammar.tolabel[label]:
		result += b')'


def getderiv(v, RankedEdge ej, Chart chart, bytes debin):
	"""Convert a RankedEdge to a string with a tree in bracket notation.

	A RankedEdge consists of an edge and a rank tuple: ``(e, j)`` notation
	('derivation with backpointers'). For example, given an edge based on the
	rule ``S => NP VP`` and the tuple ``(2, 1)``, this identifies a derivation
	headed by S and having the 2nd best NP and the 1st best VP as children.

	:param debin: perform on-the-fly debinarization, identify intermediate
		nodes using the substring ``debin``."""
	cdef bytearray result = bytearray()
	_getderiv(result, v, ej, chart, debin)
	return str(result.decode('ascii'))


def lazykbest(Chart chart, int k, bytes debin=None, bint derivs=True):
	"""Wrapper function to run ``lazykthbest``.

	Produces the ranked chart, as well as derivations as strings (when
	``derivs`` is True). chart is a monotone hypergraph; should be acyclic
	unless probabilities resolve the cycles (maybe nonzero weights for unary
	productions are sufficient?).

	:param k: the number of derivations to enumerate.
	:param debin: debinarize derivations."""
	cdef Entry entry
	cdef set explored = set()
	# assert not chart.rankededges, 'kbest derivations already extracted?'
	chart.rankededges = {}
	derivations = []
	cand = {}
	root = chart.root()
	lazykthbest(root, k, k, cand, chart, explored)
	chart.rankededges[root] = [entry for entry
			in chart.rankededges[root][:k]
			if explorederivation(root, entry.key, chart, explored, 100)]
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
	dcchart.updateprob(3, 0, 3, -log(0.0126000))
	dcchart.updateprob(3, 0, 5, -log(0.0005040))
	dcchart.updateprob(4, 1, 3, -log(0.1260000))
	dcchart.updateprob(4, 1, 5, -log(0.0050400))
	dcchart.updateprob(4, 1, 5, -log(0.0037800))
	dcchart.updateprob(1, 2, 5, -log(0.0072000))
	dcchart.updateprob(2, 3, 5, -log(0.1000000))
	dcchart.updateprob(1, 0, 1, -log(0.1))   # 'astronomers'
	dcchart.updateprob(1, 1, 2, -log(0.04))  # NP => 'saw'
	dcchart.updateprob(5, 1, 2, -log(1))     # V => 'saw'
	dcchart.updateprob(1, 2, 3, -log(0.18))  # 'stars'
	dcchart.updateprob(6, 3, 4, -log(1))     # 'with'
	dcchart.updateprob(1, 4, 5, -log(0.1))   # 'telescopes'
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
	scchart.updateprob(3, 0, 3, -log(0.0126000))
	scchart.updateprob(3, 0, 5, -log(0.0005040))
	scchart.updateprob(4, 1, 3, -log(0.1260000))
	scchart.updateprob(4, 1, 5, -log(0.0050400))
	scchart.updateprob(4, 1, 5, -log(0.0037800))
	scchart.updateprob(1, 2, 5, -log(0.0072000))
	scchart.updateprob(2, 3, 5, -log(0.1000000))
	scchart.updateprob(1, 0, 1, -log(0.1))   # 'astronomers'
	scchart.updateprob(1, 1, 2, -log(0.04))  # NP => 'saw'
	scchart.updateprob(5, 1, 2, -log(1))     # V => 'saw'
	scchart.updateprob(1, 2, 3, -log(0.18))  # 'stars'
	scchart.updateprob(6, 3, 4, -log(1))     # 'with'
	scchart.updateprob(1, 4, 5, -log(0.1))   # 'telescopes'
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

__all__ = ['getderiv', 'lazykbest', 'lazykthbest']
