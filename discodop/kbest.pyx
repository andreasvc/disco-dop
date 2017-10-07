"""Extract the k-best derivations from a probabilistic parse forest.

Implementation of Huang & Chiang (2005): Better k-best parsing.
http://www.cis.upenn.edu/~lhuang3/huang-iwpt-correct.pdf"""
from __future__ import print_function
from operator import itemgetter
from .containers import Grammar

cimport cython
from libcpp.string cimport string
from libc.stdio cimport sprintf
from libc.stdlib cimport abort
include "constants.pxi"

cdef RankedEdgeAgenda[Prob] getcandidates(Chart chart, ItemNo v, int k):
	""":returns: a heap with up to k candidate arcs starting from vertex v."""
	# NB: the priority queue should either do a stable sort, or should
	# sort on rank vector as well to have ties resolved in FIFO order;
	# otherwise the sequence (0, 0) -> (1, 0) -> (1, 1) -> (0, 1) -> (1, 1)
	# can occur (given that the first two have probability x and the latter
	# three probability y), in which case insertion order should count.
	# Otherwise (1, 1) ends up in chart.rankededges[v] after which (0, 1)
	# generates it as a neighbor and puts it in cand[v] for a second time.
	cdef RankedEdgeAgenda[Prob] agenda
	cdef vector[pair[RankedEdge, Prob]] entries
	cdef pair[RankedEdge, Prob] entry
	cdef RankedEdge re
	cdef Edge e
	cdef Prob prob
	# loop over edges
	# compute viterbi prob from rule.prob + viterbi probs of children
	for e in chart.parseforest[v]:
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
		re = RankedEdge(v, e, left, right)
		entry.first = re
		entry.second = prob
		entries.push_back(entry)
	# FIXME: can we guarantee that 'entries' does not have duplicates?
	agenda.kbest_entries(entries, k)
	return agenda


cdef lazykthbest(ItemNo v, int k, int k1, agendas_type& cand, Chart chart,
		RankedEdgeSet& explored, int depthlimit):
	""""Explore up to *k*-best derivations headed by vertex *v*.

	:param k1: the global k, with ``k <= k1``
	:param cand: contains a queue of edges to consider for each vertex
	:param explored: a set of all edges already visited."""
	cdef pair[RankedEdge, Prob] entry
	cdef RankedEdge ej, ej1
	cdef int ji
	cdef bint inrankededges = chart.rankededges[v].size() != 0
	# FIXME: cand could be a vector; but need to distinguish first visit from
	# empty agenda.
	# first visit of vertex v?
	if cand.find(v) == cand.end():
		# initialize the heap
		cand[v] = getcandidates(chart, v, k1)
	while not inrankededges or <int>chart.rankededges[v].size() < k:
		inrankededges = inrankededges or chart.rankededges[v].size() != 0
		if inrankededges:
			# last derivation
			entry = chart.rankededges[v][chart.rankededges[v].size() - 1]
			ej = entry.first
			# update the heap, adding the successors of last derivation
			# start of inlined lazynext(v, ej, k1, cand, chart, explored)
			for i in range(2):  # add the |e| neighbors
				if i == 0 and ej.left >= 0:
					ei = chart.left(ej)
					ej1 = RankedEdge(v, ej.edge, ej.left + 1, ej.right)
				elif i == 1 and ej.right >= 0:
					ei = chart.right(ej)
					ej1 = RankedEdge(v, ej.edge, ej.left, ej.right + 1)
				else:
					break
				ji = ej1.left if i == 0 else ej1.right
				if depthlimit > 0:
					# recursively solve a subproblem
					# NB: increment j[i] again, j is zero-based and k is not
					lazykthbest(ei, ji + 1, k1, cand, chart, explored,
							depthlimit - 1)
					# if it exists and is not in heap yet
					if (chart.rankededges[ei].size() != 0
							and ji < <int>chart.rankededges[ei].size()
							and explored.count(ej1) == 0):
						# 	and cand[ej1.head]): gives duplicates
						# add it to the heap
						cand[v].setitem(ej1, getprob(chart, v, ej1))
						explored.insert(ej1)
			# end of lazynext
		if cand[v].empty():
			break
		# get the next best derivation and delete it from the heap
		entry = cand[v].pop()
		chart.rankededges[v].push_back(entry)


cdef inline Prob getprob(Chart chart, ItemNo v, RankedEdge& ej) except -1:
	"""Get subtree probability of ``ej``.

	Try looking in ``chart.rankededges``, or else use viterbi probability."""
	cdef Prob prob = ej.edge.rule.prob
	ei = chart.left(ej)
	if ej.left == 0:
		prob += chart.subtreeprob(ei)
	elif chart.rankededges[ei].size():
		prob += chart.rankededges[ei][ej.left].second
	else:
		raise ValueError('non-zero rank vector %d not in explored '
				'derivations for %s' % (ej.right, chart.itemstr(v)))
	if ej.right == -1:
		return prob
	ei = chart.right(ej)
	if ej.right == 0:
		prob += chart.subtreeprob(ei)
	elif chart.rankededges[ei].size():
		prob += chart.rankededges[ei][ej.right].second
	else:
		raise ValueError('non-zero rank vector %d not in explored '
				'derivations for %s' % (ej.right, chart.itemstr(v)))
	return prob


cdef int explorederivation(ItemNo v, RankedEdge& ej, Chart chart,
		RankedEdgeSet& explored, int depthlimit) except -2:
	"""Traverse derivation to ensure all 1-best RankedEdges are present.

	:returns: True when ``ej`` is a valid, complete derivation."""
	cdef pair[RankedEdge, Prob] entry
	cdef RankedEdgeAgenda[Prob] tmp
	if depthlimit <= 0:  # to prevent cycles
		return False
	if ej.edge.rule is NULL:
		return True
	if ej.left != -1:
		leftitem = chart.left(ej)
		if not chart.rankededges[leftitem].size():
			assert ej.left == 0, '%d-best edge for %s of left item missing' % (
						ej.left, chart.itemstr(v))
			tmp = getcandidates(chart, leftitem, 1)
			if tmp.size() < 1:
				abort()
			entry = tmp.pop()
			chart.rankededges[leftitem].push_back(entry)
			explored.insert(entry.first)
		if not explorederivation(leftitem,
				chart.rankededges[leftitem][ej.left].first,
				chart, explored, depthlimit - 1):
			return False
	if ej.right != -1:
		rightitem = chart.right(ej)
		if not chart.rankededges[rightitem].size():
			assert ej.right == 0, (('%d-best edge for right child '
					'of %s missing') % (ej.right, chart.itemstr(v)))
			tmp = getcandidates(chart, rightitem, 1)
			if tmp.size() < 1:
				abort()
			entry = tmp.pop()
			chart.rankededges[rightitem].push_back(entry)
			explored.insert(entry.first)
		return explorederivation(rightitem,
				chart.rankededges[rightitem][ej.right].first,
				chart, explored, depthlimit - 1)
	return True


cdef inline _getderiv(string &result, ItemNo v, RankedEdge& ej, Chart chart):
	"""Auxiliary function for ``getderiv()``.

	:param result: provide an empty list for the initial call."""
	cdef RankedEdge rankededge
	cdef Label label = chart.label(v)
	cdef char buf[10]
	cdef int retval
	result.append(b'(')
	result.append(chart.grammar.tolabel.ob[label])
	if ej.edge.rule is NULL:  # lexical rule, left child is terminal
		# C++ way to convert int to string requires streams, C++11, or boost
		retval = sprintf(buf, ' %d)', chart.lexidx(ej.edge))
		if retval <= 0:
			raise ValueError('sprintf error; return value %d' % retval)
		result.append(buf)
	else:
		result.append(b' ')
		item = chart.left(ej)
		rankededge = chart.rankededges[item][ej.left].first
		_getderiv(result, item, rankededge, chart)
		if ej.right != -1:
			item = chart.right(ej)
			result.append(b' ')
			rankededge = chart.rankededges[item][ej.right].first
			_getderiv(result, item, rankededge, chart)
		result.append(b')')


cdef string getderiv(ItemNo v, RankedEdge ej, Chart chart):
	"""Convert a RankedEdge to a string with a tree in bracket notation.

	A RankedEdge consists of an edge and a rank tuple: ``(e, j)`` notation
	('derivation with backpointers'). For example, given an edge based on the
	rule ``S => NP VP`` and the tuple ``(2, 1)``, this identifies a derivation
	headed by S and having the 2nd best NP and the 1st best VP as children."""
	cdef string result
	_getderiv(result, v, ej, chart)
	return result  # result.decode('utf8')


def lazykbest(Chart chart, int k, bint derivs=True):
	"""Wrapper function to run ``lazykthbest``.

	Produces the ranked chart, as well as derivations as strings (when
	``derivs`` is True). ``chart.parseforest`` should be a monotone hypergraph;
	should be acyclic unless probabilities resolve the cycles (maybe nonzero
	weights for unary productions are sufficient?).

	:param k: the number of derivations to enumerate."""
	cdef agendas_type cand
	cdef RankedEdgeSet explored
	cdef pair[RankedEdge, Prob] entry
	cdef vector[pair[RankedEdge, Prob]] tmp
	cdef ItemNo root = chart.root()
	cdef int n = 0
	if root == 0:
		raise ValueError('kbest: no complete derivation in chart')
	chart.rankededges.clear()
	chart.rankededges.resize(chart.parseforest.size())
	lazykthbest(root, k, k, cand, chart, explored, MAX_DEPTH)
	for entry in chart.rankededges[root]:
		if explorederivation(root, entry.first, chart, explored, MAX_DEPTH):
			tmp.push_back(entry)
		n += 1
		if n >= k:
			break
	chart.rankededges[root] = tmp
	if derivs:
		# for entry in chart.rankededges[root]:
		# 	chart.derivations.push_back(getderiv(root, entry.first, chart))
		return [(getderiv(root, entry.first, chart).decode('utf8'),
				entry.second) for entry in chart.rankededges[root]]
	return None

__all__ = ['lazykbest']
