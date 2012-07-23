""" Implementation of Huang & Chiang (2005): Better k-best parsing
"""
from math import exp
from agenda import Agenda
from containers import ChartItem, Edge, RankedEdge
from operator import itemgetter

from agenda cimport Entry, Agenda, nsmallest
from containers cimport ChartItem, Edge, RankedEdge

unarybest = (0, )
binarybest = (0, 0)

cdef inline getcandidates(dict chart, ChartItem v, int k):
	""" Return a heap with up to k candidate arcs starting from vertex v """
	# NB: the priority queue should either do a stable sort, or should
	# sort on rank vector as well to have ties resolved in FIFO order;
	# otherwise the sequence (0, 0) -> (1, 0) -> (1, 1) -> (0, 1) -> (1, 1)
	# can occur (given that the first two have probability x and the latter
	# three probability y), in which case insertion order should count.
	# Otherwise (1, 1) ends up in D[v] after which (0. 1) generates it
	# as a neighbor and puts it in cand[v] for a second time.
	cdef Edge edge
	if v not in chart: return Agenda() #raise error?
	return Agenda(
		[(RankedEdge(v, edge, 0, 0 if edge.right.label else -1), edge.inside)
						for edge in nsmallest(k, chart[v])])

cpdef inline lazykthbest(ChartItem v, int k, int k1, dict D, dict cand,
		dict chart, set explored):
	cdef Entry entry
	cdef RankedEdge ej
	# k1 is the global k
	# first visit of vertex v?
	if v not in cand:
		# initialize the heap
		cand[v] = getcandidates(chart, v, k1)
	while v not in D or len(D[v]) < k:
		if v in D:
			# last derivation
			entry = D[v][-1]
			ej = entry.key
			# update the heap, adding the successors of last derivation
			lazynext(ej, k1, D, cand, chart, explored)
		# get the next best derivation and delete it from the heap
		if cand[v]:
			D.setdefault(v, []).append(cand[v].popentry())
		else: break
	return D

cdef inline lazynext(RankedEdge ej, int k1, dict D, dict cand, dict chart,
		set explored):
	cdef RankedEdge ej1
	cdef double prob
	# add the |e| neighbors
	for i in range(2):
		if i == 0:
			ei = ej.edge.left
			ej1 = RankedEdge(ej.head, ej.edge, ej.left + 1, ej.right)
		elif i == 1 and ej.right >= 0: #edge.right.label:
			ei = ej.edge.right
			ej1 = RankedEdge(ej.head, ej.edge, ej.left, ej.right + 1)
		else: break
		# recursively solve a subproblem
		# NB: increment j1[i] again because j is zero-based and k is not
		lazykthbest(ei, (ej1.right if i else ej1.left) + 1, k1,
							D, cand, chart, explored)
		# if it exists and is not in heap yet
		if ((ei in D and (ej1.right if i else ej1.left) < len(D[ei]))
			and ej1 not in explored): #cand[ej1.head]): <= gives duplicates
			prob = getprob(chart, D, ej1)
			# add it to the heap
			cand[ej1.head][ej1] = prob
			explored.add(ej1)

cdef inline double getprob(dict chart, dict D, RankedEdge ej) except -1.0:
	cdef ChartItem ei
	cdef Edge e
	cdef Entry entry
	cdef double result, prob
	e = ej.edge
	if e.left in D: entry = D[e.left][ej.left]; prob = entry.value
	elif ej.left == 0: edge = chart[e.left][0]; prob = edge.inside
	else: raise ValueError(
		"non-zero rank vector not part of explored derivations")
	result = e.prob + prob
	if ej.right >= 0: #if e.right.label:
		if e.right in D: entry = D[e.right][ej.right]; prob = entry.value
		elif ej.right == 0: edge = chart[e.right][0]; prob = edge.inside
		else: raise ValueError(
			"non-zero rank vector not part of explored derivations")
		result += prob
	return result

cdef inline str getderivation(RankedEdge ej, dict D, dict chart, dict tolabel,
		int n, str debin):
	""" Translate the (e, j) notation to an actual tree string in
	bracket notation.  e is an edge, j is a vector prescribing the rank of the
	corresponding tail node. For example, given the edge <S, [NP, VP], 1.0> and
	vector [2, 1], this points to the derivation headed by S and having the 2nd
	best NP and the 1st best VP as children.
	If `debin' is specified, will perform on-the-fly debinarization of nodes
	with labels containing `debin' an a substring. """
	cdef Edge edge
	cdef RankedEdge rankededge
	cdef ChartItem ei = ej.edge.left
	cdef str children = "", child
	cdef int i = ej.left
	if n > 100: return ""	#hardcoded limit to prevent cycles
	while i != -1:
		if ei not in chart:
			# this must be a terminal
			children = " %d" % ei.vec
			break
		if ei in D:
			rankededge = (<Entry>D[ei][i]).key
		else:
			assert i == 0, "non-best edge missing in derivations"
			edge = nsmallest(1, chart[ei]).pop()
			rankededge = RankedEdge(ei, edge, 0, 0 if edge.right.label else -1)
		child = getderivation(rankededge, D, chart, tolabel, n + 1, debin)
		if child == "": return ""
		children += " %s" % child
		if ei is ej.edge.right: break
		ei = ej.edge.right
		i = ej.right
	if debin is not None and debin in tolabel[ej.head.label]:
		return children
	return "(%s%s)" % (tolabel[ej.head.label], children)

cdef inline tuple getderivationtuple(RankedEdge ej, dict D, dict chart,
		dict tolabel, int n, str debin):
	""" Translate the (e, j) notation to a tuple with integer IDs for labels.
	e is an edge, j is a vector prescribing the rank of the
	corresponding tail node. For example, given the edge <S, [NP, VP], 1.0> and
	vector [2, 1], this points to the derivation headed by S and having the 2nd
	best NP and the 1st best VP as children.
	If `debin' is specified, will perform on-the-fly debinarization of nodes
	with labels containing `debin' an a substring. """
	cdef Edge edge
	cdef RankedEdge rankededge
	cdef ChartItem ei = ej.edge.left
	cdef tuple children = (), child
	cdef int i = ej.left
	if n > 100: return ()	#hardcoded limit to prevent cycles
	while i != -1:
		if ei not in chart:
			# this must be a terminal
			children = (ei.vec, )
			break
		if ei in D:
			rankededge = (<Entry>D[ei][i]).key
		else:
			assert i == 0, "non-best edge missing in derivations"
			edge = nsmallest(1, chart[ei]).pop()
			rankededge = RankedEdge(ei, edge, 0, 0 if edge.right.label else -1)
		child = getderivationtuple(rankededge, D, chart, tolabel, n + 1, debin)
		if child == (): return ()
		children += (child,)
		if ei is ej.edge.right: break
		ei = ej.edge.right
		i = ej.right
	if debin is not None and debin in tolabel[ej.head.label]:
		return children
	return (ej.head.label,) + children

cpdef list lazykbest(dict chart, ChartItem goal, int k, dict tolabel,
		str debin=None, bint tuplerepr=False):
	""" wrapper function to run lazykthbest and get the actual derivations.
	chart is a monotone hypergraph; should be acyclic unless probabilities
	resolve the cycles (maybe nonzero weights for unary productions are
	sufficient?).
	maps ChartItems to lists of tuples with ChartItems and a weight. The
	items in each list are to be ordered as they were added by the viterbi
	parse, with the best item first.
	goal is a ChartItem that is to be the root node of the derivations.
	k is the number of derivations desired.
	tolabel is a dictionary mapping numeric IDs to the original nonterminal
	labels.  """
	cdef Entry entry
	cdef dict D = {}, cand = {}
	cdef set explored = set()
	cdef double prod
	lazykthbest(goal, k, k, D, cand, chart, explored)
	if tuplerepr:
		return filter(itemgetter(0), [
				(getderivationtuple(entry.key, D, chart, tolabel, 0, debin),
					entry.value) for entry in D[goal]])
	return filter(itemgetter(0), [
			(getderivation(entry.key, D, chart, tolabel, 0, debin), entry.value)
			for entry in D[goal]])

cpdef main():
	from math import log
	cdef ChartItem v, ci
	cdef Edge ed
	cdef Entry entry
	toid = dict([a[::-1] for a in enumerate(
			"Epsilon S NP V ADV VP VP2 PN Mary walks quickly".split())])
	tolabel = dict([a[::-1] for a in toid.items()])
	NONE = ("Epsilon", 0)			# sentinel node
	chart = {
			("S", 0b111) : [
				((0.5*0.4), 0.4,
						("NP", 0b100), ("VP", 0b011)),
				((0.25*0.7), 0.7,
						("NP", 0b100), ("VP2", 0b011))],
			("VP", 0b011) : [
				(0.5, 0.5, ("V", 0b010), ("ADV", 0b001)),
				(0.4, 0.4, ("walks", 1), ("ADV", 0b001))],
			("VP2", 0b011) : [
				(0.5, 0.5, ("V", 0b010), ("ADV", 0b001)),
				(0.4, 0.4, ("walks", 1), ("ADV", 0b001))],
			("NP", 0b100) : [(0.5, 0.5, ("Mary", 0), NONE),
							(0.9, 0.9, ("PN", 0b100), NONE)],
			("PN", 0b100) : [(1.0, 1.0, ("Mary", 0), NONE),
							(0.9, 0.9, ("NP", 0b100), NONE)
							],
			("V", 0b010) : [(1.0, 1.0, ("walks", 1), NONE)],
			("ADV", 0b001) : [(1.0, 1.0, ("quickly", 2), NONE)]
		}
	for a in list(chart):
		chart[ChartItem(toid[a[0]], a[1])] = [Edge(-log(c), -log(c), -log(d),
				ChartItem(toid[e], f), ChartItem(toid[g], h))
				for c, d, (e,f), (g,h) in chart.pop(a)]
	assert ChartItem(toid["NP"], 0b100) == ChartItem(toid["NP"], 0b100)
	cand = {}
	D = {}
	k = 10
	goal = ChartItem(toid["S"], 0b111)
	for v, b in lazykthbest(goal, k, k, D, cand, chart, set()).items():
		print tolabel[v.label], bin(v.vec)[2:]
		for entry in b:
			ed = entry.key.edge
			j = (entry.key.left,)
			if entry.key.right != -1: j += (entry.key.right,)
			ip = entry.value
			print tolabel[v.label], ":",
			print " ".join([tolabel[ci.label]
				for ci, _ in zip((ed.left, ed.right), j)]),
			print exp(-ed.prob), j, exp(-ip)
		print
	from pprint import pprint
	print "tolabel",
	pprint(tolabel)
	print "candidates",
	for a in cand:
		print a, len(cand[a]),
		pprint(cand[a].items())

	print "\n%d derivations" % (len(D[goal]))
	derivations = lazykbest(chart, goal, k, tolabel)
	for a, p in derivations:
		print exp(-p), a
	assert len(D[goal]) == len(set(D[goal]))
	assert len(derivations) == len(set(derivations))
	assert len(set(derivations)) == len(dict(derivations))

if __name__ == '__main__': main()
