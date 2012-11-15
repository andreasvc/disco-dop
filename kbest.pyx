""" Implementation of Huang & Chiang (2005): Better k-best parsing. """
from math import exp
from agenda import Agenda
from containers import ChartItem, Edge, RankedEdge
from operator import itemgetter

from agenda cimport Entry, Agenda, nsmallest
from containers cimport ChartItem, Edge, RankedEdge, SmallChartItem, \
	FatChartItem, CFGChartItem, CFGEdge, LCFRSEdge, new_CFGChartItem, \
	RankedCFGEdge, UChar, UInt

cdef tuple unarybest = (0, ), binarybest = (0, 0)

cdef inline getcandidates(dict chart, ChartItem v, int k):
	""" Return a heap with up to k candidate arcs starting from vertex v """
	# NB: the priority queue should either do a stable sort, or should
	# sort on rank vector as well to have ties resolved in FIFO order;
	# otherwise the sequence (0, 0) -> (1, 0) -> (1, 1) -> (0, 1) -> (1, 1)
	# can occur (given that the first two have probability x and the latter
	# three probability y), in which case insertion order should count.
	# Otherwise (1, 1) ends up in D[v] after which (0. 1) generates it
	# as a neighbor and puts it in cand[v] for a second time.
	cdef LCFRSEdge el
	if v not in chart: return Agenda() #raise error?
	cell = chart[v]
	return Agenda(
		[(RankedEdge(v, el, 0, 0 if el.right.label else -1), el.inside)
						for el in nsmallest(k, cell)])

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
	cdef CFGEdge ec
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
	cdef Edge edge
	cdef CFGEdge ec
	cdef Entry entry
	cdef double result, prob
	ei = ej.edge.left
	if ei in D: entry = D[ei][ej.left]; prob = entry.value
	elif ej.left == 0: edge = chart[ei][0]; prob = edge.inside
	else: raise ValueError(
		"non-zero rank vector not part of explored derivations")
	result = ej.edge.prob + prob
	if ej.right >= 0: #if e.right.label:
		ei = ej.edge.right
		if ei in D: entry = D[ei][ej.right]; prob = entry.value
		elif ej.right == 0: edge = chart[ei][0]; prob = edge.inside
		else: raise ValueError(
			"non-zero rank vector not part of explored derivations")
		result += prob
	return result

# --- start CFG specific
cdef inline getcandidatescfg(list chart, UInt label,
		UChar start, UChar end, int k):
	""" Return a heap with up to k candidate arcs starting from vertex v """
	# NB: the priority queue should either do a stable sort, or should
	# sort on rank vector as well to have ties resolved in FIFO order;
	# otherwise the sequence (0, 0) -> (1, 0) -> (1, 1) -> (0, 1) -> (1, 1)
	# can occur (given that the first two have probability x and the latter
	# three probability y), in which case insertion order should count.
	# Otherwise (1, 1) ends up in D[v] after which (0. 1) generates it
	# as a neighbor and puts it in cand[v] for a second time.
	cdef CFGEdge ec
	if label not in chart[start][end]: return Agenda()
	cell = chart[start][end][label]
	return Agenda(
		[(RankedCFGEdge(label, start, end, ec, 0, 0 if ec.rule is not NULL
						and ec.rule.rhs2 else -1), ec.inside)
						for ec in nsmallest(k, cell)])

cpdef inline lazykthbestcfg(UInt label, UChar start, UChar end, int k, int k1,
		list D, list cand, list chart, set explored):
	cdef Entry entry
	cdef RankedCFGEdge ej
	# k1 is the global k
	# first visit of vertex v?
	if label not in cand[start][end]:
		# initialize the heap
		cand[start][end][label] = getcandidatescfg(chart, label, start, end, k1)
	while label not in D[start][end] or len(D[start][end][label]) < k:
		if label in D[start][end]:
			# last derivation
			entry = D[start][end][label][-1]
			ej = entry.key
			# update the heap, adding the successors of last derivation
			lazynextcfg(ej, k1, D, cand, chart, explored)
		# get the next best derivation and delete it from the heap
		if cand[start][end][label]:
			D[start][end].setdefault(label, []).append(
				cand[start][end][label].popentry())
		else: break
	return D

cdef inline lazynextcfg(RankedCFGEdge ej, int k1, list D, list cand, list chart,
		set explored):
	cdef RankedCFGEdge ej1
	cdef CFGEdge ec = ej.edge
	cdef double prob
	cdef UInt label
	cdef UChar start, end
	# add the |e| neighbors
	# left child
	label = 0 if ec.rule is NULL else ec.rule.rhs1; start = ej.start; end = ec.mid
	ej1 = RankedCFGEdge(ej.label, ej.start, ej.end, ej.edge, ej.left + 1, ej.right)
	# recursively solve a subproblem
	# NB: increment j1[i] again because j is zero-based and k is not
	lazykthbestcfg(label, start, end, ej1.left + 1, k1,
						D, cand, chart, explored)
	# if it exists and is not in heap yet
	if ((label in D[start][end] and ej1.left < len(D[start][end][label]))
		and ej1 not in explored): #cand[ej1.head]): <= gives duplicates
		prob = getprobcfg(chart, D, ej1)
		# add it to the heap
		cand[ej1.start][ej1.end][ej1.label][ej1] = prob
		explored.add(ej1)
	# right child?
	if ej.right == -1: return
	label = 0 if ec.rule is NULL else ec.rule.rhs2; start = ec.mid; end = ej.end
	ej1 = RankedCFGEdge(ej.label, ej.start, ej.end, ej.edge, ej.left, ej.right + 1)
	lazykthbestcfg(label, start, end, ej1.right + 1, k1,
						D, cand, chart, explored)
	# if it exists and is not in heap yet
	if ((label in D[start][end] and ej1.right < len(D[start][end][label]))
		and ej1 not in explored): #cand[ej1.head]): <= gives duplicates
		prob = getprobcfg(chart, D, ej1)
		# add it to the heap
		cand[ej1.start][ej1.end][ej1.label][ej1] = prob
		explored.add(ej1)

cdef inline double getprobcfg(list chart, list D, RankedCFGEdge ej) except -1.0:
	cdef CFGEdge ec, edge
	cdef Entry entry
	cdef double result, prob
	ec = ej.edge
	label = 0 if ec.rule is NULL else ec.rule.rhs1; start = ej.start; end = ec.mid
	if label in D[start][end]:
		entry = D[start][end][label][ej.left]; prob = entry.value
	elif ej.left == 0: edge = chart[start][end][label][0]; prob = edge.inside
	else: raise ValueError(
		"non-zero rank vector not part of explored derivations")
	# NB: edge.inside if preterminal, 0.0 for terminal
	result = (0.0 if ec.rule is NULL else ec.rule.prob) + prob
	if ej.right >= 0: #if e.right.label:
		label = 0 if ec.rule is NULL else ec.rule.rhs2
		start = ec.mid; end = ej.end
		if label in D[start][end]:
			entry = D[start][end][label][ej.right]
			prob = entry.value
		elif ej.right == 0: edge = chart[start][end][label][0]; prob = edge.inside
		else: raise ValueError(
			"non-zero rank vector not part of explored derivations")
		result += prob
	return result

cpdef list lazykbestcfg(list chart, CFGChartItem goal, int k):
	""" wrapper function to run lazykthbestcfg.
	does not give actual derivations, but the ranked chart D. """
	cdef Entry entry
	cdef list D = [[{} for _ in x] for x in chart]
	cdef list cand = [[{} for _ in x] for x in chart]
	cdef set explored = set()
	lazykthbestcfg(goal.label, goal.start, goal.end, k, k, D, cand, chart, explored)
	return D

cdef inline str getderivationcfg(RankedCFGEdge ej, list  D, list chart,
		dict tolabel, int n, str debin):
	""" Translate the (e, j) notation to an actual tree string in
	bracket notation.  e is an edge, j is a vector prescribing the rank of the
	corresponding tail node. For example, given the edge <S, [NP, VP], 1.0> and
	vector [2, 1], this points to the derivation headed by S and having the 2nd
	best NP and the 1st best VP as children.
	If `debin' is specified, will perform on-the-fly debinarization of nodes
	with labels containing `debin' an a substring. """
	cdef Entry entry
	cdef CFGEdge edge
	cdef RankedCFGEdge rankededge
	cdef str children = "", child
	cdef int i = ej.left
	cdef UInt label
	cdef UChar start, end
	if n > 100: return ""	#hardcoded limit to prevent cycles
	label = 0 if ej.edge.rule is NULL else ej.edge.rule.rhs1
	start = ej.start; end = ej.edge.mid
	while i != -1:
		if label not in chart[start][end]:
			# this must be a terminal
			children = " %d" % start
			break
		if label in D[start][end]:
			rankededge = (<Entry>D[start][end][label][i]).key
		else:
			assert i == 0, "non-best edge missing in derivations"
			entry = getcandidatescfg(chart, label, start, end, 1).popentry()
			D[start][end][label] = [entry]
			rankededge = entry.key
		child = getderivationcfg(rankededge, D, chart, tolabel, n + 1, debin)
		if child == "": return ""
		children += " %s" % child
		if end == ej.end: break
		label = 0 if ej.edge.rule is NULL else ej.edge.rule.rhs2
		start = ej.edge.mid; end = ej.end
		i = ej.right
	if debin is not None and debin in tolabel[ej.label]:
		return children
	return "(%s%s)" % (tolabel[ej.label], children)
# --- end CFG specific

def getderiv(RankedEdge ej, dict D, dict chart, dict tolabel, int n, str debin):
	return getderivation(ej, D, chart, tolabel, n, debin)

cdef inline str getderivation(RankedEdge ej, dict D, dict chart, dict tolabel,
		int n, str debin):
	""" Translate the (e, j) notation to an actual tree string in
	bracket notation.  e is an edge, j is a vector prescribing the rank of the
	corresponding tail node. For example, given the edge <S, [NP, VP], 1.0> and
	vector [2, 1], this points to the derivation headed by S and having the 2nd
	best NP and the 1st best VP as children.
	If `debin' is specified, will perform on-the-fly debinarization of nodes
	with labels containing `debin' an a substring. """
	cdef Entry entry
	cdef LCFRSEdge edge
	cdef RankedEdge rankededge
	cdef ChartItem ei
	cdef str children = "", child
	cdef int i = ej.left
	if n > 100: return ""	#hardcoded limit to prevent cycles
	ei = ej.edge.left
	while i != -1:
		if ei not in chart:
			# this must be a terminal
			children = " %d" % ei.lexidx()
			break
		elif ei in D:
			rankededge = (<Entry>D[ei][i]).key
		else:
			assert i == 0, "non-best edge missing in derivations"
			entry = getcandidates(chart, ei, 1).popentry()
			D[ei] = [entry]
			rankededge = entry.key
		child = getderivation(rankededge, D, chart, tolabel, n + 1, debin)
		if child == "": return ""
		children += " %s" % child
		if ei is ej.edge.right: break
		ei = ej.edge.right
		i = ej.right
	if debin is not None and debin in tolabel[ej.head.label]:
		return children
	return "(%s%s)" % (tolabel[ej.head.label], children)

cpdef tuple lazykbest(chart, ChartItem goal, int k, dict tolabel,
		str debin=None):
	""" wrapper function to run lazykthbest and get the actual derivations,
	as well as the ranked chart.
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
	cdef set explored = set()
	cdef list derivations = [], tmp = []
	if isinstance(goal, CFGChartItem):
		D = [[{} for _ in x] for x in chart]
		cand = [[{} for _ in x] for x in chart]
		start = (<CFGChartItem>goal).start
		end = (<CFGChartItem>goal).end
		lazykthbestcfg(goal.label, start, end, k, k, D, cand, chart, explored)
		for entry in D[start][end][goal.label]:
			d = getderivationcfg(entry.key, D, chart, tolabel, 0, debin)
			if d:
				derivations.append((d, entry.value))
				tmp.append(entry)
		D[start][end][goal.label] = tmp
	else:
		D = {}; cand = {}
		lazykthbest(goal, k, k, D, cand, chart, explored)
		for entry in D[goal]:
			d = getderivation(entry.key, D, chart, tolabel, 0, debin)
			if d:
				derivations.append((d, entry.value))
				tmp.append(entry)
		D[goal] = tmp
	return derivations, D

cpdef main():
	from math import log
	cdef SmallChartItem v, ci
	cdef LCFRSEdge ed
	cdef RankedEdge re
	cdef Entry entry
	toid = dict([a[::-1] for a in enumerate(
			"Epsilon S NP V ADV VP VP2 PN".split())])
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
		chart[SmallChartItem(toid[a[0]], a[1])] = [LCFRSEdge(-log(c), -log(c),
			-log(d), 0, SmallChartItem(toid.get(e, 0), f),
			SmallChartItem(toid.get(g, 0), h))
			for c, d, (e,f), (g,h) in chart.pop(a)]
	assert SmallChartItem(toid["NP"], 0b100) == SmallChartItem(toid["NP"], 0b100)
	cand = {}
	D = {}
	k = 10
	goal = SmallChartItem(toid["S"], 0b111)
	for v, b in lazykthbest(goal, k, k, D, cand, chart, set()).items():
		print tolabel[v.label], bin(v.vec)[2:]
		for entry in b:
			re = entry.key
			ed = re.edge
			j = (re.left,)
			if re.right != -1: j += (re.right,)
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
	derivations = lazykbest(chart, goal, k, tolabel)[0]
	for a, p in derivations:
		print exp(-p), a
	assert len(D[goal]) == len(set(D[goal]))
	assert len(derivations) == len(set(derivations))
	assert len(set(derivations)) == len(dict(derivations))

if __name__ == '__main__': main()
