""" Implementation of Huang & Chiang (2005): Better k-best parsing. """
from __future__ import print_function
from math import exp
from agenda import Agenda
from containers import ChartItem, Edge, RankedEdge
from operator import itemgetter

from agenda cimport Entry, Agenda, nsmallest
from containers cimport ChartItem, SmallChartItem, FatChartItem, CFGChartItem, \
	new_CFGChartItem, Edge, LCFRSEdge, new_LCFRSEdge, CFGEdge, \
	RankedEdge, RankedCFGEdge, UChar, UInt, Rule

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
	if v not in chart:
		return Agenda() #raise error?
	return Agenda(
		[(RankedEdge(v, el, 0, 0 if el.right.label else -1), el.inside)
						for el in nsmallest(k, chart[v].values())])

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
			D.setdefault(v, []).append((<Agenda>cand[v]).popentry())
		else:
			break
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
		else:
			break
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
	cdef Entry entry
	cdef double result, prob
	ei = ej.edge.left
	if ei in D:
		entry = D[ei][ej.left]
		prob = entry.value
	elif ej.left == 0:
		edge = min(chart[ei])
		prob = edge.inside
	else:
		raise ValueError("non-zero rank vector not part of explored derivations")
	result = ej.edge.rule.prob + prob
	if ej.right >= 0: #if e.right.label:
		ei = ej.edge.right
		if ei in D:
			entry = D[ei][ej.right]
			prob = entry.value
		elif ej.right == 0:
			edge = min(chart[ei])
			prob = edge.inside
		else:
			raise ValueError(
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
	cell = chart[start][end]
	if not cell.get(label):
		return Agenda()
	return Agenda(
		[(RankedCFGEdge(label, start, end, ec, 0, 0 if ec.rule is not NULL
						and ec.rule.rhs2 else -1), ec.inside)
						for ec in nsmallest(k, cell[label].values())])

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
				(<Agenda>cand[start][end][label]).popentry())
		else:
			break
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
	label = 0 if ec.rule is NULL else ec.rule.rhs1
	start = ej.start
	end = ec.mid
	ej1 = RankedCFGEdge(ej.label, ej.start, ej.end, ej.edge,
			ej.left + 1, ej.right)
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
	if ej.right == -1:
		return
	label = 0 if ec.rule is NULL else ec.rule.rhs2
	start = ec.mid
	end = ej.end
	ej1 = RankedCFGEdge(ej.label, ej.start, ej.end, ej.edge,
			ej.left, ej.right + 1)
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
	label = 0 if ec.rule is NULL else ec.rule.rhs1
	start = ej.start
	end = ec.mid
	if label in D[start][end]:
		entry = D[start][end][label][ej.left]
		prob = entry.value
	elif ej.left == 0:
		edge = min(chart[start][end][label])
		prob = edge.inside
	else:
		raise ValueError("non-zero rank vector not part of explored derivations")
	# NB: edge.inside if preterminal, 0.0 for terminal
	result = (0.0 if ec.rule is NULL else ec.rule.prob) + prob
	if ej.right >= 0: #if e.right.label:
		label = 0 if ec.rule is NULL else ec.rule.rhs2
		start = ec.mid
		end = ej.end
		if label in D[start][end]:
			entry = D[start][end][label][ej.right]
			prob = entry.value
		elif ej.right == 0:
			edge = min(chart[start][end][label])
			prob = edge.inside
		else:
			raise ValueError(
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
	lazykthbestcfg(goal.label, goal.start, goal.end, k, k, D, cand,
			chart, explored)
	return D

cdef inline bint explorederivationcfg(RankedCFGEdge ej,
		list D, list chart, set explored, int n):
	""" Traverse a derivation to ensured RankedEdges are present in D
	for every edge. """
	cdef Entry entry
	cdef RankedCFGEdge rankededge
	cdef UInt label
	if n > 100:
		return False #hardcoded limit to prevent cycles
	elif ej.edge.rule is NULL:
		return True
	label = ej.edge.rule.rhs1
	if ej.left != -1 and label in chart[ej.start][ej.edge.mid]:
		if label not in D[ej.start][ej.edge.mid]:
			assert ej.left == 0, "non-best edge missing in derivations"
			entry = (<Agenda>getcandidatescfg(chart, label, ej.start,
					ej.edge.mid, 1)).popentry()
			D[ej.start][ej.edge.mid][label] = [entry]
			explored.add(entry.key)
		if not explorederivationcfg(<RankedCFGEdge>
				(<Entry>D[ej.start][ej.edge.mid][label][ej.left]).key,
				D, chart, explored, n + 1):
			return False
	label = ej.edge.rule.rhs2
	if ej.right != -1 and label in chart[ej.edge.mid][ej.end]:
		if label not in D[ej.edge.mid][ej.end]:
			assert ej.right == 0, "non-best edge missing in derivations"
			entry = (<Agenda>getcandidatescfg(chart, label, ej.edge.mid,
					ej.end, 1)).popentry()
			explored.add(entry.key)
			D[ej.edge.mid][ej.end][label] = [entry]
		return explorederivationcfg(<RankedCFGEdge>
				(<Entry>D[ej.edge.mid][ej.end][label][ej.right]).key,
				D, chart, explored, n + 1)
	return True

cdef inline getderivationcfg(result, RankedCFGEdge ej, list  D,
		list chart, dict tolabel, str debin):
	cdef Entry entry
	cdef RankedCFGEdge rankededge
	cdef UInt label
	if debin is None or debin not in tolabel[ej.label]:
		result += '('
		result += tolabel[ej.label]
		result += ' '
	if ej.edge.rule is NULL: # this must be a terminal
		result += str(ej.start)
	elif ej.left != -1:
		label = ej.edge.rule.rhs1
		rankededge = (<Entry>D[ej.start][ej.edge.mid][label][ej.left]).key
		getderivationcfg(result, rankededge, D, chart, tolabel, debin)
		result += ' '
		if ej.right != -1:
			label = ej.edge.rule.rhs2
			rankededge = (<Entry>D[ej.edge.mid][ej.end][label][ej.right]).key
			getderivationcfg(result, rankededge, D, chart, tolabel, debin)
	if debin is None or debin not in tolabel[ej.habel]:
		result += ')'
# --- end CFG specific

cdef bint explorederivation(RankedEdge ej, dict D, dict chart, set explored,
		int n):
	""" Traverse a derivation to ensure RankedEdges are present
	in D for every edge. """
	cdef Entry entry
	cdef RankedEdge rankededge
	if n > 100:
		return False #hardcoded limit to prevent cycles
	if ej.left != -1 and ej.edge.left in chart:
		if ej.edge.left not in D:
			assert ej.left == 0, "non-best edge missing in derivations"
			entry = (<Agenda>getcandidates(chart, ej.edge.left, 1)).popentry()
			D[ej.edge.left] = [entry]
			explored.add(entry.key)
		if not explorederivation(
				<RankedEdge>(<Entry>D[ej.edge.left][ej.left]).key,
				D, chart, explored, n + 1):
			return False
	if ej.right != -1 and ej.edge.right in chart:
		if ej.edge.right not in D:
			assert ej.right == 0, "non-best edge missing in derivations"
			entry = (<Agenda>getcandidates(chart, ej.edge.right, 1)).popentry()
			D[ej.edge.right] = [entry]
			explored.add(entry.key)
		return explorederivation(
				<RankedEdge>(<Entry>D[ej.edge.right][ej.right]).key,
				D, chart, explored, n + 1)
	return True

cdef inline getderivation(result, RankedEdge ej, dict D,
		dict chart, dict tolabel, str debin):
	cdef Entry entry
	cdef RankedEdge rankededge
	cdef ChartItem item
	if debin is None or debin not in tolabel[ej.head.label]:
		result += '('
		result += tolabel[ej.head.label]
		result += ' '
	item = ej.edge.left
	if item not in chart:
		# this must be a terminal
		result += str(item.lexidx())
	else:
		rankededge = (<Entry>D[item][ej.left]).key
		getderivation(result, rankededge, D, chart, tolabel, debin)
	if ej.right != -1:
		item = ej.edge.right
		result += ' '
		if item in chart:
			rankededge = (<Entry>D[item][ej.right]).key
			getderivation(result, rankededge, D, chart, tolabel, debin)
		else:
			result += str(item.lexidx())
	if debin is None or debin not in tolabel[ej.head.label]:
		result += ')'

def getderiv(ej, D, chart, dict tolabel, str debin):
	""" Translate the (e, j) notation to an actual tree string in
	bracket notation.  e is an edge, j is a vector prescribing the rank of the
	corresponding tail node. For example, given the edge <S, [NP, VP], 1.0> and
	vector [2, 1], this points to the derivation headed by S and having the 2nd
	best NP and the 1st best VP as children.
	If `debin' is specified, will perform on-the-fly debinarization of nodes
	with labels containing `debin' an a substring. """
	result = bytearray()
	if isinstance(ej, RankedEdge):
		getderivation(result, ej, D, chart, tolabel, debin)
	elif isinstance(ej, RankedCFGEdge):
		getderivationcfg(result, ej, D, chart, tolabel, debin)
	return str(result)

cpdef tuple lazykbest(chart, ChartItem goal, int k, dict tolabel=None,
		str debin=None, bint derivs=True):
	""" wrapper function to run lazykthbest and get the actual derivations,
	(except when derivs is False) as well as the ranked chart.
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
	derivations = []
	if isinstance(goal, CFGChartItem):
		D = [[{} for _ in x] for x in chart]
		cand = [[{} for _ in x] for x in chart]
		start = (<CFGChartItem>goal).start
		end = (<CFGChartItem>goal).end
		lazykthbestcfg(goal.label, start, end, k, k, D, cand, chart, explored)
		D[start][end][goal.label] = [entry
				for entry in D[start][end][goal.label][:k]
				if explorederivationcfg(entry.key, D, chart, explored, 0)]
		if derivs:
			derivations = [(getderiv(
					entry.key, D, chart, tolabel, debin), entry.value)
					for entry in D[start][end][goal.label]]
	else:
		D = {}
		cand = {}
		lazykthbest(goal, k, k, D, cand, chart, explored)
		D[goal] = [entry for entry in D[goal][:k]
				if explorederivation(entry.key, D, chart, explored, 0)]
		if derivs:
			derivations = [(getderiv(
					entry.key, D, chart, tolabel, debin), entry.value)
					for entry in D[goal]]
	return derivations, D, explored

cpdef main():
	from math import log
	cdef SmallChartItem v, ci
	cdef LCFRSEdge ed
	cdef RankedEdge re
	cdef Entry entry
	cdef Rule rules[11]
	toid = {a: n for n, a in enumerate(
			"Epsilon S NP V ADV VP VP2 PN".split())}
	tolabel = {b: a for a, b in toid.items()}
	NONE = ("Epsilon", 0)			# sentinel node
	chart = {
			("S", 0b111): [
				((0.7 * 0.9 * 0.5), 0.7,
						("NP", 0b100), ("VP2", 0b011)),
				((0.4 * 0.9 * 0.5), 0.4,
						("NP", 0b100), ("VP", 0b011))],
			("VP", 0b011): [
				(0.5, 0.5, ("V", 0b010), ("ADV", 0b001)),
				(0.4, 0.4, ("walks", 1), ("ADV", 0b001))],
			("VP2", 0b011): [
				(0.5, 0.5, ("V", 0b010), ("ADV", 0b001)),
				(0.4, 0.4, ("walks", 1), ("ADV", 0b001))],
			("NP", 0b100): [(0.5, 0.5, ("Mary", 0), NONE),
							(0.9, 0.9, ("PN", 0b100), NONE)],
			("PN", 0b100): [(1.0, 1.0, ("Mary", 0), NONE),
							(0.9, 0.9, ("NP", 0b100), NONE)],
			("V", 0b010): [(1.0, 1.0, ("walks", 1), NONE)],
			("ADV", 0b001): [(1.0, 1.0, ("quickly", 2), NONE)]
		}
	# a hack to make Rule structs with the right probabilities.
	# rules[7] will be a Rule with probability 0.7
	for a in range(1, 11):
		rules[a].prob = -log(a / 10.0)
	for a in list(chart):
		chart[SmallChartItem(toid[a[0]], a[1])] = {x: x
			for x in [new_LCFRSEdge(-log(c), -log(c), &(rules[int(d * 10)]),
			SmallChartItem(toid.get(e, 0), f),
			SmallChartItem(toid.get(g, 0), h))
			for c, d, (e, f), (g, h) in chart.pop(a)]}
	assert SmallChartItem(toid["NP"], 0b100) == SmallChartItem(
			toid["NP"], 0b100)
	cand = {}
	D = {}
	k = 10
	goal = SmallChartItem(toid["S"], 0b111)
	for v, b in lazykthbest(goal, k, k, D, cand, chart, set()).items():
		print(tolabel[v.label], bin(v.vec)[2:])
		for entry in b:
			re = entry.key
			ed = re.edge
			j = (re.left, )
			if re.right != -1:
				j += (re.right, )
			ip = entry.value
			print(tolabel[v.label], ":",
				" ".join([tolabel[ci.label] for ci, _
				in zip((ed.left, ed.right), j)]),
				exp(-ed.rule.prob), j, exp(-ip))
		print()
	from pprint import pprint
	print("tolabel", end='')
	pprint(tolabel)
	print("candidates", end='')
	for a in cand:
		print(a, len(cand[a]), end='')
		pprint(list(cand[a].items()))

	print("\n%d derivations" % (len(D[goal])))
	derivations = lazykbest(chart, goal, k, tolabel)[0]
	for a, p in derivations:
		print(exp(-p), a)
	assert len(D[goal]) == len(set(D[goal]))
	assert len(derivations) == len(set(derivations))
	assert len(set(derivations)) == len(dict(derivations))

if __name__ == '__main__':
	main()
