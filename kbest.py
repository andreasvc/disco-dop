""" Implementation of Huang & Chiang (2005): Better k-best parsing
"""
from math import exp, fsum
from agenda import Agenda, Entry
from containers import ChartItem, Edge, RankedEdge
from operator import itemgetter
try: assert nsmallest(1, [1]) == [1]
except NameError: from heapq import *

unarybest = (0, )
binarybest = (0, 0)

def getcandidates(chart, v, k):
	""" Return a heap with up to k candidate arcs starting from vertex v """
	# NB: the priority queue should either do a stable sort, or should
	# sort on rank vector as well to have ties resolved in FIFO order;
	# otherwise the sequence (0, 0) -> (1, 0) -> (1, 1) -> (0, 1) -> (1, 1)
	# can occur (given that the first two have probability x and the latter
	# three probability y), in which case insertion order should count.
	# Otherwise (1, 1) ends up in D[v] after which (0. 1) generates it
	# as a neighbor and puts it in cand[v] for a second time.
	if v not in chart: return Agenda() #raise error?
	return Agenda(
		[(RankedEdge(v, edge, 0, 0 if edge.right.label else -1), edge.inside)
						for edge in nsmallest(k, chart[v])])

def lazykthbest(v, k, k1, D, cand, chart, explored):
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

def lazynext(ej, k1, D, cand, chart, explored):
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

def getprob(chart, D, ej):
	e = ej.edge
	if e.left in D: entry = D[e.left][ej.left]; prob = entry.value
	elif ej.left == 0: edge = chart[e.left][0]; prob = edge.inside
	else: raise ValueError("non-zero rank vector not part of explored derivations")
	result = e.prob + prob
	if ej.right >= 0: #if e.right.label:
		if e.right in D: entry = D[e.right][ej.right]; prob = entry.value
		elif ej.right == 0: edge = chart[e.right][0]; prob = edge.inside
		else: raise ValueError("non-zero rank vector not part of explored derivations")
		result += prob
	return result

def getderivation(ej, D, chart, tolabel, n):
	""" Translate the (e, j) notation to an actual tree string in
	bracket notation.  e is an edge, j is a vector prescribing the rank of the
	corresponding tail node. For example, given the edge <S, [NP, VP], 1.0> and
	vector [2, 1], this points to the derivation headed by S and having the 2nd
	best NP and the 1st best VP as children.
	"""
	if n > 100: return ""	#hardcoded limit to prevent cycles
	e = ej.edge
	children = []
	for ei, i in ((e.left, ej.left), (e.right, ej.right)):
		if i == -1: break
		if ei in chart:
			if ei in D:
				entry = D[ei][i]
				children.append(
					getderivation(entry.key, D, chart, tolabel, n + 1))
			else:
				if i == 0:
					edge = nsmallest(1, chart[ei]).pop()
					children.append(getderivation(
						RankedEdge(ei, edge, 0, 0 if edge.right.label else -1),
						D, chart, tolabel, n + 1))
				else: raise ValueError("non-best edge missing in derivations")
		else:
			# this must be a terminal
			children.append(str(ei.vec))

	if "" in children: return ""
	return "(%s %s)" % (tolabel[ej.head.label], " ".join(children))

def lazykbest(chart, goal, k, tolabel):
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
	import logging
	D = {}
	cand = {}
	explored = set()
	lazykthbest(goal, k, k, D, cand, chart, explored)
	return filter(itemgetter(0), [
			(getderivation(entry.key, D, chart, tolabel, 0), entry.value)
			for entry in D[goal]])

toid = {}
from math import log
def ci(label, vec): return ChartItem(toid[label], vec)
def l(a): return -log(a)
def ed(i, p, l, r): return Edge(i, i, p, l, r)

def main():
	from containers import Edge
	toid.update([a[::-1] for a in enumerate(
			"Epsilon S NP V ADV VP VP2 PN Mary walks quickly".split())])
	tolabel = dict([a[::-1] for a in toid.items()])
	NONE = ci("Epsilon", 0)			# sentinel node
	goal = ci("S", 0b111)
	chart = {
			ci("S", 0b111) : [
				ed(l(0.5*0.4), l(0.4),
						ci("NP", 0b100), ci("VP", 0b011)),
				ed(l(0.25*0.7), l(0.7),
						ci("NP", 0b100), ci("VP2", 0b011))],
			ci("VP", 0b011) : [
				ed(l(0.5), l(0.5), ci("V", 0b010), ci("ADV", 0b001)),
				ed(l(0.4), l(0.4), ci("walks", 1), ci("ADV", 0b001))],
			ci("VP2", 0b011) : [
				ed(l(0.5), l(0.5), ci("V", 0b010), ci("ADV", 0b001)),
				ed(l(0.4), l(0.4), ci("walks", 1), ci("ADV", 0b001))],
			ci("NP", 0b100) : [ed(l(0.5), l(0.5), ci("Mary", 0), NONE),
							ed(l(0.9), l(0.9), ci("PN", 0b100), NONE)],
			ci("PN", 0b100) : [ed(l(1.0), l(1.0), ci("Mary", 0), NONE),
							ed(l(0.9), l(0.9), ci("NP", 0b100), NONE)
							],
			ci("V", 0b010) : [ed(l(1.0), l(1.0), ci("walks", 1), NONE)],
			ci("ADV", 0b001) : [ed(l(1.0), l(1.0), ci("quickly", 2), NONE)]
		}
	assert ci("NP", 0b100) == ci("NP", 0b100)
	cand = {}
	D = {}
	k = 10
	for v, b in lazykthbest(goal, k, k, D, cand, chart, set()).items():
		print tolabel[v.label], bin(v.vec)[2:]
		for entry in b:
			e = entry.key.edge
			j = (entry.key.left,)
			if entry.key.right != -1: j += (entry.key.right,)
			ip = entry.value
			print tolabel[v.label], ":",
			print " ".join([tolabel[c.label] for c, _ in zip((e.left, e.right), j)]),
			print exp(-e.prob), j, exp(-ip)
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
