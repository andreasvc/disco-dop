# Implementation of Huang & Chiang (2005): Better k-best parsing
from heapq import nsmallest #heappop, heappush, heapify
from heapdict import heapdict
from collections import defaultdict
from nltk import memoize, Tree
from math import log, exp
try: frozenset
except NameError: from sets import ImmutableSet as frozenset

def getcandidates(chart, v, k):
	""" Return a heap with up to k candidate arcs starting from vertex v """
	temp = [(Edge(v, a, p), tuple(1 for x in a)) for  a,p in chart[v]]
	return heapdict(nsmallest(k, (((e,j), getprob(chart, e, j)) for e,j in temp), key=lambda x: x[1]))

def lazykthbest(chart, v, k, k1, D, cand):
	# k1 is the global k
	# kth derivation already computed?
	if len(D[v]) >= k: return
	# first visit of vertex v?
	if v not in cand:
		# initialize the heap
		cand[v] = getcandidates(chart, v, k1)
		# 1-best
		if cand[v]: D[v].append(cand[v].popitem())
		#if cand[v]: D[v].append(heappop(cand[v]))
	while len(D[v]) < k and cand[v]:
		# last derivation
		e, j = D[v][-1][0]
		# update the heap, adding the successors of last derivation
		lazynext(cand, v, e, j, k1, D, chart)
		# get the next best derivation and delete it from the heap
		D[v].append(cand[v].popitem())
		#D[v].append(heappop(cand[v]))
	return D

def lazynext(cand, v, e, j, k1, D, chart):
	# add the |e| neighbors
	for i,ei in enumerate(e.tailnodes):
		j1 = tuple(x+1 if xi == i else x for xi, x in enumerate(j))
		# recursively solve a subproblem
		lazykthbest(chart, ei, j1[i], k1, D, cand)
		# if it exists and is not in heap yet
		if j1[i] <= len(D[ei]) and (e, j1) not in cand[v]:
			# add it to the heap
			cand[v][e,j1] = getprob(chart, e, j1)
			#heappush(cand[v], (getprob(chart, e, j1), e, j1))

@memoize
def getprob(chart, e, j):
	if e.tailnodes == []: return 0.0
	return e.weight + sum(getprob(chart, 
								Edge(tail, chart[tail][-ji][0], chart[tail][-ji][1]), 
								tuple(1 for x in chart[tail][-ji][0])) #!?! assume 1-best
				for tail, ji in zip(e.tailnodes, j) if tail in chart)

def lazykbest(chart, goal, k):
	""" wrapper function to run lazykthbest and get the actual derivations """
	D = defaultdict(list)
	cand = {}
	lazykthbest(chart, goal, k, k, D, cand)
	return [(getderivation(D, ej), p) for ej, p in D[goal]]

def getderivation(D, (e, j)):
	""" Translate the (e, j) notation to an actual nltk Tree.
	e is an edge, j is a vector prescribing the rank of the corresponding tail
	node. For example, given the edge <S, [NP, VP], 1.0> and vector [2, 1], this
	points to the derivation headed by S and having the 2nd best NP and the 1st
	best VP as children. """
	return Tree(e.head.label, [getderivation(D, D[ei][i-1][0]) if ei in D else ei.label
				for ei,i in zip(e.tailnodes, j)])
	
# http://bob.pythonmac.org/archives/2005/03/04/frozendict/
# this does not enforce immutability
class frozendefaultdict(defaultdict):
	__slots__ = ('_hash',)
	def __hash__(self):
		rval = getattr(self, '_hash', None)
		if rval is None:
			rval = self._hash = hash(frozenset(self.iteritems()))
		return rval

class Edge:
	""" An edge is defined as an arc between a head node and zero or more tail
	nodes, with a given weight. The case of zero tail nodes corresponds to a terminal. """
	__slots__ = ("head", "tailnodes", "weight", "_hash")
	def __init__(self, head, tailnodes, weight):
		self.head = head; self.tailnodes = tailnodes; self.weight = weight
		self._hash = hash((head, tailnodes, weight))
	def __hash__(self):
		return self._hash
	def __repr__(self):
		return "<%s, [%s], %f>" % (self.head, ", ".join(repr(a) for a in self.tailnodes), exp(-self.weight))

if __name__ == '__main__':
	from plcfrs import ChartItem
	# a monotone hypergraph; should be acyclic unless probabilities resolve the
	# cycles (maybe nonzero weights for unary productions are sufficient?)
	# chart should be ordered with the 1-best items last
	chart1 = {
			ChartItem("S", 0b111) : [
				((ChartItem("NP", 0b100), ChartItem("V", 0b010), ChartItem("ADV", 0b001)), -log(0.4)),
				((ChartItem("NP", 0b100), ChartItem("VP", 0b011)), -log(0.7))],
			ChartItem("VP", 0b011) : [((ChartItem("V", 0b010), ChartItem("ADV", 0b001)), -log(0.5))],
			ChartItem("NP", 0b100) : [((ChartItem("Mary", 0),), -log(0.5)), ((ChartItem("PN", 0b100),), -log(0.8))],
			ChartItem("PN", 0b100) : [((ChartItem("Mary", 0),), -log(1.0))],
			ChartItem("V", 0b010) : [((ChartItem("walks", 1),), -log(0.5))],
			ChartItem("ADV", 0b001) : [((ChartItem("quickly", 2),), -log(0.5))]
			}
	chart = frozendefaultdict(tuple); chart.update((a, tuple(b)) for a,b in chart1.items())
	goal = ChartItem("S", 0b111)
	D = defaultdict(list)
	cand = {}
	for a,b in lazykthbest(chart, goal, 10, 10, D, cand).items():
		print a
		for ((e,j),p) in b: print e, j, exp(-p)
		print
	for a,p in lazykbest(chart, goal, 10):
		print exp(-p),a
