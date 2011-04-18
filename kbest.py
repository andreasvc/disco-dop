# Implementation of Huang & Chiang (2005): Better k-best parsing
from heapq import nsmallest
from heapdict import heapdict
from collections import defaultdict
from nltk import memoize, Tree
from math import log, exp
try: frozenset
except NameError: from sets import ImmutableSet as frozenset
infinity = float('infinity')

def getcandidates(chart, v, k):
	""" Return a heap with up to k candidate arcs starting from vertex v """
	temp = [((Edge(v, a, p), (1,) * len(a)), ip) for ip,p,a in chart[v]]
	return heapdict(nsmallest(k, [(a,p) for a,p in temp if p < infinity], key=lambda x: x[1]))

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
	while len(D[v]) < k and len(cand[v]):
		# last derivation
		e, j = D[v][-1][0]
		# update the heap, adding the successors of last derivation
		lazynext(cand, v, e, j, k1, D, chart)
		# get the next best derivation and delete it from the heap
		D[v].append(cand[v].popitem())
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
			try: cand[v][e,j1] = getprob(chart, e, j1)
			except IndexError: pass	# kludge

#@memoize
def getprob(chart, e, j, visited=frozenset()):
	try: dic = getattr(getprob, "memoize_dic")
	except: dic = {}; setattr(getprob, "memoize_dic", dic)
	if (e,j) in dic: return dic[e,j]

	#print "computing", e, j	
	if e.tailnodes == []: dic[e,j] = 0.0
	elif e.tailnodes[0] in visited: dic[e,j] = infinity
	elif j == (1,) or j == (1, 1):
		dic[e,j] = e.weight + sum(chart[tail][0][0] for tail in e.tailnodes if tail in chart)
	else: 
		dic[e,j] = e.weight + sum(getprob(chart, 
							Edge(tail, chart[tail][ji-1][2], chart[tail][-ji][1]), 
							tuple(1 for x in chart[tail][ji-1][2]), #!?! assume 1-best
							visited | frozenset(e.tailnodes) if len(e.tailnodes) == 1 else frozenset())
						for tail, ji in zip(e.tailnodes, j) if tail in chart)
	return dic[e,j]

def lazykbest(chart, goal, k, tolabel):
	""" wrapper function to run lazykthbest and get the actual derivations.
	chart is a monotone hypergraph; should be acyclic unless probabilities
	resolve the cycles (maybe nonzero weights for unary productions are
	sufficient?). 
	maps ChartItems to lists of tuples with ChartItems and a weight. The
	items in each list are to be ordered as they were added by the viterbi parse,
	with the best item last.
	goal is a ChartItem that is to be the root node of the derivations.
	k is the number of derivations desired.
	tolabel is a dictionary mapping numeric IDs to the original nonterminal
	labels.  """
	D = defaultdict(list)
	cand = {}
	fchart = frozendefaultdict(tuple)
	fchart.update((a, tuple(b)) for a,b in chart.items())
	lazykthbest(fchart, goal, k, k, D, cand)
	print "derivations", len(D[goal])
	return [(getderivation(chart, D, ej, tolabel), p) for ej, p in D[goal] if p < infinity]

def getderivation(chart, D, (e, j), tolabel):
	""" Translate the (e, j) notation to an actual nltk Tree / string in bracket notation.
	e is an edge, j is a vector prescribing the rank of the corresponding tail
	node. For example, given the edge <S, [NP, VP], 1.0> and vector [2, 1], this
	points to the derivation headed by S and having the 2nd best NP and the 1st
	best VP as children. """
	# this perversely complicated expressions is necessary because D may
	# contain backpointers to vertices which are not yet in D, so we fall back
	# to the chart.
	return "(%s %s)" % (tolabel[e.head.label],
						" ".join(getderivation(chart, D, 
								D[ei][i-1][0] if ei in D else 
									(Edge(ei, chart[ei][i-1][2], chart[ei][i-1][1]), tuple(1 for x in chart[ei][i-1][2])),
							tolabel) 
						if ei in chart else str(ei.vec)
				for ei,i in zip(e.tailnodes, j)))
	#return Tree(tolabel[e.head.label], [getderivation(D, D[ei][i-1][0], tolabel) if ei in D else ei.label
	#			for ei,i in zip(e.tailnodes, j)])
	
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
	nodes, with a given weight. The case of zero tail nodes corresponds to a 
	terminal (a source vertex). """
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
	chart1 = {
			ChartItem("S", 0b111) : [
				(-log(0.25), -log(0.4), (ChartItem("NP", 0b100), ChartItem("V", 0b010), ChartItem("ADV", 0b001))),
				(-log(0.4), -log(0.7), (ChartItem("NP", 0b100), ChartItem("VP", 0b011)))],
			ChartItem("VP", 0b011) : [(-log(0.25), -log(0.5), (ChartItem("V", 0b010), ChartItem("ADV", 0b001)))],
			ChartItem("NP", 0b100) : [(-log(0.25), -log(0.5), (ChartItem("Mary", 0),)), 
										(-log(0.25), -log(0.5), (ChartItem("PN", 0b100),))],
			ChartItem("PN", 0b100) : [(-log(1.0), -log(1.0), (ChartItem("Mary", 0),))],
			ChartItem("V", 0b010) : [(-log(0.25), -log(0.5), (ChartItem("walks", 1),))],
			ChartItem("ADV", 0b001) : [(-log(0.25), -log(0.5), (ChartItem("quickly", 2),))]
			}
	tolabel = dict((a, a) for a in "S NP V ADV VP PN".split())
	chart = frozendefaultdict(tuple); chart.update((a, tuple(b)) for a,b in chart1.items())
	goal = ChartItem("S", 0b111)
	D = defaultdict(list)
	cand = {}
	k = 10
	for a,b in lazykthbest(chart, goal, k+1, k+1, D, cand).items():
		print a
		for ((e,j),p) in b: print e, j, exp(-p)
		print
	for a,p in lazykbest(chart1, goal, k, tolabel):
		print exp(-p),a
