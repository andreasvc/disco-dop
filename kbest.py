# Implementation of Huang & Chiang (2005): Better k-best parsing
from heapq import nsmallest
from heapdict import heapdict	# fixme: heapdict does not do stable sort
from collections import defaultdict
from nltk import Tree
from math import log, exp
from plcfrs import ChartItem
infinity = float('infinity')

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
		return "<%s, [%s], %f>" % (self.head, ", ".join(map(repr, self.tailnodes)), exp(-self.weight))

def second(x): return x[1]
def getcandidates(chart, v, k):
	""" Return a heap with up to k candidate arcs starting from vertex v """
	temp = [((Edge(v, a, p), (1,) * len(a)), ip) for ip,p,a in chart[v]]
	return heapdict(nsmallest(k, [(a,ip) for a,ip in temp if ip < infinity], key=second))

def lazykthbest(chart, v, k, k1, D, cand):
	# k1 is the global k
	# kth derivation already computed?
	if len(D.get(v,[])) >= k: return
	# first visit of vertex v?
	if v not in cand:
		# initialize the heap
		cand[v] = getcandidates(chart, v, k1)
		# 1-best
		if cand[v]: D.setdefault(v, []).append(cand[v].popitem())
	# NB: there seems to be an error in the pseudocode of Huang & Chiang,
	# their while loop checks if cand[v] is nonempty, this check means
	# that the successors of the last candidate will not be explored.
	if not cand[v]: return D
	while len(D[v]) < k: # and len(cand[v]):
		# last derivation
		e, j = D[v][-1][0]
		# update the heap, adding the successors of last derivation
		lazynext(cand, v, e, j, k1, D, chart)
		# get the next best derivation and delete it from the heap
		try: D[v].append(cand[v].popitem())
		except IndexError: break
	return D

def lazynext(cand, v, e, j, k1, D, chart):
	# add the |e| neighbors
	for i,ei in enumerate(e.tailnodes):
		j1 = tuple([x+1 if xi == i else x for xi, x in enumerate(j)])
		# recursively solve a subproblem
		lazykthbest(chart, ei, j1[i], k1, D, cand)
		# if it exists and is not in heap yet
		#if len(chart[ei]) >= j1[i] <= len(D[ei]) and (e, j1) not in cand[v]:
		if j1[i] <= len(D[ei]) and (e, j1) not in cand[v]:
			# add it to the heap
			#cand[v][e,j1] = D[ei][j1[i] - 1][1]
			cand[v][e,j1] = getprob(chart, D, e, j1)
			#except IndexError: pass	# kludge

def getprob(chart, D, e, j, memoize={}): #frozenset()):
	#memoize = getprob.memoize
	if (e,j) in memoize: return memoize[e,j]

	if e.tailnodes == []: memoize[e,j] = 0.0
	#elif e.tailnodes[0] in visited: memoize[e,j] = infinity
	elif j == (1,) or j == (1, 1):
		memoize[e,j] = e.weight + sum(chart[tail][0][0] for tail in e.tailnodes if tail in chart)
	else: 
		memoize[e,j] = e.weight + sum(
						D[tail][ji - 1][1] if tail in D else chart[tail][ji - 1][0] 
						#	getprob(chart, 
						#		Edge(tail, chart[tail][ji-1][2], chart[tail][ji-1][1]), 
						#		(1,) * len(chart[tail][ji-1][2]), #assume 1-best
						#		visited | frozenset(e.tailnodes) if len(e.tailnodes) == 1 else frozenset())
						for tail, ji in zip(e.tailnodes, j) if tail in chart)
	return memoize[e,j]
#getprob.memoize = {}

def getderivation(chart, D, ej, tolabel):
	""" Translate the (e, j) notation to an actual nltk Tree / string in bracket notation.
	e is an edge, j is a vector prescribing the rank of the corresponding tail
	node. For example, given the edge <S, [NP, VP], 1.0> and vector [2, 1], this
	points to the derivation headed by S and having the 2nd best NP and the 1st
	best VP as children. """
	e, j = ej
	# this perversely complicated expressions is necessary because D may
	# contain backpointers to vertices which are not yet in D, so we fall back
	# to the chart.
	return "(%s %s)" % (tolabel[e.head.label],
						" ".join([getderivation(chart, D, 
								D[ei][i-1][0] if ei in D else 
									(Edge(ei, chart[ei][i-1][2], chart[ei][i-1][1]), (1,) * len(chart[ei][i-1][2])),
							tolabel) 
						if ei in chart else str(ei.vec)
				for ei,i in zip(e.tailnodes, j)]))
	#return Tree(tolabel[e.head.label], [getderivation(D, D[ei][i-1][0], tolabel) if ei in D else ei.label
	#			for ei,i in zip(e.tailnodes, j)])
	
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
	D = {} #defaultdict(list)
	cand = {}
	chart = dict(chart)
	lazykthbest(chart, goal, k, k, D, cand)
	#for v,l in D.items():
	#	print tolabel[v.label], bin(v.vec)
	#	for a,b in l: print a,exp(-b)
	#print "derivations", len(D[goal])
	return [(getderivation(chart, D, ej, tolabel), p) for ej, p in D[goal] if p < infinity]

def main():
	toid = dict([a[::-1] for a in enumerate("S NP V ADV VP PN Mary walks quickly".split())])
	tolabel = dict([a[::-1] for a in toid.items()])
	def ci(label, vec):
		return ChartItem(toid[label], vec)
	goal = ci("S", 0b111)
	chart = {
			ci("S", 0b111) : [
				(-log(0.5*0.4), -log(0.4), (ci("NP", 0b100), ci("V", 0b010), ci("ADV", 0b001))),
				(-log(0.25*0.7), -log(0.7), (ci("NP", 0b100), ci("VP", 0b011)))],
			ci("VP", 0b011) : [(-log(0.5), -log(0.5), (ci("V", 0b010), ci("ADV", 0b001)))],
			ci("NP", 0b100) : [(-log(0.5), -log(0.5), (ci("Mary", 0),)), 
										(-log(0.5), -log(0.5), (ci("PN", 0b100),))],
			ci("PN", 0b100) : [(-log(1.0), -log(1.0), (ci("Mary", 0),))],
			ci("V", 0b010) : [(-log(1.0), -log(1.0), (ci("walks", 1),))],
			ci("ADV", 0b001) : [(-log(1.0), -log(1.0), (ci("quickly", 2),))]
			}
	D = {} #defaultdict(list)
	cand = {}
	k = 10
	for a,b in lazykthbest(chart, goal, k+1, k+1, D, cand).items():
		print a
		for ((e,j),p) in b: print e, j, exp(-p)
		print
	for a,p in lazykbest(chart, goal, k, tolabel):
		print exp(-p), a
if __name__ == '__main__': main()
