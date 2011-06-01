""" Implementation of Huang & Chiang (2005): Better k-best parsing
"""
from math import exp, fsum
from cpq import heapdict
from heapq import nsmallest
from operator import itemgetter
from containers import ChartItem
#from plcfrs import ChartItem

NONE = ChartItem(0, 0)		# sentinel node

#class Edge(object):
#	""" An edge is defined as an arc between a head node and zero or more tail
#	nodes, with a given weight. The case of zero tail nodes corresponds to a 
#	terminal (a source vertex). """
#	__slots__ = ("tailnodes", "weight", "_hash")
#	def __init__(self, tailnodes, weight):
#		self.tailnodes = tailnodes; self.weight = weight
#		self._hash = hash((tailnodes, weight))
#	def __hash__(self):
#		return self._hash
#	def __cmp__(self, other):
#		if (self.tailnodes == other.tailnodes
#			and self.weight == other.weight):
#			return 0
#		else: return 1	# we only care about defining equality
#	def __repr__(self):
#		return "<[%s], %f>" % (", ".join(map(repr, self.tailnodes)),
#									exp(-self.weight))

def getcandidates(chart, v, k):
	""" Return a heap with up to k candidate arcs starting from vertex v """
	# NB: the priority queue should either do a stable sort, or should
	# sort on rank vector as well to have ties resolved in FIFO order;
	# otherwise the sequence (0, 0) -> (1, 0) -> (1, 1) -> (0, 1) -> (1, 1)
	# can occur (given that the first two have probability x and the latter
	# three probability y), in which case insertion order should count.
	# Otherwise (1, 1) ends up in D[v] after which (0. 1) generates it
	# as a neighbor and puts it in cand[v] for a second time.
	return heapdict([((edge, (0,) * (1 if edge.right == NONE else 2)),
				edge.inside) for edge in nsmallest(k, chart.get(v, []))])
	#return heapdict([((Edge(rhs, p), (0,) * len(rhs)), (ip, (0,) * len(rhs)))
	#			for ip,p,rhs in 
	#			nsmallest(k, chart.get(v, []), key=itemgetter(0))])

def lazykthbest(v, k, k1, D, cand, chart):
	# k1 is the global k
	# first visit of vertex v?
	if v not in cand:
		# initialize the heap
		cand[v] = getcandidates(chart, v, k1)
	while v not in D or len(D[v]) < k:
		if v in D:
			# last derivation
			e, j = D[v][-1][0]
			# update the heap, adding the successors of last derivation
			lazynext(v, e, j, k1, D, cand, chart)
		# get the next best derivation and delete it from the heap
		if cand[v]:
			D.setdefault(v, []).append(cand[v].popitem())
		else: break
	return D

def lazynext(v, e, j, k1, D, cand, chart):
	unary = e.right == NONE
	# add the |e| neighbors
	for i in range(1 if unary else 2):
		if i == 0:
			ei = e.left
			j1 = (j[0] + 1,) if unary else (j[0] + 1, j[1])
		elif i == 1:
			ei = e.right
			j1 = (j[0], j[1] + 1)
		# recursively solve a subproblem
		# NB: increment j1[i] again because j is zero-based and k is not
		lazykthbest(ei, j1[i] + 1, k1, D, cand, chart)
		# if it exists and is not in heap yet
		if (ei in D and j1[i] < len(D[ei])) and (e, j1) not in cand[v]:
			# add it to the heap
			cand[v][e, j1] = getprob(chart, D, e, j1)

def getprob(chart, D, e, j):
	result = [e.prob]
	for ei, i in zip((e.left, e.right), j):	#zip will truncate according to j
		if ei in D:
			result.append(D[ei][i][1])
		elif i == 0:
			edge = chart[ei][0]
			result.append(edge.inside)
		else: raise ValueError("non-zero rank vector not part of explored derivations")
	# it's probably pointless to use fsum for only 3 values, but well...
	return fsum(result)

def getderivation(v, ej, D, chart, tolabel):
	""" Translate the (e, j) notation to an actual tree string in
	bracket notation.  e is an edge, j is a vector prescribing the rank of the
	corresponding tail node. For example, given the edge <S, [NP, VP], 1.0> and
	vector [2, 1], this points to the derivation headed by S and having the 2nd
	best NP and the 1st best VP as children.
	"""
	e, j = ej; children = []
	for ei, i in zip((e.left, e.right), j):
		if ei in chart:
			if ei not in D:
				if i == 0:
					edge = chart[ei][i]
					D[ei] = [((edge, (0,) * (1 if edge.right == NONE else 2)),
																edge.inside)]
				else: raise ValueError
			children.append(getderivation(ei, D[ei][i][0], D, chart, tolabel))
		else:
			# this must be a terminal
			children.append(str(ei.vec))
	## debugging duplicates
	#s = "(%s %s)" % (tolabel[v.label], " ".join(children))
	#if s not in getderivation.mem:
	#	getderivation.mem[s] = ej
	#elif getderivation.mem[s] != ej:
	#	print 'DUPLICATE', ej[::-1], 'and', getderivation.mem[s][::-1],
	#	print '=>\n', s
	#	for ee, jj in (ej, getderivation.mem[s]):
	#		print jj, ':', ee
	#		agenda = zip(ee.tailnodes, jj)
	#		while agenda:
	#			ei, i = agenda.pop()
	#			if ei in D:
	#				((e, j), w) = D[ei][i]
	#				agenda.extend(zip(e.tailnodes, j))
	#				print tolabel[ei.label],
	#				print bin(ei.vec)[2:][::-1], ":",
	#				print " ".join(["%s[%s]" % (tolabel[a.label],
	#							bin(a.vec)[2:][::-1]) for a in e.tailnodes]),
	#				print j, exp(-w)
	#			else:
	#				print 'terminal', ei.vec
	#		print
	#	exit()
			
	return "(%s %s)" % (tolabel[v.label], " ".join(children))
#getderivation.mem = {}
	
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
	D = {}
	cand = {}
	lazykthbest(goal, k, k, D, cand, chart)
	return [(getderivation(goal, ej, D, chart, tolabel), p) for ej, p in D[goal]]

def main():
	from math import log
	from containers import Edge
	toid = dict([a[::-1] for a in
			enumerate("S NP V ADV VP VP2 PN Mary walks quickly".split())])
	tolabel = dict([a[::-1] for a in toid.items()])
	def ci(label, vec):
		return ChartItem(toid[label], vec)
	def l(a): return -log(a)
	goal = ci("S", 0b111)
	chart = {
			ci("S", 0b111) : [
				Edge(l(0.5*0.4), l(0.4),
						ci("NP", 0b100), ci("VP", 0b011)),
				Edge(l(0.25*0.7), l(0.7),
						ci("NP", 0b100), ci("VP2", 0b011))],
			ci("VP", 0b011) : [
				Edge(l(0.5), l(0.5), ci("V", 0b010), ci("ADV", 0b001)),
				Edge(l(0.4), l(0.4), ci("walks", 1), ci("ADV", 0b001))],
			ci("VP2", 0b011) : [
				Edge(l(0.5), l(0.5), ci("V", 0b010), ci("ADV", 0b001)),
				Edge(l(0.4), l(0.4), ci("walks", 1), ci("ADV", 0b001))],
			ci("NP", 0b100) : [Edge(l(0.5), l(0.5), ci("Mary", 0), NONE),
							Edge(l(0.5), l(0.5), ci("PN", 0b100), NONE)],
			ci("PN", 0b100) : [Edge(l(1.0), l(1.0), ci("Mary", 0), NONE),
							Edge(l(0.9), l(0.9), ci("NP", 0b100), NONE)],
			ci("V", 0b010) : [Edge(l(1.0), l(1.0), ci("walks", 1), NONE)],
			ci("ADV", 0b001) : [Edge(l(1.0), l(1.0), ci("quickly", 2), NONE)]
		}
	assert ci("NP", 0b100) == ci("NP", 0b100)
	cand = {}
	D = {}
	k = 10
	for v, b in lazykthbest(goal, k, k, D, cand, chart).items():
		print tolabel[v.label], bin(v.vec)[2:]
		for (e, j), ip in b:
			print tolabel[v.label], ":",
			print " ".join([tolabel[a.label] for a, _ in zip((e.left, e.right), j)]),
			print exp(-e.prob), j, exp(-ip)
		print
	from pprint import pprint
	print "tolabel",
	pprint(tolabel)
	print "candidates",
	for a in cand:
		print a, len(cand[a]),
		pprint(cand[a].items())
	
	print "\nderivations"
	for a,p in lazykbest(chart, goal, k, tolabel):
		print exp(-p), a

if __name__ == '__main__': main()
