""" Implementation of Huang & Chiang (2005): Better k-best parsing
"""
from math import log, exp, fsum
from cpq import heapdict
#from heapq import nsmallest

class Edge(object):
	""" An edge is defined as an arc between a head node and zero or more tail
	nodes, with a given weight. The case of zero tail nodes corresponds to a 
	terminal (a source vertex). """
	__slots__ = ("head", "tailnodes", "weight", "_hash")
	def __init__(self, head, tailnodes, weight):
		self.head = head; self.tailnodes = tailnodes; self.weight = weight
		self._hash = hash((head, tailnodes, weight))
	def __hash__(self):
		return self._hash
	def __cmp__(self, other):
		if (self.head == other.head
			and self.tailnodes == other.tailnodes
			and self.weight == other.weight):
			return 0
		else: return 1	# we only care about defining equality
	def __repr__(self):
		return "<%s, [%s], %f>" % (self.head,
			", ".join(map(repr, self.tailnodes)), exp(-self.weight))

def getcandidates(chart, v, k):
	""" Return a heap with up to k candidate arcs starting from vertex v """
	# sort on rank vector as well to have ties resolved in FIFO order
	# otherwise the sequence (0, 0) -> (1, 0) -> (1, 1) -> (0, 1) -> (1, 1)
	# can occur (given that the first two have probability x and the latter
	# three probability y), in which case insertion order should count.
	# Otherwise (1, 1) ends up in D[v] after which (0. 1) generates it
	# as a neighbor and puts it in cand[v] for a second time.
	return heapdict([((Edge(v, rhs, p), (0,) * len(rhs)),
						(ip, (0,) * len(rhs)))
				for ip,p,rhs in chart.get(v, [])[:k]])

def lazykthbest(v, k, k1, D, cand, chart):
	# k1 is the global k
	# first visit of vertex v?
	if v not in cand:
		# initialize the heap
		cand[v] = getcandidates(chart, v, k1)
	while len(D.get(v, [])) < k:
		if v in D:
			# last derivation
			e, j = D[v][-1][0]
			# update the heap, adding the successors of last derivation
			lazynext(e, j, k1, D, cand, chart)
		# get the next best derivation and delete it from the heap
		if cand[v]:
			a, b = cand[v].popitem()
			D.setdefault(v, []).append((a, b[0]))
			#print "D[v] <=", D[v][-1]
		else: break
	return D

def lazynext(e, j, k1, D, cand, chart):
	# add the |e| neighbors
	for i, ei in enumerate(e.tailnodes):
		# j1 is j but incremented at index i
		j1 = j[:i] + (j[i] + 1,) + j[i + 1:]
		assert j != j1
		# recursively solve a subproblem
		# NB: increment j1[i] again because j is zero-based and k isn't
		lazykthbest(ei, j1[i] + 1, k1, D, cand, chart)
		# if it exists and is not in heap yet
		if j1[i] < len(D.get(ei, [])) and (e, j1) not in cand[e.head]:
			# add it to the heap
			cand[e.head][e, j1] = (getprob(chart, D, e, j1), j1)
			#print "cand[v] <=", e, j1

def getprob(chart, D, e, j):
	result = [e.weight]
	for ei, i in zip(e.tailnodes, j):
		if ei in D: result.append(D[ei][i][1])
		elif i == 0: result.append(chart[ei][0][0])
		else: raise ValueError
	return fsum(result)

def getderivation(chart, D, ej, tolabel):
	""" Translate the (e, j) notation to an actual tree string in
	bracket notation.  e is an edge, j is a vector prescribing the rank of the
	corresponding tail node. For example, given the edge <S, [NP, VP], 1.0> and
	vector [2, 1], this points to the derivation headed by S and having the 2nd
	best NP and the 1st best VP as children.
	"""
	e, j = ej; children = []
	for ei, i in zip(e.tailnodes, j):
		if ei in chart:
			if ei not in D:
				if i == 0:
					ip, p, rhs = chart[ei][i]
					D[ei] = [((Edge(ei, rhs, p), (0,) * len(rhs)), ip)]
				else: raise ValueError
			children.append(getderivation(chart, D, D[ei][i][0], tolabel))
		else:
			# this must be a terminal
			children.append(str(ei.vec))
	"""
	# debugging duplicates
	s = "(%s %s)" % (tolabel[e.head.label], " ".join(children))
	if s not in getderivation.mem:
		getderivation.mem[s] = ej
	elif getderivation.mem[s] != ej:
		print 'DUPLICATE', ej[::-1], 'and', getderivation.mem[s][::-1],
		print '=>\n', s
		for ee, jj in (ej, getderivation.mem[s]):
			print jj, ':', ee
			agenda = zip(ee.tailnodes, jj)
			while agenda:
				ei, i = agenda.pop()
				if ei in D:
					((e, j), w) = D[ei][i]
					agenda.extend(zip(e.tailnodes, j))
					print tolabel[e.head.label],
					print bin(e.head.vec)[2:][::-1], ":",
					print " ".join(["%s[%s]" % (tolabel[a.label],
								bin(a.vec)[2:][::-1]) for a in e.tailnodes]),
					print j, exp(-w)
				else:
					print 'terminal', ei.vec
			print
		exit()
	"""
			
	return "(%s %s)" % (tolabel[e.head.label], " ".join(children))
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
	return [(getderivation(chart, D, ej, tolabel), p) for ej, p in D[goal]]

def main():
	from plcfrs import ChartItem
	toid = dict([a[::-1] for a in
			enumerate("S NP V ADV VP PN Mary walks quickly".split())])
	tolabel = dict([a[::-1] for a in toid.items()])
	def ci(label, vec):
		return ChartItem(toid[label], vec)
	goal = ci("S", 0b111)
	chart = {
			ci("S", 0b111) : [
				(-log(0.5*0.4), -log(0.4),
						(ci("NP", 0b100), ci("V", 0b010), ci("ADV", 0b001))),
				(-log(0.25*0.7), -log(0.7),
						(ci("NP", 0b100), ci("VP", 0b011)))],
			ci("VP", 0b011) : [
					(-log(0.5), -log(0.5), (ci("V", 0b010), ci("ADV", 0b001))),
					(-log(0.4), -log(0.4), (ci("walks", 1), ci("ADV", 0b001)))
					],
			ci("NP", 0b100) : [(-log(0.5), -log(0.5), (ci("Mary", 0),)),
								(-log(0.5), -log(0.5), (ci("PN", 0b100),))],
			ci("PN", 0b100) : [(-log(1.0), -log(1.0), (ci("Mary", 0),)),
							],  # (-log(0.9), -log(0.9), (ci("NP", 0b100),))],
			ci("V", 0b010) : [(-log(1.0), -log(1.0), (ci("walks", 1),))],
			ci("ADV", 0b001) : [(-log(1.0), -log(1.0), (ci("quickly", 2),))]
			}
	assert ci("NP", 0b100) == ci("NP", 0b100)
	D = {}
	cand = {}
	k = 10
	for a,b in lazykthbest(goal, k, k, D, cand, chart).items():
		print tolabel[a.label], bin(a.vec)[2:]
		for (e, j), p in b:
			print tolabel[e.head.label], ":",
			print " ".join([tolabel[a.label] for a in e.tailnodes]),
			print exp(-e.weight), j, exp(-p)
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
