import cython
from plcfrs cimport ChartItem

cdef class Edge:
	#cdef public ChartItem head
	cdef public object head
	cdef public tuple tailnodes
	cdef public double weight
	cdef long _hash

cdef getcandidates(dict chart, v, int k)
cdef lazykthbest(dict chart, v, int k, int k1, dict D, dict cand)
cdef lazynext(dict cand, Edge e, tuple j, int k1, dict D, dict chart)
cdef getprob(dict chart, dict D, Edge e, tuple j, memoize=*)
cdef getderivation(dict chart, dict D, tuple ej, dict tolabel)
cpdef lazykbest(chart, goal, int k, dict tolabel)
