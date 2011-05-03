import cython
from plcfrs cimport ChartItem

cdef class Edge:
	# commented because it won't work with plcfrs_cython.ChartItem ...
	#cdef public ChartItem head
	cdef public object head
	cdef public tuple tailnodes
	cdef public double weight
	cdef long _hash

@cython.locals(temp=list)
cdef inline getcandidates(dict chart, v, int k)
@cython.locals(e=Edge, j=tuple)
cdef inline lazykthbest(dict chart, v, int k, int k1, dict D, dict cand)
@cython.locals(j1=tuple)
cdef inline lazynext(dict cand, Edge e, tuple j, int k1, dict D, dict chart)
@cython.locals(result=cython.double, jj=cython.int)
cdef inline double getprob(dict chart, dict D, Edge e, tuple j)
@cython.locals(e=Edge, j=tuple, children=list, i=cython.int, ip=cython.double, p=cython.double, rhs=tuple)
cdef inline str getderivation(dict chart, dict D, tuple ej, dict tolabel)
cpdef lazykbest(chart, goal, int k, dict tolabel)
