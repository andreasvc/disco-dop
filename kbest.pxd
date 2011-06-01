import cython
from cpq cimport heapdict
#from plcfrs cimport ChartItem
from plcfrs_cython cimport ChartItem

cdef class Edge:
	cdef public tuple tailnodes
	cdef public double weight
	cdef long _hash

@cython.locals(temp=list)
cdef inline getcandidates(dict chart, ChartItem v, int k)
@cython.locals(e=Edge, j=tuple)
cdef inline lazykthbest(ChartItem v, int k, int k1, dict D, dict cand, dict chart)
@cython.locals(j1=tuple)
cdef inline lazynext(ChartItem v, Edge e, tuple j, int k1, dict D, dict cand, dict chart)
@cython.locals(result=list, i=cython.int, ei=ChartItem)
cdef inline double getprob(dict chart, dict D, Edge e, tuple j)
@cython.locals(e=Edge, j=tuple, children=list, i=cython.int, ip=cython.double, p=cython.double, rhs=tuple)
cdef inline str getderivation(ChartItem v, tuple ej, dict chart, dict D, dict tolabel)
cpdef lazykbest(chart, goal, int k, dict tolabel)
