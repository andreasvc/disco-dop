cimport cython
from agenda cimport Entry, Agenda, nsmallest
from containers cimport ChartItem, Edge, RankedEdge

@cython.locals(edge=Edge)
cdef inline getcandidates(dict chart, ChartItem v, int k)

@cython.locals(ej=RankedEdge, entry=Entry)
cpdef inline lazykthbest(ChartItem v, int k, int k1,
			dict D, dict cand, dict chart, set explored)

@cython.locals(ej1=RankedEdge, prob=cython.double)
cdef inline lazynext(RankedEdge ej, int k1,
			dict D, dict cand, dict chart, set explored)

@cython.locals(result=cython.double, prob=cython.double, i=cython.int,
			ei=ChartItem, edge=Edge, entry=Entry)
cdef inline double getprob(dict chart, dict D, RankedEdge ej)

@cython.locals(ej=RankedEdge, edge=Edge, children=list, ei=ChartItem,
	i=cython.int, entry=Entry)
cdef inline str getderivation(RankedEdge ej,
			dict chart, dict D, dict tolabel, int n)

@cython.locals(entry=Entry, prob=cython.double)
cpdef list lazykbest(dict chart, ChartItem goal, int k, dict tolabel)

@cython.locals(v=ChartItem, c=ChartItem, e=Edge, entry=Entry)
cpdef main()
