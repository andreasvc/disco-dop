from kbest cimport lazykbest, lazykthbest
from containers cimport ChartItem, Edge, getlabel, getvec

@cython.locals(
	edge=Edge)
cpdef mostprobablederivation(chart, start, tolabel)

@cython.locals(
	edge=Edge)
cdef getmpd(chart, ChartItem start, tolabel)

@cython.locals(
	parsetrees=dict,
	prob=cython.double,
	maxprob=cython.double,
	m=cython.int,
	edge=Edge)
cpdef mostprobableparse(chart, start, tolabel, n=*, sample=*, both=*, shortest=*, secondarymodel=*)

@cython.locals(
	edge=Edge,
	child=ChartItem)
cpdef samplechart(dict chart, ChartItem start, dict tolabel)

cpdef getsamples(dict chart, ChartItem start, int n, dict tolabel)

cdef inline double sumderivs(ts, derivations)
cdef inline int minunaddressed(tt, idsremoved)
