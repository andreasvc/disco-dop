cimport cython
from kbest cimport lazykbest, lazykthbest
from containers cimport ChartItem, Edge, getlabel, getvec, edgecast

@cython.locals(
	edge=Edge)
cpdef viterbiderivation(chart, start, tolabel)

@cython.locals(
	edge=Edge)
cdef getviterbi(chart, ChartItem start, tolabel)

@cython.locals(
	parsetrees=dict,
	prob=cython.double,
	maxprob=cython.double,
	m=cython.int,
	edge=Edge)
cpdef marginalize(chart, start, tolabel, n=*, sample=*, both=*, shortest=*, secondarymodel=*, mpd=*, backtransform=*)

@cython.locals(
	edge=Edge,
	child=ChartItem)
cpdef samplechart(dict chart, ChartItem start, dict tolabel, dict tables)

#cpdef getsamples(dict chart, ChartItem start, int n, dict tolabel)

cdef inline double sumderivs(ts, derivations)
cdef inline int minunaddressed(tt, idsremoved)

cpdef recoverfromfragments(derivation, dict backtransform)

@cython.locals(
	leaves=list,
	leafmap=dict,
	prod=str,
	rprod=str,
	result=str,
	frontier=str,
	replacement=str)
cpdef str recoverfromfragments_str(derivation, dict backtransform)
