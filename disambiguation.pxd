cimport cython
from kbest cimport lazykbest, lazykthbest
from containers cimport ChartItem, Edge, Grammar, getlabel, getvec, edgecast

@cython.locals(
	edge=Edge)
cpdef viterbiderivation(chart, start, tolabel)

@cython.locals(
	edge=Edge)
cdef getviterbi(chart, ChartItem start, tolabel)

@cython.locals(
	parsetrees=dict,
	treestr=str,
	deriv=str,
	prob=cython.double,
	maxprob=cython.double,
	m=cython.int,
	edge=Edge)
cpdef marginalize(dict chart, ChartItem start, dict tolabel, int n=*,
	bint sample=*, bint both=*, bint shortest=*, secondarymodel=*,
	bint mpd=*, dict backtransform=*)

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
