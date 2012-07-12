cimport cython
from kbest cimport lazykthbest, lazykbest
from agenda cimport Entry
from containers cimport ChartItem, Edge, RankedEdge, Grammar,\
			dictcast, getlabel, getvec, itemcast, edgecast

@cython.locals(
	kbest=dict,
	d=dict,
	whitelist=list,
	Ih=ChartItem)
cpdef list prunechart(dict chart, ChartItem goal, Grammar coarse, Grammar fine,
	int k, bint removeparentannotation=*, bint mergesplitnodes=*, int reduceh=*)

@cython.locals(
	newchart=dict)
cdef dict merged_kbest(dict chart, ChartItem start, int k, grammar)

@cython.locals(
	entry=Entry,
	e=Edge,
	D=dict,
	outside=dict)
cpdef dict kbest_items(dict chart, ChartItem start, int k)

@cython.locals(
	e=Edge,
	eejj=RankedEdge,
	prob=cython.double,
	entry=Entry)
cdef getitems(RankedEdge ej, double rootprob, dict D,
	dict chart, dict outside)

cpdef filterchart(chart, start)

@cython.locals(
	edge=Edge,
	item=ChartItem)
cdef void filter_subtree(ChartItem start, dict chart, dict chart2)
