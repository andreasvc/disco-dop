from containers cimport ChartItem, Edge, dictcast, getlabel, getvec, itemcast, edgecast
from kbest cimport lazykthbest, lazykbest

@cython.locals(
	kbest=dict,
	d=dict,
	prunetoid=dict,
	prunetolabel=dict,
	toid=dict,
	tolabel=dict,
	prunelist=list,
	Ih=ChartItem)
cpdef list prunelist_fromchart(dict chart, ChartItem goal,
        coarsegrammar, finegrammar, int k,
        bint removeparentannotation=*, bint mergesplitnodes=*,
        int reduceh=*)

@cython.locals(
	newchart=dict)
cpdef dict merged_kbest(dict chart, ChartItem start, int k, grammar)

cpdef dict kbest_outside(dict chart, ChartItem start, int k)


@cython.locals(
	ee=Edge,
	ee2=Edge)
cdef void getitems(Edge e, tuple j, Edge rootedge, dict D,
							dict chart, dict outside)

cpdef filterchart(chart, start)

@cython.locals(
	edge=Edge,
	item=ChartItem)
cdef void filter_subtree(ChartItem start, dict chart, dict chart2)
