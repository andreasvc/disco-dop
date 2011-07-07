cimport cython
from containers cimport ChartItem, Edge, Rule, Terminal
from agenda cimport heapdict

cdef extern from "bit.h":
	int nextset(unsigned long vec, int pos)
	int nextunset(unsigned long vec, int pos)
	int bitcount(unsigned long vec)
	bint testbit(unsigned long vec, int pos)
	bint testbitc(unsigned char vec, int pos)
	bint testbitint(unsigned int vec, int pos)
	bint testbitshort(unsigned short vec, int pos)
	bint bitminmax(unsigned long a, unsigned long b)

@cython.locals(
	m=cython.int,
	maxA=cython.int,
	lensent=cython.int,
	y=cython.double,
	p=cython.double,
	unary=list,
	lbinary=list,
	rbinary=list,
	lexical=dict,
	toid=dict,
	tolabel=dict,
	C=dict,
	Cx=list,
	terminal=Terminal,
	edge=Edge,
	Ih=ChartItem,
	I1h=ChartItem,
	goal=ChartItem,
	A=heapdict)
cpdef tuple parse(list sent, grammar, list tags=*, int start=*, bint exhaustive=*, estimate=*, prune=*, prunetoid=*)

@cython.locals(
	y=cython.double,
	edge=Edge,
	rule=Rule,
	I=cython.int,
	Ir=cython.ulong,
	I1h=ChartItem,
	result=list)
cdef inline list deduced_from(ChartItem Ih, double x, list Cx, list unary, list lbinary, list rbinary)

@cython.locals(
	lpos=cython.int,
	rpos=cython.int,
	n=cython.int,
	m=cython.int,
	b=cython.int)
cdef inline bint concat(Rule rule, unsigned long lvec, unsigned long rvec)

@cython.locals(
	ee=Edge,
	ee2=Edge,
	jj=tuple)
cdef getitems(Edge e, tuple j, Edge rootedge, dict D, dict chart, dict outside)

cdef Edge iscore(Edge e)

@cython.locals(
	edge=Edge,
	item=ChartItem)
cdef filter_subtree(start, chart, chart2)

@cython.locals(
	edge=Edge)
cpdef mostprobablederivation(chart, start, tolabel)

@cython.locals(
	edge=Edge)
cdef getmpd(chart, ChartItem start, tolabel)

@cython.locals(edge=Edge)
cpdef samplechart(dict chart, ChartItem start, dict tolabel)

@cython.locals(
	a=ChartItem,
	edge=Edge)
cpdef pprint_chart(dict chart, list sent, dict tolabel)

cpdef binrepr(ChartItem a, list sent)
