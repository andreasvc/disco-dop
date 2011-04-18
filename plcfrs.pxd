import cython
from heapdict import heapdict

cdef extern from "bit.h":
	int nextset(unsigned long vec, int pos)
	int nextunset(unsigned long vec, int pos)
	int bitcount(unsigned long vec)
	bint testbit(unsigned long vec, unsigned long pos)
	bint bitminmax(unsigned long a, unsigned long b)

cdef class ChartItem:
	cdef public int label
	cdef public unsigned long vec
	cdef long _hash

@cython.locals(
	m=cython.int,
	maxA=cython.int,
	lensent=cython.int,
	y=cython.double,
	p=cython.double,
	iscore=cython.double,
	oscore=cython.double,
	scores=tuple,
	rhs=tuple,
	unary=dict,
	lbinary=dict,
	rbinary=dict,
	lexical=dict,
	toid=dict,
	tolabel=dict,
	C=dict,
	Cx=dict,
	Ih=ChartItem,
	I1h=ChartItem,
	goal=ChartItem)
	#,A=heapdict)
cpdef tuple parse(list sent, grammar, list tags=*, start=*, bint viterbi=*, int n=*, estimate=*)

@cython.locals(
	z=cython.double,
	y=cython.double,
	I=cython.int,
	Ir=cython.ulong,
	I1h=ChartItem,
	result=list,
	rule=tuple,
	yf=tuple)
cdef inline list deduced_from(ChartItem Ih, double x, Cx, unary, lbinary, rbinary)

@cython.locals(
	lpos=cython.int,
	rpos=cython.int,
	n=cython.int,
	m=cython.int,
	b=cython.int)
cdef inline bint concat(tuple yieldfunction, unsigned long lvec, unsigned long rvec)

@cython.locals(
	entry=tuple,
	p=cython.double)
cpdef samplechart(chart, ChartItem start, dict tolabel)

