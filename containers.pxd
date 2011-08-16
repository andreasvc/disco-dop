from array cimport array

cdef class ChartItem:
	cdef public unsigned int label
	cdef public unsigned long vec

cdef class Edge:
	cdef public double score
	cdef public double inside
	cdef public double prob
	cdef public ChartItem left
	cdef public ChartItem right

cdef class Terminal:
	cdef public unsigned int lhs
	cdef public unsigned int rhs1
	cdef public unsigned int rhs2
	cdef public unicode word
	cdef public double prob

cdef class Rule:
	cdef public unsigned int lhs
	cdef public unsigned int rhs1
	cdef public unsigned int rhs2
	cdef public double prob
	cdef public array args
	cdef public array lengths
	cdef unsigned int * _args
	cdef unsigned short * _lengths

cdef class RankedEdge:
	cdef public ChartItem head
	cdef public Edge edge
	cdef public int left
	cdef public int right

cdef struct DTree:
	void *rule
	unsigned long vec
	bint islexical
	DTree *left, *right

cdef DTree new_DTree(Rule rule, unsigned long vec, bint islexical, DTree left, DTree right)

cpdef inline unsigned int getlabel(ChartItem a)
cpdef inline unsigned long getvec(ChartItem a)
cpdef inline double getscore(Edge a)
cpdef inline dict dictcast(d)
cpdef inline ChartItem itemcast(i)
cpdef inline Edge edgecast(e)
