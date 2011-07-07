from array cimport array

cdef class ChartItem:
	cdef unsigned int label
	cdef unsigned long vec
	cdef long _hash

cdef class Edge:
	cdef double score
	cdef double inside
	cdef double prob
	cdef ChartItem left
	cdef ChartItem right
	cdef long _hash

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

cdef struct DTree:
	void *rule
	unsigned long vec
	bint islexical
	DTree *left, *right

cdef DTree new_DTree(Rule rule, unsigned long vec, bint islexical, DTree left, DTree right)
