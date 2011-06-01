from array cimport array

cdef class Pair:
	cdef object a
	cdef object b

cdef class ChartItem:
	cdef public int label
	cdef public unsigned long vec
	cdef long _hash

cdef class Edge:
	cdef public double inside
	cdef public double prob
	cdef public ChartItem left
	cdef public ChartItem right

cdef class Terminal:
	cdef public int lhs
	cdef public int rhs1
	cdef public int rhs2
	cdef public unicode word
	cdef public double prob

cdef class Rule:
	cdef public int lhs
	cdef public int rhs1
	cdef public int rhs2
	cdef public array args
	cdef public array lengths
	cdef public double prob
