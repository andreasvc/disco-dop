from libc.stdlib cimport malloc, free
from libc.string cimport memcmp
from array cimport array

ctypedef unsigned long ULong

cdef class ChartItem:
	cdef public unsigned int label
	cdef public unsigned long long vec

cdef class Edge:
	cdef public double score
	cdef public double inside
	cdef public double prob
	cdef public ChartItem left
	cdef public ChartItem right

cdef class LexicalRule:
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

cdef struct Node:
	int label, prod
	short left, right

cdef struct NodeArray:
	int len
	Node *nodes

cdef class Ctrees:
	cdef int len, max, maxnodes
	cdef NodeArray *data
	cpdef alloc(self, int numtrees, int numnodes, int maxnodes)
	cpdef add(self, list tree, dict labels, dict prods)

cdef class CBitset:
	cdef ULong *data
	cdef long _hash
	cdef char SLOTS
	cdef inline sethash(self)

cpdef inline unsigned int getlabel(ChartItem a)
cpdef inline unsigned long long getvec(ChartItem a)
cpdef inline double getscore(Edge a)
cpdef inline dict dictcast(d)
cpdef inline ChartItem itemcast(i)
cpdef inline Edge edgecast(e)
