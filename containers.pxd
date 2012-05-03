from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcmp
from array cimport array
cimport cython

ctypedef unsigned long long ULLong
ctypedef unsigned long ULong
ctypedef unsigned int UInt
ctypedef unsigned char UChar

cdef extern:
	int __builtin_ffsll (ULLong)
	int __builtin_ctzll (ULLong)
	int __builtin_clzll (ULLong)
	int __builtin_ctzl (ULong)
	int __builtin_popcountl (ULong)
	int __builtin_popcountll (ULLong)

cdef extern from "bit.h":
	int BITSIZE
	int BITSLOT(int b)

@cython.final
cdef class ChartItem:
	cdef public UInt label
	cdef public ULLong vec

@cython.final
cdef class Edge:
	cdef public double score
	cdef public double inside
	cdef public double prob
	cdef public ChartItem left
	cdef public ChartItem right

@cython.final
cdef class LexicalRule:
	cdef public UInt lhs
	cdef public UInt rhs1
	cdef public UInt rhs2
	cdef public unicode word
	cdef public double prob

@cython.final
cdef class Rule:
	cdef public UInt lhs
	cdef public UInt rhs1
	cdef public UInt rhs2
	cdef public double prob
	cdef public array args
	cdef public array lengths
	cdef UInt * _args
	cdef unsigned short * _lengths

@cython.final
cdef class RankedEdge:
	cdef public ChartItem head
	cdef public Edge edge
	cdef public int left
	cdef public int right

cdef struct Node:
	int label, prod
	short left, right

cdef struct NodeArray:
	short len, root
	Node *nodes

@cython.final
cdef class Ctrees:
	cdef public int maxnodes
	cdef public long nodes
	cdef int len, max,
	cdef long nodesleft
	cdef NodeArray *data
	cpdef alloc(self, int numtrees, long numnodes)
	cdef realloc(self, int len)
	cpdef add(self, list tree, dict labels, dict prods)

@cython.final
cdef class CBitset:
	cdef char *data
	cdef UChar slots
	cdef int bitcount(self)
	cdef int nextset(self, UInt pos)
	cdef int nextunset(self, UInt pos)
	cdef void setunion(self, CBitset src)
	cdef bint superset(self, CBitset op)
	cdef bint subset(self, CBitset op)
	cdef bint disjunct(self, CBitset op)

@cython.final
cdef class FrozenArray:
	cdef array data

@cython.final
cdef class MemoryPool:
	cdef int poolsize, limit, n, leftinpool
	cdef void **pool
	cdef void *cur
	cdef void *malloc(self, int size)
	cdef void reset(MemoryPool self)

cdef inline FrozenArray new_FrozenArray(array data)
cpdef inline UInt getlabel(ChartItem a)
cpdef inline ULLong getvec(ChartItem a)
cpdef inline double getscore(Edge a)
cpdef inline dict dictcast(d)
cpdef inline ChartItem itemcast(i)
cpdef inline Edge edgecast(e)
