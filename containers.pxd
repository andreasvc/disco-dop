from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcmp, memset
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

cdef extern from "macros.h":
	int BITSIZE
	int BITSLOT(int b)

@cython.final
cdef class Grammar:
	cdef Rule **unary, **lbinary, **rbinary, **bylhs
	cdef public dict lexical, lexicalbylhs, toid, tolabel
	cdef public list unaryclosure
	cdef frozenset donotprune, origrules
	cdef size_t nonterminals, numrules

@cython.final
cdef class ChartItem:
	cdef public ULLong vec
	cdef public UInt label

cdef class FatChartItem:
	cdef ULLong vec[7]
	cdef public UInt label

@cython.final
cdef class Edge:
	cdef public double score
	cdef public double inside
	cdef public double prob
	cdef public ChartItem left
	cdef public ChartItem right

#@cython.final
cdef struct Rule:
	double prob # 8 bytes
	UInt args # 4 bytes => 32 max vars per rule
	UInt lengths # 4 bytes => same
	UInt lhs # 4 bytes
	UInt rhs1 # 4 bytes
	UInt rhs2 # 4 bytes
	UChar fanout # 1 byte
	# total: 29 bytes (32 bytes w/padding).

@cython.final
cdef class LexicalRule:
	cdef public UInt lhs
	cdef public UInt rhs1
	cdef public UInt rhs2
	cdef public unicode word
	cdef public double prob

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
	Node *nodes
	short len, root

@cython.final
cdef class Ctrees:
	cpdef alloc(self, int numtrees, long numnodes)
	cdef realloc(self, int len)
	cpdef add(self, list tree, dict labels, dict prods)
	cdef NodeArray *data
	cdef long nodesleft
	cdef public long nodes
	cdef public int maxnodes
	cdef int len, max

@cython.final
cdef class CBitset:
	cdef int bitcount(self)
	cdef int nextset(self, UInt pos)
	cdef int nextunset(self, UInt pos)
	cdef void setunion(self, CBitset src)
	cdef bint superset(self, CBitset op)
	cdef bint subset(self, CBitset op)
	cdef bint disjunct(self, CBitset op)
	cdef char *data
	cdef UChar slots

@cython.final
cdef class FrozenArray:
	cdef array data

@cython.final
cdef class MemoryPool:
	cdef void reset(MemoryPool self)
	cdef void *malloc(self, int size)
	cdef void **pool
	cdef void *cur
	cdef int poolsize, limit, n, leftinpool

cdef inline FrozenArray new_FrozenArray(array data)
cdef str yfrepr(Rule rule)
cpdef inline UInt getlabel(ChartItem a)
cpdef inline ULLong getvec(ChartItem a)
cpdef inline double getscore(Edge a)
cpdef inline dict dictcast(d)
cpdef inline ChartItem itemcast(i)
cpdef inline Edge edgecast(e)
