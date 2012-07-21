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
	cdef UChar *fanout
	cdef UInt *mapping, **splitmapping
	cdef size_t nonterminals, numrules
	cdef public dict lexical, lexicalbylhs, toid, tolabel
	cdef public list unaryclosure
	cdef frozenset origrules
	cpdef getmapping(Grammar self, striplabelre, neverblockre, Grammar coarse,
			bint splitprune, bint markorigin)
	cdef str rulerepr(self, Rule rule)
	cdef str yfrepr(self, Rule rule)

cdef struct Rule:
	double prob # 8 bytes
	UInt args # 4 bytes => 32 max vars per rule
	UInt lengths # 4 bytes => same
	UInt lhs # 4 bytes
	UInt rhs1 # 4 bytes
	UInt rhs2 # 4 bytes
	# total: 28 bytes (32 bytes w/padding).
	#UChar fanout # 1 byte

@cython.final
cdef class ChartItem:
	cdef public ULLong vec
	cdef public UInt label

cdef class FatChartItem:
	cdef ULLong vec[7]
	cdef public UInt label


cdef union VecType:
	ULLong vec
	ULong *vecptr

cdef class NewChartItem:
	cdef VecType vec
	cdef public UInt label

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

# to avoid overhead of __init__ and __cinit__ constructors
cdef inline FrozenArray new_FrozenArray(array data):
	cdef FrozenArray item = FrozenArray.__new__(FrozenArray)
	item.data = data
	return item

cdef inline ChartItem new_ChartItem(UInt label, ULLong vec):
	cdef ChartItem item = ChartItem.__new__(ChartItem)
	item.label = label; item.vec = vec
	return item

cdef inline Edge new_Edge(double score, double inside, double prob,
	ChartItem left, ChartItem right):
	cdef Edge edge = Edge.__new__(Edge)
	edge.score = score; edge.inside = inside; edge.prob = prob
	edge.left = left; edge.right = right
	return edge
