from array import array
from cpython.array cimport array
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcmp, memset
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
	ULong BITMASK(int b)
	int BITNSLOTS(int nb)
	void SETBIT(ULong a[], int b)
	ULong TESTBIT(ULong a[], int b)
	#int SLOTS # doesn't work
cdef extern from "arrayarray.h": pass

# FIXME: find a way to make this a constant, yet shared across modules.
DEF SLOTS = 2

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
			bint splitprune, bint markorigin, bint debug=*)
	cdef str rulerepr(self, Rule rule)
	cdef str yfrepr(self, Rule rule)

cdef struct Rule:
	double prob # 8 bytes
	UInt args # 4 bytes => 32 max vars per rule
	UInt lengths # 4 bytes => same
	UInt lhs # 4 bytes
	UInt rhs1 # 4 bytes
	UInt rhs2 # 4 bytes
	int no # 4 bytes
	# total: 32 bytes.
	#UChar fanout # 1 byte

@cython.final
cdef class LexicalRule:
	cdef UInt lhs
	cdef UInt rhs1
	cdef UInt rhs2
	cdef unicode word
	cdef double prob

cdef class ParseForest:
	""" the chart representation of bitpar. seems to require parsing
	in 3 stages: recognizer, enumerate analyses, get probs. """
	#keys
	cdef list catnum		#lhs
	cdef list firstanalysis	#idx to lists below.
	# from firstanalysis[n] to firstanalysis[n+1] or end
	#values.
	cdef list rulenumber
	cdef list firstchild
	#positive means index to lists above, negative means0 terminal index
	cdef list child

cdef class ChartItem:
	cdef UInt label
cdef class SmallChartItem(ChartItem):
	cdef ULLong vec
cdef class FatChartItem(ChartItem):
	cdef ULong vec[SLOTS]
cdef class CFGChartItem(ChartItem):
	cdef UChar start, end

cdef SmallChartItem CFGtoSmallChartItem(UInt label, UChar start, UChar end)
cdef FatChartItem CFGtoFatChartItem(UInt label, UChar start, UChar end)

# start scratch
cdef union VecType:
	ULLong vec
	ULong *vecptr

cdef class NewChartItem:
	cdef VecType vec
	cdef UInt label

cdef class DiscNode:
	cdef int label
	cdef tuple children
	cdef CBitset leaves
# end scratch

cdef class Edge:
	cdef double inside
cdef class LCFRSEdge(Edge):
	cdef double score
	cdef double prob # we could eliminate prob by using ruleno
	cdef ChartItem left
	cdef ChartItem right
	cdef long _hash
	cdef int ruleno
cdef class CFGEdge(Edge):
	cdef Rule *rule
	cdef UChar mid

@cython.final
cdef class RankedEdge:
	cdef ChartItem head
	cdef LCFRSEdge edge
	cdef int left
	cdef int right
	cdef long _hash

@cython.final
cdef class RankedCFGEdge:
	cdef UInt label
	cdef UChar start, end
	cdef CFGEdge edge
	cdef int left
	cdef int right
	cdef long _hash

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

#@cython.final
cdef class FrozenArray:
	cdef array obj

@cython.final
cdef class MemoryPool:
	cdef void reset(MemoryPool self)
	cdef void *malloc(self, int size)
	cdef void **pool
	cdef void *cur
	cdef int poolsize, limit, n, leftinpool

cdef binrepr(ULong *vec)

# to avoid overhead of __init__ and __cinit__ constructors
cdef inline FrozenArray new_FrozenArray(array data):
	cdef FrozenArray item = FrozenArray.__new__(FrozenArray)
	item.obj = data
	return item

cdef inline FatChartItem new_FatChartItem(UInt label):
	cdef FatChartItem item = FatChartItem.__new__(FatChartItem)
	item.label = label
	return item

cdef inline SmallChartItem new_ChartItem(UInt label, ULLong vec):
	cdef SmallChartItem item = SmallChartItem.__new__(SmallChartItem)
	item.label = label; item.vec = vec
	return item

cdef inline CFGChartItem new_CFGChartItem(UInt label, UChar start, UChar end):
	cdef CFGChartItem item = CFGChartItem.__new__(CFGChartItem)
	item.label = label; item.start = start; item.end = end
	return item

cdef inline LCFRSEdge new_Edge(double score, double inside, double prob,
	int rule, ChartItem left, ChartItem right):
	cdef LCFRSEdge edge = LCFRSEdge.__new__(LCFRSEdge)
	edge.score = score; edge.inside = inside; edge.prob = prob
	edge.ruleno = rule; edge.left = left; edge.right = right
	return edge

cdef inline CFGEdge new_CFGEdge(double inside, Rule *rule, UChar mid):
	cdef CFGEdge edge = CFGEdge.__new__(CFGEdge)
	edge.inside = inside; edge.rule = rule; edge.mid = mid
	return edge
