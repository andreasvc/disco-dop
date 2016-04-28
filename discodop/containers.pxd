from math import isinf, exp, log, fsum
from libc.stdlib cimport malloc, calloc, realloc, free, abort, \
		qsort, atol, strtod
from libc.string cimport memcmp, memset, memcpy
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from cpython.array cimport array
cimport cython
include "constants.pxi"

# NB: For PCFG parsing sentences longer than 256 words, change this to uint16_t
ctypedef uint8_t Idx


cdef extern from *:
	cdef bint PY2


cdef extern from "macros.h":
	int BITSIZE
	int BITSLOT(int b)
	int BITNSLOTS(int nb)
	void SETBIT(uint64_t a[], int b)
	void CLEARBIT(uint64_t a[], int b)
	uint64_t TESTBIT(uint64_t a[], int b)
	uint64_t BITMASK(int b)


@cython.final
cdef class Grammar:
	cdef ProbRule **bylhs
	cdef ProbRule **unary
	cdef ProbRule **lbinary
	cdef ProbRule **rbinary
	cdef uint32_t *mapping
	cdef uint32_t *revmap
	cdef uint32_t **splitmapping
	cdef uint8_t *fanout
	cdef uint64_t *chainvec
	cdef uint64_t *mask
	cdef readonly int currentmodel
	cdef readonly size_t nonterminals, phrasalnonterminals
	cdef readonly size_t numrules, numunary, numbinary, maxfanout
	cdef readonly bint logprob, bitpar, binarized
	cdef readonly object models
	cdef readonly str origrules, origlexicon, start
	cdef readonly list tolabel, lexical, modelnames, rulemapping
	cdef readonly dict toid, lexicalbyword, lexicalbylhs, lexicalbynum, rulenos
	cdef _convertrules(self, list rulelines, dict fanoutdict)
	cdef _indexrules(self, ProbRule **dest, int idx, int filterlen)
	cpdef rulestr(self, int n)
	cdef yfstr(self, ProbRule rule)


# chart improvements done:
# [x] only store probs for viterbi chart; parse forest is symbolic
# [x] common API for CFG / LCFRS: CFG: key=integer index; LCFRS: key=ChartItem
# [x] LCFRS for sent > 64 words
# [X] for CFG: index vit. probs by [cell + lhs]; need efficient way to look up
# 		vit. prob for lhs in given cell.
#     for LCFRS: index vit. probs by [lhs][item]; iterate over items for lhs.
# [x] pruning: either (1) in probs; return nan for blocked, inf for missing,
# 		or (2) prepopulate parse forest; advantage: non-probabilistic
# 		parsing can be pruned, viterbi probabilities can be obtained in
# 		separate stage.
# 		=> (3) external whitelist (current)
# [x] sampling; not well tested.
# [x] unroll list of edges in parse forest: list w/blocks of 1000 edges
#     in arrays

# chart improvements todo:
# [ ] better to use e.g., C++ vector or other existing dynamic array for edges
# [ ] inside-outside parsing; current numpy arrays can be replaced with compact
# 		indexed arrays, to save 50%, and be consistent with chart API
# [ ] symbolic parsing; separate viterbi stage
# [ ] can we exploit bottom-up order of parser or previous ctf stages
# 		to pack the parse forest?
# [ ] is it useful to have a recognition phase before making parse forest?
cdef class Chart:
	cdef readonly dict rankededges  # [item][n] => DoubleEntry(RankedEdge, prob)
	cdef object itemsinorder  # array or list
	cdef dict inside, outside
	cdef Grammar grammar
	cdef readonly list sent
	cdef short lensent
	cdef uint32_t start
	cdef readonly bint logprob  # False: 0 < p <= 1; True: 0 <= -log(p) < inf
	cdef readonly bint viterbi  # False: inside probs; True: viterbi 1-best
	cdef double subtreeprob(self, item)
	cdef int lexidx(self, Edge *edge)
	cdef double lexprob(self, item, Edge *edge)
	cdef edgestr(self, item, Edge *edge)
	cdef _left(self, item, Edge *edge)
	cdef _right(self, item, Edge *edge)
	cdef left(self, RankedEdge edge)
	cdef right(self, RankedEdge edge)
	cdef copy(self, item)
	cdef uint32_t label(self, item)
	cdef ChartItem asChartItem(self, item)
	cdef size_t asCFGspan(self, item, size_t nonterminals)
	cdef Edges getedges(self, item)


cdef struct ProbRule:  # total: 32 bytes.
	double prob  # 8 bytes
	uint32_t lhs  # 4 bytes
	uint32_t rhs1  # 4 bytes
	uint32_t rhs2  # 4 bytes
	uint32_t args  # 4 bytes => 32 max vars per rule
	uint32_t lengths  # 4 bytes => same
	uint32_t no  # 4 bytes


@cython.final
cdef class LexicalRule:
	cdef readonly double prob
	cdef readonly uint32_t lhs
	cdef readonly str word


@cython.freelist(1000)
cdef class ChartItem:
	cdef uint32_t label
	cdef double prob


@cython.final
cdef class SmallChartItem(ChartItem):
	cdef uint64_t vec
	cdef copy(self)


@cython.final
cdef class FatChartItem(ChartItem):
	cdef uint64_t vec[SLOTS]
	cdef copy(self)


cdef SmallChartItem CFGtoSmallChartItem(uint32_t label, Idx start, Idx end)
cdef FatChartItem CFGtoFatChartItem(uint32_t label, Idx start, Idx end)


cdef union Position:  # 8 bytes
	short mid  # CFG, end index of left child
	uint64_t lvec  # LCFRS, bit vector of left child
	uint64_t *lvec_fat  # LCFRS > 64 words, pointer to bit vector of left child;
	# 		NB: this assumes left child is not garbage collected!


cdef struct Edge:  # 16 bytes
	ProbRule *rule  # ruleno may take less space than pointer, but not convenient
	Position pos


@cython.final
cdef class Edges:
	cdef short len  # the number of edges in the head of the linked list
	cdef MoreEdges *head  # head of linked list


cdef struct EdgesStruct:
	short len  # the number of edges in the head of the linked list
	MoreEdges *head  # head of linked list


cdef struct MoreEdges:  # An unrolled linked list
	MoreEdges *prev
	Edge data[EDGES_SIZE]
	# this array is always full, unless this is the head of
	# the linked list


@cython.final
cdef class RankedEdge:
	# NB: 'head' is unnecessary because the head will also be the dictionary
	# key for a ranked edge, but having it as part of the object is convenient.
	cdef Edge *edge  # rule / spans of children
	cdef object head  # span / label of this node
	cdef int left, right  # rank of left / right child


# start scratch
#
#
# cdef struct CompactEdge:
# 	uint32_t ruleno  # => idx to grammar.bylhs; define sentinel.
# 	uint32_t posno  # => idx to an array of positions
# 	# 8 bytes, but more indirection, less convenience
#
#
# cdef class ParseForest:
# 	"""Chart representation used by bitpar.
#
# 	Seems to require parsing in 3 stages:
#   1. recognizer: determine which triples <lhs, start, end> are possible
#   2. enumerate analyses: determine edges for each <lhs, start, end>
# 	3. get probabilities: Viterbi/inside probability for each <lhs, start, end>
#   """
# 	# keys
# 	cdef uint32_t *catnum			# no. of chart item -> lhs
# 	cdef size_t *firstanalysis	# no. of chart item -> idx to arrays below.
# 	# from firstanalysis[n] to firstanalysis[n+1] or end values.
# 	cdef size_t *firstchild     # idx to child array below
# 	cdef double *insideprobs	# no. of edge -> inside prob
# 	cdef uint32_t *ruleno
# 	# positive means index to lists above, negative means terminal index
# 	cdef uint32_t *child
#
#
# cdef class DiscNode:
# 	cdef int label
# 	cdef tuple children
# 	cdef CBitset leaves
#
#
# end scratch


# start fragments stuff

cdef struct Node:  # a node of a binary tree
	int prod  # non-negative, ID of a phrasal or lexical production
	short left  # >= 0: array idx to child Node; <0: idx sent[-left - 1];
	short right  # >=0: array idx to child Node; -1: empty (unary Node)


cdef struct NodeArray:  # a tree as an array of Node structs
	unsigned int offset  # index to Node array where this tree starts
	short len, root  # number of nodes, index to root node


@cython.final
cdef class Ctrees:
	cdef Node *nodes
	cdef NodeArray *trees
	cdef long nodesleft, max
	cdef readonly size_t numnodes, numwords
	cdef readonly short maxnodes
	cdef readonly int len
	cdef readonly object prodindex
	cdef object _state
	cpdef alloc(self, int numtrees, long numnodes)
	cdef realloc(self, int numtrees, int extranodes)
	cdef addnodes(self, Node *source, int cnt, int root)


cdef struct Rule:
	uint32_t lhs, rhs1, rhs2, lengths, args


cdef union ItemType:
	void *ptr
	char *aschar
	uint32_t *asint


cdef struct DArray:
	uint8_t itemsize
	uint32_t len
	uint32_t capacity
	ItemType d


cdef class Vocabulary:
	cdef readonly dict prods  # production str. => int
	cdef readonly dict labels  # label/word str => int
	#
	cdef DArray prodbuf  # single string with all productions concatented
	cdef DArray labelbuf  # single string with all labels/words concatented
	cdef DArray labelidx  # label id => offset in labelbuf
	#
	cdef str idtolabel(self, uint32_t i)
	cdef str getlabel(self, int prodno)
	cdef str getword(self, int prodno)
	cdef bint islexical(self, int prodno)
	cdef int getprod(self, tuple r, tuple yf) except -2
	cdef int _getprodid(self, bytes prod) except -2
	cdef int _getlabelid(self, str label) except -1


cdef class FixedVocabulary(Vocabulary):
	cdef object state  # to keep buffer alive


# end fragments stuff


# ---------------------------------------------------------------
#                          INLINED FUNCTIONS
# ---------------------------------------------------------------


cdef inline FatChartItem new_FatChartItem(uint32_t label):
	cdef FatChartItem item = FatChartItem.__new__(FatChartItem)
	item.label = label
	# NB: since item.vec is a static array, its elements are initialized to 0.
	return item


cdef inline SmallChartItem new_SmallChartItem(uint32_t label, uint64_t vec):
	cdef SmallChartItem item = SmallChartItem.__new__(SmallChartItem)
	item.label = label
	item.vec = vec
	return item


cdef inline RankedEdge new_RankedEdge(
		object head, Edge *edge, int left, int right):
	cdef RankedEdge rankededge = RankedEdge.__new__(RankedEdge)
	rankededge.head = head
	rankededge.edge = edge
	rankededge.left = left
	rankededge.right = right
	return rankededge


# defined here because circular import.
cdef inline size_t cellidx(short start, short end, short lensent,
		uint32_t nonterminals):
	"""Return an index for a regular three dimensional array.

	``chart[start][end][0] => chart[idx]`` """
	return (start * lensent + (end - 1)) * nonterminals


cdef inline size_t compactcellidx(short start, short end, short lensent,
		uint32_t nonterminals):
	"""Return an index to a triangular array, given start < end.
	The result of this function is the index to chart[start][end][0]."""
	return nonterminals * (lensent * start
			- ((start - 1) * start / 2) + end - start - 1)


cdef object log1e200 = log(1e200)


cdef inline logprobadd(x, y):
	""" Add two log probabilities in log space.

	>>> a = b = 0.25
	>>> logprobadd(log(a), log(b)) == log(a + b) == log(0.5)
	True

	:param x, y: Python floats with log probabilities; -inf <= x, y <= 0.
	:source: https://facwiki.cs.byu.edu/nlp/index.php/Log_Domain_Computations
	"""
	if isinf(x):
		return y
	elif isinf(y):
		return x
	# If one value is much smaller than the other, keep the larger value.
	elif x < (y - 460):  # log(1e200)
		return y
	elif y < (x - 460):  # log(1e200)
		return x
	diff = y - x
	assert not isinf(diff)
	if isinf(exp(diff)):  # difference is too large
		return x if x > y else y
	# otherwise return the sum.
	return x + log(1.0 + exp(diff))


cdef inline double logprobsum(list logprobs):
	"""Sum a list of log probabilities producing a normal probability.

	>>> a = b = c = 0.25
	>>> logprobsum([log(a), log(b), log(c)]) == sum([a, b, c]) == 0.75
	True

	:param logprobs: a list of Python floats with negative log probilities,
		s.t. 0 <= p <= inf for each p in ``logprobs``.
	:returns: a probability p with 0 < p <= 1.0
	:source: http://blog.smola.org/post/987977550/

	Comparison of different methods: https://gist.github.com/andreasvc/6204982
	"""
	maxprob = max(logprobs)
	return exp(maxprob) * fsum([exp(prob - maxprob) for prob in logprobs])


cdef inline str yieldranges(list leaves):
	"""Convert a sorted list of indices into a string with intervals.

	Intended for discontinuous trees. The intervals are of the form
	``start:end``, where ``end`` is part of the interval. e.g.:

	>>> yieldranges([0, 1, 2, 3, 4])
	'0:1= 2:4='"""
	cdef list yields = []
	cdef int a, start = -2, prev = -2
	for a in leaves:
		if a - 1 != prev:
			if prev != -2:
				yields.append('%d:%d' % (start, prev))
			start = a
		prev = a
	yields.append('%d:%d' % (start, prev))
	return ' '.join(yields)


cdef inline short termidx(short x):
	"""Translate representation for terminal indices."""
	return -x - 1
