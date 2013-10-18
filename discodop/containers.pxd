from math import isinf, exp, log, fsum
from libc.stdlib cimport malloc, calloc, realloc, free, qsort
from libc.string cimport memcmp
cimport cython
include "constants.pxi"

ctypedef unsigned long long ULLong
ctypedef unsigned long ULong
ctypedef unsigned int UInt
ctypedef unsigned short UShort
ctypedef unsigned char UChar


cdef extern:
	int __builtin_ffsll (ULLong)
	int __builtin_ctzll (ULLong)
	int __builtin_clzll (ULLong)
	int __builtin_popcountll (ULLong)
	int __builtin_ctzl (ULong)
	int __builtin_popcountl (ULong)


cdef extern from "macros.h":
	int BITSIZE
	int BITSLOT(int b)
	int BITNSLOTS(int nb)
	void SETBIT(ULong a[], int b)
	ULong TESTBIT(ULong a[], int b)
	ULong BITMASK(int b)


@cython.final
cdef class Grammar:
	cdef Rule **bylhs, **unary, **lbinary, **rbinary
	cdef UInt *mapping, **splitmapping
	cdef UChar *fanout
	cdef ULong *chainvec
	cdef readonly currentmodel
	cdef readonly size_t nonterminals, phrasalnonterminals
	cdef readonly size_t numrules, numunary, numbinary, maxfanout
	cdef readonly bint logprob, bitpar
	cdef readonly object models
	cdef readonly bytes origrules, start
	cdef readonly unicode origlexicon
	cdef readonly list tolabel, lexical, modelnames, rulemapping
	cdef readonly dict toid, lexicalbyword, lexicalbylhs, lexicalbynum, rulenos
	cdef _convertrules(Grammar self, list rulelines, dict fanoutdict)
	cdef _indexrules(Grammar self, Rule **dest, int idx, int filterlen)
	cdef rulestr(self, Rule rule)
	cdef yfstr(self, Rule rule)


# chart improvements done:
# [x] only store probs for viterbi chart; parse forest is symbolic
# [x] common API for CFG / LCFRS: CFG: key=integer index; LCFRS: key=ChartItem
# [x] LCFRS for sent > 64 words
# [X] for CFG: index vit. probs by [cell + lhs]; need efficient way to look up
# 		vit. prob for lhs in given cell.
#     for LCFRS: index vit. probs by [lhs][item]; iterate over items for lhs.
# [x] pruning: either (1) in probs; return nan for blocked, inf for missing,
#		or (2) prepopulate parse forest; advantage: non-probabilistic
#		parsing can be pruned, viterbi probabilities can be obtained in
#		separate stage.
#		=> (3) external whitelist (current)

# [x] sampling; not well tested.
# [x] unroll list of edges in parse forest: list w/blocks of 1000 edges in arrays
# chart improvements todo:
# [ ] better to use e.g., C++ vector or other existing dynamic array for edges
# [ ] inside-outside parsing; current numpy arrays can be replaced with compact
# 		indexed arrays, to save 50%, and be consistent with chart API
# [ ] symbolic parsing; separate viterbi stage
# [ ] can we exploit bottom-up order of parser or previous ctf stages
#		to pack the parse forest?
# [ ] is it useful to have a recognition phase before making parse forest?
cdef class Chart:
	cdef public dict rankededges  # [item][n] => Entry(RankedEdge, prob)
	cdef list itemsinorder
	cdef Grammar grammar
	cdef list sent
	cdef dict inside, outside
	cdef UInt start
	cdef short lensent
	cdef public bint logprob  # False => 0 < p <= 1; True => 0 <= -log(p) < inf
	cdef public bint viterbi  # False => inside probs; True => viterbi 1-best
	cdef double subtreeprob(self, item)
	cdef lexidx(self, item, Edge *edge)
	cdef edgestr(self, item, Edge *edge)
	cdef _left(self, item, Edge *edge)
	cdef _right(self, item, Edge *edge)
	cdef left(self, RankedEdge edge)
	cdef right(self, RankedEdge edge)
	cdef copy(self, item)
	cdef ChartItem asChartItem(self, item)
	cdef size_t asCFGspan(self, item, size_t nonterminals)
	cdef getitems(self)
	cdef list getedges(self, item)


cdef struct Rule:  # total: 32 bytes.
	double prob # 8 bytes
	UInt lhs # 4 bytes
	UInt rhs1 # 4 bytes
	UInt rhs2 # 4 bytes
	UInt args # 4 bytes => 32 max vars per rule
	UInt lengths # 4 bytes => same
	UInt no # 4 bytes


@cython.final
cdef class LexicalRule:
	cdef double prob
	cdef UInt lhs
	cdef unicode word


@cython.freelist(1000)
cdef class ChartItem:
	cdef UInt label


@cython.final
cdef class SmallChartItem(ChartItem):
	cdef ULLong vec
	cdef copy(self)


@cython.final
cdef class FatChartItem(ChartItem):
	cdef ULong vec[SLOTS]
	cdef copy(self)


cdef SmallChartItem CFGtoSmallChartItem(UInt label, UChar start, UChar end)
cdef FatChartItem CFGtoFatChartItem(UInt label, UChar start, UChar end)


cdef union Position: # 8 bytes
	short mid  # CFG, end index of left child
	ULLong lvec  # LCFRS, bit vector of left child
	ULong *lvec_fat  # LCFRS > 64 words, pointer to bit vector of left child;
	# 		NB: this assumes left child is not garbage collected!


cdef struct Edge:  # 16 bytes
	Rule *rule  # ruleno may take less space as pointer, but not convenient
	Position pos


@cython.final
cdef class RankedEdge:
	# NB: 'head' is unnecessary because the head will also be the dictionary
	# key for a ranked edge, but having it as part of the object is convenient.
	cdef Edge *edge  # rule / spans of children
	cdef object head  # span / label of this node
	cdef int left, right  # rank of left / right child


cdef class Edges:
	cdef short len
	cdef Edge data[EDGES_SIZE]


# start scratch
#
#
#cdef struct CompactEdge:
#	UInt ruleno  # => idx to grammar.bylhs; define sentinel.
#	UInt posno  # => idx to an array of positions
#	# 8 bytes, but more indirection, less convenience
#
#
#cdef class ParseForest:
#	""" the chart representation of bitpar. seems to require parsing
#	in 3 stages: recognizer, enumerate analyses, get probs. """
#	#keys
#	cdef UInt *catnum			# no. of chart item -> lhs
#	cdef size_t *firstanalysis	# no. of chart item -> idx to arrays below.
#	# from firstanalysis[n] to firstanalysis[n+1] or end values.
#	cdef size_t *firstchild     # idx to child array below
#	cdef double *insideprobs	# no. of edge -> inside prob
#	cdef UInt *ruleno
#	#positive means index to lists above, negative means terminal index
#	cdef UInt *child
#
#
#cdef class DiscNode:
#	cdef int label
#	cdef tuple children
#	cdef CBitset leaves
#
#
# end scratch


# start fragments stuff

cdef struct Node:  # a node of a binary tree
	int prod # non-negative, ID of a phrasal or lexical production
	short left # >= 0: array idx to child Node; <0: idx sent[-left - 1];
	short right # >=0: array idx to child Node; -1: empty (unary Node)


cdef struct NodeArray:  # a tree as an array of Node structs
	size_t offset # index to array of nodes in treebank where this tree starts
	short len, root # number of nodes, index to root node


@cython.final
cdef class Ctrees:
	cdef Node *nodes
	cdef NodeArray *trees
	cdef long nodesleft, max
	cdef readonly size_t numnodes
	cdef readonly short maxnodes
	cdef readonly int len
	cdef list treeswithprod
	cpdef alloc(self, int numtrees, long numnodes)
	cdef realloc(self, int numtrees, int extranodes)
	cpdef add(self, list tree, dict prods)
	cdef addnodes(self, Node *source, int cnt, int root)

# end fragments stuff


# ---------------------------------------------------------------
#                          INLINED FUNCTIONS
# ---------------------------------------------------------------


cdef inline FatChartItem new_FatChartItem(UInt label):
	cdef FatChartItem item = FatChartItem.__new__(FatChartItem)
	item.label = label
	# NB: since item.vec is a static array, its elements are initialized to 0.
	return item


cdef inline SmallChartItem new_SmallChartItem(UInt label, ULLong vec):
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
		UInt nonterminals):
	""" Return an index for a regular three dimensional array:
	``chart[start][end][0] => chart[idx]`` """
	return (start * lensent + (end - 1)) * nonterminals


cdef object log1e200 = log(1e200)


cdef inline logprobadd(x, y):
	""" Add two log probabilities in log space; i.e.:

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
	if isinf(exp(diff)):	# difference is too large
		return x if x > y else y
	# otherwise return the sum.
	return x + log(1.0 + exp(diff))


cdef inline double logprobsum(list logprobs):
	""" Takes a list of log probabilities and sums them producing a
	normal probability 0 < p <= 1.0; i.e.:

	>>> a = b = c = 0.25
	>>> logprobsum([log(a), log(b), log(c)]) == sum([a, b, c]) == 0.75
	True

	:param logprobs: a list of Python floats with negative log probilities,
		s.t. 0 <= p <= inf for each p in ``logprobs``.
	:source: http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point-numbers

	Comparison of different methods: https://gist.github.com/andreasvc/6204982
	"""
	maxprob = max(logprobs)
	return exp(maxprob) * fsum([exp(prob - maxprob) for prob in logprobs])
