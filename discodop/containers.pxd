from math import isinf, exp, fsum, log
from libc.stdlib cimport malloc, calloc, realloc, free, abort, \
		qsort, atol, strtod
from libc.math cimport fabs, log, exp
from libc.string cimport memcmp, memset, memcpy
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int16_t, int32_t
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp.string cimport string
from cpython.array cimport array
from libc.stdio cimport FILE, fopen, fwrite, fclose
from cpython.buffer cimport PyBUF_SIMPLE, Py_buffer, PyObject_CheckBuffer, \
		PyObject_GetBuffer, PyBuffer_Release
cimport cython
from cpython.array cimport array, clone
from .bit cimport nextset, nextunset, anextset, anextunset
from .bit cimport bit_popcount

include "constants.pxi"

ctypedef uint16_t Idx  # PCFG chart indices; max 16 bits.

cdef extern from "_containers.h":
	ctypedef uint32_t ItemNo  # numeric ID for chart item
	ctypedef uint32_t Label  # numeric ID for nonterminal label; max 32 bits.
	ctypedef double Prob  # precision for regular or log probabilities.
	# FIXME: considerations when setting to float precision: numpy arrays,
	# cpython arrays and functions from math.h may need to be changed; Python
	# float is double.

cdef extern from "macros.h":
	int BITSIZE
	int BITSLOT(int b)
	int BITNSLOTS(int nb)
	void SETBIT(uint64_t a[], int b)
	void CLEARBIT(uint64_t a[], int b)
	uint64_t TESTBIT(uint64_t a[], int b)
	uint64_t BITMASK(int b)


# Instead of Python's dict, we want a C/C++ hash table that directly stores
# values instead of boxed objects using pointers. This gives both a speed and
# memory advantage. One option is to use klib, used by e.g., pandas. klib
# offers a generic hash table using macros; a disadvantage is that in Cython
# this requires instatiating a new hash table type in a C header for each
# combination of <key_type, value_type>. There are also several C++ hash table
# implementations following the unordered_map API: e.g., google sparse hash.
# These can be instantiated from Cython code with a specified value type.
# However, non-primitive key types requiring a custom hash and equality
# function still need to be defined in a header, because Cython does not
# support non-type parameters or overloading the std::hash function.

cdef extern from "_containers.h" namespace "spp" nogil:
	# Original, fully generic version.
	# Only usable with types that already have a hash and equals function
	# (e.g., numeric, string).
	cdef cppclass sparse_hash_map[K, D]:
		sparse_hash_map()
		sparse_hash_map(uint64_t n)
		void swap(sparse_hash_map&)
		K& key_type
		D& data_type
		pair[K, D]& value_type
		uint64_t size_type
		bint empty()
		uint64_t size()
		cppclass iterator:
			pair[K, D]& operator*() nogil
			iterator operator++() nogil
			iterator operator--() nogil
			bint operator==(iterator) nogil
			bint operator!=(iterator) nogil
		iterator begin()
		iterator end()
		iterator find(K& k)
		D& operator[](K&) nogil
		pair[iterator, bint] insert(pair[K, D]) nogil
		uint64_t erase(K& k)
		uint64_t bucket(K& key)
		uint64_t max_size()
		uint64_t count(K& k)
		uint64_t bucket_count()
		uint64_t bucket_size(uint64_t i)
		void rehash(uint64_t n)
		void resize(uint64_t n)
		void reserve(size_t n)
		void clear_deleted_key()
		void erase(iterator pos)
		void erase(iterator first, iterator last)
		void clear()
		void clear_no_resize()

	cdef cppclass sparse_hash_set[V]:
		ctypedef V value_type
		sparse_hash_set()
		sparse_hash_set(uint64_t n)
		void swap(sparse_hash_set&)
		bint empty()
		uint64_t size()
		cppclass iterator:
			V& operator*() nogil
			iterator operator++() nogil
			iterator operator--() nogil
			bint operator==(iterator) nogil
			bint operator!=(iterator) nogil
		iterator begin()
		iterator end()
		iterator find(V& k)
		V& operator[](V&) nogil
		pair[iterator, bint] insert(V& v) nogil
		uint64_t count(V& k)
		uint64_t erase(V& k)
		uint64_t bucket(V& key)
		uint64_t max_size()
		uint64_t bucket_count()
		uint64_t bucket_size(uint64_t i)
		void rehash(uint64_t n)
		void resize(uint64_t n)
		void reserve(size_t n)
		void clear_deleted_key()
		void erase(iterator pos)
		void erase(iterator first, iterator last)
		void clear()
		void clear_no_resize()


cdef extern from "../cpp-btree/btree_set.h" namespace "btree" nogil:
	cdef cppclass btree_set[T]:
		ctypedef T value_type
		cppclass iterator:
			T& operator*()
			iterator operator++()
			iterator operator--()
			bint operator==(iterator)
			bint operator!=(iterator)
		btree_set() except +
		btree_set(btree_set&) except +
		btree_set(iterator, iterator) except +
		bint operator==(btree_set&, btree_set&)
		bint operator!=(btree_set&, btree_set&)
		bint operator<(btree_set&, btree_set&)
		bint operator>(btree_set&, btree_set&)
		bint operator<=(btree_set&, btree_set&)
		bint operator>=(btree_set&, btree_set&)
		void swap(btree_set&)
		void clear()
		size_t size()
		size_t count(const T&)
		size_t max_size()
		bint empty()
		iterator begin()
		iterator end()
		iterator find(T&)
		iterator lower_bound(T&)
		iterator upper_bound(const T&)
		pair[iterator, iterator] equal_range(const T&)
		iterator insert(iterator, const T&) except +
		pair[iterator, bint] insert(const T&) except +
		void erase(iterator)
		void erase(iterator, iterator)
		size_t erase(T&)


cdef extern from "_containers.h" nogil:
	cdef cppclass ProbRule:
		Prob prob
		Label lhs
		Label rhs1
		Label rhs2
		uint32_t args
		uint32_t lengths
		uint32_t no
	# NB: variant of ProbRule without probability, rule number (no).
	cdef cppclass Rule:
		Label lhs
		Label rhs1
		Label rhs2
		uint32_t args
		uint32_t lengths
	cdef cppclass LexicalRule:
		Prob prob
		Label lhs
		string word
	cdef union Position:
		short mid
		uint64_t lvec
		size_t lidx
	cdef cppclass Edge:
		ProbRule *rule
		Position pos
	cdef cppclass SmallChartItem:
		Label label
		uint64_t vec
		SmallChartItem()
		SmallChartItem(Label _label, uint64_t _vec)
		bint operator==(SmallChartItem&)
		bint operator>(SmallChartItem&)
		bint operator<(SmallChartItem&)
	cdef cppclass FatChartItem:
		Label label
		uint64_t vec[SLOTS]
		FatChartItem()
		FatChartItem(Label _label)
		bint operator==(FatChartItem&)
		bint operator<(FatChartItem&)
		bint operator>(FatChartItem&)
	cdef cppclass RankedEdge:
		Edge edge
		ItemNo head
		int left, right
		RankedEdge()
		RankedEdge(ItemNo _head, Edge _edge, int _left, int _right)
		bint operator==(RankedEdge&)
		bint operator<(RankedEdge&)
		bint operator>(RankedEdge&)

	# Wrappers for data structures with specific, user-defined key types
	# because Cython does not support non-type parameters
	# or overloading of std::hash.
	cdef cppclass Agenda[K, V]:
		ctypedef pair[K, V] item_type
		ctypedef pair[item_type, uint32_t] entry_type
		Agenda()
		void reserve(size_t n)
		void replace_entries(vector[item_type] entries)
		void kbest_entries(vector[item_type] entries, size_t k)
		size_t size()
		bint empty()
		bint member(K k)
		void setitem(K k, V v)
		void setifbetter(K k, V v)
		item_type pop()
	cdef cppclass SmallChartItemAgenda[V]:
		ctypedef pair[SmallChartItem, V] item_type
		ctypedef pair[item_type, uint32_t] entry_type
		SmallChartItemAgenda()
		void reserve(size_t n)
		void replace_entries(vector[item_type] entries)
		void kbest_entries(vector[item_type] entries, size_t k)
		size_t size()
		bint empty()
		bint member(SmallChartItem k)
		void setitem(SmallChartItem k, V v)
		void setifbetter(SmallChartItem k, V v)
		item_type pop()
	cdef cppclass RankedEdgeAgenda[V]:
		ctypedef pair[RankedEdge, V] item_type
		ctypedef pair[item_type, uint32_t] entry_type
		RankedEdgeAgenda()
		void reserve(size_t n)
		void replace_entries(vector[item_type] entries)
		void kbest_entries(vector[item_type] entries, size_t k)
		size_t size()
		bint empty()
		bint member(RankedEdge k)
		void setitem(RankedEdge k, V v)
		void setifbetter(RankedEdge, V v)
		item_type pop()

	cdef cppclass SmallChartItemSet:
		ctypedef SmallChartItem value_type
		SmallChartItemSet()
		SmallChartItemSet(uint64_t n)
		void swap(SmallChartItemSet&)
		bint empty()
		uint64_t size()
		cppclass iterator:
			SmallChartItem& operator*() nogil
			iterator operator++() nogil
			iterator operator--() nogil
			bint operator==(iterator) nogil
			bint operator!=(iterator) nogil
		iterator begin()
		iterator end()
		iterator find(SmallChartItem& k)
		SmallChartItem& operator[](SmallChartItem&) nogil
		pair[iterator, bint] insert(SmallChartItem& v) nogil
		uint64_t count(SmallChartItem& k)
		uint64_t erase(SmallChartItem& k)
		uint64_t bucket(SmallChartItem& key)
		uint64_t max_size()
		uint64_t bucket_count()
		uint64_t bucket_size(uint64_t i)
		void rehash(uint64_t n)
		void resize(uint64_t n)
		void reserve(size_t n)
		void clear_deleted_key()
		void erase(iterator pos)
		void erase(iterator first, iterator last)
		void clear()
		void clear_no_resize()
	cdef cppclass FatChartItemSet:
		ctypedef FatChartItem value_type
		FatChartItemSet()
		FatChartItemSet(uint64_t n)
		void swap(FatChartItemSet&)
		bint empty()
		uint64_t size()
		cppclass iterator:
			FatChartItem& operator*() nogil
			iterator operator++() nogil
			iterator operator--() nogil
			bint operator==(iterator) nogil
			bint operator!=(iterator) nogil
		iterator begin()
		iterator end()
		iterator find(FatChartItem& k)
		FatChartItem& operator[](FatChartItem&) nogil
		pair[iterator, bint] insert(FatChartItem& v) nogil
		uint64_t count(FatChartItem& k)
		uint64_t erase(FatChartItem& k)
		uint64_t bucket(FatChartItem& key)
		uint64_t max_size()
		uint64_t bucket_count()
		uint64_t bucket_size(uint64_t i)
		void rehash(uint64_t n)
		void resize(uint64_t n)
		void reserve(size_t n)
		void clear_deleted_key()
		void erase(iterator pos)
		void erase(iterator first, iterator last)
		void clear()
		void clear_no_resize()
	cdef cppclass RankedEdgeSet:
		ctypedef RankedEdge value_type
		RankedEdgeSet()
		RankedEdgeSet(uint64_t n)
		void swap(RankedEdgeSet&)
		bint empty()
		uint64_t size()
		cppclass iterator:
			RankedEdge& operator*() nogil
			iterator operator++() nogil
			iterator operator--() nogil
			bint operator==(iterator) nogil
			bint operator!=(iterator) nogil
		iterator begin()
		iterator end()
		iterator find(RankedEdge& k)
		RankedEdge& operator[](RankedEdge&) nogil
		pair[iterator, bint] insert(RankedEdge& v) nogil
		uint64_t count(RankedEdge& k)
		uint64_t erase(RankedEdge& k)
		uint64_t bucket(RankedEdge& key)
		uint64_t max_size()
		uint64_t bucket_count()
		uint64_t bucket_size(uint64_t i)
		void rehash(uint64_t n)
		void resize(uint64_t n)
		void reserve(size_t n)
		void clear_deleted_key()
		void erase(iterator pos)
		void erase(iterator first, iterator last)
		void clear()
		void clear_no_resize()
	cdef cppclass RuleHashMap[D]:
		RuleHashMap()
		RuleHashMap(uint64_t n)
		void swap(RuleHashMap&)
		Rule& key_type
		D& data_type
		pair[Rule, D]& value_type
		uint64_t size_type
		bint empty()
		uint64_t size()
		cppclass iterator:
			pair[Rule, D]& operator*() nogil
			iterator operator++() nogil
			iterator operator--() nogil
			bint operator==(iterator) nogil
			bint operator!=(iterator) nogil
		iterator begin()
		iterator end()
		iterator find(Rule& k)
		D& operator[](Rule&) nogil
		pair[iterator, bint] insert(pair[Rule, D]) nogil
		uint64_t erase(Rule& k)
		uint64_t bucket(Rule& key)
		uint64_t max_size()
		uint64_t count(Rule& k)
		uint64_t bucket_count()
		uint64_t bucket_size(uint64_t i)
		void rehash(uint64_t n)
		void resize(uint64_t n)
		void reserve(size_t n)
		void clear_deleted_key()
		void erase(iterator pos)
		void erase(iterator first, iterator last)
		void clear()
		void clear_no_resize()

	cdef cppclass SmallChartItemBtreeMap[V]:
		cppclass iterator:
			pair[SmallChartItem, V]& operator*()
			iterator operator++()
			iterator operator--()
			bint operator==(iterator)
			bint operator!=(iterator)
		cppclass const_iterator:
			pair[const SmallChartItem, V]& operator*()
			const_iterator operator++()
			const_iterator operator--()
			bint operator==(const_iterator)
			bint operator!=(const_iterator)
		cppclass reverse_iterator:
			pair[SmallChartItem, V]& operator*()
			iterator operator++()
			iterator operator--()
			bint operator==(reverse_iterator)
			bint operator!=(reverse_iterator)
		cppclass const_reverse_iterator(reverse_iterator):
			pass
		SmallChartItemBtreeMap() except +
		SmallChartItemBtreeMap(SmallChartItemBtreeMap&) except +
		V& operator[](SmallChartItem&)
		bint operator==(SmallChartItemBtreeMap&, SmallChartItemBtreeMap&)
		bint operator!=(SmallChartItemBtreeMap&, SmallChartItemBtreeMap&)
		bint operator<(SmallChartItemBtreeMap&, SmallChartItemBtreeMap&)
		bint operator>(SmallChartItemBtreeMap&, SmallChartItemBtreeMap&)
		bint operator<=(SmallChartItemBtreeMap&, SmallChartItemBtreeMap&)
		bint operator>=(SmallChartItemBtreeMap&, SmallChartItemBtreeMap&)
		V& at(const SmallChartItem&) except +
		iterator begin()
		const_iterator const_begin "begin" ()
		void clear()
		size_t count(const SmallChartItem&)
		bint empty()
		iterator end()
		const_iterator const_end "end" ()
		pair[iterator, iterator] equal_range(const SmallChartItem&)
		void erase(iterator)
		void erase(iterator, iterator)
		size_t erase(const SmallChartItem&)
		iterator find(const SmallChartItem&)
		const_iterator const_find "find" (const SmallChartItem&)
		pair[iterator, bint] insert(pair[SmallChartItem, V]) except +
		iterator insert(iterator, pair[SmallChartItem, V]) except +
		iterator lower_bound(const SmallChartItem&)
		const_iterator const_lower_bound "lower_bound"(const SmallChartItem&)
		size_t max_size()
		reverse_iterator rbegin()
		const_reverse_iterator const_rbegin "rbegin"()
		reverse_iterator rend()
		const_reverse_iterator const_rend "rend"()
		size_t size()
		void swap(SmallChartItemBtreeMap&)
		iterator upper_bound(const SmallChartItem&)
		const_iterator const_upper_bound "upper_bound"(const SmallChartItem&)
	cdef cppclass FatChartItemBtreeMap[V]:
		cppclass iterator:
			pair[FatChartItem, V]& operator*()
			iterator operator++()
			iterator operator--()
			bint operator==(iterator)
			bint operator!=(iterator)
		cppclass const_iterator:
			pair[const FatChartItem, V]& operator*()
			const_iterator operator++()
			const_iterator operator--()
			bint operator==(const_iterator)
			bint operator!=(const_iterator)
		cppclass reverse_iterator:
			pair[FatChartItem, V]& operator*()
			iterator operator++()
			iterator operator--()
			bint operator==(reverse_iterator)
			bint operator!=(reverse_iterator)
		cppclass const_reverse_iterator(reverse_iterator):
			pass
		FatChartItemBtreeMap() except +
		FatChartItemBtreeMap(FatChartItemBtreeMap&) except +
		V& operator[](FatChartItem&)
		bint operator==(FatChartItemBtreeMap&, FatChartItemBtreeMap&)
		bint operator!=(FatChartItemBtreeMap&, FatChartItemBtreeMap&)
		bint operator<(FatChartItemBtreeMap&, FatChartItemBtreeMap&)
		bint operator>(FatChartItemBtreeMap&, FatChartItemBtreeMap&)
		bint operator<=(FatChartItemBtreeMap&, FatChartItemBtreeMap&)
		bint operator>=(FatChartItemBtreeMap&, FatChartItemBtreeMap&)
		V& at(const FatChartItem&) except +
		iterator begin()
		const_iterator const_begin "begin" ()
		void clear()
		size_t count(const FatChartItem&)
		bint empty()
		iterator end()
		const_iterator const_end "end" ()
		pair[iterator, iterator] equal_range(const FatChartItem&)
		void erase(iterator)
		void erase(iterator, iterator)
		size_t erase(const FatChartItem&)
		iterator find(const FatChartItem&)
		const_iterator const_find "find" (const FatChartItem&)
		pair[iterator, bint] insert(pair[FatChartItem, V]) except +
		iterator insert(iterator, pair[FatChartItem, V]) except +
		iterator lower_bound(const FatChartItem&)
		const_iterator const_lower_bound "lower_bound"(const FatChartItem&)
		size_t max_size()
		reverse_iterator rbegin()
		const_reverse_iterator const_rbegin "rbegin"()
		reverse_iterator rend()
		const_reverse_iterator const_rend "rend"()
		size_t size()
		void swap(FatChartItemBtreeMap&)
		iterator upper_bound(const FatChartItem&)
		const_iterator const_upper_bound "upper_bound"(const FatChartItem&)


@cython.final
cdef class StringList(object):
	cdef vector[string] ob


@cython.final
cdef class StringIntDict(object):
	cdef sparse_hash_map[string, Label] ob


@cython.final
cdef class Grammar:
	cdef ProbRule **bylhs
	cdef ProbRule **unary
	cdef ProbRule **lbinary
	cdef ProbRule **rbinary
	cdef vector[LexicalRule] lexical
	cdef sparse_hash_map[string, vector[uint32_t]] lexicalbyword
	cdef sparse_hash_map[Label, sparse_hash_map[string, uint32_t]] lexicalbylhs
	cdef Label *mapping
	cdef Label *selfmapping
	cdef Label **splitmapping
	cdef uint32_t *revrulemap
	cdef uint64_t *mask
	cdef vector[vector[Label]] revmap
	cdef vector[uint8_t] fanout
	cdef vector[ProbRule] buf
	cdef RuleHashMap[uint32_t] rulenos
	cdef readonly size_t nonterminals, phrasalnonterminals
	cdef readonly size_t numrules, numunary, numbinary, maxfanout
	cdef readonly bint logprob, bitpar
	cdef readonly str start
	cdef readonly str rulesfile, lexiconfile, altweightsfile
	cdef readonly object ruletuples
	cdef StringList tolabel
	cdef StringIntDict toid
	cdef readonly list rulemapping, selfrulemapping
	cdef readonly dict tblabelmapping
	cdef readonly str currentmodel
	cdef vector[Prob] defaultmodel
	cdef readonly object models  # serialized numpy arrays
	cdef _indexrules(self, ProbRule **dest, int idx, int filterlen)
	cpdef rulestr(self, int n)
	cpdef noderuleno(self, node)
	cpdef getruleno(self, tuple r, tuple yf)
	cdef yfstr(self, ProbRule rule)


# chart improvements todo:
# [ ] is it useful to have a recognition phase before making parse forest?
# [ ] symbolic parsing; separate viterbi stage
# [ ] can we exploit bottom-up order of parser or previous CTF stages
# 		to pack the parse forest?
cdef class Chart:
	cdef vector[Prob] probs
	cdef vector[Prob] inside
	cdef vector[Prob] outside
	cdef vector[vector[Edge]] parseforest
	cdef vector[vector[pair[RankedEdge, Prob]]] rankededges
	# cdef vector[string] derivations  # corresponds to rankededges[chart.root()]
	# list of (str, float); corresponds to rankededges[chart.root()]:
	cdef readonly list derivations
	cdef Grammar grammar
	cdef readonly list sent
	cdef short lensent
	cdef Label start
	cdef readonly bint logprob  # False: 0 < p <= 1; True: 0 <= -log(p) < inf
	cdef readonly bint viterbi  # False: inside probs; True: viterbi 1-best
	cdef int lexidx(self, Edge edge) except -1
	cdef Prob subtreeprob(self, ItemNo itemidx)
	cdef Prob lexprob(self, ItemNo itemidx, Edge edge) except -1
	cdef edgestr(self, ItemNo itemidx, Edge edge)
	cdef ItemNo _left(self, ItemNo itemidx, Edge edge)
	cdef ItemNo _right(self, ItemNo itemidx, Edge edge)
	cdef ItemNo left(self, RankedEdge edge)
	cdef ItemNo right(self, RankedEdge edge)
	cdef Label label(self, ItemNo itemidx)
	cdef ItemNo getitemidx(self, uint64_t idx)
	cdef SmallChartItem asSmallChartItem(self, ItemNo itemidx)
	cdef FatChartItem asFatChartItem(self, ItemNo itemidx)
	cdef size_t asCFGspan(self, ItemNo itemidx)


@cython.final
cdef class Whitelist:
	cdef vector[sparse_hash_set[Label]] cfg  # span -> set of labels
	# cdef vector[vector[Label]] cfg  # span -> sorted array of fine labels
	# cdef vector[btree_set[Label]] cfg  # span -> set of fine labels
	cdef vector[SmallChartItemSet] small  # label -> set of items
	cdef vector[FatChartItemSet] fat   # label -> set of items
	cdef Label *mapping  # maps of labels to ones in this whitelist
	cdef Label **splitmapping


cdef SmallChartItem CFGtoSmallChartItem(Label label, Idx start, Idx end)
cdef FatChartItem CFGtoFatChartItem(Label label, Idx start, Idx end)


# start scratch
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
# 	cdef Prob *insideprobs	# no. of edge -> inside prob
# 	cdef uint32_t *ruleno
# 	# positive means index to lists above, negative means terminal index
# 	cdef int32_t *child
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

cdef packed struct Node:  # a node of a binary tree
	int32_t prod # >= 0: production ID; -1: unseen production
	int16_t left  # >= 0: array idx to child Node; <0: idx sent[-left - 1];
	int16_t right  # >=0: array idx to child Node; -1: empty (unary Node)


cdef packed struct NodeArray:  # a tree as an array of Node structs
	uint32_t offset  # index to Node array where this tree starts
	int16_t len, root  # number of nodes, index to root node


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
	cdef DArray prodbuf  # single string with all productions concatented
	cdef DArray labelbuf  # single string with all labels/words concatented
	cdef DArray labelidx  # label id => offset in labelbuf
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


# defined here because circular import.
cdef inline size_t cellidx(short start, short end, short lensent,
		Label nonterminals):
	"""Return an index for a regular three dimensional array.

	``chart[start][end][0] => chart[idx]`` """
	return (start * lensent + (end - 1)) * nonterminals


cdef inline size_t compactcellidx(short start, short end, short lensent,
		Label nonterminals):
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


cdef inline double logprobsum(vector[Prob]& logprobs):
	"""Sum a list of log probabilities producing a regular probability.

	>>> a = b = c = 0.25
	>>> logprobsum([log(a), log(b), log(c)]) == sum([a, b, c]) == 0.75
	True

	:param logprobs: log probilities, s.t. -inf < p <= 0 for each p in
		``logprobs``.
	:returns: a probability p with 0 < p <= 1.0

	source: http://blog.smola.org/post/987977550/ and
	http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
	(streaming version)
	Comparison of different methods: https://gist.github.com/andreasvc/6204982
	"""
	# maxprob = max(logprobs)
	# return exp(maxprob) * fsum([exp(prob - maxprob) for prob in logprobs])
	cdef double maxprob = -800
	cdef double result = 0
	for prob in logprobs:
		if prob <= maxprob:
			result += exp(prob - maxprob)
		else:
			result *= exp(maxprob - prob)
			result += 1.0
			maxprob = prob
	return exp(maxprob) * result


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
