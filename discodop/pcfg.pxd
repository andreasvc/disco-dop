cimport cython
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from cpython.dict cimport PyDict_Contains, PyDict_GetItem
from cpython.float cimport PyFloat_AS_DOUBLE
from .containers cimport (Chart, Grammar, ProbRule, LexicalRule,
		Edge, RankedEdge, Idx, Prob, Label, ItemNo, cellidx, compactcellidx,
		compactcell_to_start, compactcell_to_end,
		sparse_hash_map, sparse_hash_set, Agenda, Whitelist,
		SmallChartItem, FatChartItem, CFGtoSmallChartItem, CFGtoFatChartItem)

cdef extern from "<cmath>" namespace "std" nogil:
	bint isfinite(double v)
	bint isinf(double v)

cdef extern from "macros.h":
	uint64_t TESTBIT(uint64_t a[], int b)


# Records the minimum and maximum mid points for (left/right index, label)
cdef struct MidFilter:
	vector[short] minleft, maxleft, minright, maxright


ctypedef fused CFGChart_fused:
	DenseCFGChart
	SparseCFGChart


# Representation:
# Dense: encode items directly as uint64_t array indices;
# 	no difference between items and itemidx.
#   n'th item given by chart.getitemidx(n)
# Sparse: encode items as struct that fits in uint64_t;
# 	itemidx maps to items.
#   n'th item given by itemidx == n
# in both cases, an item can be constructed from a cell (an item with label 0)
# by adding the label to it.
cdef packed struct SparseCFGItem:
		Label label
		Idx start
		Idx end


cdef union CFGItem:
	uint64_t dt
	SparseCFGItem st


cdef class CFGChart(Chart):
	cdef vector[uint64_t] items
	cdef vector[Prob] beambuckets
	cdef ItemNo getitemidx(self, uint64_t idx)


@cython.final
cdef class DenseCFGChart(CFGChart):
	cdef void addedge(self, uint64_t item, Idx mid, ProbRule *rule)
	cdef bint updateprob(self, uint64_t item, Prob prob, Prob beam)
	cdef Label _label(self, uint64_t item)
	cdef Prob _subtreeprob(self, uint64_t item)
	cdef bint _hasitem(self, uint64_t item)


@cython.final
cdef class SparseCFGChart(CFGChart):
	cdef sparse_hash_map[uint64_t, ItemNo] itemindex
	cdef void addedge(self, uint64_t item, Idx mid, ProbRule *rule)
	cdef bint updateprob(self, uint64_t item, Prob prob, Prob beam)
	cdef Label _label(self, uint64_t item)
	cdef Prob _subtreeprob(self, uint64_t item)
	cdef bint _hasitem(self, uint64_t item)


# @cython.final
# cdef class SparseBucketCFGChart(CFGChart):
# 	# cell => label => idx in cell
# 	cdef vector[sparse_hash_map[Label, uint32_t]] itemindex
# 	cdef void addedge(
# 			self, uint64_t cell, Label lhs, Idx mid, ProbRule *rule)
# 	cdef bint updateprob(
# 			self, uint64_t cell, Label lhs, Prob prob, Prob beam)
# 	cdef Label _label(self, uint64_t cell, Label lhs)
# 	cdef Prob _subtreeprob(self, uint64_t cell, Label lhs)
# 	cdef bint _hasitem(self, uint64_t cell, Label lhs)
