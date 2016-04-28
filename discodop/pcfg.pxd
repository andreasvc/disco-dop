cimport cython
from libc.stdlib cimport malloc, calloc, free, abort
from libc.math cimport isinf, isfinite
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from cpython.dict cimport PyDict_Contains, PyDict_GetItem
from cpython.float cimport PyFloat_AS_DOUBLE
from .plcfrs cimport DoubleAgenda, new_DoubleEntry
from .containers cimport Chart, Grammar, ProbRule, LexicalRule, \
		Edge, Edges, EdgesStruct, MoreEdges, RankedEdge, Idx, \
		cellidx, compactcellidx, PY2

cdef extern from "macros.h":
	uint64_t TESTBIT(uint64_t a[], int b)

ctypedef fused CFGChart_fused:
	DenseCFGChart
	SparseCFGChart


cdef class CFGChart(Chart):
	pass


@cython.final
cdef class DenseCFGChart(CFGChart):
	cdef EdgesStruct *parseforest  # chartitem => EdgesStruct(...)
	cdef double *probs
	cdef void addedge(self, uint32_t lhs, Idx start, Idx end, Idx mid,
			ProbRule *rule)
	cdef bint updateprob(self, uint32_t lhs, Idx start, Idx end, double prob,
			double beam)
	cdef double _subtreeprob(self, size_t item)
	cpdef bint hasitem(self, item)


@cython.final
cdef class SparseCFGChart(CFGChart):
	cdef readonly dict parseforest  # chartitem => Edges(...)
	cdef dict probs
	cdef void addedge(self, uint32_t lhs, Idx start, Idx end, Idx mid,
			ProbRule *rule)
	cdef bint updateprob(self, uint32_t lhs, Idx start, Idx end, double prob,
			double beam)
	cdef double _subtreeprob(self, size_t item)
	cpdef bint hasitem(self, item)
