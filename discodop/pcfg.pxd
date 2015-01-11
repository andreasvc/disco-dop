cimport cython
from libc.stdlib cimport malloc, calloc, free
from libc.math cimport isinf, isfinite
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from cpython.dict cimport PyDict_Contains, PyDict_GetItem
from cpython.float cimport PyFloat_AS_DOUBLE
from discodop.plcfrs cimport DoubleAgenda, new_DoubleEntry
from discodop.containers cimport Chart, Grammar, Rule, LexicalRule, \
		Edge, Edges, RankedEdge, Idx, cellidx, compactcellidx

cdef extern from "macros.h":
	uint64_t TESTBIT(uint64_t a[], int b)

ctypedef fused CFGChart_fused:
	DenseCFGChart
	SparseCFGChart


cdef class CFGChart(Chart):
	pass


@cython.final
cdef class DenseCFGChart(CFGChart):
	cdef readonly list parseforest  # chartitem => [Edge(lvec, rule), ...]
	cdef double *probs
	cdef void addedge(self, uint32_t lhs, Idx start, Idx end, Idx mid,
			Rule *rule)
	cdef bint updateprob(self, uint32_t lhs, Idx start, Idx end, double prob,
			double beam)
	cdef double _subtreeprob(self, size_t item)
	cdef bint hasitem(self, size_t item)


@cython.final
cdef class SparseCFGChart(CFGChart):
	cdef readonly dict parseforest  # chartitem => [Edge(lvec, rule), ...]
	cdef dict probs
	cdef void addedge(self, uint32_t lhs, Idx start, Idx end, Idx mid,
			Rule *rule)
	cdef bint updateprob(self, uint32_t lhs, Idx start, Idx end, double prob,
			double beam)
	cdef double _subtreeprob(self, size_t item)
	cdef bint hasitem(self, size_t item)
