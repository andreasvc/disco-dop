cimport cython
from libc.string cimport memcmp
from libc.stdlib cimport malloc, calloc, free, abort
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from cpython.list cimport PyList_GET_ITEM, PyList_GET_SIZE
from cpython.set cimport PySet_Contains
from cpython.float cimport PyFloat_AS_DOUBLE
from .containers cimport Chart, Grammar, ProbRule, LexicalRule, \
		ChartItem, SmallChartItem, FatChartItem, new_SmallChartItem, \
		new_FatChartItem, Edge, Edges, MoreEdges, Chart, CFGtoFatChartItem
from .bit cimport nextset, nextunset, bitcount, bitlength, \
	testbit, anextset, anextunset, abitcount, abitlength, setunion
from libc.string cimport memset, memcpy

cdef extern from "macros.h":
	int BITSLOT(int b)
	uint64_t BITMASK(int b)
	int BITNSLOTS(int nb)
	void SETBIT(uint64_t a[], int b)
	uint64_t TESTBIT(uint64_t a[], int b)

ctypedef fused LCFRSChart_fused:
	SmallLCFRSChart
	FatLCFRSChart

ctypedef fused LCFRSItem_fused:
	SmallChartItem
	FatChartItem

cdef class LCFRSChart(Chart):
	cdef readonly dict parseforest  # chartitem => Edges(...)
	cdef list probs
	cdef void addlexedge(self, item, short wordidx)
	cdef void updateprob(self, ChartItem item, double prob)
	cdef void addprob(self, ChartItem item, double prob)
	cdef double _subtreeprob(self, ChartItem item)
	cpdef hasitem(self, ChartItem item)


@cython.final
cdef class SmallLCFRSChart(LCFRSChart):
	cdef SmallChartItem tmpitem
	cdef void addedge(self, SmallChartItem item, SmallChartItem left,
			ProbRule *rule)


@cython.final
cdef class FatLCFRSChart(LCFRSChart):
	cdef FatChartItem tmpitem
	cdef void addedge(self, FatChartItem item, FatChartItem left,
			ProbRule *rule)


# FIXME: Entry/Agenda are used in multiple modules; put in containers?
@cython.final
cdef class Entry:
	cdef readonly object key
	cdef readonly object value
	cdef readonly uint64_t count


@cython.final
cdef class DoubleEntry:
	cdef readonly object key
	cdef readonly double value
	cdef readonly uint64_t count


ctypedef fused Entry_fused:
	Entry
	DoubleEntry


cdef inline Entry new_Entry(object k, object v, uint64_t c):
	cdef Entry entry = Entry.__new__(Entry)
	entry.key = k
	entry.value = v
	entry.count = c
	return entry


cdef inline DoubleEntry new_DoubleEntry(object k, double v, uint64_t c):
	cdef DoubleEntry entry = DoubleEntry.__new__(DoubleEntry)
	entry.key = k
	entry.value = v
	entry.count = c
	return entry


cdef class Agenda:
	cdef uint64_t length, counter
	cdef readonly list heap
	cdef dict mapping


@cython.final
cdef class DoubleAgenda(Agenda):
	cdef inline void setitem(self, key, double value)
	cdef inline void setifbetter(self, key, double value)
	cdef double getitem(self, key)
	cdef double replace(self, key, double value)
	cdef DoubleEntry popentry(self)
	cdef DoubleEntry peekentry(self)
	cdef update_entries(self, list entries)

cdef list nsmallest(int n, list entries)
