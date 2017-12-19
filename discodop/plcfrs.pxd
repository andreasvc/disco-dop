cimport cython
from libc.string cimport memcmp, memset, memcpy
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from cpython.set cimport PySet_Contains
from .containers cimport Chart, Grammar, Prob, Label, ItemNo, \
		ProbRule, LexicalRule, SmallChartItem, FatChartItem, Edge, \
		Whitelist, Agenda, SmallChartItemBtreeMap, FatChartItemBtreeMap, \
		BITSIZE, CFGtoSmallChartItem, CFGtoFatChartItem, compactcellidx
from .bit cimport nextset, nextunset, bitcount, bitlength, \
	testbit, anextset, anextunset, abitcount, abitlength, setunion

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
	cdef void addlexedge(self, ItemNo itemidx, short wordidx)
	cdef void updateprob(self, ItemNo itemidx, Prob prob)
	cdef void addprob(self, ItemNo itemidx, Prob prob)
	cdef Prob _subtreeprob(self, ItemNo itemidx)


@cython.final
cdef class SmallLCFRSChart(LCFRSChart):
	cdef vector[SmallChartItem] items
	cdef SmallChartItemBtreeMap[ItemNo] itemindex
	cdef SmallChartItemBtreeMap[Prob] beambuckets
	cdef SmallChartItem _root(self)
	cdef Label _label(self, ItemNo itemidx)
	cdef void addedge(self, ItemNo itemidx, ItemNo leftitemidx,
			SmallChartItem& left, ProbRule *rule)


@cython.final
cdef class FatLCFRSChart(LCFRSChart):
	cdef vector[FatChartItem] items
	cdef FatChartItemBtreeMap[ItemNo] itemindex
	cdef FatChartItemBtreeMap[Prob] beambuckets
	cdef FatChartItem _root(self)
	cdef Label _label(self, ItemNo itemidx)
	cdef void addedge(self, ItemNo itemidx, ItemNo leftitemidx,
			FatChartItem& left, ProbRule *rule)
