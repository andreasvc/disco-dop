cimport cython
from libc.string cimport memcmp
from cpython.list cimport PyList_GET_ITEM, PyList_GET_SIZE
from cpython.dict cimport PyDict_Contains
#from cpython.float cimport PyFloat_AS_DOUBLE
from discodop.containers cimport Chart, Grammar, Rule, LexicalRule, \
		ChartItem, SmallChartItem, FatChartItem, new_SmallChartItem, \
		new_FatChartItem, Edge, Edges, Chart, CFGtoFatChartItem, \
		UChar, UInt, ULong, ULLong
from discodop.bit cimport nextset, nextunset, bitcount, bitlength, \
	testbit, anextset, anextunset, abitcount, abitlength, \
	ulongset, ulongcpy, setunion

cdef extern from "Python.h":
	double PyFloat_AS_DOUBLE(object pyfloat)

cdef extern from "macros.h":
	int BITSLOT(int b)
	ULong BITMASK(int b)
	int BITNSLOTS(int nb)
	void SETBIT(ULong a[], int b)
	ULong TESTBIT(ULong a[], int b)

ctypedef fused LCFRSChart_fused:
	SmallLCFRSChart
	FatLCFRSChart

ctypedef fused LCFRSItem_fused:
	SmallChartItem
	FatChartItem



cdef class LCFRSChart(Chart):
	cdef readonly dict parseforest # chartitem => [Edge(lvec, rule), ...]
	cdef list probs
	cdef void addlexedge(self, item, short wordidx)
	cdef void updateprob(self, ChartItem item, double prob)
	cdef void addprob(self, ChartItem item, double prob)
	cdef double _subtreeprob(self, ChartItem item)


@cython.final
cdef class SmallLCFRSChart(LCFRSChart):
	cdef SmallChartItem tmpleft, tmpright
	cdef void addedge(self, SmallChartItem item, SmallChartItem left,
			Rule *rule)


@cython.final
cdef class FatLCFRSChart(LCFRSChart):
	cdef FatChartItem tmpleft, tmpright
	cdef void addedge(self, FatChartItem item, FatChartItem left, Rule *rule)


# FIXME: Entry is (mis)used by kbest; put in containers?
@cython.final
cdef class Entry:
	cdef object key, value
	cdef unsigned long count


ctypedef bint (*CmpFun)(Entry a, Entry b)


cdef inline Entry new_Entry(object k, object v, unsigned long c):
	cdef Entry entry = Entry.__new__(Entry)
	entry.key = k
	entry.value = v
	entry.count = c
	return entry


@cython.final
cdef class Agenda:
	cdef unsigned long length, counter
	cdef readonly list heap
	cdef dict mapping
	cdef void setitem(self, key, object value)
	cdef void setifbetter(self, key, object value)
	cdef object getitem(self, key)
	cdef object replace(self, object key, object value)
	cdef Entry popentry(self)
	cdef Entry peekentry(self)
	cdef bint contains(self, key)


@cython.final
cdef class DoubleAgenda:
	cdef unsigned long length, counter
	cdef list heap
	cdef dict mapping
	cdef inline void setitem(self, key, double value)
	cdef inline void setifbetter(self, key, double value)
	cdef double getitem(self, key)
	cdef double replace(self, key, double value)
	cdef Entry popentry(self)
	cdef Entry peekentry(self)
	cdef bint contains(self, key)

cdef list nsmallest(int n, object iterable, key=*)
