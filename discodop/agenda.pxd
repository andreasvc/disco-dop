from discodop.containers cimport Edge, ChartItem, LCFRSEdge
from cpython cimport PyList_GET_ITEM, PyList_GET_SIZE
cimport cython

@cython.final
cdef class Entry:
	cdef object key
	cdef object value
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
	cdef list heap
	cdef dict mapping
	cdef setitem(self, key, object value)
	cdef setifbetter(self, key, object value)
	cdef object getitem(self, key)
	cdef object replace(self, object key, object value)
	cdef Entry popentry(self)
	cdef Entry peekentry(self)
	cdef bint contains(self, key)

@cython.final
cdef class EdgeAgenda:
	cdef unsigned long length, counter
	cdef list heap
	cdef dict mapping
	cdef setitem(self, key, object value)
	cdef setifbetter(self, key, object value)
	cdef LCFRSEdge getitem(self, key)
	cdef LCFRSEdge replace(self, object key, object value)
	cdef Entry popentry(self)
	cdef Entry peekentry(self)
	cdef bint contains(self, key)

cdef list nsmallest(int n, object iterable)
