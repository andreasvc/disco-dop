from containers cimport Edge, ChartItem, LCFRSEdge
from cpython cimport PyList_GET_ITEM, PyList_GET_SIZE
cimport cython

@cython.final
cdef class Entry:
	cdef object key
	cdef object value
	cdef unsigned long count
ctypedef bint (*CmpFun)(Entry a, Entry b)

@cython.final
cdef class Agenda:
	cdef setitem(self, key, object value)
	cdef setifbetter(self, key, object value)
	cdef object getitem(self, key)
	cdef object replace(self, object key, object value)
	cdef Entry popentry(self)
	cdef Entry peekentry(self)
	cdef bint contains(self, key)
	cdef unsigned long length, counter
	cdef list heap
	cdef dict mapping
	cdef CmpFun cmpfun

@cython.final
cdef class EdgeAgenda:
	cdef setitem(self, key, object value)
	cdef setifbetter(self, key, object value)
	cdef LCFRSEdge getitem(self, key)
	cdef LCFRSEdge replace(self, object key, object value)
	cdef Entry popentry(self)
	cdef Entry peekentry(self)
	cdef bint contains(self, key)
	cdef unsigned long length, counter
	cdef list heap
	cdef dict mapping
	cdef CmpFun cmpfun

cdef list nsmallest(int n, list items)
