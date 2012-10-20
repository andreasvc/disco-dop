from containers cimport Edge, ChartItem, LCFRSEdge
from cpython cimport PyList_GET_ITEM as list_getitem,\
					PyList_GET_SIZE as list_getsize,\
					PyDict_GetItem as dict_getitem

ctypedef bint (*CmpFun)(Entry a, Entry b)
cdef class Entry:
	cdef object key
	cdef object value
	cdef unsigned long count

cdef class Agenda:
	cpdef setitem(self, key, object value)
	cpdef setifbetter(self, key, object value)
	cpdef Entry popentry(self)
	cpdef Entry peekentry(self)
	cdef object getitem(self, key)
	cdef object replace(self, object key, object value)
	cdef bint contains(self, key)
	cdef unsigned long length
	cdef unsigned long counter
	cdef list heap
	cdef dict mapping
	cdef CmpFun cmpfun

cdef class EdgeAgenda(Agenda):
	cpdef setitem(self, key, object value)
	cpdef setifbetter(self, key, object value)
	cpdef Entry popentry(self)
	cpdef Entry peekentry(self)
	cdef bint contains(self, key)
	cdef object getitem(self, key)
	cdef object replace(self, object key, object value)

cdef list nsmallest(int n, list items)
