from containers cimport Edge, ChartItem
from cpython cimport PyList_GET_ITEM as list_getitem,\
					PyList_GET_SIZE as list_getsize,\
					PyDict_GetItem as dict_getitem

ctypedef bint (*CmpFun)(Entry a, Entry b)
cdef class Entry:
	cdef object key
	cdef object value
	cdef unsigned long count

cdef class Agenda(dict):
	cpdef Entry popentry(self)
	cpdef Entry peekentry(self)
	cpdef setitem(self, key, object value)
	cpdef setifbetter(self, key, object value)
	cpdef list getheap(self)
	cpdef object getkey(self, Entry entry)
	cpdef object getval(self, Entry entry)
	cdef object getitem(self, key)
	cdef object replace(self, object key, object value)
	cdef bint contains(self, key)
	cdef unsigned long length
	cdef list heap
	cdef dict mapping
	cdef unsigned long counter
	cdef CmpFun cmpfun

cdef class EdgeAgenda(Agenda):
	cpdef Entry popentry(self)
	cpdef Entry peekentry(self)
	cpdef setitem(self, key, object value)
	cpdef setifbetter(self, key, object value)
	cpdef list getheap(self)
	cpdef object getval(self, Entry entry)
	cdef object getitem(self, key)
	cdef bint contains(self, key)
	cdef object replace(self, object key, object value)

cdef inline list nsmallest(int n, list items)
