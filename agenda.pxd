from containers cimport Edge
from cpython cimport PyList_Append as append,\
					PyList_GET_ITEM as list_getitem,\
					PyList_GET_SIZE as list_getsize,\
					PyDict_Contains as dict_contains,\
					PyDict_GetItem as dict_getitem

cdef class Entry:
	cdef public object key
	cdef public object value
	cdef unsigned long count

cdef class Function:
	cdef bint cmpfun(self, Entry a, Entry b)
cdef class EdgeCmp(Function):
	cdef bint cmpfun(self, Entry a, Entry b)
cdef class NormalCmp(Function):
	cdef bint cmpfun(self, Entry a, Entry b)

cdef class Agenda(dict):
	cpdef Entry popentry(self)
	cpdef setitem(self, key, object value)
	cpdef setifbetter(self, key, object value)
	cpdef list getheap(self)
	cpdef object getkey(self, Entry entry)
	cpdef object getval(self, Entry entry)
	cdef object getitem(self, key)
	cdef object replace(self, object key, object value)
	cdef bint contains(self, key)
	cdef public unsigned long length
	cdef public list heap
	cdef dict mapping
	cdef unsigned long counter
	cdef Function cmpfun

cdef class EdgeAgenda(Agenda):
	cpdef Entry popentry(self)
	cpdef setitem(self, key, object value)
	cpdef setifbetter(self, key, object value)
	cpdef list getheap(self)
	cpdef object getval(self, Entry entry)
	cdef object getitem(self, key)
	cdef bint contains(self, key)
	cdef object replace(self, object key, object value)

cdef inline list nsmallest(int n, list items)
