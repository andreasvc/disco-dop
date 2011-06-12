from containers cimport Edge

cdef class Entry:
	cdef object key
	cdef Edge value
	cdef unsigned long count

cdef class heapdict(dict):
	cdef list heap
	cdef dict mapping
	cdef unsigned long counter
	cdef unsigned long length
	cpdef tuple popitem(heapdict self)
	cpdef inline Entry popentry(heapdict self)
	cpdef inline Edge replace(heapdict self, object key, Edge value)
	cdef inline Edge getitem(heapdict self, key)
	cdef inline void setitem(heapdict self, key, Edge value)
	cdef inline void setifbetter(heapdict self, key, Edge value)
	cdef inline bint contains(heapdict self, key)

cdef inline list nsmallest(int n, list items)
