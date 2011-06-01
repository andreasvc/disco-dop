cdef extern from "bit.h":
	int nextset(unsigned long vec, int pos)
	int nextunset(unsigned long vec, int pos)
	int bitcount(unsigned long vec)
	bint testbit(unsigned long vec, unsigned long pos)
	bint bitminmax(unsigned long a, unsigned long b)

cdef class ChartItem:
	cdef public int label
	cdef public unsigned long vec
	cdef long _hash
