from containers cimport ULLong, ULong, UInt, UChar

cdef extern:
	int __builtin_ffsll (ULLong)
	int __builtin_ctzll (ULLong)
	int __builtin_clzll (ULLong)
	int __builtin_ctzl (ULong)
	int __builtin_popcountl (ULong)
	int __builtin_popcountll (ULLong)

cdef extern from "macros.h":
	int BITSIZE
	int BITSLOT(int b)
	ULong TESTBIT(ULong a[], int b)

#on python integers
cpdef inline int fanout(arg)
cpdef inline int pyintnextset(a, int pos)

#on ULLongs
cpdef inline int nextset(ULLong vec, UInt pos)
cpdef inline int nextunset(ULLong vec, UInt pos)
cpdef inline int bitcount(ULLong vec)
cpdef inline int bitlength(ULLong vec)
cpdef inline bint bitminmax(ULLong a, ULLong b)
cpdef inline bint testbit(ULLong vec, UInt pos)
cpdef inline bint testbitc(UChar arg, UInt pos)
cpdef inline bint testbitshort(unsigned short arg, UInt pos)
cpdef inline bint testbitint(UInt arg, UInt pos)

#on arrays of unsigned long
cdef inline int abitcount(ULong *vec, UInt slots)
cdef inline int anextset(ULong *vec, UInt pos, UInt slots)
cdef inline int anextunset(ULong *vec, UInt pos, UInt slots)
cdef inline bint subset(ULong *vec1, ULong *vec2, UInt slots)
cdef inline void setunion(ULong *dest, ULong *src, UInt slots)
cdef inline void ulongset(ULong *dest, ULong value, UInt slots)
cdef inline void ulongcpy(ULong *dest, ULong *src, UInt slots)
