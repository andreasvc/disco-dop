from libc.stdlib cimport malloc, free
from libc.string cimport memset
from containers cimport ULong, UInt, UChar
from containers cimport Node, NodeArray, Ctrees, FrozenArray, new_FrozenArray
from array cimport new_array, array
from bit cimport anextset, abitcount, subset, setunion, ulongcpy, ulongset


cdef extern from "bit.h":
	int BITSIZE
	int BITMASK(int b)
	int BITSLOT(int b)
	void SETBIT(ULong a[], int b)
	void CLEARBIT(ULong a[], int b)
	ULong TESTBIT(ULong a[], int b)
	int BITNSLOTS(int nb)
	int IDX(int i, int j, int jmax, int kmax)
#	int abitcount(ULong *vec, int slots)
#	int anextset(ULong *vec, int pos, UInt slots)
#	int subset(ULong *vec1, ULong *vec2, UInt slots)
#	void setunion(ULong *dest, ULong *src, UInt slots)
#	void ulongcpy(ULong *dest, ULong *src, UInt slots)
#	void ulongset(ULong *dest, ULong value, UInt slots)

