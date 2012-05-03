from libc.stdlib cimport malloc, free
from libc.string cimport memset
from containers cimport ULong, UInt, UChar
from containers cimport Node, NodeArray, Ctrees, FrozenArray, new_FrozenArray
from array cimport array, clone
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
