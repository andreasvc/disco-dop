from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from containers cimport ULong, UChar
from containers cimport Node, NodeArray, Ctrees, FrozenArray, new_FrozenArray
from array cimport new_array, array

cdef extern from "bit.h":
	int BITSIZE
	int BITMASK(int b)
	int BITSLOT(int b)
	void SETBIT(ULong a[], int b)
	void CLEARBIT(ULong a[], int b)
	ULong TESTBIT(ULong a[], int b)
	int BITNSLOTS(int nb)
	int IDX(int i, int j, int jmax, int kmax)
	int abitcount(ULong vec[], int slots)
	int anextset(ULong vec[], int pos, int slots)
	int subset(ULong vec1[], ULong vec2[], int slots)
	void setunion(ULong vec1[], ULong vec2[], int slots)
