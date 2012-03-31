from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from bit cimport pyintnextset
from containers cimport ULong, Node, NodeArray, Ctrees

cdef extern from "bit.h":
	int BITSIZE
	int BITMASK(int b)
	int BITSLOT(int b)
	int SETBIT(ULong a[], int b)
	int CLEARBIT(ULong a[], int b)
	int TESTBIT(ULong a[], int b)
	int BITNSLOTS(int nb)
	int IDX(int i, int j, int jmax, int kmax)
	int abitcount(ULong vec[], int slots)
	int anextset(ULong vec[], int pos, int slots)
	int subset(ULong vec1[], ULong vec2[], int slots)
