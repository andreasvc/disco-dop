from libc.stdlib cimport malloc, free

cdef extern from "bit.h":
	int BITMASK(int b)
	int BITSLOT(int b)
	int BITSET(long a[], int b)
	int BITTEST(long a[], int b)
	int BITNSLOTS(int nb)
	int abitcount(long vec[], int slots)
	int anextset(long vec[], int pos, int slots)
	int subset(long vec1[], long vec2[], int slots)
	void setunion(long vec1[], long vec2[], int slots)

cdef struct node:
	int label, prod
	short left, right

cdef struct treetype:
	int len
	node *nodes

DEF MAXNODE = 256
DEF BITSIZE = 64 #(8*sizeof(long))
DEF slots = 4 #BITNSLOTS(MAXNODE)
