from libc.stdlib cimport malloc, free

cdef extern from "bit.h":
	int BITSIZE
	int BITMASK(int b)
	int BITSLOT(int b)
	int SETBIT(long a[], int b)
	int BITTEST(long a[], int b)
	int BITNSLOTS(int nb)
	int GET3DIDX(int i, int j, int jmax, int kmax)
	int abitcount(long vec[], int slots)
	int anextset(long vec[], int pos, int slots)
	int subset(long vec1[], long vec2[], int slots)

cdef struct node:
	int label, prod
	short left, right

cdef struct treetype:
	int len
	node *nodes

