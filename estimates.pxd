# cython: boundscheck: False
# cython: nonecheck: False
# cython: profile: False

import cython
from containers cimport ChartItem, Edge, Rule, Terminal
from plcfrs_cython cimport new_Edge, new_ChartItem
from cpq cimport heapdict, Entry

if cython.compiled:
	print "Yep, I'm compiled"
else:
	print "interpreted"

cdef extern from "bit.h":
	int nextset(unsigned long vec, int pos)
	int nextunset(unsigned long vec, int pos)
	int bitcount(unsigned long vec)
	bint bitminmax(unsigned long a, unsigned long b)
	bint testbit(unsigned long vec, unsigned int pos)
	bint testbitc(unsigned char c, unsigned int pos)
	bint testbitshort(unsigned short c, unsigned int pos)

cdef class Item:
	cdef int state, length, lr, gaps
	cdef long _hash
	#def __init__(self, len state, int length, int lr, int gaps)

@cython.locals(item=Item)
cdef Item new_Item(int state, int length, int lr, int gaps)

@cython.locals(
	length=cython.int,
	left=cython.int,
	right=cython.int,
	lr=cython.int,
	gaps=cython.int)
cpdef double getoutside(list outside, int maxlen, int slen, int label, unsigned long vec)

@cython.locals(
	I=ChartItem,
	e=Edge,
	nil=ChartItem,
	entry=Entry)
cpdef doinside(grammar, int maxlen, concat, insidescores)

@cython.locals(
	entry=Entry,
	newitem=Item,
	nil=ChartItem,
	I=Item,
	e=Edge,
	infinity=cython.double,
	x=cython.double,
	y=cython.double,
	insidescore=cython.double,
	score=cython.double,
	m=cython.int,
	n=cython.int,
	a=cython.int,
	b=cython.int,
	c=cython.int,
	fanout=cython.int,
	addgaps=cython.int,
	addright=cython.int,
	addleft=cython.int,
	leftarity=cython.int,
	rightarity=cython.int,
	lenA=cython.int,
	lenB=cython.int,
	lr=cython.int,
	ga=cython.int,
	totlen=cython.int,
	rstate=cython.int,
	lstate=cython.int,
	stopaddright=cython.bint,
	stopaddleft=cython.bint,
	rule=Rule,
	arg=cython.ushort,
	#yieldfunction=cython.tuple
	)
cpdef list outsidelr(grammar, insidescores, int maxlen, int goal)


