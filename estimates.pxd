# cython: boundscheck: False
# cython: wraparound: False
# cython: nonecheck: False
# cython: profile: False

cimport cython
from containers cimport ChartItem, Edge, Rule, Terminal
from plcfrs_cython cimport new_Edge, new_ChartItem
from cpq cimport heapdict, Entry
from array cimport array
cimport numpy as np

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
	foo=cython.int,
	bar=cython.int,
	right=cython.int,
	lr=cython.int,
	gaps=cython.int)
cdef double getoutside(np.ndarray[np.double_t, ndim=4] outside, int maxlen, int slen, int label, unsigned long vec)

@cython.locals(
	I=ChartItem,
	e=Edge,
	nil=ChartItem,
	entry=Entry)
cpdef doinside(grammar, int maxlen, concat, dict insidescores)

@cython.locals(
	n=cython.uint,
	m=cython.uint,
	d2=dict,
	val=np.double_t)
cdef void twodim_dict_to_array(dict d, np.ndarray[np.double_t, ndim=2] a)

@cython.locals(
#	a=cython.int,
#	b=cython.int
	npinsidescores=np.ndarray)
cpdef np.ndarray outsidelr(grammar, dict insidescores, int maxlen, int goal)

@cython.locals(
	score=np.double_t,
	entry=Entry,
	newitem=Item,
	nil=ChartItem,
	I=Item,
	e=Edge,
	bylhs=list,
	rules=list,
	infinity=cython.double,
	x=cython.double,
	y=cython.double,
	insidescore=cython.double,
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
	arg=cython.ushort
	)
cdef void computeoutsidelr(grammar, np.ndarray[np.double_t, ndim=2] insidescores, int maxlen, int goal, np.ndarray[np.double_t, ndim=4] outside)

