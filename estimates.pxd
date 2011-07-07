cimport cython
from containers cimport ChartItem, Edge, Rule, Terminal
from plcfrs_cython cimport new_Edge, new_ChartItem
from agenda cimport heapdict, Entry
from array cimport array
cimport numpy as np

if cython.compiled:
	print "Yep, I'm compiled"
else:
	print "interpreted"

cdef extern from "bit.h":
	bint bitminmax(unsigned long a, unsigned long b)
	bint testbit(unsigned long vec, unsigned int pos)
	bint testbitc(unsigned char c, unsigned int pos)
	bint testbitint(unsigned int c, unsigned int pos)
	bint testbitshort(unsigned short c, unsigned int pos)
	int nextset(unsigned long vec, unsigned int pos)
	int nextunset(unsigned long vec, unsigned int pos)
	int bitcount(unsigned long vec)
	int bitlength(unsigned long vec)

cdef extern from "math.h":
	bint isnan(double x)
	bint isfinite(double x)

cdef class Item:
	cdef public int state, length, lr, gaps
	cdef public long _hash
	#def __init__(self, len state, int length, int lr, int gaps)

@cython.locals(item=Item)
cdef Item new_Item(unsigned int state, unsigned int length, unsigned int lr, unsigned int gaps)

@cython.locals(
	length=cython.uint,
	left=cython.uint,
	right=cython.uint,
	lr=cython.uint,
	gaps=cython.uint)
cdef double getoutside(np.ndarray[np.double_t, ndim=4] outside, unsigned int maxlen, unsigned int slen, unsigned int label, unsigned long vec)

@cython.locals(
	I=ChartItem,
	e=Edge,
	nil=ChartItem,
	entry=Entry)
cpdef dict inside(grammar, unsigned int maxlen, dict insidescores)

@cython.locals(
	I=ChartItem,
	e=Edge,
	rule=Rule,
	nil=ChartItem,
	entry=Entry,
	agenda=heapdict,
	infinity=np.double_t,
	lbinary=list,
	rbinary=list,
	unary=list,
	rules=list,
	x=np.double_t,
	vec=cython.ulong)
cpdef simpleinside(grammar, unsigned int maxlen, np.ndarray[np.double_t, ndim=2] insidescores)

@cython.locals(
	current=np.double_t,
	score=np.double_t,
	entry=Entry,
	newitem=Item,
	nil=ChartItem,
	I=Item,
	e=Edge,
	bylhs=list,
	rules=list,
	rule=Rule,
	#arity=array,
	infinity=cython.double,
	x=cython.double,
	y=cython.double,
	insidescore=cython.double,
	m=cython.int,
	n=cython.int,
	totlen=cython.int,
	addgaps=cython.int,
	addright=cython.int,
	addleft=cython.int,
	leftarity=cython.int,
	rightarity=cython.int,
	lenA=cython.int,
	lenB=cython.int,
	lr=cython.int,
	ga=cython.int,
	stopaddright=cython.bint,
	stopaddleft=cython.bint
	)
cdef void outsidelr(grammar, np.ndarray[np.double_t, ndim=2] insidescores, unsigned int maxlen, unsigned int goal, array[unsigned char] arity, np.ndarray[np.double_t, ndim=4] outside) except *

