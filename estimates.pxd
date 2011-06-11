# cython: boundscheck: False
# cython: wraparound: False
# cython: nonecheck: False
# cython: profile: False

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
	bint testbitshort(unsigned short c, unsigned int pos)
	int nextset(unsigned long vec, unsigned int pos)
	int nextunset(unsigned long vec, unsigned int pos)
	int bitcount(unsigned long vec)
	int bitlength(unsigned long vec)

cdef class Item:
	cdef public unsigned int state, length, lr, gaps
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

# can we specify signature of concat?
@cython.locals(
	I=ChartItem,
	e=Edge,
	nil=ChartItem,
	entry=Entry)
cpdef doinside(grammar, unsigned int maxlen, concat, dict insidescores)

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
cpdef np.ndarray outsidelr(grammar, dict insidescores, unsigned int maxlen, unsigned int goal)

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
	arity=array,
	infinity=cython.double,
	x=cython.double,
	y=cython.double,
	insidescore=cython.double,
	m=cython.uint,
	n=cython.uint,
	addgaps=cython.uint,
	addright=cython.uint,
	addleft=cython.uint,
	leftarity=cython.int,
	rightarity=cython.int,
	lenA=cython.uint,
	lenB=cython.uint,
	lr=cython.uint,
	ga=cython.uint,
	totlen=cython.uint,
	stopaddright=cython.bint,
	stopaddleft=cython.bint
	)
cdef void computeoutsidelr(grammar, np.ndarray[np.double_t, ndim=2] insidescores, unsigned int maxlen, unsigned int goal, np.ndarray[np.double_t, ndim=4] outside) except *

