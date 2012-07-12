cimport cython
from containers cimport ChartItem, Grammar, Rule, UInt
from plcfrs cimport new_ChartItem
from agenda cimport Agenda, Entry
from array cimport array
cimport numpy as np
from bit cimport nextset, nextunset, bitcount, bitlength, testbit, testbitint,\
	bitminmax

if cython.compiled: print "Yep, I'm compiled"
else: print "interpreted"

cdef extern from "math.h":
	bint isnan(double x)
	bint isfinite(double x)

cdef class Item:
	cdef public long _hash
	cdef public int state, length, lr, gaps
	#def __init__(self, len state, int length, int lr, int gaps)

@cython.locals(item=Item)
cdef Item new_Item(UInt state, UInt length, UInt lr, UInt gaps)

@cython.locals(
	length=cython.uint,
	left=cython.uint,
	right=cython.uint,
	lr=cython.uint,
	gaps=cython.uint)
cdef double getoutside(np.ndarray[np.double_t, ndim=4] outside, UInt maxlen, UInt slen, UInt label, unsigned long long vec)

@cython.locals(
	I=ChartItem,
	nil=ChartItem,
	i=size_t,
	entry=Entry)
cpdef dict inside(Grammar grammar, UInt maxlen, dict insidescores)

cdef insideconcat(a, b, Rule rule, maxlen)

@cython.locals(
	I=ChartItem,
	rule=Rule,
	nil=ChartItem,
	entry=Entry,
	agenda=Agenda,
	infinity=np.double_t,
	lbinary=list,
	rbinary=list,
	unary=list,
	rules=list,
	i=size_t,
	x=np.double_t,
	vec=cython.ulong)
cpdef simpleinside(Grammar grammar, UInt maxlen, np.ndarray[np.double_t, ndim=2] insidescores)

@cython.locals(
	current=np.double_t,
	score=np.double_t,
	entry=Entry,
	newitem=Item,
	nil=ChartItem,
	I=Item,
	bylhs=list,
	rules=list,
	rule=Rule,
	infinity=cython.double,
	x=cython.double,
	y=cython.double,
	insidescore=cython.double,
	i=size_t,
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
cdef void outsidelr(Grammar grammar, np.ndarray[np.double_t, ndim=2] insidescores, UInt maxlen, UInt goal, np.ndarray[np.double_t, ndim=4] outside)

cpdef testestimates(Grammar grammar, maxlen, goal)
cpdef getestimates(Grammar grammar, maxlen, goal)
