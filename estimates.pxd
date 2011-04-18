from plcfrs_cython import ChartItem
import cython

if cython.compiled:
	print "Yep, I'm compiled"
else:
	print "interpreted"

cdef extern from "bit.h":
	int nextset(unsigned long vec, int pos)
	int nextunset(unsigned long vec, int pos)
	int bitcount(unsigned long vec)
	bint testbit(unsigned long vec, unsigned long pos)
	bint bitminmax(unsigned long a, unsigned long b)

cdef class Item:
	cdef int state, len, lr, gaps
	cdef long _hash
	#def __init__(self, str state, int len, int lr, int gaps)

@cython.locals(
	len=cython.int,
	left=cython.int,
	right=cython.int,
	lr=cython.int,
	gaps=cython.int)
cpdef double getoutside(list outside, int maxlen, int slen, int label, unsigned long vec)

@cython.locals(
	newitem=Item,
	I=Item,
	infinity=cython.double,
	x=cython.double,
	y=cython.double,
	insidescore=cython.double,
	score=cython.double,
	a=cython.int,
	b=cython.int,
	c=cython.int,
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
	#yieldfunction=cython.tuple
	)
cpdef list outsidelr(grammar, insidescores, int maxlen, int goal)
