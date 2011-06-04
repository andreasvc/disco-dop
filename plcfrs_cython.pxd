from cpq cimport Entry
from kbest cimport lazykbest
from containers cimport ChartItem, Edge, Rule, Terminal
from array cimport array
from cpq cimport heapdict
#from cqueue cimport Queue, QueueValue, queue_new, queue_free, \
#				queue_push_tail, queue_pop_head, queue_is_empty

cdef extern from "bit.h":
	int nextset(unsigned long vec, int pos)
	int nextunset(unsigned long vec, int pos)
	int bitcount(unsigned long vec)
	bint testbit(unsigned long vec, unsigned long pos)
	bint testbitc(unsigned char arg, unsigned long pos)
	bint bitminmax(unsigned long a, unsigned long b)
