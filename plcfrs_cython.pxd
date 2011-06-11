cimport numpy as np
from agenda cimport Entry
from kbest cimport lazykbest, lazykthbest
from containers cimport ChartItem, Edge, Rule, Terminal
from array cimport array
from agenda cimport heapdict

cdef extern from "bit.h":
	bint testbit(unsigned long vec, unsigned int pos)
	bint testbitc(unsigned char arg, unsigned int pos)
	bint testbitshort(unsigned short arg, unsigned int pos)
	bint bitminmax(unsigned long a, unsigned long b)
	int nextset(unsigned long vec, unsigned int pos)
	int nextunset(unsigned long vec, unsigned int pos)
	int bitcount(unsigned long vec)
	int bitlength(unsigned long vec)

cdef inline ChartItem new_ChartItem(unsigned int label, unsigned long vec)
cdef inline Edge new_Edge(double score, double inside, double prob, ChartItem left, ChartItem right)

from estimates cimport * #getoutside
