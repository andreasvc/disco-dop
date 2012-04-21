from cpython cimport PyList_Append as append,\
					PyList_GET_ITEM as list_getitem,\
					PyList_GET_SIZE as list_getsize,\
					PyDict_Contains as dict_contains,\
					PyDict_GetItem as dict_getitem
cimport numpy as np
from agenda cimport Entry
from kbest cimport lazykbest, lazykthbest
from containers cimport ChartItem, Edge, Rule, LexicalRule, UInt, ULLong
from array cimport array
from agenda cimport EdgeAgenda
from bit cimport nextset, nextunset, bitcount, bitlength, testbit, testbitint,\
	bitminmax

cdef extern from "math.h":
	bint isinf(double x)
	bint isfinite(double x)

cdef inline ChartItem new_ChartItem(UInt label, ULLong vec)
cdef inline Edge new_Edge(double score, double inside, double prob,
	ChartItem left, ChartItem right)
