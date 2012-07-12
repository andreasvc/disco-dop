from cpython cimport PyDict_Contains, PyDict_GetItem
cimport numpy as np
from agenda cimport Entry, EdgeAgenda
from kbest cimport lazykbest, lazykthbest
from containers cimport ChartItem, Edge, Grammar, Rule, LexicalRule,\
	UChar, UInt, ULong, ULLong
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
