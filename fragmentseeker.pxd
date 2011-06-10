cimport cython

@cython.locals(
	a=dict,
	b=dict,
	i=tuple,
	j=tuple,
	l=set,
	x=tuple,
	x=frozenset,
	y=frozenset,
	m=cython.int,
	n=cython.int,
	mem=dict
	)
cpdef extractfragments(list trees, list sents)


@cython.locals(
	n=cython.int,
	ii=tuple,
	jj=tuple)
cdef set extractmaxfragments(dict a, dict b, tuple i, tuple j, list asent, list bsent, dict mem)
