cimport cython
from containers cimport new_DTree, DTree

#cdef DTree make_DTree(tree, sent)

@cython.locals(
	a=dict,
	b=dict,
	x=tuple,
	asent=list,
	bsent=list,
	m=cython.int,
	n=cython.int,
	lentrees=cython.int,
	mem=dict
	)
cpdef extractfragments(list trees, list sents)

@cython.locals(
	l=set,
	i=tuple,
	j=tuple,
	x=frozenset,
	y=frozenset)
cdef set extractfrom(dict a, dict b, list asent, list bsent)

@cython.locals(
	n=cython.int,
	ii=tuple,
	jj=tuple)
cdef set extractmaxfragments(dict a, dict b, tuple i, tuple j, list asent, list bsent, dict mem)
