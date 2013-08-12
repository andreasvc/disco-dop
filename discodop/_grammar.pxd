from libc.stdlib cimport malloc, calloc, free, qsort
cimport cython
from containers cimport ULLong, ULong, UInt, UChar, Rule, LexicalRule

cdef extern from "macros.h":
	int BITNSLOTS(int nb)
	void SETBIT(ULong a[], int b)


@cython.final
cdef class Grammar:
	cdef Rule **unary, **lbinary, **rbinary, **bylhs
	cdef ULong *chainvec
	cdef UInt *mapping, **splitmapping
	cdef UChar *fanout
	cdef readonly currentmodel
	cdef readonly size_t nonterminals, numrules, numunary, numbinary, maxfanout
	cdef readonly bint logprob, bitpar
	cdef readonly object models
	cdef readonly bytes origrules, start
	cdef readonly unicode origlexicon
	cdef readonly list tolabel, lexical, modelnames, rulemapping
	cdef readonly dict toid, lexicalbyword, lexicalbylhs, lexicalbynum, rulenos
	cdef _convertrules(Grammar self, list rulelines, dict fanoutdict)
	cdef _indexrules(Grammar self, Rule **dest, int idx, int filterlen)
	cdef rulestr(self, Rule rule)
	cdef yfstr(self, Rule rule)
