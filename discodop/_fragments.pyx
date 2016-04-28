"""Fragment extraction with tree kernels.

Implements:

- van Cranenburgh (2014), Extraction of Phrase-Structure Fragments
  with a Linear Average Time Tree-Kernel
- Sangati et al. (2010), Efficiently extract recurring tree fragments from
  large treebanks
- Moschitti (2006): Making Tree Kernels practical for Natural Language Learning
"""

from __future__ import print_function
import re
import io
import os
import sys
from collections import Counter
from functools import partial
from itertools import islice
from array import array
from roaringbitmap import RoaringBitmap
from .tree import Tree, escape
from .util import openread
from .grammar import lcfrsproductions, printrule
from .treetransforms import binarize, handledisc

cimport cython
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memset, memcpy
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from cpython.array cimport array, clone, extend_buffer, resize
from .containers cimport Node, NodeArray, Ctrees, Vocabulary, Rule, \
		yieldranges, termidx, PY2
from .bit cimport iteratesetbits, abitcount, subset, setunioninplace

cdef extern from "macros.h":
	int BITNSLOTS(int nb) nogil
	void SETBIT(uint64_t a[], int b) nogil
	void CLEARBIT(uint64_t a[], int b) nogil
	uint64_t TESTBIT(uint64_t a[], int b) nogil

# a template to create arrays of this type
cdef array ulongarray = array(b'L' if PY2 else 'L')
cdef array uintarray = array(b'I' if PY2 else 'I')
cdef array intarray = array(b'i' if PY2 else 'i')
cdef array shortarray = array(b'h' if PY2 else 'h')

FRONTIERORTERMRE = re.compile(r' ([0-9]+)(?:=|:[0-9]+\b)')  # all leaves
TERMINDICESRE = re.compile(r'\([^(]+ ([0-9]+)=[^ ()]+\)')  # term. indices
FRONTIERRE = re.compile(r' ([0-9]+):([0-9]+)\b')  # non-terminal frontiers
TREEPARSEMSG = [
		'0 nodes found',  # 0
		'unexpected end of string after opening paren',  # 1
		'unexpected end of string after label',  # 2
		'unexpected end of string after closing paren',  # 3
		'unexpected data after end of tree',  # 4
		'unexpected data after terminal',  # 5
		'missing terminal index',  # 6
		'stack not empty after end of string',  # 7
		]


# bitsets representing fragments are uint64_t arrays with the number of
# elements determined by SLOTS. While SLOTS is not a constant nor a global,
# it is set just once for a treebank to fit its largest tree.
# After SLOTS elements, this struct follows:
cdef packed struct BitsetTail:
	uint32_t id  # tree no. of which this fragment was extracted
	short root  # index of the root of the fragment in the NodeArray


# we wrap bitsets in bytes objects, because these can be (1) used in
# dictionaries and (2) passed between processes.
# we leave one byte for NUL-termination:
# data (SLOTS * 8), id (4), root (2), NUL (1)
cdef inline bytes wrap(uint64_t *data, short SLOTS):
	"""Wrap bitset in bytes object for handling in Python space."""
	return (<char *>data)[:SLOTS * sizeof(uint64_t) + sizeof(BitsetTail)]


# use getters & setters because a cdef class would incur overhead & indirection
# of a python object, and with a struct the root & id fields must be in front
# which seems to lead to bad hashing behavior (?)
cdef inline uint64_t *getpointer(object wrapper):
	"""Get pointer to bitset from wrapper."""
	return <uint64_t *><char *><bytes>wrapper


cdef inline uint32_t getid(uint64_t *data, short SLOTS):
	"""Get id of fragment in a bitset."""
	return (<BitsetTail *>&data[SLOTS]).id


cdef inline short getroot(uint64_t *data, short SLOTS):
	"""Get root of fragment in a bitset."""
	return (<BitsetTail *>&data[SLOTS]).root


cdef inline void setrootid(uint64_t *data, short root, uint32_t id,
		short SLOTS):
	"""Set root and id of fragment in a bitset."""
	cdef BitsetTail *tail = <BitsetTail *>&data[SLOTS]
	tail.id = id
	tail.root = root


cpdef extractfragments(Ctrees trees1, int start1, int end1, Vocabulary vocab,
		Ctrees trees2=None, int start2=0, int end2=0,
		bint approx=True, bint debug=False,
		bint disc=False, bint complement=False,
		str twoterms=None, bint adjacent=False):
	"""Find the largest fragments in treebank(s) with the fast tree kernel.

	- scenario 1: recurring fragments in single treebank, use::
		extractfragments(trees1, start1, end1, vocab)
	- scenario 2: common fragments in two treebanks::
		extractfragments(trees1, start1, end1, vocab, trees2)

	:param start1, end1: specify slice of treebank to be used; can be used to
		divide the work over multiple processes; they are indices of ``trees1``
		to work on (pass 0 for both to use all trees).
	:param start2, end2: idem for trees2.
	:param approx: return approximate counts instead of bitsets.
	:param debug: if True, a table of common productions is printed for each
		pair of trees
	:param disc: if True, return trees with indices as leaves.
	:param complement: if True, the complement of the recurring
		fragments in each pair of trees is extracted as well.
	:param twoterms: only return fragments with at least two terminals,
		one of which has a POS tag matching the given regex.
	:param adjacent: only extract fragments from sentences with adjacent
		indices.
	:returns: a dictionary; keys are fragments as strings; values are
		either counts (if approx=True), or bitsets describing fragments of
		``trees1``.
	"""
	cdef:
		int n, m
		short minterms = 2 if twoterms else 0
		short SLOTS  # the number of uint32_ts needed to cover the largest tree
		uint64_t *matrix = NULL  # bit matrix of common productions in tree pair
		uint64_t *scratch
		NodeArray a
		Node *anodes
		list asent
		dict fragments = {}
		set inter = set(), contentwordprods = None, lexicalprods = None
		list tmp = []
	if twoterms:
		contentword = re.compile(twoterms)
		contentwordprods = {n for n in range(len(vocab.prods))
				if contentword.match(vocab.getlabel(n))}
		lexicalprods = {n for n in range(len(vocab.prods))
				if vocab.islexical(n)}
	if trees2 is None:
		trees2 = trees1
	SLOTS = BITNSLOTS(max(trees1.maxnodes, trees2.maxnodes) + 1)
	matrix = <uint64_t *>malloc(trees2.maxnodes * SLOTS * sizeof(uint64_t))
	scratch = <uint64_t *>malloc((SLOTS + 2) * sizeof(uint64_t))
	if matrix is NULL or scratch is NULL:
		raise MemoryError('allocation error')
	end2 = min(end2 or trees2.len, trees2.len)
	# loop over tree pairs to extract fragments from
	for n in range(start1, min(end1 or trees1.len, trees1.len)):
		a = trees1.trees[n]
		asent = trees1.extractsent(n, vocab)
		anodes = &trees1.nodes[a.offset]
		if adjacent:
			m = n + 1
			if m < trees2.len:
				extractfrompair(a, anodes, trees2, n, m, complement, debug,
						vocab, inter, minterms, matrix, scratch, SLOTS)
		elif twoterms:
			for m in twoterminals(a, anodes, trees2,
					contentwordprods, lexicalprods):
				if trees1 is trees2 and m <= n:
					continue
				elif start2 > m or m > end2:
					continue
				elif m < 0 or m >= trees2.len:
					raise ValueError('illegal index %d' % m)
				extractfrompair(a, anodes, trees2, n, m, complement, debug,
						vocab, inter, minterms, matrix, scratch, SLOTS)
		else:  # all pairs
			if trees1 is trees2:
				start2 = max(n + 1, start2)
			for m in range(start2, end2):
				extractfrompair(a, anodes, trees2, n, m,
						complement, debug, vocab, inter, minterms, matrix,
						scratch, SLOTS)
		collectfragments(fragments, inter, anodes, asent, vocab,
				disc, approx, False, tmp, SLOTS)
	free(matrix)
	free(scratch)
	return fragments


cdef inline extractfrompair(NodeArray a, Node *anodes, Ctrees trees2,
		int n, int m, bint complement, bint debug, Vocabulary vocab, set inter,
		short minterms, uint64_t *matrix, uint64_t *scratch, short SLOTS):
	"""Extract the bitsets of maximal overlapping fragments for a tree pair."""
	cdef NodeArray b = trees2.trees[m]
	cdef Node *bnodes = &trees2.nodes[b.offset]
	# initialize table
	memset(<void *>matrix, 0, b.len * SLOTS * sizeof(uint64_t))
	# fill table
	fasttreekernel(anodes, bnodes, a.len, b.len, matrix, SLOTS)
	# dump table
	if debug:
		print(n, m)
		dumpmatrix(matrix, a, b, anodes, bnodes, vocab, scratch, SLOTS)
	# extract results
	extractbitsets(matrix, anodes, bnodes, b.root, n,
			inter, minterms, scratch, SLOTS)
	# extract complementary fragments?
	if complement:
		# combine bitsets of inter together with bitwise or
		memset(<void *>scratch, 0, SLOTS * sizeof(uint64_t))
		for wrapper in inter:
			setunioninplace(scratch, getpointer(wrapper), SLOTS)
		# extract bitsets in A from result, without regard for B
		extractcompbitsets(scratch, anodes, a.root, n, inter,
				SLOTS, NULL)


cdef inline collectfragments(dict fragments, set inter, Node *anodes,
		list asent, Vocabulary vocab, bint disc, bint approx,
		bint indices, list tmp, short SLOTS):
	"""Collect string representations of fragments given as bitsets."""
	cdef uint64_t *bitset
	for wrapper in inter:
		bitset = getpointer(wrapper)
		getsubtree(tmp, anodes, bitset, vocab,
				disc, getroot(bitset, SLOTS))
		try:
			frag = getsent(''.join(tmp)) if disc else ''.join(tmp)
		except:
			print(asent)
			print(tmp)
			raise
		del tmp[:]
		if approx:
			if frag not in fragments:
				fragments[frag] = 0
			fragments[frag] += 1
		elif indices:
			if frag not in fragments:
				fragments[frag] = Counter()
			fragments[frag][getid(bitset, SLOTS)] += 1
		elif frag not in fragments:  # FIXME: is this condition useful?
			fragments[frag] = wrapper
	inter.clear()


cdef inline void fasttreekernel(Node *a, Node *b, int alen, int blen,
		uint64_t *matrix, short SLOTS):
	"""Fast Tree Kernel (average case linear time).

	Expects trees to be sorted according to their productions (in descending
	order, with terminals as -1). This algorithm is from the pseudocode in
	Moschitti (2006): Making Tree Kernels practical for Natural Language
	Learning."""
	# i is an index to a, j to b, and ii is a temp index starting at i.
	cdef int i = 0, j = 0, ii = 0
	while True:
		if a[i].prod < b[j].prod:
			i += 1
			if i >= alen:
				return
		elif a[i].prod > b[j].prod:
			j += 1
			if j >= blen:
				return
		else:
			while a[i].prod == b[j].prod:
				ii = i
				while a[ii].prod == b[j].prod:
					SETBIT(&matrix[j * SLOTS], ii)
					ii += 1
					if ii >= alen:
						break
				j += 1
				if j >= blen:
					return


cdef inline extractbitsets(uint64_t *matrix, Node *a, Node *b, short j, int n,
		set results, int minterms, uint64_t *scratch, short SLOTS):
	"""Visit nodes of ``b`` top-down and store bitsets of common nodes.

	Stores bitsets of connected subsets of ``a`` as they are encountered,
	following the common nodes specified in bit matrix. ``j`` is the node in
	``b`` to start with, which changes with each recursive step. ``n`` is the
	identifier of this tree which is stored with extracted fragments."""
	cdef uint64_t *bitset = &matrix[j * SLOTS]
	cdef uint64_t cur = bitset[0]
	cdef int idx = 0, terms
	cdef int i = iteratesetbits(bitset, SLOTS, &cur, &idx)
	while i != -1:
		memset(<void *>scratch, 0, SLOTS * sizeof(uint64_t))
		terms = extractat(matrix, scratch, a, b, i, j, SLOTS)
		if terms >= minterms:
			setrootid(scratch, i, n, SLOTS)
			results.add(wrap(scratch, SLOTS))
		i = iteratesetbits(bitset, SLOTS, &cur, &idx)
	if b[j].left >= 0:
		extractbitsets(matrix, a, b, b[j].left, n,
				results, minterms, scratch, SLOTS)
		if b[j].right >= 0:
			extractbitsets(matrix, a, b, b[j].right, n,
					results, minterms, scratch, SLOTS)


cdef inline int extractat(uint64_t *matrix, uint64_t *result, Node *a, Node *b,
		short i, short j, short SLOTS):
	"""Traverse tree ``a`` and ``b`` in parallel to extract a connected subset.
	"""
	cdef int terms = 0
	SETBIT(result, i)
	CLEARBIT(&matrix[j * SLOTS], i)
	if a[i].left < 0:
		return 1
	elif TESTBIT(&matrix[b[j].left * SLOTS], a[i].left):
		terms += extractat(matrix, result, a, b, a[i].left, b[j].left, SLOTS)
	if a[i].right < 0:
		return 0
	elif TESTBIT(&matrix[b[j].right * SLOTS], a[i].right):
		terms += extractat(matrix, result, a, b, a[i].right, b[j].right, SLOTS)
	return terms


cpdef exactcounts(Ctrees trees1, Ctrees trees2, list bitsets,
		int indices=False, maxnodes=None):
	"""Get exact counts or indices of occurrence for fragments.

	:param trees1, bitsets: ``bitsets`` defines fragments of trees in
		``trees1`` to search for (the needles).
	:param trees2: the trees to search in (haystack); may be equal
		to ``trees2``.
	:param indices: whether to collect indices or counts of fragments.

		:0: return a single count per fragment.
		:1: collect the indices (sentence numbers) in which fragments occur.
		:2: collect both sentence numbers, and node numbers of fragments.

	:param maxnodes: the maximum number of nodes in a single tree to fix the
		bitset size; use the same value as the function that generated these
		bitsets. For ``extractfragments``, it is the maximum value across both
		treebanks, which is also the default here.
	:returns: depending on ``indices``:

		:0: an array of counts, corresponding to ``bitsets``.
		:1: a list of arrays, each array being a sorted sequence of indices
			for the corresponding bitset; multiple occurrences of a fragment in
			the same tree are reflected as multiple occurrences of the same
			index.
		:2: a list of pairs of arrays, tree indices paired with node numbers.
	"""
	cdef:
		array counts = None
		array treenums = None
		array nodenums = None
		list theindices = None
		object candidates  # RoaringBitmap()
		short i, j, SLOTS
		uint32_t n, m
		uint32_t *countsp = NULL
		NodeArray *a
		NodeArray *b
		Node *anodes
		Node *bnodes
		uint64_t *bitset
	if maxnodes:
		SLOTS = BITNSLOTS(maxnodes + 1)
	else:
		SLOTS = BITNSLOTS(max(trees1.maxnodes, trees2.maxnodes) + 1)
	if indices == 0:
		counts = clone(uintarray, len(bitsets), True)
		countsp = counts.data.as_uints
	elif indices == 1:
		theindices = [clone(uintarray, 0, False) for _ in bitsets]
	elif indices == 2:
		theindices = [(clone(uintarray, 0, False), clone(shortarray, 0, False))
				for _ in bitsets]
	# compare one bitset to each tree for each unique fragment.
	for n, wrapper in enumerate(bitsets):
		bitset = getpointer(wrapper)
		a = &(trees1.trees[getid(bitset, SLOTS)])  # fragment is from this tree
		anodes = &trees1.nodes[a.offset]
		candidates = getcandidates(anodes, bitset, trees2, a.len,
				0, 0, SLOTS)
		if candidates is None:  # ran across unseen production
			continue
		if indices == 1:
			treenums = theindices[n]
		elif indices == 2:
			treenums, nodenums = theindices[n]
		i = getroot(bitset, SLOTS)  # root of fragment in tree 'a'
		for m in candidates:
			b = &(trees2.trees[m])
			bnodes = &trees2.nodes[b.offset]
			for j in range(b.len):
				if anodes[i].prod == bnodes[j].prod:
					if containsbitset(anodes, bnodes, bitset, i, j):
						if indices == 0:
							countsp[n] += 1
						elif indices == 1:
							treenums.append(m)
						elif indices == 2:
							treenums.append(m)
							nodenums.append(j)
				elif anodes[i].prod < bnodes[j].prod:
					break
	return theindices if indices else counts


cpdef exactcountsslice(Ctrees trees1, Ctrees trees2, list bitsets,
		int indices=0, maxnodes=None, start=None, end=None, maxresults=None):
	"""Get counts of fragments in a slice of the treebank.

	Variant of exactcounts() that releases the GIL in the inner loop and is
	intended for searching in subsets of ``trees2``.

	:param start, end: only search through this interval of trees from
		``trees2`` (defaults to all trees).
	:param maxresults: stop searching after this number of matchs.
	:returns: depending on ``indices``:

		:0: an array of counts, corresponding to ``bitsets``.
		:1: a list of arrays, each array being a sequence of indices
			for the corresponding bitset.
		:2: a list of pairs of arrays, tree indices paired with node numbers.
	"""
	cdef:
		array counts = None
		array candidatesarray = None
		array tmp = None, tmp2 = None
		list theindices = None
		object candidates  # RoaringBitmap
		short i, SLOTS
		size_t nummatches = 0, cnt = 0, allocated = 1024
		size_t maxresults_ = maxresults or -1
		uint32_t n, numcandidates
		uint32_t *countsp = NULL
		uint32_t *treenums = NULL
		short *nodenums = NULL
		NodeArray *a
		Node *anodes
		uint64_t *bitset
		int start_ = start or 0, end_ = end or trees2.len
	if maxnodes:
		SLOTS = BITNSLOTS(maxnodes + 1)
	else:
		SLOTS = BITNSLOTS(max(trees1.maxnodes, trees2.maxnodes) + 1)
	if indices:
		treenums = <uint32_t *>malloc(allocated * sizeof(int))
		if treenums is NULL:
			raise MemoryError
		if indices == 1:
			theindices = [clone(uintarray, 0, False) for _ in bitsets]
		elif indices == 2:
			theindices = [
					(clone(uintarray, 0, False), clone(shortarray, 0, False))
					for _ in bitsets]
			nodenums = <short *>malloc(allocated * sizeof(int))
			if nodenums is NULL:
				raise MemoryError
	else:
		counts = clone(uintarray, len(bitsets), True)
		countsp = counts.data.as_uints
	candidatesarray = clone(uintarray, 0, False)
	# compare one bitset to each tree for each unique fragment.
	for n, wrapper in enumerate(bitsets):
		bitset = getpointer(wrapper)
		a = &(trees1.trees[getid(bitset, SLOTS)])  # fragment is from this tree
		anodes = &trees1.nodes[a.offset]
		candidates = getcandidates(anodes, bitset, trees2, a.len,
				start_, end_, SLOTS)
		if candidates is None:  # ran across unseen production
			continue
		candidatesarray.extend(candidates)
		numcandidates = len(candidatesarray)
		if indices == 1:
			tmp = theindices[n]
		elif indices == 2:
			tmp, tmp2 = theindices[n]
		i = getroot(bitset, SLOTS)  # root of fragment in tree 'a'
		with nogil:
			cnt = countbitset(bitset, anodes, i, n,
					candidatesarray.data.as_uints,
					numcandidates, indices, trees2.trees, trees2.nodes,
					maxresults_, &nummatches, &allocated,
					countsp, &treenums, &nodenums)
		if cnt == -1:
			raise MemoryError
		if indices == 1:
			extend_buffer(tmp, <char *>treenums, cnt)
		elif indices == 2:
			extend_buffer(tmp, <char *>treenums, cnt)
			extend_buffer(tmp2, <char *>nodenums, cnt)
		resize(candidatesarray, 0)
	if indices:
		free(treenums)
		if indices == 2:
			free(nodenums)
	return theindices if indices else counts


cdef int countbitset(uint64_t *bitset, Node *anodes, int i, int n,
		uint32_t *candidatesarrayp, int numcandidates, int indices,
		NodeArray *trees, Node *nodes, size_t maxresults, size_t *nummatches,
		size_t *allocated, uint32_t *countsp, uint32_t **treenums,
		short **nodenums) nogil:
	"""Internal function to process a single fragment."""
	cdef int cnt = 0
	cdef NodeArray b
	cdef Node *bnodes
	for x in range(numcandidates):
		m = candidatesarrayp[x]
		b = trees[m]
		bnodes = &nodes[b.offset]
		for j in range(b.len):
			if anodes[i].prod == bnodes[j].prod:
				if containsbitset(anodes, bnodes, bitset, i, j):
					if indices == 0:
						countsp[n] += 1
					elif indices == 1:
						treenums[0][cnt] = m
					elif indices == 2:
						treenums[0][cnt] = m
						nodenums[0][cnt] = j
					nummatches[0] += 1
					cnt += 1
					if nummatches[0] >= maxresults:
						return cnt
					elif cnt >= allocated[0]:
						allocated[0] = cnt * 2
						treenums[0] = <uint32_t *>realloc(
								treenums[0], 2 * cnt * sizeof(int))
						if treenums[0] is NULL:
							return -1
						if indices == 2:
							nodenums[0] = <short *>realloc(
									nodenums[0], 2 * cnt * sizeof(int))
							if nodenums[0] is NULL:
								return -1
			elif anodes[i].prod < bnodes[j].prod:
				break
	return cnt


cdef getcandidates(Node *a, uint64_t *bitset, Ctrees trees, short alen,
		int start, int end, short SLOTS):
	"""Get candidates from productions in fragment ``bitset`` at ``a[i]``."""
	cdef uint64_t cur = bitset[0]
	cdef int i, idx = 0
	cdef list tmp = []
	while True:
		i = iteratesetbits(bitset, SLOTS, &cur, &idx)
		if i == -1 or i >= alen:  # FIXME. why is 2nd condition necessary?
			break
		tmp.append(a[i].prod)
	return trees.prodindex.intersection(
			tmp, start=start, stop=end or trees.len)


cdef inline int containsbitset(Node *a, Node *b, uint64_t *bitset,
		short i, short j) nogil:
	"""Test whether the fragment ``bitset`` at ``a[i]`` occurs at ``b[j]``."""
	if a[i].prod != b[j].prod:
		return 0
	elif a[i].left < 0:
		return 1
	elif TESTBIT(bitset, a[i].left):
		if not containsbitset(a, b, bitset, a[i].left, b[j].left):
			return 0
	if a[i].right < 0:
		return 1
	elif TESTBIT(bitset, a[i].right):
		return containsbitset(a, b, bitset, a[i].right, b[j].right)
	return 1


cpdef completebitsets(Ctrees trees, Vocabulary vocab,
		short maxnodes, bint disc=False, start=None, end=None, tostring=True):
	"""Generate bitsets corresponding to whole trees in the input.

	:param tostring: do not create list of trees as strings
	:returns: a pair of lists with trees as strings and their bitsets,
		respectively.

	A tree with a discontinuous substitution site is expected to be binarized
	with ``rightmostunary=True``:

	>>> from discodop.treebank import brackettree
	>>> tree, sent = brackettree(u'(S (X 0= 2= 4=))')
	>>> print(handledisc(tree))
	(S (X 0 (X|<> 2 (X|<> 4))))

	These auxiliary nodes will not be part of the returned tree / bitset:

	>>> tmp = getctrees([(tree, sent)])
	>>> print(completebitsets(tmp['trees1'], tmp['vocab'], 2, disc=True)[0][0])
	(S (X 0= 2= 4=))
	"""
	cdef:
		list result = [] if tostring else None
		list bitsets = []
		list tmp = []
		int n, i
		short SLOTS = BITNSLOTS(maxnodes + 1)
		uint64_t *scratch = <uint64_t *>malloc((SLOTS + 2) * sizeof(uint64_t))
		Node *nodes
	start, end, unused_step = slice(start, end).indices(trees.len)
	for n in range(<int>start, <int>end):
		memset(<void *>scratch, 0, SLOTS * sizeof(uint64_t))
		nodes = &trees.nodes[trees.trees[n].offset]
		# The set of terminal indices (i.e., minus frontier non-terminals)
		termindices = {termidx(nodes[m].left)
				for m in range(trees.trees[n].len)
				if nodes[m].left < 0 and vocab.islexical(nodes[m].prod)}
		for i in range(trees.trees[n].len):
			if nodes[i].left >= 0 or termidx(nodes[i].left) in termindices:
				SETBIT(scratch, i)
		setrootid(scratch, trees.trees[n].root, n, SLOTS)
		if tostring:
			tmp[:] = []
			getsubtree(tmp, nodes, scratch, vocab,
					disc, trees.trees[n].root)
			tree = getsent(''.join(tmp)) if disc else ''.join(tmp)
			result.append(tree)
		bitsets.append(wrap(scratch, SLOTS))
	free(scratch)
	return result, bitsets


cdef twoterminals(NodeArray a, Node *anodes,
		Ctrees trees2, set contentwordprods, set lexicalprods):
	"""Produce tree pairs that share at least two words.

	Specifically, tree pairs sharing one content word and one additional word,
	where content words are recognized by a POS tag from the Penn treebank tag
	set.

	if trees2 is None, pairs (n, m) are such that n < m."""
	cdef int i, j
	cdef object tmp, candidates = RoaringBitmap()
	# select candidates from 'trees2' that share productions with tree 'a'
	# want to select at least 1 content POS tag, 1 other lexical prod
	for i in range(a.len):
		if anodes[i].left >= 0 or anodes[i].prod not in contentwordprods:
			continue
		tmp = trees2.prodindex[anodes[i].prod]
		for j in range(a.len):
			if (i != j and anodes[j].left < 0 and
					anodes[j].prod in lexicalprods):
				candidates |= tmp & trees2.prodindex[anodes[j].prod]
	return candidates


cdef extractcompbitsets(uint64_t *bitset, Node *a,
		int i, int n, set results, short SLOTS, uint64_t *scratch):
	"""Like ``extractbitsets()`` but following complement of ``bitset``."""
	cdef bint start = scratch is NULL and not TESTBIT(bitset, i)
	if start:  # start extracting a fragment
		# need to create a new array because further down in the recursion
		# other fragments may be encountered which should not overwrite
		# this one
		scratch = <uint64_t *>malloc((SLOTS + 2) * sizeof(uint64_t))
		if scratch is NULL:
			raise MemoryError('allocation error')
		setrootid(scratch, i, n, SLOTS)
	elif TESTBIT(bitset, i):  # stop extracting for this fragment
		scratch = NULL
	if scratch is not NULL:
		SETBIT(scratch, i)
	if a[i].left >= 0:
		extractcompbitsets(bitset, a, a[i].left, n, results, SLOTS, scratch)
		if a[i].right >= 0:
			extractcompbitsets(bitset, a, a[i].right, n, results, SLOTS, scratch)
	if start:  # store fragment
		results.add(wrap(scratch, SLOTS))
		free(scratch)


def allfragments(Ctrees trees, Vocabulary vocab,
		int maxdepth, int maxfrontier=999, bint disc=True,
		bint indices=False):
	"""Return all fragments of trees up to maxdepth.

	:param maxdepth: maximum depth of fragments; depth 1 gives fragments that
		are equivalent to a treebank grammar.
	:param maxfrontier: maximum number of frontier non-terminals (substitution
		sites) in fragments; a limit of 0 only gives fragments that bottom out
		in terminals; 999 is unlimited for practical purposes.
	:returns: dictionary fragments with tree strings as keys and integer counts
		as values."""
	cdef NodeArray tree
	cdef Node *nodes
	cdef int n, SLOTS = BITNSLOTS(trees.maxnodes)
	cdef list table, tmp = []
	cdef set inter = set()
	cdef dict fragments = {}
	cdef uint64_t *scratch = <uint64_t *>malloc((SLOTS + 2) * sizeof(uint64_t))
	if scratch is NULL:
		raise MemoryError('allocation error')
	for n in range(trees.len):
		tree = trees.trees[n]
		sent = trees.extractsent(n, vocab)
		nodes = &trees.nodes[tree.offset]
		# for each node, list of bitsets wrapped in bytes
		table = [[] for _ in range(tree.len)]
		traverse(tree, nodes, tree.root, n, maxdepth, maxfrontier, table,
				scratch, SLOTS)
		# collect subtrees at each node in a single set
		for i, frags in enumerate(table):
			for frag in frags:
				# in the table, the root & id attributes are used to store
				# the depth and frontiers of fragments; replace these
				depth = getroot(getpointer(frag), SLOTS)
				frontiers = getid(getpointer(frag), SLOTS)
				assert depth <= maxdepth
				assert frontiers <= maxfrontier
				memcpy(scratch, getpointer(frag), SLOTS * sizeof(uint64_t))
				setrootid(scratch, i, n, SLOTS)
				inter.add(wrap(scratch, SLOTS))
		collectfragments(fragments, inter, nodes, sent, vocab,
				disc, not indices, indices, tmp, SLOTS)
		del table[:]
	free(scratch)
	return fragments


cdef traverse(NodeArray tree, Node *nodes, int i, int n, int maxdepth,
		int maxfrontier, list table, uint64_t *scratch, int SLOTS):
	"""Collect all fragments of a tree up to maxdepth, maxfrontier."""
	cdef bytes lfrag, rfrag
	cdef int ldepth, rdepth, lfrontiers, rfrontiers
	# First collect fragments of children
	if nodes[i].left >= 0:
		traverse(tree, nodes, nodes[i].left, n, maxdepth, maxfrontier,
				table, scratch, SLOTS)
		if nodes[i].right >= 0:
			traverse(tree, nodes, nodes[i].right, n, maxdepth, maxfrontier,
					table, scratch, SLOTS)
	# Production at current idx is a depth 1 fragment; 0, 1, or 2 frontiers
	if nodes[i].left < 0:
		lfrontiers = 0
	elif nodes[i].right < 0:
		lfrontiers = 1
	else:
		lfrontiers = 2
	if lfrontiers <= maxfrontier:
		memset(<void *>scratch, 0, SLOTS * sizeof(uint64_t))
		SETBIT(scratch, i)
		setrootid(scratch, 1, lfrontiers, SLOTS)
		table[i].append(wrap(scratch, SLOTS))
	# Then combine with fragments of children to form prunes of this subtree
	if nodes[i].left >= 0:
		# unary, or right node as frontier with all lfrags
		for lfrag in table[nodes[i].left]:
			ldepth = getroot(getpointer(lfrag), SLOTS)
			lfrontiers = getid(getpointer(lfrag), SLOTS)
			if (ldepth + 1 <= maxdepth
					and lfrontiers + 1 <= maxfrontier):
				# bitset with current idx + left bitset
				memset(<void *>scratch, 0, SLOTS * sizeof(uint64_t))
				SETBIT(scratch, i)
				setunioninplace(scratch, getpointer(lfrag), SLOTS)
				setrootid(scratch, ldepth + 1, lfrontiers + 1, SLOTS)
				table[i].append(wrap(scratch, SLOTS))
		if nodes[i].right >= 0:  # binary node
			# left node as frontier with all rfrags
			for rfrag in table[nodes[i].right]:
				rdepth = getroot(getpointer(rfrag), SLOTS)
				rfrontiers = getid(getpointer(rfrag), SLOTS)
				if (rdepth + 1 <= maxdepth
						and rfrontiers + 1 <= maxfrontier):
					# bitset with current idx + right bitset
					memset(<void *>scratch, 0, SLOTS * sizeof(uint64_t))
					SETBIT(scratch, i)
					setunioninplace(scratch, getpointer(rfrag), SLOTS)
					setrootid(scratch, rdepth + 1, rfrontiers + 1, SLOTS)
					table[i].append(wrap(scratch, SLOTS))
			# cartesian product of left x right fragments
			for lfrag in table[nodes[i].left]:
				ldepth = getroot(getpointer(lfrag), SLOTS)
				lfrontiers = getid(getpointer(lfrag), SLOTS)
				for rfrag in table[nodes[i].right]:
					rdepth = getroot(getpointer(rfrag), SLOTS)
					rfrontiers = getid(getpointer(rfrag), SLOTS)
					if (max(ldepth, rdepth) + 1 <= maxdepth
							and lfrontiers + rfrontiers <= maxfrontier):
						# bitset with current idx + pairs of bitsets in l, r
						memset(<void *>scratch, 0, SLOTS * sizeof(uint64_t))
						SETBIT(scratch, i)
						setunioninplace(scratch, getpointer(lfrag), SLOTS)
						setunioninplace(scratch, getpointer(rfrag), SLOTS)
						setrootid(scratch, max(ldepth, rdepth) + 1,
								lfrontiers + rfrontiers, SLOTS)
						table[i].append(wrap(scratch, SLOTS))


cdef inline getsubtree(list result, Node *tree, uint64_t *bitset,
		Vocabulary vocab, bint disc, int i):
	"""Get string of tree fragment denoted by bitset; indices as terminals.

	:param result: provide an empty list for the initial call.
	:param disc: pass True to get a tree with indices as leaves
		(discontinuous trees); otherwise the result will be a
		continuous tree with words as leaves."""
	result.append('(')
	result.append(vocab.getlabel(tree[i].prod))
	result.append(' ')
	if TESTBIT(bitset, i):
		if tree[i].left >= 0:
			getsubtree(result, tree, bitset, vocab, disc, tree[i].left)
			if tree[i].right >= 0:
				result.append(' ')
				getsubtree(result, tree, bitset, vocab, disc, tree[i].right)
		elif disc:
			if vocab.islexical(tree[i].prod):
				result.append('%d=%s' % (
						termidx(tree[i].left), vocab.getword(tree[i].prod)))
			else:
				result.append('%d:%d' % (
						termidx(tree[i].left), termidx(tree[i].left)))
		else:
			result.append(vocab.getword(tree[i].prod) or '')
	elif disc:  # node not in bitset, frontier non-terminal
		result.append(yieldranges(sorted(getyield(tree, i))))
	result.append(')')


cdef inline list getyield(Node *tree, int i):
	"""Recursively collect indices of terminals under a node."""
	if tree[i].left < 0:  # terminal / substitution site
		if tree[i].right >= 0:  # substitution site with more indices
			return [termidx(tree[i].left)] + getyield(tree, tree[i].right)
		return [termidx(tree[i].left)]
	elif tree[i].right < 0:  # unary node
		return getyield(tree, tree[i].left)
	# binary node
	return getyield(tree, tree[i].left) + getyield(tree, tree[i].right)


def pygetsent(str frag):
	"""Wrapper of ``getsent()`` to make doctests possible.

	>>> print(pygetsent(u'(S (NP 2=man) (VP 4=walks))'))
	(S (NP 0=man) (VP 2=walks))
	>>> print(pygetsent(u'(VP (VB 0=Wake) (PRT 3=up))'))
	(VP (VB 0=Wake) (PRT 2=up))
	>>> print(pygetsent(u'(S (NP 2:2 4:4) (VP 1:1 3:3))'))
	(S (NP 1= 3=) (VP 0= 2=))
	>>> print(pygetsent(u'(ROOT (S 0:2) ($. 3=.))'))
	(ROOT (S 0=) ($. 1=.))
	>>> print(pygetsent(u'(ROOT (S 0=Foo) ($. 3=.))'))
	(ROOT (S 0=Foo) ($. 2=.))
	>>> print(pygetsent(
	... u'(S|<VP>_2 (VP_3 0:1 3:3 16:16) (VAFIN 2=wird))'))
	(S|<VP>_2 (VP_3 0= 2= 4=) (VAFIN 1=wird))
	"""
	try:
		return getsent(frag)
	except:
		print(frag)
		raise


cdef getsent(str frag):
	"""Renumber indices in fragment and select terminals it contains.

	Returns a transformed copy of fragment and sentence. Replace words that do
	not occur in the fragment with None and renumber terminals in fragment such
	that the first index is 0 and any gaps have a width of 1. Expects a tree as
	string where frontier nodes are marked with intervals."""
	cdef int n, m = 0, maxl
	cdef dict leafmap = {}
	cdef dict spans = {int(start): int(end) + 1
			for start, end in FRONTIERRE.findall(frag)}
	cdef list leaves = [int(a) for a in TERMINDICESRE.findall(frag)]
	for n in leaves:
		spans[n] = n + 1
	maxl = max(spans)
	for n in sorted(spans):
		leafmap[n] = ' %d=' % m
		m += 1
		if spans[n] not in spans and n != maxl:  # a gap
			m += 1
	frag = FRONTIERORTERMRE.sub(repl(leafmap), frag)
	return frag


cdef dumpmatrix(uint64_t *matrix, NodeArray a, NodeArray b, Node *anodes,
		Node *bnodes, Vocabulary vocab,
		uint64_t *scratch, short SLOTS):
	"""Dump a table of the common productions of a tree pair."""
	dumptree(a, anodes, [], vocab, scratch)  # FIXME get sent
	dumptree(b, bnodes, [], vocab, scratch)
	print('\t'.join([''] + ['%2d' % x for x in range(b.len)
		if bnodes[x].prod != -1]))
	for m in range(b.len):
		print('\t', vocab.getlabel((<Node>bnodes[m]).prod)[:3], end='')
	for n in range(a.len):
		print('\n%2d' % n, vocab.getlabel((<Node>anodes[n]).prod)[:3], end='')
		for m in range(b.len):
			print('\t', end='')
			print('1' if TESTBIT(&matrix[m * SLOTS], n) else ' ', end='')
	print('\ncommon productions:', end=' ')
	print(len({anodes[n].prod for n in range(a.len)} &
			{bnodes[n].prod for n in range(b.len)}))
	print('found:', end=' ')
	print('horz', sum([abitcount(&matrix[n * SLOTS], SLOTS) > 0
			for n in range(b.len)]),
		'vert', sum([any([TESTBIT(&matrix[n * SLOTS], m)
			for n in range(b.len)]) for m in range(a.len)]),
		'both', abitcount(matrix, b.len * SLOTS))


cdef dumptree(NodeArray a, Node *anodes, list asent, Vocabulary vocab,
		uint64_t *scratch):
	"""Print debug information of a given tree."""
	for n in range(a.len):
		print('idx=%2d\tleft=%2d\tright=%2d\tprod=%2d\tlabel=%s' % (n,
				termidx(anodes[n].left) if anodes[n].left < 0
				else anodes[n].left,
				anodes[n].right, anodes[n].prod,
				vocab.getlabel(anodes[n].prod)),
				end='')
		if anodes[n].left < 0:
			if vocab.islexical(anodes[n].prod):
				print('\t%s=%s' % ('terminal', vocab.getword(anodes[n].prod)))
			else:
				print('\tfrontier non-terminal')
		else:
			print()
	tmp = []
	memset(<void *>scratch, 255, BITNSLOTS(a.len) * sizeof(uint64_t))
	getsubtree(tmp, anodes, scratch, vocab, True, a.root)
	print(''.join(tmp), '\n', asent, '\n')


def repl(d):
	"""A function for use with re.sub that looks up numeric IDs in a dict.
	"""
	def f(x):
		return d[int(x.group(1))]
	return f


def nonfrontier(sent):
	def nonfrontierfun(x):
		return isinstance(x[0], Tree) or sent[x[0]] is not None
	return nonfrontierfun


cdef int readtree(
		str line, Vocabulary vocab, Node *nodes, array stack,
		array labels, list children) except -9:
	"""Parse tree in bracket format into pre-allocated array of Node structs.

	Tree will be binarized on the fly, equivalent to
	```binarize(tree, dot=True)```. Whitespace is not significant.
	Terminals must have preterminals, without siblings; e.g., ``(X x y)``
	will raise an error.

	:param line: a complete bracketed tree.
	:param vocab: collects productions, labels, and words.
	:parem nodes: primary result.
	:param stack: provide an empty ``array.array('h')``.
	:param labels: provide empty ``array.array('i')``.
	:param children: provide empty list.
	"""
	# :param disc: whether to expect discbracket format.
	cdef short idx = 0, n = 0, m, y, cur, parent = -1
	cdef short lenline = len(line), lensent = 0
	cdef int label, binlabel, binlabel2
	cdef str terminal = None
	cdef bytes prod = None
	cdef list binchildren, binlabels
	cdef Rule rule
	while n < lenline:
		while n < lenline and line[n] in ' \t\n':
			n += 1
		if n >= lenline:
			break
		elif line[n] == '(':
			n += 1
			while n < lenline and line[n] in ' \t\n':
				n += 1
			if n >= lenline:
				return -1  # unexpected end of string after opening paren
			startlabel = n
			while n + 1 < lenline and line[n + 1] not in ' \t\n()':
				n += 1
			if n + 1>= lenline:
				return -2  # unexpected end of string after label
			label = vocab._getlabelid(line[startlabel:n + 1])
			parent = stack.data.as_shorts[len(stack) - 1] if stack else -1
			stack.append(idx)
			labels.append(label)
			nodes[idx].left = nodes[idx].right = nodes[idx].prod = -1
			idx += 1
			if parent >= 0 and nodes[parent].right >= 0:
				children.append([nodes[parent].left, nodes[parent].right,
						stack.data.as_shorts[len(stack) - 1]])
				nodes[parent].right = -2
			elif parent >= 0 and nodes[parent].right == -2:
				children[len(children) - 1].append(
						stack.data.as_shorts[len(stack) - 1])
		elif line[n] == ')':
			prod = terminal = None
			cur = stack.pop()
			if stack:
				parent = stack.data.as_shorts[len(stack) - 1]
				# lookahead for next paren
				while n + 1 < lenline and line[n + 1] in ' \t\n':
					n += 1
				if n + 1 >= lenline:
					return -3  # unexpected end of string after closing paren
				if nodes[cur].prod == -1:  # frontier non-terminal?
					nodes[cur].left = termidx(lensent)
					lensent += 1
					rule.lhs = labels.data.as_ints[cur]
					rule.rhs1 = rule.rhs2 = rule.args = rule.lengths = 0
					prod = (<char *>&rule)[:sizeof(Rule)]
					nodes[cur].prod = vocab._getprodid(prod)
				if nodes[parent].left == -1:
					nodes[parent].left = cur
					if line[n + 1] == ')':  # first and only child, unary node
						rule.lhs = labels.data.as_ints[parent]
						rule.rhs1 = labels.data.as_ints[cur]
						rule.rhs2 = rule.args = 0
						rule.lengths = 1
						prod = (<char *>&rule)[:sizeof(Rule)]
				elif nodes[parent].right == -1:
					nodes[parent].right = cur
					if line[n + 1] == ')':  # 2nd and last child, binary node
						rule.lhs = labels.data.as_ints[parent]
						rule.rhs1 = labels.data.as_ints[parent + 1]
						rule.rhs2 = labels.data.as_ints[cur]
						rule.args = rule.lengths = 0b10
						prod = (<char *>&rule)[:sizeof(Rule)]
				elif line[n + 1] == ')':  # last of > 2 children
					# collect labels
					binchildren = children.pop()
					binlabels = [labels.data.as_ints[m] for m in binchildren]
					binlabel = vocab._getlabelid('%s|<%s.%s>' % (
							vocab.idtolabel(labels.data.as_ints[parent]),
							vocab.idtolabel(binlabels[0]),
							','.join(vocab.idtolabel(x)
								for x in binlabels[1:])))
					nodes[parent].right = idx
					rule.lhs = labels.data.as_ints[parent]
					rule.rhs1 = labels.data.as_ints[parent + 1]
					rule.rhs2 = binlabel
					rule.args = rule.lengths = 0b10
					prod = (<char *>&rule)[:sizeof(Rule)]
					# add intermediate nodes
					for m, y in enumerate(
							binchildren[1:len(binchildren) - 1], 1):
						nodes[idx].left = y
						if m == len(binlabels) - 2:
							nodes[idx].right = binchildren[m + 1]
							binlabel2 = binlabels[len(binlabels) - 1]
						else:
							nodes[idx].right = idx + 1
							binlabel2 = vocab._getlabelid('%s|<%s.%s>' % (
									vocab.idtolabel(
										labels.data.as_ints[parent]),
									','.join(vocab.idtolabel(x)
										for x in binlabels[:m + 1]),
									','.join(vocab.idtolabel(x)
										for x in binlabels[m + 1:])))
						nodes[parent].prod = vocab._getprodid(prod)
						parent = idx
						label, binlabel = binlabel, binlabel2
						rule.lhs, rule.rhs1, rule.rhs2 = (
								label, labels.data.as_ints[y], binlabel)
						rule.args = rule.lengths = 0b10
						prod = (<char *>&rule)[:sizeof(Rule)]
						idx += 1
						labels.append(label)
				# else:  # there is no else.
			else:  # end of stack, should be end of tree
				n += 1
				while n < lenline and line[n] in ' \t\n':
					n += 1
				if n < lenline:
					return -4  # unexpected data after end of tree
		else:  # terminal
			start = n
			while n < lenline and line[n] not in ') \t\n':
				n += 1
			terminal = line[start:n]
			while n < lenline and line[n] in ' \t\n':
				n += 1
			if line[n] != ')':
				return -5  # unexpected data after terminal
			n -= 1
			parent = stack.data.as_shorts[len(stack) - 1]
			# if disc:
			# 	m = terminal.find('=')
			# 	if m <= 0:
			# 		return -6  # missing terminal index
			# 	m, terminal = int(terminal[:m]), terminal[m + 1:]
			# 	nodes[parent].left = termidx(m)
			# else:
			# 	nodes[parent].left = termidx(lensent)
			nodes[parent].left = termidx(lensent)
			lensent += 1
			rule.lhs = labels.data.as_ints[parent]
			rule.rhs1 = rule.rhs2 = rule.lengths = 0
			rule.args = vocab._getlabelid(terminal)
			prod = (<char *>&rule)[:sizeof(Rule)]
		if prod is None:
			pass
		else:
			nodes[parent].prod = vocab._getprodid(prod)
		n += 1

	if len(stack) != 0 or len(children) != 0:
		return -7  # stack / children not empty after end of string
	return idx


cdef inline copynodes(tree, list prodsintree, Node *result, int *idx):
	"""Convert a binarized Tree object to an array of Node structs."""
	cdef size_t n = idx[0]
	if not isinstance(tree, Tree):
		raise ValueError('Expected Tree node, got %s\n%r' % (type(tree), tree))
	elif not 1 <= len(tree) <= 2:
		raise ValueError('trees must be non-empty and binarized\n%s' % tree)
	result[n].prod = prodsintree[n]
	idx[0] += 1
	if isinstance(tree[0], int):  # a terminal index
		result[n].left = -tree[0] - 1
	else:  # non-terminal
		result[n].left = idx[0]
		copynodes(tree[0], prodsintree, result, idx)
	if len(tree) == 1:  # unary node
		result[n].right = -1
	elif isinstance(tree[1], int):  # a terminal index
		raise ValueError('right child can only be non-terminal.')
	else:  # binary node
		result[n].right = idx[0]
		copynodes(tree[1], prodsintree, result, idx)


def getctrees(items1, items2=None, Vocabulary vocab=None, bint index=True):
	"""Convert binarized Tree objects to Ctrees object.

	:param items1: an iterable with tuples of the form ``(tree, sent)``.
	:param items2: optionally, a second iterable of trees.
	:param update: if False, do not update ``vocab``; unseen productions are
		assigned a sentinel that never matches.
	:param index: whether to create production index of trees.
	:returns: dictionary with same keys as arguments, where trees1 and
		trees2 are Ctrees objects for disc. binary trees and sentences."""
	cdef Ctrees ctrees, ctrees1, ctrees2 = None
	cdef Node *scratch
	cdef int cnt
	maxnodes = 512
	scratch = <Node *>malloc(maxnodes * sizeof(Node))
	if scratch is NULL:
		raise MemoryError('allocation error')
	if vocab is None:
		vocab = Vocabulary()
	ctrees = ctrees1 = Ctrees()
	ctrees.alloc(512, 512 * 512)
	for m, items in enumerate((items1, items2)):
		if items is None:
			break
		elif m == 1:
			ctrees = ctrees2 = Ctrees()
			ctrees.alloc(512, 512 * 512)
		for tree, sent in items:
			cnt = 0
			prodsintree = []
			for r, yf in lcfrsproductions(tree, sent, frontiers=True):
				prodsintree.append(vocab.getprod(r, yf))
				cnt += 1
			if cnt > maxnodes:
				maxnodes = cnt
				scratch = <Node *>realloc(scratch, maxnodes * sizeof(Node))
				if scratch is NULL:
					raise MemoryError('allocation error')
			cnt = 0
			copynodes(tree, prodsintree, scratch, &cnt)
			ctrees.addnodes(scratch, cnt, 0)
		if index:
			ctrees.indextrees(vocab)
	free(scratch)
	return dict(trees1=ctrees1, trees2=ctrees2, vocab=vocab)


def readtreebank(treebankfile, Vocabulary vocab,
		fmt='bracket', limit=None, encoding='utf8'):
	"""Read a treebank from a given filename.

	``vocab`` should be re-used when reading multiple treebanks.

	:returns: tuple of Ctrees object and list of sentences."""
	cdef Ctrees ctrees
	cdef Node *scratch
	cdef short maxnodes = 512
	cdef short binfactor = 2  # conservative estimate to accommodate binarization
	cdef int cnt, n
	cdef str line
	cdef array stack, labels
	cdef list children
	if treebankfile is None:
		return None
	ctrees = Ctrees()
	ctrees.alloc(512, 512 * 512)  # dummy values, array will be realloc'd
	scratch = <Node *>malloc(maxnodes * binfactor * sizeof(Node))
	if scratch is NULL:
		raise MemoryError('allocation error')
	if fmt != 'bracket':
		from .treebank import READERS
		corpus = READERS[fmt](treebankfile, encoding=encoding)
		for _, item in corpus.itertrees(0, limit):
			tree = binarize(handledisc(item.tree), dot=True)
			prodsintree = []
			cnt = 0
			for r, yf in lcfrsproductions(tree, item.sent, frontiers=True):
				prodsintree.append(vocab.getprod(r, yf))
				cnt += 1
			if cnt > maxnodes:
				maxnodes = cnt
				scratch = <Node *>realloc(scratch,
						maxnodes * binfactor * sizeof(Node))
				if scratch is NULL:
					raise MemoryError('allocation error')
			cnt = 0
			copynodes(tree, prodsintree, scratch, &cnt)
			ctrees.addnodes(scratch, cnt, 0)
	else:  # do incremental reading of bracket trees
		stack = clone(shortarray, 0, False)
		labels = clone(intarray, 0, False)
		children = []  # list of lists
		data = openread(treebankfile, encoding=encoding)
		for n, line in enumerate(islice(data, limit), 1):
			cnt = line.count('(')
			if cnt > maxnodes:
				maxnodes = 2 * cnt
				scratch = <Node *>realloc(scratch,
						maxnodes * binfactor * sizeof(Node))
				if scratch is NULL:
					raise MemoryError('allocation error')
			cnt = readtree(line, vocab, scratch, stack, labels, children)
			if cnt <= 0:
				raise ValueError('error %d in line %d: %s\n%s' % (
						cnt, n, TREEPARSEMSG[-cnt], line))
			ctrees.addnodes(scratch, cnt, 0)
			del labels[:], children[:]
		if not ctrees.len:
			raise ValueError('%r appears to be empty' % treebankfile)
	free(scratch)
	return ctrees


__all__ = ['extractfragments', 'exactcounts', 'completebitsets',
		'allfragments', 'repl', 'pygetsent', 'nonfrontier', 'getctrees',
		'readtreebank', 'exactcountsslice']
