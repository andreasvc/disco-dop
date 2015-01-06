"""Fragment extraction with tree kernels.

Implements:

- Sangati et al. (2010), Efficiently extract recurring tree fragments from
  large treebanks.
- Moschitti (2006): Making Tree Kernels practical for Natural Language Learning
"""

from __future__ import print_function
import re
import io
import sys
from collections import Counter
from functools import partial
from itertools import islice
from array import array
from discodop.tree import Tree
from discodop.grammar import lcfrsproductions
from discodop.treetransforms import binarize

from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memset, memcpy
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from cpython.array cimport array, clone
from discodop.containers cimport Node, NodeArray, Ctrees, yieldranges
from discodop.bit cimport iteratesetbits, abitcount, subset, setunioninplace

cdef extern from "macros.h":
	int BITNSLOTS(int nb)
	void SETBIT(uint64_t a[], int b)
	void CLEARBIT(uint64_t a[], int b)
	uint64_t TESTBIT(uint64_t a[], int b)
	int IDX(int i, int j, int jmax, int kmax)

# a template to create arrays of this type
cdef array uintarray = array('I' if sys.version[0] >= '3' else b'I', ())

# NB: (?: ) is a non-grouping operator; the second ':' is part of what we match
FRONTIERORTERMRE = re.compile(r' ([0-9]+)(?::[0-9]+)?\b')  # all leaf nodes
TERMINDICESRE = re.compile(r'\([^(]+ ([0-9]+)\)')  # leaf nodes w/term.indices
FRONTIERRE = re.compile(r' ([0-9]+):([0-9]+)\b')  # non-terminal frontiers
LABEL = re.compile(r' *\( *([^ ()]+) *')


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


cpdef extractfragments(Ctrees trees1, list sents1, int offset, int end,
		list labels, Ctrees trees2=None, list sents2=None, bint approx=True,
		bint debug=False, bint discontinuous=False, bint complement=False,
		bint twoterms=False, bint adjacent=False):
	"""Find the largest fragments in treebank(s) with the fast tree kernel.

	- scenario 1: recurring fragments in single treebank, use:
		``extractfragments(trees1, sents1, offset, end, labels)``
	- scenario 2: common fragments in two treebanks:
		``extractfragments(trees1, sents1, offset, end, labels, trees2, sents2)``

	``offset`` and ``end`` can be used to divide the work over multiple
	processes; they are indices of ``trees1`` to work on (default is all);
	when ``debug`` is enabled a table of common productions is shown for each
	pair of trees; when ``complement`` is true, the complement of the recurring
	fragments in each pair of trees is extracted as well."""
	cdef:
		int n, m, start = 0, end2
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
		contentword = re.compile(
				r'^(?:NN(?:[PS]|PS)?|(?:JJ|RB)[RS]?|VB[DGNPZ])$')
		contentwordprods = {n for n, label in enumerate(labels)
				if contentword.match(label)}
		lexical = re.compile(r'^[A-Z]+$')
		lexicalprods = {n for n, label in enumerate(labels)
				if lexical.match(label)}
	if trees2 is None:
		trees2 = trees1
	SLOTS = BITNSLOTS(max(trees1.maxnodes, trees2.maxnodes) + 1)
	matrix = <uint64_t *>malloc(trees2.maxnodes * SLOTS * sizeof(uint64_t))
	scratch = <uint64_t *>malloc((SLOTS + 2) * sizeof(uint64_t))
	if matrix is NULL or scratch is NULL:
		raise MemoryError('allocation error')
	end2 = trees2.len
	# loop over tree pairs to extract fragments from
	for n in range(offset, min(end or trees1.len, trees1.len)):
		a = trees1.trees[n]
		asent = <list>(sents1[n])
		anodes = &trees1.nodes[a.offset]
		if adjacent:
			m = n + 1
			if m < trees2.len:
				extractfrompair(a, anodes, trees2, n, m,
						complement, debug, asent, sents2 or sents1,
						labels, inter, minterms, matrix, scratch, SLOTS)
		elif twoterms:
			for m in twoterminals(a, anodes, trees2,
					contentwordprods, lexicalprods):
				if sents2 is None and m <= n:
					continue
				elif m < 0 or m >= trees2.len:
					raise ValueError('illegal index %d' % m)
				extractfrompair(a, anodes, trees2, n, m,
						complement, debug, asent, sents2 or sents1,
						labels, inter, minterms, matrix, scratch, SLOTS)
		else:  # all pairs
			if sents2 is None:
				start = n + 1
			for m in range(start, end2):
				extractfrompair(a, anodes, trees2, n, m,
						complement, debug, asent, sents2 or sents1,
						labels, inter, minterms, matrix, scratch, SLOTS)
		collectfragments(fragments, inter, anodes, asent, labels,
				discontinuous, approx, False, tmp, SLOTS)
	free(matrix)
	free(scratch)
	return fragments


cdef inline extractfrompair(NodeArray a, Node *anodes, Ctrees trees2,
		int n, int m, bint complement, bint debug, list asent, list sents,
		list labels, set inter, short minterms, uint64_t *matrix,
		uint64_t *scratch, short SLOTS):
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
		dumpmatrix(matrix, a, b, anodes, bnodes, asent, sents[m], labels,
				scratch, SLOTS)
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
		list asent, list labels, bint discontinuous, bint approx,
		bint indices, list tmp, short SLOTS):
	"""Collect string representations of fragments given as bitsets."""
	cdef uint64_t *bitset
	for wrapper in inter:
		bitset = getpointer(wrapper)
		getsubtree(tmp, anodes, bitset, labels,
				<list>(None if discontinuous else asent),
				getroot(bitset, SLOTS))
		frag = getsent(''.join(tmp), asent) if discontinuous else ''.join(tmp)
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
	"""Traverse tree `a` and `b` in parallel to extract a connected subset."""
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
		bint indices=False):
	"""Get exact counts or indices of occurrence for fragments.

	Given a set of fragments from ``trees1`` as bitsets, find all occurrences
	of those fragments in ``trees2`` (which may be equal to ``trees1``).

	By default, exact counts are collected. When ``indices`` is ``True``, a
	multiset of indices is returned for each fragment; the indices point to the
	trees where the fragments (described by the bitsets) occur. This is useful
	to look up the contexts of fragments, or when the order of sentences has
	significance which makes it possible to interpret the set of indices as
	time-series data. The reason we need to do this separately from extracting
	maximal fragments is that a fragment can occur in other trees where it was
	not a maximal."""
	cdef:
		array counts = None
		list theindices = None
		object matches = None  # Counter()
		set candidates
		short i, j, SLOTS = BITNSLOTS(max(trees1.maxnodes, trees2.maxnodes) + 1)
		uint32_t n, m
		uint32_t *countsp = NULL
		NodeArray a, b
		Node *anodes
		Node *bnodes
		uint64_t *bitset
	if indices:
		theindices = [Counter() for _ in bitsets]
	else:
		counts = clone(uintarray, len(bitsets), True)
		countsp = counts.data.as_uints
	# compare one bitset to each tree for each unique fragment.
	for n, wrapper in enumerate(bitsets):
		bitset = getpointer(wrapper)
		a = trees1.trees[getid(bitset, SLOTS)]  # fragment came from this tree
		anodes = &trees1.nodes[a.offset]
		i = getroot(bitset, SLOTS)  # root of fragment in tree 'a'
		try:
			candidates = getcandidates(anodes, bitset, i, trees2)
		except KeyError:
			candidates = set()
		if indices:
			matches = theindices[n]
		for m in candidates:
			b = trees2.trees[m]
			bnodes = &trees2.nodes[b.offset]
			for j in range(b.len):
				if anodes[i].prod == bnodes[j].prod:
					if containsbitset(anodes, bnodes, bitset, i, j):
						if indices:
							matches[m] += 1
						else:
							countsp[n] += 1
				elif anodes[i].prod < bnodes[j].prod:
					break
	return theindices if indices else counts


cdef set getcandidates(Node *a, uint64_t *bitset, short i, Ctrees trees):
	"""Get candidates based on trigrams in fragment ``bitset`` at ``a[i]``."""
	if a[i].left < 0:
		return trees.prodindex[a[i].prod]
	elif a[i].right < 0:
		if TESTBIT(bitset, a[i].left):
			return (trees.trigramindex[
					a[i].prod,
					a[a[i].left].prod,
					a[i].right]
					& getcandidates(a, bitset, a[i].left, trees))
		return trees.prodindex[a[i].prod]
	elif TESTBIT(bitset, a[i].left):
		if TESTBIT(bitset, a[i].right):
			return (trees.trigramindex[
						a[i].prod,
						a[a[i].left].prod, a[a[i].right].prod]
					& getcandidates(a, bitset, a[i].left, trees)
					& getcandidates(a, bitset, a[i].right, trees))
		else:
			return (trees.prodindex[a[i].prod]
					& getcandidates(a, bitset, a[i].left, trees))
	elif TESTBIT(bitset, a[i].right):
		return (trees.prodindex[a[i].prod]
				& getcandidates(a, bitset, a[i].right, trees))
	# this is a lexical or frontier node
	return trees.prodindex[a[i].prod]


cdef inline int containsbitset(Node *a, Node *b, uint64_t *bitset,
		short i, short j):
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


cpdef dict completebitsets(Ctrees trees, list sents, list labels,
		short maxnodes, bint discontinuous=False):
	"""Generate bitsets corresponding to whole trees in the input.

	Important: if multiple treebanks are used, maxnodes should equal
	``max(trees1.maxnodes, trees2.maxnodes)``"""
	cdef:
		dict result = {}
		list sent
		int n, i
		short SLOTS = BITNSLOTS(maxnodes + 1)
		uint64_t *scratch = <uint64_t *>malloc((SLOTS + 2) * sizeof(uint64_t))
		Node *nodes
		list tmp = []
	for n in range(trees.len):
		memset(<void *>scratch, 0, SLOTS * sizeof(uint64_t))
		nodes = &trees.nodes[trees.trees[n].offset]
		sent = sents[n]
		for i in range(trees.trees[n].len):
			if nodes[i].left >= 0 or sent[termidx(nodes[i].left)] is not None:
				SETBIT(scratch, i)
		setrootid(scratch, trees.trees[n].root, n, SLOTS)
		getsubtree(tmp, nodes, scratch, labels,
				None if discontinuous else sent, trees.trees[n].root)
		frag = (''.join(tmp), tuple(sent)) if discontinuous else ''.join(tmp)
		del tmp[:]
		result[frag] = wrap(scratch, SLOTS)
	return result


cdef set twoterminals(NodeArray a, Node *anodes,
		Ctrees trees2, set contentwordprods, set lexicalprods):
	"""Produce tree pairs that share at least two words.

	Specifically, tree pairs sharing one content word and one additional word,
	where content words are recognized by a POS tag from the Penn treebank tag
	set.

	if trees2 is None, pairs (n, m) are such that n < m."""
	cdef int i, j
	cdef set tmp, candidates = set()
	# select candidates from 'trees2' that share productions with tree 'a'
	# want to select at least 1 content POS tag, 1 other lexical prod
	for i in range(a.len):
		if anodes[i].left >= 0 or anodes[i].prod not in contentwordprods:
			continue
		tmp = <set>(trees2.prodindex[anodes[i].prod])
		for j in range(a.len):
			if (i != j and anodes[j].left < 0 and
					anodes[j].prod in lexicalprods):
				candidates |= tmp & <set>(trees2.prodindex[anodes[j].prod])
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


def allfragments(Ctrees trees, list sents, list labels,
		int maxdepth, int maxfrontier=999, bint discontinuous=True,
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
		collectfragments(fragments, inter, nodes, sents[n], labels,
				discontinuous, not indices, indices, tmp, SLOTS)
		del table[:]
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


cdef inline short termidx(short x):
	"""Translate representation for terminal indices."""
	return -x - 1


cdef inline getsubtree(list result, Node *tree, uint64_t *bitset,
		list labels, list sent, int i):
	"""Get string of tree fragment denoted by bitset; indices as terminals.

	:param result: provide an empty list for the initial call."""
	result.append('(')
	result.append(labels[tree[i].prod])
	result.append(' ')
	if TESTBIT(bitset, i):
		if tree[i].left >= 0:
			getsubtree(result, tree, bitset, labels, sent, tree[i].left)
			if tree[i].right >= 0:
				result.append(' ')
				getsubtree(result, tree, bitset, labels, sent, tree[i].right)
		elif sent is None:
			result.append(str(termidx(tree[i].left)))
		else:
			result.append(sent[termidx(tree[i].left)])
	elif sent is None:  # node not in bitset, frontier non-terminal
		result.append(yieldranges(sorted(getyield(tree, i))))
	result.append(')')


cdef inline list getyield(Node *tree, int i):
	"""Recursively collect indices of terminals under a node."""
	if tree[i].left < 0:
		return [termidx(tree[i].left)]
	elif tree[i].right < 0:
		return getyield(tree, tree[i].left)
	return getyield(tree, tree[i].left) + getyield(tree, tree[i].right)


def repl(d):
	"""A function for use with re.sub that looks up numeric IDs in a dict.
	"""
	def f(x):
		return d[int(x.group(1))]
	return f


def pygetsent(str frag, list sent):
	"""Wrapper of ``getsent()`` to make doctests possible.

	>>> frag, sent = pygetsent(u'(S (NP 2) (VP 4))',
	... [u'The', u'tall', u'man', u'there', u'walks'])
	>>> print(frag)
	(S (NP 0) (VP 2))
	>>> print(repr(sent).replace("u'", "'"))
	('man', None, 'walks')
	>>> frag, sent = pygetsent(u'(VP (VB 0) (PRT 3))',
	...	[u'Wake', u'your', u'friend', u'up'])
	>>> print(frag)
	(VP (VB 0) (PRT 2))
	>>> print(repr(sent).replace("u'", "'"))
	('Wake', None, 'up')
	>>> frag, sent = pygetsent(u'(S (NP 2:2 4:4) (VP 1:1 3:3))',
	... [u'Walks',u'the',u'quickly',u'man'])
	>>> print(frag)
	(S (NP 1 3) (VP 0 2))
	>>> print(sent)
	(None, None, None, None)
	>>> frag, sent = pygetsent(u'(ROOT (S 0:2) ($. 3))',
	... [u'Foo', u'bar', u'zed', u'.'])
	>>> print(frag)
	(ROOT (S 0) ($. 1))
	>>> print(repr(sent).replace("u'", "'"))
	(None, '.')
	>>> frag, sent = pygetsent(u'(ROOT (S 0) ($. 3))',
	... [u'Foo', u'bar', u'zed',u'.'])
	>>> print(frag)
	(ROOT (S 0) ($. 2))
	>>> print(repr(sent).replace("u'", "'"))
	('Foo', None, '.')
	>>> frag, sent = pygetsent(u'(S|<VP>_2 (VP_3 0:1 3:3 16:16) (VAFIN 2))',
	... u'''In Japan wird offenbar die Fusion der Geldkonzerne Daiwa und
	...  Sumitomo zur gr\\xf6\\xdften Bank der Welt vorbereitet .'''.split())
	>>> print(frag)
	(S|<VP>_2 (VP_3 0 2 4) (VAFIN 1))
	>>> print(repr(sent).replace("u'", "'"))
	(None, 'wird', None, None, None)"""
	try:
		return getsent(frag, sent)
	except:
		print(frag, '\n', sent)
		raise


cdef getsent(str frag, list sent):
	"""Renumber indices in fragment and select terminals it contains.

	Returns a transformed copy of fragment and sentence. Replace words that do
	not occur in the fragment with None and renumber terminals in fragment such
	that the first index is 0 and any gaps have a width of 1. Expects a tree as
	string where frontier nodes are marked with intervals."""
	cdef:
		int n, m = 0, maxl
		list newsent = []
		dict leafmap = {}
		dict spans = {int(start): int(end) + 1
				for start, end in FRONTIERRE.findall(frag)}
		list leaves = list(map(int, TERMINDICESRE.findall(frag)))
	for n in leaves:
		spans[n] = n + 1
	maxl = max(spans)
	for n in sorted(spans):
		newsent.append(sent[n] if n in leaves else None)
		leafmap[n] = ' %d' % m
		m += 1
		if spans[n] not in spans and n != maxl:  # a gap
			newsent.append(None)
			m += 1
	frag = FRONTIERORTERMRE.sub(repl(leafmap), frag)
	return frag, tuple(newsent)


cdef dumpmatrix(uint64_t *matrix, NodeArray a, NodeArray b, Node *anodes,
		Node *bnodes, list asent, list bsent, list labels, uint64_t *scratch,
		short SLOTS):
	"""Dump a table of the common productions of a tree pair."""
	dumptree(a, anodes, asent, labels, scratch)
	dumptree(b, bnodes, bsent, labels, scratch)
	print('\t'.join([''] + ['%2d' % x for x in range(b.len)
		if bnodes[x].prod != -1]))
	for m in range(b.len):
		print('\t', labels[(<Node>bnodes[m]).prod][:3], end='')
	for n in range(a.len):
		print('\n%2d' % n, labels[(<Node>anodes[n]).prod][:3], end='')
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


cdef dumptree(NodeArray a, Node *anodes, list asent, list labels,
		uint64_t *scratch):
	"""Print debug information of a given tree."""
	for n in range(a.len):
		print('idx=%2d\tleft=%2d\tright=%2d\tprod=%2d\tlabel=%s' % (n,
				termidx(anodes[n].left) if anodes[n].left < 0
				else anodes[n].left,
				anodes[n].right, anodes[n].prod, labels[anodes[n].prod]),
				end='')
		if anodes[n].left < 0:
			if asent[termidx(anodes[n].left)]:
				print('\t%s=%s' % ('terminal', asent[termidx(anodes[n].left)]))
			else:
				print('\tfrontier non-terminal')
		else:
			print()
	tmp = []
	memset(<void *>scratch, 255, BITNSLOTS(a.len) * sizeof(uint64_t))
	getsubtree(tmp, anodes, scratch, labels, None, a.root)
	print(tmp, '\n', asent, '\n')


def getprods(tree, sent, disc=True):
	"""Get a list of LCFRS/CFG productions for a tree.

	:param disc: pass disc=False to get CFG productions."""
	if disc:
		return lcfrsproductions(tree, sent, frontiers=True)
	else:
		result = []
		for a in tree.subtrees():
			if len(a) == 0:
				result.append((a.label, ))
			elif len(a) == 1:
				result.append((a.label,
						a[0].label if isinstance(a[0], Tree)
						else sent[a[0]]))
			elif len(a) == 2:
				result.append((a.label, a[0].label, a[1].label))
			else:
				raise ValueError('tree not binarized.')
		return result


def nonfrontier(sent):
	def nonfrontierfun(x):
		return isinstance(x[0], Tree) or sent[x[0]] is not None
	return nonfrontierfun


cdef readnode(str label, str line, short start, short end,
		list labels, dict prods, Node *result, size_t *idx, list sent,
		str binlabel):
	"""Parse tree in bracket format into pre-allocated array of Node structs.

	:param idx: a counter to keep track of the number of Node structs used.
	:param sent: collects the terminals encountered.
	:param binlabel: used for on-the-fly binarization, pass None initially."""
	cdef:
		short n, parens = 0, left = -1, right = -1
		short startchild1 = 0, startchild2 = 0, endchild1 = 0, endchild2 = 0
		list childlabels = None
		str labelchild1 = None, labelchild2 = None
	# walk through the string and find the first two children
	for n in range(start, end):
		if line[n] == '(':
			if parens == 0:
				start = n
			parens += 1
		elif line[n] == ')':
			parens -= 1
			if parens == -1:
				if startchild1 == 0:
					startchild1, endchild1 = start, n
			elif parens == 0:
				if startchild1 == 0:
					startchild1, endchild1 = start, n + 1
				elif startchild2 == 0:
					startchild2, endchild2 = start, n + 1
				else:  # will do on the fly binarization
					childlabels = []
					break
	# get left child label
	match1 = LABEL.match(line, startchild1)
	if match1 is not None:  # non-terminal label
		labelchild1 = match1.group(1)
		startchild1 = match1.end()
	elif startchild2 != 0:  # insert preterminal
		labelchild1 = '/'.join((label, line[startchild1:endchild1]))
	else:  # terminal following pre-terminal; store terminal
		# leading space distinguishes terminal from non-terminal
		labelchild1 = ' ' + line[startchild1:endchild1]
		left = termidx(len(sent))
		sent.append(line[startchild1:endchild1]
				if startchild1 < endchild1 else None)
	# if there were more children, collect labels for a binarized constituent
	if childlabels is not None:
		for n in range(startchild2, end):
			if line[n] == '(':
				if parens == 0:
					start = n
				parens += 1
			elif line[n] == ')':
				parens -= 1
				if parens == 0:
					match = LABEL.match(line, start)
					if match is None:  # introduce preterminal
						childlabels.append(
								'/'.join((label, line[start:n + 1])))
					else:
						childlabels.append(match.group(1))
		if not binlabel:
			binlabel = label + '|<' + labelchild1
			labelchild2 = binlabel + '.' + ','.join(childlabels) + '>'
		else:
			binlabel += ',' + labelchild1
			labelchild2 = binlabel + '.' + ','.join(childlabels) + '>'
		endchild2 = end
	if parens != -1:
		raise ValueError('unbalanced parentheses: %d\n%r' % (parens, line))
	if startchild2 == 0:
		prod = (label, labelchild1) if startchild1 else (label, )
	else:
		if labelchild2 is None:
			match = LABEL.match(line, startchild2)
			if match is not None:
				labelchild2 = match.group(1)
				startchild2 = match.end()
			else:  # insert preterminal
				labelchild2 = '/'.join((label, line[startchild2:endchild2]))
		prod = (label, labelchild1, labelchild2)
	if prod not in prods:  # assign new ID?
		prods[prod] = len(prods)
		labels.append(label)
	n = idx[0]
	idx[0] += 1
	if match1 is not None or startchild2 != 0:
		left = idx[0]
		readnode(labelchild1, line, startchild1, endchild1, labels,
				prods, result, idx, sent, None)
		if startchild2 != 0:
			right = idx[0]
			readnode(labelchild2, line, startchild2, endchild2, labels,
					prods, result, idx, sent, childlabels and binlabel)
	# store node
	result[n].prod = prods[prod]
	result[n].left = left
	result[n].right = right


cdef inline copynodes(tree, list prodsintree, dict prods,
		Node *result, size_t *idx):
	"""Convert a binarized Tree object to an array of Node structs."""
	cdef size_t n = idx[0]
	if not isinstance(tree, Tree):
		raise ValueError('Expected Tree node, got %s\n%r' % (type(tree), tree))
	elif not 1 <= len(tree) <= 2:
		raise ValueError('trees must be non-empty and binarized\n%s' % tree)
	result[n].prod = prods[prodsintree[n]]
	idx[0] += 1
	if isinstance(tree[0], int):  # a terminal index
		result[n].left = -tree[0] - 1
		result[n].right = -1
	else:
		result[n].left = idx[0]
		copynodes(tree[0], prodsintree, prods, result, idx)
		if len(tree) == 1:  # unary node
			result[n].right = -1
		else:  # binary node
			result[n].right = idx[0]
			copynodes(tree[1], prodsintree, prods, result, idx)


def getctrees(trees1, sents1, disc=True, trees2=None, sents2=None):
	""":returns: Ctrees object for disc. binary trees and sentences."""
	cdef Ctrees ctrees, ctrees1, ctrees2 = None
	cdef Node *scratch
	cdef size_t cnt
	maxnodes = 512
	scratch = <Node *>malloc(maxnodes * sizeof(Node))
	if scratch is NULL:
		raise MemoryError('allocation error')
	labels = []
	prods = {}
	ctrees = ctrees1 = Ctrees()
	ctrees.alloc(512, 512 * 512)
	for m, (trees, sents) in enumerate(((trees1, sents1), (trees2, sents2))):
		if trees is None or sents is None:
			break
		if m == 1:
			ctrees = ctrees2 = Ctrees()
			ctrees.alloc(512, 512 * 512)
		for tree, sent in zip(trees, sents):
			prodsintree = getprods(tree, sent, disc=disc)
			for prod in prodsintree:
				if prod not in prods:
					labels.append(prod[0][0] if disc else prod[0])
					prods[prod] = len(prods)
			if len(prodsintree) > maxnodes:
				maxnodes = len(prodsintree)
				scratch = <Node *>realloc(scratch,
						maxnodes * sizeof(Node))
				if scratch is NULL:
					raise MemoryError('allocation error')
			cnt = 0
			copynodes(tree, prodsintree, prods, scratch, &cnt)
			ctrees.addnodes(scratch, cnt, 0)
		ctrees.indextrees(prods)
	return dict(trees1=ctrees1, sents1=sents1, trees2=ctrees2, sents2=sents2,
			prods=prods, labels=labels)


def readtreebank(treebankfile, list labels, dict prods,
		fmt='bracket', limit=None, encoding='utf-8'):
	"""Read a treebank from a given filename.

	labels and prods should be re-used when reading multiple treebanks."""
	cdef Ctrees ctrees
	cdef Node *scratch
	cdef size_t cnt, n
	cdef str line
	if treebankfile is None:
		return None, None
	ctrees = Ctrees()
	ctrees.alloc(512, 512 * 512)  # dummy values, array will be realloc'd
	sents = []
	maxnodes = 512
	binfactor = 2  # conservative estimate to accommodate binarization
	scratch = <Node *>malloc(maxnodes * binfactor * sizeof(Node))
	if scratch is NULL:
		raise MemoryError('allocation error')
	if fmt != 'bracket':
		from discodop.treebank import READERS
		from discodop.treetransforms import canonicalize
		corpus = READERS[fmt](treebankfile, encoding=encoding)
		for _, item in corpus.itertrees(0, limit):
			tree = canonicalize(binarize(item.tree, dot=True))
			prodsintree = getprods(tree, item.sent)
			for prod in prodsintree:
				if prod not in prods:
					labels.append(prod[0][0])
					prods[prod] = len(prods)
			if len(prodsintree) > maxnodes:
				maxnodes = len(prodsintree)
				scratch = <Node *>realloc(scratch,
						maxnodes * binfactor * sizeof(Node))
				if scratch is NULL:
					raise MemoryError('allocation error')
			cnt = 0
			copynodes(tree, prodsintree, prods, scratch, &cnt)
			ctrees.addnodes(scratch, cnt, 0)
			sents.append(item.sent)
	else:  # do incremental reading of bracket trees
		# could use BracketCorpusReader or expect trees/sents as input, but
		# incremental reading reduces memory requirements.
		data = io.open(treebankfile, encoding=encoding)
		for n, line in enumerate(islice(data, limit), 1):
			if line.count('(') > maxnodes:
				maxnodes = 2 * line.count('(')
				scratch = <Node *>realloc(scratch,
						maxnodes * binfactor * sizeof(Node))
				if scratch is NULL:
					raise MemoryError('allocation error')
			cnt = 0
			sent = []
			match = LABEL.match(line)
			if match is None:
				raise ValueError('malformed tree in %s:%d\n%r' % (
						treebankfile, n, line))
			readnode(match.group(1), line, match.end(), len(line),
					labels, prods, scratch, &cnt, sent, None)
			ctrees.addnodes(scratch, cnt, 0)
			sents.append(sent)
		if not sents:
			raise ValueError('%r appears to be empty' % treebankfile)
		free(scratch)
	return ctrees, sents


__all__ = ['extractfragments', 'exactcounts', 'completebitsets',
		'allfragments', 'repl', 'pygetsent', 'nonfrontier', 'getctrees',
		'readtreebank']
