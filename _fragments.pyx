""" Implementation of Sangati et al. (2010), Efficiently extract recurring tree
fragments from large treebanks.
Moschitti (2006): Making Tree Kernels practical for Natural Language
Learning. """

import re, codecs, sys
from collections import defaultdict, Counter as multiset
from itertools import count
from functools import partial
from array import array
from tree import Tree
from grammar import lcfrs_productions
from containers import Terminal
from treetransforms import binarize, introducepreterminals

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cpython.array cimport array, clone
from containers cimport ULong, UInt, UChar
from containers cimport Node, NodeArray, Ctrees, FrozenArray, new_FrozenArray
from bit cimport anextset, abitcount, subset, ulongcpy, ulongset

cdef extern from "macros.h":
	int BITSIZE
	int BITSLOT(int b)
	void SETBIT(ULong a[], int b)
	void TOGGLEBIT(ULong a[], int b)
	void CLEARBIT(ULong a[], int b)
	ULong TESTBIT(ULong a[], int b)
	ULong BITMASK(int b)
	int BITNSLOTS(int nb)
	int IDX(int i, int j, int jmax, int kmax)

# these arrays are never used but serve as template to create
# arrays of this type.
cdef array uintarray = array("I", []), ulongarray = array("L", [])

# used to generate values for defaultdicts
def one():
	return 1

cpdef fastextractfragments(Ctrees trees1, list sents1, int offset, int end,
	list revlabel, Ctrees trees2=None, list sents2=None, bint approx=True,
	bint debug=False, bint discontinuous=False, bint complement=False):
	""" Seeks the largest fragments in treebank(s) with a linear time tree
	kernel
	- scenario 1: recurring fragments in single treebank, use:
      fastextractfragments(trees1, sents1, offset, end, revlabel)
	- scenario 2: common fragments in two treebanks:
      fastextractfragments(trees1, sents1, offset, end, trees2, sents2)
	offset and end can be used to divide the work over multiple processes.
	they are indices of trees1 to work on (default is all).
	when debug is enabled a contingency table is shown for each pair of trees.
	when complement is true, the complement of the recurring fragments in each
	pair of trees is extracted as well.
	"""
	cdef:
		int n, m, x, start = 0, end2
		UChar SLOTS	# the number of bitsets needed to cover the largest tree
		ULong *CST = NULL # common subtree table
		ULong *bitset # a temporary variable
		NodeArray a, b, *ctrees1, *ctrees2
		list asent
		dict fragments = {}
		set inter = set()
		FrozenArray frozenarray
	if approx:
		fragments = <dict>defaultdict(one)

	if trees2 is None:
		trees2 = trees1
	ctrees1 = trees1.data
	ctrees2 = trees2.data
	SLOTS = BITNSLOTS(max(trees1.maxnodes, trees2.maxnodes))
	CST = <ULong *>malloc(trees2.maxnodes * SLOTS * sizeof(ULong))
	assert CST is not NULL
	end2 = trees2.len
	# find recurring fragments
	for n in range(offset, min(end or trees1.len, trees1.len)):
		a = ctrees1[n]
		asent = <list>(sents1[n])
		if sents2 is None:
			start = n + 1
		for m in range(start, end2):
			b = ctrees2[m]
			# initialize table
			ulongset(CST, 0UL, b.len * SLOTS)
			# fill table
			fasttreekernel(a.nodes, b.nodes, CST, SLOTS)
			# dump table
			if debug:
				print n, m
				dumpCST(CST, a, b, asent, (sents2 or sents1)[m],
					revlabel, SLOTS, True)
			# extract results
			extractbitsets(CST, a.nodes, b.nodes, b.root, n, inter, SLOTS)
			# extract complementary fragments?
			if complement:
				# combine bitsets of inter together with bitwise or
				ulongset(CST, 0, SLOTS)
				for frozenarray in inter:
					for x in range(SLOTS):
						CST[x] |= frozenarray.obj.data.as_ulongs[x]
				# extract bitsets in A from result, without regard for B
				extractcompbitsets(CST, a.nodes, a.root, n, inter, SLOTS, NULL)
		# collect string representations of fragments
		for frozenarray in inter:
			bitset = frozenarray.obj.data.as_ulongs
			frag = getsubtree(a.nodes, bitset, revlabel,
					None if discontinuous else asent, bitset[SLOTS])
			if discontinuous:
				frag = getsent(frag, asent)
			if approx:
				fragments[frag] += 1
			elif frag not in fragments:
				fragments[frag] = frozenarray.obj
		inter.clear()
	free(CST)
	return fragments

cdef inline void fasttreekernel(Node *a, Node *b, ULong *CST, UChar SLOTS):
	""" Fast Tree Kernel (average case linear time). Expects trees to be sorted
	according to their productions (in descending order, with terminals as -1).
	Moschitti (2006): Making Tree Kernels practical for Natural Language
	Learning.  """
	# i as an index to a, j to b, and jj is a temp index starting at j.
	cdef int i = 0, j = 0, jj = 0
	while a[i].prod != -1 and b[j].prod != -1:
		if   a[i].prod < b[j].prod:
			j += 1
		elif a[i].prod > b[j].prod:
			i += 1
		else:
			while a[i].prod != -1 and a[i].prod == b[j].prod:
				jj = j
				while b[jj].prod != -1 and a[i].prod == b[jj].prod:
					SETBIT(&CST[jj * SLOTS], i)
					jj += 1
				i += 1

cdef inline void extractbitsets(ULong *CST, Node *a, Node *b,
	short j, int n, set results, UChar SLOTS):
	""" Visit nodes of 'b' in pre-order traversal. store bitsets of connected
	subsets of 'a' as they are encountered, following to the common nodes
	specified in CST. j is the node in 'b' to start with, which changes with
	each recursive step. 'n' is the identifier of this tree which is stored
	with extracted fragments. """
	cdef array pyarray
	cdef ULong *bitset = &CST[j * SLOTS]
	cdef short i = anextset(bitset, 0, SLOTS)
	while i != -1:
		pyarray = clone(ulongarray, (SLOTS + 2), True)
		extractat(CST, pyarray.data.as_ulongs, a, b, i, j, SLOTS)
		pyarray.data.as_ulongs[SLOTS] = i
		pyarray.data.as_ulongs[SLOTS + 1] = n
		results.add(new_FrozenArray(pyarray))
		i = anextset(bitset, i + 1, SLOTS)
	if b[j].left < 0:
		return
	if b[b[j].left].prod >= 0:
		extractbitsets(CST, a, b, b[j].left, n, results, SLOTS)
	if b[j].right >= 0 and b[b[j].right].prod >= 0:
		extractbitsets(CST, a, b, b[j].right, n, results, SLOTS)

cdef inline void extractat(ULong *CST, ULong *result, Node *a, Node *b,
	short i, short j, UChar SLOTS):
	""" Traverse tree a and b in parallel to extract a connected subset """
	SETBIT(result, i)
	CLEARBIT(&CST[j * SLOTS], i)
	if a[i].left < 0:
		return
	elif TESTBIT(&CST[b[j].left * SLOTS], a[i].left):
		extractat(CST, result, a, b, a[i].left, b[j].left, SLOTS)
	if a[i].right < 0:
		return
	elif TESTBIT(&CST[b[j].right * SLOTS], a[i].right):
		extractat(CST, result, a, b, a[i].right, b[j].right, SLOTS)

cdef inline void extractcompbitsets(ULong *bitset, Node *a, int i, int n,
		set results, UChar SLOTS, ULong *scratch):
	""" Visit nodes of 'a' in pre-order traversal. store bitsets of connected
	subsets of 'a' as they are encountered, following the nodes specified in
	'bitset'. i is the node in 'a' to start with, which changes with each
	recursive step. 'n' is the identifier of this tree which is stored with
	extracted fragments. """
	cdef array pyarray
	cdef bint start = scratch is NULL and not TESTBIT(bitset, i)
	if start:
		pyarray = clone(ulongarray, (SLOTS + 2), True)
		scratch = pyarray.data.as_ulongs
	else:
		scratch = NULL
	if scratch is not NULL:
		SETBIT(scratch, i)
	if a[i].left >= 0 and a[a[i].left].prod >= 0:
		extractcompbitsets(bitset, a, a[i].left, n, results, SLOTS, scratch)
	if a[i].right >= 0 and a[a[i].right].prod >= 0:
		extractcompbitsets(bitset, a, a[i].right, n, results, SLOTS, scratch)
	if start:
		pyarray.data.as_ulongs[SLOTS] = i
		pyarray.data.as_ulongs[SLOTS + 1] = n
		results.add(new_FrozenArray(pyarray))

cpdef dict coverbitsets(Ctrees trees, list sents, list treeswithprod,
		list revlabel, bint discontinuous):
	""" Utility function to generate one bitset for each type of production. """
	cdef dict result = {}
	cdef array pyarray
	cdef int p, m, n = -1
	cdef UChar SLOTS = BITNSLOTS(trees.maxnodes)
	for p, treeindices in enumerate(treeswithprod):
		pyarray = clone(ulongarray, SLOTS + 2, True)
		n = iter(treeindices).next()
		for m in range(trees.data[n].len):
			if trees.data[n].nodes[m].prod == p:
				SETBIT(pyarray.data.as_ulongs, m)
				break
		else:
			raise ValueError("production not found. wrong index?")
		pyarray.data.as_ulongs[SLOTS] = m
		pyarray.data.as_ulongs[SLOTS + 1] = n
		frag = getsubtree(trees.data[n].nodes, pyarray.data.as_ulongs, revlabel,
				None if discontinuous else sents[n], m)
		if discontinuous:
			frag = getsent(frag, sents[n])
		result[frag] = pyarray
	return result

cpdef dict completebitsets(Ctrees trees, list sents, list revlabel):
	""" Utility function to generate bitsets corresponding to whole trees
	in the input."""
	cdef dict result = {}
	cdef array pyarray
	cdef int n, m
	cdef UChar SLOTS = BITNSLOTS(trees.maxnodes)
	for n in range(trees.len):
		pyarray = clone(ulongarray, SLOTS + 2, True)
		for m in range(trees.data[n].len):
			if trees.data[n].nodes[m].prod != -1:
				SETBIT(pyarray.data.as_ulongs, m)
		pyarray.data.as_ulongs[SLOTS] = trees.data[n].root
		pyarray.data.as_ulongs[SLOTS + 1] = n
		frag = strtree(trees.data[n].nodes, revlabel,
			None if sents is None else sents[n], trees.data[n].root)
		if sents is not None:
			frag = getsent(frag, sents[n])
		result[frag] = pyarray
	return result

cpdef exactcounts(Ctrees trees1, Ctrees trees2, list bitsets,
	list treeswithprod, bint fast=True):
	""" Given a set of fragments from trees2 as bitsets, produce an exact
	count of those fragments in trees1 (which may be equal to trees2).
	The reason we need to do this separately from extracting maximal fragments
	is that a fragment can occur in other trees where it was not a maximal. """
	cdef:
		UInt count, n, m, i, j
		array counts = clone(uintarray, len(bitsets), True)
		array bitset = bitsets[0]
		UChar SLOTS = len(bitset) - 2
		NodeArray a, b, *ctrees1 = trees1.data, *ctrees2 = trees2.data
		set candidates
	assert SLOTS
	# compare one bitset to each tree for each unique fragment.
	for n, bitset in enumerate(bitsets):
		a = ctrees2[bitset.data.as_ulongs[SLOTS + 1]] # the fragment
		candidates = {m for m in range(trees1.len)}
		for i in range(a.len):
			if TESTBIT(bitset.data.as_ulongs, i):
				candidates &= <set>(treeswithprod[a.nodes[i].prod])
		i = bitset.data.as_ulongs[SLOTS] # root of fragment in a

		count = 0
		for m in candidates:
			b = ctrees1[m]
			for j in range(b.len):
				if a.nodes[i].prod != -1 and a.nodes[i].prod == b.nodes[j].prod:
					# exploit the fact that True == 1, False == 0
					count += containsbitset(a, b, bitset.data.as_ulongs, i, j)
				elif fast and a.nodes[i].prod > b.nodes[j].prod:
					break
		counts.data.as_uints[n] = count
	return counts

cpdef list exactindices(Ctrees trees1, Ctrees trees2, list bitsets,
	list treeswithprod, bint fast=True):
	""" Given a set of fragments from trees2 as bitsets, produce a mapping of
	fragments to the multiset of indices to trees1 which contain those
	fragments. This is a multiset to account for multiple occurrences of a
	fragment in a tree. (NB: trees1 may be equal to trees2). The reason we need
	to do this separately is that a fragment can occur in other trees where it
	was not a maximal fragment.
	Returns a list with a multiset of indices for each bitset in the input;
	the indices point to the trees where the fragments (described by the
	bitsets) occur. This is useful to look up the contexts of fragments, or
	when the order of sentences has significance which makes it possible to
	interpret the set of indices as time-series data. """
	cdef:
		int n, m, i, j
		UInt count
		UChar SLOTS = 0
		array bitset
		NodeArray a, b, *ctrees1 = trees1.data, *ctrees2 = trees2.data
		set candidates
		list indices = [set() for _ in bitsets]
	if SLOTS == 0:
		pyarray = bitsets[0]
		SLOTS = len(pyarray) - 2
		assert SLOTS
	# compare one bitset to each tree for each unique fragment.
	for n, bitset in enumerate(bitsets):
		a = ctrees2[bitset.data.as_ulongs[SLOTS + 1]] # the fragment
		candidates = {m for m in range(trees1.len)}
		for i in range(a.len):
			if TESTBIT(bitset.data.as_ulongs, i):
				candidates &= <set>(treeswithprod[a.nodes[i].prod])
		i = bitset.data.as_ulongs[SLOTS] # root of fragment in a

		matches = multiset()
		for m in candidates:
			b = ctrees1[m]
			for j in range(b.len):
				if a.nodes[i].prod != -1 and a.nodes[i].prod == b.nodes[j].prod:
					if containsbitset(a, b, bitset.data.as_ulongs, i, j):
						matches[m] += 1
				elif fast and a.nodes[i].prod > b.nodes[j].prod:
					break
		indices[n] = matches
	return indices

cdef inline int containsbitset(NodeArray A, NodeArray B, ULong *bitset,
	short i, short j):
	""" Recursively check whether fragment starting from A[i] described by
	bitset is equal to B[j], i.e., whether B contains that fragment from A. """
	cdef Node a = A.nodes[i], b = B.nodes[j]
	if a.prod != b.prod:
		return 0
	# lexical production, no recursion
	if A.nodes[a.left].prod == -1:
		return 1
	# test for structural mismatch
	if (a.left < 0) != (a.left < 0) or (a.right < 0) != (b.right < 0):
		return 0
	# recurse on further nodes
	if a.left < 0:
		return 1
	if TESTBIT(bitset, a.left):
		if not containsbitset(A, B, bitset, a.left, b.left):
			return 0
	if a.right < 0:
		return 1
	if TESTBIT(bitset, a.right):
		return containsbitset(A, B, bitset, a.right, b.right)
	return 1

cpdef extractfragments(Ctrees trees1, list sents1, int offset, int end,
	list revlabel, Ctrees trees2=None, list sents2=None, bint approx=True,
	bint debug=False, bint discontinuous=False):
	""" Seeks the largest fragments in treebank(s)
	- scenario 1: recurring fragments in single treebank, use:
		extractfragments(trees1, sents1, offset, end)
	- scenario 2: common fragments in two treebanks:
		extractfragments(trees1, sents1, offset, end, trees2, sents2)
	offset and end can be used to divide the work over multiple processes.
	offset is the starting point in trees1, end is the number of trees from
	trees1 to work on.
	when debug is enabled a contingency table is shown for each pair of trees.
	"""
	cdef:
		int n, m, aa, bb, start = 0
		UChar SLOTS
		ULong *CST = NULL
		NodeArray a, b, result
		NodeArray *ctrees1, *ctrees2
		FrozenArray frozenarray
		list asent
		dict fragments = {}
		set inter = set()

	if approx:
		fragments = <dict>defaultdict(one)
	if trees2 is None:
		trees2 = trees1
	ctrees1 = trees1.data
	ctrees2 = trees2.data
	SLOTS = BITNSLOTS(max(trees1.maxnodes, trees2.maxnodes))
	CST = <ULong *>malloc(trees1.maxnodes * trees2.maxnodes
		* SLOTS * sizeof(ULong))
	assert CST is not NULL

	# find recurring fragments
	for n in range(offset, (end or trees1.len)):
		a = ctrees1[n]
		asent = <list>(sents1[n])
		if sents2 is None:
			start = n + 1
		for m in range(start, trees2.len):
			b = ctrees2[m]
			# initialize table
			ulongset(CST, 0UL, a.len * b.len * SLOTS)
			# fill table
			for aa in range(a.len): # skip terminals
				if a.nodes[aa].prod == -1:
					break
				for bb in range(b.len): #skip terminals
					if b.nodes[bb].prod == -1:
						break
					elif not TESTBIT(&CST[IDX(aa, bb, b.len, SLOTS)], 0):
						getCST(a, b, CST, aa, bb, SLOTS)
			# dump table
			if debug:
				print n, m
				dumpCST(CST, a, b, asent,
					(sents2[m] if sents2 else sents1[m]), revlabel, SLOTS)
			# extract results
			inter.update(getnodeset(CST, a.len, b.len, n, SLOTS))
		for frozenarray in inter:
			frag = getsubtree(a.nodes, frozenarray.obj.data.as_ulongs, revlabel,
				None if discontinuous else asent,
				frozenarray.obj.data.as_ulongs[SLOTS])
			if discontinuous:
				frag = getsent(frag, asent)
			if approx:
				fragments[frag] += 1
			else:
				fragments[frag] = frozenarray.obj
		inter.clear()
	free(CST)
	return fragments

cdef inline void getCST(NodeArray A, NodeArray B, ULong *CST, int i, int j,
	UChar SLOTS):
	"""Recursively build common subtree table (CST) for subtrees A[i] & B[j]"""
	cdef Node a = A.nodes[i], b = B.nodes[j]
	cdef ULong *child, *bitset = &CST[IDX(i, j, B.len, SLOTS)]
	SETBIT(bitset, 0) # mark cell as visited
	# compare label & arity / terminal; assume presence of arity markers.
	if a.prod == b.prod:
		SETBIT(bitset, i + 1)
		# lexical production, no recursion
		if A.nodes[a.left].prod == -1:
			return
		# normal production, recurse or use cached value
		child = &CST[IDX(a.left, b.left, B.len, SLOTS)]
		if not child[0] & 1:
			getCST(A, B, CST, a.left, b.left, SLOTS)
		for n in range(SLOTS):
			bitset[n] |= child[n]
		if a.right != -1: # and b.right != -1:	#sentinel node
			child = &CST[IDX(a.right, b.right, B.len, SLOTS)]
			if not child[0] & 1:
				getCST(A, B, CST, a.right, b.right, SLOTS)
			for n in range(SLOTS):
				bitset[n] |= child[n]

cdef inline set getnodeset(ULong *CST, int lena, int lenb, int t, UChar SLOTS):
	""" Extract the largest, disjuncts bitsets from the common subtree
	table (CST). """
	cdef ULong *bitset
	cdef int n, m
	cdef array pyarray
	cdef FrozenArray bs, tmp
	cdef set finalnodeset = set()
	tmp = new_FrozenArray(clone(ulongarray, SLOTS, False))
	for n in range(lena):
		for m in range(lenb):
			bitset = &CST[IDX(n, m, lenb, SLOTS)]
			if not TESTBIT(bitset, 0) or abitcount(bitset, SLOTS) < 2:
				continue
			shiftright(bitset, SLOTS)
			ulongcpy(tmp.obj.data.as_ulongs, bitset, SLOTS)
			if tmp in finalnodeset:
				continue
			for bs in finalnodeset:
				if subset(bitset, bs.obj.data.as_ulongs, SLOTS):
					break
				elif subset(bs.obj.data.as_ulongs, bitset, SLOTS):
					pyarray = clone(ulongarray, SLOTS + 2, False)
					ulongcpy(pyarray.data.as_ulongs, bitset, SLOTS)
					pyarray.data.as_ulongs[SLOTS] = n
					pyarray.data.as_ulongs[SLOTS + 1] = t
					finalnodeset.add(new_FrozenArray(pyarray))
					finalnodeset.discard(bs)
					break
			else:
				# completely new (disjunct) bitset
				pyarray = clone(ulongarray, SLOTS + 2, False)
				ulongcpy(pyarray.data.as_ulongs, bitset, SLOTS)
				pyarray.data.as_ulongs[SLOTS] = n
				pyarray.data.as_ulongs[SLOTS + 1] = t
				finalnodeset.add(new_FrozenArray(pyarray))
	return finalnodeset

cdef void shiftright(ULong *bitset, UChar SLOTS):
	""" Shift an array of bitsets one bit to the right. """
	cdef int x
	cdef ULong mask = (1 << 1) - 1
	bitset[0] >>= 1
	for x from 1 <= x < SLOTS:
		bitset[x-1] |= bitset[x] & mask
		bitset[x] >>= 1

cdef inline unicode strtree(Node *tree, list revlabel, list sent, int i):
	""" produce string representation of (complete) tree. """
	if tree[i].prod == -1:
		if sent is None:
			return unicode(tree[i].label)
		return u"" if sent[tree[i].label] is None else sent[tree[i].label]
	if tree[i].left >= 0:
		if tree[i].right >= 0:
			return u"(%s %s %s)" % (revlabel[tree[i].label],
				strtree(tree, revlabel, sent, tree[i].left),
				strtree(tree, revlabel, sent, tree[i].right))
		return u"(%s %s)" % (revlabel[tree[i].label],
			strtree(tree, revlabel, sent, tree[i].left))
	return u"(%s )" % (revlabel[tree[i].label])

cdef inline unicode getsubtree(Node *tree, ULong *bitset, list revlabel,
	list sent, int i):
	""" Turn bitset into string representation of tree.  """
	if TESTBIT(bitset, i) and tree[i].left >= 0:
		if tree[i].right >= 0:
			return u"(%s %s %s)" % (revlabel[tree[i].label],
				getsubtree(tree, bitset, revlabel, sent, tree[i].left),
				getsubtree(tree, bitset, revlabel, sent, tree[i].right))
		return u"(%s %s)" % (revlabel[tree[i].label],
			getsubtree(tree, bitset, revlabel, sent, tree[i].left))
	elif tree[i].prod == -1: # a terminal
		return unicode(tree[i].label) if sent is None else sent[tree[i].label]
	elif sent is None:
		return u"(%s %s)" % (revlabel[tree[i].label],
				yieldranges(tree, sent, i))
	else:
		return u"(%s )" % (revlabel[tree[i].label])

cdef inline unicode yieldranges(Node *tree, list sent, int i):
	""" For discontinuous trees, return a string with the intervals of indices
	corresponding to the components in the yield of a node.
	The intervals are of the form start:end, where `end' is part of the
	interval. e.g., "0:1 2:4" corresponds to (0, 1) and (2, 3, 4). """
	cdef list yields = []
	cdef list leaves = sorted(getyield(tree, sent, i))
	cdef int a, start = -2, prev = -2
	for a in leaves:
		if a - 1 != prev:
			if prev != -2:
				yields.append(u"%d:%d" % (start, prev))
			start = a
		prev = a
	yields.append(u"%d:%d" % (start, prev))
	return u" ".join(yields)

cdef inline list getyield(Node *tree, list sent, int i):
	""" Recursively collect indices of terminals under a node. """
	if tree[i].prod == -1:
		return [tree[i].label]
	elif tree[i].left < 0:
		return [] #FIXME?
	elif tree[i].right < 0:
		return getyield(tree, sent, tree[i].left)
	return (getyield(tree, sent, tree[i].left)
		+ getyield(tree, sent, tree[i].right))

# match all leaf nodes containing indices
# by requiring a preceding space, only terminals are matched
# lookahead to ensure we match whole tokens ("23" instead of 2 or 3)
#termsre = re.compile(r" ([0-9]+)(?=[ )])")
# detect word boundary; less precise than  '[ )]',
# but works with linear-time regex libs
termsre = re.compile(r"\([^(]+ ([0-9]+)\)")
frontierre = re.compile(r" ([0-9]+):([0-9]+)\b")
frontierortermre = re.compile(r" ([0-9]+)(?::[0-9]+)?\b")
def repl(d):
	def f(x):
		return d[int(x.group(1))]
	return f

def getsent(frag, list sent):
	""" Select the words that occur in the fragment and renumber terminals in
	fragment such that the first index is 0 and any gaps have a width of 1.
	Expects a tree as string where frontier nodes are marked with intervals.
	>>> getsent("(S (NP 2) (VP 4))", ['The', 'tall', 'man', 'there', 'walks'])
	(u'(S (NP 0) (VP 2))', ('man', None, 'walks'))
	>>> getsent("(VP (VB 0) (PRT 3))", ['Wake', 'your', 'friend', 'up'])
	(u'(VP (VB 0) (PRT 2))', ('Wake', None, 'up'))
	>>> getsent("(S (NP 2:2 4:4) (VP 1:1 3:3))",['Walks','the','quickly','man'])
	(u'(S (NP 1 3) (VP 0 2))', (None, None, None, None))
	>>> getsent("(ROOT (S 0:2) ($. 3))", ['Foo', 'bar', 'zed', '.'])
	(u'(ROOT (S 0) ($. 1))', (None, '.'))
	>>> getsent("(ROOT (S 0) ($. 3))", ['Foo', 'bar', 'zed','.'])
	(u'(ROOT (S 0) ($. 2))', ('Foo', None, '.'))
	>>> getsent("(S|<VP>_2 (VP_3 0:1 3:3 16:16) (VAFIN 2))", "In Japan wird \
	offenbar die Fusion der Geldkonzerne Daiwa und Sumitomo zur \
	gr\\xf6\\xdften Bank der Welt vorbereitet .".split(" "))
	(u'(S|<VP>_2 (VP_3 0 2 4) (VAFIN 1))', (None, 'wird', None, None, None))
	"""
	cdef int n, m = 0, maxl
	cdef list newsent = []
	cdef dict leafmap = {}
	cdef dict spans = {int(start):
		int(end) + 1
			for start, end in frontierre.findall(frag)}
	cdef list leaves = map(int, termsre.findall(frag))
	spans.update((a, a + 1) for a in leaves)
	maxl = max(spans)
	for n in sorted(spans):
		if n in leaves:
			newsent.append(sent[n]) # a terminal
		else:
			newsent.append(None) # a frontier node
		leafmap[n] = u" " + unicode(m)
		m += 1
		if spans[n] not in spans and n != maxl: # a gap
			newsent.append(None)
			m += 1
	frag = frontierortermre.sub(repl(leafmap), frag)
	return frag, tuple(newsent)

cdef dumptree(NodeArray a, list revlabel):
	""" print a human-readable representation of a tree struct. """
	cdef int n
	for n in range(a.len):
		if a.nodes[n].prod == -1:
			break #skip terminals
		print "idx=%2d\tleft=%2d\tright=%2d\tprod=%2d\tlabel=%2d=%s" % (
			n, a.nodes[n].left, a.nodes[n].right, a.nodes[n].prod,
			a.nodes[n].label, revlabel[a.nodes[n].label])
	print

cdef dumpCST(ULong *CST, NodeArray a, NodeArray b, list asent, list bsent,
	list revlabel, UChar SLOTS, bint bitmatrix=False):
	""" print a table of the common subtrees of two trees. """
	cdef int n, m
	cdef Node aa, bb
	dumptree(a, revlabel)
	dumptree(b, revlabel)
	print strtree(a.nodes, revlabel, asent, a.root).encode('unicode-escape')
	print strtree(b.nodes, revlabel, bsent, b.root).encode('unicode-escape')
	print '\t'.join([''] + ["%2d"%x for x in range(b.len)
		if b.nodes[x].prod != -1]), '\n',
	for m in range(b.len):
		bb = b.nodes[m]
		if bb.prod == -1:
			break
		print '\t', revlabel[bb.label][:3],
	for n in range(a.len):
		aa = a.nodes[n]
		if aa.prod == -1:
			break
		print "\n%2d"%n, revlabel[aa.label][:3],
		for m in range(b.len):
			print '\t',
			if b.nodes[m].prod == -1:
				break
			elif bitmatrix:
				print "1" if TESTBIT(&CST[m * SLOTS], n) else " ",
			else:
				if TESTBIT(&CST[IDX(n, m, b.len, SLOTS)], 0):
					if abitcount(&CST[IDX(n, m, b.len, SLOTS)], SLOTS) - 1:
						print abitcount(&CST[IDX(n, m, b.len, SLOTS)], SLOTS) - 1,
				else:
					print '-',
	print "\ncommon productions:",
	print len({a.nodes[n].prod for n in range(a.len)} &
		{b.nodes[n].prod for n in range(b.len)} - {-1})
	print "found:",
	if bitmatrix:
		print "horz", sum([abitcount(&CST[n * SLOTS], SLOTS) > 0
			for n in range(b.len)]),
		print "vert", sum([any([TESTBIT(&CST[n * SLOTS], m)
			for n in range(b.len)]) for m in range(a.len)]),
		print "both", abitcount(CST, b.len * SLOTS)
	else:
		print sum([any([abitcount(&CST[IDX(n,m,b.len,SLOTS)], SLOTS) > 1
			for m in range(b.len)]) for n in range(a.len)])

def add_lcfrs_rules(tree, sent):
	for a, b in zip(tree.subtrees(),
			lcfrs_productions(tree, sent, frontiers=True)):
		a.prod = b
	return tree

def getprods(trees, prods):
	prods.update((p, n) for n, p in enumerate(sorted({st.prod
		for tree in trees for st in tree[0].subtrees()} - prods.viewkeys())))

def getlabels(trees, labels):
	labels.update((l, n) for n, l in enumerate(sorted({st.node
		for tree in trees for st in tree[0].subtrees()} - labels.viewkeys())))

def nonfrontier(sent):
	def nonfrontierfun(x):
		return isinstance(x[0], Tree) or sent[x[0]] is not None
	return nonfrontierfun
def ispreterminal(n):
	return not isinstance(n[0], Tree)
def tolist(tree, sent):
	""" Convert Tree object to a list of nodes in pre-order traversal,
	except for the terminals, which come last. """
	for a in tree.subtrees(ispreterminal):
		a[0] = Terminal(a[0])
	result = list(tree.subtrees(nonfrontier(sent))) + tree.leaves()
	for n in reversed(range(len(result))):
		a = result[n]
		a.idx = n
		assert isinstance(a, Terminal) or a.node, (
				"labels should be non-empty. "
				"tree: %s\nsubtree: %s\nindex %d, label %r" % (
				tree, a, n, a.node))
	result[0].root = 0
	return result

cpdef list indextrees(Ctrees trees, dict prods):
	""" Create an index from specific productions to trees containing that
	production. Productions are represented as integer IDs, trees are given
	as sets of integer indices. """
	cdef list result = [set() for _ in prods]
	cdef NodeArray a
	cdef int n, m
	for n in range(trees.len):
		a = trees.data[n]
		for m in range(a.len):
			if a.nodes[m].prod >= 0:
				(<set>result[a.nodes[m].prod]).add(n)
			elif a.nodes[m].prod == -1:
				break
	return result

def frontierorterm(x):
	# this expression is true when x is empty,
	# or any of its children is a terminal
	return not all(isinstance(y, Tree) for y in x)

def pathsplit(p):
	return p.rsplit("/", 1) if "/" in p else (".", p)

def getprodid(prods, node):
	return -prods.get(node.prod, -1)
def getctrees(trees, sents, trees2=None, sents2=None):
	""" Return Ctrees object for a list of disc. trees and sentences. """
	from treetransforms import canonicalize
	labels = {}
	prods = {}
	# make deep copies to avoid side effects.
	trees12 = trees = [tolist(add_lcfrs_rules(canonicalize(x.copy(True)), y), y)
					for x, y in zip(trees, sents)]
	if trees2:
		trees2 = [tolist(add_lcfrs_rules(canonicalize(x.copy(True)), y), y)
					for x, y in zip(trees2, sents2)]
		trees12 = trees + trees2
	getlabels(trees12, labels)
	getprods(trees12, prods)
	for tree in trees12:
		root = tree[0]
		# reverse sort so that terminals end up last
		tree.sort(key=partial(getprodid, prods))
		for n, a in enumerate(tree):
			a.idx = n
		tree[0].root = root.idx
	trees = Ctrees(trees, labels, prods)
	if trees2:
		trees2 = Ctrees(trees2, labels, prods)
	revlabel = sorted(labels, key=labels.get)
	treeswithprod = indextrees(trees, prods)
	return dict(trees1=trees, sents1=sents, trees2=trees2, sents2=sents2,
		labels=labels, prods=prods, revlabel=revlabel,
		treeswithprod=treeswithprod)

def readtreebank(treebankfile, labels, prods, sort=True, discontinuous=False,
	limit=0, encoding="utf-8"):
	# this could use BracketCorpusReader or expect trees/sents as input,
	# but this version reads the treebank incrementally which reduces memory
	# requirements.
	if treebankfile is None:
		return None, None
	if discontinuous:
		# no incremental reading w/disc trees
		from treetransforms import canonicalize
		from treebank import NegraCorpusReader
		corpus = NegraCorpusReader(*pathsplit(treebankfile), encoding=encoding)
		trees = corpus.parsed_sents().values()
		sents = corpus.sents().values()
		if limit:
			trees = trees[:limit]
		sents = sents[:limit]
		for tree in trees:
			tree.chomsky_normal_form()
		trees = [tolist(add_lcfrs_rules(canonicalize(x), y), y)
						for x, y in zip(trees, sents)]
		getlabels(trees, labels)
		getprods(trees, prods)
		if sort:
			for tree in trees:
				root = tree[0]
				# reverse sort so that terminals end up last
				tree.sort(key=partial(getprodid, prods))
				for n, a in enumerate(tree):
					a.idx = n
				tree[0].root = root.idx
		trees = Ctrees(trees, labels, prods)
		return trees, sents
	sents = []
	tmp = Tree("TMP", [])
	lines = codecs.open(treebankfile, encoding=encoding).readlines()
	if limit:
		lines = lines[:limit]
	numtrees = len(lines)
	numnodes = sum(a.count(" ") for a in lines) + numtrees
	trees = Ctrees()
	trees.alloc(numtrees, numnodes)
	for m, line in enumerate(lines):
		# create Tree object from string
		try:
			tree = Tree(line)
		except:
			print >> sys.stderr, "malformed line:\n%s\nline %d" % (line, m)
			raise
		sent = tree.leaves()
		# every terminal should have its own preterminal
		introducepreterminals(tree)
		# binarize (could be a command line option)
		tmp[:] = [tree]
		try:
			binarize(tmp)
		except AttributeError:
			print >> sys.stderr, "malformed tree:\n%s\nline %d" % (tree, m)
			raise
		# convert tree to list
		for n, a in enumerate(tree.subtrees(frontierorterm)):
			# fixme: get rid of this, maybe exclude terminals altogether
			if len(a) == 1:
				a[0] = Terminal(n)
			elif len(a) == 0:
				sent.insert(n, None)
				a.append(Terminal(n))
			else:
				raise ValueError("Expected binarized tree "
					"with a preterminal for each terminal.\ntree: %s" %
					tree.pprint(margin=999))
		tree = list(tree.subtrees()) + tree.leaves()
		# collect new labels and productions
		for a in tree:
			if not isinstance(a, Tree):
				continue
			if a.node not in labels:
				labels[a.node] = len(labels)
			a.prod = (a.node, ) + tuple(unicode(x.node
					if isinstance(x, Tree) else sent[x]) for x in a)
			if a.prod not in prods:
				prods[a.prod] = len(prods)
		root = tree[0]
		if sort:
			tree.sort(key=partial(getprodid, prods))
		for n, a in enumerate(tree):
			assert isinstance(a, (Tree, Terminal)), (
				"Expected Tree or Terminal, got: "
				"%s.\nnode: %r\nsent: %r\ntree: %s" % (
				type(a), a, sent, root.pprint()))
			a.idx = n
		tree[0].root = root.idx
		trees.add(tree, labels, prods)
		sents.append(sent)
	return trees, sents

cdef test():
	""" Test some of the bitvector operations """
	cdef UChar SLOTS = 4
	cdef int n, m
	cdef ULong vec[4]
	for n in range(SLOTS):
		vec[n] = 0
	bits = SLOTS * BITSIZE
	print "BITSIZE:", BITSIZE
	print "bits:", bits, "in", SLOTS, "SLOTS"
	for n in range(bits):
		SETBIT(vec, n)
		assert abitcount(vec, SLOTS) == 1, abitcount(vec, SLOTS)
		for m in range(bits):
			if m == n:
				assert TESTBIT(vec, n)
			else:
				assert not TESTBIT(vec, m), (n, m)
		CLEARBIT(vec, n)
		assert abitcount(vec, SLOTS) == 0, abitcount(vec, SLOTS)
		for m in range(bits):
			assert not TESTBIT(vec, n)
	print "bitset test successful"

def main():
	from doctest import testmod, NORMALIZE_WHITESPACE, ELLIPSIS
	test()

	treebank = map(Tree, """\
(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (JJ hungry) (NN dog))))
(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (JJ yellow) (NN cat))))
(S (NP (DT The) (JJ little) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN cat)) (VP (VBP ate) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP ate) (NP (DT the) (NN cat))))\
		""".splitlines())
	for tree in treebank:
		tree.chomsky_normal_form()
	sents = [tree.leaves() for tree in treebank]
	for tree in treebank:
		for n, idx in enumerate(tree.treepositions('leaves')):
			tree[idx] = n
	params = getctrees(treebank, sents)
	treeswithprod = indextrees(params['trees1'], params['prods'])
	fragments = fastextractfragments(params['trees1'], params['sents1'], 0, 0,
			params['revlabel'], discontinuous=True, approx=False)
	counts = exactcounts(params['trees1'], params['trees1'],
			list(fragments.values()), treeswithprod, fast=True)
	print "number of fragments:", len(fragments)
	assert len(fragments) == 25
	for (a, b), c in sorted(zip(fragments, counts)):
		print "%s\t%d" % (re.sub("[0-9]+", lambda x: b[int(x.group())], a), c)
		#print "%s\t%s\t%d" % (a, b, c)

if __name__ == '__main__':
	main()
