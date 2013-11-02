""" Implementation of Sangati et al. (2010), Efficiently extract recurring tree
fragments from large treebanks.
Moschitti (2006): Making Tree Kernels practical for Natural Language
Learning. """

from __future__ import print_function
import re
import io
from collections import defaultdict, Counter as multiset
from functools import partial
from array import array
from discodop.tree import Tree
from discodop.grammar import lcfrs_productions
from discodop.treetransforms import binarize, introducepreterminals

cimport cython
from libc.stdlib cimport malloc, free
from cpython.array cimport array, clone
from discodop.containers cimport ULong, UInt
from discodop.containers cimport Node, NodeArray, Ctrees
from discodop.bit cimport iteratesetbits, abitcount, subset, ulongcpy, \
		ulongset, setunioninplace

cdef extern from "macros.h":
	int BITNSLOTS(int nb)
	void SETBIT(ULong a[], int b)
	void CLEARBIT(ULong a[], int b)
	ULong TESTBIT(ULong a[], int b)
	int IDX(int i, int j, int jmax, int kmax)

# template to create arrays of this type.
cdef array uintarray = array('I', ())

# NB: (?: ) is a non-grouping operator; the second ':' is part of what we match
FRONTIERORTERMRE = re.compile(r' ([0-9]+)(?::[0-9]+)?\b')  # all leaf nodes
TERMINDICESRE = re.compile(r'\([^(]+ ([0-9]+)\)')  # leaf nodes w/term. indices
FRONTIERRE = re.compile(r' ([0-9]+):([0-9]+)\b')  # non-terminal frontiers
LABEL = re.compile(' *\( *([^ ()]+) *')


# bitsets representing fragments are ULong arrays with the number of elements
# determined by SLOTS. While SLOTS is not a constant nor a global, it is
# set just once for a treebank to fit its largest tree.
# After SLOTS elements, this struct follows:
cdef packed struct BitsetTail:
	UInt id  # tree no. of which this fragment was extracted
	short root  # index of the root of the fragment in the NodeArray


# we wrap bitsets in bytes objects, because these can be (1) used in
# dictionaries and (2) passed between processes.
# we leave one byte for NUL-termination:
# data (SLOTS * 8), id (4), root (2), NUL (1)
cdef inline bytes wrap(ULong *data, short SLOTS):
	""" Wrap bitset in bytes object for handling in Python space. """
	return (<char *>data)[:SLOTS * sizeof(ULong) + sizeof(BitsetTail)]


# use getters & setters because a cdef class would incur overhead & indirection
# of a python object, and with a struct the root & id fields must be in front
# which seems to lead to bad hashing behavior (?)
cdef inline ULong *getpointer(object wrapper):
	""" Get pointer to bitset from wrapper. """
	return <ULong *><char *><bytes>wrapper


cdef inline UInt getid(ULong *data, short SLOTS):
	""" Get id of fragment in a bitset. """
	return (<BitsetTail *>&data[SLOTS]).id


cdef inline short getroot(ULong *data, short SLOTS):
	""" Get root of fragment in a bitset. """
	return (<BitsetTail *>&data[SLOTS]).root


cdef inline void setrootid(ULong *data, short root, UInt id, short SLOTS):
	""" Set root and id of fragment in a bitset. """
	cdef BitsetTail *tail = <BitsetTail *>&data[SLOTS]
	tail.id = id
	tail.root = root


cpdef fastextractfragments(Ctrees trees1, list sents1, int offset, int end,
		list labels, Ctrees trees2=None, list sents2=None, bint approx=True,
		bint debug=False, bint discontinuous=False, bint complement=False):
	""" Seeks the largest fragments in treebank(s) with a linear time tree
	kernel

	- scenario 1: recurring fragments in single treebank, use:
		``fastextractfragments(trees1, sents1, offset, end, labels)``
	- scenario 2: common fragments in two treebanks:
		``fastextractfragments(trees1, sents1, offset, end, labels, trees2, sents2)``

	``offset`` and ``end`` can be used to divide the work over multiple
	processes; they are indices of ``trees1`` to work on (default is all); when
	``debug`` is enabled a contingency table is shown for each pair of trees;
	when ``complement`` is true, the complement of the recurring fragments in
	each pair of trees is extracted as well. """
	cdef:
		int n, m, start = 0, end2
		short SLOTS  # the number of bitsets needed to cover the largest tree
		ULong *CST = NULL  # Common Subtree Table
		ULong *scratch, *bitset  # temporary variables
		NodeArray a, b, *ctrees1, *ctrees2
		Node *anodes, *bnodes
		list asent
		dict fragments = {}
		set inter = set()
	if approx:
		fragments = <dict>defaultdict(one)

	if trees2 is None:
		trees2 = trees1
	ctrees1 = trees1.trees
	ctrees2 = trees2.trees
	SLOTS = BITNSLOTS(max(trees1.maxnodes, trees2.maxnodes) + 1)
	CST = <ULong *>malloc(trees2.maxnodes * SLOTS * sizeof(ULong))
	scratch = <ULong *>malloc((SLOTS + 2) * sizeof(ULong))
	assert CST is not NULL and scratch is not NULL
	end2 = trees2.len
	# find recurring fragments
	for n in range(offset, min(end or trees1.len, trees1.len)):
		a = ctrees1[n]
		asent = <list>(sents1[n])
		anodes = &trees1.nodes[a.offset]
		if sents2 is None:
			start = n + 1
		for m in range(start, end2):
			b = ctrees2[m]
			bnodes = &trees2.nodes[b.offset]
			# initialize table
			ulongset(CST, 0UL, b.len * SLOTS)
			# fill table
			fasttreekernel(anodes, bnodes, a.len, b.len, CST, SLOTS)
			# dump table
			if debug:
				print(n, m)
				dumpCST(CST, a, b, anodes, bnodes, asent, (sents2 or sents1)[m],
					labels, SLOTS, True)
			# extract results
			extractbitsets(CST, anodes, bnodes, b.root, n, inter,
					scratch, SLOTS)
			# extract complementary fragments?
			if complement:
				# combine bitsets of inter together with bitwise or
				ulongset(scratch, 0UL, SLOTS)
				for wrapper in inter:
					setunioninplace(scratch, getpointer(wrapper), SLOTS)
				# extract bitsets in A from result, without regard for B
				extractcompbitsets(scratch, anodes, a.root, n, inter,
						SLOTS, NULL)
		# collect string representations of fragments
		for wrapper in inter:
			bitset = getpointer(wrapper)
			if discontinuous:
				frag = getsubtree(anodes, bitset, labels,
						getroot(bitset, SLOTS))
				frag = getsent(frag, asent)
			else:
				frag = getsubtreeunicode(anodes, bitset, labels, asent,
						getroot(bitset, SLOTS))
			if approx:
				fragments[frag] += 1
			elif frag not in fragments:
				fragments[frag] = wrapper
		inter.clear()
	free(CST)
	free(scratch)
	return fragments


cdef inline void fasttreekernel(Node *a, Node *b, int alen, int blen,
		ULong *CST, short SLOTS):
	""" Fast Tree Kernel (average case linear time). Expects trees to be sorted
	according to their productions (in descending order, with terminals as -1).
	Moschitti (2006): Making Tree Kernels practical for Natural Language
	Learning. """
	# i as an index to a, j to b, and jj is a temp index starting at j.
	cdef int i = 0, j = 0, jj = 0
	while i < alen and j < blen:
		if a[i].prod < b[j].prod:
			i += 1
		elif a[i].prod > b[j].prod:
			j += 1
		else:
			while i < alen and a[i].prod == b[j].prod:
				jj = j
				while jj < blen and a[i].prod == b[jj].prod:
					SETBIT(&CST[jj * SLOTS], i)
					jj += 1
				i += 1


cdef inline extractbitsets(ULong *CST, Node *a, Node *b,
		short j, int n, set results, ULong *scratch, short SLOTS):
	""" Visit nodes of 'b' in pre-order traversal. store bitsets of connected
	subsets of 'a' as they are encountered, following the common nodes
	specified in CST. j is the node in 'b' to start with, which changes with
	each recursive step. 'n' is the identifier of this tree which is stored
	with extracted fragments. """
	cdef ULong *bitset = &CST[j * SLOTS]
	cdef ULong cur = bitset[0]
	cdef short idx = 0
	cdef short i = iteratesetbits(bitset, SLOTS, &cur, &idx)
	while i != -1:
		ulongset(scratch, 0UL, SLOTS)
		extractat(CST, scratch, a, b, i, j, SLOTS)
		setrootid(scratch, i, n, SLOTS)
		results.add(wrap(scratch, SLOTS))
		i = iteratesetbits(bitset, SLOTS, &cur, &idx)
	if b[j].left >= 0:
		extractbitsets(CST, a, b, b[j].left, n, results, scratch, SLOTS)
		if b[j].right >= 0:
			extractbitsets(CST, a, b, b[j].right, n, results, scratch, SLOTS)


cdef inline void extractat(ULong *CST, ULong *result, Node *a, Node *b,
		short i, short j, short SLOTS):
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


cpdef exactcounts(Ctrees trees1, Ctrees trees2, list bitsets,
		bint fast=True, bint indices=False):
	""" Given a set of fragments from trees1 as bitsets, find all occurences
	of those fragments in trees2 (which may be equal to trees1).
	By default, exact counts are collected. When indices is True, a multiset
	of indices is returned for each fragment; the indices point to the trees
	where the fragments (described by the bitsets) occur. This is useful to
	look up the contexts of fragments, or when the order of sentences has
	significance which makes it possible to interpret the set of indices as
	time-series data.
	The reason we need to do this separately from extracting maximal fragments
	is that a fragment can occur in other trees where it was not a maximal. """
	cdef:
		array counts = None
		list theindices = None
		object matches = None  # multiset()
		set candidates
		short i, j, SLOTS = BITNSLOTS(max(trees1.maxnodes, trees2.maxnodes) + 1)
		UInt n, m, *countsp = NULL
		NodeArray a, b
		Node *anodes, *bnodes
		ULong cur, *bitset
		short idx
	if indices:
		theindices = [multiset() for _ in bitsets]
	else:
		counts = clone(uintarray, len(bitsets), True)
		countsp = counts.data.as_uints
	# compare one bitset to each tree for each unique fragment.
	for n, wrapper in enumerate(bitsets):
		bitset = getpointer(wrapper)
		a = trees1.trees[getid(bitset, SLOTS)]  # fragment came from this tree
		anodes = &trees1.nodes[a.offset]
		cur, idx = bitset[0], 0
		i = iteratesetbits(bitset, SLOTS, &cur, &idx)
		assert i != -1
		candidates = <set>(trees2.treeswithprod[anodes[i].prod]).copy()
		while True:
			i = iteratesetbits(bitset, SLOTS, &cur, &idx)
			if i == -1 or i >= a.len:  # FIXME. why is 2nd condition necessary?
				break
			candidates &= <set>(trees2.treeswithprod[anodes[i].prod])
		i = getroot(bitset, SLOTS)  # root of fragment in tree 'a'
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
				elif fast and anodes[i].prod < bnodes[j].prod:
					break
	return theindices if indices else counts


cdef inline int containsbitset(Node *a, Node *b, ULong *bitset,
		short i, short j):
	""" Recursively check whether fragment starting from a[i] described by
	bitset is equal to b[j], i.e., whether b contains that fragment from a. """
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


cpdef dict coverbitsets(Ctrees trees, list sents, list labels,
		short maxnodes, bint discontinuous):
	""" Utility function to generate one bitset for each type of production.
	Important: if multiple treebanks are used, maxnodes should equal
	``max(trees1.maxnodes, trees2.maxnodes)`` """
	cdef:
		dict result = {}
		int p, i, n = -1
		short SLOTS = BITNSLOTS(maxnodes + 1)
		ULong *scratch = <ULong *>malloc((SLOTS + 2) * sizeof(ULong))
		Node *nodes
	assert scratch is not NULL
	assert SLOTS, SLOTS
	for p, treeindices in enumerate(trees.treeswithprod):
		ulongset(scratch, 0UL, SLOTS)
		# slightly convoluted way of getting an arbitrary set member:
		n = next(iter(treeindices))
		nodes = &trees.nodes[trees.trees[n].offset]
		for i in range(trees.trees[n].len):
			if nodes[i].prod == p:
				SETBIT(scratch, i)
				setrootid(scratch, i, n, SLOTS)
				break
		else:
			raise ValueError("production not found. wrong index?")
		if discontinuous:
			frag = getsubtree(nodes, scratch, labels, i)
			frag = getsent(frag, sents[n])
		else:
			frag = getsubtreeunicode(nodes, scratch, labels,
					sents[n], i)
		result[frag] = wrap(scratch, SLOTS)
	return result


cpdef dict completebitsets(Ctrees trees, list sents, list labels,
		short maxnodes, bint discontinuous=False):
	""" Utility function to generate bitsets corresponding to whole trees
	in the input.
	Important: if multiple treebanks are used, maxnodes should equal
	``max(trees1.maxnodes, trees2.maxnodes)`` """
	cdef:
		dict result = {}
		list sent
		int n, i
		short SLOTS = BITNSLOTS(maxnodes + 1)
		ULong *scratch = <ULong *>malloc((SLOTS + 2) * sizeof(ULong))
		Node *nodes
	for n in range(trees.len):
		ulongset(scratch, 0UL, SLOTS)
		nodes = &trees.nodes[trees.trees[n].offset]
		sent = sents[n]
		for i in range(trees.trees[n].len):
			if nodes[i].left >= 0 or sent[termidx(nodes[i].left)] is not None:
				SETBIT(scratch, i)
		setrootid(scratch, trees.trees[n].root, n, SLOTS)
		if discontinuous:
			frag = (strtree(nodes, labels, trees.trees[n].root), tuple(sent))
		else:
			frag = unicodetree(nodes, labels, sent, trees.trees[n].root)
		result[frag] = wrap(scratch, SLOTS)
	return result


cdef inline void extractcompbitsets(ULong *bitset, Node *a,
		int i, int n, set results, short SLOTS, ULong *scratch):
	""" Visit nodes of 'a' in pre-order traversal. store bitsets of connected
	subsets of 'a' as they are encountered, following the nodes specified in
	the complement of 'bitset'. i is the node in 'a' to start with, which
	changes with each recursive step. 'n' is the identifier of this tree which
	is stored with extracted fragments. """
	cdef bint start = scratch is NULL and not TESTBIT(bitset, i)
	if start:  # start extracting a fragment
		# need to create a new array because further down in the recursion
		# other fragments may be encountered which should not overwrite
		# this one
		scratch = <ULong *>malloc((SLOTS + 2) * sizeof(ULong))
		assert scratch is not NULL
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


cpdef extractfragments(Ctrees trees1, list sents1, int offset, int end,
		list labels, Ctrees trees2=None, list sents2=None, bint approx=True,
		bint debug=False, bint discontinuous=False):
	""" Seeks the largest fragments in treebank(s)

	- scenario 1: recurring fragments in single treebank, use:
		``fastextractfragments(trees1, sents1, offset, end)``
	- scenario 2: common fragments in two treebanks:
		``fastextractfragments(trees1, sents1, offset, end, trees2, sents2)``

	``offset`` and ``end`` can be used to divide the work over multiple
	processes; they are indices of ``trees1`` to work on (default is all); when
	``debug`` is enabled a contingency table is shown for each pair of trees;
	when ``complement`` is true, the complement of the recurring fragments in
	each pair of trees is extracted as well. """
	cdef:
		int n, m, aa, bb, start = 0
		short SLOTS
		ULong *CST, *scratch, *bitset
		NodeArray *ctrees1, *ctrees2, a, b
		Node *anodes, *bnodes
		list asent
		dict fragments = {}
		set inter = set()

	if approx:
		fragments = <dict>defaultdict(one)
	if trees2 is None:
		trees2 = trees1
	ctrees1 = trees1.trees
	ctrees2 = trees2.trees
	SLOTS = BITNSLOTS(max(trees1.maxnodes, trees2.maxnodes) + 1)
	CST = <ULong *>malloc(trees1.maxnodes * trees2.maxnodes
		* SLOTS * sizeof(ULong))
	scratch = <ULong *>malloc((SLOTS + 2) * sizeof(ULong))
	assert CST is not NULL and scratch is not NULL

	# find recurring fragments
	for n in range(offset, (end or trees1.len)):
		a = ctrees1[n]
		anodes = &trees1.nodes[a.offset]
		asent = <list>(sents1[n])
		if sents2 is None:
			start = n + 1
		for m in range(start, trees2.len):
			b = ctrees2[m]
			bnodes = &trees2.nodes[b.offset]
			# initialize table
			ulongset(CST, 0UL, a.len * b.len * SLOTS)
			# fill table
			for aa in range(a.len):
				for bb in range(b.len):
					if not TESTBIT(&CST[IDX(aa, bb, b.len, SLOTS)], a.len):
						getCST(anodes, bnodes, a.len, b.len, CST, aa, bb, SLOTS)
			# dump table
			if debug:
				print(n, m)
				dumpCST(CST, a, b, anodes, bnodes, asent,
					(sents2[m] if sents2 else sents1[m]), labels, SLOTS)
			# extract results
			inter.update(getnodeset(CST, a.len, b.len, n, scratch, SLOTS))
		for wrapper in inter:
			bitset = getpointer(wrapper)
			if discontinuous:
				frag = getsubtree(anodes, bitset, labels, getroot(bitset, SLOTS))
				frag = getsent(frag, asent)
			else:
				frag = getsubtreeunicode(anodes, bitset, labels, asent,
						getroot(bitset, SLOTS))
			if approx:
				fragments[frag] += 1
			else:
				fragments[frag] = wrapper
		inter.clear()
	free(CST)
	return fragments


cdef inline void getCST(Node *a, Node *b, int alen, int blen, ULong *CST,
		int i, int j, short SLOTS):
	""" Recursively build common subtree table (CST) for subtrees a[i], b[j]. """
	cdef ULong *child, *bitset = &CST[IDX(i, j, blen, SLOTS)]
	SETBIT(bitset, alen)  # mark cell as visited
	# compare label & arity / terminal; assume presence of arity markers.
	if a[i].prod == b[j].prod:
		SETBIT(bitset, i)
		# lexical production, no recursion
		if a[i].left < 0:
			return
		# normal production, recurse or use cached value
		child = &CST[IDX(a[i].left, b[j].left, blen, SLOTS)]
		if not TESTBIT(child, alen):
			getCST(a, b, alen, blen, CST, a[i].left, b[j].left, SLOTS)
		for n in range(SLOTS):
			bitset[n] |= child[n]
		if a[i].right < 0:  # and b[j].right < 0: # unary node
			return
		child = &CST[IDX(a[i].right, b[j].right, blen, SLOTS)]
		if not TESTBIT(child, alen):
			getCST(a, b, alen, blen, CST, a[i].right, b[j].right, SLOTS)
		for n in range(SLOTS):
			bitset[n] |= child[n]


cdef inline set getnodeset(ULong *CST, int alen, int blen, int n,
		ULong *scratch, short SLOTS):
	""" Extract the largest, connected bitsets from the common subtree
	table (CST). """
	cdef:
		ULong *bitset
		int i, j
		set finalnodeset = set()
	for i in range(alen):
		for j in range(blen):
			bitset = &CST[IDX(i, j, blen, SLOTS)]
			if not TESTBIT(bitset, alen) or abitcount(bitset, SLOTS) < 2:
				continue
			ulongcpy(scratch, bitset, SLOTS)
			CLEARBIT(scratch, alen)
			setrootid(scratch, i, n, SLOTS)
			tmp = wrap(scratch, SLOTS)
			if tmp in finalnodeset:
				continue
			for bs in finalnodeset:
				if subset(scratch, getpointer(bs), SLOTS):
					break
				elif subset(getpointer(bs), scratch, SLOTS):
					finalnodeset.remove(bs)
					finalnodeset.add(tmp)
					break
			else:
				# completely new (disjunct) bitset
				finalnodeset.add(tmp)
	return finalnodeset


cdef inline short termidx(x):
	""" Translate representation for terminal indices. """
	return -x - 1


cdef inline str strtree(Node *tree, list labels, int i):
	""" produce string representation of (complete) tree, with indices as
	terminals. """
	if tree[i].left < 0:
		return "(%s %d)" % (labels[tree[i].prod], termidx(tree[i].left))
	elif tree[i].right < 0:
		return "(%s %s)" % (labels[tree[i].prod],
				strtree(tree, labels, tree[i].left))
	return "(%s %s %s)" % (labels[tree[i].prod],
			strtree(tree, labels, tree[i].left),
			strtree(tree, labels, tree[i].right))


cdef inline unicode unicodetree(Node *tree, list labels, list sent, int i):
	""" produce string representation of (complete) tree, with words as
	terminals. """
	if tree[i].left < 0:
		return u"(%s %s)" % (labels[tree[i].prod],
				(u"%d" % termidx(tree[i].left)
				if sent is None else
				sent[termidx(tree[i].left)] or u''))
	elif tree[i].right < 0:
		return u"(%s %s)" % (labels[tree[i].prod],
				unicodetree(tree, labels, sent, tree[i].left))
	return u"(%s %s %s)" % (labels[tree[i].prod],
			unicodetree(tree, labels, sent, tree[i].left),
			unicodetree(tree, labels, sent, tree[i].right))


cdef inline str getsubtree(Node *tree, ULong *bitset, list labels, int i):
	""" Turn bitset into string representation of tree,
	with indices as terminals. """
	if TESTBIT(bitset, i):
		if tree[i].left < 0:
			return "(%s %d)" % (labels[tree[i].prod], termidx(tree[i].left))
		elif tree[i].right < 0:
			return "(%s %s)" % (labels[tree[i].prod],
					getsubtree(tree, bitset, labels, tree[i].left))
		return "(%s %s %s)" % (labels[tree[i].prod],
				getsubtree(tree, bitset, labels, tree[i].left),
				getsubtree(tree, bitset, labels, tree[i].right))
	# node not in bitset, frontier non-terminal
	return "(%s %s)" % (labels[tree[i].prod], yieldranges(tree, i))


cdef inline unicode getsubtreeunicode(Node *tree, ULong *bitset, list labels,
		list sent, int i):
	""" Turn bitset into string representation of tree,
	with words as terminals. """
	if TESTBIT(bitset, i):
		if tree[i].left < 0:
			return u"(%s %s)" % (labels[tree[i].prod],
				(u"%d" % termidx(tree[i].left) if sent is None
				else sent[termidx(tree[i].left)]))
		elif tree[i].right < 0:
			return u"(%s %s)" % (labels[tree[i].prod],
				getsubtreeunicode(tree, bitset, labels, sent, tree[i].left))
		return u"(%s %s %s)" % (labels[tree[i].prod],
			getsubtreeunicode(tree, bitset, labels, sent, tree[i].left),
			getsubtreeunicode(tree, bitset, labels, sent, tree[i].right))
	# node not in bitset, frontier non-terminal
	return u"(%s )" % labels[tree[i].prod]


cdef inline yieldranges(Node *tree, int i):
	""" For discontinuous trees, return a string with the intervals of indices
	corresponding to the components in the yield of a node.
	The intervals are of the form start:end, where `end` is part of the
	interval. e.g., "0:1 2:4" corresponds to (0, 1) and (2, 3, 4). """
	cdef list yields = [], leaves = sorted(getyield(tree, i))
	cdef int a, start = -2, prev = -2
	for a in leaves:
		if a - 1 != prev:
			if prev != -2:
				yields.append("%d:%d" % (start, prev))
			start = a
		prev = a
	yields.append("%d:%d" % (start, prev))
	return ' '.join(yields)


cdef inline list getyield(Node *tree, int i):
	""" Recursively collect indices of terminals under a node. """
	if tree[i].left < 0:
		return [termidx(tree[i].left)]
	elif tree[i].right < 0:
		return getyield(tree, tree[i].left)
	return getyield(tree, tree[i].left) + getyield(tree, tree[i].right)


def repl(d):
	def f(x):
		return d[int(x.group(1))]
	return f


cdef getsent(frag, list sent):
	""" Select the words that occur in the fragment and renumber terminals in
	fragment such that the first index is 0 and any gaps have a width of 1.
	Expects a tree as string where frontier nodes are marked with intervals.

	>>> getsent('(S (NP 2) (VP 4))', ['The', 'tall', 'man', 'there', 'walks'])
	('(S (NP 0) (VP 2))', ('man', None, 'walks'))
	>>> getsent('(VP (VB 0) (PRT 3))', ['Wake', 'your', 'friend', 'up'])
	('(VP (VB 0) (PRT 2))', ('Wake', None, 'up'))
	>>> getsent('(S (NP 2:2 4:4) (VP 1:1 3:3))',['Walks','the','quickly','man'])
	('(S (NP 1 3) (VP 0 2))', (None, None, None, None))
	>>> getsent('(ROOT (S 0:2) ($. 3))', ['Foo', 'bar', 'zed', '.'])
	('(ROOT (S 0) ($. 1))', (None, '.'))
	>>> getsent('(ROOT (S 0) ($. 3))', ['Foo', 'bar', 'zed','.'])
	('(ROOT (S 0) ($. 2))', ('Foo', None, '.'))
	>>> getsent('(S|<VP>_2 (VP_3 0:1 3:3 16:16) (VAFIN 2))', '''In Japan wird
	...  offenbar die Fusion der Geldkonzerne Daiwa und Sumitomo zur
	...  gr\\xf6\\xdften Bank der Welt vorbereitet .'''.split(' '))
	('(S|<VP>_2 (VP_3 0 2 4) (VAFIN 1))', (None, 'wird', None, None, None)) """
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
		if n in leaves:
			newsent.append(sent[n])  # a terminal
		else:
			newsent.append(None)  # a frontier node
		leafmap[n] = " %d" % m
		m += 1
		if spans[n] not in spans and n != maxl:  # a gap
			newsent.append(None)
			m += 1
	frag = FRONTIERORTERMRE.sub(repl(leafmap), frag)
	return frag, tuple(newsent)


cdef dumptree(NodeArray a, Node *anodes, list asent, list labels):
	""" dump the node structs of a tree showing numeric IDs as well
	as a strings representation of the tree in bracket notation. """
	cdef int n
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
	print(unicodetree(anodes, labels, asent, a.root), '\n', asent, '\n')


cdef dumpCST(ULong *CST, NodeArray a, NodeArray b, Node *anodes, Node *bnodes,
		list asent, list bsent, list labels, short SLOTS, bint bitmatrix=False):
	""" dump a table of the common subtrees of two trees. """
	cdef int n, m
	cdef Node aa, bb
	dumptree(a, anodes, asent, labels)
	dumptree(b, bnodes, bsent, labels)
	print('\t'.join([''] + ["%2d" % x for x in range(b.len)
		if bnodes[x].prod != -1]))
	for m in range(b.len):
		bb = bnodes[m]
		print('\t', labels[bb.prod][:3], end='')
	for n in range(a.len):
		aa = anodes[n]
		print("\n%2d" % n, labels[aa.prod][:3], end='')
		for m in range(b.len):
			print('\t', end='')
			if bitmatrix:
				print("1" if TESTBIT(&CST[m * SLOTS], n) else ' ', end='')
			else:
				if TESTBIT(&CST[IDX(n, m, b.len, SLOTS)], a.len):
					if abitcount(&CST[IDX(n, m, b.len, SLOTS)], SLOTS) - 1:
						print(abitcount(&CST[IDX(n, m, b.len, SLOTS)],
								SLOTS) - 1, end='')
				else:
					print('-', end='')
	print("\ncommon productions:", end='')
	print(len({anodes[n].prod for n in range(a.len)} &
			{bnodes[n].prod for n in range(b.len)}))
	print("found:", end='')
	if bitmatrix:
		print("horz", sum([abitcount(&CST[n * SLOTS], SLOTS) > 0
				for n in range(b.len)]),
			"vert", sum([any([TESTBIT(&CST[n * SLOTS], m)
				for n in range(b.len)]) for m in range(a.len)]),
			"both", abitcount(CST, b.len * SLOTS))
	else:
		print(sum([any([abitcount(&CST[IDX(n, m, b.len, SLOTS)], SLOTS) > 1
			for m in range(b.len)]) for n in range(a.len)]))


def add_lcfrs_rules(tree, sent):
	""" Set attribute ``.prod`` on nodes of tree with the LCFRS production for
	that node. """
	for a, b in zip(tree.subtrees(),
			lcfrs_productions(tree, sent, frontiers=True)):
		a.prod = b
	return tree


def getlabelsprods(trees, labels, prods):
	""" collect label/prod attributes from ``trees`` and index them. """
	pnum = len(prods)
	for tree in trees:
		for st in tree:
			if st.prod not in prods:
				labels.append(st.label)  # LHS label associated with this prod
				prods[st.prod] = pnum
				pnum += 1


def nonfrontier(sent):
	def nonfrontierfun(x):
		return isinstance(x[0], Tree) or sent[x[0]] is not None
	return nonfrontierfun


def tolist(tree, sent):
	""" Convert Tree object to list of non-terminal nodes in pre-order
	traversal; add indices to nodes reflecting their position in the list. """
	result = list(tree.subtrees(nonfrontier(sent)))
	for n in reversed(range(len(result))):
		a = result[n]
		a.idx = n
		assert a.label, ("labels should be non-empty. tree: "
				"%s\nsubtree: %s\nindex %d, label %r" % (tree, a, n, a.label))
	result[0].rootidx = 0
	return result


def pathsplit(p):
	return p.rsplit('/', 1) if '/' in p else ('.', p)


def getprodid(prods, node):
	return prods.get(node.prod, -1)


def getctrees(trees, sents, trees2=None, sents2=None):
	""" :returns: Ctrees object for disc. binary trees and sentences. """
	# make deep copies to avoid side effects.
	trees12 = trees = [tolist(add_lcfrs_rules(Tree.convert(a), b), b)
			for a, b in zip(trees, sents)]
	if trees2:
		trees2 = [tolist(add_lcfrs_rules(Tree.convert(a), b), b)
					for a, b in zip(trees2, sents2)]
		trees12 = trees + trees2
	labels = []
	prods = {}
	getlabelsprods(trees12, labels, prods)
	for tree in trees12:
		root = tree[0]
		tree.sort(key=partial(getprodid, prods))
		for n, a in enumerate(tree):
			a.idx = n
		tree[0].rootidx = root.idx
	trees = Ctrees(trees, prods)
	trees.indextrees(prods)
	if trees2:
		trees2 = Ctrees(trees2, prods)
		trees2.indextrees(prods)
	return dict(trees1=trees, sents1=sents, trees2=trees2, sents2=sents2,
			prods=prods, labels=labels)


cdef readnode(line, list labels, dict prods, Node *result,
		size_t *idx, list sent, origlabel):
	""" Parse an s-expression in a string, and store in an array of Node
	structs (pre-allocated). ``idx`` is a counter to keep track of the number
	of Node structs used; ``sent`` collects the terminals encountered. """
	cdef:
		int n, parens = 0, start = 0, startchild2 = 0, left = -1, right = -1
		list childlabels = None
		#bytes a, label, rest, labelchild1, labelchild2
		#bytes child1 = None, child2 = None
	match = LABEL.match(line)
	if match is None:
		raise ValueError('malformed tree:\n%s' % line)
	label = match.group(1)
	rest = line[match.end():]
	child1 = child2 = labelchild2 = None
	# walk through the string and find the first two children
	for n, a in enumerate(rest):
		if a == '(':
			if parens == 0:
				start = n
			parens += 1
		elif a == ')':
			parens -= 1
			if parens == -1:
				if child1 is None:
					child1 = rest[start:n]
			elif parens == 0:
				if child1 is None:
					child1 = rest[start:n + 1]
				elif child2 is None:
					child2 = rest[start:n + 1]
					startchild2 = start
				else:  # will do on the fly binarization
					childlabels = []
					break
	# if there were more children, collect labels for a binarized constituent
	if childlabels is not None:
		for n, a in enumerate(rest[startchild2:], startchild2):
			if a == '(':
				if parens == 0:
					start = n
				parens += 1
			elif a == ')':
				parens -= 1
				if parens == 0:
					match = LABEL.match(rest, start)
					if match is None:  # introduce preterminal
						childlabels.append('/'.join((label, rest[start:n + 1])))
					else:
						childlabels.append(match.group(1))
		# not optimal, parsing this string can be avoided
		# in fact we already have: label, child2, rest
		child2 = ('(' + (origlabel or label) + '|<' + ','.join(childlabels)
				+ '> ' + rest[startchild2:])
	assert parens == -1, "unbalanced parentheses: %d\n%r" % (parens, line)
	match1 = LABEL.match(child1)
	if match1 is not None:  # non-terminal label
		labelchild1 = match1.group(1)
	elif child2 is not None:  # insert preterminal
		labelchild1 = '/'.join((label, child1))
		child1 = '(' + ' '.join((labelchild1, child1)) + ')'
	else:  # terminal following pre-terminal; store terminal
		# leading space distinguishes terminal from non-terminal
		labelchild1 = ' ' + child1
		left = termidx(len(sent))
		sent.append(child1 or None)
	if child2 is None:
		prod = (label, labelchild1) if child1 else (label, )
	else:
		match = LABEL.match(child2)
		if match is not None:
			labelchild2 = match.group(1)
		else:  # insert preterminal
			labelchild2 = '/'.join((label, child2))
			child2 = '(' + ' '.join((labelchild2, child2)) + ')'
		prod = (label, labelchild1, labelchild2)
	if prod not in prods:  # assign new ID?
		prods[prod] = len(prods)
		labels.append(label)
	n = idx[0]
	idx[0] += 1
	if match1 is not None or child2 is not None:
		left = idx[0]
		readnode(child1, labels, prods, result, idx, sent, None)
		if child2 is not None:
			right = idx[0]
			readnode(child2, labels, prods, result, idx, sent,
					childlabels and (origlabel or label))
	# store node
	result[n].prod = prods[prod]
	result[n].left = left
	result[n].right = right


def readtreebank(treebankfile, list labels, dict prods, bint sort=True,
		fmt='bracket', limit=None, encoding="utf-8"):
	""" Read a treebank from a given filename. labels and prods should be a
	list and a dictionary, with the same ones used when reading multiple
	treebanks. """
	# this could use BracketCorpusReader or expect trees/sents as input,
	# but this version reads the treebank incrementally which reduces memory
	# requirements.
	cdef size_t cnt
	cdef Node *scratch
	cdef Ctrees ctrees
	if treebankfile is None:
		return None, None
	if fmt != 'bracket':
		from itertools import islice
		from discodop.treebank import READERS
		from discodop.treetransforms import canonicalize
		corpus = READERS[fmt](*pathsplit(treebankfile), encoding=encoding)
		ctrees = Ctrees()
		ctrees.alloc(512, 512 * 512)  # dummy values, array will be realloc'd
		sents = []
		for _, tree, sent in islice(corpus.parsed_sents_iter(), limit):
			tree = tolist(add_lcfrs_rules(
					canonicalize(binarize(tree)), sent), sent)
			for st in tree:
				if st.prod not in prods:
					labels.append(st.label)
					prods[st.prod] = len(prods)
			if sort:
				root = tree[0]
				tree.sort(key=partial(getprodid, prods))
				for n, a in enumerate(tree):
					a.idx = n
				tree[0].rootidx = root.idx
			ctrees.add(tree, prods)
			sents.append(sent)
	else:  # do incremental reading of bracket trees
		data = io.open(treebankfile, encoding=encoding).read()
		if limit:
			data = re.match(r'(?:[^\n\r]+[\n\r]+){0,%d}' % limit, data).group()
		lines = [line for line in data.splitlines() if line.strip()]
		numtrees = len(lines)
		assert numtrees, "%r appears to be empty" % treebankfile
		sents = []
		nodesperline = [line.count('(') for line in lines]
		numnodes = sum(nodesperline)
		ctrees = Ctrees()
		ctrees.alloc(numtrees, numnodes)
		ctrees.maxnodes = max(nodesperline)
		binfactor = 2  # conservative estimate to accommodate binarization
		scratch = <Node *>malloc(ctrees.maxnodes * binfactor * sizeof(Node))
		assert scratch is not NULL
		for line in lines:
			cnt = 0
			sent = []
			readnode(line, labels, prods, scratch, &cnt, sent, None)
			ctrees.addnodes(scratch, cnt, 0)
			sents.append(sent)
		free(scratch)
	return ctrees, sents


def one():
	""" used to generate the value 1 for defaultdicts. """
	return 1


def test():
	treebank = [binarize(Tree(x)) for x in """\
(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (JJ hungry) (NN dog))))
(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (JJ yellow) (NN cat))))
(S (NP (DT The) (JJ little) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN cat)) (VP (VBP ate) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP ate) (NP (DT the) (NN cat))))\
		""".splitlines()]
	sents = [tree.leaves() for tree in treebank]
	for tree in treebank:
		for n, idx in enumerate(tree.treepositions('leaves')):
			tree[idx] = n
	params = getctrees(treebank, sents)
	fragments = fastextractfragments(params['trees1'], params['sents1'], 0, 0,
			params['labels'], discontinuous=True, approx=False)
	counts = exactcounts(params['trees1'], params['trees1'],
			list(fragments.values()), fast=True)
	assert len(fragments) == 25
	for (a, b), c in sorted(zip(fragments, counts), key=repr):
		print("%s\t%d" % (re.sub("[0-9]+", lambda x: b[int(x.group())], a), c))
