""" Implementation of Sangati et al. (2010), Efficiently extract recurring tree
fragments from large treebanks.
Moschitti (2006): Making Tree Kernels practical for Natural Language
Learning. """

import re, codecs
from collections import defaultdict
from itertools import count
from array import array
from nltk import Tree
from grammar import srcg_productions, alpha_normalize, freeze
from containers import Terminal

# these arrays are never used but serve as template to create
# arrays of this type.
cdef array uintarray = array("I", []), ulongarray = array("L", [])

cpdef extractfragments(Ctrees trees1, list sents1, int offset, int end,
	dict labels, dict prods, list revlabel,
	Ctrees trees2=None, list sents2=None, bint approx=True,
	bint debug=False, bint discontinuous=False):
	""" Seeks the largest fragments in treebank(s)
	- scenario 1: recurring fragments in single treebank, use:
		extractfragments(trees1, sents1, offset, end, labels, prods, None, None)
	- scenario 2: common fragments in two treebanks:
		extractfragments(trees1, sents1, offset, end, labels, prods,
			trees2, sents2)
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
		FrozenArray pyarray
		list asent
		dict fragments = {}
		set inter = set()

	if approx: fragments = <dict>defaultdict(one)

	if trees2 is None: trees2 = trees1
	ctrees1 = trees1.data
	ctrees2 = trees2.data
	SLOTS = BITNSLOTS(max(trees1.maxnodes, trees2.maxnodes))
	CST = <ULong *>malloc(trees1.maxnodes * trees2.maxnodes
		* SLOTS * sizeof(ULong))
	assert CST is not NULL

	# find recurring fragments
	for n in range(offset, (end or trees1.len)):
		a = ctrees1[n]; asent = <list>(sents1[n])
		if sents2 is None: start = n + 1
		for m in range(start, trees2.len):
			b = ctrees2[m]
			# initialize table
			ulongset(CST, 0UL, a.len * b.len * SLOTS)
			# fill table
			for aa in range(a.len): # skip terminals
				if a.nodes[aa].prod == -1: break
				for bb in range(b.len): #skip terminals
					if b.nodes[bb].prod == -1: break
					elif not TESTBIT(&CST[IDX(aa, bb, b.len, SLOTS)], 0):
						getCST(a, b, CST, aa, bb, SLOTS)
			# dump table
			if debug:
				print n, m
				dumpCST(CST, a, b, asent,
					(sents2[m] if sents2 else sents1[m]), revlabel, SLOTS)
			# extract results
			inter.update(getnodeset(CST, a.len, b.len, n, SLOTS))
		for pyarray in inter:
			frag = getsubtree(a.nodes, pyarray.data._L, revlabel,
				None if discontinuous else asent, pyarray.data._L[SLOTS])
			if discontinuous: frag = getsent(frag, asent)
			if approx: fragments[frag] += 1
			else: fragments[frag] = pyarray.data
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
		if A.nodes[a.left].prod == -1: return
		# normal production, recurse or use cached value
		child = &CST[IDX(a.left, b.left, B.len, SLOTS)]
		if not child[0] & 1: getCST(A, B, CST, a.left, b.left, SLOTS)
		for n in range(SLOTS): bitset[n] |= child[n]
		if a.right != -1: # and b.right != -1:	#sentinel node
			child = &CST[IDX(a.right, b.right, B.len, SLOTS)]
			if not child[0] & 1: getCST(A, B, CST, a.right, b.right, SLOTS)
			for n in range(SLOTS): bitset[n] |= child[n]

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
			ulongcpy(tmp.data._L, bitset, SLOTS)
			if tmp in finalnodeset: continue
			for bs in finalnodeset:
				if subset(bitset, bs.data._L, SLOTS): break
				elif subset(bs.data._L, bitset, SLOTS):
					pyarray = clone(ulongarray, SLOTS + 2, False)
					ulongcpy(pyarray._L, bitset, SLOTS)
					pyarray._L[SLOTS] = n
					pyarray._L[SLOTS + 1] = t
					finalnodeset.add(new_FrozenArray(pyarray))
					finalnodeset.discard(bs)
					break
			else:
				# completely new (disjunct) bitset
				pyarray = clone(ulongarray, SLOTS + 2, False)
				ulongcpy(pyarray._L, bitset, SLOTS)
				pyarray._L[SLOTS] = n
				pyarray._L[SLOTS + 1] = t
				finalnodeset.add(new_FrozenArray(pyarray))
	return finalnodeset

cdef void shiftright(ULong *bitset, UChar SLOTS):
	""" Shift an array of bitsets one bit to the right. """
	cdef int x
	cdef ULong mask = (1 << 1) - 1
	bitset[0] >>= 1
	for x in range(1, SLOTS):
		bitset[x-1] |= bitset[x] & mask
		bitset[x] >>= 1

cdef inline unicode strtree(Node *tree, list revlabel, list sent, int i):
	""" produce string representation of (complete) tree. """
	if tree[i].prod == -1:
		if sent is None: return unicode(tree[i].label)
		return "" if sent[tree[i].label] is None else sent[tree[i].label]
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
	elif tree[i].prod == -1:
		return unicode(tree[i].label) if sent is None else sent[tree[i].label]
	elif sent is None:
		return u"(%s %s)" % (revlabel[tree[i].label], yieldheads(tree, sent, i))
	else: return u"(%s )" % (revlabel[tree[i].label])

cdef inline unicode yieldheads(Node *tree, list sent, int i):
	y = getyield(tree, sent, i)
	return " ".join([unicode(a) for a in sorted(y) if a - 1 not in y])

cdef inline list getyield(Node *tree, list sent, int i):
	if tree[i].prod == -1: return [tree[i].label]
	elif tree[i].left < 0: return [] #??
	elif tree[i].right < 0: return getyield(tree, sent, tree[i].left)
	else: return (getyield(tree, sent, tree[i].left)
		+ getyield(tree, sent, tree[i].right))

# match all leaf nodes containing indices
# by requiring a preceding space, only terminals are matched
# lookahead to ensure we match whole tokens ("23" instead of 2 or 3)
termsre = re.compile(r" ([0-9]+)(?=[ )])")
def repl(d):
	def f(x): return d[int(x.groups()[0])]
	return f

def getsent(frag, list sent):
	""" Select the words that occur in the fragment and renumber terminals
	in fragment. Expects a tree as string.
	>>> getsent("(S (NP 0) (VP 2))", ['The', 'man', 'walks'])
	("(S (NP 0) (VP 2))", [None, 'walks'])
	"""
	cdef int x = 0, n, maxl
	cdef list newsent = []
	cdef dict leafmap = {}
	leaves = set(int(x) for x in termsre.findall(frag))
	if not leaves: return frag, ()
	maxl = max(leaves)
	for n in sorted(leaves):
		leafmap[n] = " " + unicode(x)
		newsent.append(sent[n])
		x += 1
		if n + 1 not in leaves and n != maxl:
			leafmap[n+1] = " " + unicode(x)
			newsent.append(None)
			x += 1
	frag = termsre.sub(repl(leafmap), frag)
	return frag, tuple(newsent)

cdef dumptree(NodeArray a, list revlabel):
	""" print a human-readable representation of a tree struct. """
	cdef int n
	for n in range(a.len):
		if a.nodes[n].prod == -1: break #skip terminals
		print "idx=%2d\tleft=%2d\tright=%2d\tprod=%2d\tlabel=%2d=%s" % (
			n, a.nodes[n].left, a.nodes[n].right, a.nodes[n].prod,
			a.nodes[n].label, revlabel[a.nodes[n].label])
	print

cdef dumpCST(ULong *CST, NodeArray a, NodeArray b, list asent, list bsent,
	list revlabel, UChar SLOTS, bint bitmatrix=False):
	""" print a table of the common subtrees of two trees. """
	cdef int n, m
	cdef Node aa, bb
	dumptree(a, revlabel); dumptree(b, revlabel)
	print strtree(a.nodes, revlabel, asent, a.root)
	print strtree(b.nodes, revlabel, bsent, b.root)
	print '\t'.join([''] + ["%2d"%x for x in range(b.len)
		if b.nodes[x].prod != -1]), '\n',
	for m in range(b.len):
		bb = b.nodes[m]
		if bb.prod == -1: break
		print '\t', revlabel[bb.label][:3],
	for n in range(a.len):
		aa = a.nodes[n]
		if aa.prod == -1: break
		print "\n%2d"%n, revlabel[aa.label][:3],
		for m in range(b.len):
			print '\t',
			if b.nodes[m].prod == -1: break
			elif bitmatrix:
				print "1" if TESTBIT(&CST[m * SLOTS], n) else " ",
			else:
				if TESTBIT(&CST[IDX(n,m,b.len,SLOTS)], 0):
					if abitcount(&CST[IDX(n,m,b.len,SLOTS)], SLOTS) - 1:
						print abitcount(&CST[IDX(n,m,b.len,SLOTS)], SLOTS) - 1,
				else: print '-',
	print "\ncommon productions:",
	print len(set([a.nodes[n].prod for n in range(a.len)]) &
		set([b.nodes[n].prod for n in range(b.len)]) - set([-1]))
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

def add_cfg_rules(tree):
	for a, b in zip(tree.subtrees(), tree.productions()):
		a.prod = (b.lhs().symbol(),) + tuple(unicode(x) for x in b.rhs())
	return tree

def add_srcg_rules(tree, sent):
	for a, b in zip(tree.subtrees(), srcg_productions(tree, sent, False)):
		a.prod = freeze(alpha_normalize(b))
	return tree

def getprods(trees, prods):
	prods.update((p, n) for n, p in enumerate(sorted(set(st.prod
		for tree in trees for st in tree[0].subtrees()) - prods.viewkeys())))

def getlabels(trees, labels):
	labels.update((l, n) for n, l in enumerate(sorted(set(st.node
		for tree in trees for st in tree[0].subtrees()) - labels.viewkeys())))

def tolist(tree):
	""" Convert NLTK tree to a list of nodes in pre-order traversal,
	except for the terminals, which come last."""
	for a in tree.subtrees(lambda n: len(n)==1 and not isinstance(n[0], Tree)):
		a[0] = Terminal(a[0])
	result = list(tree.subtrees()) + tree.leaves()
	for n in reversed(range(len(result))):
		a = result[n]
		a.idx = n
	result[0].root = 0
	return result

def one(): return 1

cpdef fastextractfragments(Ctrees trees1, list sents1, int offset, int end,
	dict labels, dict prods, list revlabel,
	Ctrees trees2=None, list sents2=None, bint approx=True,
	bint debug=False, bint discontinuous=False):
	""" Seeks the largest fragments in treebank(s) with a linear time tree
	kernel
	- scenario 1: recurring fragments in single treebank, use:
      fastextractfragments(trees1, sents1, offset, end, labels,prods,None,None)
	- scenario 2: common fragments in two treebanks:
      fastextractfragments(trees1, sents1, offset, end, labels,prods,
      trees2, sents2)
	offset and end can be used to divide the work over multiple processes.
	they are indices of trees1 to work on (default is all).
	when debug is enabled a contingency table is shown for each pair of trees.
	"""
	cdef:
		int n, m, x, start = 0, end2
		UChar SLOTS
		ULong *bitset = NULL, *CST = NULL
		NodeArray a, b, *ctrees1, *ctrees2
		list asent
		dict fragments = {}
		set inter = set()
		array pyarray = array("L")
		FrozenArray frozenarray
	if approx: fragments = <dict>defaultdict(one)

	if trees2 is None: trees2 = trees1
	ctrees1 = trees1.data
	ctrees2 = trees2.data
	SLOTS = BITNSLOTS(max(trees1.maxnodes, trees2.maxnodes))
	bitset = <ULong *>malloc((SLOTS+2) * sizeof(ULong))
	CST = <ULong *>malloc(trees2.maxnodes * SLOTS * sizeof(ULong))
	end2 = trees2.len
	assert bitset is not NULL
	assert CST is not NULL
	# find recurring fragments
	for n in range(offset, min(end or trees1.len, trees1.len)):
		a = ctrees1[n]; asent = <list>(sents1[n])
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
				dumpCST(CST, a, b, asent, (sents2[m] if sents2 else sents1[m]),
					revlabel, SLOTS, True)
			# extract results
			extractbitsets(CST, bitset, a.nodes, b.nodes, b.root, n,
				pyarray, inter, discontinuous, SLOTS)
		for frozenarray in inter:
			pyarray = frozenarray.data
			x = pyarray._L[SLOTS]
			frag = getsubtree(a.nodes, pyarray._L, revlabel,
					None if discontinuous else asent, x)
			if discontinuous: frag = getsent(frag, asent)
			if approx: fragments[frag] += 1
			elif frag not in fragments:
				fragments[frag] = pyarray
		inter.clear()
	free(bitset)
	free(CST)
	return fragments

cdef inline void fasttreekernel(Node *a, Node *b, ULong *CST, UChar SLOTS):
	""" Fast Tree Kernel (average case linear time). Expects trees to be sorted
	according to their productions (in descending order, with terminals as -1).
	Moschitti (2006): Making Tree Kernels practical for Natural Language
	Learning.  """
	cdef int i = 0, j = 0, jj = 0
	while a[i].prod != -1 and b[j].prod != -1:
		if   a[i].prod < b[j].prod: j += 1
		elif a[i].prod > b[j].prod: i += 1
		else:
			while a[i].prod != -1 and a[i].prod == b[j].prod:
				jj = j
				while b[jj].prod != -1 and a[i].prod == b[jj].prod:
					SETBIT(&CST[jj * SLOTS], i)
					jj += 1
				i += 1

cdef inline void extractbitsets(ULong *CST, ULong *scratch, Node *a, Node *b,
	short j, int n, array pyarray, set results, bint discontinuous,
	UChar SLOTS):
	""" visit nodes of b in pre-order traversal. store bitsets of connected
	subsets of A as they are encountered. """
	cdef ULong *bitset = &CST[j * SLOTS]
	cdef short i = anextset(bitset, 0, SLOTS)
	while i != -1:
		ulongset(scratch, 0UL, SLOTS)
		extractat(CST, scratch, a, b, i, j, SLOTS)
		pyarray = clone(pyarray, (SLOTS+2), False)
		ulongcpy(pyarray._L, scratch, SLOTS)
		pyarray._L[SLOTS] = i; pyarray._L[SLOTS+1] = n
		results.add(new_FrozenArray(pyarray))
		i = anextset(bitset, i + 1, SLOTS)
	if b[j].left < 0: return
	if b[b[j].left].prod >= 0:
		extractbitsets(CST, scratch, a, b, b[j].left, n, pyarray, results,
			discontinuous, SLOTS)
	if b[j].right >= 0 and b[b[j].right].prod >= 0:
		extractbitsets(CST, scratch, a, b, b[j].right, n, pyarray, results,
			discontinuous, SLOTS)

cdef inline void extractat(ULong *CST, ULong *result, Node *a, Node *b,
	short i, short j, UChar SLOTS):
	"""Traverse tree a and b in parallel to extract a connected subset """
	SETBIT(result, i)
	CLEARBIT(&CST[j * SLOTS], i)
	if a[i].left < 0: return
	elif TESTBIT(&CST[b[j].left * SLOTS], a[i].left):
		extractat(CST, result, a, b, a[i].left, b[j].left, SLOTS)
	if a[i].right < 0: return
	elif TESTBIT(&CST[b[j].right * SLOTS], a[i].right):
		extractat(CST, result, a, b, a[i].right, b[j].right, SLOTS)

cpdef array exactcounts(Ctrees trees1, list sents1, list bitsets,
	bint discontinuous, list revlabel, list treeswithprod, bint fast=True):
	""" Given a set of fragments from trees1 as bitsets, produce an exact
	count of those fragments in trees1. The reason we need to do this is that
	a fragment can occur in other trees where it was not a maximal fragment"""
	cdef:
		int n, m, i, x, f
		UInt count
		UChar SLOTS = 0
		array pyarray, counts = clone(uintarray, len(bitsets), False)
		ULong *bitset = NULL
		NodeArray a, b, *ctrees1 = trees1.data
		set candidates
	if SLOTS == 0:
		pyarray = bitsets[0]
		SLOTS = pyarray.length - 2
		assert SLOTS
	# compare one bitset to each tree for each unique fragment.
	for f, pyarray in enumerate(bitsets):
		bitset = pyarray._L
		i = bitset[SLOTS]
		n = bitset[SLOTS + 1]
		a = ctrees1[n]
		candidates = set()
		candidates |= <set>(treeswithprod[a.nodes[i].prod])
		for x in range(a.len):
			if TESTBIT(bitset, x):
				candidates &= <set>(treeswithprod[a.nodes[x].prod])

		count = 0
		for m in candidates:
			b = ctrees1[m]
			for x in range(b.len):
				if a.nodes[i].prod == b.nodes[x].prod:
					count += containsbitset(a, b, bitset, i, x)
				elif fast and a.nodes[i].prod > b.nodes[x].prod: break
		counts[f] = count
	return counts

cdef inline bint containsbitset(NodeArray A, NodeArray B, ULong *bitset,
	short i, short j):
	""" Recursively check whether fragment starting from A[i] described by
	bitset is equal to B[j], i.e., whether B contains that fragment from A. """
	cdef Node a = A.nodes[i], b = B.nodes[j]
	if a.prod != b.prod: return False
	# lexical production, no recursion
	if A.nodes[a.left].prod == -1: return True
	# test for structural mismatch
	if (a.left  < 0) != (a.left  < 0): return False
	if (a.right < 0) != (b.right < 0): return False
	# recurse on further nodes
	if a.left < 0: return True
	if TESTBIT(bitset, a.left):
		if not containsbitset(A, B, bitset, a.left, b.left): return False
	if a.right < 0: return True
	if TESTBIT(bitset, a.right):
		return containsbitset(A, B, bitset, a.right, b.right)
	else: return True

cpdef list indextrees(Ctrees trees, dict prods):
	cdef list result = [set() for _ in range(len(prods))]
	cdef NodeArray a
	cdef int n, m
	for n in range(trees.len):
		a = trees.data[n]
		for m in range(a.len):
			if a.nodes[m].prod >= 0: (<set>result[a.nodes[m].prod]).add(n)
			elif a.nodes[m].prod == -1: break
	return result

cpdef dict completebitsets(Ctrees trees, list sents, list revlabel,
	bint discontinuous):
	cdef dict result = {}
	cdef array pyarray
	cdef int n
	cdef UChar SLOTS = BITNSLOTS(trees.maxnodes)
	for n in range(trees.len):
		pyarray = clone(ulongarray, SLOTS + 2, False)
		ulongset(pyarray._L, ~0UL, SLOTS)
		pyarray._L[SLOTS] = trees.data[n].root
		pyarray._L[SLOTS + 1] = n
		frag = strtree(trees.data[n].nodes, revlabel,
			None if discontinuous else sents[n], trees.data[n].root)
		if discontinuous: frag = getsent(frag, sents[n])
		result[frag] = pyarray
	return result

frontierortermre = re.compile("[^)]\)")
frontiersre = re.compile(" \)")
def frontierorterm(x):
	return len(x) == 0 or (len(x) == 1 and not isinstance(x[0], Tree))

def pathsplit(p):
	return p.rsplit("/", 1) if "/" in p else (".", p)

def readtreebank(treebank, labels, prods, sort=True, discontinuous=False,
	limit=0, encoding="utf-8"):
	if treebank is None: return None, None
	if discontinuous:
		# no incremental reading w/disc trees
		from grammar import canonicalize
		from treebank import NegraCorpusReader
		corpus = NegraCorpusReader(*pathsplit(treebank), encoding=encoding)
		trees = corpus.parsed_sents(); sents = corpus.sents()
		if limit: trees = trees[:limit]; sents = sents[:limit]
		for tree in trees: tree.chomsky_normal_form()
		trees = [tolist(add_srcg_rules(canonicalize(x), y))
						for x, y in zip(trees, sents)]
		getlabels(trees, labels)
		getprods(trees, prods)
		if sort:
			for tree in trees:
				root = tree[0]
				# reverse sort so that terminals end up last
				tree.sort(key=lambda n: -prods.get(n.prod, -1))
				for n, a in enumerate(tree): a.idx = n
				tree[0].root = root.idx
		trees = Ctrees(trees, labels, prods)
		return trees, sents
	sents = []
	tmp = Tree("TMP", [])
	lines = codecs.open(treebank, encoding=encoding).readlines()
	if limit: lines = lines[:limit]
	numtrees = len(lines)
	numnodes = sum(a.count(" ") for a in lines) + numtrees
	trees = Ctrees()
	trees.alloc(numtrees, numnodes)
	for m, line in enumerate(lines):
		# create Tree object from string
		try: tree = Tree(line)
		except: print "malformed line:\n%s\nline %d" % (line, m); raise
		sent = tree.leaves()
		# binarize (could be a command line option)
		tmp[:] = [tree]
		try: tmp.chomsky_normal_form()
		except: print "malformed tree:\n%s\nline %d" % (tree, m); raise
		# convert tree to list
		productions = tree.productions()
		for n, a in enumerate(tree.subtrees(frontierorterm)):
			# fixme: get rid of this, maybe exclude terminals altogether
			if a: a[0] = Terminal(n)
			else:
				sent.insert(n, None)
				a.append(Terminal(n))
		tree = list(tree.subtrees()) + tree.leaves()
		# collect new labels and productions
		for a, b in zip(tree, productions):
			if a.node not in labels: labels[a.node] = len(labels)
			a.prod = (b.lhs().symbol(),) + tuple(unicode(x) for x in b.rhs())
			if a.prod not in prods: prods[a.prod] = len(prods)
		root = tree[0]
		if sort: tree.sort(key=lambda n: -prods.get(n.prod, -1))
		for n, a in enumerate(tree): a.idx = n
		tree[0].root = root.idx
		trees.add(tree, labels, prods)
		sents.append(sent)
	return trees, sents

def extractfragments1(trees, sents):
	""" Wrapper. """
	cdef Ctrees treesx
	from grammar import canonicalize
	# read and convert input
	trees = [tolist(add_srcg_rules(canonicalize(x.copy(True)), y))
				for x, y in zip(trees, sents)]
	labels = {}; getlabels(trees, labels)
	prods = {}; getprods(trees, prods)
	revlabel = sorted(labels, key=labels.get)
	treesx = Ctrees(trees, labels, prods)
	# build actual fragments
	fragments = extractfragments(treesx, sents, 0, 0, labels, prods,
		revlabel, discontinuous=True)
	return fragments

cdef test():
	""" Test some of the bitvector operations """
	cdef UChar SLOTS = 4
	cdef int n, m
	cdef ULong vec[4]
	for n in range(SLOTS): vec[n] = 0
	bits = SLOTS * BITSIZE
	print "BITSIZE:", BITSIZE
	print "bits:", bits, "in", SLOTS, "SLOTS"
	for n in range(bits):
		SETBIT(vec, n)
		assert abitcount(vec, SLOTS) == 1, abitcount(vec, SLOTS)
		for m in range(bits):
			if m == n: assert TESTBIT(vec, n)
			else: assert not TESTBIT(vec, m), (n, m)
		CLEARBIT(vec, n)
		assert abitcount(vec, SLOTS) == 0, abitcount(vec, SLOTS)
		for m in range(bits): assert not TESTBIT(vec, n)
	print "bitset test successful"

def main():
	import sys, codecs
	# this fixes utf-8 output when piped through e.g. less
	# won't help if the locale is not actually utf-8, of course
	sys.stdout = codecs.getwriter('utf8')(sys.stdout)
	from doctest import testmod, NORMALIZE_WHITESPACE, ELLIPSIS
	# do doctests, but don't be pedantic about whitespace (I suspect it is the
	# militant anti-tab faction who are behind this obnoxious default)
	optionflags=NORMALIZE_WHITESPACE | ELLIPSIS
	fail, attempted = testmod(verbose=False, optionflags=optionflags)
	if attempted and not fail:
		print "%s: %d doctests succeeded!" % (__file__, attempted)
	test()

	treebank1 = """\
(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (JJ hungry) (NN dog))))
(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (JJ yellow) (NN cat))))
(S (NP (DT The) (JJ little) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN cat)) (VP (VBP ate) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP ate) (NP (DT the) (NN cat))))"""
	treebank = map(Tree, treebank1.splitlines())
	for tree in treebank: tree.chomsky_normal_form()
	sents = [tree.leaves() for tree in treebank]
	for tree in treebank:
		for n, idx in enumerate(tree.treepositions('leaves')): tree[idx] = n
	#for (a,aa),b in sorted(extractfragments1(treebank, sents).items()):
	#	print "%s\t%s\t%d" % (a, aa, b)
	for a,b in sorted(extractfragments1(treebank, sents).items()):
		print "%s\t%s\t%d" % (a[0], a[1], b)

if __name__ == '__main__':
	main()
