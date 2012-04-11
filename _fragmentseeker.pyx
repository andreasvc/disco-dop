""" Implementation of Sangati et al. (2010), Efficiently extract recurring tree
fragments from large treebanks.
Moschitti (2006): Making Tree Kernels practical for Natural Language
Learning.
"""

import re
from collections import defaultdict
from itertools import count
from nltk import Tree
from grammar import srcg_productions, canonicalize, alpha_normalize, freeze
from containers import Terminal

cpdef extractfragments(Ctrees trees1, list sents1, int offset, int end,
	dict labels, dict prods, Ctrees trees2=None, list sents2=None,
	bint debug=False, bint complete=False, bint discontinuous=False):
	""" Seeks the largest fragments in treebank(s)
	- scenario 1: recurring fragments in single treebank, use:
		 extractfragments(trees1, sents1, offset, end, labels, prods, None, None)
	- scenario 2: common fragments in two treebanks:
		 extractfragments(trees1, sents1, offset, end, labels, prods, trees2, sents2)
	offset and end can be used to divide the work over multiple processes.
	offset is the starting point in trees1, end is the number of trees from
	trees1 to work on.
	when debug is enabled a contingency table is shown for each pair of trees.
	when complete is enabled only complete matches of trees in trees1 are
	reported (much faster).  """
	cdef:
		int n, m, aa, bb, x, root, start = 0, MAXNODE, SLOTS
		ULong *bitset = NULL, *CST = NULL
		NodeArray a, b, result
		NodeArray *ctrees1, *ctrees2
		list asent, revlabel = sorted(labels, key=labels.get)
		dict fragments = <dict>defaultdict(set)
		dict inter = <dict>defaultdict(set)
	
	ctrees1 = trees1.data
	if trees2 is None:
		ctrees2 = ctrees1
		trees2 = trees1
		MAXNODE = max([ctrees1[n].len for n in range(trees1.len)])
	else:
		ctrees2 = trees2.data
		MAXNODE = max([ctrees1[n].len for n in range(trees1.len)]
			+ [ctrees2[n].len for n in range(trees2.len)])
	SLOTS = BITNSLOTS(MAXNODE)
	CST = <ULong *>malloc(MAXNODE * MAXNODE * SLOTS * sizeof(ULong))
	assert CST is not NULL, (MAXNODE, SLOTS, MAXNODE * MAXNODE * SLOTS * sizeof(ULong))
	# find recurring fragments
	for n in range(offset, (end or trees1.len) - (sents2 is None)):
		a = ctrees1[n]; asent = <list>(sents1[n])
		if sents2 is None: start = n + 1
		for m in range(start, trees2.len):
			b = ctrees2[m]
			if complete:
				for bb in range(b.len): #skip terminals
					if containsfragment(a, b, 0, bb):
						(<set>inter[n]).add(m)
						break
			else:
				# initialize table
				memset(CST, 0, a.len * b.len * SLOTS * sizeof(ULong))
				# fill table
				for aa in range(a.len): # skip terminals
					if a.nodes[aa].prod == -1: break
					for bb in range(b.len): #skip terminals
						if b.nodes[bb].prod == -1: break
						elif not TESTBIT(&CST[IDX(aa, bb, b.len, SLOTS)], 0):
							getCST(a, b, CST, aa, bb, SLOTS)
				# dump table
				if debug:
					dumpCST(CST, a, b, asent,
						(sents2[m] if sents2 else sents1[m]), revlabel, SLOTS)
				# extract results
				#bb = getnodeset1(CST, a.len, b.len, SLOTS) * SLOTS
				#for aa in range(0, bb):
				#	bs = CST[aa*SLOTS] >> 1
				#	for x in range(1, SLOTS):
				#		tmp = CST[aa*SLOTS+x]
				#		bs |= tmp << (x * BITSIZE - 1)
				for bs in getnodeset(CST, a.len, b.len, SLOTS):
					if sents2 is None: (<set>inter[bs]).add(m)
					else: inter[bs]
		if complete:
			if discontinuous:
				for found_in in inter.itervalues():
					frag = (strtree(a.nodes, revlabel, asent, a.root), asent)
					(<set>found_in).add(n)
					(<set>fragments[frag]).update(found_in)
				
			else:
				for found_in in inter.itervalues():
					frag = strtree(a.nodes, revlabel, asent, a.root)
					(<set>found_in).add(n)
					(<set>fragments[frag]).update(found_in)
		else:
			if discontinuous:
				for bs, found_in in inter.iteritems():
					frag = getsubtree(a.nodes, bs, revlabel, None,
						pyintnextset(bs, 0))
					(<set>found_in).add(n)
					(<set>fragments[getsent(frag, asent)]).update(found_in)
			else:
				for bs, found_in in inter.iteritems():
					frag = getsubtree(a.nodes, bs, revlabel, asent,
						pyintnextset(bs, 0))
					(<set>found_in).add(n)
					(<set>fragments[frag]).update(found_in)
		inter.clear()
	free(CST)
	return fragments

cdef inline void getCST(NodeArray A, NodeArray B, ULong *CST, int i, int j,
	int SLOTS):
	"""Recursively build common subtree table (CST) for subtrees A[i] & B[j]"""
	cdef Node a = A.nodes[i], b = B.nodes[j]
	cdef ULong *child, *bitset = &CST[IDX(i, j, B.len, SLOTS)]
	SETBIT(bitset, 0) # mark cell as visited
	# compare label & arity / terminal; assume presence of arity markers.
	if a.prod == b.prod:
		SETBIT(bitset, i + 1)
		# lexical production, no recursion
		if A.nodes[a.left].prod == -1:
			SETBIT(bitset, a.left + 1)
			return
		# normal production, recurse or use cached value
		child = &CST[IDX(a.left, b.left, B.len, SLOTS)]
		if not child[0] & 1: getCST(A, B, CST, a.left, b.left, SLOTS)
		for n in range(SLOTS): bitset[n] |= child[n]
		if a.right != -1: # and b.right != -1:	#sentinel node
			child = &CST[IDX(a.right, b.right, B.len, SLOTS)]
			if not child[0] & 1: getCST(A, B, CST, a.right, b.right, SLOTS)
			for n in range(SLOTS): bitset[n] |= child[n]
	elif a.label == b.label: SETBIT(bitset, i + 1)

cdef inline bint containsfragment(NodeArray A, NodeArray B, int i, int j):
	""" Recursively check whether tree A[i] is equal to B[j],
	i.e., whether B contains A. """
	cdef Node a = A.nodes[i], b = B.nodes[j]
	# compare label & arity / terminal; assume presence of arity markers.
	if a.prod < 0: return a.label == b.label
	elif a.prod == b.prod:
		# lexical production, no recursion
		if A.nodes[a.left].prod == -1: return True
		# normal production, recurse
		if not containsfragment(A, B, a.left, b.left): return False
		if a.right != -1: # and b.right != -1:	#sentinel node
			return containsfragment(A, B, a.right, b.right)
		return True
	return False

cdef inline int getnodeset1(ULong *CST, int lena, int lenb, int SLOTS):
	""" Extract the largest, disjuncts bitsets from the common subtree table
	(CST). Results are written to the first row of CST (and may overflow to the
	next rows). """
	cdef ULong *bitset
	cdef int n, m, x, cnt = 0
	for n in range(lena):
		for m in range(lenb):
			bitset = &CST[IDX(n, m, lenb, SLOTS)]
			if not bitset[0] & 1 or abitcount(bitset, SLOTS) <= 1: continue
			for x in range(cnt):
				other = &CST[x * SLOTS]
				if subset(bitset, other, SLOTS): break # other contains bitset
				elif subset(other, bitset, SLOTS): # bitset contains other
					memcpy(other, bitset, SLOTS * sizeof(ULong))
					break
			else: # completely new (disjunct) bitset
				memcpy(&CST[cnt * SLOTS], bitset, SLOTS * sizeof(ULong))
				cnt += 1
	return cnt

cdef inline set getnodeset(ULong *CST, int lena, int lenb, int SLOTS):
	""" Extract the largest, disjuncts bitsets from the common subtree 
	table (CST). """
	cdef ULong *bitset
	cdef int n, m, x
	cdef set finalnodeset = set()
	for n in range(lena):
		for m in range(lenb):
			bitset = &CST[IDX(n, m, lenb, SLOTS)]
			if not bitset[0] & 1 or abitcount(bitset, SLOTS) <= 2: continue
			coll = 0
			for x in range(SLOTS):
				tmp = bitset[x]
				coll |= tmp << (x * BITSIZE)
			coll >>= 1
			#assert coll > 0
			for bs in finalnodeset:
				if bs & coll == coll: break # bs contains coll
				elif bs & coll == bs: # coll contains bs
					finalnodeset.add(coll)
					finalnodeset.discard(bs)
					break
			else: finalnodeset.add(coll) # completely new (disjunct) bitset
	return finalnodeset

cdef inline str strtree(Node *tree, list labels, list sent, int i):
	""" produce string representation of (complete) tree. """
	if tree[i].prod == -1:
		if sent is None: return str(tree[i].label)
		return "" if sent[tree[i].label] is None else sent[tree[i].label]
	if tree[i].left >= 0:
		if tree[i].right >= 0:
			return "(%s %s %s)" % (labels[tree[i].label],
				strtree(tree, labels, sent, tree[i].left),
				strtree(tree, labels, sent, tree[i].right))
		return "(%s %s)" % (labels[tree[i].label],
			strtree(tree, labels, sent, tree[i].left))
	return "(%s )" % (labels[tree[i].label])

cdef inline str getsubtree(Node *tree, object bs, list labels, list sent,
	int i):
	""" turn pyint bitset into string representation of tree. """
	if tree[i].prod == -1:
		return str(tree[i].label) if sent is None else sent[tree[i].label]
	pyint = tree[i].left
	if tree[i].left >= 0 and bs & (1 << pyint):
		if tree[i].right >= 0:
			return "(%s %s %s)" % (labels[tree[i].label],
				getsubtree(tree, bs, labels, sent, tree[i].left),
				getsubtree(tree, bs, labels, sent, tree[i].right))
		return "(%s %s)" % (labels[tree[i].label],
			getsubtree(tree, bs, labels, sent, tree[i].left))
	if sent is None:
		return "(%s %s)" % (labels[tree[i].label], yieldheads(tree, sent, i))
	return "(%s )" % (labels[tree[i].label])

cdef inline yieldheads(Node *tree, list sent, int i):
	y = getyield(tree, sent, i)
	return " ".join([str(a) for a in sorted(y) if a - 1 not in y])

cdef inline getyield(Node *tree, list sent, int i):
	if tree[i].prod == -1: return [tree[i].label]
	elif tree[i].left < 0: return [] #??
	elif tree[i].right < 0: return getyield(tree, sent, tree[i].left)
	else: return getyield(tree, sent, tree[i].left) + getyield(tree, sent, tree[i].right)

cdef inline str getsubtree1(Node *tree, ULong *bitset, list labels, list sent,
	int i):
	""" Turn bitset into string representation of tree.
	Subtle difference with getsubtree: bits in bitset stand for productions, not nodes.
	indices of terminals and frontier nodes do not need to be set."""
	if TESTBIT(bitset, i) and tree[i].left >= 0:
		if tree[i].right >= 0:
			return "(%s %s %s)" % (labels[tree[i].label],
				getsubtree1(tree, bitset, labels, sent, tree[i].left),
				getsubtree1(tree, bitset, labels, sent, tree[i].right))
		return "(%s %s)" % (labels[tree[i].label],
			getsubtree1(tree, bitset, labels, sent, tree[i].left))
	elif tree[i].prod == -1:
		return str(tree[i].label) if sent is None else sent[tree[i].label]
	elif sent is None:
		return "(%s %s)" % (labels[tree[i].label], yieldheads(tree, sent, i))
	else: return "(%s )" % (labels[tree[i].label])

termsre = re.compile(r"([^ )]+)\)")
def repl(d):
	def f(x): return d[int(x.groups()[0])] + ")"
	return f

def getsent(frag, list sent):
	""" Select the words that occur in the fragment and renumber terminals
	in fragment. Expects a tree as string."""
	cdef:
		int x = 0, n, maxl
		list newsent = []
		dict leafmap = {}
	leaves = set(map(int, termsre.findall(frag)))
	if not leaves: return frag, ()
	maxl = max(leaves)
	for n in sorted(leaves):
		leafmap[n] = str(x)
		newsent.append(sent[n])
		x += 1
		if n+1 not in leaves and n != maxl:
			leafmap[n+1] = str(x)
			newsent.append(None)
			x += 1
	frag = termsre.sub(repl(leafmap), frag)
	return frag, tuple(newsent)

cdef dumpCST(ULong *CST, NodeArray a, NodeArray b, list asent, list bsent, list revlabel, int SLOTS, bint bitmatrix=False):
	cdef int n, m
	cdef Node aa, bb
	print "trees:\n%s\n%s" % (strtree(a.nodes, revlabel, asent, a.root), strtree(b.nodes, revlabel, bsent, b.root))
	print '\t'.join([''] + ["%2d"%x for x in range(b.len) if b.nodes[x].prod != -1]), '\n',
	for m in range(b.len):
		bb = b.nodes[m]
		if bb.prod == -1: break
		print '\t', bsent[bb.label] if bb.prod == -1 else revlabel[bb.label][:3],
	for n in range(a.len):
		aa = a.nodes[n]
		if aa.prod == -1: break
		print "\n%2d"%n, (asent[aa.label] if aa.prod == -1 else revlabel[aa.label][:3]),
		for m in range(b.len):
			print '\t',
			if b.nodes[m].prod == -1: break
			elif bitmatrix:
				print "1" if TESTBIT(&CST[n * SLOTS], m) else " ",
			else:
				if TESTBIT(&CST[IDX(n,m,b.len,SLOTS)], 0):
					if abitcount(&CST[IDX(n,m,b.len,SLOTS)], SLOTS) -1:
						print abitcount(&CST[IDX(n,m,b.len,SLOTS)], SLOTS) - 1,
				else: print '-',
	print
	print "common productions:", len(set([a.nodes[n].prod for n in range(a.len)]) &
		set([b.nodes[n].prod for n in range(b.len)]))
	print "found:",
	if bitmatrix: print sum([abitcount(&CST[n * SLOTS], SLOTS) > 0 for n in range(a.len)])
	else: print sum([any([abitcount(&CST[IDX(n,m,b.len,SLOTS)], SLOTS) > 2 for m in range(b.len)]) for n in range(a.len)])

def add_cfg_rules(tree):
	for a, b in zip(tree.subtrees(), tree.productions()):
		a.prod = (b.lhs().symbol(),) + tuple(str(x) for x in b.rhs())
	return tree

def add_srcg_rules(tree, sent):
	for a, b in zip(tree.subtrees(), srcg_productions(tree, sent, False)):
		a.prod = freeze(alpha_normalize(b))
	return tree

def getprods(trees):
	return dict((p, n) for n, p in enumerate(sorted(set(st.prod
		for tree in trees for st in tree[0].subtrees()))))

def getlabels(trees):
	return dict((l, n) for n, l in enumerate(sorted(set(st.node
		for tree in trees for st in tree[0].subtrees()))))

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

def sorttrees(trees, prods):
	for tree in trees:
		# reverse sort so that terminals end up last
		tree.sort(key=lambda n: -prods.get(n.prod, -1))
		for n, a in enumerate(tree):
			if a.idx == 0: a.root = root = n
			a.idx = n
		tree[0].root = root

cpdef fastextractfragments(Ctrees trees1, list sents1, int offset, int end,
	dict labels, dict prods, Ctrees trees2=None, list sents2=None,
	bint debug=False, bint complete=False, bint discontinuous=False):
	""" Seeks the largest fragments in treebank(s)
	- scenario 1: recurring fragments in single treebank, use:
	  extractfragments(trees1, sents1, offset, end, labels, prods, None, None)
	- scenario 2: common fragments in two treebanks:
	  extractfragments(trees1, sents1, offset, end, labels, prods,
	  	trees2, sents2)
	offset and end can be used to divide the work over multiple processes.
	they are indices of trees1 to work on (default is all).
	when debug is enabled a contingency table is shown for each pair of trees.
	when complete is enabled only complete matches of trees in trees1 are
	reported (much faster).  """
	cdef:
		int n, m, x, root, start = 0, MAXNODE, SLOTS
		ULong *bitset = NULL, *CST = NULL
		NodeArray a, b, result, *ctrees1, *ctrees2
		list asent, bsent, revlabel = sorted(labels, key=labels.get)
		dict fragments = <dict>defaultdict(set)
	
	ctrees1 = trees1.data
	if trees2 is None:
		ctrees2 = ctrees1
		trees2 = trees1
		MAXNODE = max([ctrees1[n].len for n in range(trees1.len)])
	else:
		ctrees2 = trees2.data
		MAXNODE = max([ctrees1[n].len for n in range(trees1.len)]
			+ [ctrees2[n].len for n in range(trees2.len)])
	SLOTS = BITNSLOTS(MAXNODE)
	bitset = <ULong *>malloc(SLOTS * sizeof(ULong))
	CST = <ULong *>malloc(MAXNODE * SLOTS * sizeof(ULong))
	assert bitset is not NULL
	assert CST is not NULL
	# find recurring fragments
	for n in range(offset, (end or trees1.len) - (sents2 is None)):
		a = ctrees1[n]; asent = <list>(sents1[n])
		if sents2 is None: start = n + 1
		for m in range(start, trees2.len):
			b = ctrees2[m]
			bsent = <list>(sents1[m]) if sents2 is None else <list>(sents2[m])
			if complete:
				for x in range(b.len):
					if b.nodes[x].prod == -1: break #skip terminals
					if containsfragment(a, b, a.root, x):
						(<set>fragments[n]).add(m)
						frag = strtree(a.nodes, revlabel,
							None if discontinuous else asent, a.root)
						if discontinuous: (<set>fragments[frag, asent]).add(m)
						else: (<set>fragments[frag]).add(m)
						break
				continue
			# initialize table
			memset(CST, 0, a.len * SLOTS * sizeof(ULong))
			# fill table
			fasttreekernel(b.nodes, a.nodes, CST, SLOTS)
			# dump table
			if debug:
				print n, m
				dumpCST(CST, a, b, asent, (sents2[m] if sents2 else sents1[m]),
					revlabel, SLOTS, True)
			# extract results
			x = m if sents2 is None else -1
			#todo: exact counts! i.e., track multiple occurrences per tree
			extractbitsets(CST, bitset, b.nodes, a.nodes, a.root, n, m,
				fragments, discontinuous, revlabel, bsent, SLOTS)
	free(bitset)
	free(CST)
	return fragments

cdef void fasttreekernel(Node *a, Node *b, ULong *CST, int SLOTS):
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
	short j, int n, int m, dict results, bint discontinuous, list revlabel,
	list asent, int SLOTS):
	""" visit nodes of b in pre-order traversal. store bitsets of connected 
	subsets of A as they are encountered. """
	cdef ULong *bitset = &CST[j * SLOTS]
	cdef short x, i = anextset(bitset, 0, SLOTS)
	cdef str frag
	cdef set s
	while i != -1:
		memset(scratch, 0, SLOTS * sizeof(ULong))
		extractat(CST, scratch, a, b, i, j, SLOTS)
		frag = getsubtree1(a, scratch, revlabel,
			None if discontinuous else asent, i)
		if discontinuous: s = (<set>results[getsent(frag, asent)])
		else: s = (<set>results[frag])
		s.add(n); s.add(m)
		i = anextset(bitset, i + 1, SLOTS)
	if b[j].left < 0: return
	extractbitsets(CST, scratch, a, b, b[j].left, n, m, results,
		discontinuous, revlabel, asent, SLOTS)
	if b[j].right < 0: return
	extractbitsets(CST, scratch, a, b, b[j].right, n, m, results,
		discontinuous, revlabel, asent, SLOTS)

cdef inline void extractat(ULong *CST, ULong *result, Node *a, Node *b,
	short i, short j, int SLOTS):
	"""Traverse tree a and b in parallel to extract a connected subset """
	SETBIT(result, i)
	CLEARBIT(&CST[j * SLOTS], i)
	if a[i].left < 0 or b[j].left < 0: return
	elif TESTBIT(&CST[b[j].left * SLOTS], a[i].left):
		extractat(CST, result, a, b, a[i].left, b[j].left, SLOTS)
	if a[i].right < 0 or b[j].right < 0: return
	elif TESTBIT(&CST[b[j].right * SLOTS], a[i].right):
		extractat(CST, result, a, b, a[i].right, b[j].right, SLOTS)

def extractfragments1(trees, sents):
	""" Seeks the largest fragments occurring at least twice in the corpus. """
	cdef NodeArray *ctrees1 = <NodeArray *>malloc(len(trees) * sizeof(NodeArray))
	cdef Ctrees treesx
	# read and convert input
	trees = [tolist(add_srcg_rules(canonicalize(x.copy(True)), y))
				for x, y in zip(trees, sents)]
	labels = getlabels(trees)
	prods = getprods(trees)
	treesx = Ctrees(trees, labels, prods)
	# build actual fragments
	fragments = extractfragments(treesx, sents, 0, len(trees), labels, prods, None, None, discontinuous=True)
	del treesx
	return dict([(x, len(y)) for x, y in fragments.iteritems()])

cdef test():
	#define BITSIZE 			(8*sizeof(long))
	#define BITMASK(b) 			(1 << ((b) % BITSIZE))
	#define BITSLOT(b) 			((b) / BITSIZE)
	#define SETBIT(a, b) 		((a)[BITSLOT(b)] |= BITMASK(b))
	#define CLEARBIT(a, b) 		((a)[BITSLOT(b)] &= ~BITMASK(b))
	#define TESTBIT(a, b) 		((a)[BITSLOT(b)] & BITMASK(b))
	#define BITNSLOTS(nb) 		((nb + BITSIZE - 1) / BITSIZE)

	#define IDX(i,j,jmax,kmax)		((i * jmax + j) * kmax)
	#define CSTIDX(i,j)		(&CST[(i * MAXNODE + j) * SLOTS])

	#int abitcount(ULong vec[], int slots) {
	#int anextset(ULong vec[], unsigned int pos, int slots) {
	#void setunion(ULong vec1[], ULong vec2[], int slots) {
	#int subset(ULong vec1[], ULong vec2[], int slots) {
	cdef int SLOTS = 4, n, m
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
	from treebank import NegraCorpusReader
	import sys, codecs
	# this fixes utf-8 output when piped through e.g. less
	# won't help if the locale is not actually utf-8, of course
	sys.stdout = codecs.getwriter('utf8')(sys.stdout)
	from doctest import testmod, NORMALIZE_WHITESPACE, ELLIPSIS
	# do doctests, but don't be pedantic about whitespace (I suspect it is the
	# militant anti-tab faction who are behind this obnoxious default)
	fail, attempted = testmod(verbose=False, optionflags=NORMALIZE_WHITESPACE | ELLIPSIS)
	if attempted and not fail:
		print "%s: %d doctests succeeded!" % (__file__, attempted)
	test()

	treebank1 = """(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (JJ hungry) (NN dog))))
(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (JJ yellow) (NN cat))))
(S (NP (DT The) (JJ little) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN cat)) (VP (VBP ate) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP ate) (NP (DT the) (NN cat))))"""
	treebank1 = "\n".join(treebank1 for _ in range(1))
	treebank = map(Tree, treebank1.splitlines())
	for tree in treebank: tree.chomsky_normal_form()
	sents = [tree.leaves() for tree in treebank]
	for tree, sent in zip(treebank, sents):
		for n, idx in enumerate(tree.treepositions('leaves')): tree[idx] = n
	#for (a,aa),b in sorted(extractfragments1(treebank, sents).items()):
	#	print "%s\t%s\t%d" % (a, aa, b)
	for a,b in sorted(extractfragments1(treebank, sents).items()):
		print "%s\t%d" % (a, b)

if __name__ == '__main__':
	main()
