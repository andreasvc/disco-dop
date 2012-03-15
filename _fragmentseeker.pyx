""" Implementation of Sangati et al., Efficiently extract recurring tree
fragments from large treebanks. """
	
from collections import defaultdict, deque
from itertools import count
from nltk import Tree
from grammar import srcg_productions, canonicalize, alpha_normalize, freeze
from bit import bitcount, nextset
from libc.stdlib cimport malloc, free

#DEF ...
BITSIZE = (8*sizeof(long))
MAXNODE = 1024 #264 #256
SLOTS = 16 #15 #16 #4 #BITNSLOTS(MAXNODE)

class Terminal():
	def __init__(self, node): self.node = node
	def __repr__(self): return repr(self.node)
	def __hash__(self): return hash(self.node)

def add_srcg_rules(tree, sent):
	for a, b in zip(tree.subtrees(), srcg_productions(tree, sent, False)):
		a.prod = freeze(alpha_normalize(zip(*b)))
	return tree

def tolist(tree):
	result = []
	for n, i in enumerate(tree.treepositions()):
		a = tree[i]
		if isinstance(a, Tree): a.left = n + 1
		else: tree[i] = a = Terminal(a)
		a.idx = n
		result.append(a)
	return result

def levelorder(node):
	queue = deque([node]); result = []
	while queue:
		node = queue.popleft()
		node.idx = len(result)
		result.append(node)
		if isinstance(node, Tree):
			node.left = len(result) + len(queue)
			if not isinstance(node[0], Tree): node[0] = Terminal(node[0])
			queue.append(node[0])
			if len(node) == 2: queue.append(node[1])
	return result

def getprods(trees):
	return dict((p, n) for n, p in enumerate(sorted(set(st.prod
		for tree in trees for st in tree[0].subtrees()))))

def getlabels(trees):
	return dict((l, n) for n, l in enumerate(sorted(set(st.node
		for tree in trees for st in tree[0].subtrees()))))

cdef indices(tree, dict labels, dict prods, node *result):
	cdef int n
	for n, a in enumerate(tree):
		if isinstance(a, Tree):
			assert 1 <= len(a) <= 2, "trees must be binarized:\n%s" % a
			result[n].label = labels[a.node]
			result[n].prod = prods[a.prod]
			result[n].left = a[0].idx
			result[n].right = a[1].idx if len(a) == 2 else -1
		elif isinstance(a, Terminal):
			result[n].label = a.node
			result[n].prod = -1
			result[n].left = result[n].right = -1
		else: assert False

cdef dumpCST(long *CST, a, b, asent, bsent):
	print "\t".join(['',''] + [str(y) for y, x in enumerate(b) if x is not None])
	print "\t".join(['',''] + [x.node if isinstance(x, Tree) else
		bsent[x.node] for x in b if x is not None])
	for n, aa in enumerate(a):
		print "\t".join([str(n),
			aa.node if isinstance(aa, Tree) else asent[aa.node]]
			+ [str(bitcount(CST[GET3DIDX(n,m,MAXNODE,SLOTS)])) if CST[GET3DIDX(n,m,MAXNODE,SLOTS)] > 0 else ''
			for m, bb in enumerate(b) if bb is not None])

cpdef extractfragments(list trees1, list sents1, int offset, int chunk, dict labels, dict prods, list trees2=None, list sents2=None):
	""" Seeks the largest fragments occurring at least twice in the corpus.
	- scenario 1: recurring fragments in treebank, use:
		 extractfragments(trees1, sents1, offset, chunk, labels, prods, None, None)
	- scenario 2: common fragments in two treebanks:
		 extractfragments(trees1, sents1, offset, chunk, labels, prods, trees2, sents2)
	offset and chunk is used to divide the work over multiple processes.
	offset is the starting point in trees1, chunk is the number of trees to
	work on."""
	cdef int lentrees1 = len(trees1), lentrees2 = len(trees2 or trees1)
	cdef int n, m, aa, bb, start = 0
	#cdef long CST1[MAXNODE][MAXNODE][SLOTS]
	cdef long *bitset, *CST #= &CST1[0][0][0]
	#cdef long (*CST)[MAXNODE] = malloc(MAXNODE * MAXNODE * sizeof(long[SLOTS]));
	cdef treetype a, b, result
	cdef treetype *newtrees1 = <treetype *>malloc(lentrees1 * sizeof(treetype))
	cdef treetype *newtrees2
	assert newtrees1 != NULL
	inter = defaultdict(set)
	CST = <long *>malloc(MAXNODE * MAXNODE * SLOTS)
	assert CST != NULL
	# convert input
	if trees2 is not None:
		newtrees2 = <treetype *>malloc(lentrees2 * sizeof(treetype))
		assert newtrees2 != NULL
		for n, tree, sent in zip(count(), trees2, sents2):
			assert len(tree) <= BITSIZE * SLOTS, "Tree too large. Nodes: %d, MAXNODE: %d, BITSIZE: %d, SLOTS: %d." % (len(tree), MAXNODE, BITSIZE, SLOTS)
			newtrees2[n].len = len(tree)
			newtrees2[n].nodes = <node *>malloc(newtrees2[n].len * sizeof(node))
			assert newtrees2[n].nodes != NULL
			indices(tree, labels, prods, newtrees2[n].nodes)
	else:
		newtrees2 = newtrees1
	for n, tree, sent in zip(count(), trees1, sents1):
		assert len(tree) <= BITSIZE * SLOTS, len(tree)
		newtrees1[n].len = len(tree)
		newtrees1[n].nodes = <node *>malloc(newtrees1[n].len * sizeof(node))
		assert newtrees1[n].nodes != NULL
		indices(tree, labels, prods, newtrees1[n].nodes)
	# find recurring fragments
	for n in range(offset, min(offset+(chunk or lentrees1), lentrees1)):
		a = newtrees1[n]; asent = sents1[n]
		#inter.clear()
		if trees2 is None: start = n + 1
		for m in range(start, lentrees2):
			b = newtrees2[m]
			# initialize table
			for aa in range(a.len):
				for bb in range(b.len):
					#bitset = CST[aa][bb]
					bitset = &CST[GET3DIDX(aa, bb, MAXNODE, SLOTS)]
					bitset[0] = -1
					for z in range(1, SLOTS): bitset[z] = 0
			# fill table
			for aa in range(a.len):
				if a.nodes[aa].prod == -1: continue
				for bb in range(b.len):
					#bitset = CST[aa][bb]
					bitset = &CST[GET3DIDX(aa, bb, MAXNODE, SLOTS)]
					if b.nodes[bb].prod != -1 and bitset[0] == -1:
						getCST(a, b, aa, bb, CST)
			# dump table
			#dumpCST(CST, trees[n], trees[m], asent, sents[m]); exit()
			if trees2 is None:
				#for bs in getnodeset(CST, a.len, b.len): inter[bs].add(m)
				for bs in getnodeset(CST, a.len, b.len):
					s = inter[n, bs]
					s.add(n); s.add(m)
			else:
				#for bs in getnodeset(CST, a.len, b.len): inter[bs] = None
				for bs in getnodeset(CST, a.len, b.len): inter[n, bs].add(n)
		# build actual fragments
		#for bs in inter:
		#	if bitcount(bs) == 1: continue
		#	frag = getsent(getsubtree(trees1[n][nextset(bs, 0)], bs), asent)
		#	s = fragments[frag]
		#	s.add(n)
		#	if trees2 is None: s.update(inter[bs])
	for n in range(lentrees1): free(newtrees1[n].nodes)
	free(newtrees1)
	if trees2 is not None:
		for n in range(lentrees2): free(newtrees2[n].nodes)
		free(newtrees2)
	return inter

def extractfragments1(trees, sents):
	""" Seeks the largest fragments occurring at least twice in the corpus. """
	fragments = defaultdict(set)
	# read and convert input
	trees = [levelorder(add_srcg_rules(canonicalize(x.copy(True)), y))
				for x, y in zip(trees, sents)]
	labels = getlabels(trees)
	prods = getprods(trees)

	# build actual fragments
	inter = extractfragments(trees, sents, 0, len(trees), labels, prods, None, None)
	for n, bs in inter:
		frag = getsent(getsubtree(trees[n][nextset(bs, 0)], bs), sents[n])
		s = fragments[frag]
		s.add(n)
		s.update(inter[n, bs])
	return dict([(x, len(y)) for x, y in fragments.iteritems()])

#cdef void getCST(treetype A, treetype B, int i, int j, long CST[][MAXNODE][SLOTS]):
cdef void getCST(treetype A, treetype B, int i, int j, long *CST):
	""" Recursively build common subtree table (CST) for subtrees A[i] & B[j]. """
	cdef node a = A.nodes[i], b = B.nodes[j]
	cdef long *bitset, *child
	#bitset = CST[i][j]
	bitset = &CST[GET3DIDX(i, j, MAXNODE, SLOTS)]
	bitset[0] = 0
	# compare label & arity / terminal; assume presence of arity markers.
	if a.label == b.label:
		SETBIT(bitset, i)
		if a.prod == b.prod:
			# lexical production, no recursion
			if A.nodes[a.left].prod == -1:
				SETBIT(bitset, a.left)
				return
			# normal production, recurse or use cached value
			#child = CST[a.left][b.left]
			child = &CST[GET3DIDX(a.left, b.left, MAXNODE, SLOTS)]
			if child[0] == -1: getCST(A, B, a.left, b.left, CST)
			for n in range(SLOTS): bitset[n] |= child[n]
			if a.right != -1: # and b.right != -1:	#sentinel nodes
				#child = CST[a.right][b.right]
				child = &CST[GET3DIDX(a.right, b.right, MAXNODE, SLOTS)]
				if child[0] == -1: getCST(A, B, a.right, b.right, CST)
				for n in range(SLOTS): bitset[n] |= child[n]

#cdef set getnodeset(long CST[][MAXNODE][SLOTS], int lena, int lenb):
cdef set getnodeset(long *CST, int lena, int lenb):
	""" Extract the largest, disjuncts bitsets from the common subtree 
	table (CST). """
	cdef set finalnodeset = set()
	cdef long *bitset
	cdef int n, m, x
	for n in range(lena):
		for m in range(lenb):
			bitset = &CST[GET3DIDX(n, m, MAXNODE, SLOTS)]
			# FIXME -1 okay in bitset ??
			if bitset[0] == -1 or abitcount(bitset, SLOTS) <= 1: continue
			coll = 0
			for x in range(SLOTS):
				tmp = bitset[x]
				coll |= tmp << (x * BITSIZE)
			for bs in finalnodeset:
				if bs & coll == coll: break # bs contains coll
				elif bs & coll == bs: # coll contains bs
					finalnodeset.add(coll)
					finalnodeset.discard(bs)
					break
			else: finalnodeset.add(coll) #completely new (disjunct) bitset
			#coll = CST[n][m]
			#for x in range(cnt):
			#	if subset(coll, CST[0][x], SLOTS): break # bs contains coll
			#	elif subset(CST[0][x], coll, SLOTS): # coll contains bs
			#		for y in range(SLOTS): CST[0][x][y] = coll[y]
			#		break
			#else:
			#	cnt += 1
			#	for y in range(SLOTS): CST[0][cnt][y] = coll[y]
	return finalnodeset

def getsubtree(node, bs):
	""" Turn a bitset into an actual Tree object. """
	if not isinstance(node, Tree): return node
	return Tree(node.node,
		[getsubtree(a, bs) for a in node]
		if bs & (1 << node.left) else [])

def getsent(frag, sent, penn=False):
	""" Select the words that occur in the fragment and renumber terminals
	in fragment. """
	if penn:
		for n in frag.treepositions('leaves'): frag[n] = sent[frag[n].node]
		return frag.pprint(margin=9999)
	newsent = []; leafmap = {}
	leaves = set(a.node for a in frag.leaves())
	if leaves:
		minl = min(leaves)
		for n in range(min(leaves), max(leaves) + 1):
			leafmap[n] = n - minl
			newsent.append(sent[n] if n in leaves else None)
		for n in frag.treepositions('leaves'): frag[n] = leafmap[frag[n].node]
	return frag.pprint(margin=9999), tuple(newsent)

def main():
	from treebank import NegraCorpusReader
	import sys, codecs
	# this fixes utf-8 output when piped through e.g. less
	# won't help if the locale is not actually utf-8, of course
	sys.stdout = codecs.getwriter('utf8')(sys.stdout)
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
	for (a,aa),b in sorted(extractfragments1(treebank, sents).items()):
		print "%s\t%s\t%d" % (a, aa, b)

if __name__ == '__main__':
	from doctest import testmod, NORMALIZE_WHITESPACE, ELLIPSIS
	# do doctests, but don't be pedantic about whitespace (I suspect it is the
	# militant anti-tab faction who are behind this obnoxious default)
	fail, attempted = testmod(verbose=False, optionflags=NORMALIZE_WHITESPACE | ELLIPSIS)
	main()
	if attempted and not fail:
		print "%s: %d doctests succeeded!" % (__file__, attempted)
