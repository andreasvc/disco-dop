""" Implementation of Sangati et al., Efficiently extract recurring tree
fragments from large treebanks. """
	
from collections import defaultdict, deque
from nltk import Tree
from grammar import srcg_productions, canonicalize, alpha_normalize, freeze
from bit import bitcount, nextset
from libc.stdlib cimport malloc, free

cdef extern from "bit.h":
	int BITMASK(int b)
	int BITSLOT(int b)
	int BITSET(long a[], int b)
	int BITTEST(long a[], int b)
	int BITNSLOTS(int nb)
	int abitcount(long vec[], int slots)
	int anextset(long vec[], int pos, int slots)
	int subset(long vec1[], long vec2[], int slots)
	void setunion(long vec1[], long vec2[], int slots)

cdef struct node:
	int label, prod
	short left, right

cdef struct treetype:
	int len
	node *nodes

DEF MAXNODE = 256
DEF BITSIZE = 64 #(8*sizeof(long))
DEF slots = 4 #BITNSLOTS(MAXNODE)

class Terminal():
	def __init__(self, node):
		self.node = node
	def __repr__(self): return repr(self.node)
	def __hash__(self): return hash(self.node)

def add_srcg_rules(tree, sent):
	for a, b in zip(tree.subtrees(), srcg_productions(tree, sent, False)):
		a.prod = freeze(alpha_normalize(zip(*b)))
	return tree

def levelorder(node):
	queue = deque([node]); result = []
	while queue:
		node = queue.popleft()
		node.idx = len(result)
		result.append(node)
		if isinstance(node, Tree):
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
			assert 1 <= len(a) <= 2, "trees must be binarized"
			result[n].label = labels[a.node]
			result[n].prod = prods[a.prod]
			result[n].left = a.left = a[0].idx
			result[n].right = a[1].idx if len(a) == 2 else -1
		elif isinstance(a, Terminal):
			result[n].label = a.node
			result[n].prod = -1
			result[n].left = result[n].right = -1
		else: assert False

cdef dumpCST(long CST[][MAXNODE][slots], a, b, asent, bsent):
	print "\t".join(['',''] + [str(y) for y, x in enumerate(b) if x is not None])
	print "\t".join(['',''] + [x.node if isinstance(x, Tree) else
		bsent[x.node] for x in b if x is not None])
	for n, aa in enumerate(a):
		print "\t".join([str(n),
			aa.node if isinstance(aa, Tree) else asent[aa.node]]
			+ [str(bitcount(CST[n][m][0])) if CST[n][m][0] > 0 else ''
			for m, bb in enumerate(b) if bb is not None])

cpdef extractfragments(trees, sents):
	""" Seeks the largest fragments occurring at least twice in the corpus. """
	cdef int cnt = 0, lentrees = len(trees), n, m, aa, bb
	cdef long CST[MAXNODE][MAXNODE][slots]
	#cdef bitset *CST = <bitset *>malloc(256 * 256 * 4 * 8)
	cdef treetype a, b, result
	cdef treetype *newtrees = <treetype *>malloc(lentrees * sizeof(treetype))
	assert newtrees != NULL
	fragments = defaultdict(set)
	inter = defaultdict(set)
	# read and convert input
	trees = [levelorder(add_srcg_rules(canonicalize(x.copy(True)), y))
				for x, y in zip(trees, sents)]
	labels = getlabels(trees)
	prods = getprods(trees)
	for tree, sent in zip(trees, sents):
		newtrees[cnt].len = len(tree)
		assert len(tree) <= sizeof(long) * 8, len(tree)
		newtrees[cnt].nodes = <node *>malloc(newtrees[cnt].len * sizeof(node))
		assert newtrees[cnt].nodes != NULL
		indices(tree, labels, prods, newtrees[cnt].nodes)
		cnt += 1
	# find recurring fragments
	for n in range(lentrees):
		a = newtrees[n]; asent = sents[n]
		inter.clear()
		for m in range(n + 1, lentrees):
			b = newtrees[m]
			# initialize table
			for aa in range(a.len):
				for bb in range(b.len):
					CST[aa][bb][0] = -1
					for z in range(1, 4): CST[aa][bb][z] = 0
			# fill table
			for aa in range(a.len):
				if a.nodes[aa].prod == -1: continue
				for bb in range(b.len):
					if b.nodes[bb].prod != -1 and CST[aa][bb][0] == -1:
						getCST(a, b, aa, bb, CST)
			# dump table
			#dumpCST(CST, trees[n], trees[m], asent, sents[m]); exit()
			for bs in getnodeset(CST, a.len, b.len): inter[bs].add(m)
		# build actual fragments
		for bs in inter:
			if bitcount(bs) == 1: continue
			frag = getsent(getsubtree(trees[n][nextset(bs, 0)], bs), asent)
			s = fragments[frag]
			s.add(n)
			s.update(inter[bs])
	for n in range(lentrees): free(newtrees[n].nodes)
	free(newtrees)
	return dict([(x, len(y)) for x, y in fragments.iteritems()])

cdef void getCST(treetype A, treetype B, int i, int j, long CST[][MAXNODE][slots]):
	""" Recursively build common subtree table (CST) for subtrees A[i] & B[j]. """
	cdef node *a = A.nodes, *b = B.nodes
	CST[i][j][0] = 0
	# compare label & arity / terminal; assume presence of arity markers.
	if a[i].label == b[j].label:
		BITSET(CST[i][j], i)
		if a[i].prod == b[j].prod:
			# lexical production, no recursion
			if a[a[i].left].prod == -1:
				BITSET(CST[i][j], a[i].left)
				return
			# normal production, recurse or use cached value
			if CST[a[i].left][b[j].left][0] == -1:
				getCST(A, B, a[i].left, b[j].left, CST)
			for n in range(slots): CST[i][j][n] |= CST[a[i].left][b[j].left][n]
			if a[i].right != -1 and b[j].right != -1:	#sentinel nodes
				if CST[a[i].right][b[j].right][0] == -1:
					getCST(A, B, a[i].right, b[j].right, CST)
				for n in range(slots): CST[i][j][n] |= CST[a[i].right][b[j].right][n]

cdef set getnodeset(long CST[][MAXNODE][4], int lena, int lenb):
	""" Extract the largest, disjuncts bitsets from the common subtree 
	table (CST). """
	cdef set finalnodeset = set()
	cdef int n, m, x #, y, cnt = 0
	#cdef long *coll
	for n in range(lena):
		for m in range(lenb):
			if CST[n][m][0] == -1 or (CST[n][m][0] ==
				CST[n][m][1] == CST[n][m][2] ==
				CST[n][m][3] == 0): continue
			coll = 0
			for x in range(slots):
				coll |= CST[n][m][x] << (x * 8 * sizeof(long))
			for bs in finalnodeset:
				if bs & coll == coll: break # bs contains coll
				elif bs & coll == bs: # coll contains bs
					finalnodeset.add(coll)
					finalnodeset.discard(bs)
					break
			else: finalnodeset.add(coll) #completely new (disjunct) bitset
			#coll = CST[n][m]
			#for x in range(cnt):
			#	if subset(coll, CST[0][x], slots): break # bs contains coll
			#	elif subset(CST[0][x], coll, slots): # coll contains bs
			#		for y in range(slots): CST[0][x][y] = coll[y]
			#		break
			#else:
			#	cnt += 1
			#	for y in range(slots): CST[0][cnt][y] = coll[y]
	return finalnodeset

def getsubtree(node, bs):
	""" Turn a bitset into an actual Tree object. """
	if not isinstance(node, Tree): return node
	return Tree(node.node,
		[getsubtree(a, bs) for a in node]
		if bs & (1 << node.left) else [])

def getsent(frag, sent):
	""" Select the words that occur in the fragment and renumber terminals
	in fragment. """
	leaves = set(a.node for a in frag.leaves())
	if not leaves: return frag.freeze(), ()
	minl = min(leaves); maxl = max(leaves)
	newsent = []; leafmap = {}; cnt = 0
	for n, a in enumerate(sent):
		if minl <= n <= maxl:
			leafmap[n] = cnt; cnt += 1
			newsent.append(a if n in leaves else None)
	for n in frag.treepositions('leaves'): frag[n] = leafmap[frag[n].node]
	return frag.freeze(), tuple(newsent)

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
	treebank1 = "\n".join(treebank1 for _ in range(100))
	treebank = map(Tree, treebank1.splitlines())
	for tree in treebank: tree.chomsky_normal_form()
	sents = [tree.leaves() for tree in treebank]
	for tree, sent in zip(treebank, sents):
		for n, idx in enumerate(tree.treepositions('leaves')): tree[idx] = n
	for (a,aa),b in sorted(extractfragments(treebank, sents).items()):
		print "%s\t%s\t%d" % (a.pprint(margin=9999), aa, b)

if __name__ == '__main__':
	from doctest import testmod, NORMALIZE_WHITESPACE, ELLIPSIS
	# do doctests, but don't be pedantic about whitespace (I suspect it is the
	# militant anti-tab faction who are behind this obnoxious default)
	fail, attempted = testmod(verbose=False, optionflags=NORMALIZE_WHITESPACE | ELLIPSIS)
	main()
	if attempted and not fail:
		print "%s: %d doctests succeeded!" % (__file__, attempted)
