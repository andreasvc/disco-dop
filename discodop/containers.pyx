""" Data types for chart items, edges, &c. """
from math import exp, log, fsum
from libc.math cimport log, exp
from discodop.tree import Tree

DEF SLOTS = 3
maxbitveclen = SLOTS * sizeof(ULong) * 8


cdef class LexicalRule:
	""" A weighted rule of the form 'non-terminal --> word'. """
	def __init__(self, lhs, word, prob):
		self.lhs = lhs
		self.word = word
		self.prob = prob

	def __repr__(self):
		return "%s%r" % (self.__class__.__name__,
				(self.lhs, self.word, self.prob))


cdef class SmallChartItem:
	""" Item with word sized bitvector """
	def __init__(SmallChartItem self, label, vec):
		self.label = label
		self.vec = vec

	def __hash__(SmallChartItem self):
		""" juxtapose bits of label and vec, rotating vec if > 33 words:
		64              32            0
		|               ..........label
		|vec[0] 1st half
		|               vec[0] 2nd half
		------------------------------- XOR """
		return (self.label ^ (self.vec << (sizeof(self.vec) / 2 - 1))
				^ (self.vec >> (sizeof(self.vec) / 2 - 1)))

	def __richcmp__(SmallChartItem self, SmallChartItem ob, int op):
		if op == 2:
			return self.label == ob.label and self.vec == ob.vec
		elif op == 3:
			return self.label != ob.label or self.vec != ob.vec
		elif op == 5:
			return self.label >= ob.label or self.vec >= ob.vec
		elif op == 1:
			return self.label <= ob.label or self.vec <= ob.vec
		elif op == 0:
			return self.label < ob.label or self.vec < ob.vec
		elif op == 4:
			return self.label > ob.label or self.vec > ob.vec

	def __nonzero__(SmallChartItem self):
		return self.label != 0 and self.vec != 0

	def __repr__(self):
		return "%s(%d, %s)" % (self.__class__.__name__,
				self.label, self.binrepr())

	def lexidx(self):
		assert self.label == 0
		return self.vec

	def copy(SmallChartItem self):
		return SmallChartItem(self.label, self.vec)

	def binrepr(SmallChartItem self, int lensent=0):
		return bin(self.vec)[2:].zfill(lensent)[::-1]


cdef class FatChartItem:
	""" Item with fixed-with bitvector. """
	def __hash__(self):
		cdef long n, _hash
		""" juxtapose bits of label and vec:
		64              32            0
		|               ..........label
		|vec[0] 1st half
		|               vec[0] 2nd half
		|........ rest of vec .........
		------------------------------- XOR """
		_hash = (self.label ^ self.vec[0] << (8 * sizeof(self.vec[0]) / 2 - 1)
				^ self.vec[0] >> (8 * sizeof(self.vec[0]) / 2 - 1))
		# add remaining bits
		for n in range(sizeof(self.vec[0]), sizeof(self.vec)):
			_hash *= 33 ^ (<UChar *>self.vec)[n]
		return _hash

	def __richcmp__(FatChartItem self, FatChartItem ob, int op):
		cdef int cmp = memcmp(<UChar *>self.vec, <UChar *>ob.vec,
			sizeof(self.vec))
		cdef bint labelmatch = self.label == ob.label
		if op == 2:
			return labelmatch and cmp == 0
		elif op == 3:
			return not labelmatch or cmp != 0
		elif op == 5:
			return self.label >= ob.label or (labelmatch and cmp >= 0)
		elif op == 1:
			return self.label <= ob.label or (labelmatch and cmp <= 0)
		elif op == 0:
			return self.label < ob.label or (labelmatch and cmp < 0)
		elif op == 4:
			return self.label > ob.label or (labelmatch and cmp > 0)

	def __nonzero__(self):
		cdef int n
		if self.label:
			for n in range(SLOTS):
				if self.vec[n]:
					return True
		return False

	def __repr__(self):
		return "%s(%d, %s)" % (self.__class__.__name__,
			self.label, self.binrepr())

	def lexidx(self):
		assert self.label == 0
		return self.vec[0]

	def copy(FatChartItem self):
		cdef FatChartItem a = FatChartItem(self.label)
		for n in range(SLOTS):
			a.vec[n] = self.vec[n]
		return a

	def binrepr(FatChartItem self, lensent=0):
		cdef int m, n = SLOTS - 1
		cdef str result
		while n and self.vec[n] == 0:
			n -= 1
		result = bin(self.vec[n])
		for m in range(n - 1, -1, -1):
			result += bin(self.vec[m])[2:].zfill(BITSIZE)
		return result.zfill(lensent)[::-1]


cdef class CFGChartItem:
	""" Item for CFG parsing; span is denoted with start and end indices. """
	def __init__(self, label, start, end):
		self.label = label
		self.start = start
		self.end = end

	def __hash__(self):
		""" juxtapose bits of label and indices of span:
		|....end...start...label
		64    40      32       0 """
		return (self.label ^ <ULong>self.start << (8 * sizeof(long) / 2)
				^ <ULong>self.end << (8 * sizeof(long) / 2 + 8))

	def __richcmp__(CFGChartItem self, CFGChartItem ob, int op):
		cdef bint labelmatch = self.label == ob.label
		cdef bint startmatch = self.start == ob.start
		if op == 2:
			return labelmatch and startmatch and self.end == ob.end
		elif op == 3:
			return not labelmatch or not startmatch or self.end != ob.end
		elif op == 5:
			return self.label >= ob.label or (labelmatch and (self.start
					>= ob.start or startmatch and self.end >= ob.end))
		elif op == 1:
			return self.label <= ob.label or (labelmatch and (self.start
					<= ob.start or (startmatch and self.end <= ob.end)))
		elif op == 0:
			return self.label < ob.label or (labelmatch and (self.start
					< ob.start or (startmatch and self.end < ob.end)))
		elif op == 4:
			return self.label > ob.label or (labelmatch and (self.start
					> ob.start or (startmatch and self.end > ob.end)))

	def __nonzero__(self):
		return self.label and self.end

	def __repr__(self):
		return "%s(%d, %d, %d)" % (self.__class__.__name__,
				self.label, self.start, self.end)

	def lexidx(self):
		assert self.label == 0
		return self.start

	def copy(CFGChartItem self):
		return new_CFGChartItem(self.label, self.start, self.end)


cdef SmallChartItem CFGtoSmallChartItem(UInt label, UChar start, UChar end):
	return new_ChartItem(label, (1ULL << end) - (1ULL << start))


cdef FatChartItem CFGtoFatChartItem(UInt label, UChar start, UChar end):
	cdef FatChartItem fci = new_FatChartItem(label)
	if BITSLOT(start) == BITSLOT(end):
		fci.vec[BITSLOT(start)] = (1ULL << end) - (1ULL << start)
	else:
		fci.vec[BITSLOT(start)] = ~0UL << (start % BITSIZE)
		for n in range(BITSLOT(start) + 1, BITSLOT(end)):
			fci.vec[n] = ~0UL
		fci.vec[BITSLOT(end)] = BITMASK(end) - 1
	return fci


cdef class LCFRSEdge:
	""" NB: hash / (in)equality considers all elements except inside score,
	order is determined by inside score only. """
	def __hash__(LCFRSEdge self):
		cdef long _hash = 0x345678UL
		# this condition could be avoided by using a dedicated sentinel Rule
		if self.rule is not NULL:
			_hash = (1000003UL * _hash) ^ <long>self.rule.no
		# we only look at the left item, because this edge will only be
		# compared to other edges for the same parent item
		# FIXME: we cannot compute hash directly here, because
		# left can be of different subtypes.
		_hash = (1000003UL * _hash) ^ <long>self.left.__hash__()
		return _hash

	def __richcmp__(LCFRSEdge self, LCFRSEdge ob, int op):
		if op == 0:
			return self.score < ob.score
		elif op == 1:
			return self.score <= ob.score
		elif op == 2 or op == 3:
			# right matches iff left matches, so skip that check
			return (op == 2) == (self.rule is ob.rule and self.left == ob.left)
		elif op == 4:
			return self.score > ob.score
		elif op == 5:
			return self.score >= ob.score
		elif op == 1:
			return self.score <= ob.score
		elif op == 0:
			return self.score < ob.score

	def __repr__(self):
		return "%s(%g, %g, Rule(%g, 0x%x, 0x%x, %d, %d, %d, %d), %r, %r)" % (
				self.__class__.__name__, self.score, self.inside,
				self.rule.prob, self.rule.args, self.rule.lengths,
				self.rule.lhs, self.rule.rhs1, self.rule.rhs2, self.rule.no,
				self.left, self.right)

	def copy(self):
		return new_LCFRSEdge(self.score, self.inside, self.rule,
				self.left.copy(), self.right.copy())


cdef class CFGEdge:
	""" NB: hash / (in)equality considers all elements except inside score,
	order is determined by inside score only. """
	def __hash__(CFGEdge self):
		cdef long _hash = 0x345678UL
		_hash = (1000003UL * _hash) ^ <long>self.rule
		_hash = (1000003UL * _hash) ^ <long>self.mid
		return _hash

	def __richcmp__(CFGEdge self, CFGEdge ob, int op):
		if op == 0:
			return self.inside < ob.inside
		elif op == 1:
			return self.inside <= ob.inside
		elif op == 2 or op == 3:
			return (op == 2) == (self.rule is ob.rule and self.mid == ob.mid)
		elif op == 4:
			return self.inside > ob.inside
		elif op == 5:
			return self.inside >= ob.inside
		elif op == 1:
			return self.inside <= ob.inside
		elif op == 0:
			return self.inside < ob.inside

	def __repr__(self):
		return "%s(%g, Rule(%g, 0x%x, 0x%x, %d, %d, %d, %d), %r)" % (
			self.__class__.__name__, self.inside, self.rule.prob,
			self.rule.args, self.rule.lengths, self.rule.lhs, self.rule.rhs1,
			self.rule.rhs2, self.rule.no, self.mid)


cdef class RankedEdge:
	""" An edge, including the ChartItem to which it points, along with
	ranks for its children, to denote a k-best derivation. """
	def __cinit__(self, ChartItem head, LCFRSEdge edge, int j1, int j2):
		self.head = head
		self.edge = edge
		self.left = j1
		self.right = j2

	def __hash__(self):
		cdef long _hash = 0x345678UL
		_hash = (1000003UL * _hash) ^ hash(self.head)
		_hash = (1000003UL * _hash) ^ hash(self.edge)
		_hash = (1000003UL * _hash) ^ self.left
		_hash = (1000003UL * _hash) ^ self.right
		return _hash

	def __richcmp__(RankedEdge self, RankedEdge ob, int op):
		if op == 2 or op == 3:
			return (op == 2) == (self.left == ob.left and self.right ==
					ob.right and self.head == ob.head and self.edge == ob.edge)
		return NotImplemented

	def __repr__(self):
		return "%s(%r, %r, %d, %d)" % (self.__class__.__name__,
			self.head, self.edge, self.left, self.right)


cdef class RankedCFGEdge:
	""" An edge, including the ChartItem to which it points, along with
	ranks for its children, to denote a k-best derivation. """
	def __cinit__(self, UInt label, UChar start, UChar end, Edge edge,
			int j1, int j2):
		self.label = label
		self.start = start
		self.end = end
		self.edge = edge
		self.left = j1
		self.right = j2

	def __hash__(self):
		cdef long _hash = 0x345678UL
		_hash = (1000003UL * _hash) ^ hash(self.edge)
		_hash = (1000003UL * _hash) ^ self.label
		_hash = (1000003UL * _hash) ^ self.start
		_hash = (1000003UL * _hash) ^ self.end
		_hash = (1000003UL * _hash) ^ self.left
		_hash = (1000003UL * _hash) ^ self.right
		return _hash

	def __richcmp__(RankedCFGEdge self, RankedCFGEdge ob, int op):
		if op == 2 or op == 3:
			return (op == 2) == (self.left == ob.left and self.right ==
					ob.right and self.label == ob.label and self.start ==
					ob.start and self.end == ob.end and self.edge == ob.edge)
		return NotImplemented

	def __repr__(self):
		return "%s(%r, %r, %r, %r, %d, %d)" % (self.__class__.__name__,
			self.label, self.start, self.end, self.edge, self.left, self.right)


cdef class Ctrees:
	"""
	Auxiliary class to be able to pass around collections of NodeArrays
	in Python.

	When trees is given, prods should be given as well.
	When trees is not given, the alloc() method should be called and
	trees added one by one using the add() or addnodes() methods. """
	def __cinit__(self):
		self.trees = self.nodes = NULL

	def __init__(self, list trees=None, dict prods=None):
		self.len = self.max = 0
		self.numnodes = self.maxnodes = self.nodesleft = 0
		if trees is not None:
			assert prods is not None
			self.alloc(len(trees), sum(map(len, trees)))
			for tree in trees:
				self.add(tree, prods)

	cpdef alloc(self, int numtrees, long numnodes):
		""" Initialize an array of trees of nodes structs. """
		self.max = numtrees
		self.trees = <NodeArray *>malloc(numtrees * sizeof(NodeArray))
		self.nodes = <Node *>malloc(numnodes * sizeof(Node))
		assert self.trees is not NULL and self.nodes is not NULL
		self.nodesleft = numnodes

	cdef realloc(self, int len):
		""" Increase size of array (handy with incremental binarization) """
		self.nodesleft += len
		#estimate how many new nodes will be needed
		self.nodesleft += (self.max - self.len) * (self.numnodes / self.len)
		self.nodes = <Node *>realloc(self.nodes,
				(self.numnodes + self.nodesleft) * sizeof(Node))
		assert self.nodes is not NULL

	cpdef add(self, list tree, dict prods):
		""" Trees can be incrementally added to the node array; useful
		when dealing with large numbers of NLTK trees (say 100,000). """
		assert self.len < self.max, ("either no space left (len >= max) or "
			"alloc() has not been called (max=0). max = %d" % self.max)
		if self.nodesleft < len(tree):
			self.realloc(len(tree))
		self.trees[self.len].len = len(tree)
		self.trees[self.len].offset = self.numnodes
		copynodes(tree, prods, &self.nodes[self.numnodes])
		self.trees[self.len].root = tree[0].rootidx
		self.len += 1
		self.nodesleft -= len(tree)
		self.numnodes += len(tree)
		self.maxnodes = max(self.maxnodes, len(tree))

	cdef addnodes(self, Node *source, int cnt, int root):
		""" Trees can be incrementally added to the node array; this version
		copies a tree that has already been converted to an array of nodes. """
		cdef dict prodsintree, sortidx
		cdef int n, m
		cdef Node *dest
		assert self.len < self.max, ("either no space left (len >= max) or "
				"alloc() has not been called (max=0).\n"
				"len = %d, max = %d" % (self.len, self.max))
		if self.nodesleft < cnt:
			self.realloc(cnt)
		prodsintree = {n: source[n].prod for n in range(cnt)}
		sortidx = {m: n for n, m in enumerate(
				sorted(range(cnt), key=prodsintree.get))}
		# copy nodes to allocated array, while translating indices
		dest = &self.nodes[self.numnodes]
		for n, m in sortidx.iteritems():
			dest[m] = source[n]
			if dest[m].left >= 0:
				dest[m].left = sortidx[source[n].left]
				if dest[m].right >= 0:
					dest[m].right = sortidx[source[n].right]
		self.trees[self.len].offset = self.numnodes
		self.trees[self.len].root = sortidx[root]
		self.trees[self.len].len = cnt
		self.len += 1
		self.nodesleft -= cnt
		self.numnodes += cnt
		if cnt > self.maxnodes:
			self.maxnodes = cnt

	def indextrees(self, dict prods):
		""" Create an index from specific productions to trees containing that
		production. Productions are represented as integer IDs, trees are given
		as sets of integer indices. """
		cdef:
			list result = [set() for _ in prods]
			NodeArray a
			Node *nodes
			int n, m
		for n in range(self.len):
			a = self.trees[n]
			nodes = &self.nodes[a.offset]
			for m in range(a.len):
				(<set>result[nodes[m].prod]).add(n)
		self.treeswithprod = result

	def __dealloc__(Ctrees self):
		if self.nodes is not NULL:
			free(self.nodes)
			self.nodes = NULL
		if self.trees is not NULL:
			free(self.trees)
			self.trees = NULL

	def __len__(self):
		return self.len


cdef inline copynodes(tree, dict prods, Node *result):
	""" Convert NLTK tree to an array of Node structs. """
	cdef int n
	for n, a in enumerate(tree):
		assert isinstance(a, Tree), (
				'Expected Tree node, got %s\n%r' % (type(a), a))
		assert 1 <= len(a) <= 2, (
				"trees must be non-empty and binarized\n%s\n%s" % (a, tree[0]))
		result[n].prod = prods[a.prod]
		if isinstance(a[0], int):  # a terminal index
			result[n].left = -a[0] - 1
		else:
			result[n].left = a[0].idx
			if len(a) == 2:
				result[n].right = a[1].idx
			else:  # unary node
				result[n].right = -1
