from nltk import Tree
from  collections import Set, Sequence, Iterable

cdef class ChartItem:
	def __init__(ChartItem self, label, vec):
		self.label = label
		self.vec = vec
	def __hash__(ChartItem self):
		cdef long h
		# juxtapose bits of label and vec, rotating vec if > 33 words
		h = self.label ^ (self.vec << 31UL) ^ (self.vec >> 31UL)
		return -2 if h == -1 else h
	def __richcmp__(ChartItem self, ChartItem other, int op):
		if op == 2: return self.label == other.label and self.vec == other.vec
		elif op == 3: return self.label != other.label or self.vec != other.vec
		elif op == 5: return self.label >= other.label or self.vec >= other.vec
		elif op == 1: return self.label <= other.label or self.vec <= other.vec
		elif op == 0: return self.label < other.label or self.vec < other.vec
		elif op == 4: return self.label > other.label or self.vec > other.vec
	def __nonzero__(ChartItem self):
		return self.label != 0 and self.vec != 0
	def __repr__(ChartItem self):
		return "ChartItem(%d, %s)" % (self.label, bin(self.vec))

cdef class Edge:
	def __init__(self, score, inside, prob, left, right):
		self.score = score; self.inside = inside; self.prob = prob
		self.left = left; self.right = right
	def __hash__(self):
		cdef long h
		#self._hash = hash((inside, prob, left, right))
		# this is the hash function used for tuples, apparently
		h = (1000003UL * 0x345678UL) ^ <long>self.inside
		h = (1000003UL * h) ^ <long>self.prob
		h = (1000003UL * h) ^ (<ChartItem>self.left).vec
		h = (1000003UL * h) ^ (<ChartItem>self.left).label
		h = (1000003UL * h) ^ (<ChartItem>self.right).vec
		h = (1000003UL * h) ^ (<ChartItem>self.right).label
		return -2 if h == -1 else h
	def __richcmp__(Edge self, other, int op):
		# the ordering only depends on the estimate / inside score
		if op == 0: return self.score < (<Edge>other).score
		elif op == 1: return self.score <= (<Edge>other).score
		# (in)equality compares all elements
		# boolean trick: equality and inequality in one expression i.e., the
		# equality between the two boolean expressions acts as biconditional
		elif op == 2 or op == 3:
			return (op == 2) == (
				self.score == (<Edge>other).score
				and self.inside == (<Edge>other).inside
				and self.prob == (<Edge>other).prob
				and self.left == (<Edge>other).left
				and self.right == (<Edge>other).right)
		elif op == 4: return self.score > other.score
		elif op == 5: return self.score >= other.score
	def __repr__(self):
		return "Edge(%g, %g, %g, %r, %r)" % (
				self.score, self.inside, self.prob, self.left, self.right)

cdef class RankedEdge:
	def __cinit__(RankedEdge self, ChartItem head, Edge edge, int j1, int j2):
		self.head = head; self.edge = edge
		self.left = j1; self.right = j2
	def __hash__(RankedEdge self):
		cdef long h
		#h = hash((head, edge, j1, j2))
		h = (1000003UL * 0x345678UL) ^ hash(self.head)
		h = (1000003UL * h) ^ hash(self.edge)
		h = (1000003UL * h) ^ self.left
		h = (1000003UL * h) ^ self.right
		if h == -1: h = -2
		return h
	def __richcmp__(RankedEdge self, RankedEdge other, int op):
		if op == 2 or op == 3:
			return (op == 2) == (
				self.left == other.left
				and self.right == other.right
				and self.head == other.head
				and self.edge == other.edge)
		else:
			raise NotImplemented
	def __repr__(RankedEdge self):
		return "RankedEdge(%r, %r, %d, %d)" % (
			self.head, self.edge, self.left, self.right)

cdef class Grammar:
	def __init__(self, unary, lbinary, rbinary, lexical, bylhs, lexicalbylhs,
		toid, tolabel, arity):
		self.unary = unary; self.lbinary = lbinary; self.rbinary = rbinary
		self.lexical = lexical; self.bylhs = bylhs
		self.toid = toid; self.tolabel = tolabel
		self.lexicalbylhs = lexicalbylhs; self.arity = arity
	def __repr__(self):
		return repr(dict(unary=self.unary, lbinary=self.lbinary,
			rbinary=self.rbinary, lexical=self.lexical, bylhs=self.bylhs,
			lexicalbylhs=self.lexicalbylhs, toid=self.toid,
			tolabel=self.tolabel))

cdef class LexicalRule:
	def __init__(self, lhs, rhs1, rhs2, word, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.word = word; self.prob = prob
	def __repr__(self):
		return repr((self.lhs, self.rhs1, self.rhs2, self.word, self.prob))

cdef class Rule:
	def __init__(self, lhs, rhs1, rhs2, args, lengths, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.args = args; self.lengths = lengths
		self._args = self.args._I; self._lengths = self.lengths._H
		self.prob = prob
	def __repr__(self):
		return repr((self.lhs, self.rhs1, self.rhs2, self.args, self.lengths,
			self.prob))

cdef class Ctrees:
	"""auxiliary class to be able to pass around collections of trees in
	Python"""
	def __init__(Ctrees self, list trees=None, dict labels=None, dict prods=None):
		self.len=0; self.max=0; self.maxnodes = 0; self.nodesleft = 0
		if trees is None: return
		else: assert labels is not None and prods is not None
		self.alloc(len(trees), sum(map(len, trees)))
		for tree in trees: self.add(tree, labels, prods)
	cpdef alloc(Ctrees self, int numtrees, long numnodes):
		""" Initialize an array of trees of nodes structs. """
		self.max = numtrees
		self.data = <NodeArray *>malloc(numtrees * sizeof(NodeArray))
		assert self.data is not NULL
		self.data[0].nodes = <Node *>malloc(numnodes * sizeof(Node))
		assert self.data[0].nodes is not NULL
		self.nodes = self.nodesleft = numnodes
	cdef realloc(Ctrees self, int len):
		""" Increase size of array (handy with incremental binarization) """
		#other options: new alloc: fragmentation (maybe not so bad)
		#memory pool: idem
		cdef Node *new = NULL
		self.nodes += (self.max - self.len) * len #estimate
		new = <Node *>realloc(self.data[0].nodes, self.nodes * sizeof(Node))
		assert new is not NULL
		if new != self.data[0].nodes: # need to update all previous pointers
			self.data[0].nodes = new
			for n in range(1, self.len):
				# derive pointer from previous tree offset by its size
				self.data[n].nodes = &(
					self.data[n-1].nodes)[self.data[n-1].len]
	cpdef add(Ctrees self, list tree, dict labels, dict prods):
		""" Trees can be incrementally added to the node array; useful
		when dealing with large numbers of NLTK trees (say 100,000)."""
		assert self.len < self.max, ("either no space left (len >= max) or "
			"alloc() has not been called (max=0). max = %d" % self.max)
		if self.nodesleft < len(tree): self.realloc(len(tree))
		self.data[self.len].len = len(tree)
		if self.len: # derive pointer from previous tree offset by its size
			self.data[self.len].nodes = &(
				self.data[self.len-1].nodes)[self.data[self.len-1].len]
		indices(tree, labels, prods, self.data[self.len].nodes)
		self.data[self.len].root = tree[0].root
		self.len += 1
		self.nodesleft -= len(tree)
		self.maxnodes = max(self.maxnodes, len(tree))
	def __dealloc__(Ctrees self):
		free(self.data[0].nodes)
		free(self.data)
	def __len__(Ctrees self): return self.len

cdef inline indices(tree, dict labels, dict prods, Node *result):
	""" Convert NLTK tree to an array of Node structs. """
	cdef int n
	for n, a in enumerate(tree):
		if isinstance(a, Tree):
			assert 1 <= len(a) <= 2, "trees must be binarized:\n%s" % a
			result[n].label = labels.get(a.node, -2)
			if len(a.prod) == 1: result[n].prod = -2
			else: result[n].prod = prods.get(a.prod, -2)
			result[n].left = a[0].idx
			if len(a) == 2:
				result[n].right = a[1].idx
			else: result[n].right = -1
		elif isinstance(a, Terminal):
			result[n].label = a.node
			result[n].prod = result[n].left = result[n].right = -1
		else: assert isinstance(a, Tree) or isinstance(a, Terminal)

class Terminal:
	"""auxiliary class to be able to add indices to terminal nodes of NLTK
	trees"""
	def __init__(self, node): self.prod = self.node = node
	def __repr__(self): return repr(self.node)
	def __hash__(self): return hash(self.node)

cdef class FrozenArray:
	def __init__(self, array data):
		self.data = data
	def __hash__(FrozenArray self):
		cdef int n, _hash
		_hash = 5381
		for n in range(self.data.length):
			_hash *= 33 ^ self.data._B[n]
		return _hash
	def __richcmp__(FrozenArray self, FrozenArray other, int op):
		cdef int cmp = memcmp(self.data._B, other.data._B,
			self.data.length*sizeof(ULong))
		if op == 2: return cmp == 0
		elif op == 3: return cmp != 0
		elif op == 0: return cmp < 0
		elif op == 4: return cmp > 0
		elif op == 1: return cmp <= 0
		return cmp >= 0

cdef inline FrozenArray new_FrozenArray(array data):
	cdef FrozenArray item = FrozenArray.__new__(FrozenArray)
	item.data = data
	return item

cdef class CBitset:
	"""auxiliary class to be able to pass around bitsets in Python"""
	def __cinit__(CBitset self, UChar slots):
		self.slots = slots
	def __hash__(CBitset self):
		cdef int n, _hash
		_hash = 5381
		for n in range(self.slots * sizeof(ULong)):
			_hash *= 33 ^ (<char *>self.data)[n]
		return _hash
	def __richcmp__(CBitset self, CBitset other, int op):
		# value comparisons
		cdef int cmp = memcmp(<void *>self.data, <void *>other.data,
					self.slots)
		if op == 2: return cmp == 0
		elif op == 3: return cmp != 0
		elif op == 0: return cmp < 0
		elif op == 4: return cmp > 0
		elif op == 1: return cmp <= 0
		return cmp >= 0

	cdef int bitcount(self):
		""" number of set bits in variable length bitvector """
		cdef int a, result = __builtin_popcountl(self.data[0])
		for a in range(1, self.slots):
			result += __builtin_popcountl(self.data[a])
		return result

	cdef int nextset(self, UInt pos):
		""" return next set bit starting from pos, -1 if there is none. """
		cdef UInt a = BITSLOT(pos), offset = pos % BITSIZE
		if self.data[a] >> offset:
			return pos + __builtin_ctzl(self.data[a] >> offset)
		for a in range(a + 1, self.slots):
			if self.data[a]: return a * BITSIZE + __builtin_ctzl(self.data[a])
		return -1

	cdef int nextunset(self, UInt pos):
		""" return next unset bit starting from pos. """
		cdef UInt a = BITSLOT(pos), offset = pos % BITSIZE
		if ~(self.data[a] >> offset):
			return pos + __builtin_ctzl(~(self.data[a] >> offset))
		a += 1
		while self.data[a] == ~0UL: a += 1
		return a * BITSIZE + __builtin_ctzl(~(self.data[a]))

	cdef void setunion(self, CBitset src):
		""" dest gets the union of dest and src; both operands must have at
		least `slots' slots. """
		cdef int a
		for a in range(self.slots): self.data[a] |= src.data[a]

	cdef bint superset(self, CBitset op):
		""" test whether `op' is a superset of this bitset; i.e., whether
		all bits of this bitset are in op. """
		cdef int a
		for a in range(self.slots):
			if self.data[a] != (self.data[a] & op.data[a]): return False
		return True

	cdef bint subset(self, CBitset op):
		""" test whether `op' is a subset of this bitset; i.e., whether
		all bits of op are in this bitset. """
		cdef int a
		for a in range(self.slots):
			if (self.data[a] & op.data[a]) != op.data[a]: return False
		return True

	cdef bint disjunct(self, CBitset op):
		""" test whether `op' is disjunct from this bitset; i.e., whether
		no bits of op are in this bitset & vice versa. """
		cdef int a
		for a in range(self.slots):
			if (self.data[a] & op.data[a]): return False
		return True

cdef class MemoryPool:
	"""A memory pool that allocates chunks of poolsize, up to limit times.
	Memory is automatically freed when object is deallocated. """
	def __cinit__(MemoryPool self, int poolsize, int limit):
		cdef int x
		self.poolsize = poolsize
		self.limit = limit
		self.n = 0
		self.pool = <void **>malloc(limit * sizeof(void *))
		assert self.pool is not NULL
		for x in range(limit): self.pool[x] = NULL
		self.cur = self.pool[0] = <ULong *>malloc(self.poolsize)
		assert self.cur is not NULL
		self.leftinpool = self.poolsize
	cdef void *malloc(MemoryPool self, int size):
		cdef void *ptr
		if size > self.poolsize: return NULL
		elif self.leftinpool < size:
			self.n += 1
			assert self.n < self.limit
			if self.pool[self.n] is NULL:
				self.pool[self.n] = <ULong *>malloc(self.poolsize)
			self.cur = self.pool[self.n]
			assert self.cur is not NULL
			self.leftinpool = self.poolsize
		ptr = self.cur
		self.cur = &((<char *>self.cur)[size])
		self.leftinpool -= size
		return ptr
	cdef void reset(MemoryPool self):
		self.n = 0
		self.cur = self.pool[0]
		self.leftinpool = self.poolsize
	def __dealloc__(MemoryPool self):
		cdef int x
		for x in range(self.n+1): free(self.pool[x])
		free(self.pool)


class OrderedSet(Set):
	""" A frozen, ordered set which maintains a regular list/tuple and set. """
	def __init__(self, iterable=None):
		if iterable:
			self.seq = tuple(iterable)
			self.theset = frozenset(self.seq)
		else:
			self.seq = ()
			self.theset = frozenset()
	def __hash__(self):
		return hash(self.theset)
	def __contains__(self, value):
		return value in self.theset
	def __len__(self):
		return len(self.theset)
	def __iter__(self):
		return iter(self.seq)
	def __getitem__(self, n):
		return self.seq[n]
	def __reversed__(self):
		return reversed(self.seq)
	def __repr__(self):
		if not self.seq:
			return '%s()' % (self.__class__.__name__,)
		return '%s(%r)' % (self.__class__.__name__, self.seq)
	def __eq__(self, other):
		#if isinstance(other, (OrderedSet, Sequence)):
		#	return len(self) == len(other) and list(self) == list(other)
		# equality is defined _without_ regard for order
		return self.theset == set(other)
	def __and__(self, other):
		""" maintain the order of the left operand. """
		if not isinstance(other, Iterable):
			return NotImplemented
		return self._from_iterable(value for value in self if value in other)
	
# some helper functions that only serve to bridge cython & python code
cpdef inline UInt getlabel(ChartItem a):
	return a.label
cpdef inline ULLong getvec(ChartItem a):
	return a.vec
cpdef inline double getscore(Edge a):
	return a.score
cpdef inline dict dictcast(d):
	return <dict>d
cpdef inline ChartItem itemcast(i):
	return <ChartItem>i
cpdef inline Edge edgecast(e):
	return <Edge>e
