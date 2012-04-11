from nltk import Tree

cdef class ChartItem:
	def __init__(self, label, vec):
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
	def __cinit__(self, ChartItem head, Edge edge, int j1, int j2):
		self.head = head; self.edge = edge
		self.left = j1; self.right = j2
	def __hash__(self):
		cdef long h
		#h = hash((head, edge, j1, j2))
		h = (1000003UL * 0x345678UL) ^ hash(self.head)
		h = (1000003UL * h) ^ hash(self.edge)
		h = (1000003UL * h) ^ self.left
		h = (1000003UL * h) ^ self.right
		if h == -1: h = -2
		return h
	def __richcmp__(self, RankedEdge other, int op):
		if op == 2 or op == 3:
			return (op == 2) == (
				self.left == other.left
				and self.right == other.right
				and self.head == other.head
				and self.edge == other.edge)
		else:
			raise NotImplemented
	def __repr__(self):
		return "RankedEdge(%r, %r, %d, %d)" % (
					self.head, self.edge, self.left, self.right)

cdef class LexicalRule:
	def __init__(self, lhs, rhs1, rhs2, word, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.word = word; self.prob = prob

cdef class Rule:
	def __init__(self, lhs, rhs1, rhs2, args, lengths, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.args = args; self.lengths = lengths
		self._args = self.args._I; self._lengths = self.lengths._H
		self.prob = prob

cdef class Ctrees:
	"""auxiliary class to be able to pass around collections of trees in
	Python"""
	def __init__(self, list trees=None, dict labels=None, dict prods=None):
		self.len=0; self.max=0
		if trees is None: return
		self.alloc(len(trees), sum(map(len, trees)), len(max(trees, key=len)))
		for tree in trees: self.add(tree, labels, prods)
	cpdef alloc(self, int numtrees, int numnodes, int maxnodes):
		""" Initialize an array of trees of nodes structs. """
		self.max = numtrees
		self.maxnodes = maxnodes
		self.data = <NodeArray *>malloc(numtrees * sizeof(NodeArray))
		assert self.data != NULL
		self.data[0].nodes = <Node *>malloc(numnodes * sizeof(Node))
		assert self.data[0].nodes != NULL
	cpdef add(self, list tree, dict labels, dict prods):
		""" Trees can be incrementally added to the node array; useful
		when dealing with large numbers of NLTK trees (say 100,000)."""
		assert self.len < self.max, ("either no space left (len >= max) or "
			"alloc() has not been called (max=0). max = %d" % self.max)
		assert len(tree) <= self.maxnodes, (
			"Tree too large. Nodes: %d, MAXNODE: %d." % (
			len(tree), self.maxnodes))
		self.data[self.len].len = len(tree)
		if self.len: # derive pointer from previous tree offset by its size
			self.data[self.len].nodes = &(
				self.data[self.len-1].nodes)[self.data[self.len-1].len]
		indices(tree, labels, prods, self.data[self.len].nodes)
		self.data[self.len].root = tree[0].root
		self.len += 1
	def __dealloc__(self):
		free(self.data[0].nodes)
		free(self.data)
	def __len__(self): return self.len

cdef inline void indices(tree, dict labels, dict prods, Node *result):
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

class Terminal():
	"""auxiliary class to be able to add indices to terminal nodes of NLTK
	trees"""
	def __init__(self, node): self.prod = self.node = node
	def __repr__(self): return repr(self.node)
	def __hash__(self): return hash(self.node)

cdef class CBitset:
	"""auxiliary class to be able to pass around bitsets in Python"""
	cdef inline sethash(self):
		cdef int n
		self._hash = 5381
		for n in range(self.SLOTS * sizeof(ULong)):
			self._hash = self._hash * 33 ^ <char>self.data[n]
	def __hash__(self): return self._hash
	def __richcmp__(self, CBitset other, int op):
		cdef int cmp = memcmp(<void *>self.data, <void *>other.data,
					self.SLOTS * sizeof(ULong))
		if op == 2: return cmp == 0
		elif op == 3: return cmp != 0
		elif op == 0: return cmp < 0
		elif op == 4: return cmp > 0
		elif op == 1: return cmp <= 0
		return cmp >= 0
	
# some helper functions that only serve to bridge cython & python code
cpdef inline unsigned int getlabel(ChartItem a):
	return a.label
cpdef inline unsigned long long getvec(ChartItem a):
	return a.vec
cpdef inline double getscore(Edge a):
	return a.score
cpdef inline dict dictcast(d):
	return <dict>d
cpdef inline ChartItem itemcast(i):
	return <ChartItem>i
cpdef inline Edge edgecast(e):
	return <Edge>e
