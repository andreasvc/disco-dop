
cdef class ChartItem:
	def __init__(self, label, vec):
		self.label = label
		self.vec = vec
		#self._hash = hash((self.label, self.vec))
		self._hash = (<unsigned long>1000003 * ((<unsigned long>1000003 * <unsigned long>0x345678) ^ label)) ^ (vec & ((1 << 15) - 1) + (vec >> 15))
		if self._hash == -1: self._hash = -2
	def __hash__(ChartItem self):
		return self._hash
	def __richcmp__(ChartItem self, ChartItem other, int op):
		if op == 2: return self.label == other.label and self.vec == other.vec
		elif op == 3: return self.label != other.label or self.vec != other.vec
		elif op == 5: return self.label >= other.label or self.vec >= other.vec
		elif op == 1: return self.label <= other.label or self.vec <= other.vec
		elif op == 0: return self.label < other.label or self.vec < other.vec
		elif op == 4: return self.label > other.label or self.vec > other.vec
	def __getitem__(ChartItem self, int n):
		if n == 0: return self.label
		elif n == 1: return self.vec
	def __nonzero__(ChartItem self):
		return self.vec and self.label
	def __repr__(ChartItem self):
		#would need bitlen for proper padding
		return "%s[%s]" % (self.label, bin(self.vec)[2:][::-1])

cdef class Edge:
	def __init__(self, inside, prob, left, right):
		cdef long _hash
		self.inside = inside; self.prob = prob
		self.left = left; self.right = right
		#self._hash = hash((inside, prob, left, right))
		# this is the hash function used for tuples, apparently
		_hash = (<unsigned long>1000003 * <unsigned long>0x345678) ^ <long>inside
		_hash = (<unsigned long>1000003 * _hash) ^ <long>prob
		_hash = (<unsigned long>1000003 * _hash) ^ (<ChartItem>left)._hash
		_hash = (<unsigned long>1000003 * _hash) ^ (<ChartItem>right)._hash
		if _hash == -1: self._hash = -2
		else: self._hash = _hash
	def __hash__(self):
		return self._hash
	def __richcmp__(Edge self, other, int op):
		# the ordering only depends on inside probobality
		# (or only on estimate / outside score when added)
		if op == 0: return self.inside < (<Edge>other).inside
		elif op == 1: return self.inside <= (<Edge>other).inside
		# (in)equality compares all elements
		# boolean trick: equality and inequality in one expression i.e., the
		# equality between the two boolean expressions acts as biconditional
		elif op == 2 or op ==3:
			return (op == 2) == (
				(self.inside == (<Edge>other).inside
				and self.prob == (<Edge>other).prob
				and self.left == (<Edge>other).right
				and self.right == (<Edge>other).right))
		elif op == 4: return self.inside > other.inside
		elif op == 5: return self.inside >= other.inside
	def __repr__(self):
		return "<%g, %g, [%r, %s]>" % (self.inside, self.prob,
					self.left, repr(self.right) if self.right else 'None')

cdef class RankedEdge(Edge):
	def __cinint__(self, Edge edge, double ip, int j1, int j2):
		self.inside = ip; self.prob = edge.prob
		self.left = edge.left; self.right = edge.right
		self.leftrank = j1; self.rightrank = j2
		self._hash = hash((ip, edge.prob, edge.left, edge.right, j1, j2))
	def __repr__(self):
		return "<%g, %g, [%r[%d], %s[%d]]>" % (self.inside, self.prob,
					self.left, self.leftrank,
					repr(self.right) if self.right else 'None', self.rightrank)

cdef class Terminal:
	def __init__(self, lhs, rhs1, rhs2, word, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.word = word; self.prob = prob

cdef class Rule:
	def __init__(self, lhs, rhs1, rhs2, args, lengths, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.args = args; self.lengths = lengths; self.prob = prob
