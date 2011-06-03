
cdef class ChartItem:
	def __init__(self, label, vec):
		self.label = label
		self.vec = vec
		self._hash = hash((self.label, self.vec))
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
		self.inside = inside; self.prob = prob
		self.left = left; self.right = right
	def __hash__(self):
		return  hash((self.inside, self.prob, self.left, self.right))
	def __richcmp__(Edge self, Edge other, int op):
		# the ordering, only depends on inside probobality
		# (or only on estimate / outside score when added)
		if op == 0: return self.inside < other.inside
		elif op == 1: return self.inside <= other.inside
		elif op == 4: return self.inside > other.inside
		elif op == 5: return self.inside >= other.inside
		# (in)equality compares all elements
		# boolean trick: equality and inequality in one expression
		else: return (op == 2) == (
				(self.inside == other.inside and self.prob == other.prob
				and self.left == other.right and self.right == other.right))
	def __repr__(self):
		return "<%g, %g, [%r, %r]>" % (self.inside, self.prob,
							self.left, self.right if self.right else 'None')

cdef class Terminal:
	def __init__(self, lhs, rhs1, rhs2, word, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.word = word; self.prob = prob

cdef class Rule:
	def __init__(self, lhs, rhs1, rhs2, args, lengths, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.args = args; self.lengths = lengths; self.prob = prob
