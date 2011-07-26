
cdef class ChartItem:
	def __init__(self, label, vec):
		self.label = label
		self.vec = vec
		#self._hash = hash((self.label, self.vec))
		self._hash = (1000003UL * ((1000003UL * 0x345678UL) ^ <long>label)) ^ (vec & ((1 << 15) - 1) + (vec >> 15))
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
	def __nonzero__(ChartItem self):
		return self.label != 0 and self.vec != 0
	def __repr__(ChartItem self):
		return "ChartItem(%d, %s)" % (self.label, bin(self.vec))

cdef class Edge:
	def __init__(self, score, inside, prob, left, right):
		cdef long _hash
		self.score = score; self.inside = inside; self.prob = prob
		self.left = left; self.right = right
		#self._hash = hash((inside, prob, left, right))
		# this is the hash function used for tuples, apparently
		_hash = (1000003UL * 0x345678UL) ^ <long>inside
		_hash = (1000003UL * _hash) ^ <long>prob
		_hash = (1000003UL * _hash) ^ (<ChartItem>left)._hash
		_hash = (1000003UL * _hash) ^ (<ChartItem>right)._hash
		if _hash == -1: self._hash = -2
		else: self._hash = _hash
	def __hash__(self):
		return self._hash
	def __richcmp__(Edge self, other, int op):
		# the ordering only depends on the estimate / inside score
		if op == 0: return self.score < (<Edge>other).score
		elif op == 1: return self.score <= (<Edge>other).score
		# (in)equality compares all elements
		# boolean trick: equality and inequality in one expression i.e., the
		# equality between the two boolean expressions acts as biconditional
		elif op == 2 or op ==3:
			return (op == 2) == (
				(self.score == (<Edge>other).score
				and self.inside == (<Edge>other).inside
				and self.prob == (<Edge>other).prob
				and self.left == (<Edge>other).right
				and self.right == (<Edge>other).right))
		elif op == 4: return self.score > other.score
		elif op == 5: return self.score >= other.score
	def __repr__(self):
		return "Edge(%g, %g, %g, %r, %r)" % (
				self.score, self.inside, self.prob, self.left, self.right)

def getlabel(ChartItem a):
	return a.label
def getvec(ChartItem a):
	return a.vec
def getscore(Edge a):
	return a.score

#cdef class RankEdge(Edge):
#	def __cinint__(self, Edge edge, int j1, int j2):
#		self.inside = ip; self.prob = edge.prob
#		self.left = edge.left; self.right = edge.right
#		self.leftrank = j1; self.rightrank = j2
#		self._hash = hash((ip, edge.prob, edge.left, edge.right, j1, j2))
#	def __repr__(self):
#		return "<%g, %g, [%r[%d], %s[%d]]>" % (self.inside, self.prob,
#					self.left, self.leftrank,
#					repr(self.right) if self.right else 'None', self.rightrank)

cdef class Terminal:
	def __init__(self, lhs, rhs1, rhs2, word, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.word = word; self.prob = prob

cdef class Rule:
	def __init__(self, lhs, rhs1, rhs2, args, lengths, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.args = args; self.lengths = lengths
		self._args = self.args._I; self._lengths = self.lengths._H
		self.prob = prob
	
cdef struct DTree:
	void *rule
	unsigned long vec
	bint islexical
	DTree *left, *right

cdef DTree new_DTree(Rule rule, unsigned long vec, bint islexical, DTree left, DTree right):
	return DTree(<void *>rule, vec, islexical, &left, &right)

