
class ChartItem:
	__slots__ = ('label', 'vec')
	def __init__(self, label, vec):
		self.label = label
		self.vec = vec
	def __hash__(self):
		# juxtapose bits of label and vec, rotating vec if > 33 words
		h = self.label ^ (self.vec << 31) ^ (self.vec >> 31)
		return -2 if h == -1 else h
	def __eq__(self, other): return self.label == other.label and self.vec == other.vec
	def __ne__(self, other): return self.label != other.label or self.vec != other.vec
	def __ge__(self, other): return self.label >= other.label or self.vec >= other.vec
	def __le__(self, other): return self.label <= other.label or self.vec <= other.vec
	def __lt__(self, other): return self.label < other.label or self.vec < other.vec
	def __gt__(self, other): return self.label > other.label or self.vec > other.vec
	def __getitem__(self, n):
		if n == 0: return self.label
		elif n == 1: return self.vec
	def __nonzero__(self):
		return self.vec and self.label
	def __repr__(self):
		return "ChartItem(%s, %s)" % (self.label, bin(self.vec))

class Edge:
	__slots__ = ('score', 'inside', 'prob', 'left', 'right')
	def __init__(self, score, inside, prob, left, right):
		self.score = score; self.inside = inside; self.prob = prob
		self.left = left; self.right = right
	def __hash__(self):
		return hash((self.inside, self.prob, self.left, self.right))
	def __lt__(self, other):
		# the ordering only depends on inside probobality
		# (or only on estimate / outside score when added)
		return self.score < other.score
	def __le__(self, other):
		return self.score <= other.score
	def __ne__(self, other):
		return not self.__eq__(other)
	def __eq__(self, other):
		return (self.inside == other.inside
				and self.prob == other.prob
				and self.left == other.left
				and self.right == other.right)
	def __gt__(self, other):
		return self.score > other.score
	def __ge__(self, other):
		return self.score >= other.score
	def __repr__(self):
		return "Edge(%g, %g, %g, %r, %r)" % (self.score, self.inside,
					self.prob,
					self.left,
					self.right)
					#repr(self.right) if self.right else 'None')

class RankedEdge:
	def __init__(self, head, edge, j1, j2):
		self.head = head; self.edge = edge
		self.left = j1; self.right = j2
	def __hash__(self):
		#h = hash((head, edge, j1, j2))
		h = (1000003L * 0x345678L) ^ hash(self.head)
		h = (1000003L * h) ^ hash(self.edge)
		h = (1000003L * h) ^ self.left
		h = (1000003L * h) ^ self.right
		if h == -1: h = -2
		return h
	def __eq__(self, other):
		return (isinstance(other, RankedEdge) and self.left == other.left
			and self.right == other.right and self.head == other.head
			and self.edge == other.edge)
	def __repr__(self):
		return "RankedEdge(%r, %r, %d, %d)" % (
					self.head, self.edge, self.left, self.right)

class Terminal:
	__slots__ = ('lhs', 'rhs1', 'rhs2', 'word', 'prob')
	def __init__(self, lhs, rhs1, rhs2, word, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.word = word; self.prob = prob
	def __repr__(self):
		return "Terminal" + repr((self.lhs, self.rhs1, self.rhs2,
				self.word, self.prob))

class Rule:
	__slots__ = ('lhs', 'rhs1', 'rhs2', 'prob',
				'args', 'lengths', '_args', 'lengths')
	def __init__(self, lhs, rhs1, rhs2, args, lengths, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.args = args; self.lengths = lengths; self.prob = prob
		self._args = self.args; self._lengths = self.lengths
	def __repr__(self):
		return "Rule" + repr((self.lhs, self.rhs1, self.rhs2,
							self.args, self.lengths, self.prob))

def getlabel(a): return a.label
def getvec(a): return a.vec
def getscore(a): return a.score
def dictcast(d): return d
def itemcast(d): return d
def edgecast(d): return d

if __name__ == '__main__':
	from array import array
	c = ChartItem(0, 0)
	e = Edge(0., 0., 0., c, c)
	h = hash(c); h = hash(e)
	c[0]; c[1]; bool(c); repr(c); repr(e)
	e < e; e > e; e <= e; e == e; e >= e; e != e
	t = Terminal(0, 0, 0, "foo", 0.)
	r = Rule(0, 0, 0, array('H', [0, 0, 0]), array('I', [0, 0, 0]), 0.)
