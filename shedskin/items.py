
class ChartItem(object):
	__slots__ = ("label", "vec", "_hash")
	def __init__(self, label, vec):
		self.label = label		#the category of this item (NP/PP/VP etc)
		self.vec = vec			#bitvector describing the spans of this item
		self._hash = hash((self.label, self.vec))
	def __hash__(self):
		return self._hash
	#def __cmp__(self, other):
	#	if self.label == other.label and self.vec == other.vec: return 0
	#	elif self.label < other.label or (self.label == other.label
	#								and self.vec < other.vec): return -1
	#	return 1
	def __eq__(self, other):
		if other is None: return False
		return self.label == other.label and self.vec == other.vec
	#def __lt__(self, other):
	#	if other is None: return False
	#	return self.label < other.label or (self.label == other.label
	#								and self.vec < other.vec)
	#def __gt__(self, other):
	#	if other is None: return False
	#	return self.label > other.label or (self.label == other.label
	#								and self.vec > other.vec)
	#def __ge__(self, other):
	#	if other is None: return False
	#	return self.label > other.label or (self.label == other.label
	#								and self.vec >= other.vec)
	#def __le__(self, other):
	#	if other is None: return False
	#	return self.label < other.label or (self.label == other.label
	#								and self.vec <= other.vec)
	#def __getitem__(self, n):
	#	if n == 0: return self.label
	#	elif n == 1: return self.vec
	def __repr__(self):
		#would need sentence length to properly pad with trailing zeroes
		return "%s[%s]" % (self.label, bin(self.vec)[2:][::-1])

class Edge(object):
	__slots__ = ('inside', 'prob', 'left', 'right')
	def __init__(self, inside, prob, left, right):
		self.inside = inside
		self.prob = prob
		self.left = left
		self.right = right
	def __eq__(self, other):
		return (self.inside == other.inside and self.prob == other.prob
				and self.left == other.left and self.right == other.right)

class Production: pass

class Terminal(Production):
	__slots__ = ('lhs', 'rhs1', 'rhs2', 'word', 'prob')
	def __init__(self, lhs, rhs1, word, prob):
		self.lhs = lhs
		self.rhs1 = rhs1
		self.rhs2 = 0
		self.word = word
		self.prob = prob

class Rule(Production):
	__slots__ = ('lhs', 'rhs1', 'rhs2', 'yf', 'prob')
	def __init__(self, lhs, rhs1, rhs2, yieldfunction, prob):
		self.lhs = lhs
		self.rhs1 = rhs1
		self.rhs2 = rhs2
		self.yf = yieldfunction		#??? want 2D boolean array here
		self.prob = prob

def maini():
	item = ChartItem(0, 0)
	edge = Edge(0.0, 0.0, item, item)
	t = Terminal(0, 0, ['spass', []], 0.0)
	r = Rule(0, 0, 0, [[0, 1], [1]], 0.5)
	assert edge.left == item
	assert edge == edge
	assert ChartItem(0, 0) < ChartItem(1, 0)
	assert ChartItem(1, 0) > ChartItem(0, 0)
	assert ChartItem(1, 0) >= ChartItem(0, 0)
	assert ChartItem(0, 0) <= ChartItem(0, 0)
	assert ChartItem(0, 0) <= ChartItem(1, 0)
	assert ChartItem(0, 0) == ChartItem(0, 0)
	assert ChartItem(0, 0) != ChartItem(1, 0)
	assert ChartItem(0, 0)
	print hash(item), repr(item)
	#print item[0], item[1], item.__cmp__(ChartItem(1, 0))
	print 'it worked'
if __name__ == '__main__': maini()
