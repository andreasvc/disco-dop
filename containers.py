
class ChartItem:
	def __init__(self, label, vec):
		self.label = label
		self.vec = vec
		self._hash = hash((self.label, self.vec))
		#self._hash = (<unsigned long>1000003 * ((<unsigned long>1000003 * <unsigned long>0x345678) ^ <long>label)) ^ (vec & ((1 << 15) - 1) + (vec >> 15))
		#if self._hash == -1: self._hash = -2
	def __hash__(self):
		return self._hash
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
		#would need bitlen for proper padding
		return "%s[%s]" % (self.label, bin(self.vec)[2:][::-1])

class Edge:
	def __init__(self, score, inside, prob, left, right):
		self.score = score; self.inside = inside; self.prob = prob
		self.left = left; self.right = right
		self._hash = hash((inside, prob, left, right))
		# this is the hash function used for tuples, apparently
		#_hash = (<unsigned long>1000003 * <unsigned long>0x345678) ^ <long>inside
		#_hash = (<unsigned long>1000003 * _hash) ^ <long>prob
		#_hash = (<unsigned long>1000003 * _hash) ^ (<ChartItem>left)._hash
		#_hash = (<unsigned long>1000003 * _hash) ^ (<ChartItem>right)._hash
		#if _hash == -1: self._hash = -2
		#else: self._hash = _hash
	def __hash__(self):
		return self._hash
	def __lt__(self, other):
		# the ordering only depends on inside probobality
		# (or only on estimate / outside score when added)
		return self.score < other.score
	def __le__(self, other):
		return self.score <= other.score
	def __ne__(self, other):
		return not self.__eq__(self, other)
	def __eq__(self, other):
		return (self.inside == other.inside
				and self.prob == other.prob
				and self.left == other.right
				and self.right == other.right)
	def __gt__(self, other):
		return self.score > other.score
	def __ge__(self, other):
		return self.score >= other.score
	def __repr__(self):
		return "<%g, %g, [%r, %s]>" % (self.inside, self.prob,
					self.left, repr(self.right) if self.right else 'None')

class Terminal:
	def __init__(self, lhs, rhs1, rhs2, word, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.word = word; self.prob = prob

class Rule:
	def __init__(self, lhs, rhs1, rhs2, args, lengths, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.args = args; self.lengths = lengths; self.prob = prob
