
class AOUOEUAEOUTHAOEAOEUT(object): pass

class ChartItem(AOUOEUAEOUTHAOEAOEUT):
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
	def __lt__(self, other):
		if other is None: return False
		return self.label < other.label or (self.label == other.label
									and self.vec < other.vec)
	def __gt__(self, other):
		if other is None: return False
		return self.label > other.label or (self.label == other.label
									and self.vec > other.vec)
	def __ge__(self, other):
		if other is None: return False
		return self.label > other.label or (self.label == other.label
									and self.vec >= other.vec)
	def __le__(self, other):
		if other is None: return False
		return self.label < other.label or (self.label == other.label
									and self.vec <= other.vec)
	#def __getitem__(self, n):
	#	if n == 0: return self.label
	#	elif n == 1: return self.vec
	def __repr__(self):
		#would need sentence length to properly pad with trailing zeroes
		return "%s[%s]" % (self.label, bin(self.vec)[2:][::-1])

class NoChartItem(AOUOEUAEOUTHAOEAOEUT):
	def __init__(self): pass
	def __hash__(self): return hash(0)
	def __eq__(self, other):
		return ((not isinstance(other, ChartItem))
			and isinstance(other, AOUOEUAEOUTHAOEAOEUT))

def main():
	item = ChartItem(0, 0)
	assert ChartItem(0, 0) < ChartItem(1, 0)
	assert ChartItem(1, 0) > ChartItem(0, 0)
	assert ChartItem(1, 0) >= ChartItem(0, 0)
	assert ChartItem(0, 0) <= ChartItem(0, 0)
	assert ChartItem(0, 0) <= ChartItem(1, 0)
	assert ChartItem(0, 0) == ChartItem(0, 0)
	assert ChartItem(0, 0) != ChartItem(1, 0)
	assert ChartItem(0, 0)
	assert ChartItem(0, 0) != NoChartItem()
	print hash(item), repr(item), hash(NoChartItem())
	#print item[0], item[1], item.__cmp__(ChartItem(1, 0))
	print 'it worked'
if __name__ == '__main__': main()
