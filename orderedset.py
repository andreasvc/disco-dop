import collections

class OrderedSet(collections.Set):
	""" A frozen, ordered set which maintains a regular list/tuple and set. """
	def __init__(self, iterable=None):
		if iterable:
			self.seq = tuple(iterable)
			self.theset = frozenset(self.seq)
		else:
			self.seq = ()
			self.theset = frozenset()
	def __hash__(self):
		return hash(self.seq)
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
		if isinstance(other, (OrderedSet, collections.Sequence)):
			return len(self) == len(other) and list(self) == list(other)
		return self.theset == set(other)
	
if __name__ == '__main__':
	print(OrderedSet('abracadaba'))
	print(OrderedSet('simsalabim'))
