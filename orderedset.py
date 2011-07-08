import collections

class UniqueList(collections.MutableSet):
	""" An ordered set which maintains a regular list and set. """
	def __init__(self, iterable=None):
		self.list = []
		self.set = set()
		if iterable:
			self.list.extend(iterable)
			self.set.update(self.list)
	def __contains__(self, value):
		return value in self.set
	def __len__(self):
		return len(self.set)
	def __iter__(self):
		return iter(self.list)
	def __getitem__(self, n):
		return self.list[n]
	def __reversed__(self):
		return reversed(self.list)
	def add(self, value):
		if value not in self.set:
			self.set.add(value)
			self.list.append(value)
		else:
			print "already in set:", value
			raise ValueError("already in set")
	def discard(self, value):
		if value in self.set:
			self.set.discard(value)
			self.list.remove(value)
	def pop(self, value=None):
		if not self.list:
			raise KeyError('set is empty')
		if value is None:
			value = self.list.pop()
		else:
			value = self.list.pop(value)
		self.set.discard(value)
		return value
	def __repr__(self):
		if not self.list:
			return '%s()' % (self.__class__.__name__,)
		return '%s(%r)' % (self.__class__.__name__, self.list)
	def __eq__(self, other):
		if isinstance(other, (UniqueList, collections.Sequence)):
			return len(self) == len(other) and list(self) == list(other)
		return self.set == set(other)
	def __del__(self):
		self.set.clear()
		del self.list[:]
	
if __name__ == '__main__':
	print(UniqueList('abracadaba'))
	print(UniqueList('simsalabim'))
