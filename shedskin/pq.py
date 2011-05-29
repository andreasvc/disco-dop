# Priority Queue based on binary heap which implements decrease-key
# by marking entries as invalid
# Based on notes in http://docs.python.org/library/heapq.html

from heapq import heappush, heappop, heapify
from itertools import count, imap, izip
from chartitem import ChartItem

INVALID = 0
VALUE, COUNT, KEY = range(3)

class Entry(object):
	__slots__ = ('key', 'value', 'count')
	def __init__(self, key, value, count):
		self.key = key
		self.value = value
		self.count = count

class heapdict(object):
	def __init__(self, iterable=None):
		self.heap = []						# the priority queue list
		self.counter = 1					# unique sequence count
		self.mapping = {}					# mapping of keys to entries
		if iterable:
			self.heap = [Entry(k, v, n + 1)
							for n, (k,v) in enumerate(dict(iterable).items())]
			heapify(self.heap)
			self.mapping = dict((entry.key, entry) for entry in self.heap)
			self.counter += len(self.heap)

	def __setitem__(self, key, value):
		if key in self.mapping:
			oldentry = self.mapping[key]
			entry = Entry(key, value, oldentry.count)
			self.mapping[key] = entry
			heappush(self.heap, entry)
			oldentry.count = INVALID
		else:
			entry = Entry(key, value, self.counter)
			self.counter += 1
			self.mapping[key] = entry
			heappush(self.heap, entry)

	def __getitem__(self, key):
		return self.mapping[key].value

	def __delitem__(self, key):
		self.mapping.pop(key).count = INVALID

	def __iter__(self):
		return iter(self.mapping)

	def __len__(self):
		return len(self.mapping)

	def keys(self):
		return self.mapping.keys()

	def values(self):
		return map(lambda x: x.value, self.mapping.values())

	def itervalues(self):
		return imap(lambda x: x.value, self.mapping.values())

	def items(self):
		return zip(self.keys(), self.values())

	def iteritems(self):
		return izip(self.iterkeys(), self.itervalues())

	def peekitem(self):
		while self.heap[0].count is INVALID:
			entry = heappop(self.heap)
			del self.mapping[entry.key]
		return self.heap[0].key, self.heap[0].value

	def popitem(self):
		entry = heappop(self.heap)
		try: del self.mapping[entry.key]
		except KeyError: pass
		while entry.count is INVALID:
			entry = heappop(self.heap)
			try: del self.mapping[entry.key]
			except KeyError: pass
		return entry.key, entry.value

	def pop(self, key):
		entry = self.mapping.pop(key)
		entry.count = INVALID
		return entry.value

def main():
	h = heapdict()
	e = Entry(ChartItem(0, 0), ((0.0, 0.0), (ChartItem(0, 0), None)), 1)
	e = Entry(ChartItem(0, 0), ((0.0, 0.0), (ChartItem(0, 0), ChartItem(0, 0))), 1)
	h[ChartItem(0, 0)] = ((0.0, 0.0), (ChartItem(0, 0), ChartItem(0, 0)))
	h[ChartItem(0, 0)] = ((0.0, 0.0), (ChartItem(0, 0), None))
	h[ChartItem(2, 0)] = ((0.0, 0.0), (ChartItem(0, 0), None))
	del h[ChartItem(2, 0)]
	assert ChartItem(2, 0) not in h
	assert h[ChartItem(0, 0)] == ((0.0, 0.0), (ChartItem(0, 0), None))
	assert h.keys() == [ChartItem(0, 0)]
	assert h.values() == [((0.0, 0.0), (ChartItem(0, 0), None))]
	assert h.items() == [(ChartItem(0,0), ((0.0, 0.0), (ChartItem(0, 0), None)))]
	assert h.peekitem() == (ChartItem(0, 0), ((0.0, 0.0), (ChartItem(0, 0), None)))
	assert h.popitem() == (ChartItem(0, 0), ((0.0, 0.0), (ChartItem(0, 0), None)))
	h = heapdict([(ChartItem(0,0), ((0.0, 0.0), (ChartItem(0, 0), None)))])
	assert h.popitem() == (ChartItem(0, 0), ((0.0, 0.0), (ChartItem(0, 0), None)))
	assert len(h) == 0
	h[ChartItem(0, 0)] = ((0.0, 0.0), (ChartItem(0, 0), None))
	assert h.pop(ChartItem(0, 0)) == ((0.0, 0.0), (ChartItem(0, 0), None))
	assert len(h) == 0

if __name__ == '__main__': main()
