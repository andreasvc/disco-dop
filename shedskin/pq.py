# Priority Queue based on binary heap which implements decrease-key
# by marking entries as invalid
# Based on notes in http://docs.python.org/library/heapq.html

from heapq import heappush, heappop, heapify
from itertools import count, imap, izip
from chartitem import ChartItem

INVALID = 0
VALUE, COUNT, KEY = range(3)

class heapdict(object):
	def __init__(self, iterable=None):
		self.heap = []						# the priority queue list
		self.counter = 1					# unique sequence count
		self.mapping = {}					# mapping of keys to entries
		if iterable:
			self.heap = [[v, n + 1, k]
							for n, (k,v) in enumerate(dict(iterable).items())]
			heapify(self.heap)
			self.mapping = dict((entry[KEY], entry) for entry in self.heap)
			self.counter += len(self.heap)

	def __setitem__(self, key, value):
		try:
			oldentry = self.mapping[key]
		except KeyError:
			entry = [value, self.counter, key]
			self.counter += 1
			self.mapping[key] = entry
			heappush(self.heap, entry)
		else:
			entry = [value, oldentry[COUNT], key]
			self.mapping[key] = entry
			heappush(self.heap, entry)
			oldentry[COUNT] = INVALID

	def __getitem__(self, key):
		return self.mapping[key][VALUE]

	def __delitem__(self, key):
		self.mapping.pop(key)[COUNT] = INVALID

	def __iter__(self):
		return iter(self.mapping)

	def __len__(self):
		return len(self.mapping)

	def keys(self):
		return self.mapping.keys()
	def values(self):
		return map(lambda x: x[VALUE], self.mapping.values())

	def itervalues(self):
		return imap(lambda x: x[VALUE], self.mapping.values())

	def items(self):
		return zip(self.keys(), self.values())

	def iteritems(self):
		return izip(self.iterkeys(), self.itervalues())

	def peekitem(self):
		while self.heap[0][COUNT] is INVALID:
			value, cnt, key = heappop(self.heap)
			del self.mapping[key]
		return self.heap[0][KEY], self.heap[0][VALUE]

	def popitem(self):
		cnt = INVALID
		while cnt is INVALID:
			value, cnt, key = heappop(self.heap)
			try: del self.mapping[key]
			except KeyError: pass
		return key, value

	def pop(self, key):
		if key is None:
			entry = self.popitem()
		else:
			entry = self.mapping.pop(key)
		entry[COUNT] = INVALID
		return entry[VALUE]

def main():
	h = heapdict()
	h[ChartItem(0, 0)] = ((0.0, 0.0), (ChartItem(0, 0), ChartItem(0, 0)))
	h[ChartItem(0, 0)] = ((0.0, 0.0), (ChartItem(0, 0),))
	h[ChartItem(2, 0)] = ((0.0, 0.0), (ChartItem(0, 0),))
	del h[ChartItem(2, 0)]
	assert ChartItem(2, 0) not in h
	assert h[ChartItem(0, 0)] == ((0.0, 0.0), (ChartItem(0, 0),))
	assert h.keys() == [ChartItem(0, 0)]
	assert h.values() == [((0.0, 0.0), (ChartItem(0, 0),))]
	assert h.items() == [(ChartItem(0,0), ((0.0, 0.0), (ChartItem(0, 0),)))]
	assert h.peekitem() == (ChartItem(0, 0), ((0.0, 0.0), (ChartItem(0, 0),)))
	assert h.popitem() == (ChartItem(0, 0), ((0.0, 0.0), (ChartItem(0, 0),)))
	h = heapdict([(ChartItem(0,0), ((0.0, 0.0), (ChartItem(0, 0),)))])
	assert h.popitem() == (ChartItem(0, 0), ((0.0, 0.0), (ChartItem(0, 0),)))
	assert len(h) == 0
	h[ChartItem(0, 0)] = ((0.0, 0.0), (ChartItem(0, 0),))
	assert h.pop(ChartItem(0, 0)) == ((0.0, 0.0), (ChartItem(0, 0),))
	assert len(h) == 0

if __name__ == '__main__': main()
