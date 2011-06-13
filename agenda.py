"""
Priority Queue based on binary heap which implements decrease-key and remove
by marking entries as invalid. Provides dictionary-like interface.
Based on notes in the documentation for heapq, see:
http://docs.python.org/library/heapq.html

This version is specialised to be used as agenda with edges.
"""

from itertools import count, imap, izip
from operator import itemgetter

INVALID = 0

class Entry:
	__slots__ = ('key', 'value', 'count')

def make_entry(k, v, c):
	#entry = Entry.__new__(Entry)
	entry = Entry()
	entry.key = k; entry.value = v; entry.count = c
	return entry

class heapdict(dict):
	def __init__(self, iterable=None):
		""" NB: when initialized with an iterable, we don't guarantee that
			order of equivalent values in this iterable is preserved, and
			duplicate keys will be arbitrarily pruned (not necessarily keeping
			the ones with best priorities), as we use a dict to make sure we
			only have unique keys.
			if order needs to be preserved, insert them one by one. """
		self.counter = 1
		self.length = 0
		self.heap = []
		self.mapping = {}
		if iterable:
			# put elements in a dictionary to remove duplicate keys
			temp = dict(iterable)
			self.length = len(temp)
			self.counter = self.length + 1
			for i in range(1, self.counter):
				k, v = temp.popitem()
				entry = make_entry(k, v, i)
				self.mapping[k] = entry
				self.heap.append(entry)
			assert temp == {}
			heapify(self.heap)

	def __setitem__(self, key, value):
		if key in self.mapping:
			oldentry = self.mapping[key]
			entry = Entry() #Entry.__new__(Entry)
			entry.key =  key; entry.value = value; entry.count = oldentry.count
			self.mapping[key] = entry
			heappush(self.heap, entry)
			oldentry.count = INVALID
		else:
			self.counter += 1
			self.length += 1
			entry = Entry() #Entry.__new__(Entry)
			entry.key =  key; entry.value = value; entry.count = self.counter
			self.mapping[key] = entry
			heappush(self.heap, entry)

	def __getitem__(self, key):
		entry = self.mapping[key]
		return entry.value

	def setitem(self, key, value):
		if key in self.mapping:
			oldentry = self.mapping[key]
			entry = Entry() #Entry.__new__(Entry)
			entry.key =  key; entry.value = value; entry.count = oldentry.count
			self.mapping[key] = entry
			heappush(self.heap, entry)
			oldentry.count = INVALID
		else:
			self.counter += 1
			self.length += 1
			entry = Entry() #Entry.__new__(Entry)
			entry.key =  key; entry.value = value; entry.count = self.counter
			self.mapping[key] = entry
			heappush(self.heap, entry)

	def setifbetter(self, key, value):
		""" sets an item, but only if item is new or has lower score """
		if key in self.mapping:
			oldentry = self.mapping[key]
			if value.score >= oldentry.value.score: return
			entry = Entry() #Entry.__new__(Entry)
			entry.key = key; entry.value = value; entry.count = oldentry.count
			self.mapping[key] = entry
			heappush(self.heap, entry)
			oldentry.count = INVALID
		else:
			self.counter += 1
			self.length += 1
			entry = Entry() #Entry.__new__(Entry)
			entry.key =  key; entry.value = value; entry.count = self.counter
			self.mapping[key] = entry
			heappush(self.heap, entry)

	def getitem(self, key):
		entry = self.mapping[key]
		return entry.value

	def contains(self, key):
		return key in self.mapping

	def __delitem__(self, key):
		(self.mapping[key]).count = INVALID
		self.length -= 1
		del self.mapping[key]

	def __contains__(self, key):
		return key in self.mapping

	def __iter__(self):
		return iter(self.mapping)

	def __len__(self):
		return self.length #len(self.mapping)

	def __repr__(self):
		return '%s({%s})' % (self.__class__.__name__, ", ".join(
				['%r: %r' % (a.key, a.value) for a in self.heap if a.count]))

	def __str__(self):
		return self.__repr__()

	def keys(self):
		return self.mapping.keys()

	def values(self):
		return [(e).value for e in self.mapping.values()]

	def items(self):
		return zip(self.keys(), self.values())

	def iterkeys(self):
		return self.mapping.iterkeys()

	def itervalues(self):
		return imap(lambda entry: entry.value, self.mapping.itervalues())

	def iteritems(self):
		return izip(self.iterkeys(), self.itervalues())

	def peekitem(self):
		while not (self.heap[0]).count:
			entry = heappop(self.heap)
			try: del self.mapping[entry.key]
			except KeyError: pass
		return (self.heap[0]).key, (self.heap[0]).value

	def popitem(self):
		entry = self.popentry()
		return entry.key, entry.value

	def popentry(self):
		""" like popitem, but avoids tuple construction """
		while True:
			try: entry = heappop(self.heap)
			except IndexError: raise KeyError("popitem(): heapdict is empty")
			if entry.count: break
		self.length -= 1
		return entry

	def replace(self, key, value):
		""" return current value for key, and also change its value.
		equivalent to vv = d[k]; d[k] = v; return vv """
		oldentry = self.mapping[key]
		entry = Entry() #Entry.__new__(Entry)
		entry.key =  key; entry.value = value; entry.count = oldentry.count
		self.mapping[key] = entry
		heappush(self.heap, entry)
		oldentry.count = INVALID
		return oldentry.value

	def pop(self, key):
		if key is None:
			return self.popentry().value
		entry = self.mapping.pop(key)
		entry.count = INVALID
		self.length -= 1
		return entry.value

	def update(self, *a, **kw):
		for k, v in dict(*a, **kw).iteritems():
			self[k] = v

	def clear(self):
		self.counter = 1
		del self.heap[:]
		self.mapping.clear()

# this is _significantly_ faster than relying on __richcmp__
def lessthan(a, b):
	return (a.value.score < b.value.score
				or (a.value.score == b.value.score and a.count < b.count))

# heap operations (without heapdict's reheapify, adapted from heapq)

def heappop(heap):
	entry = (heap[0])
	if len(heap) == 1:
		heap.pop()
	else:
		#replace first element with last element and restore heap invariant
		heap[0] = heap.pop()
		siftup(heap, 0)
	return entry

def heappush(heap, entry):
	# place at the end and swap with parents until heap invariant holds
	heap.append(entry)
	siftdown(heap, 0, len(heap) - 1)

def heapify(heap):
	"""Transform list into a heap, in-place, in O(len(heap)) time."""
	for i in range(len(heap) // 2, -1, -1):
		siftup(heap, i)

def nsmallest(n, items):
	""" return an _unsorted_ list of the n best items in a list """
	if len(items) > 1:
		quickfindFirstK(items, 0, len(items) - 1, n)
	return items[:n]

def quickfindFirstK(items, left, right, k):
	""" quicksort k-best selection """
	# select pivotIndex between left and right
	# middle between left & right
	pivot = left + (right - left) // 2
	pivotNewIndex = partition(items, left, right, pivot)
	if pivotNewIndex > k:
		if pivotNewIndex - 1 > left:
			# new condition
			quickfindFirstK(items, left, pivotNewIndex - 1, k)
	elif pivotNewIndex < k:
		if right > pivotNewIndex + 1:
			quickfindFirstK(items, pivotNewIndex + 1, right, k)

def partition(items, left, right, pivot):
	pivotValue = items[pivot]
	# Move pivot to end
	items[pivot], items[right] = items[right], items[pivot]
	storeIndex = left
	for i in range(left, right):
		if items[i].inside < pivotValue.inside:
			items[i], items[storeIndex] = items[storeIndex], items[i]
			storeIndex += 1
	# Move pivot to its final place
	items[storeIndex], items[right] = items[right], items[storeIndex]
	return storeIndex

def _parent(i):
	return (i - 1) >> 1

def _left(i):
	return (i << 1) + 1

def _right(i):
	return (i + 1) << 1

def siftup(heap, pos):
	startpos = pos; childpos = _left(pos)
	endpos = len(heap)
	newitem = heap[pos]
	while childpos < endpos:
		rightpos = childpos + 1
		if (rightpos < endpos and not lessthan((heap[childpos]),
												(heap[rightpos]))):
			childpos = rightpos
		heap[pos] = heap[childpos]
		pos = childpos
		childpos = _left(pos)
	heap[pos] = newitem
	siftdown(heap, startpos, pos)

def siftdown(heap, startpos, pos):
	newitem = heap[pos]
	while pos > startpos:
		parentpos = _parent(pos)
		parent = (heap[parentpos])
		if lessthan(newitem, parent):
			heap[pos] = parent
			pos = parentpos
			continue
		break
	heap[pos] = newitem
