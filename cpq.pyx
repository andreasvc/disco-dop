# cython: profile=False
# cython: boundscheck=False
"""
Priority Queue based on binary heap which implements decrease-key and remove
by marking entries as invalid. Provides dictionary-like interface.
Based on notes in the documentation for heapq, see:
http://docs.python.org/library/heapq.html

This version is specialised to be used as agenda with edges.
"""

from itertools import count, imap, izip
from operator import itemgetter

DEF INVALID = 0

cdef Entry make_entry(object k, Edge v, unsigned long c):
	cdef Entry entry = Entry.__new__(Entry)
	entry.key = k; entry.value = v; entry.count = c
	return entry

cdef class heapdict(dict):
	def __init__(self, iterable=None):
		""" NB: when initialized with an iterable, we don't guarantee that
			order of equivalent values in this iterable is preserved, and
			duplicate keys will be arbitrarily pruned (not necessarily keeping
			the ones with best priorities), as we use a dict to make sure we
			only have unique keys.
			if order needs to be preserved, insert them one by one. """
		cdef Entry entry
		cdef dict temp
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
		cdef Entry oldentry, entry
		if key in self.mapping:
			oldentry = <Entry>self.mapping[key]
			entry = <Entry>Entry.__new__(Entry)
			entry.key =  key; entry.value = value; entry.count = oldentry.count
			self.mapping[key] = entry
			heappush(self.heap, entry)
			oldentry.count = INVALID
		else:
			self.counter += 1
			self.length += 1
			entry = <Entry>Entry.__new__(Entry)
			entry.key =  key; entry.value = value; entry.count = self.counter
			self.mapping[key] = entry
			heappush(self.heap, entry)

	def __getitem__(self, key):
		cdef Entry entry
		entry = <Entry>self.mapping[key]
		return entry.value

	cdef inline void setitem(self, key, Edge value):
		cdef Entry oldentry, entry
		if key in self.mapping:
			oldentry = <Entry>self.mapping[key]
			entry = <Entry>Entry.__new__(Entry)
			entry.key =  key; entry.value = value; entry.count = oldentry.count
			self.mapping[key] = entry
			heappush(self.heap, entry)
			oldentry.count = INVALID
		else:
			self.counter += 1
			self.length += 1
			entry = <Entry>Entry.__new__(Entry)
			entry.key =  key; entry.value = value; entry.count = self.counter
			self.mapping[key] = entry
			heappush(self.heap, entry)

	cdef inline Edge getitem(self, key):
		cdef Entry entry
		entry = <Entry>self.mapping[key]
		return <Edge>entry.value

	cdef inline bint contains(self, key):
		return key in self.mapping

	def __delitem__(self, key):
		(<Entry>self.mapping[key]).count = INVALID
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
		return [(<Entry>e).value for e in self.mapping.values()]

	def items(self):
		return zip(self.keys(), self.values())

	def iterkeys(self):
		return self.mapping.iterkeys()

	def itervalues(self):
		return imap(lambda entry: entry.value, self.mapping.itervalues())

	def iteritems(self):
		return izip(self.iterkeys(), self.itervalues())

	def peekitem(self):
		cdef Entry entry
		while not <Entry>(self.heap[0]).count:
			entry = <Entry>heappop(self.heap)
			try: del self.mapping[entry.key]
			except KeyError: pass
		return <Entry>(self.heap[0]).key, <Entry>(self.heap[0]).value

	cpdef tuple popitem(self):
		cdef Entry entry
		entry.count = INVALID
		while not entry.count:
			try: entry = <Entry>heappop(self.heap)
			except IndexError: raise KeyError("popitem(): heapdict is empty")
			try: del self.mapping[entry.key]
			except KeyError: pass
		self.length -= 1
		return entry.key, entry.value

	cpdef Entry popentry(self):
		""" like popitem, but avoids tuple construction """
		cdef Entry entry
		entry.count = INVALID
		while not entry.count:
			try: entry = <Entry>heappop(self.heap)
			except IndexError: raise KeyError("popitem(): heapdict is empty")
			try: del self.mapping[entry.key]
			except KeyError: pass
		self.length -= 1
		return entry

	cpdef Edge replace(self, key, Edge value):
		""" return current value for key, and also change its value.
		equivalent to vv = d[k]; d[k] = v; return vv """
		oldentry = <Entry>self.mapping[key]
		entry = <Entry>Entry.__new__(Entry)
		entry.key =  key; entry.value = value; entry.count = oldentry.count
		self.mapping[key] = entry
		heappush(self.heap, entry)
		oldentry.count = INVALID
		return <Edge>oldentry.value

	def pop(self, key):
		cdef Entry entry
		if key is None:
			return self.popitem()[1]
		entry = <Entry>(self.mapping.pop(key))
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
cdef inline bint lessthan(Entry a, Entry b):
	return (a.value.score < b.value.score
				or (a.value.score == b.value.score and a.count < b.count))

# heap operations (without heapdict's reheapify, adapted from heapq)

cdef inline Entry heappop(list heap):
	cdef Entry entry = <Entry>(heap[0])
	if len(heap) == 1:
		heap.pop()
	else:
		#replace first element with last element and restore heap invariant
		heap[0] = heap.pop()
		siftup(heap, 0)
	return entry

cdef inline void heappush(list heap, Entry entry):
	# place at the end and swap with parents until heap invariant holds
	heap.append(entry)
	siftdown(heap, 0, len(heap) - 1)

cdef inline void heapify(list heap):
	"""Transform list into a heap, in-place, in O(len(heap)) time."""
	cdef int i
	for i in range(len(heap) // 2, -1, -1):
		siftup(heap, i)

cdef inline list nsmallest(int n, list items):
	""" return an _unsorted_ list of the n best items in a list """
	if len(items) > 1:
		quickfindFirstK(items, 0, len(items) - 1, n)
	return items[:n]

cdef inline void quickfindFirstK(list items, int left, int right, int k):
	""" quicksort k-best selection """
	# select pivotIndex between left and right
	# middle between left & right
	cdef int pivot = left + (right - left) // 2
	cdef int pivotNewIndex = partition(items, left, right, pivot)
	if pivotNewIndex > k:
		if pivotNewIndex - 1 > left:
			# new condition
			quickfindFirstK(items, left, pivotNewIndex - 1, k)
	elif pivotNewIndex < k:
		if right > pivotNewIndex + 1:
			quickfindFirstK(items, pivotNewIndex + 1, right, k)

cdef inline int partition(list items, int left, int right, int pivot):
	cdef Edge pivotValue = <Edge>(items[pivot])
	# Move pivot to end
	items[pivot], items[right] = items[right], items[pivot]
	cdef int i, storeIndex = left
	for i in range(left, right):
		if (<Edge>items[i]).inside < pivotValue.inside:
			items[i], items[storeIndex] = items[storeIndex], items[i]
			storeIndex += 1
	# Move pivot to its final place
	items[storeIndex], items[right] = items[right], items[storeIndex]
	return storeIndex

cdef inline int _parent(int i):
	return (i - 1) >> 1

cdef inline int _left(int i):
	return (i << 1) + 1

cdef inline int _right(int i):
	return (i + 1) << 1

cdef inline void siftup(list heap, int pos):
	cdef int startpos = pos, childpos = _left(pos), rightpos
	cdef int endpos = len(heap)
	cdef Entry newitem = heap[pos]
	while childpos < endpos:
		rightpos = childpos + 1
		if (rightpos < endpos and not lessthan(<Entry>(heap[childpos]),
												<Entry>(heap[rightpos]))):
			childpos = rightpos
		heap[pos] = heap[childpos]
		pos = childpos
		childpos = _left(pos)
	heap[pos] = newitem
	siftdown(heap, startpos, pos)

cdef inline void siftdown(list heap, int startpos, int pos):
	cdef int parentpos
	cdef Entry parent, newitem = heap[pos]
	while pos > startpos:
		parentpos = _parent(pos)
		parent = <Entry>(heap[parentpos])
		if lessthan(newitem, parent):
			heap[pos] = parent
			pos = parentpos
			continue
		break
	heap[pos] = newitem
