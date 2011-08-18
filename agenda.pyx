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

cdef Entry make_entry(object k, object v, unsigned long c):
	cdef Entry entry = Entry.__new__(Entry)
	entry.key = k; entry.value = v; entry.count = c
	return entry

cdef class Entry: pass #defined in pxd file

cdef class Function:
	cdef inline bint cmpfun(self, Entry a, Entry b): raise NotImplemented
cdef class EdgeCmp(Function):
	# this is _significantly_ faster than relying on __richcmp__
	cdef inline bint cmpfun(self, Entry a, Entry b):
		return (a.value.score < b.value.score
				or (a.value.score == b.value.score and a.count < b.count))
cdef class NormalCmp(Function):
	cdef inline bint cmpfun(self, Entry a, Entry b):
		return (a.value < b.value
				or (a.value == b.value and a.count < b.count))

cdef class Agenda(dict):
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
		self.cmpfun = NormalCmp()
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
			heapify(self.heap, self.cmpfun)

	def __getitem__(self, key):
		return self.getitem(key)

	def __setitem__(self, key, value):
		self.setitem(key, value)

	cdef inline getitem(self, key):
		cdef Entry entry
		entry = <Entry>self.mapping[key]
		return entry.value

	cpdef inline setitem(self, key, value):
		cdef Entry oldentry, entry
		if key in self.mapping:
			oldentry = <Entry>self.mapping[key]
			entry = <Entry>Entry.__new__(Entry)
			entry.key =  key; entry.value = value; entry.count = oldentry.count
			self.mapping[key] = entry
			heappush(self.heap, entry, self.cmpfun)
			oldentry.count = INVALID
		else:
			self.counter += 1
			self.length += 1
			entry = <Entry>Entry.__new__(Entry)
			entry.key =  key; entry.value = value; entry.count = self.counter
			self.mapping[key] = entry
			heappush(self.heap, entry, self.cmpfun)

	cpdef inline setifbetter(self, key, value):
		""" sets an item, but only if item is new or has lower score """
		cdef Entry oldentry, entry
		if key in self.mapping:
			oldentry = <Entry>self.mapping[key]
			if value >= oldentry.value: return
		self.setitem(key, value)

	def peekitem(self):
		cdef Entry entry
		while not (<Entry>(self.heap[0])).count:
			entry = <Entry>heappop(self.heap, self.cmpfun)
		return (<Entry>(self.heap[0])).key, (<Entry>(self.heap[0])).value

	cpdef Entry popentry(self):
		""" like popitem, but avoids tuple construction by returning an Entry
		object """
		cdef Entry entry
		while True:
			entry = <Entry>heappop(self.heap, self.cmpfun)
			if entry.count: break
		del self.mapping[entry.key]
		self.length -= 1
		return entry

	cdef object replace(self, key, value):
		""" return current value for key, and also change its value.
		equivalent to vv = d[k]; d[k] = v; return vv """
		cdef Entry entry, oldentry = <Entry>self.mapping[key]
		entry = <Entry>Entry.__new__(Entry)
		entry.key =  key; entry.value = value; entry.count = oldentry.count
		self.mapping[key] = entry
		heappush(self.heap, entry, self.cmpfun)
		oldentry.count = INVALID
		return oldentry.value

	def pop(self, key):
		cdef Entry entry
		if key is None:
			return self.popentry().value
		entry = <Entry>(self.mapping.pop(key))
		entry.count = INVALID
		self.length -= 1
		return entry.value

	def popitem(self):
		cdef Entry entry = self.popentry()
		return entry.key, entry.value

	def update(self, *a, **kw):
		for k, v in dict(*a, **kw).iteritems():
			self[k] = v

	def clear(self):
		self.counter = 1
		del self.heap[:]
		self.mapping.clear()

	def __contains__(self, key):
		return self.contains(key)

	cdef inline bint contains(self, key):
		return key in self.mapping

	def __delitem__(self, key):
		(<Entry>self.mapping[key]).count = INVALID
		self.length -= 1
		del self.mapping[key]

	def __iter__(self):
		return iter(self.mapping)

	def __len__(self):
		return self.length

	def __repr__(self):
		return '%s({%s})' % (self.__class__.__name__, ", ".join(
				['%r: %r' % ((<Entry>a).key, (<Entry>a).value)
				for a in self.heap if (<Entry>a).count]))

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
		return imap(lambda entry: (<Entry>entry).value, self.mapping.itervalues())

	def iteritems(self):
		return izip(self.iterkeys(), self.itervalues())

	cpdef list getheap(self):
		return self.heap
	cpdef object getkey(self, Entry entry):
		return entry.key
	cpdef object getval(self, Entry entry):
		return entry.value

cdef class EdgeAgenda(Agenda):
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
		self.cmpfun = EdgeCmp()
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
			heapify(self.heap, self.cmpfun)

	cpdef inline Edge getitem(self, key):
		cdef Entry entry
		entry = <Entry>self.mapping[key]
		return <Edge>entry.value

	cpdef inline setifbetter(self, key, value):
		""" sets an item, but only if item is new or has lower score """
		cdef Entry oldentry, entry
		if key in self.mapping:
			oldentry = <Entry>self.mapping[key]
			if (<Edge>value).score >= (<Edge>oldentry.value).score:
				return
		self.setitem(key, <Edge>value)

	cdef Edge replace(self, key, value):
		""" return current value for key, and also change its value.
		equivalent to vv = d[k]; d[k] = v; return vv """
		cdef Entry entry, oldentry = <Entry>self.mapping[key]
		entry = <Entry>Entry.__new__(Entry)
		entry.key =  key; entry.value = value; entry.count = oldentry.count
		self.mapping[key] = entry
		heappush(self.heap, entry, self.cmpfun)
		oldentry.count = INVALID
		return <Edge>oldentry.value

#a more efficient nsmallest implementation. Assumes items are Edge objects.
cdef inline list nsmallest(int n, list items):
	""" return an _unsorted_ list of the n best items in a list """
	if len(items) > 1:
		quickfindfirstk(items, 0, len(items) - 1, n)
	return items[:n]

cdef inline void quickfindfirstk(list items, int left, int right, int k):
	""" quicksort k-best selection """
	# select pivotIndex between left and right
	# middle between left & right
	cdef int pivot = left + (right - left) // 2
	cdef int pivotnewindex = partition(items, left, right, pivot)
	if pivotnewindex > k:
		if pivotnewindex - 1 > left:
			# new condition
			quickfindfirstk(items, left, pivotnewindex - 1, k)
	elif pivotnewindex < k:
		if right > pivotnewindex + 1:
			quickfindfirstk(items, pivotnewindex + 1, right, k)

cdef inline int partition(list items, int left, int right, int pivot):
	cdef Edge pivotvalue = <Edge>(items[pivot])
	# Move pivot to end
	items[pivot], items[right] = items[right], items[pivot]
	cdef int i, storeindex = left
	for i in range(left, right):
		if (<Edge>items[i]).inside < pivotvalue.inside:
			items[i], items[storeindex] = items[storeindex], items[i]
			storeindex += 1
	# Move pivot to its final place
	items[storeindex], items[right] = items[right], items[storeindex]
	return storeindex

# heap operations (adapted from heapq)
cdef inline Entry heappop(list heap, Function cmpfun):
	cdef Py_ssize_t n = list_getsize(heap)
	cdef Entry entry
	if n == 0:
		raise IndexError("pop from empty heap")
	elif n == 1:
		entry = <Entry>heap.pop()
	else:
		#replace first element with last element and restore heap invariant
		entry = <Entry>(list_getitem(heap, 0))
		heap[0] = heap.pop()
		siftup(heap, 0, cmpfun)
	return entry

cdef inline void heappush(list heap, Entry entry, Function cmpfun):
	# place at the end and swap with parents until heap invariant holds
	append(heap, entry)
	siftdown(heap, 0, list_getsize(heap) - 1, cmpfun)

cdef inline void heapify(list heap, Function cmpfun):
	"""Transform list into a heap, in-place, in O(len(heap)) time."""
	cdef int i
	for i in range(list_getsize(heap) // 2, -1, -1):
		siftup(heap, i, cmpfun)

cdef inline int _parent(int i):
	return (i - 1) >> 1

cdef inline int _left(int i):
	return (i << 1) + 1

cdef inline int _right(int i):
	return (i + 1) << 1

cdef inline void siftup(list heap, int pos, Function cmpfun):
	cdef int startpos = pos, childpos = _left(pos), rightpos
	cdef int endpos = list_getsize(heap)
	cdef Entry newitem = <Entry>list_getitem(heap, pos)
	while childpos < endpos:
		rightpos = childpos + 1
		if (rightpos < endpos and not
			cmpfun.cmpfun(<Entry>(list_getitem(heap, childpos)),
					<Entry>(list_getitem(heap, rightpos)))):
			childpos = rightpos
		heap[pos] = <Entry>list_getitem(heap, childpos)
		pos = childpos
		childpos = _left(pos)
	heap[pos] = newitem
	siftdown(heap, startpos, pos, cmpfun)

cdef inline void siftdown(list heap, int startpos, int pos, Function cmpfun):
	cdef int parentpos
	cdef Entry parent, newitem = <Entry>list_getitem(heap, pos)
	while pos > startpos:
		parentpos = _parent(pos)
		parent = <Entry>list_getitem(heap, parentpos)
		if cmpfun.cmpfun(newitem, parent):
			heap[pos] = parent
			pos = parentpos
			continue
		break
	heap[pos] = newitem
