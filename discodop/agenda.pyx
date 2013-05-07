""" Priority Queue based on binary heap which implements decrease-key and
remove by marking entries as invalid. Provides dictionary-like interface. Based
on notes in the documentation for heapq, see:
http://docs.python.org/library/heapq.html

There is a version specialized to be used as agenda with edges.
"""

from __future__ import print_function
from operator import itemgetter

DEF INVALID = 0
DEF HEAP_ARITY = 4
# 2 for binary heap, 4 for quadtree heap

def getkey(Entry entry):
	return entry.key

cdef inline bint cmpfun(Entry a, Entry b):
	return (a.value < b.value or (a.value == b.value and a.count < b.count))
# this is _significantly_ faster than going through __richcmp__ of objects
cdef inline bint edgecmpfun(Entry a, Entry b):
	return ((<LCFRSEdge>a.value).score < (<LCFRSEdge>b.value).score
		or ((<LCFRSEdge>a.value).score == (<LCFRSEdge>b.value).score
		and a.count < b.count))

cdef class Agenda:
	""" Priority Queue based on binary heap which implements decrease-key and
	remove by marking entries as invalid. Provides dictionary-like interface.
	"""
	def __init__(self, iterable=None):
		""" Can be initialized with an iterable; order of equivalent values
		remains and the best priorities are retained on duplicate keys. """
		cdef Entry entry, oldentry
		self.counter = 1
		self.length = 0
		self.heap = []
		self.mapping = {}
		if iterable:
			for k, v in iterable:
				entry = new_Entry(k, v, self.counter)
				if k in self.mapping:
					oldentry = <Entry>self.mapping[k]
					if cmpfun(entry, oldentry):
						oldentry.count = INVALID
						self.mapping[k] = entry
				else:
					self.mapping[k] = entry
					self.counter += 1
				self.heap.append(entry)
			self.length = len(self.mapping)
			heapify(self.heap, cmpfun)

	cdef setitem(self, key, value):
		cdef Entry oldentry, entry
		if key in self.mapping:
			oldentry = <Entry>self.mapping[key]
			entry = <Entry>Entry.__new__(Entry)
			entry.key =  key
			entry.value = value
			entry.count = oldentry.count
			self.mapping[key] = entry
			heappush(self.heap, entry, cmpfun)
			oldentry.count = INVALID
		else:
			self.counter += 1
			self.length += 1
			entry = <Entry>Entry.__new__(Entry)
			entry.key =  key
			entry.value = value
			entry.count = self.counter
			self.mapping[key] = entry
			heappush(self.heap, entry, cmpfun)

	cdef setifbetter(self, key, value):
		""" sets an item, but only if item is new or has lower score """
		cdef Entry oldentry
		if key in self.mapping:
			oldentry = <Entry>self.mapping[key]
			if value >= oldentry.value:
				return
		self.setitem(key, value)

	cdef getitem(self, key):
		cdef Entry entry
		entry = <Entry>self.mapping[key]
		return entry.value

	cdef object replace(self, key, value):
		""" return current value for key, and also change its value.
		equivalent to vv = d[k]; d[k] = v; return vv """
		cdef Entry entry, oldentry = <Entry>self.mapping[key]
		entry = <Entry>Entry.__new__(Entry)
		entry.key =  key
		entry.value = value
		entry.count = oldentry.count
		self.mapping[key] = entry
		heappush(self.heap, entry, cmpfun)
		oldentry.count = INVALID
		return oldentry.value

	cdef Entry peekentry(self):
		cdef Entry entry
		cdef Py_ssize_t n = PyList_GET_SIZE(self.heap)
		if n == 0:
			raise IndexError("peek at empty heap")
		entry = <Entry>(self.heap[0])
		while entry.count == 0:
			if n == 1:
				raise IndexError("peek at empty heap")
			#replace first element with last element
			self.heap[0] = self.heap.pop()
			#and restore heap invariant
			siftdown(self.heap, 0, cmpfun)
			n -= 1
			entry = <Entry>(self.heap[0])
		return entry

	cdef Entry popentry(self):
		""" like popitem(), but avoids tuple construction by returning an Entry
		object """
		cdef Entry entry = <Entry>heappop(self.heap, cmpfun)
		while not entry.count:
			entry = <Entry>heappop(self.heap, cmpfun)
		del self.mapping[entry.key]
		self.length -= 1
		return entry

	cdef bint contains(self, key):
		return key in self.mapping

	def peekitem(self):
		cdef Entry entry = self.peekentry()
		return entry.key, entry.value

	# standard dict() methods
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
	def __delitem__(self, key):
		(<Entry>self.mapping[key]).count = INVALID
		self.length -= 1
		del self.mapping[key]
	def update(self, *a, **kw):
		for b in a:
			for k, v in b:
				self[k] = v
		for k, v in kw.items():
			self[k] = v
	def clear(self):
		self.counter = 1
		del self.heap[:]
		self.mapping.clear()
	def __repr__(self):
		return '%s({%s})' % (self.__class__.__name__, ", ".join(
				['%r: %r' % ((<Entry>a).key, (<Entry>a).value)
				for a in self.heap if (<Entry>a).count]))
	def __contains__(self, key): return key in self.mapping
	def __getitem__(self, key): return self.getitem(key)
	def __setitem__(self, key, value): self.setitem(key, value)
	def __str__(self): return self.__repr__()
	def __iter__(self): return iter(self.mapping)
	def __len__(self): return self.length
	def __nonzero__(self): return self.length != 0
	def keys(self): return self.mapping.keys()
	def values(self): return map(getval, self.mapping.values())
	def items(self): return zip(self.keys(), self.values())

cdef class EdgeAgenda:
	""" Priority Queue based on binary heap which implements decrease-key and
	remove by marking entries as invalid. Provides dictionary-like interface.

	This version is specialized to be used as agenda with edges.
	"""
	def __init__(self, iterable=None):
		""" Can be initialized with an iterable; order of equivalent values
		remains and the best priorities are retained on duplicate keys. """
		cdef Entry entry, oldentry
		self.counter = 1
		self.length = 0
		self.heap = []
		self.mapping = {}
		if iterable:
			for k, v in iterable:
				entry = new_Entry(k, v, self.counter)
				if k in self.mapping:
					oldentry = <Entry>self.mapping[k]
					if edgecmpfun(entry, oldentry):
						oldentry.count = INVALID
						self.mapping[k] = entry
				else:
					self.mapping[k] = entry
					self.counter += 1
				self.heap.append(entry)
			self.length = len(self.mapping)
			heapify(self.heap, edgecmpfun)

	cdef LCFRSEdge getitem(self, key):
		cdef Entry entry
		entry = <Entry>self.mapping[key]
		return <LCFRSEdge>entry.value

	cdef setifbetter(self, key, value):
		""" sets an item, but only if item is new or has lower score """
		cdef Entry oldentry
		if key in self.mapping:
			oldentry = <Entry>self.mapping[key]
			if (<LCFRSEdge>value).score >= (<LCFRSEdge>oldentry.value).score:
				return
		self.setitem(key, <LCFRSEdge>value)

	cdef LCFRSEdge replace(self, key, value):
		""" return current value for key, and also change its value.
		equivalent to vv = d[k]; d[k] = v; return vv """
		cdef Entry entry, oldentry = <Entry>self.mapping[key]
		entry = <Entry>Entry.__new__(Entry)
		entry.key =  key
		entry.value = value
		entry.count = oldentry.count
		self.mapping[key] = entry
		self.heap.append(entry)
		siftup(self.heap, 0, PyList_GET_SIZE(self.heap) - 1, edgecmpfun)
		oldentry.count = INVALID
		return <LCFRSEdge>oldentry.value

	# the following are identical except for `edgecmpfun'
	cdef Entry popentry(self):
		""" like popitem, but avoids tuple construction by returning an Entry
		object """
		cdef Entry entry = <Entry>heappop(self.heap, edgecmpfun)
		while not entry.count:
			entry = <Entry>heappop(self.heap, edgecmpfun)
		del self.mapping[entry.key]
		self.length -= 1
		return entry

	cdef Entry peekentry(self):
		cdef Entry entry
		cdef Py_ssize_t n = PyList_GET_SIZE(self.heap)
		if n == 0:
			raise IndexError("peek at empty heap")
		entry = <Entry>(self.heap[0])
		while entry.count == 0:
			if n == 1:
				raise IndexError("peek at empty heap")
			#replace first element with last element
			self.heap[0] = self.heap.pop()
			#and restore heap invariant
			siftdown(self.heap, 0, edgecmpfun)
			n -= 1
			entry = <Entry>(self.heap[0])
		return entry

	cdef setitem(self, key, value):
		cdef Entry oldentry, entry
		if key in self.mapping:
			oldentry = <Entry>self.mapping[key]
			entry = <Entry>Entry.__new__(Entry)
			entry.key =  key
			entry.value = value
			entry.count = oldentry.count
			self.mapping[key] = entry
			self.heap.append(entry)
			siftup(self.heap, 0, PyList_GET_SIZE(self.heap) - 1, edgecmpfun)
			oldentry.count = INVALID
		else:
			self.counter += 1
			self.length += 1
			entry = <Entry>Entry.__new__(Entry)
			entry.key =  key
			entry.value = value
			entry.count = self.counter
			self.mapping[key] = entry
			self.heap.append(entry)
			siftup(self.heap, 0, PyList_GET_SIZE(self.heap) - 1, edgecmpfun)

	# identical to Agenda() methods
	cdef bint contains(self, key):
		return key in self.mapping
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
	def __delitem__(self, key):
		(<Entry>self.mapping[key]).count = INVALID
		self.length -= 1
		del self.mapping[key]
	def update(self, *a, **kw):
		for b in a:
			for k, v in b:
				self[k] = v
		for k, v in kw.items():
			self[k] = v
	def clear(self):
		self.counter = 1
		del self.heap[:]
		self.mapping.clear()
	def __repr__(self):
		return '%s({%s})' % (self.__class__.__name__, ", ".join(
				['%r: %r' % ((<Entry>a).key, (<Entry>a).value)
				for a in self.heap if (<Entry>a).count]))
	def __contains__(self, key): return key in self.mapping
	def __getitem__(self, key): return self.getitem(key)
	def __setitem__(self, key, value): self.setitem(key, value)
	def __str__(self): return self.__repr__()
	def __iter__(self): return iter(self.mapping)
	def __len__(self): return self.length
	def __nonzero__(self): return self.length != 0
	def keys(self): return self.mapping.keys()
	def values(self): return map(getval, self.mapping.values())
	def items(self): return zip(self.keys(), self.values())

#a more efficient nsmallest implementation. Assumes items are Edge objects.
cdef list nsmallest(int n, object iterable):
	""" return an _unsorted_ list of the n best items in a list """
	cdef list items = list(iterable)
	if len(items) > 1:
		quickfindfirstk(items, 0, len(items) - 1, n)
	return items[:n]

cdef inline void quickfindfirstk(list items, int left, int right, int k):
	""" quicksort k-best selection """
	# select pivot index between left and right
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
cdef inline Entry heappop(list heap, CmpFun cmpfun):
	cdef Py_ssize_t n = PyList_GET_SIZE(heap)
	cdef Entry entry
	if n == 0:
		raise IndexError("pop from empty heap")
	elif n == 1:
		entry = <Entry>heap.pop()
	else:
		#replace first element with last element and restore heap invariant
		entry = <Entry>(PyList_GET_ITEM(heap, 0))
		heap[0] = heap.pop()
		siftdown(heap, 0, cmpfun)
	return entry

cdef inline void heappush(list heap, Entry entry, CmpFun cmpfun):
	# place at the end and swap with parents until heap invariant holds
	heap.append(entry)
	siftup(heap, 0, PyList_GET_SIZE(heap) - 1, cmpfun)

cdef inline void heapify(list heap, CmpFun cmpfun):
	"""Transform list into a heap, in-place, in O(len(heap)) time."""
	cdef int i
	for i in range(PyList_GET_SIZE(heap) // HEAP_ARITY, -1, -1):
		siftdown(heap, i, cmpfun)

# shifts only apply for binary tree
cdef inline int _parent(int i):
	return (i - 1) // HEAP_ARITY
	#return (i - 1) >> 1

cdef inline int _left(int i):
	return i * HEAP_ARITY + 1
	#return (i << 1) + 1

cdef inline int _right(int i):
	""" for documentation purposes; not used. """
	return i * HEAP_ARITY + 2
	#return (i + 1) << 1

cdef inline void siftdown(list heap, int pos, CmpFun cmpfun):
	cdef int startpos = pos, childpos = _left(pos), rightpos
	cdef int endpos = PyList_GET_SIZE(heap)
	cdef Entry newitem = <Entry>PyList_GET_ITEM(heap, pos)
	while childpos < endpos:
		for rightpos in range(childpos + 1, childpos + HEAP_ARITY):
			if (rightpos < endpos and
				cmpfun(<Entry>(PyList_GET_ITEM(heap, rightpos)),
					<Entry>(PyList_GET_ITEM(heap, childpos)))):
				childpos = rightpos
		heap[pos] = <Entry>PyList_GET_ITEM(heap, childpos)
		pos = childpos
		childpos = _left(pos)
	heap[pos] = newitem
	siftup(heap, startpos, pos, cmpfun)

cdef inline void siftup(list heap, int startpos, int pos, CmpFun cmpfun):
	cdef int parentpos
	cdef Entry parent, newitem = <Entry>PyList_GET_ITEM(heap, pos)
	while pos > startpos:
		parentpos = _parent(pos)
		parent = <Entry>PyList_GET_ITEM(heap, parentpos)
		if cmpfun(parent, newitem):
			break
		heap[pos] = parent
		pos = parentpos
	heap[pos] = newitem

#==============================================================================
from unittest import TestCase
testN = 1000

def getval(Entry entry):
	return entry.value

class TestHeap(TestCase):
	def check_invariants(self, Agenda h):
		for i in range(len(h)):
			if i > 0:
				self.assertTrue(getval(h.heap[_parent(i)]) <= getval(h.heap[i]))

	def make_data(self):
		from random import random
		pairs = [(random(), random()) for i in range(testN)]
		h = Agenda()
		d = {}
		for k, v in pairs:
			h[k] = v
			d[k] = v

		pairs.sort(key=itemgetter(1), reverse=True)
		return h, pairs, d

	def test_contains(self):
		h, pairs, d = self.make_data()
		h, pairs2, d = self.make_data()
		for k, v in pairs + pairs2:
			self.assertEqual(k in h, k in d)

	def test_len(self):
		h, pairs, d = self.make_data()
		self.assertEqual(len(h), len(d))

	def test_popitem(self):
		h, pairs, d = self.make_data()
		while pairs:
			v = h.popitem()
			v2 = pairs.pop(-1)
			self.assertEqual(v, v2)
			d.pop(v[0])
			self.assertEqual(len(h), len(d))
			self.assertEqual(set(h.items()), set(d.items()))
		self.assertEqual(len(h), 0)

	def test_popitem_ties(self):
		h = Agenda()
		for i in range(testN):
			h[i] = 0.
		for i in range(testN):
			k, v = h.popitem()
			self.assertEqual(v, 0.)
			self.check_invariants(h)

	def test_popitem_ties_fifo(self):
		h = Agenda()
		for i in range(testN):
			h[i] = 0.
		for i in range(testN):
			k, v = h.popitem()
			self.assertEqual(k, i)
			self.assertEqual(v, 0.)
			self.check_invariants(h)

	def test_peek(self):
		h, pairs, d = self.make_data()
		while pairs:
			v = h.peekitem()[0]
			h.popitem()
			v2 = pairs.pop(-1)
			self.assertEqual(v, v2[0])
		self.assertEqual(len(h), 0)

	def test_iter(self):
		h, pairs, d = self.make_data()
		self.assertEqual(list(h), list(d))

	def test_keys(self):
		h, pairs, d = self.make_data()
		self.assertEqual(sorted(h.keys()), sorted(d.keys()))

	def test_values(self):
		h, pairs, d = self.make_data()
		self.assertEqual(sorted(h.values()), sorted(d.values()))

	def test_items(self):
		h, pairs, d = self.make_data()
		self.assertEqual(sorted(h.items()), sorted(d.items()))

	def test_del(self):
		h, pairs, d = self.make_data()
		while pairs:
			k, v = pairs.pop(len(pairs) // 2)
			del h[k]
			del d[k]
			self.assertEqual(len(h), len(d))
			self.assertEqual(set(h.items()), set(d.items()))
		self.assertEqual(len(h), 0)

	def test_pop(self):
		h, pairs, d = self.make_data()
		while pairs:
			k, v = pairs.pop(-1)
			v2 = h.pop(k)
			v3 = d.pop(k)
			self.assertEqual(v, v2)
			self.assertEqual(v, v3)
			self.assertEqual(len(h), len(d))
			self.assertEqual(set(h.items()), set(d.items()))
		self.assertEqual(len(h), 0)

	def test_change(self):
		h, pairs, d = self.make_data()
		k, v = pairs[testN // 2]
		h[k] = 0.5
		pairs[testN // 2] = (k, 0.5)
		pairs.sort(key=itemgetter(1), reverse=True)
		while pairs:
			v = h.popitem()
			v2 = pairs.pop()
			self.assertEqual(v, v2)
		self.assertEqual(len(h), 0)

	def test_init(self):
		h, pairs, d = self.make_data()
		h = Agenda(d.items())
		while pairs:
			v = h.popitem()
			v2 = pairs.pop()
			self.assertEqual(v, v2)
			d.pop(v[0])
		self.assertEqual(len(h), len(d))
		self.assertEqual(len(h), 0)

	def test_repr(self):
		h, pairs, d = self.make_data()
		#self.assertEqual(h, eval(repr(h)))
		tmp = repr(h)	# 'Agenda({....})'
		#strip off class name
		dstr = tmp[tmp.index('(') + 1:tmp.rindex(')')]
		self.assertEqual(d, eval(dstr))

def test(verbose=False):
	import sys
	if sys.version[0] >= '3':
		import test.support as test_support # Python 3
	else:
		import test.test_support as test_support # Python 2
	test_support.run_unittest(TestHeap)

	# verify reference counting
	if verbose and hasattr(sys, "gettotalrefcount"):
		import gc
		counts = [None] * 5
		for i in xrange(len(counts)):
			test_support.run_unittest(TestHeap)
			gc.collect()
			counts[i] = sys.gettotalrefcount()
		print(counts)
