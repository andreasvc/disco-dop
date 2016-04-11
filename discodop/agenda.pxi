"""Priority Queue, quick select, and n-way merge using D-ary array-based heap.

Based on source and notes in:

- http://docs.python.org/library/heapq.html
- https://en.wikipedia.org/wiki/D-ary_heap
- https://github.com/valyala/gheap
- https://en.wikipedia.org/wiki/Quickselect"""

from operator import itemgetter
include "constants.pxi"

DEF INVALID = 0


@cython.final
cdef class Entry:
	def __repr__(self):
		return '%s(%r, %r, %r)' % (
				self.__class__.__name__, self.key, self.value, self.count)


@cython.final
cdef class DoubleEntry:
	def __repr__(self):
		return '%s(%r, %r, %r)' % (
				self.__class__.__name__, self.key, self.value, self.count)


cdef inline bint cmpfun(Entry_fused a, Entry_fused b):
	"""Generic comparison function for Entry objects."""
	return (a.value < b.value or (a.value == b.value and a.count < b.count))


cdef class Agenda:
	"""Priority Queue implemented with array-based n-ary heap.

	Implements decrease-key and remove operations by marking entries as
	invalid. Provides dictionary-like interface.

	Can be initialized with an iterable; equivalent values are preserved in
	insertion order and the best priorities are retained on duplicate keys."""
	def __init__(self, iterable=None):
		cdef Entry entry = None, oldentry
		self.counter = 1
		self.length = 0
		self.heap = []
		self.mapping = {}
		if iterable:
			for k, v in iterable:
				entry = new_Entry(k, v, self.counter)
				if k in self.mapping:
					oldentry = self.mapping[k]
					if cmpfun(entry, oldentry):
						entry.count = oldentry.count
						oldentry.count = INVALID
						self.mapping[k] = entry
						self.heap.append(entry)
				else:
					self.mapping[k] = entry
					self.heap.append(entry)
					self.counter += 1
			self.length = len(self.mapping)
			heapify(self.heap, entry)

	def peekitem(self):
		"""Get the current best (key, value) pair, while keeping it on the
		agenda."""
		cdef Entry entry
		cdef Py_ssize_t n = PyList_GET_SIZE(self.heap)
		if n == 0:
			raise IndexError("peek at empty heap")
		entry = self.heap[0]
		while entry.count == 0:
			if n == 1:
				raise IndexError("peek at empty heap")
			# replace first element with last element
			self.heap[0] = self.heap.pop()
			# and restore heap invariant
			siftup(self.heap, 0, entry)
			n -= 1
			entry = self.heap[0]
		return entry.key, entry.value

	# standard dict() methods
	def pop(self, key):
		""":returns: value for agenda[key] and remove it."""
		cdef Entry entry
		if key is None:
			return self.popitem()[1]
		entry = self.mapping.pop(key)
		entry.count = INVALID
		self.length -= 1
		return entry.value

	def popitem(self):
		""":returns: best scoring (key, value) pair; removed from agenda."""
		cdef Entry entry = None
		entry = heappop(self.heap, entry)
		while not entry.count:
			entry = heappop(self.heap, entry)
		del self.mapping[entry.key]
		self.length -= 1
		return entry.key, entry.value

	def update(self, *a, **kw):
		"""Change score of items given a sequence of (key, value) pairs."""
		for b in a:
			for k, v in b:
				self[k] = v
		for k, v in kw.items():
			self[k] = v

	def clear(self):
		"""Remove all items from agenda."""
		self.counter = 1
		del self.heap[:]
		self.mapping.clear()

	def __contains__(self, key):
		return key in self.mapping

	def __getitem__(self, key):
		return self.mapping[key].value

	def __setitem__(self, key, value):
		cdef Entry oldentry, entry = new_Entry(key, value, self.counter)
		if key in self.mapping:
			oldentry = <Entry>self.mapping[key]
			entry.count = oldentry.count
			oldentry.count = INVALID
		else:
			self.length += 1
			self.counter += 1
		self.mapping[key] = entry
		heappush(self.heap, entry)

	def __delitem__(self, key):
		"""Remove key from heap."""
		(<Entry>self.mapping[key]).count = INVALID
		self.length -= 1
		del self.mapping[key]

	def __repr__(self):
		return '%s({%s})' % (self.__class__.__name__, ", ".join(
				['%r: %r' % (a.key, a.value) for a in self.heap if a.count]))

	def __str__(self):
		return self.__repr__()

	def __iter__(self):
		return iter(self.mapping)

	def __len__(self):
		return self.length

	def __bool__(self):
		return self.length != 0

	def keys(self):
		""":returns: keys in agenda."""
		return self.mapping.keys()

	def values(self):
		""":returns: values in agenda."""
		return [entry.value for entry in self.mapping.values()]

	def items(self):
		""":returns: (key, value) pairs in agenda."""
		return zip(self.keys(), self.values())


# Unfortunately it seems we cannot template a cdef class with a fused type yet,
# otherwise Agenda[double] would have been possible.
@cython.final
cdef class DoubleAgenda(Agenda):
	"""Priority Queue where priorities are C doubles.

	Implements decrease-key and remove operations by marking entries as
	invalid. Provides dictionary-like interface.

	Can be initialized with an iterable of DoubleEntry objects; order of
	equivalent values remains and the best priorities are retained on duplicate
	keys.

	This version is specialized to be used as agenda with C doubles as
	priorities (values); keys are hashable Python objects."""
	def __init__(self, iterable=None):
		cdef DoubleEntry entry = None, oldentry
		self.counter = 1
		self.length = 0
		self.heap = []
		self.mapping = {}
		if iterable:
			for k, v in iterable:
				entry = new_DoubleEntry(k, v, self.counter)
				if k in self.mapping:
					oldentry = <DoubleEntry>self.mapping[k]
					if cmpfun(entry, oldentry):
						entry.count = oldentry.count
						oldentry.count = INVALID
						self.mapping[k] = entry
						self.heap.append(entry)
				else:
					self.mapping[k] = entry
					self.heap.append(entry)
					self.counter += 1
			self.length = len(self.mapping)
			heapify(self.heap, entry)

	cdef double getitem(self, key):
		"""Like agenda[key], but bypass Python API."""
		return (<DoubleEntry>self.mapping[key]).value

	cdef inline void setifbetter(self, key, double value):
		"""Sets an item, but only if item is new or has lower score."""
		cdef DoubleEntry oldentry
		if key in self.mapping:
			oldentry = <DoubleEntry>self.mapping[key]
			if value >= oldentry.value:
				return
		self.setitem(key, value)

	cdef double replace(self, key, double value):
		"""Return current value for key, and also change its value.

		Equivalent to vv = d[k]; d[k] = v; return vv"""
		cdef DoubleEntry entry, oldentry = <DoubleEntry>self.mapping[key]
		entry = new_DoubleEntry(key, value, oldentry.count)
		self.mapping[key] = entry
		self.heap.append(entry)
		siftdown(self.heap, 0, PyList_GET_SIZE(self.heap) - 1, entry)
		oldentry.count = INVALID
		return oldentry.value

	# the following are methods specialized for `DoubleEntry` objects
	cdef DoubleEntry popentry(self):
		"""Like popitem, but avoid tuple construction by returning a
		DoubleEntry object."""
		cdef DoubleEntry entry = None
		entry = <DoubleEntry>heappop(self.heap, entry)
		while not entry.count:
			entry = <DoubleEntry>heappop(self.heap, entry)
		del self.mapping[entry.key]
		self.length -= 1
		return entry

	cdef DoubleEntry peekentry(self):
		"""Get the current best entry, while keeping it on the agenda."""
		cdef DoubleEntry entry
		cdef Py_ssize_t n = PyList_GET_SIZE(self.heap)
		if n == 0:
			raise IndexError("peek at empty heap")
		entry = <DoubleEntry>(self.heap[0])
		while entry.count == 0:
			if n == 1:
				raise IndexError("peek at empty heap")
			# replace first element with last element
			self.heap[0] = self.heap.pop()
			# and restore heap invariant
			siftup(self.heap, 0, entry)
			n -= 1
			entry = <DoubleEntry>(self.heap[0])
		return entry

	cdef inline void setitem(self, key, double value):
		"""Like agenda[key] = value, but bypass Python API."""
		cdef DoubleEntry oldentry
		cdef DoubleEntry entry = new_DoubleEntry(key, value, self.counter)
		if key in self.mapping:
			oldentry = <DoubleEntry>self.mapping[key]
			entry.count = oldentry.count
			oldentry.count = INVALID
		else:
			self.length += 1
			self.counter += 1
		self.mapping[key] = entry
		self.heap.append(entry)
		siftdown(self.heap, 0, PyList_GET_SIZE(self.heap) - 1, entry)

	cdef update_entries(self, list entries):
		"""Like ``update()``, but expects a list of DoubleEntry objects."""
		cdef DoubleEntry entry = None
		for entry in entries:
			entry.count = self.counter
			if entry.key in self.mapping:
				oldentry = <DoubleEntry>self.mapping[entry.key]
				if cmpfun(entry, oldentry):
					entry.count = oldentry.count
					oldentry.count = INVALID
					self.mapping[entry.key] = entry
					self.heap.append(entry)
			else:
				self.mapping[entry.key] = entry
				self.heap.append(entry)
				self.counter += 1
		self.length = len(self.mapping)
		heapify(self.heap, entry)

	# Override these methods to ensure that only DoubleEntry objects enter heap
	def __setitem__(self, key, value):
		cdef DoubleEntry oldentry
		cdef DoubleEntry entry = new_DoubleEntry(key, value, self.counter)
		if key in self.mapping:
			oldentry = <DoubleEntry>self.mapping[key]
			entry.count = oldentry.count
			oldentry.count = INVALID
		else:
			self.length += 1
			self.counter += 1
		self.mapping[key] = entry
		heappush(self.heap, entry)

	def __delitem__(self, key):
		"""Remove key from heap."""
		(<DoubleEntry>self.mapping[key]).count = INVALID
		self.length -= 1
		del self.mapping[key]

	def popitem(self):
		cdef DoubleEntry entry = self.popentry()
		return entry.key, entry.value

	def pop(self, key):
		""":returns: value for agenda[key] and remove it."""
		cdef DoubleEntry entry
		if key is None:
			return self.popentry().value
		entry = self.mapping.pop(key)
		entry.count = INVALID
		self.length -= 1
		return entry.value

	def peekitem(self):
		"""Get the current best (key, value) pair, while keeping it on the
		agenda."""
		cdef DoubleEntry entry
		cdef Py_ssize_t n = PyList_GET_SIZE(self.heap)
		if n == 0:
			raise IndexError("peek at empty heap")
		entry = self.heap[0]
		while entry.count == 0:
			if n == 1:
				raise IndexError("peek at empty heap")
			# replace first element with last element
			self.heap[0] = self.heap.pop()
			# and restore heap invariant
			siftup(self.heap, 0, entry)
			n -= 1
			entry = self.heap[0]
		return entry.key, entry.value


# A quicksort nsmallest implementation.
cdef list nsmallest(int n, list entries):
	"""Return an _unsorted_ list of the n smallest DoubleEntry objects.

	``entries`` is modified in-place."""
	if len(entries) > 1:
		quickfindfirstk(entries, 0, len(entries) - 1, n)
	return entries[:n]


cdef inline void quickfindfirstk(list entries, int left, int right, int k):
	"""Quicksort k-best selection."""
	# select pivot index between left and right
	# middle between left & right
	cdef int pivot = left + (right - left) // 2
	cdef int pivotnewindex = partition(entries, left, right, pivot)
	if pivotnewindex > k:
		if pivotnewindex - 1 > left:
			# new condition
			quickfindfirstk(entries, left, pivotnewindex - 1, k)
	elif pivotnewindex < k:
		if right > pivotnewindex + 1:
			quickfindfirstk(entries, pivotnewindex + 1, right, k)


cdef inline int partition(list entries, int left, int right, int pivot):
	cdef double pivotvalue = (<DoubleEntry>entries[pivot]).value
	# Move pivot to end
	entries[pivot], entries[right] = entries[right], entries[pivot]
	cdef int i, storeindex = left
	for i in range(left, right):
		if (<DoubleEntry>entries[i]).value < pivotvalue:
			entries[i], entries[storeindex] = entries[storeindex], entries[i]
			storeindex += 1
	# Move pivot to its final place
	entries[storeindex], entries[right] = entries[right], entries[storeindex]
	return storeindex


# heap operations (adapted from Python heapq module)
# dummy variables are only used to select the right fused type.
cdef inline Entry_fused heappop(list heap, Entry_fused dummy):
	cdef Py_ssize_t n = PyList_GET_SIZE(heap)
	cdef Entry_fused entry
	if n == 0:
		raise IndexError('pop from empty heap')
	elif n == 1:
		entry = <Entry_fused>heap.pop()
	else:
		# replace first element with last element and restore heap invariant
		entry = <Entry_fused>(PyList_GET_ITEM(heap, 0))
		heap[0] = heap.pop()
		siftup(heap, 0, dummy)
	return entry


cdef inline void heappush(list heap, Entry_fused entry):
	# place at the end and swap with parents until heap invariant holds
	heap.append(entry)
	siftdown(heap, 0, PyList_GET_SIZE(heap) - 1, entry)


cdef inline Entry_fused heapreplace(list heap, Entry_fused entry):
	"""Pop and return the current smallest value, and add the new item.

	NB: returned item may be larger than new item."""
	cdef Py_ssize_t n = PyList_GET_SIZE(heap)
	cdef Entry_fused oldentry
	if n == 0:
		raise IndexError("pop from empty heap")
	else:
		oldentry = <Entry_fused>(PyList_GET_ITEM(heap, 0))
		heap[0] = entry
		siftup(heap, 0, entry)
	return oldentry


cdef inline void heapify(list heap, Entry_fused dummy):
	"""Transform list into a heap, in-place, in O(len(heap)) time."""
	cdef int i
	if PyList_GET_SIZE(heap) > 1:
		for i in range((PyList_GET_SIZE(heap) - 2) // HEAP_ARITY, -1, -1):
			siftup(heap, i, dummy)


# shifts only apply for binary tree
cdef inline int _parent(int i):
	return (i - 1) // HEAP_ARITY
	# return (i - 1) >> 1


cdef inline int _left(int i):
	return i * HEAP_ARITY + 1
	# return (i << 1) + 1


cdef inline int _right(int i):
	"""For documentation purposes; not used."""
	return i * HEAP_ARITY + 2
	# return (i + 1) << 1


def getparent(i):
	"""Python version of Cython-only _parent() function."""
	return (i - 1) // HEAP_ARITY


# NB: the naming of siftdown / siftup follows the Python heapq module;
# gheap reverses this terminology.
cdef inline void siftdown(list heap, int startpos, int pos, Entry_fused dummy):
	"""`heap` is a heap at all indices >= startpos, except possibly for pos.
	`pos` is the index of a leaf with a possibly out-of-order value.
	Restore the heap invariant."""
	cdef int parentpos
	cdef Entry_fused parent, newitem = <Entry_fused>PyList_GET_ITEM(heap, pos)
	while pos > startpos:
		parentpos = _parent(pos)
		parent = <Entry_fused>PyList_GET_ITEM(heap, parentpos)
		if cmpfun(parent, newitem):
			break
		heap[pos] = parent
		pos = parentpos
	heap[pos] = newitem


cdef inline void siftup(list heap, int pos, Entry_fused dummy):
	"""The child indices of heap index pos are already heaps, and we want to
	make a heap at index pos too.  We do this by bubbling the smaller child of
	pos up (and so on with that child's children, etc) until hitting a leaf,
	then using `siftdown` to move the oddball originally at index pos into
	place."""
	cdef int startpos = pos, childpos = _left(pos), rightpos
	cdef int endpos = PyList_GET_SIZE(heap)
	cdef Entry_fused newitem = <Entry_fused>PyList_GET_ITEM(heap, pos)
	while childpos < endpos:
		for rightpos in range(childpos + 1, childpos + HEAP_ARITY):
			if (rightpos < endpos and
				cmpfun(<Entry_fused>(PyList_GET_ITEM(heap, rightpos)),
					<Entry_fused>(PyList_GET_ITEM(heap, childpos)))):
				childpos = rightpos
		heap[pos] = <Entry_fused>PyList_GET_ITEM(heap, childpos)
		pos = childpos
		childpos = _left(pos)
	heap[pos] = newitem
	siftdown(heap, startpos, pos, dummy)


def merge(*iterables, key=None):
	"""Generator that performs an n-way merge of sorted iterables.

	>>> list(merge([0, 1, 2], [0, 1, 2, 3]))
	[0, 0, 1, 1, 2, 2, 3]

	NB: while a sort key may be specified, the individual iterables must
	already be sorted with this key."""
	cdef list heap = []
	cdef uint64_t cnt
	cdef Entry entry = None, dummy = None
	if key is None:
		def key(x):
			return x

	for cnt, it in enumerate(iterables, 1):
		items = iter(it)
		try:
			item = next(items)
		except StopIteration:
			pass
		else:
			heap.append(new_Entry((item, items), key(item), cnt))
	heapify(heap, entry)

	while len(heap) > 1:
		try:
			while True:
				entry = heap[0]
				item, iterable = entry.key
				yield item
				item = next(iterable)
				entry.key = (item, iterable)
				entry.value = key(item)
				dummy = heapreplace(heap, entry)
		except StopIteration:
			dummy = heappop(heap, entry)

	if heap:  # only a single iterator remains, skip heap
		entry = heappop(heap, entry)
		item, iterable = entry.key
		yield item
		for item in iterable:
			yield item
