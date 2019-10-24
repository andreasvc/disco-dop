"""Misc code to avoid cyclic imports."""
import os
import re
import sys
import gzip
import codecs
import traceback
import subprocess
from contextlib import contextmanager
from heapq import heapify, heappush, heappop, heapreplace
from functools import wraps
from collections.abc import Set, Iterable


def which(program, exception=True):
	"""Return first match for program in search path.

	:param exception: By default, ValueError is raised when program not found.
		Pass False to return None in this case.
	"""
	for path in os.environ.get('PATH', os.defpath).split(":"):
		if path and os.path.exists(os.path.join(path, program)):
			return os.path.join(path, program)
	if exception:
		raise ValueError('%r not found in path; please install it.' % program)


def workerfunc(func):
	"""Wrap a multiprocessing worker function to produce a full traceback."""
	@wraps(func)
	def wrapper(*args, **kwds):
		"""Apply decorated function."""
		# NB: only concurrent.futures on Python 3.3+ will exit gracefully.
		try:
			return func(*args, **kwds)
		except Exception:  # pylint: disable=W0703
			# Put traceback as string into an exception and raise that
			raise Exception('in worker process\n%s' %
					''.join(traceback.format_exception(*sys.exc_info())))
	return wrapper


@contextmanager
def genericdecompressor(cmd, filename, encoding='utf8'):
	"""Run command line decompressor on file and return file object.

	:param encoding: if None, mode is binary; otherwise, text."""
	with subprocess.Popen(
			[which(cmd), '-d', '-c', '-q', filename],
			stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
		# FIXME: should use select to avoid deadlocks due to OS pipe buffers
		# filling up and blocking the child process.
		yield proc.stdout if encoding is None else codecs.getreader(
				encoding)(proc.stdout)
		retcode = proc.wait()
		if retcode:  # FIXME: retcode 2 means warning. allow warnings?
			raise ValueError('non-zero exit code %s from compressor %s:\n%r'
					% (retcode, cmd, proc.stderr.read()))


@contextmanager
def genericcompressor(cmd, filename, encoding='utf8', compresslevel=8):
	"""Run command line compressor on file and return file object.

	:param encoding: if None, mode is binary; otherwise, text."""
	with open(filename, 'wb') as out, subprocess.Popen(
			[which(cmd), '-c', '-q', '-f', '-%s' % compresslevel],
			stdin=subprocess.PIPE, stdout=out, stderr=subprocess.PIPE) as proc:
		yield proc.stdin if encoding is None else codecs.getwriter(
				encoding)(proc.stdin)
		proc.stdin.close()
		retcode = proc.wait()
		if retcode:
			raise ValueError('non-zero exit code %s from compressor %s:\n%r'
					% (retcode, cmd, proc.stderr.read()))


def openread(filename, encoding='utf8'):
	"""Open stdin/file for reading; decompress gz/lz4/zst files on-the-fly.

	:param encoding: if None, mode is binary; otherwise, text."""
	mode = 'rb' if encoding is None else 'rt'
	if filename == '-':  # TODO: decompress stdin on-the-fly
		return open(sys.stdin.fileno(), mode=mode, encoding=encoding)
	if not isinstance(filename, int):
		if filename.endswith('.gz'):
			return gzip.open(filename, mode=mode, encoding=encoding)
		elif filename.endswith('.zst'):
			return genericdecompressor('zstd', filename, encoding)
		elif filename.endswith('.lz4'):
			return genericdecompressor('lz4', filename, encoding)
	return open(filename, mode=mode, encoding=encoding)


def readbytes(filename):
	"""Read bytes from stdin/file; decompress gz/lz4/zst files on-the-fly."""
	with openread(filename, encoding=None) as inp:
		return inp.read()


def slice_bounds(seq, slice_obj, allow_step=False):
	"""Calculate the effective (start, stop) bounds of a slice.

	Takes into account ``None`` indices and negative indices.

	:returns: tuple ``(start, stop, 1)``, s.t. ``0 <= start <= stop <= len(seq)``
	:raises ValueError: if slice_obj.step is not None.
	:param allow_step: If true, then the slice object may have a non-None step.
		If it does, then return a tuple (start, stop, step)."""
	start, stop = (slice_obj.start, slice_obj.stop)
	if allow_step:
		slice_obj.step = 1 if slice_obj.step is None else slice_obj.step
		# Use a recursive call without allow_step to find the slice
		# bounds. If step is negative, then the roles of start and
		# stop (in terms of default values, etc), are swapped.
		if slice_obj.step < 0:
			start, stop, _ = slice_bounds(seq, slice(stop, start))
		else:
			start, stop, _ = slice_bounds(seq, slice(start, stop))
		return start, stop, slice_obj.step
	elif slice_obj.step not in (None, 1):
		raise ValueError('slices with steps are not supported by %s' %
				seq.__class__.__name__)
	start = 0 if start is None else start
	stop = len(seq) if stop is None else stop
	start = max(0, len(seq) + start) if start < 0 else start
	stop = max(0, len(seq) + stop) if stop < 0 else stop
	if stop > 0:  # Make sure stop doesn't go past the end of the list.
		try:  # Avoid calculating len(seq), may be expensive for lazy sequences
			seq[stop - 1]
		except IndexError:
			stop = len(seq)
	start = min(start, stop)
	return start, stop, 1


class OrderedSet(Set):
	"""A frozen, ordered set which maintains a regular list/tuple and set.

	The set is indexable. Equality is defined _without_ regard for order."""

	def __init__(self, iterable=None):
		if iterable:
			self.seq = tuple(iterable)
			self.theset = frozenset(self.seq)
		else:
			self.seq = ()
			self.theset = frozenset()

	def __hash__(self):
		return hash(self.theset)

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
			return '%s()' % self.__class__.__name__
		return '%s(%r)' % (self.__class__.__name__, self.seq)

	def __eq__(self, other):
		"""equality is defined _without_ regard for order."""
		return self.theset == set(other)

	def __and__(self, other):
		"""maintain the order of the left operand."""
		if not isinstance(other, Iterable):
			return NotImplemented
		return self._from_iterable(value for value in self if value in other)


INVALID = 0


class Entry(object):
	"""A PyAgenda entry."""
	def __init__(self, key, value, count):
		self.key = key
		self.value = value
		self.count = count

	def __lt__(self, b):
		return (self.value < b.value
				or (self.value == b.value and self.count < b.count))

	def __repr__(self):
		return '%s(%r, %r, %r)' % (
				self.__class__.__name__, self.key, self.value, self.count)


class PyAgenda:
	"""Priority Queue implemented with array-based heap.

	Implements decrease-key and remove operations by marking entries as
	invalid. Provides dictionary-like interface.

	Can be initialized with an iterable; equivalent values are preserved in
	insertion order and the best priorities are retained on duplicate keys."""
	def __init__(self, iterable=None):
		entry = None
		self.counter = 1
		self.length = 0
		self.heap = []
		self.mapping = {}
		if iterable:
			for k, v in iterable:
				entry = Entry(k, v, self.counter)
				if k in self.mapping:
					oldentry = self.mapping[k]
					if entry < oldentry:
						entry.count = oldentry.count
						oldentry.count = INVALID
						self.mapping[k] = entry
						self.heap.append(entry)
				else:
					self.mapping[k] = entry
					self.heap.append(entry)
					self.counter += 1
			self.length = len(self.mapping)
			heapify(self.heap)

	def peekitem(self):
		"""Get the current best (key, value) pair; keep it on the agenda."""
		n = len(self.heap)
		if n == 0:
			raise IndexError("peek at empty heap")
		entry = self.heap[0]
		while entry.count == 0:
			if n == 1:
				raise IndexError("peek at empty heap")
			entry = heappop(self.heap)
			n -= 1
		return entry.key, entry.value

	# standard dict() methods
	def pop(self, key):
		""":returns: value for agenda[key] and remove it."""
		if key is None:
			return self.popitem()[1]
		entry = self.mapping.pop(key)
		entry.count = INVALID
		self.length -= 1
		return entry.value

	def popitem(self):
		""":returns: best scoring (key, value) pair; removed from agenda."""
		entry = None
		entry = heappop(self.heap)
		while not entry.count:
			entry = heappop(self.heap)
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
		entry = Entry(key, value, self.counter)
		if key in self.mapping:
			oldentry = self.mapping[key]
			entry.count = oldentry.count
			oldentry.count = INVALID
		else:
			self.length += 1
			self.counter += 1
		self.mapping[key] = entry
		heappush(self.heap, entry)

	def __delitem__(self, key):
		"""Remove key from heap."""
		self.mapping[key].count = INVALID
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


def merge(*iterables, key=None):
	"""Generator that performs an n-way merge of sorted iterables.

	>>> list(merge([0, 1, 2], [0, 1, 2, 3]))
	[0, 0, 1, 1, 2, 2, 3]

	Similar to ``heapq.merge``, but ``key`` can be specified.

	NB: while a sort key may be specified, the individual iterables must
	already be sorted with this key."""
	def defaultkey(x):
		"""Default key() function (identity)."""
		return x

	heap = []
	entry = None
	if key is None:
		key = defaultkey

	for cnt, it in enumerate(iterables, 1):
		items = iter(it)
		try:
			item = next(items)
		except StopIteration:
			pass
		else:
			heap.append(Entry((item, items), key(item), cnt))
	heapify(heap)

	while len(heap) > 1:
		try:
			while True:
				entry = heap[0]
				item, iterable = entry.key
				yield item
				item = next(iterable)
				entry.key = (item, iterable)
				entry.value = key(item)
				heapreplace(heap, entry)
		except StopIteration:
			heappop(heap)

	if heap:  # only a single iterator remains, skip heap
		entry = heappop(heap)
		item, iterable = entry.key
		yield item
		yield from iterable


FRENCHCONTRACTIONS = 'aujourd|jusqu|lorsqu|presqu|puisqu|qu|quelqu|quoiqu'
# List of contractions adapted from Robert MacIntyre's tokenizer.
CONTRACTIONS = [
		r"(.)('ll|'re|'ve|n't|'s|'m|'d)",
		r"\b(can)(not)",
		r"\b(D)('ye)",
		r"\b(Gim)(me)",
		r"\b(Gon)(na)",
		r"\b(Got)(ta)",
		r"\b(Lem)(me)",
		r"\b(Mor)('n)",
		r"\b(T)(is)",
		r"\b(T)(was)",
		r"\b(Wan)(na)",
		r"\b((?:[cdjlmnst]|%s)')(\w+)" % FRENCHCONTRACTIONS,
		]
CONTRACTIONSRE = re.compile(
		r"(?i)(?:%s)\b" % "|".join(CONTRACTIONS), flags=re.UNICODE)
CONTRACTIONSREPL = ''.join(  # r'\1\3\5... \2\4\6...',
		['\\%d' % n for n in range(1, 2 * len(CONTRACTIONS) + 1, 2)]
		+ [' ']
		+ ['\\%d' % n for n in range(2, 2 * len(CONTRACTIONS) + 1, 2)])


def tokenize(text):
	"""A basic tokenizer following English/French PTB/FTB conventions.

	Adapted from nltk.tokenize.TreebankTokenizer."""
	text = CONTRACTIONSRE.sub(CONTRACTIONSREPL, text)
	# Separate most punctuation
	text = re.sub(r"([^\w\.\'\-\/,&])", r' \1 ', text, flags=re.UNICODE)
	# Separate commas if they're followed by space; e.g., don't separate 2,500
	# Separate single quotes if they're followed by a space.
	text = re.sub(r"(\S\S+)([,'](?:\s|$))", r'\1 \2', text, flags=re.UNICODE)
	# hack to revert "qu '" etc back to "qu'"
	text = re.sub(r"\b(%s) ' " % FRENCHCONTRACTIONS,
			r"\1' ", text, flags=re.IGNORECASE)
	text = re.sub(r"^'", r"' ", text)
	# Separate periods near end of string.
	text = re.sub(r'\.(\W*$)', r' . \1', text)
	return text.split()


def run(*popenargs, **kwargs):
	"""Run command with arguments and return (returncode, stdout, stderr).

	All arguments are the same as for the Popen constructor."""
	with subprocess.Popen(*popenargs, **kwargs,
			stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
		try:
			stdout, stderr = proc.communicate()
		except Exception:
			proc.kill()
			proc.wait()
			raise
		retcode = proc.poll()
	return retcode, stdout, stderr


ANSICOLOR = {
		'black': 30,
		'red': 31,
		'green': 32,
		'yellow': 33,
		'blue': 34,
		'magenta': 35,
		'cyan': 36,
		'white': 37,
}

__all__ = ['which', 'workerfunc', 'openread', 'slice_bounds',
		'OrderedSet', 'ANSICOLOR']
