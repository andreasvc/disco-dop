"""Misc code to avoid cyclic imports."""
import io
import os
import sys
import gzip
import codecs
import traceback
from functools import wraps
from collections import Set, Iterable


def ishead(tree):
	"""Test whether this node is the head of the parent constituent."""
	return getattr(tree, 'head', False)


def which(program):
	"""Return first match for program in search path."""
	for path in os.environ.get('PATH', os.defpath).split(":"):
		if path and os.path.exists(os.path.join(path, program)):
			return os.path.join(path, program)
	raise ValueError('%r not found in path; please install it.' % program)


def workerfunc(func):
	"""Wrap a multiprocessing worker function to produce a full traceback."""
	@wraps(func)
	def wrapper(*args, **kwds):
		"""Apply decorated function."""
		try:
			import faulthandler
			faulthandler.enable()  # Dump information on segfault.
		except (ImportError, io.UnsupportedOperation):
			pass
		# NB: only concurrent.futures on Python 3.3+ will exit gracefully.
		try:
			return func(*args, **kwds)
		except Exception:  # pylint: disable=W0703
			# Put traceback as string into an exception and raise that
			raise Exception('in worker process\n%s' %
					''.join(traceback.format_exception(*sys.exc_info())))
	return wrapper


def openread(filename, encoding='utf8'):
	"""Open stdin/text file for reading; decompress .gz files on-the-fly."""
	if filename == '-':
		return io.open(sys.stdin.fileno(), encoding=encoding)
	if not isinstance(filename, int) and filename.endswith('.gz'):
		return codecs.getreader(encoding)(gzip.open(filename))
	else:
		return io.open(filename, encoding=encoding)


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

__all__ = ['ishead', 'which', 'workerfunc', 'openread', 'slice_bounds',
		'OrderedSet', 'ANSICOLOR']
