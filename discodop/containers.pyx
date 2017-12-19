"""Data types for chart items, edges, &c."""
from __future__ import print_function
import io
import os
import re
import mmap
import logging
import numpy as np
from array import array
from math import isinf, fsum
from roaringbitmap import RoaringBitmap, MultiRoaringBitmap
from .tree import escape, unescape
from .util import readbytes
from .tree import Tree
cimport cython
from cython.operator cimport dereference
from libc.string cimport strchr
from libc.stdio cimport FILE, fopen, fread, fclose
include "constants.pxi"

cdef array chararray = array('b')
cdef array dblarray = array('d')
cdef int maxbitveclen = SLOTS * sizeof(uint64_t) * 8

include "_grammar.pxi"


cdef SmallChartItem CFGtoSmallChartItem(Label label, Idx start, Idx end):
	cdef SmallChartItem result = SmallChartItem(
			label, (1UL << end) - (1UL << start))
	return result


cdef FatChartItem CFGtoFatChartItem(Label label, Idx start, Idx end):
	cdef FatChartItem fci = FatChartItem(label)
	cdef short n
	if BITSLOT(start) == BITSLOT(end):
		fci.vec[BITSLOT(start)] = (1UL << end) - (1UL << start)
	else:
		fci.vec[BITSLOT(start)] = ~0UL << (start % BITSIZE)
		for n in range(BITSLOT(start) + 1, BITSLOT(end)):
			fci.vec[n] = ~0UL
		fci.vec[BITSLOT(end)] = BITMASK(end) - 1
	return fci


cdef class Chart:
	"""Base class for charts. Provides methods available on all charts.

	The subclass hierarchy for charts has three levels:

		(0) base class, methods for chart traversal.
		(1) formalism, methods specific to CFG vs. LCFRS parsers.
		(2) data structures optimized for short/long sentences, small/large
			grammars.

	Level 1/2 defines a type for labeled spans referred to as ``item``."""
	def root(self):
		"""Return item with root label spanning the whole sentence."""
		raise NotImplementedError

	cdef ItemNo _left(self, ItemNo itemidx, Edge edge):
		"""Return the left item that edge points to."""
		raise NotImplementedError

	cdef ItemNo _right(self, ItemNo itemidx, Edge edge):
		"""Return the right item that edge points to."""
		raise NotImplementedError

	cdef Label label(self, ItemNo itemidx):
		raise NotImplementedError

	cdef Prob subtreeprob(self, ItemNo itemidx):
		"""Return probability of subtree headed by item."""
		raise NotImplementedError

	cdef ItemNo left(self, ItemNo v, RankedEdge rankededge):
		"""Given a ranked edge, return the left item it points to."""
		return self._left(v, rankededge.edge)

	cdef ItemNo right(self, ItemNo v, RankedEdge rankededge):
		"""Given a ranked edge, return the right item it points to."""
		return self._right(v, rankededge.edge)

	cdef ItemNo getitemidx(self, uint64_t n):
		"""Get itemidx of n'th item."""
		return n

	cdef SmallChartItem asSmallChartItem(self, ItemNo itemidx):
		"""Convert/copy item to SmallChartItem instance."""
		raise NotImplementedError

	cdef FatChartItem asFatChartItem(self, ItemNo itemidx):
		"""Convert/copy item to FatChartItem instance."""
		raise NotImplementedError

	cdef size_t asCFGspan(self, ItemNo itemidx):
		"""Convert item for chart to compact span."""
		raise NotImplementedError

	def indices(self, item):
		"""Return a list of indices dominated by ``item``."""
		raise NotImplementedError

	cdef int lexidx(self, Edge edge) except -1:
		"""Return sentence index of the terminal child given a lexical edge."""
		cdef short result = edge.pos.mid - 1
		assert 0 <= result < self.lensent, (result, self.lensent)
		return result

	cdef int lexruleno(self, ItemNo itemidx, Edge edge) except -1:
		"""Return lexical rule number given a lexical edge."""
		cdef Label label = self.label(itemidx)
		cdef string word = self.sent[self.lexidx(edge)].encode('utf8')
		it = self.grammar.lexicalbylhs.find(label)
		if it != self.grammar.lexicalbylhs.end():
			it2 = dereference(it).second.find(word)
			if it2 != dereference(it).second.end():
				return dereference(it2).second
		raise ValueError

	cdef Prob lexprob(self, ItemNo itemidx, Edge edge) except -1:
		"""Return lexical probability given a lexical edge."""
		cdef int ruleno = self.lexruleno(itemidx, edge)
		if ruleno >= 0:
			prob = self.grammar.lexical[ruleno].prob
			return exp(-prob) if self.logprob else prob
		return 0 if self.logprob else 1

	def numitems(self):
		"""Number of items in chart; NB: this includes 1 sentinel item."""
		return self.parseforest.size()

	def itemid(self, str label, indices, Whitelist whitelist=None):
		"""Get integer ID for labeled span in the chart (0 if non-existent)."""
		raise NotImplementedError

	def __bool__(self):
		"""Return true when the root item is in the chart.

		i.e., test whether sentence has been parsed successfully."""
		return self.root() in self

	def __contains__(self, item):
		"""Return true when item is in the chart."""
		return (item is not None
				and item < self.parseforest.size()
				and self.parseforest[item].size() != 0)

	def filter(self):
		"""Drop edges not part of a derivation headed by root of chart."""
		cdef set itemstokeep = set()
		cdef ItemNo item
		if self.parseforest.size() == 0:
			return
		_filtersubtree(self, self.root(), itemstokeep)
		for item in {self.getitemidx(n) for n in range(1, self.numitems())
				} - itemstokeep:
			self.parseforest[item].clear()

	cdef edgestr(self, ItemNo itemidx, Edge edge):
		"""Return string representation of item and edge belonging to it."""
		if edge.rule is NULL:
			return "'%s' %g" % (
					self.sent[self.lexidx(edge)],
					self.lexprob(itemidx, edge))
		else:
			return '%s %s %g' % (
					self.itemstr(self._left(itemidx, edge)),
					self.itemstr(self._right(itemidx, edge))
						if edge.rule.rhs2 else '',
					exp(-edge.rule.prob)
						if self.grammar.logprob else edge.rule.prob)

	def __str__(self):
		"""Pretty-print chart and *k*-best derivations."""
		cdef RankedEdge rankededge
		cdef Edge edge
		cdef pair[RankedEdge, Prob] entry
		cdef uint64_t n, m
		cdef ItemNo item
		result = []
		for n in range(1, self.numitems()):
			item = self.getitemidx(n)
			result.append(' '.join((
					self.itemstr(item).ljust(20),
					('vitprob=%g' % (
						exp(-self.subtreeprob(item)) if self.logprob
						else self.subtreeprob(item))).ljust(17),
					((' ins=%g' % self.inside[item]).ljust(14)
						if self.inside.size() else ''),
					((' out=%g' % self.outside[item]).ljust(14)
						if self.outside.size() else ''))))
			for edge in self.parseforest[item]:
				result.append('\t=> %s' % self.edgestr(item, edge))
		if self.rankededges.size():
			result.append('\nranked edges:')
			for n in range(1, self.numitems()):
				item = self.getitemidx(n)
				if self.rankededges[item].size() == 0:
					continue
				result.append(self.itemstr(item))
				m = 0
				for entry in self.rankededges[item]:
					rankededge = entry.first
					result.append('\t%d: %10g, %s %d %d' % (
							m, exp(-entry.second) if self.logprob
								else entry.second,
							self.edgestr(item, rankededge.edge),
							rankededge.left, rankededge.right))
					m += 1
		return '\n'.join(result)

	def stats(self):
		"""Return a short string with counts of items, edges."""
		return 'items %d, edges %d' % (
				self.numitems() - 1,
				sum([self.parseforest[self.getitemidx(n)].size()
					for n in range(1, self.numitems())]))
		# more stats:
		# labels: len({self.label(item) for item in range(1, self.numitems())}),
		# spans: ...


cdef void _filtersubtree(Chart chart, item, set items):
	"""Recursively collect items that lead to a complete derivation."""
	cdef Edge edge
	items.add(item)
	for edge in chart.parseforest[item]:
		if edge.rule is NULL:
			continue
		leftitem = chart._left(item, edge)
		if leftitem not in items:
			_filtersubtree(chart, leftitem, items)
		if edge.rule.rhs2 == 0:
			continue
		rightitem = chart._right(item, edge)
		if rightitem not in items:
			_filtersubtree(chart, rightitem, items)


@cython.final
cdef class StringList(object):
	"""Proxy class to expose vector<string> with read-only list interface.

	Decodes utf8 to unicode."""
	def __len__(self):
		return self.ob.size()

	def __getitem__(self, size_t i):
		cdef string result
		if i >= self.ob.size():
			raise IndexError('index %d out of bounds (len=%d)' % (
					i, self.ob.size()))
		result = self.ob[i]
		return result.decode('utf8')


@cython.final
cdef class StringIntDict(object):
	"""Proxy class to expose sparse_hash_map with read-only dict interface.

	Decodes utf8 to unicode."""
	def __len__(self):
		return self.ob.size()

	def __getitem__(self, str key):
		cdef string cppkey = key.encode('utf8')
		it = self.ob.find(cppkey)
		if it == self.ob.end():
			raise KeyError(key)
		return dereference(it).second

	def get(self, str key, default=None):
		cdef string cppkey = key.encode('utf8')
		it = self.ob.find(cppkey)
		if it == self.ob.end():
			return default
		return dereference(it).second

	def __contains__(self, str key):
		cdef string cppkey = key.encode('utf8')
		return self.ob.find(cppkey) != self.ob.end()

	def __iter__(self):
		for it in self.ob:
			yield it.first.decode('utf8')


@cython.final
cdef class Ctrees:
	"""An indexed, binarized treebank stored as array.

	Indexing depends on an external Vocabulary object that maps
	productions and labels to unique integers across different sets of trees.
	First call the alloc() method with (estimated) number of nodes & trees.
	Then add trees one by one using the addnodes() method."""
	def __cinit__(self):
		self.trees = self.nodes = NULL

	def __init__(self):
		self.len = self.max = 0
		self.numnodes = self.maxnodes = self.nodesleft = self.numwords = 0
		self._state = None

	cpdef alloc(self, int numtrees, long numnodes):
		"""Initialize an array of trees of nodes structs."""
		self.max = numtrees
		self.trees = <NodeArray *>malloc(numtrees * sizeof(NodeArray))
		self.nodes = <Node *>malloc(numnodes * sizeof(Node))
		if self.trees is NULL or self.nodes is NULL:
			raise MemoryError('allocation error')
		self.nodesleft = numnodes

	cdef realloc(self, int numtrees, int extranodes):
		"""Increase size of array (handy with incremental binarization)."""
		# based on Python's listobject.c list_resize()
		cdef numnodes
		if numtrees > self.max:
			# overallocate to get linear-time amortized behavior
			numtrees += (numtrees >> 3) + (3 if numtrees < 9 else 6)
			self.trees = <NodeArray *>realloc(self.trees,
					numtrees * sizeof(NodeArray))
			if self.trees is NULL:
				raise MemoryError('allocation error')
			self.max = numtrees
		if extranodes > self.nodesleft:
			numnodes = self.numnodes + extranodes
			# estimate how many new nodes will be needed
			# self.nodesleft += (self.max - self.len) * (self.numnodes / self.len)
			# overallocate to get linear-time amortized behavior
			numnodes += (numnodes >> 3) + (3 if numnodes < 9 else 6)
			self.nodes = <Node *>realloc(self.nodes, numnodes * sizeof(Node))
			if self.nodes is NULL:
				raise MemoryError('allocation error')
			self.nodesleft = numnodes - self.numnodes

	cdef addnodes(self, Node *source, int cnt, int root):
		"""Incrementally add tree to the node array.

		Copies a tree that has already been converted to an array of nodes."""
		cdef dict prodsintree, sortidx
		cdef int n, m, lensent = 0
		cdef Node *dest
		if not self.max:
			raise ValueError('alloc() has not been called (max=0).')
		if self.len >= self.max or self.nodesleft < cnt:
			self.realloc(self.len + 1, cnt)
		prodsintree = {n: source[n].prod for n in range(cnt)}
		sortidx = {m: n for n, m in enumerate(
				sorted(range(cnt), key=prodsintree.get))}
		# copy nodes to allocated array, while translating indices
		dest = &self.nodes[self.numnodes]
		for n, m in sortidx.items():
			dest[m] = source[n]
			if dest[m].left >= 0:
				dest[m].left = sortidx[source[n].left]
			else:
				lensent += 1
			if dest[m].right >= 0:
				dest[m].right = sortidx[source[n].right]
		self.trees[self.len].offset = self.numnodes
		self.trees[self.len].root = sortidx[root]
		self.trees[self.len].len = cnt
		self.len += 1
		self.nodesleft -= cnt
		self.numnodes += cnt
		if cnt > self.maxnodes:
			self.maxnodes = cnt
		self.numwords += lensent

	def indextrees(self, Vocabulary vocab):
		"""Create index from productions to trees containing that production.

		Productions are represented as integer IDs, trees are given as sets of
		integer indices."""
		cdef:
			list prodindex = [None] * len(vocab.prods)
			Node *nodes
			int n, m
		for n in range(self.len):
			nodes = &self.nodes[self.trees[n].offset]
			for m in range(self.trees[n].len):
				if nodes[m].prod >= 0:
					# Add to production index
					rb = prodindex[nodes[m].prod]
					if rb is None:
						rb = prodindex[nodes[m].prod] = RoaringBitmap()
					rb.add(n)
		self.prodindex = MultiRoaringBitmap(prodindex)

	def extract(self, int n, Vocabulary vocab, bint disc=True, int node=-1):
		"""Return given tree in discbracket format.

		:param node: if given, extract specific subtree instead of whole tree.
		"""
		result = []
		if n < 0 or n >= self.len:
			raise IndexError
		gettree(result, &(self.nodes[self.trees[n].offset]),
				vocab, self.trees[n].root if node == -1 else node, disc)
		return ''.join(result)

	def extractsent(self, int n, Vocabulary vocab):
		"""Return sentence as a list for given tree."""
		if n < 0 or n >= self.len:
			raise IndexError
		sent = {termidx(self.nodes[m].left): vocab.getword(self.nodes[m].prod)
				for m in range(self.trees[n].offset,
					self.trees[n].offset + self.trees[n].len)
				if self.nodes[m].left < 0}
		return [sent.get(m, None) for m in range(max(sent) + 1)]

	def printrepr(self, int n, Vocabulary vocab):
		"""Print repr of a tree for debugging purposes."""
		tree = self.extract(n, vocab, disc=True)
		print('tree no.:', n)
		print('complete tree:', tree)
		offset = self.trees[n].offset
		print('no. leftidx rightidx prodno        lhslabel     prod')
		for m in range(self.trees[n].len):
			print('%2d. %7d %8d %6d %15s     %s' % (
					m, self.nodes[offset + m].left,
					self.nodes[offset + m].right,
					self.nodes[offset + m].prod,
					vocab.getlabel(self.nodes[offset + m].prod),
					vocab.prodrepr(self.nodes[offset + m].prod)))
		print()
		# for a, m in sorted(vocab.prods.items(), key=lambda x: x[1]):
		# 	print("%d. '%s' '%s' '%s'" % (m, vocab.labels[m], vocab.words[m], a))
		# print()

	def __dealloc__(self):
		if isinstance(self._state, tuple):
			self._state[1].close()
			os.close(self._state[0])
			self._state = None
			return
		elif isinstance(self._state, dict):
			return
		if self.nodes is not NULL:
			free(self.nodes)
			self.nodes = NULL
		if self.trees is not NULL:
			free(self.trees)
			self.trees = NULL

	def __len__(self):
		return self.len

	def __reduce__(self):
		"""Helper function for pickling."""
		return (Ctrees, (),
				dict(nodes=<bytes>(<char *>self.nodes)[
						:self.numnodes * sizeof(self.nodes[0])],
					trees=<bytes>(<char *>self.trees)[
						:self.len * sizeof(self.trees[0])],
					len=self.len,
					numnodes=self.numnodes,
					numwords=self.numwords,
					maxnodes=self.maxnodes,
					prodindex=self.prodindex))

	def __setstate__(self, state):
		self.len = self.max = state['len']
		self.numnodes = state['numnodes']
		self.numwords = state['numwords']
		self.maxnodes = state['maxnodes']
		self.prodindex = state['prodindex']
		# self.alloc(self.len, self.numnodes)
		self.nodesleft = 0
		self.nodes = <Node *><char *><bytes>state['nodes']
		self.trees = <NodeArray *><char *><bytes>state['trees']
		self._state = state  # keep reference alive

	def tofile(self, filename):
		cdef array out
		cdef uint64_t *ptr
		cdef int offset
		cdef bytes tmp = self.prodindex.__getstate__()
		out = clone(chararray,
				4 * sizeof(uint64_t)
				+ len(tmp)
				+ self.len * sizeof(NodeArray)
				+ self.numnodes * sizeof(Node),
				False)
		ptr = <uint64_t *>out.data.as_chars
		ptr[0] = self.len
		ptr[1] = self.numnodes
		ptr[2] = self.numwords
		ptr[3] = self.maxnodes
		offset = 4 * sizeof(uint64_t)
		memcpy(&(out.data.as_chars[offset]), <char *>tmp, len(tmp))
		offset += len(tmp)
		memcpy(&(out.data.as_chars[offset]), <char *>self.trees,
				self.len * sizeof(NodeArray))
		offset += self.len * sizeof(NodeArray)
		memcpy(&(out.data.as_chars[offset]), <char *>self.nodes,
				self.numnodes * sizeof(Node))
		with open(filename, 'wb') as outfile:
			out.tofile(outfile)

	@classmethod
	def fromfile(cls, filename):
		cdef Ctrees ob = Ctrees.__new__(Ctrees)
		cdef size_t prodidxsize
		cdef Py_buffer buffer
		cdef Py_ssize_t size = 0
		cdef char *ptr = NULL
		cdef uint64_t *header = NULL
		cdef int result
		fileno = os.open(filename, os.O_RDONLY)
		buf = mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)
		ob._state = (fileno, buf)
		result = getbufptr(buf, &ptr, &size, &buffer)
		if result != 0:
			raise ValueError('could not get buffer from mmap.')
		header = <uint64_t *>ptr
		ob.len = ob.max = header[0]
		ob.numnodes, ob.numwords, ob.maxnodes = header[1], header[2], header[3]
		ob.nodesleft = 0
		ob.prodindex = MultiRoaringBitmap.frombuffer(buf, 4 * sizeof(uint64_t))
		prodidxsize = ob.prodindex.bufsize()
		ob.trees = <NodeArray *>&(ptr[4 * sizeof(uint64_t) + prodidxsize])
		ob.nodes = <Node *>&ob.trees[ob.len]
		PyBuffer_Release(&buffer)
		return ob


cdef inline getyf(tuple yf, uint32_t *args, uint32_t *lengths):
	cdef int m = 0
	args[0] = lengths[0] = 0
	for part in yf:
		for a in part:
			if a == 1:
				args[0] += 1 << m
			elif a != 0:
				raise ValueError('expected: 0 or 1; got: %r' % a)
			m += 1
		lengths[0] |= 1 << (m - 1)


cdef inline darray_init(DArray *self, uint8_t itemsize):
	self.len = 0
	self.capacity = 4
	self.itemsize = itemsize
	self.d.ptr = malloc(self.capacity * itemsize)
	if self.d.ptr is NULL:
		raise MemoryError


cdef inline darray_append_uint32(DArray *self, uint32_t elem):
	cdef void *tmp
	if self.len == self.capacity:
		self.capacity *= 2
		tmp = realloc(self.d.ptr, self.capacity * self.itemsize)
		if tmp is NULL:
			raise MemoryError
		self.d.ptr = tmp
	self.d.asint[self.len] = elem
	self.len += 1


cdef inline darray_extend(DArray *self, bytes data):
	cdef void *tmp
	cdef int n = len(data)
	if self.len + n >= self.capacity:
		self.capacity = 2 * (self.capacity + n)
		tmp = realloc(self.d.ptr, self.capacity * self.itemsize)
		if tmp is NULL:
			raise MemoryError
		self.d.ptr = tmp
	memcpy(&(self.d.aschar[self.len]), <char *>data, n)
	self.len += n


cdef class Vocabulary:
	"""A mapping of productions, labels, words to integers.

	- Vocabulary.getprod(): get prod no and add to index (mutating).
	- FixedVocabulary.getprod(): lookup prod no given labels/words.
		(no mutation, but requires makeindex())
	- .getlabel(): lookup label/word given prod no (no mutation, arrays only)
	"""
	def __init__(self):
		cdef bytes epsilon = b'Epsilon'
		self.prods = {}  # bytes objects => prodno
		self.labels = {epsilon.decode('utf8'): 0}  # str => labelno
		darray_init(&self.prodbuf, sizeof(char))
		darray_init(&self.labelbuf, sizeof(char))
		darray_init(&self.labelidx, sizeof(uint32_t))
		darray_append_uint32(&self.labelidx, 0)
		darray_append_uint32(&self.labelidx, len(epsilon))
		darray_extend(&self.labelbuf, epsilon)

	def __dealloc__(self):
		if self.prodbuf.capacity == 0:  # FixedVocabulary object
			return
		free(self.prodbuf.d.ptr)
		free(self.labelbuf.d.ptr)
		free(self.labelidx.d.ptr)
		self.prodbuf.d.ptr = self.labelbuf.d.ptr = self.labelidx.d.ptr = NULL

	cdef int getprod(self, tuple r, tuple yf) except -2:
		"""Lookup/assign production ID. Add IDs for labels/words."""
		cdef Rule rule
		cdef char *tmp = <char *>&rule
		rule.lhs = self._getlabelid(r[0])
		rule.rhs1 = self._getlabelid(r[1]) if len(r) > 1 else 0
		rule.rhs2 = self._getlabelid(r[2]) if len(r) > 2 else 0
		if rule.rhs1 == 0 and len(r) > 1:
			rule.args = self._getlabelid(yf[0])
			rule.lengths = 0
		else:
			getyf(yf, &rule.args, &rule.lengths)
		prod = <bytes>tmp[:sizeof(Rule)]
		return self._getprodid(prod)

	cdef int _getprodid(self, bytes prod) except -2:
		cdef uint32_t n
		if prod not in self.prods:
			n = len(self.prods)
			self.prods[prod] = n
			darray_extend(&self.prodbuf, prod)
			return n
		return self.prods[prod]

	cdef int _getlabelid(self, str label) except -1:
		if label not in self.labels:
			self.labels[label] = len(self.labels)
			darray_extend(&self.labelbuf, label.encode('utf8'))
			darray_append_uint32(&self.labelidx, self.labelbuf.len)
		return self.labels[label]

	cdef str idtolabel(self, uint32_t i):
		return self.labelbuf.d.aschar[self.labelidx.d.asint[i]:
				self.labelidx.d.asint[i + 1]].decode('utf8')

	cdef str getlabel(self, int prodno):
		cdef Rule *rule
		if prodno < 0:
			return '<UNKNOWN>'
		rule = <Rule *>(&self.prodbuf.d.aschar[
				prodno * sizeof(Rule)])
		return self.labelbuf.d.aschar[self.labelidx.d.asint[rule.lhs]:
				self.labelidx.d.asint[rule.lhs + 1]].decode('utf8')

	cdef str getword(self, int prodno):
		cdef Rule *rule
		if prodno < 0:
			return '<UNKNOWN>'
		rule = <Rule *>(&self.prodbuf.d.aschar[
				prodno * sizeof(Rule)])
		if rule.args == 0:
			return None
		return self.labelbuf.d.aschar[self.labelidx.d.asint[rule.args]:
				self.labelidx.d.asint[rule.args + 1]].decode('utf8')

	cdef bint islexical(self, int prodno):
		cdef Rule *rule
		if prodno < 0:
			return False
		rule = <Rule *>(&self.prodbuf.d.aschar[
				prodno * sizeof(Rule)])
		return rule.args != 0 and rule.lengths == 0

	def prodrepr(self, int prodno):
		cdef Rule *rule
		cdef int fanout, n, m = 0
		cdef str yf = ''
		if prodno < 0:
			return '<UNKNOWN>'
		rule = <Rule *>(&self.prodbuf.d.aschar[
				prodno * sizeof(Rule)])
		fanout = bit_popcount(<uint64_t>rule.lengths)
		for n in range(8 * sizeof(rule.args)):
			yf += '1' if (rule.args >> n) & 1 else '0'
			if (rule.lengths >> n) & 1:
				m += 1
				if m == fanout:
					break
				else:
					yf += ','
		rhs1 = rhs2 = word = ''
		lhs = self.idtolabel(rule.lhs)
		if rule.rhs1 == 0:
			word = repr(self.idtolabel(rule.args) if rule.args else None)
		else:
			rhs1 = self.idtolabel(rule.rhs1)
		if rule.rhs2 != 0:
			rhs2 = self.idtolabel(rule.rhs2)
		return '%s %s %s %s' % (yf if rhs1 else '',
				lhs, rhs1 or word, rhs2)

	def __repr__(self):
		if self.prods is None:
			return '<Vocabulary object>'
		return 'labels: %d, prods: %d' % (len(set(self.labels)), len(self.prods))

	def tofile(self, str filename):
		"""Helper function for pickling."""
		cdef FILE *out = fopen(filename.encode('utf8'), b'wb')
		cdef size_t written = 0
		cdef uint32_t header[3]
		header[:] = [len(self.prods), len(self.labels) + 1, self.labelbuf.len]
		if out is NULL:
			raise ValueError('could not open file.')
		try:
			written += fwrite(<void *>header, 1, sizeof(header), out)
			written += fwrite(self.prodbuf.d.ptr, 1, self.prodbuf.len, out)
			written += fwrite(self.labelidx.d.ptr, 1,
					self.labelidx.len * sizeof(uint32_t), out)
			written += fwrite(self.labelbuf.d.ptr, 1, self.labelbuf.len, out)
			if (written != sizeof(header) + self.prodbuf.len
					+ self.labelidx.len * sizeof(uint32_t) + self.labelbuf.len):
				raise ValueError('error writing to file.')
		finally:
			fclose(out)

	@classmethod
	def fromfile(cls, filename):
		"""Create a mutable Vocabulary object from a file."""
		cdef Vocabulary ob = Vocabulary.__new__(Vocabulary)
		cdef uint32_t header[3]
		cdef FILE *fp
		cdef int result
		fp = fopen(filename.encode('utf8'), b'rb')
		if fp is NULL:
			raise IOError
		try:
			result = fread(&header, sizeof(uint32_t), 3, fp)
			if result != 3:
				raise IOError
			ob.prodbuf.itemsize = sizeof(char)
			ob.prodbuf.capacity = ob.prodbuf.len = header[0] * sizeof(Rule)
			ob.prodbuf.d.ptr = malloc(
					ob.prodbuf.capacity * ob.prodbuf.itemsize)
			ob.labelidx.itemsize = sizeof(uint32_t)
			ob.labelidx.capacity = ob.labelidx.len = header[1]
			ob.labelidx.d.ptr = malloc(
					ob.labelidx.capacity * ob.labelidx.itemsize)
			ob.labelbuf.itemsize = sizeof(char)
			ob.labelbuf.capacity = ob.labelbuf.len = header[2]
			ob.labelbuf.d.ptr = malloc(
					ob.labelbuf.capacity * ob.labelbuf.itemsize)
			if (ob.prodbuf.d.ptr is NULL or ob.labelidx.d.ptr is NULL
					or ob.labelbuf.d.ptr is NULL):
				raise MemoryError
			result = fread(ob.prodbuf.d.ptr, ob.prodbuf.itemsize,
					ob.prodbuf.len, fp)
			if result != ob.prodbuf.len:
				raise IOError
			result = fread(ob.labelidx.d.ptr, ob.labelidx.itemsize,
					ob.labelidx.len, fp)
			if result != ob.labelidx.len:
				raise IOError
			result = fread(ob.labelbuf.d.ptr, ob.labelbuf.itemsize,
					ob.labelbuf.len, fp)
			if result != ob.labelbuf.len:
				raise IOError
		finally:
			fclose(fp)
		ob.prods = {<bytes>ob.prodbuf.d.aschar[n * sizeof(Rule):
				(n + 1) * sizeof(Rule)]: n
				for n in range(ob.prodbuf.len // sizeof(Rule))}
		ob.labels = {ob.labelbuf.d.aschar[ob.labelidx.d.asint[n]:
				ob.labelidx.d.asint[n + 1]].decode('utf8'): n
				for n in range(ob.labelidx.len - 1)}
		return ob


@cython.final
cdef class FixedVocabulary(Vocabulary):
	@classmethod
	def fromfile(cls, filename):
		"""Return an immutable Vocabulary object from a file."""
		cdef FixedVocabulary ob = FixedVocabulary.__new__(FixedVocabulary)
		cdef size_t offset = 3 * sizeof(uint32_t)
		cdef Py_buffer buffer
		cdef Py_ssize_t size = 0
		cdef char *ptr = NULL
		cdef uint32_t *header
		cdef int result
		fileno = os.open(filename, os.O_RDONLY)
		buf = mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)
		ob.state = (fileno, buf)
		result = getbufptr(buf, &ptr, &size, &buffer)
		if result != 0:
			raise ValueError('could not get buffer from mmap.')
		header = <uint32_t *>ptr
		ob.prodbuf.d.aschar = &(ptr[offset])
		ob.prodbuf.len = header[0] * sizeof(Rule)
		ob.prodbuf.capacity = 0
		offset += header[0] * sizeof(Rule)
		ob.labelidx.d.aschar = &(ptr[offset])
		ob.labelidx.len = header[1]
		ob.labelidx.capacity = 0
		offset += header[1] * sizeof(uint32_t)
		ob.labelbuf.d.aschar = &(ptr[offset])
		ob.labelbuf.len = header[2]
		ob.labelbuf.capacity = 0
		PyBuffer_Release(&buffer)
		return ob

	def __dealloc__(self):
		if self.state:
			self.state[1].close()
			os.close(self.state[0])
			self.state = None

	def makeindex(self):
		"""Build dictionaries; necessary for getprod()."""
		self.prods = {<bytes>self.prodbuf.d.aschar[n * sizeof(Rule):
				(n + 1) * sizeof(Rule)]: n
				for n in range(self.prodbuf.len // sizeof(Rule))}
		self.labels = {self.labelbuf.d.aschar[self.labelidx.d.asint[n]:
				self.labelidx.d.asint[n + 1]].decode('utf8'): n
				for n in range(self.labelidx.len - 1)}

	cdef int getprod(self, tuple r, tuple yf) except -2:
		"""Lookup production ID. Return -1 when not seen before."""
		cdef Rule rule
		cdef char *tmp = <char *>&rule
		res = self.labels.get(r[0], None)
		if res is None:
			return None
		rule.lhs = res
		res = self.labels.get(r[1], None) if len(r) > 1 else 0
		if res is None:
			return None
		rule.rhs1 = res
		if len(r) > 2:
			res = self.labels.get(r[2], None)
			if res is None:
				return None
			rule.rhs2 = res
		else:
			rule.rhs2 = 0
		if rule.rhs1 == 0 and len(r) > 1:
			res = self.labels.get(yf[0], None)
			if res is None:
				return None
			rule.args = res
			rule.lengths = 0
		else:
			getyf(yf, &rule.args, &rule.lengths)
		prod = <bytes>tmp[:sizeof(Rule)]
		return self.prods.get(prod, -1)


cdef inline gettree(list result, Node *tree, Vocabulary vocab, int i,
		bint disc):
	"""Collect string of tree and sentence for a tree.

	:param result: provide an empty list for the initial call.
	:param i: node number to start with."""
	cdef int j = tree[i].right
	result.append('(')
	result.append(vocab.getlabel(tree[i].prod))
	result.append(' ')
	if tree[i].left >= 0:
		gettree(result, tree, vocab, tree[i].left, disc)
		if tree[i].right >= 0:
			result.append(' ')
			gettree(result, tree, vocab, tree[i].right, disc)
	elif disc:
		result.append('%d=%s' % (
				termidx(tree[i].left),
				vocab.getword(tree[i].prod) or ''))
		# append rest of indices in case of disc. substitution site
		while j >= 0:
			result.append(' %d=' % termidx(tree[j].left))
			j = tree[j].right
	else:
		result.append(vocab.getword(tree[i].prod) or '')
	result.append(')')


cdef inline int getbufptr(
		object obj, char ** ptr, Py_ssize_t * size, Py_buffer * buf):
	"""Get a pointer from bytes/buffer object ``obj``.

	On success, return 0, and set ``ptr``, ``size``, and possibly ``buf``."""
	cdef int result = -1
	ptr[0] = NULL
	size[0] = 0
	if PyObject_CheckBuffer(obj) == 1:  # new-style Buffer interface
		result = PyObject_GetBuffer(obj, buf, PyBUF_SIMPLE)
		if result == 0:
			ptr[0] = <char *>buf.buf
			size[0] = buf.len
	return result


__all__ = ['Grammar', 'Chart', 'Ctrees', 'Vocabulary', 'FixedVocabulary']
