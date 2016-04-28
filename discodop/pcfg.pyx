"""CKY parser for Probabilistic Context-Free Grammar (PCFG)."""
from __future__ import print_function
from os import unlink
import re
import sys
import subprocess
from math import exp, log as pylog
from array import array
from itertools import count
import numpy as np
from .tree import Tree
from .util import which
from .plcfrs import DoubleAgenda
from .treebank import TERMINALSRE

cimport cython
include "constants.pxi"

cdef double INFINITY = float('infinity')


cdef class CFGChart(Chart):
	"""A Chart for context-free grammars (CFG).

	An item is a Python integer made up of ``start``, ``end``, ``lhs`` indices.
	"""
	def __init__(self, Grammar grammar, list sent,
			start=None, logprob=True, viterbi=True):
		raise NotImplementedError

	cdef _left(self, item, Edge *edge):
		cdef short start
		if edge.rule is NULL:
			return None
		start = <size_t>item / (
				self.grammar.nonterminals * self.lensent)
		return cellidx(start, edge.pos.mid, self.lensent,
				self.grammar.nonterminals) + edge.rule.rhs1

	cdef _right(self, item, Edge *edge):
		cdef short end
		if edge.rule is NULL or edge.rule.rhs2 == 0:
			return None
		end = <size_t>item / self.grammar.nonterminals % self.lensent + 1
		return cellidx(edge.pos.mid, end, self.lensent,
				self.grammar.nonterminals) + edge.rule.rhs2

	def root(self):
		return cellidx(0, self.lensent,
				self.lensent, self.grammar.nonterminals) + self.start

	cdef uint32_t label(self, item):
		return <size_t>item % self.grammar.nonterminals

	def indices(self, item):
		cdef short start = (item // self.grammar.nonterminals) // self.lensent
		cdef short end = (item // self.grammar.nonterminals) % self.lensent + 1
		return list(range(start, end))

	def itemstr(self, item):
		cdef uint32_t lhs = self.label(item)
		cdef short start = (item // self.grammar.nonterminals) // self.lensent
		cdef short end = (item // self.grammar.nonterminals) % self.lensent + 1
		return '%s[%d:%d]' % (
				self.grammar.tolabel[lhs], start, end)

	def getitems(self):
		return self.parseforest


@cython.final
cdef class DenseCFGChart(CFGChart):
	"""
	A CFG chart in which edges and probabilities are stored in a dense
	array; i.e., array is contiguous and all valid combinations of indices
	``0 <= start <= mid <= end`` and ``label`` can be addressed. Whether it is
	feasible to use this chart depends on the grammar constant, specifically
	the number of non-terminal labels."""
	def __init__(self, Grammar grammar, list sent,
			start=None, logprob=True, viterbi=True):
		self.grammar = grammar
		self.sent = sent
		self.lensent = len(sent)
		self.start = grammar.toid[grammar.start if start is None else start]
		self.logprob = logprob
		self.viterbi = viterbi
		entries = compactcellidx(self.lensent - 1, self.lensent, self.lensent,
				grammar.nonterminals) + grammar.nonterminals
		self.probs = <double *>malloc(entries * sizeof(double))
		if self.probs is NULL:
			raise MemoryError('allocation error')
		for n in range(entries):
			self.probs[n] = INFINITY
		# store parse forest in array instead of dict
		# FIXME: use compactcellidx?
		entries = cellidx(self.lensent - 1, self.lensent, self.lensent,
				grammar.nonterminals) + grammar.nonterminals
		self.parseforest = <EdgesStruct *>calloc(entries, sizeof(EdgesStruct))
		if self.parseforest is NULL:
			raise MemoryError('allocation error')
		self.itemsinorder = array(b'L' if PY2 else 'L')

	def __dealloc__(self):
		cdef size_t n, entries = cellidx(
				self.lensent - 1, self.lensent, self.lensent,
				self.grammar.nonterminals) + self.grammar.nonterminals
		cdef MoreEdges *cur
		cdef MoreEdges *tmp
		if self.probs is not NULL:
			free(self.probs)
		for n in range(entries):
			cur = self.parseforest[n].head
			while cur is not NULL:
				tmp = cur
				cur = cur.prev
				free(tmp)
		free(self.parseforest)

	cdef void addedge(self, uint32_t lhs, Idx start, Idx end, Idx mid,
			ProbRule *rule):
		"""Add new edge to parse forest."""
		cdef size_t item = cellidx(
				start, end, self.lensent, self.grammar.nonterminals) + lhs
		cdef Edge *edge
		cdef EdgesStruct *edges = &(self.parseforest[item])
		cdef MoreEdges *edgelist
		if edges.head is NULL:
			edgelist = <MoreEdges *>calloc(1, sizeof(MoreEdges))
			if edgelist is NULL:
				abort()
			edgelist.prev = NULL
			edges.head = edgelist
			self.itemsinorder.append(item)
		else:
			edgelist = edges.head
			if edges.len == EDGES_SIZE:
				edgelist = <MoreEdges *>calloc(1, sizeof(MoreEdges))
				if edgelist is NULL:
					abort()
				edgelist.prev = edges.head
				edges.head = edgelist
				edges.len = 0
		edge = &(edgelist.data[edges.len])
		edge.rule = rule
		edge.pos.mid = mid
		edges.len += 1

	cdef bint updateprob(self, uint32_t lhs, Idx start, Idx end, double prob,
			double beam):
		"""Update probability for item if better than current one."""
		cdef size_t idx = compactcellidx(
				start, end, self.lensent, self.grammar.nonterminals) + lhs
		cdef size_t beamitem
		if beam:
			# the item with label 0 for a cell holds the best score in a cell
			beamitem = compactcellidx(
					start, end, self.lensent, self.grammar.nonterminals)
			if prob > self.probs[beamitem] + beam:
				return False
			elif prob < self.probs[beamitem]:
				self.probs[beamitem] = self.probs[idx] = prob
			elif prob < self.probs[idx]:
				self.probs[idx] = prob
		elif prob < self.probs[idx]:
			self.probs[idx] = prob
		return True

	cdef double _subtreeprob(self, size_t item):
		"""Get viterbi / inside probability of a subtree headed by `item`."""
		cdef short start, end
		cdef uint32_t lhs
		cdef size_t idx
		lhs = item % self.grammar.nonterminals
		item /= self.grammar.nonterminals
		start = item / self.lensent
		end = item % self.lensent + 1
		idx = compactcellidx(
				start, end, self.lensent, self.grammar.nonterminals) + lhs
		return self.probs[idx]

	cdef double subtreeprob(self, item):
		return self._subtreeprob(<size_t>item)

	def getitems(self):
		return self.itemsinorder
		# return [n for n, a in enumerate(self.parseforest) if a is not None]

	cdef Edges getedges(self, item):
		"""Get edges for item."""
		if item is None:
			return None
		result = Edges()
		result.len = self.parseforest[item].len
		result.head = self.parseforest[item].head
		return result

	cpdef bint hasitem(self, item):
		"""Test if item is in chart."""
		return (item is not None
				and self.parseforest[<size_t>item].head is not NULL)

	def setprob(self, item, double prob):
		"""Set probability for item (unconditionally)."""
		cdef short start, end
		cdef uint32_t lhs
		cdef size_t idx
		lhs = item % self.grammar.nonterminals
		item /= self.grammar.nonterminals
		start = item / self.lensent
		end = item % self.lensent + 1
		idx = compactcellidx(
				start, end, self.lensent, self.grammar.nonterminals) + lhs
		self.probs[idx] = prob

	def __bool__(self):
		"""Return true when the root item is in the chart.

		i.e., test whether sentence has been parsed successfully."""
		return self.parseforest[self.root()].head is not NULL


@cython.final
cdef class SparseCFGChart(CFGChart):
	"""
	A CFG chart which uses a dictionary for each cell so that grammars
	with a large number of non-terminal labels can be handled."""
	def __init__(self, Grammar grammar, list sent,
			start=None, logprob=True, viterbi=True):
		self.grammar = grammar
		self.sent = sent
		self.lensent = len(sent)
		self.start = grammar.toid[grammar.start if start is None else start]
		self.logprob = logprob
		self.viterbi = viterbi
		self.probs = {}
		self.parseforest = {}
		self.itemsinorder = array(b'L' if PY2 else 'L')

	def __dealloc__(self):
		cdef MoreEdges *cur
		cdef MoreEdges *tmp
		for item in self.parseforest:
			cur = (<Edges>self.parseforest[item]).head
			while cur is not NULL:
				tmp = cur
				cur = cur.prev
				free(tmp)

	cdef void addedge(self, uint32_t lhs, Idx start, Idx end, Idx mid,
			ProbRule *rule):
		"""Add new edge to parse forest."""
		cdef Edges edges
		cdef MoreEdges *edgelist
		cdef Edge *edge
		cdef size_t item = cellidx(
				start, end, self.lensent, self.grammar.nonterminals) + lhs
		if item in self.parseforest:
			edges = self.parseforest[item]
			edgelist = edges.head
			if edges.len == EDGES_SIZE:
				edgelist = <MoreEdges *>calloc(1, sizeof(MoreEdges))
				if edgelist is NULL:
					abort()
				edgelist.prev = edges.head
				edges.head = edgelist
				edges.len = 0
		else:
			edges = Edges()
			self.parseforest[item] = edges
			edgelist = <MoreEdges *>calloc(1, sizeof(MoreEdges))
			if edgelist is NULL:
				abort()
			edges.head = edgelist
			self.itemsinorder.append(item)
		edge = &(edgelist.data[edges.len])
		edge.rule = rule
		edge.pos.mid = mid
		edges.len += 1

	cdef bint updateprob(self, uint32_t lhs, Idx start, Idx end, double prob,
			double beam):
		"""Update probability for item if better than current one."""
		cdef size_t item = cellidx(
				start, end, self.lensent, self.grammar.nonterminals) + lhs
		cdef size_t beamitem
		if beam:
			# the item with label 0 for a cell holds the best score in a cell
			beamitem = cellidx(
					start, end, self.lensent, self.grammar.nonterminals)
			if beamitem not in self.probs or prob < self.probs[beamitem]:
				self.probs[item] = self.probs[beamitem] = prob
			elif prob > self.probs[beamitem] + beam:
				return False
			elif item not in self.probs or prob < self.probs[item]:
				self.probs[item] = prob
		elif item not in self.probs or prob < self.probs[item]:
			self.probs[item] = prob
		return True

	cdef double _subtreeprob(self, size_t item):
		"""Get viterbi / inside probability of a subtree headed by `item`."""
		return (PyFloat_AS_DOUBLE(self.probs[item])
				if item in self.probs else INFINITY)

	cdef double subtreeprob(self, item):
		return self._subtreeprob(item)

	cpdef bint hasitem(self, item):
		"""Test if item is in chart."""
		return item in self.parseforest

	def setprob(self, item, prob):
		"""Set probability for item (unconditionally)."""
		self.probs[item] = prob


def parse(sent, Grammar grammar, tags=None, start=None, list whitelist=None,
		bint symbolic=False, double beam_beta=0.0, int beam_delta=50):
	"""A CKY parser modeled after Bodenstab's 'fast grammar loop'.

	:param sent: A sequence of tokens that will be parsed.
	:param grammar: A ``Grammar`` object.
	:returns: a ``Chart`` object.
	:param tags: Optionally, a sequence of POS tags to use instead of
		attempting to apply all possible POS tags.
	:param start: integer corresponding to the start symbol that complete
		derivations should be headed by; e.g., ``grammar.toid['ROOT']``.
		If not given, the default specified by ``grammar`` is used.
	:param whitelist: a list of items that may enter the chart.
		The whitelist is a list of cells consisting of sets of labels:
		``whitelist = [{label1, label2, ...}, ...]``;
		The cells are indexed as compact spans; label is an integer for a
		non-terminal label. The presence of a label means the span with that
		label will not be pruned.
	:param symbolic: If ``True``, parse sentence without regard for
		probabilities. All Viterbi probabilities will be set to ``1.0``.
	:param beam_beta: keep track of the best score in each cell and only allow
		items which are within a multiple of ``beam_beta`` of the best score.
		Should be a negative log probability. Pass ``0.0`` to disable.
	:param beam_delta: the maximum span length to which beam search is applied.
	"""
	if grammar.maxfanout != 1:
		raise ValueError('Not a PCFG! fanout: %d' % grammar.maxfanout)
	if not grammar.logprob:
		raise ValueError('Expected grammar with log probabilities.')
	if grammar.nonterminals < 20000:
		chart = DenseCFGChart(grammar, sent, start)
		if symbolic:
			return parse_symbolic(sent, <DenseCFGChart>chart, grammar,
					tags=tags, whitelist=whitelist)
		return parse_main(sent, <DenseCFGChart>chart, grammar, tags,
				whitelist, beam_beta, beam_delta)
	chart = SparseCFGChart(grammar, sent, start)
	if symbolic:
		return parse_symbolic(sent, <SparseCFGChart>chart, grammar,
				tags=tags, whitelist=whitelist)
	return parse_main(sent, <SparseCFGChart>chart, grammar, tags,
			whitelist, beam_beta, beam_delta)


cdef parse_main(sent, CFGChart_fused chart, Grammar grammar, tags,
		list whitelist, double beam_beta, int beam_delta):
	cdef:
		short [:, :] minleft, maxleft, minright, maxright
		DoubleAgenda unaryagenda = DoubleAgenda()
		set cellwhitelist = None
		ProbRule *rule
		short left, right, mid, span, lensent = len(sent)
		short narrowl, narrowr, widel, wider, minmid, maxmid
		double oldscore, prob
		uint32_t n, lhs = 0, rhs1
		size_t cell, lastidx
		object it = None
	minleft, maxleft, minright, maxright = minmaxmatrices(
			grammar.nonterminals, lensent)
	# assign POS tags
	covered, msg = populatepos(grammar, chart, sent, tags, whitelist, False,
			minleft, maxleft, minright, maxright)
	if not covered:
		return chart, msg

	for span in range(2, lensent + 1):
		# constituents from left to right
		for left in range(lensent - span + 1):
			right = left + span
			cell = cellidx(left, right, lensent, grammar.nonterminals)
			lastidx = len(chart.itemsinorder)
			if whitelist is not None:
				cellwhitelist = <set>whitelist[
						compactcellidx(left, right, lensent, 1)]
			# apply binary rules; if whitelist is given, loop only over
			# whitelisted labels for cell; equivalent to:
			# for lhs in cellwhitelist or range(1, grammar.phrasalnonterminals):
			if whitelist is None:
				lhs = 0
			else:
				it = iter(cellwhitelist)
			while True:
				if whitelist is None:
					lhs += 1
					if lhs >= grammar.phrasalnonterminals:
						break
				else:
					try:
						lhs = next(it)
					except StopIteration:
						break
				n = 0
				rule = &(grammar.bylhs[lhs][n])
				oldscore = chart._subtreeprob(cell + lhs)
				while rule.lhs == lhs:
					narrowr = minright[rule.rhs1, left]
					narrowl = minleft[rule.rhs2, right]
					if (rule.rhs2 == 0 or narrowr >= right or narrowl < narrowr
							or TESTBIT(grammar.mask, rule.no)):
						n += 1
						rule = &(grammar.bylhs[lhs][n])
						continue
					widel = maxleft[rule.rhs2, right]
					minmid = narrowr if narrowr > widel else widel
					wider = maxright[rule.rhs1, left]
					maxmid = wider if wider < narrowl else narrowl
					for mid in range(minmid, maxmid + 1):
						leftitem = cellidx(left, mid,
								lensent, grammar.nonterminals) + rule.rhs1
						rightitem = cellidx(mid, right,
								lensent, grammar.nonterminals) + rule.rhs2
						if (chart.hasitem(leftitem)
								and chart.hasitem(rightitem)):
							prob = (rule.prob + chart._subtreeprob(leftitem)
									+ chart._subtreeprob(rightitem))
							if chart.updateprob(lhs, left, right, prob,
									beam_beta if span <= beam_delta else 0.0):
								chart.addedge(lhs, left, right, mid, rule)
					n += 1
					rule = &(grammar.bylhs[lhs][n])

				# update filter
				if isinf(oldscore):
					if not chart.hasitem(cell + lhs):
						continue
					if left > minleft[lhs, right]:
						minleft[lhs, right] = left
					if left < maxleft[lhs, right]:
						maxleft[lhs, right] = left
					if right < minright[lhs, left]:
						minright[lhs, left] = right
					if right > maxright[lhs, left]:
						maxright[lhs, left] = right

			# unary rules
			unaryagenda.update_entries([new_DoubleEntry(
						chart.label(item), chart._subtreeprob(item), 0)
						for item in chart.itemsinorder[lastidx:]])
			while unaryagenda.length:
				rhs1 = unaryagenda.popentry().key
				for n in range(grammar.numunary):
					rule = &(grammar.unary[rhs1][n])
					if rule.rhs1 != rhs1:
						break
					elif TESTBIT(grammar.mask, rule.no) or (
							whitelist is not None
							and rule.lhs not in cellwhitelist):
						continue
					lhs = rule.lhs
					prob = rule.prob + chart._subtreeprob(cell + rhs1)
					chart.addedge(lhs, left, right, right, rule)
					if (not chart.hasitem(cell + lhs)
							or prob < chart._subtreeprob(cell + lhs)):
						chart.updateprob(lhs, left, right, prob, 0.0)
						unaryagenda.setifbetter(lhs, prob)
					# update filter
					if left > minleft[lhs, right]:
						minleft[lhs, right] = left
					if left < maxleft[lhs, right]:
						maxleft[lhs, right] = left
					if right < minright[lhs, left]:
						minright[lhs, left] = right
					if right > maxright[lhs, left]:
						maxright[lhs, left] = right
			unaryagenda.clear()
	if not chart:
		return chart, 'no parse ' + chart.stats()
	return chart, chart.stats()


cdef parse_symbolic(sent, CFGChart_fused chart, Grammar grammar,
		tags=None, list whitelist=None):
	cdef:
		short [:, :] minleft, maxleft, minright, maxright
		list unaryagenda
		set cellwhitelist = None
		object it = None
		ProbRule *rule
		short left, right, mid, span, lensent = len(sent)
		short narrowl, narrowr, widel, wider, minmid, maxmid
		uint32_t n, lhs = 0, rhs1
		size_t cell, lastidx
		bint haditem
	minleft, maxleft, minright, maxright = minmaxmatrices(
			grammar.nonterminals, lensent)
	# assign POS tags
	covered, msg = populatepos(grammar, chart, sent, tags, whitelist, True,
			minleft, maxleft, minright, maxright)
	if not covered:
		return chart, msg

	for span in range(2, lensent + 1):
		# constituents from left to right
		for left in range(lensent - span + 1):
			right = left + span
			cell = cellidx(left, right, lensent, grammar.nonterminals)
			lastidx = len(chart.itemsinorder)
			if whitelist is not None:
				cellwhitelist = <set>whitelist[
						compactcellidx(left, right, lensent, 1)]
			# apply binary rules; if whitelist is given, loop only over
			# whitelisted labels for cell
			# for lhs in (range(1, grammar.phrasalnonterminals)
			# 		if whitelist is None else cellwhitelist):
			if whitelist is None:
				lhs = 0
			else:
				it = iter(cellwhitelist)
			while True:
				if whitelist is None:
					lhs += 1
					if lhs >= grammar.phrasalnonterminals:
						break
				else:
					try:
						lhs = next(it)
					except StopIteration:
						break
				n = 0
				rule = &(grammar.bylhs[lhs][n])
				haditem = chart.hasitem(cell + lhs)
				while rule.lhs == lhs:
					narrowr = minright[rule.rhs1, left]
					narrowl = minleft[rule.rhs2, right]
					if (rule.rhs2 == 0 or narrowr >= right or narrowl < narrowr
							or TESTBIT(grammar.mask, rule.no)):
						n += 1
						rule = &(grammar.bylhs[lhs][n])
						continue
					widel = maxleft[rule.rhs2, right]
					minmid = narrowr if narrowr > widel else widel
					wider = maxright[rule.rhs1, left]
					maxmid = wider if wider < narrowl else narrowl
					for mid in range(minmid, maxmid + 1):
						leftitem = cellidx(left, mid,
								lensent, grammar.nonterminals) + rule.rhs1
						rightitem = cellidx(mid, right,
								lensent, grammar.nonterminals) + rule.rhs2
						if (chart.hasitem(leftitem)
								and chart.hasitem(rightitem)):
							chart.addedge(lhs, left, right, mid, rule)
							chart.updateprob(lhs, left, right, 0.0, 0.0)
					n += 1
					rule = &(grammar.bylhs[lhs][n])

				# update filter
				if not haditem and chart.hasitem(cell + lhs):
					if left > minleft[lhs, right]:
						minleft[lhs, right] = left
					if left < maxleft[lhs, right]:
						maxleft[lhs, right] = left
					if right < minright[lhs, left]:
						minright[lhs, left] = right
					if right > maxright[lhs, left]:
						maxright[lhs, left] = right

			# unary rules
			unaryagenda = [chart.label(item)
					for item in chart.itemsinorder[lastidx:]]
			while unaryagenda:
				rhs1 = unaryagenda.pop()
				for n in range(grammar.numunary):
					rule = &(grammar.unary[rhs1][n])
					if rule.rhs1 != rhs1:
						break
					elif TESTBIT(grammar.mask, rule.no) or (
							whitelist is not None
							and rule.lhs not in cellwhitelist):
						continue
					lhs = rule.lhs
					chart.addedge(lhs, left, right, right, rule)
					if not chart.hasitem(cell + lhs):
						chart.updateprob(lhs, left, right, 0.0, 0.0)
					# update filter
					if left > minleft[lhs, right]:
						minleft[lhs, right] = left
					if left < maxleft[lhs, right]:
						maxleft[lhs, right] = left
					if right < minright[lhs, left]:
						minright[lhs, left] = right
					if right > maxright[lhs, left]:
						maxright[lhs, left] = right
	if not chart:
		return chart, 'no parse ' + chart.stats()
	return chart, chart.stats()


cdef populatepos(Grammar grammar, CFGChart_fused chart, sent, tags, whitelist,
		bint symbolic, short [:, :] minleft, short [:, :] maxleft,
		short [:, :] minright, short [:, :] maxright):
	"""Apply all possible lexical and unary rules on each lexical span.

	:returns: a tuple ``(success, msg)`` where ``success`` is True if a POS tag
	was found for every word in the sentence."""
	cdef:
		DoubleAgenda unaryagenda = DoubleAgenda()
		ProbRule *rule
		LexicalRule lexrule
		uint32_t n, lhs, rhs1
		short left, right, lensent = len(sent)
	for left, word in enumerate(sent):
		tag = tags[left] if tags else None
		right = left + 1
		recognized = False
		for lexrule in grammar.lexicalbyword.get(word, ()):
			# assert whitelist is None or cell in whitelist, whitelist.keys()
			if whitelist is not None and lexrule.lhs not in whitelist[
					compactcellidx(left, right, lensent, 1)]:
				continue
			lhs = lexrule.lhs
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if (tag is None or grammar.tolabel[lhs] == tag
					or grammar.tolabel[lhs].startswith(tag + '@')):
				chart.addedge(lhs, left, right, right, NULL)
				chart.updateprob(lhs, left, right,
						0.0 if symbolic else lexrule.prob, 0.0)
				unaryagenda.setitem(lhs, lexrule.prob)
				recognized = True
				# update filter
				if left > minleft[lhs, right]:
					minleft[lhs, right] = left
				if left < maxleft[lhs, right]:
					maxleft[lhs, right] = left
				if right < minright[lhs, left]:
					minright[lhs, left] = right
				if right > maxright[lhs, left]:
					maxright[lhs, left] = right
		# NB: don't allow blocking of gold tags if given
		if not recognized and tag is not None and tag in grammar.toid:
			lhs = grammar.toid[tag]
			chart.addedge(lhs, left, right, right, NULL)
			chart.updateprob(lhs, left, right, 0.0, 0.0)
			unaryagenda.setitem(lhs, 0.0)
			recognized = True
			# update filter
			if left > minleft[lhs, right]:
				minleft[lhs, right] = left
			if left < maxleft[lhs, right]:
				maxleft[lhs, right] = left
			if right < minright[lhs, left]:
				minright[lhs, left] = right
			if right > maxright[lhs, left]:
				maxright[lhs, left] = right
		elif not recognized:
			if tag is None and word not in grammar.lexicalbyword:
				return chart, 'no parse: %r not in lexicon' % word
			elif tag is not None and tag not in grammar.toid:
				return chart, 'no parse: unknown tag %r' % tag
			return chart, 'no parse: all tags for %r blocked' % word

		# unary rules on the span of this POS tag
		while unaryagenda.length:
			rhs1 = unaryagenda.popentry().key
			for n in range(grammar.numunary):
				rule = &(grammar.unary[rhs1][n])
				if rule.rhs1 != rhs1:
					break
				elif TESTBIT(grammar.mask, rule.no) or (
						whitelist is not None
						and rule.lhs not in whitelist[
							compactcellidx(left, right, lensent, 1)]):
					continue
				lhs = rule.lhs
				item = cellidx(left, right, lensent, grammar.nonterminals) + lhs
				# FIXME can vit.prob change while entry in agenda?
				# prob = rule.prob + entry.value
				prob = rule.prob + chart._subtreeprob(cellidx(
						left, right, lensent, grammar.nonterminals) + rhs1)
				if (not chart.hasitem(item) or
						prob < chart._subtreeprob(item)):
					unaryagenda.setifbetter(lhs, prob)
				chart.addedge(lhs, left, right, right, rule)
				chart.updateprob(lhs, left, right,
						0.0 if symbolic else prob, 0.0)
				# update filter
				if left > minleft[lhs, right]:
					minleft[lhs, right] = left
				if left < maxleft[lhs, right]:
					maxleft[lhs, right] = left
				if right < minright[lhs, left]:
					minright[lhs, left] = right
				if right > maxright[lhs, left]:
					maxright[lhs, left] = right
	return True, ''


def minmaxmatrices(nonterminals, lensent):
	"""Create matrices to track minima and maxima for binary splits."""
	minleft = np.empty((nonterminals, lensent + 1), dtype='int16')
	maxleft = np.empty_like(minleft)
	minleft[...], maxleft[...] = -1, lensent + 1
	minright, maxright = maxleft.copy(), minleft.copy()
	return minleft, maxleft, minright, maxright


BITPARUNESCAPE = re.compile(r"\\([\"\\ $\^'()\[\]{}=<>#])")
BITPARPARSES = re.compile(r'^vitprob=(.*)\n(\(.*\))\n', re.MULTILINE)
BITPARPARSESLOG = re.compile(r'^logvitprob=(.*)\n(\(.*\))\n', re.MULTILINE)
CPUTIME = re.compile('^raw cpu time (.+)$', re.MULTILINE)
LOG10 = pylog(10)


def parse_bitpar(grammar, rulesfile, lexiconfile, sent, n,
		startlabel, startid, tags=None):
	"""Parse a sentence with bitpar, given filenames of rules and lexicon.

	:param n: the number of derivations to return (max 1000); if n == 0, return
		parse forest instead of n-best list (requires binarized grammar).
	:returns: a dictionary of derivations with their probabilities."""
	if n < 1 or n > 1000:
		raise ValueError('with bitpar number of derivations n should be '
				'1 <= n <= 1000. got: n = %d' % n)
	chart = SparseCFGChart(grammar, sent, start=startlabel,
			logprob=True, viterbi=True)
	if n == 0:
		if not chart.grammar.binarized:
			raise ValueError('Extracing parse forest, '
					'expected binarized grammar.')
	else:
		chart.rankededges = {chart.root(): []}
	tmp = None
	if tags:
		import tempfile
		tmp = tempfile.NamedTemporaryFile(delete=False)
		# NB: this doesn't work with the tags from the DOP reduction
		tmp.writelines(set(['%s@%s\t%s@%s 1\t%s 1\n' % (t, w, t, w, t)
				for t, w in zip(tags, sent)]))
		tmp.close()
		lexiconfile = tmp.name
	tokens = [token.encode('utf8') for token in sent]
	if tags:
		tokens = ['%s@%s' % (tag, token) for tag, token in zip(tags, tokens)]
	# pass empty 'unkwown word file' to disable bitpar's smoothing
	args = ['-y'] if n == 0 else ['-b', str(n)]
	args += ['-s', startlabel, '-vp', '-u', '/dev/null', rulesfile, lexiconfile]
	proc = subprocess.Popen([which('bitpar')] + args,
			shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
			stderr=subprocess.PIPE)
	results, msg = proc.communicate(('\n'.join(tokens) + '\n').encode('utf8'))
	msg = msg.replace('Warning: Word class 0 did not occur!\n',
			'').decode('utf8').strip()
	match = CPUTIME.search(msg)
	cputime = float(match.group(1)) if match else 0.0
	if tags:
		unlink(tmp.name)
	# decode results or not?
	if not results or results.startswith('No parse'):
		return chart, cputime, '%s\n%s' % (results.decode('utf8').strip(), msg)
	elif n == 0:
		bitpar_yap_forest(results, chart)
	else:
		bitpar_nbest(results, chart)
	return chart, cputime, ''


def bitpar_yap_forest(forest, SparseCFGChart chart):
	"""Read bitpar YAP parse forest (-y option) into a Chart object.

	The forest has lines of the form::
		label start end     prob [edge1] % prob [edge2] % .. %%

	where an edge is either a quoted "word", or a rule number and one or two
	line numbers in the parse forest referring to children.
	Assumes binarized grammar. Assumes chart's Grammar object has same order of
	grammar rules as the grammar that was presented to bitpar."""
	cdef ProbRule *rule
	cdef uint32_t lhs
	cdef Idx left, right, mid
	cdef size_t ruleno, child1
	cdef double prob
	if not chart.grammar.binarized:
		raise ValueError('Extracing parse forest, expected binarized grammar.')
	forest = forest.strip().splitlines()
	midpoints = [int(line.split(None, 3)[2]) for line in forest]
	for line in forest:
		a, b, c, fields = line.rstrip('% ').split(None, 3)
		lhs, left, right = chart.grammar.toid[a], int(b), int(c)
		# store 1-best probability, other probabilities can be ignored.
		prob = -pylog(float(fields.split(None, 1)[0]))
		chart.updateprob(lhs, left, right, prob, 0.0)
		for edge in fields.split(' % '):
			unused_prob, rest = edge.split(None, 1)
			if rest.startswith('"'):
				mid = right
				rule = NULL
			else:
				restsplit = rest.split(None, 2)
				ruleno = int(restsplit[0])
				child1 = int(restsplit[1])
				# ignore second child: (midpoint + end of current node suffices)
				# child2 = restsplit[2] if len(restsplit) > 2 else None
				mid = midpoints[child1]
				rule = &(chart.grammar.bylhs[0][chart.grammar.revmap[ruleno]])
			chart.addedge(lhs, left, right, mid, rule)


def bitpar_nbest(nbest, SparseCFGChart chart):
	"""Put bitpar's list of n-best derivations into the chart.
	Parse forest is not converted."""
	lines = BITPARUNESCAPE.sub(r'\1', nbest).replace(')(', ') (')
	derivs = [(renumber(deriv), -float(prob) * LOG10)
			for prob, deriv in BITPARPARSESLOG.findall(lines)]
	if not derivs:
		derivs = [(renumber(deriv), -pylog(float(prob) or 5.e-130))
				for prob, deriv in BITPARPARSES.findall(lines)]
	chart.parseforest = {chart.root(): None}  # dummy so bool(chart) == True
	chart.rankededges[chart.root()] = derivs


def renumber(deriv):
	"""Replace terminals of CF-derivation (string) with indices."""
	it = count()
	return TERMINALSRE.sub(lambda _: ' %s)' % next(it), deriv)


def test():
	from .containers import Grammar
	from .disambiguation import getderivations, marginalize
	from operator import itemgetter
	cfg = Grammar([
		((('A', 'A'), ((0, ), )), 0.7), ((('A', 'B'), ((0, ), )), 0.6),
		((('A', 'C'), ((0, ), )), 0.5), ((('A', 'D'), ((0, ), )), 0.4),
		((('B', 'A'), ((0, ), )), 0.3), ((('B', 'B'), ((0, ), )), 0.2),
		((('B', 'C'), ((0, ), )), 0.1), ((('B', 'D'), ((0, ), )), 0.2),
		((('B', 'C'), ((0, ), )), 0.3), ((('C', 'A'), ((0, ), )), 0.4),
		((('C', 'B'), ((0, ), )), 0.5), ((('C', 'C'), ((0, ), )), 0.6),
		((('C', 'D'), ((0, ), )), 0.7), ((('D', 'A'), ((0, ), )), 0.8),
		((('D', 'B'), ((0, ), )), 0.9), ((('D', 'NP', 'VP'), ((0, 1), )), 1),
		((('D', 'C'), ((0, ), )), 0.8), ((('S', 'D'), ((0, ), )), 0.5),
		((('S', 'A'), ((0, ), )), 0.8), ((('NP', 'Epsilon'), ('mary', )), 1),
		((('VP', 'Epsilon'), ('walks', )), 1)],
		start='S')
	print(cfg)
	print('cfg parsing; sentence: mary walks')
	print('pcfg')
	chart, msg = parse('mary walks'.split(), cfg)
	assert chart, msg
	# chart, msg = parse_sparse('mary walks'.split(), cfg)
	# assert chart, msg
	print(chart)
	rules = [
		((('NP', 'NP', 'PP'), ((0, 1), )), 0.4),
		((('PP', 'P', 'NP'), ((0, 1), )), 1),
		((('S', 'NP', 'VP'), ((0, 1), )), 1),
		((('VP', 'V', 'NP'), ((0, 1), )), 0.7),
		((('VP', 'VP', 'PP'), ((0, 1), )), 0.3),
		((('NP', 'Epsilon'), ('astronomers', )), 0.1),
		((('NP', 'Epsilon'), ('ears', )), 0.18),
		((('V', 'Epsilon'), ('saw', )), 1),
		((('NP', 'Epsilon'), ('saw', )), 0.04),
		((('NP', 'Epsilon'), ('stars', )), 0.18),
		((('NP', 'Epsilon'), ('telescopes', )), 0.1),
		((('P', 'Epsilon'), ('with', )), 1)]
	cfg2 = Grammar(rules, start='S')
	sent = 'astronomers saw stars with telescopes'.split()
	cfg2.switch(u'default', True)
	chart, msg = parse(sent, cfg2)
	print(msg)
	print(chart)
	derivations, entries = getderivations(chart, 10, True, False, True)
	mpp, _ = marginalize('mpp', derivations, entries, chart)
	for a, p, _ in sorted(mpp, key=itemgetter(1), reverse=True):
		print(p, a)
	# chart1, msg1 = parse(sent, cfg2, symbolic=True)
	# print(msg, '\n', msg1)

__all__ = ['CFGChart', 'DenseCFGChart', 'SparseCFGChart', 'parse', 'renumber',
		'minmaxmatrices', 'parse_bitpar', 'bitpar_yap_forest', 'bitpar_nbest']
