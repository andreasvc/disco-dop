"""CKY parser for Probabilistic Context-Free Grammar (PCFG)."""
from __future__ import print_function
import re
import sys
import subprocess
from os import unlink
from math import exp, log as pylog
from itertools import count
import numpy as np
from .tree import Tree
from .util import which

cimport cython
from cython.operator cimport postincrement, dereference
from libc.math cimport HUGE_VAL as INFINITY
include "constants.pxi"


cdef inline uint64_t cellstruct(Idx start, Idx end):
	cdef CFGItem result
	result.st.start = start
	result.st.end = end
	result.st.label = 0
	return result.dt


cdef class CFGChart(Chart):
	"""A Chart for context-free grammars (CFG).

	An item is a triple ``(start, end, label)``."""
	def __init__(self, Grammar grammar, list sent,
			start=None, logprob=True, viterbi=True):
		raise NotImplementedError

	cdef Label label(self, ItemNo itemidx):
		raise NotImplementedError

	cdef Prob subtreeprob(self, ItemNo itemidx):
		"""Return probability of subtree headed by item."""
		raise NotImplementedError


@cython.final
cdef class DenseCFGChart(CFGChart):
	"""A CFG chart with fixed, pre-allocated arrays.

	All possible chart items are stored in dense, pre-allocated arrays; i.e.,
	array is contiguous and all valid combinations of indices ``0 <= start <=
	mid <= end`` and ``label`` can be addressed. Whether it is feasible to use
	this chart depends on the grammar constant, specifically the number of
	non-terminal labels (and to a lesser extent the sentence length)."""
	def __init__(self, Grammar grammar, list sent,
			start=None, logprob=True, viterbi=True):
		self.grammar = grammar
		self.sent = sent
		self.lensent = len(sent)
		self.start = grammar.toid[grammar.start if start is None else start]
		self.logprob = logprob
		self.viterbi = viterbi
		# FIXME: use compactcellidx?
		# entries = compactcellidx(self.lensent - 1, self.lensent,
		# 		self.lensent, grammar.nonterminals) + grammar.nonterminals
		entries = cellidx(self.lensent - 1, self.lensent, self.lensent,
				grammar.nonterminals) + grammar.nonterminals
		self.items.reserve(entries)
		self.items.push_back(0)
		# NB: resize not reserve; will not resize again.
		self.probs.resize(entries, INFINITY)
		self.parseforest.resize(entries)

	def root(self):
		return cellidx(0, self.lensent, self.lensent,
			self.grammar.nonterminals) + self.start

	def bestsubtree(self, start, end):
		cdef Prob bestprob = INFINITY, prob
		cdef uint64_t bestitem = 0
		cdef uint64_t cell = cellidx(start, end, self.lensent,
			self.grammar.nonterminals)
		for label in range(self.grammar.nonterminals):
			if ('*' in self.grammar.tolabel[label]
					or '<' in self.grammar.tolabel[label]):
				continue
			prob = self._subtreeprob(cell + label)
			if prob < bestprob:
				bestprob = prob
				bestitem = cell + label
		return bestitem

	cdef void addedge(self, uint64_t item, Idx mid, ProbRule *rule):
		"""Add new edge to parse forest."""
		cdef Edge edge
		edge.rule = rule
		edge.pos.lvec = mid
		self.parseforest[item].push_back(edge)

	cdef bint updateprob(self, uint64_t item, Prob prob, Prob beam):
		"""Update probability for item if better than current one.

		Add item if not seen before; return False if pruned."""
		cdef uint64_t beamitem, itemx
		cdef bint newitem = self.probs[item] == INFINITY
		if beam:
			itemx = item - item % self.grammar.nonterminals
			start = itemx // (self.grammar.nonterminals * self.lensent)
			end = itemx // self.grammar.nonterminals % self.lensent + 1
			beamitem = compactcellidx(start, end, self.lensent, 1)
			if prob > self.beambuckets[beamitem]:  # prob falls outside of beam
				return False
			elif prob + beam < self.beambuckets[beamitem]:  # shrink beam
				self.beambuckets[beamitem] = prob + beam
				self.probs[item] = prob
			elif prob < self.probs[item]:  # prob falls within beam
				self.probs[item] = prob
		elif prob < self.probs[item]:
			self.probs[item] = prob
		# can infer order of binary rules, but need to track unaries explicitly
		if newitem:
			self.items.push_back(item)
		return True

	cdef ItemNo _left(self, ItemNo itemidx, Edge edge):
		cdef uint64_t item = itemidx
		cdef short start
		if edge.rule is NULL:
			return 0
		start = item // (self.grammar.nonterminals * self.lensent)
		return cellidx(start, edge.pos.mid, self.lensent,
				self.grammar.nonterminals) + edge.rule.rhs1

	cdef ItemNo _right(self, ItemNo itemidx, Edge edge):
		cdef uint64_t item = itemidx
		cdef short end
		if edge.rule is NULL or edge.rule.rhs2 == 0:
			return 0
		end = item // self.grammar.nonterminals % self.lensent + 1
		return cellidx(edge.pos.mid, end, self.lensent,
				self.grammar.nonterminals) + edge.rule.rhs2

	cdef Label _label(self, uint64_t item):
		return item % self.grammar.nonterminals

	cdef Label label(self, ItemNo itemidx):
		cdef uint64_t item = itemidx
		return item % self.grammar.nonterminals

	cdef Prob _subtreeprob(self, uint64_t item):
		"""Get viterbi / inside probability of a subtree headed by `item`."""
		return self.probs[item]

	cdef Prob subtreeprob(self, ItemNo itemidx):
		cdef uint64_t item = itemidx
		return self.probs[item]

	cdef bint _hasitem(self, uint64_t item):
		"""Test if item is in chart."""
		return self.probs[item] != INFINITY
		# return self.parseforest[item].size() != 0

	def indices(self, ItemNo itemidx):
		cdef uint64_t item = itemidx
		cdef short start = (item // self.grammar.nonterminals) // self.lensent
		cdef short end = (item // self.grammar.nonterminals) % self.lensent + 1
		return list(range(start, end))

	def itemstr(self, ItemNo itemidx):
		cdef uint64_t item = itemidx
		cdef Label lhs = item % self.grammar.nonterminals
		cdef short start = (item // self.grammar.nonterminals) // self.lensent
		cdef short end = (item // self.grammar.nonterminals) % self.lensent + 1
		return '%s[%d:%d]' % (self.grammar.tolabel[lhs], start, end)

	def numitems(self):
		return self.items.size() - 1

	cdef ItemNo getitemidx(self, uint64_t n):
		"""Get itemidx of n'th item.

		:param n: an index in range(0, self.items.size())
		:returns: the ItemNo that is the n'th item in the chart.

		With the other charts the n'th item simply has itemidx n,
		but for this chart we need a level of indirection because
		itemidx is the item itself."""
		return self.items[n]

	def itemid(self, str label, indices, Whitelist whitelist=None):
		cdef Label labelid
		try:
			labelid = self.grammar.toid[label]
		except KeyError:
			return 0
		return self.itemid1(labelid, indices, whitelist)

	def itemid1(self, Label labelid, indices, Whitelist whitelist=None):
		cdef short left = min(indices)
		cdef short right = max(indices) + 1
		item = cellidx(left, right, self.lensent, self.grammar.nonterminals
				) + labelid
		if whitelist is not None:
			return whitelist.cfg[compactcellidx(left, right, self.lensent, 1)
					].count(whitelist.mapping[labelid]) != 0 and item
		return self.parseforest[item].size() != 0 and item

	cdef SmallChartItem asSmallChartItem(self, ItemNo itemidx):
		cdef CFGItem item
		item.dt = itemidx
		label = item.dt % self.grammar.nonterminals
		item.dt //= self.grammar.nonterminals
		start = item.dt // self.lensent
		end = item.dt % self.lensent + 1
		return CFGtoSmallChartItem(label, start, end)

	cdef FatChartItem asFatChartItem(self, ItemNo itemidx):
		cdef CFGItem item
		item.dt = itemidx
		label = item.dt % self.grammar.nonterminals
		item.dt //= self.grammar.nonterminals
		start = item.dt // self.lensent
		end = item.dt % self.lensent + 1
		return CFGtoFatChartItem(label, start, end)

	cdef size_t asCFGspan(self, ItemNo itemidx):
		cdef CFGItem item
		item.dt = itemidx
		item.dt //= self.grammar.nonterminals
		start = item.dt // self.lensent
		end = item.dt % self.lensent + 1
		assert 0 <= start < end <= self.lensent
		return compactcellidx(start, end, self.lensent, 1)


@cython.final
cdef class SparseCFGChart(CFGChart):
	"""A CFG chart which uses a hash table suitable for large grammars."""
	def __init__(self, Grammar grammar, list sent,
			start=None, logprob=True, viterbi=True, itemsestimate=None):
		cdef uint64_t sentinel = cellstruct(0, 0)
		self.grammar = grammar
		self.sent = sent
		self.lensent = len(sent)
		self.start = grammar.toid[grammar.start if start is None else start]
		self.logprob = logprob
		self.viterbi = viterbi
		if itemsestimate is not None:
			self.items.reserve(itemsestimate)
			self.itemindex.reserve(itemsestimate)
			self.parseforest.reserve(itemsestimate)
			self.probs.reserve(itemsestimate)
		self.items.push_back(sentinel)
		self.itemindex[sentinel] = 0
		self.probs.push_back(INFINITY)

	def root(self):
		return self.itemindex[cellstruct(0, self.lensent) + self.start]

	def bestsubtree(self, start, end):
		cdef Prob bestprob = INFINITY, prob
		cdef uint64_t bestitem = 0
		cdef uint64_t cell = cellstruct(start, end)
		# FIXME: can iterate over all available items in cell
		# by querying for appropriate range:
		# range(cellstruct(start, end),
		# 		cellstruct(start, end) + self.grammar.nonterminals)
		for label in range(self.grammar.nonterminals):
			if ('*' in self.grammar.tolabel[label]
					or '<' in self.grammar.tolabel[label]):
				continue
			prob = self._subtreeprob(cell + label)
			if prob < bestprob:
				bestprob = prob
				bestitem = self.itemindex[cell + label]
		return bestitem

	cdef void addedge(self, uint64_t item, Idx mid, ProbRule *rule):
		"""Add new edge to parse forest."""
		cdef ItemNo itemidx = self.itemindex[item]
		cdef Edge edge
		edge.rule = rule
		edge.pos.lvec = 0UL
		edge.pos.mid = mid
		self.parseforest[itemidx].push_back(edge)

	cdef bint updateprob(self, uint64_t item, Prob prob, Prob beam):
		"""Update probability for item if better than current one.

		Add item if not seen before; return False if pruned."""
		cdef CFGItem itemx
		cdef uint64_t beamitem
		cdef ItemNo itemidx = self.itemindex[item]
		cdef bint newitem = itemidx == 0
		cdef bint updateitem = newitem
		if beam:
			itemx.dt = item
			beamitem = compactcellidx(
					itemx.st.start, itemx.st.end, self.lensent, 1)
			if prob > self.beambuckets[beamitem]:  # prob falls outside of beam
				return False
			elif prob + beam < self.beambuckets[beamitem]:  # shrink beam
				self.beambuckets[beamitem] = prob + beam
				updateitem = True
			elif newitem or prob < self.probs[itemidx]:  # prob falls within beam
				updateitem = True
		elif prob < self.probs[itemidx]:
			updateitem = True
		if newitem:
			itemidx = self.itemindex[item] = self.items.size()
			self.items.push_back(item)
			self.parseforest.resize(self.items.size())
			self.probs.push_back(prob)
		elif updateitem:
			self.probs[itemidx] = prob
		return True

	cdef ItemNo _left(self, ItemNo itemidx, Edge edge):
		cdef CFGItem item
		item.dt = self.items[itemidx]
		if edge.rule is NULL:
			return 0
		return self.itemindex[cellstruct(
				item.st.start, edge.pos.mid) + edge.rule.rhs1]

	cdef ItemNo _right(self, ItemNo itemidx, Edge edge):
		cdef CFGItem item
		item.dt = self.items[itemidx]
		if edge.rule is NULL or edge.rule.rhs2 == 0:
			return 0
		return self.itemindex[cellstruct(
				edge.pos.mid, item.st.end) + edge.rule.rhs2]

	cdef Label _label(self, uint64_t item):
		cdef CFGItem itemx
		itemx.dt = item
		return itemx.st.label

	cdef Label label(self, ItemNo itemidx):
		cdef CFGItem item
		item.dt = self.items[itemidx]
		return item.st.label

	cdef Prob _subtreeprob(self, uint64_t item):
		"""Get viterbi / inside probability of a subtree headed by `item`."""
		it = self.itemindex.find(item)
		if it == self.itemindex.end():
			return INFINITY
		return self.probs[dereference(it).second]

	cdef Prob subtreeprob(self, ItemNo itemidx):
		return self.probs[itemidx]

	cdef bint _hasitem(self, uint64_t item):
		"""Test if item is in chart."""
		return self.itemindex.find(item) != self.itemindex.end()

	def indices(self, ItemNo itemidx):
		cdef CFGItem item
		item.dt = self.items[itemidx]
		return list(range(item.st.start, item.st.end))

	def itemstr(self, ItemNo itemidx):
		cdef CFGItem item
		item.dt = self.items[itemidx]
		return '%s[%d:%d]' % (
				self.grammar.tolabel[item.st.label],
				item.st.start, item.st.end)

	def itemid(self, str label, indices, Whitelist whitelist=None):
		cdef Label labelid
		try:
			labelid = self.grammar.toid[label]
		except KeyError:
			return 0
		return self.itemid1(labelid, indices, whitelist)

	def itemid1(self, Label labelid, indices, Whitelist whitelist=None):
		cdef short left = min(indices)
		cdef short right = max(indices) + 1
		item = cellstruct(left, right) + labelid
		if whitelist is not None:
			return whitelist.cfg[compactcellidx(left, right, self.lensent, 1)
					].count(whitelist.mapping[labelid]) != 0 and item
		return self.itemindex.find(item) != self.itemindex.end() and item

	cdef SmallChartItem asSmallChartItem(self, ItemNo itemidx):
		cdef CFGItem item
		item.dt = self.items[itemidx]
		return CFGtoSmallChartItem(item.st.label, item.st.start, item.st.end)

	cdef FatChartItem asFatChartItem(self, ItemNo itemidx):
		cdef CFGItem item
		item.dt = self.items[itemidx]
		return CFGtoFatChartItem(item.st.label, item.st.start, item.st.end)

	cdef size_t asCFGspan(self, ItemNo itemidx):
		cdef CFGItem item
		item.dt = self.items[itemidx]
		assert 0 <= item.st.start < item.st.end <= self.lensent
		return compactcellidx(item.st.start, item.st.end, self.lensent, 1)


def parse(sent, Grammar grammar, tags=None, start=None, whitelist=None,
		Prob beam_beta=0.0, int beam_delta=50, itemsestimate=None,
		postagging=None):
	"""PCFG parsing using CKY.

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
	:param beam_beta: keep track of the best score in each cell and only allow
		items which are within a multiple of ``beam_beta`` of the best score.
		Should be a negative log probability. Pass ``0.0`` to disable.
	:param beam_delta: the maximum span length to which beam search is applied.
	:param itemsestimate: the number of chart items to pre-allocate.
	"""
	if grammar.maxfanout != 1:
		raise ValueError('Not a PCFG! fanout: %d' % grammar.maxfanout)
	if not grammar.logprob:
		raise ValueError('Expected grammar with log probabilities.')
	if whitelist is None and grammar.nonterminals < 20000:
		chart = DenseCFGChart(grammar, sent, start)
		return parse_grammarloop[DenseCFGChart](
				sent, <DenseCFGChart>chart, tags, beam_beta, beam_delta,
				postagging)
	chart = SparseCFGChart(grammar, sent, start, itemsestimate=itemsestimate)
	if whitelist is None:
		return parse_grammarloop[SparseCFGChart](
				sent, <SparseCFGChart>chart, tags, beam_beta, beam_delta,
				postagging)
	return parse_leftchildloop(
			sent, chart, tags, whitelist, beam_beta, beam_delta, postagging)


cdef parse_grammarloop(sent, CFGChart_fused chart, tags,
		Prob beam_beta, int beam_delta, postagging):
	"""A CKY parser modeled after Bodenstab's 'fast grammar loop'."""
	cdef:
		Grammar grammar = chart.grammar
		Agenda[Label, Prob] unaryagenda
		MidFilter midfilter
		ProbRule *rule
		short left, right, mid, span, lensent = len(sent)
		short narrowl, narrowr, widel, wider, minmid, maxmid
		Prob prevprob, prob
		Label lhs = 0
		uint32_t n
		uint64_t item, leftitem, rightitem, cell, blocked = 0
		ItemNo lastidx
		size_t nts = grammar.nonterminals
		bint usemask = grammar.mask.size() != 0
	# Create matrices to track minima and maxima for binary splits.
	n = (lensent + 1) * nts + 1
	midfilter.minleft.resize(n, -1)
	midfilter.maxright.resize(n, -1)
	midfilter.maxleft.resize(n, lensent + 1)
	midfilter.minright.resize(n, lensent + 1)

	if beam_beta:
		chart.beambuckets.resize(
				compactcellidx(lensent - 1, lensent, lensent, 1) + 1,
				INFINITY)
	# assign POS tags
	covered, msg = populatepos[CFGChart_fused](chart, sent, tags,
			unaryagenda, None, &blocked, &midfilter, NULL, postagging)
	if not covered:
		return chart, msg

	for span in range(2, lensent + 1):
		# constituents from left to right
		for left in range(lensent - span + 1):
			right = left + span
			if CFGChart_fused is DenseCFGChart:
				cell = cellidx(left, right, lensent, nts)
			elif CFGChart_fused is SparseCFGChart:
				cell = cellstruct(left, right)
			lastidx = chart.items.size()
			# apply all binary rules
			for lhs in range(1, grammar.nonterminals):
				n = 0
				rule = &(grammar.bylhs[lhs][n])
				item = lhs + cell
				prevprob = chart._subtreeprob(item)
				while rule.lhs == lhs:
					narrowr = midfilter.minright[left * nts + rule.rhs1]
					narrowl = midfilter.minleft[right * nts + rule.rhs2]
					if (rule.rhs2 == 0 or narrowr >= right or narrowl < narrowr
							or (usemask and TESTBIT(
								&(grammar.mask[0]), rule.no))):
						n += 1
						rule = &(grammar.bylhs[lhs][n])
						continue
					widel = midfilter.maxleft[right * nts + rule.rhs2]
					minmid = narrowr if narrowr > widel else widel
					wider = midfilter.maxright[left * nts + rule.rhs1]
					maxmid = wider if wider < narrowl else narrowl
					for mid in range(minmid, maxmid + 1):
						if CFGChart_fused is DenseCFGChart:
							leftitem = rule.rhs1 + cellidx(left, mid,
									lensent, nts)
							rightitem = rule.rhs2 + cellidx(mid, right,
									lensent, nts)
						elif CFGChart_fused is SparseCFGChart:
							leftitem = rule.rhs1 + cellstruct(left, mid)
							rightitem = rule.rhs2 + cellstruct(mid, right)
						prob = chart._subtreeprob(leftitem)
						if isinf(prob):
							continue
						prob += chart._subtreeprob(rightitem)
						if isfinite(prob):
							if chart.updateprob(item, prob + rule.prob,
									beam_beta if span <= beam_delta else 0.0):
								chart.addedge(item, mid, rule)
							else:
								blocked += 1
					n += 1
					rule = &(grammar.bylhs[lhs][n])

				if isinf(prevprob) and isfinite(chart._subtreeprob(item)):
					updatemidfilter(midfilter, left, right, lhs, nts)

			applyunaryrules[CFGChart_fused](chart, left, right, cell, lastidx,
					unaryagenda, &midfilter, &blocked, None)

	msg = '%s%s, blocked %s' % (
			'' if chart else 'no parse; ', chart.stats(), blocked)
	return chart, msg


cdef parse_leftchildloop(sent, SparseCFGChart chart, tags,
		Whitelist whitelist, Prob beam_beta, int beam_delta, postagging):
	"""A CKY parser that iterates over items in chart and compatible rules."""
	cdef:
		Grammar grammar = chart.grammar
		Agenda[Label, Prob] unaryagenda
		ProbRule *rule
		vector[size_t] cellindex  # compact cell idx => itemidx
		Prob leftprob, rightprob
		Label rhs1
		ItemNo leftitemidx, lastidx
		uint64_t item, leftitem, rightcell, cell, ccell
		uint64_t blocked = 0
		uint32_t n
		short left, right, mid, span, lensent = len(sent)
		CFGItem li
		bint usemask = grammar.mask.size() != 0
	cellindex.resize(compactcellidx(lensent - 1, lensent, lensent, 1) + 2, 0)
	if beam_beta:
		chart.beambuckets.resize(
				compactcellidx(lensent - 1, lensent, lensent, 1) + 1, INFINITY)
	# assign POS tags
	covered, msg = populatepos(chart, sent, tags, unaryagenda, whitelist,
			&blocked, NULL, &cellindex, postagging)
	if not covered:
		return chart, msg

	for span in range(2, lensent + 1):
		# constituents from left to right
		for left in range(lensent - span + 1):
			right = left + span
			cell = cellstruct(left, right)
			ccell = compactcellidx(left, right, lensent, 1)
			cellindex[ccell] = lastidx = chart.items.size()
			# apply binary rules; if whitelist is given, skip labels not in set
			for mid in range(left + 1, right):
				rightcell = cellstruct(mid, right)
				leftitemidx = cellindex[compactcellidx(left, mid, lensent, 1)]
				leftitem = chart.items[leftitemidx]
				li.dt = leftitem
				while li.st.end == mid:
					leftprob = (<Chart>chart).subtreeprob(leftitemidx)
					rhs1 = chart._label(leftitem)
					n = 0
					rule = &(grammar.lbinary[rhs1][n])
					while rule.rhs1 == rhs1 and n < grammar.numbinary:
						# This requires a hash table lookup of right item;
						# might be better if items in cell are together
						# in own datastructure
						rightprob = chart._subtreeprob(rightcell + rule.rhs2)
						item = cell + rule.lhs
						if isfinite(rightprob):
							if (usemask and TESTBIT(&(grammar.mask[0]), rule.no
									)) or (whitelist is not None
									and whitelist.mapping[rule.lhs]
									and not whitelist.cfg[ccell].count(
										whitelist.mapping[rule.lhs])):
								blocked += 1
							elif not chart.updateprob(
									item, leftprob + rightprob + rule.prob,
									beam_beta if span <= beam_delta else 0.0):
								blocked += 1
							else:
								chart.addedge(item, mid, rule)
						n += 1
						rule = &(grammar.lbinary[rhs1][n])
					leftitemidx += 1
					leftitem = chart.items[leftitemidx]
					li.dt = leftitem

			applyunaryrules(chart, left, right, cell, lastidx, unaryagenda,
					NULL, &blocked, whitelist)
			cellindex[ccell + 1] = chart.items.size()
	msg = '%s%s, blocked %s' % (
			'' if chart else 'no parse; ', chart.stats(), blocked)
	return chart, msg


cdef populatepos(CFGChart_fused chart, sent, tags,
		Agenda[Label, Prob]& unaryagenda, Whitelist whitelist,
		uint64_t *blocked, MidFilter *midfilter, vector[size_t] *cellindex,
		object postagging):
	"""Apply all possible lexical and unary rules on each lexical span.

	:param unaryagenda: expects an empty agenda; only passed around to reuse
		allocated memory.
	:returns: a tuple ``(success, msg)`` where ``success`` is True if a POS tag
		was found for every word in the sentence."""
	cdef:
		Grammar grammar = chart.grammar
		LexicalRule lexrule
		Label lhs
		size_t nts = grammar.nonterminals
		ItemNo lastidx
		uint64_t cell, ccell = 0
		uint32_t n
		short left, right, lensent = len(sent)
		Prob openclassfactor = 0.001
	for left, word in enumerate(sent):
		tag = tags[left] if tags and tags[left] else None
		# if we are given gold tags, make sure we only allow matching
		# tags - after removing addresses introduced by the DOP reduction
		# and other state splits.
		tagre = re.compile('%s($|[-@^/])' % re.escape(tag)) if tag else None
		right = left + 1
		if CFGChart_fused is DenseCFGChart:
			cell = cellidx(left, right, lensent, nts)
		elif CFGChart_fused is SparseCFGChart:
			cell = cellstruct(left, right)
		ccell = compactcellidx(left, right, lensent, 1)
		lastidx = chart.items.size()
		if cellindex is not NULL:
			cellindex[0][ccell] = lastidx
		recognized = False
		it = grammar.lexicalbyword.find(word.encode('utf8'))
		if it == grammar.lexicalbyword.end():
			it = grammar.lexicalbyword.find(word.lower().encode('utf8'))
		if it != grammar.lexicalbyword.end():
			if (postagging and tag is None
					and not word.startswith('_UNK')
					and postagging.method == 'unknownword'
					and postagging.closedclasswords
					and word not in postagging.closedclasswords):
				reserveprob = -pylog(1 - openclassfactor)
			else:
				reserveprob = 0
			for n in dereference(it).second:
				lexrule = grammar.lexical[n]
				if (whitelist is not None and whitelist.mapping[lexrule.lhs]
						and whitelist.cfg[ccell].count(
							whitelist.mapping[lexrule.lhs]) == 0):
					blocked[0] += 1
					continue
				lhs = lexrule.lhs
				if tag is None or tagre.match(grammar.tolabel[lhs]):
					chart.updateprob(cell + lhs,
							lexrule.prob + reserveprob, 0.0)
					chart.addedge(cell + lhs, right, NULL)
					recognized = True
					if midfilter is not NULL:
						updatemidfilter(midfilter[0], left, right, lhs, nts)
		if (postagging and tag is None
				and not word.startswith('_UNK')
				and postagging.method == 'unknownword'
				and postagging.closedclasswords
				and word not in postagging.closedclasswords):
			# add tags associated with signature, scale probabilities
			sig = postagging.unknownwordfun(word, left, postagging.lexicon)
			it = grammar.lexicalbyword.find(sig.encode('utf8'))
			if it != grammar.lexicalbyword.end():
				for n in dereference(it).second:
					lexrule = grammar.lexical[n]
					# avoid POS tag already considered above
					if isfinite(chart._subtreeprob(cell + lexrule.lhs)):
						continue
					if (whitelist is not None
							and whitelist.mapping[lexrule.lhs]
							and whitelist.cfg[ccell].count(
								whitelist.mapping[lexrule.lhs]) == 0):
						blocked[0] += 1
						continue
					lhs = lexrule.lhs
					chart.updateprob(cell + lhs,
							lexrule.prob - pylog(openclassfactor), 0.0)
					chart.addedge(cell + lhs, right, NULL)
					recognized = True
					if midfilter is not NULL:
						updatemidfilter(midfilter[0], left, right, lhs, nts)
		# NB: use gold tags if given, even if (word, tag) was not part of
		# training data or if it was pruned, modulo state splits etc.
		if not recognized and tag is not None:
			for lhs in grammar.lexicallhs:
				if tagre.match(grammar.tolabel[lhs]):
					chart.updateprob(cell + lhs, 0.0, 0.0)
					chart.addedge(cell + lhs, right, NULL)
					recognized = True
					if midfilter is not NULL:
						updatemidfilter(midfilter[0], left, right, lhs, nts)
		if not recognized:
			if tag is None and it == grammar.lexicalbyword.end():
				return False, ('no parse: no gold POS tag given '
						'and word %r not in lexicon' % word)
			elif tag is not None and tag not in grammar.toid:
				return False, ('no parse: gold POS tag given '
						'but tag %r not in grammar' % tag)
			return False, 'no parse: all tags for word %r blocked' % word

		# unary rules on the span of this POS tag
		applyunaryrules[CFGChart_fused](chart, left, right, cell, lastidx,
				unaryagenda, midfilter, blocked, whitelist)
	if cellindex is not NULL:
		cellindex[0][ccell + 1] = chart.items.size()
	return True, ''


cdef inline void applyunaryrules(
		CFGChart_fused chart, short left, short right, uint64_t cell,
		ItemNo lastidx, Agenda[Label, Prob]& unaryagenda, MidFilter *midfilter,
		uint64_t *blocked, Whitelist whitelist):
	"""Apply unary rules in a given cell."""
	cdef:
		Grammar grammar = chart.grammar
		Label lhs, rhs1
		Prob prob
		ProbRule *rule
		uint64_t item, leftitem
		uint64_t ccell = compactcellidx(left, right, chart.lensent, 1)
		size_t nts = grammar.nonterminals
		bint usemask = grammar.mask.size() != 0
		# pair[Label, Prob] unaryentry
		# vector[pair[Label, Prob]] unaryentries
	# collect possible rhs items for unaries
	for itemidx in range(lastidx, chart.items.size()):
		item = chart.items[itemidx]
		unaryagenda.setifbetter(
				chart._label(item), (<Chart>chart).subtreeprob(itemidx))
	# 	unaryentry.first = chart._label(item)
	# 	unaryentry.second = chart._subtreeprob(item)
	# 	unaryentries.push_back(unaryentry)
	# FIXME heapify; possibly more efficient; but there's some bug
	# unaryagenda.replace_entries(unaryentries)
	while not unaryagenda.empty():
		rhs1 = unaryagenda.pop().first
		leftitem = cell + rhs1
		# FIXME can vit.prob change while entry in agenda?
		# prob = rule.prob + entry.second
		prob = chart._subtreeprob(leftitem)
		# FIXME: chart.updateprob here
		# FIXME: maybe better to iterate over whitelist and check for
		# unary prod. Or: compute intersection before loop;
		for n in range(grammar.numunary):
			rule = &(grammar.unary[rhs1][n])
			lhs = rule.lhs
			if rule.rhs1 != rhs1:
				break
			elif (usemask and TESTBIT(&(grammar.mask[0]), rule.no)) or (
					whitelist is not None
					and whitelist.mapping[lhs]
					and whitelist.cfg[ccell].count(
						whitelist.mapping[lhs]) == 0):
				continue
			item = cell + lhs
			if rule.prob + prob < chart._subtreeprob(item):
				if chart.updateprob(item, rule.prob + prob, 0.0):
					unaryagenda.setifbetter(lhs, rule.prob + prob)
				else:
					blocked += 1
			chart.addedge(item, right, rule)
			if midfilter is not NULL:
				updatemidfilter(midfilter[0], left, right, lhs, nts)


cdef inline void updatemidfilter(
		MidFilter& midfilter, short left, short right, Label lhs,
		size_t nts):
	"""Update mid point filter arrays."""
	if left > midfilter.minleft[right * nts + lhs]:
		midfilter.minleft[right * nts + lhs] = left
	if left < midfilter.maxleft[right * nts + lhs]:
		midfilter.maxleft[right * nts + lhs] = left
	if right < midfilter.minright[left * nts + lhs]:
		midfilter.minright[left * nts + lhs] = right
	if right > midfilter.maxright[left * nts + lhs]:
		midfilter.maxright[left * nts + lhs] = right


def testsent(sent, grammar, expected=None):
	"""Parse sentence with grammar and print 10 best derivations."""
	from .kbest import lazykbest
	print('cfg parsing; sentence:', sent)
	chart, msg = parse(sent.split(), grammar)
	print(chart)
	assert chart, msg
	if expected is None:
		print('10 best parse trees:')
	else:
		print('10 best parse trees (%d expected):' % expected)
	derivations = lazykbest(chart, 10)
	for a, p in derivations:
		print(exp(-p), a)
	if expected is not None:
		assert len(derivations) == expected, (len(derivations), expected)


def test():
	from .containers import Grammar

	cfg = Grammar([
		((('A', 'A'), ((0, ), )), 1), ((('A', 'B'), ((0, ), )), 1),
		((('A', 'C'), ((0, ), )), 1), ((('A', 'D'), ((0, ), )), 1),
		((('B', 'A'), ((0, ), )), 1), ((('B', 'B'), ((0, ), )), 1),
		((('B', 'C'), ((0, ), )), 1), ((('B', 'D'), ((0, ), )), 1),
		# ((('B', 'C'), ((0, ), )), 1),
		((('C', 'A'), ((0, ), )), 1),
		((('C', 'B'), ((0, ), )), 1), ((('C', 'C'), ((0, ), )), 1),
		((('C', 'D'), ((0, ), )), 1), ((('D', 'A'), ((0, ), )), 1),
		((('D', 'B'), ((0, ), )), 1), ((('D', 'C'), ((0, ), )), 1),
		((('D', 'NP', 'VP'), ((0, 1), )), 2),
		((('S', 'A'), ((0, ), )), 1), ((('S', 'D'), ((0, ), )), 2),
		((('NP', 'Epsilon'), ('mary', )), 1),
		((('VP', 'Epsilon'), ('walks', )), 1)],
		start='S')
	print(cfg)
	testsent('mary walks', cfg, 10)

	rules = [
		((('NP', 'NP', 'PP'), ((0, 1), )), 1),
		((('PP', 'P', 'NP'), ((0, 1), )), 1),
		((('S', 'NP', 'VP'), ((0, 1), )), 1),
		((('VP', 'V', 'NP'), ((0, 1), )), 1),
		((('VP', 'VP', 'PP'), ((0, 1), )), 2),
		((('NP', 'Epsilon'), ('astronomers', )), 1),
		((('NP', 'Epsilon'), ('ears', )), 1),
		((('V', 'Epsilon'), ('saw', )), 1),
		((('NP', 'Epsilon'), ('saw', )), 1),
		((('NP', 'Epsilon'), ('stars', )), 1),
		((('NP', 'Epsilon'), ('telescopes', )), 1),
		((('P', 'Epsilon'), ('with', )), 1)]
	cfg2 = Grammar(rules, start='S')
	testsent('astronomers saw stars with telescopes', cfg2, 2)

__all__ = ['CFGChart', 'DenseCFGChart', 'SparseCFGChart', 'parse']
