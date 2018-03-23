"""Parser for string-rewriting Linear Context-Free Rewriting Systems.

Expects binarized, epsilon-free, monotone LCFRS grammars."""
from __future__ import print_function
import re
import logging
import numpy as np
from math import exp, log as pylog
cimport cython
from cython.operator cimport postincrement, dereference
from libc.math cimport HUGE_VAL as INFINITY
include "constants.pxi"

cdef SmallChartItem COMPONENT = SmallChartItem(0, 0)
cdef SmallChartItem NONE = SmallChartItem(0, 0)
cdef FatChartItem FATNONE = FatChartItem(0)
cdef FatChartItem FATCOMPONENT = FatChartItem(0)

cdef class LCFRSChart(Chart):
	"""A chart for LCFRS grammars. An item is a ChartItem object."""
	def __init__(self, Grammar grammar, list sent,
			start=None, logprob=True, viterbi=True,
			itemsestimate=None):
		self.grammar = grammar
		self.sent = sent
		self.lensent = len(sent)
		self.start = grammar.toid[grammar.start if start is None else start]
		self.logprob = logprob
		self.viterbi = viterbi

	cdef void addlexedge(self, ItemNo itemidx, short wordidx):
		"""Add lexical edge."""
		cdef Edge edge
		edge.rule = NULL
		edge.pos.mid = wordidx + 1
		self.parseforest[itemidx].push_back(edge)

	cdef void updateprob(self, ItemNo itemidx, Prob prob):
		if prob < self.probs[itemidx]:
			self.probs[itemidx] = prob

	cdef void addprob(self, ItemNo itemidx, Prob prob):
		self.probs[itemidx] += prob

	cdef Prob _subtreeprob(self, ItemNo itemidx):
		return self.probs[itemidx]

	cdef Prob subtreeprob(self, ItemNo itemidx):
		return self.probs[itemidx]

	cdef Label label(self, ItemNo itemidx):
		raise NotImplementedError


@cython.final
cdef class SmallLCFRSChart(LCFRSChart):
	"""For sentences that fit into a single machine word."""
	def __init__(self, Grammar grammar, list sent,
			start=None, logprob=True, viterbi=True,
			itemsestimate=None):
		cdef SmallChartItem tmp = SmallChartItem(0, 0)
		super(SmallLCFRSChart, self).__init__(
				grammar, sent, start, logprob, viterbi)
		if itemsestimate is not None:
			self.items.reserve(itemsestimate)
			# NB: self.itemindex does not support reserve
			self.parseforest.reserve(itemsestimate)
			self.probs.reserve(itemsestimate)
		# first item is a sentinel
		self.items.push_back(tmp)
		self.itemindex[tmp] = 0
		self.probs.push_back(INFINITY)

	cdef void addedge(self, ItemNo itemidx, ItemNo leftitemidx,
			SmallChartItem& left, ProbRule *rule):
		"""Add new edge."""
		cdef Edge edge
		edge.rule = rule
		edge.pos.lvec = 0UL
		edge.pos.lvec = left.vec
		self.parseforest[itemidx].push_back(edge)

	cdef ItemNo _left(self, ItemNo itemidx_unused, Edge edge):
		cdef SmallChartItem tmp
		if edge.rule is NULL:
			return 0
		tmp.label = edge.rule.rhs1
		tmp.vec = edge.pos.lvec
		return self.itemindex[tmp]

	cdef ItemNo _right(self, ItemNo itemidx, Edge edge):
		cdef SmallChartItem tmp
		if edge.rule is NULL or edge.rule.rhs2 == 0:
			return 0
		tmp.label = edge.rule.rhs2
		tmp.vec = self.items[itemidx].vec ^ edge.pos.lvec
		return self.itemindex[tmp]

	cdef SmallChartItem _root(self):
		return SmallChartItem(self.start, (1UL << self.lensent) - 1)

	def root(self):
		it = self.itemindex.find(self._root())
		if it == self.itemindex.end():
			return 0
		return dereference(it).second

	cdef Label label(self, ItemNo itemidx):
		return self.items[itemidx].label

	cdef Label _label(self, ItemNo itemidx):
		return self.items[itemidx].label

	cdef ItemNo getitemidx(self, uint64_t n):
		"""Get itemidx of n'th item."""
		return n

	def indices(self, ItemNo itemidx):
		cdef SmallChartItem item = self.items[itemidx]
		return [n for n in range(len(self.sent)) if testbit(item.vec, n)]

	def itemstr(self, ItemNo itemidx):
		return '%s[%s]' % (
				self.grammar.tolabel[self._label(itemidx)],
				bin(self.items[itemidx].vec)[2:].zfill(self.lensent)[::-1])

	def itemid(self, str label, indices, Whitelist whitelist=None):
		try:
			labelid = self.grammar.toid[label]
		except KeyError:
			return 0
		return self.itemid1(labelid, indices, whitelist)

	def itemid1(self, Label labelid, indices, Whitelist whitelist=None):
		cdef SmallChartItem tmp, tmp1
		vec = sum(1 << n for n in indices)
		tmp = SmallChartItem(labelid, vec)
		if whitelist is not None:
			tmp1.label = whitelist.mapping[labelid]
			tmp1.vec = tmp.vec
			return (whitelist.small[tmp1.label].count(tmp1)
					and self.itemindex[tmp])
		return self.itemindex[tmp]

	cdef SmallChartItem asSmallChartItem(self, ItemNo itemidx):
		return self.items[itemidx]

	cdef FatChartItem asFatChartItem(self, ItemNo itemidx):
		raise ValueError

	cdef size_t asCFGspan(self, ItemNo itemidx):
		cdef SmallChartItem sitem = self.items[itemidx]
		cdef int start = nextset(sitem.vec, 0)
		cdef int end = nextunset(sitem.vec, start)
		assert nextset(sitem.vec, end) == -1
		assert 0 <= start < end <= self.lensent
		return compactcellidx(start, end, self.lensent, 1)


@cython.final
cdef class FatLCFRSChart(LCFRSChart):
	"""LCFRS chart that supports longer sentences."""
	def __init__(self, Grammar grammar, list sent,
			start=None, logprob=True, viterbi=True,
			itemsestimate=None):
		super(FatLCFRSChart, self).__init__(
				grammar, sent, start, logprob, viterbi)
		cdef FatChartItem tmp = FatChartItem(0)
		if itemsestimate is not None:
			self.items.reserve(itemsestimate)
			# NB: self.itemindex does not support reserve
			self.parseforest.reserve(itemsestimate)
			self.probs.reserve(itemsestimate)
		self.items.push_back(tmp)  # sentinel
		self.itemindex[tmp] = 0  # sentinel
		self.probs.push_back(INFINITY)

	cdef void addedge(self, ItemNo itemidx, ItemNo leftitemidx,
			FatChartItem& left, ProbRule *rule):
		"""Add new edge and update viterbi probability."""
		cdef Edge edge
		edge.rule = rule
		edge.pos.lvec = 0UL
		edge.pos.lidx = leftitemidx
		self.parseforest[itemidx].push_back(edge)

	cdef ItemNo _left(self, ItemNo itemidx_unused, Edge edge):
		if edge.rule is NULL:
			return 0
		return edge.pos.lidx

	cdef ItemNo _right(self, ItemNo itemidx, Edge edge):
		cdef FatChartItem tmp
		cdef size_t n
		if edge.rule is NULL or edge.rule.rhs2 == 0:
			return 0
		tmp = self.items[edge.pos.lidx]
		tmp.label = edge.rule.rhs2
		for n in range(SLOTS):
			tmp.vec[n] ^= self.items[itemidx].vec[n]
		return self.itemindex[tmp]

	cdef FatChartItem _root(self):
		return CFGtoFatChartItem(self.start, 0, self.lensent)

	def root(self):
		cdef FatChartItem tmp = self._root()
		it = self.itemindex.find(tmp)
		if it == self.itemindex.end():
			return 0
		return dereference(it).second

	cdef Label label(self, ItemNo itemidx):
		return self._label(itemidx)  # somehow needed

	cdef Label _label(self, ItemNo itemidx):
		return self.items[itemidx].label

	cdef ItemNo getitemidx(self, uint64_t n):
		"""Get itemidx of n'th item."""
		return n

	def indices(self, ItemNo itemidx):
		cdef FatChartItem item = self.items[itemidx]
		return [n for n in range(len(self.sent)) if TESTBIT(item.vec, n)]

	def itemstr(self, ItemNo itemidx):
		cdef int n
		cdef str result = ''
		for n in range(SLOTS):
			result += bin(self.items[itemidx].vec[n])[2:].zfill(BITSIZE)[::-1]
		result = result[:self.lensent]
		return '%s[%s]' % (self.grammar.tolabel[self._label(itemidx)],
				result)

	def itemid(self, str label, indices, Whitelist whitelist=None):
		try:
			labelid = self.grammar.toid[label]
		except KeyError:
			return 0
		return self.itemid1(labelid, indices, whitelist)

	def itemid1(self, Label labelid, indices, Whitelist whitelist=None):
		cdef FatChartItem tmp, tmp1
		cdef uint64_t n
		tmp = FatChartItem(labelid)
		for n in indices:
			if n >= SLOTS * sizeof(unsigned long) * 8:
				return 0
			SETBIT(tmp.vec, n)
		if whitelist is not None:
			tmp1 = FatChartItem(labelid)
			tmp1.label = whitelist.mapping[labelid]
			for n in indices:
				if n >= SLOTS * sizeof(unsigned long) * 8:
					return 0
				SETBIT(tmp1.vec, n)
			return (whitelist.fat[tmp1.label].count(tmp1) != 0
					and self.itemindex[tmp])
		return self.itemindex[tmp]

	cdef SmallChartItem asSmallChartItem(self, ItemNo itemidx):
		raise ValueError

	cdef FatChartItem asFatChartItem(self, ItemNo itemidx):
		return self.items[itemidx]

	cdef size_t asCFGspan(self, ItemNo itemidx):
		cdef FatChartItem fitem = self.items[itemidx]
		cdef int start = anextset(fitem.vec, 0, SLOTS)
		cdef int end = anextunset(fitem.vec, start, SLOTS)
		assert anextset(fitem.vec, end, SLOTS) == -1
		assert 0 <= start < end <= self.lensent
		return compactcellidx(start, end, self.lensent, 1)


def parse(sent, Grammar grammar, tags=None, bint exhaustive=True,
		start=None, Whitelist whitelist=None, bint splitprune=False,
		bint markorigin=False, estimates=None,
		Prob beam_beta=0.0, int beam_delta=50, itemsestimate=None,
		postagging=None):
	"""Parse sentence and produce a chart.

	:param sent: A sequence of tokens that will be parsed.
	:param grammar: A ``Grammar`` object.
	:returns: a tuple (chart, msg); a ``Chart`` object and status message.
	:param tags: Optionally, a sequence of POS tags to use instead of
		attempting to apply all possible POS tags.
	:param exhaustive: don't stop at viterbi parse, return a full chart
	:param start: integer corresponding to the start symbol that complete
		derivations should be headed by; e.g., ``grammar.toid['ROOT']``.
		If not given, the default specified by ``grammar`` is used.
	:param whitelist: a whitelist of allowed ChartItems. Anything else is not
		added to the agenda.
	:param splitprune: coarse stage used a split-PCFG where discontinuous node
		appear as multiple CFG nodes. Every discontinuous node will result
		in multiple lookups into whitelist to see whether it should be
		allowed on the agenda.
	:param markorigin: in combination with splitprune, coarse labels include an
		integer to distinguish components; e.g., CFG nodes NP*0 and NP*1
		map to the discontinuous node NP_2
	:param estimates: use context-summary estimates (heuristics, figures of
		merit) to order agenda. should be a tuple with the kind of
		estimates ('SX' or 'SXlrgaps'), and the estimates themselves in a
		4-dimensional numpy matrix. If estimates are not consistent, it is
		no longer guaranteed that the optimal parse will be found.
		experimental.
	:param beam_beta: keep track of the best score in each cell and only allow
		items which are within a multiple of ``beam_beta`` of the best score.
		Should be a negative log probability. Pass ``0.0`` to disable.
	:param beam_delta: the maximum span length to which beam search is applied.
	:param itemsestimate: the number of chart items to pre-allocate.
	"""
	if <unsigned>len(sent) < sizeof(COMPONENT.vec) * 8:
		chart = SmallLCFRSChart(grammar, list(sent), start,
			itemsestimate=itemsestimate)
		return parse_main[SmallLCFRSChart, SmallChartItem](
				<SmallLCFRSChart>chart,
				<SmallChartItem>(<SmallLCFRSChart>chart)._root(),
				sent, grammar, tags, exhaustive, whitelist,
				splitprune, markorigin, estimates, beam_beta, beam_delta,
				postagging)
	chart = FatLCFRSChart(grammar, list(sent), start,
			itemsestimate=itemsestimate)
	return parse_main[FatLCFRSChart, FatChartItem](
			<FatLCFRSChart>chart, <FatChartItem>(<FatLCFRSChart>chart)._root(),
			sent, grammar, tags, exhaustive, whitelist,
			splitprune, markorigin, estimates, beam_beta, beam_delta,
			postagging)


cdef parse_main(LCFRSChart_fused chart, LCFRSItem_fused goal, sent,
		Grammar grammar, tags, bint exhaustive, Whitelist whitelist,
		bint splitprune, bint markorigin, estimates,
		Prob beam_beta, int beam_delta, postagging):
	cdef:
		Agenda[ItemNo, pair[Prob, Prob]] agenda  # prioritized items to explore
		pair[ItemNo, pair[Prob, Prob]] entry
		vector[ItemNo] sibvec
		ProbRule *rule
		LCFRSItem_fused item, sib, newitem, tmpitem
		double [:, :, :, :] outside = None  # outside estimates, if provided
		Prob siblingprob, score, prob, newprob
		short lensent = len(sent), estimatetype = 0
		int length = 1, left = 0, right = 0, gaps = 0
		ItemNo itemidx, sibidx
		size_t blocked = 0, maxA = 0, n
		bint usemask = grammar.mask.size() != 0
	# avoid generating code for spurious fused type combinations
	if ((LCFRSItem_fused is SmallChartItem
			and LCFRSChart_fused is FatLCFRSChart)
			or (LCFRSItem_fused is FatChartItem
			and LCFRSChart_fused is SmallLCFRSChart)):
		return
	if estimates is not None:
		estimatetypestr, outside = estimates
		estimatetype = {'SX': SX, 'SXlrgaps': SXlrgaps}[estimatetypestr]
	if LCFRSItem_fused is SmallChartItem:
		newitem = SmallChartItem(0, 0)
		tmpitem = SmallChartItem(0, 0)
	elif LCFRSItem_fused is FatChartItem:
		newitem = FatChartItem(0)
		tmpitem = FatChartItem(0)
	agenda.reserve(1024)

	# assign POS tags
	covered, msg = populatepos[LCFRSChart_fused, LCFRSItem_fused](
			grammar, agenda, chart, newitem,
			sent, tags, whitelist, estimates, postagging)
	if not covered:
		return chart, msg
	assert not agenda.empty()

	while not agenda.empty():  # main parsing loop
		entry = agenda.pop()
		itemidx = entry.first
		prob = entry.second.second
		item = chart.items[itemidx]
		# store viterbi probability; cannot do this when this item is added to
		# the agenda because that would give rise to duplicate edges.
		chart.updateprob(itemidx, prob)
		if item == goal:
			if not exhaustive:
				break
		else:
			# unary
			if LCFRSItem_fused is SmallChartItem:
				length = bitcount(item.vec)
				newitem.vec = item.vec
			elif LCFRSItem_fused is FatChartItem:
				length = abitcount(item.vec, SLOTS)
				memcpy(<void *>newitem.vec, <void *>item.vec,
						SLOTS * sizeof(uint64_t))
			if estimates is not None:
				if LCFRSItem_fused is SmallChartItem:
					left = nextset(item.vec, 0)
					gaps = bitlength(item.vec) - length - left
					right = lensent - length - left - gaps
				elif LCFRSItem_fused is FatChartItem:
					left = anextset(item.vec, 0, SLOTS)
					gaps = abitlength(item.vec, SLOTS) - length - left
					right = lensent - length - left - gaps
			for n in range(grammar.numunary):
				rule = &(grammar.unary[item.label][n])
				if rule.rhs1 != item.label:
					break
				elif usemask and TESTBIT(&(grammar.mask[0]), rule.no):
					continue
				score = newprob = prob + rule.prob
				if estimatetype == SX:
					score += outside[rule.lhs, left, right, 0]
					if score > MAX_LOGPROB:
						continue
				elif estimatetype == SXlrgaps:
					score += outside[
							rule.lhs, length, left + right, gaps]
					if score > MAX_LOGPROB:
						continue
				else:
					# add length of span to score so that all items of length n
					# have a strictly lower score than items with length n + 1.
					score += length * MAX_LOGPROB
				newitem.label = rule.lhs
				if process_edge[LCFRSItem_fused, LCFRSChart_fused](
						newitem, newprob, score, rule, itemidx, item,
						agenda, chart, estimatetype, whitelist,
						splitprune and grammar.fanout[rule.lhs] != 1,
						markorigin, 0.0):
					if LCFRSItem_fused is SmallChartItem:
						newitem.vec = item.vec
					elif LCFRSItem_fused is FatChartItem:
						memcpy(<void *>newitem.vec, <void *>item.vec,
								SLOTS * sizeof(uint64_t))
				else:
					blocked += 1
			# binary production, item from agenda is on the right
			for n in range(grammar.numbinary):
				rule = &(grammar.rbinary[item.label][n])
				if rule.rhs2 != item.label:
					break
				# elif chart.probs[rule.rhs1] is None:
				# 	continue
				elif usemask and TESTBIT(&(grammar.mask[0]), rule.no):
					continue
				tmpitem.label = rule.rhs1
				itemidxit = chart.itemindex.lower_bound(tmpitem)
				sibvec.clear()
				while (itemidxit != chart.itemindex.end()
						and dereference(itemidxit).first.label == rule.rhs1):
					sib = dereference(itemidxit).first
					sibidx = dereference(itemidxit).second
					sibvec.push_back(sibidx)
					postincrement(itemidxit)
				for sibidx in sibvec:
					sib = chart.items[sibidx]
					postincrement(itemidxit)
					if concat[LCFRSItem_fused](rule, &sib, &item):
						newitem.label = rule.lhs
						combine_item[LCFRSItem_fused](&newitem, &sib, &item)
						siblingprob = chart.probs[sibidx]
						if siblingprob == INFINITY:
							continue
						score = newprob = prob + siblingprob + rule.prob
						if LCFRSItem_fused is SmallChartItem:
							length = bitcount(newitem.vec)
						elif LCFRSItem_fused is FatChartItem:
							length = abitcount(newitem.vec, SLOTS)
						if estimatetype == SX or estimatetype == SXlrgaps:
							if LCFRSItem_fused is SmallChartItem:
								left = nextset(newitem.vec, 0)
							elif LCFRSItem_fused is FatChartItem:
								left = anextset(newitem.vec, 0, SLOTS)
						if estimatetype == SX:
							right = lensent - length - left
							score += outside[rule.lhs, left, right, 0]
							if score > MAX_LOGPROB:
								continue
						elif estimatetype == SXlrgaps:
							if LCFRSItem_fused is SmallChartItem:
								gaps = bitlength(newitem.vec) - length - left
							elif LCFRSItem_fused is FatChartItem:
								gaps = abitlength(newitem.vec, SLOTS
										) - length - left
							right = lensent - length - left - gaps
							score += outside[
									rule.lhs, length, left + right, gaps]
							if score > MAX_LOGPROB:
								continue
						else:
							score += length * MAX_LOGPROB
						if process_edge[LCFRSItem_fused, LCFRSChart_fused](
								newitem, newprob, score, rule, sibidx, sib,
								agenda, chart, estimatetype, whitelist,
								splitprune and grammar.fanout[rule.lhs] != 1,
								markorigin,
								beam_beta if length <= beam_delta else 0.0):
							pass
						else:
							blocked += 1
			# binary production, item from agenda is on the left
			for n in range(grammar.numbinary):
				rule = &(grammar.lbinary[item.label][n])
				if rule.rhs1 != item.label:
					break
				# elif chart.probs[rule.rhs2] is None:
				# 	continue
				elif usemask and TESTBIT(&(grammar.mask[0]), rule.no):
					continue
				tmpitem.label = rule.rhs2
				itemidxit = chart.itemindex.lower_bound(tmpitem)
				sibvec.clear()
				while (itemidxit != chart.itemindex.end()
						and dereference(itemidxit).first.label == rule.rhs2):
					sib = dereference(itemidxit).first
					sibidx = dereference(itemidxit).second
					sibvec.push_back(sibidx)
					postincrement(itemidxit)
				for sibidx in sibvec:
					sib = chart.items[sibidx]
					if concat[LCFRSItem_fused](rule, &item, &sib):
						newitem.label = rule.lhs
						combine_item[LCFRSItem_fused](&newitem, &item, &sib)
						siblingprob = chart.probs[sibidx]
						if siblingprob == INFINITY:
							continue
						score = newprob = prob + siblingprob + rule.prob
						if LCFRSItem_fused is SmallChartItem:
							length = bitcount(newitem.vec)
						elif LCFRSItem_fused is FatChartItem:
							length = abitcount(newitem.vec, SLOTS)
						if estimatetype == SX or estimatetype == SXlrgaps:
							if LCFRSItem_fused is SmallChartItem:
								left = nextset(newitem.vec, 0)
							elif LCFRSItem_fused is FatChartItem:
								left = anextset(newitem.vec, 0, SLOTS)
						if estimatetype == SX:
							right = lensent - length - left
							score += outside[rule.lhs, left, right, 0]
							if score > MAX_LOGPROB:
								continue
						elif estimatetype == SXlrgaps:
							if LCFRSItem_fused is SmallChartItem:
								gaps = bitlength(newitem.vec) - length - left
							elif LCFRSItem_fused is FatChartItem:
								gaps = abitlength(newitem.vec, SLOTS
										) - length - left
							right = lensent - length - left - gaps
							score += outside[rule.lhs, length,
									left + right, gaps]
							if score > MAX_LOGPROB:
								continue
						else:
							score += length * MAX_LOGPROB
						if process_edge[LCFRSItem_fused, LCFRSChart_fused](
								newitem, newprob, score, rule, itemidx, item,
								agenda, chart, estimatetype, whitelist,
								splitprune and grammar.fanout[rule.lhs] != 1,
								markorigin,
								beam_beta if length <= beam_delta else 0.0):
							pass
						else:
							blocked += 1
		if agenda.size() > maxA:
			maxA = agenda.size()
	msg = ('%s, blocked %d, agenda max %d, now %d' % (
			chart.stats(), blocked, maxA, agenda.size()))
	if not chart:
		return chart, 'no parse; ' + msg
	return chart, msg


cdef inline bint process_edge(LCFRSItem_fused newitem,
		Prob prob, Prob score, ProbRule *rule,
		ItemNo leftitemidx, LCFRSItem_fused& left,
		Agenda[ItemNo, pair[Prob, Prob]]& agenda, LCFRSChart_fused chart,
		int estimatetype, Whitelist whitelist, bint splitprune,
		bint markorigin, Prob beam):
	"""Decide what to do with a newly derived edge.

	:returns: ``True`` when edge is accepted in the chart, ``False`` when
		blocked. When ``False``, ``newitem`` may be reused."""
	cdef Label label
	cdef ItemNo itemidx
	cdef bint inagenda, inchart
	cdef pair[Prob, Prob] scoreprob
	cdef Prob curprob
	# avoid generating code for spurious fused type combinations
	if ((LCFRSItem_fused is SmallChartItem
			and LCFRSChart_fused is FatLCFRSChart)
			or (LCFRSItem_fused is FatChartItem
			and LCFRSChart_fused is SmallLCFRSChart)):
		return False
	itemidxit = chart.itemindex.find(newitem)
	if itemidxit == chart.itemindex.end():
		cuprob = INFINITY
		inagenda = inchart = False
		itemidx = curprob = 0
	else:
		itemidx = dereference(itemidxit).second
		curprob = chart.subtreeprob(itemidx)
		inagenda = agenda.member(itemidx)
		inchart = chart.parseforest[itemidx].size() != 0
	scoreprob.first = score
	scoreprob.second = prob
	if not inagenda and not inchart:
		# check if we need to prune this item
		if whitelist is not None and not checkwhitelist(
				newitem, whitelist, splitprune, markorigin):
			return False
		elif beam:
			label = newitem.label
			newitem.label = 0
			it = chart.beambuckets.find(newitem)
			if (it == chart.beambuckets.end()
					or prob + beam < dereference(it).second):
				chart.beambuckets[newitem] = prob + beam
			elif prob > dereference(it).second:
				return False
			newitem.label = label
		# haven't seen this item before, won't prune, add to agenda
		itemidx = chart.itemindex[newitem] = chart.items.size()
		chart.items.push_back(newitem)
		chart.parseforest.resize(chart.items.size())
		chart.probs.push_back(INFINITY)
		agenda.setitem(itemidx, scoreprob)
	# in agenda (maybe in chart)
	elif inagenda:
		# lower score? => decrease-key in agenda
		agenda.setifbetter(itemidx, scoreprob)
	# not in agenda => must be in chart
	elif not inagenda and prob < curprob:
		# re-add to agenda because we found a better score.
		agenda.setitem(itemidx, scoreprob)
		if estimatetype != SXlrgaps:
			# This should only happen because of an inconsistent or
			# non-monotonic estimate.
			logging.warning('WARN: re-adding item to agenda already in chart:'
					' %s', chart.itemstr(itemidx))
	# store this edge, regardless of whether the item was new (unary chains)
	chart.addedge(itemidx, leftitemidx, left, rule)
	return True


cdef populatepos(Grammar grammar,
		Agenda[ItemNo, pair[Prob, Prob]]& agenda, LCFRSChart_fused chart,
		LCFRSItem_fused item, sent, tags, Whitelist whitelist, estimates,
		postagging):
	"""Apply all possible lexical and unary rules on each lexical span.

	:returns: a tuple ``(success, msg)`` where ``success`` is True if a POS tag
		was found for every word in the sentence."""
	cdef:
		LexicalRule lexrule
		LCFRSItem_fused newitem
		double [:, :, :, :] outside = None  # outside estimates, if provided
		Prob score
		short wordidx, lensent = len(sent), estimatetype = 0
		int length = 1, left = 0, right = 0, gaps = 0
		Label lhs
		bint recognized
		Prob openclassfactor = 0.001
	if estimates is not None:
		estimatetypestr, outside = estimates
		estimatetype = {'SX': SX, 'SXlrgaps': SXlrgaps}[estimatetypestr]
	# newitem will be recycled until it is added to the chart
	if LCFRSItem_fused is SmallChartItem:
		newitem = SmallChartItem(0, 0)
	elif LCFRSItem_fused is FatChartItem:
		newitem = FatChartItem(0)

	for wordidx, word in enumerate(sent):  # add preterminals to chart
		recognized = False
		if LCFRSItem_fused is SmallChartItem:
			item = SmallChartItem(0, wordidx)
		elif LCFRSItem_fused is FatChartItem:
			item = FatChartItem(0)
			item.vec[0] = wordidx
		tag = tags[wordidx] if tags and tags[wordidx] else None
		# if we are given gold tags, make sure we only allow matching
		# tags - after removing addresses introduced by the DOP reduction
		# and other state splits.
		tagre = re.compile('%s($|[-@^/])' % re.escape(tag)) if tag else None
		if estimates is not None:
			left = wordidx
			gaps = 0
			right = lensent - 1 - wordidx
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
				if not tag or tagre.match(grammar.tolabel[lexrule.lhs]):
					score = lexrule.prob
					if estimatetype == SX:
						score += outside[lexrule.lhs, left, right, 0]
						if score > MAX_LOGPROB:
							continue
					elif estimatetype == SXlrgaps:
						score += outside[
								lexrule.lhs, length, left + right, gaps]
						if score > MAX_LOGPROB:
							continue
					# NB: do NOT add length of span to score, so that the
					# scores of POS tags are all strictly smaller than any
					# unaries on them.
					newitem.label = lexrule.lhs
					if LCFRSItem_fused is SmallChartItem:
						newitem.vec = 1UL << wordidx
					elif LCFRSItem_fused is FatChartItem:
						memset(<void *>newitem.vec, 0, SLOTS * sizeof(uint64_t))
						SETBIT(newitem.vec, wordidx)
					if process_lexedge[LCFRSItem_fused, LCFRSChart_fused](
							newitem, lexrule.prob + reserveprob,
							score + reserveprob, wordidx, agenda, chart,
							whitelist):
						if LCFRSItem_fused is SmallChartItem:
							newitem = SmallChartItem(0, 0)
						elif LCFRSItem_fused is FatChartItem:
							newitem = FatChartItem(0)
						recognized = True
		if (postagging and tag is None
				and not word.startswith('_UNK')
				and postagging.method == 'unknownword'
				and postagging.closedclasswords
				and word not in postagging.closedclasswords):
			# add tags associated with signature, scale probabilities
			sig = postagging.unknownwordfun(word, wordidx, postagging.lexicon)
			it = grammar.lexicalbyword.find(sig.encode('utf8'))
			if it != grammar.lexicalbyword.end():
				for n in dereference(it).second:
					lexrule = grammar.lexical[n]
					newitem.label = lexrule.lhs
					if LCFRSItem_fused is SmallChartItem:
						newitem.vec = 1UL << wordidx
					elif LCFRSItem_fused is FatChartItem:
						memset(<void *>newitem.vec, 0, SLOTS * sizeof(uint64_t))
						SETBIT(newitem.vec, wordidx)
					score = lexrule.prob - pylog(openclassfactor)
					# process_lexedge checks that tag is not already in agenda
					if process_lexedge[LCFRSItem_fused, LCFRSChart_fused](
							newitem, lexrule.prob - pylog(openclassfactor),
							score, wordidx, agenda, chart, whitelist):
						if LCFRSItem_fused is SmallChartItem:
							newitem = SmallChartItem(0, 0)
						elif LCFRSItem_fused is FatChartItem:
							newitem = FatChartItem(0)
						recognized = True
		# NB: use gold tags if given, even if (word, tag) was not part of
		# training data, modulo state splits etc.
		if not recognized and tag is not None:
			for x in grammar.lexicalbylhs:
				lhs = x.first
				if tagre.match(grammar.tolabel[lhs]) is not None:
					score = 0.0
					if estimatetype == SX:
						score += outside[lhs, left, right, 0]
						if score > MAX_LOGPROB:
							continue
					elif estimatetype == SXlrgaps:
						score += outside[lhs, length, left + right, gaps]
						if score > MAX_LOGPROB:
							continue
					newitem.label = lhs
					if LCFRSItem_fused is SmallChartItem:
						newitem.vec = 1UL << wordidx
					elif LCFRSItem_fused is FatChartItem:
						memset(
								<void *>newitem.vec, 0,
								SLOTS * sizeof(uint64_t))
						SETBIT(newitem.vec, wordidx)
					# prevent pruning of provided tags in whitelist
					if process_lexedge[LCFRSItem_fused, LCFRSChart_fused](
							newitem, 0.0, score, wordidx, agenda, chart, None):
						if LCFRSItem_fused is SmallChartItem:
							newitem = SmallChartItem(0, 0)
						elif LCFRSItem_fused is FatChartItem:
							newitem = FatChartItem(0)
						recognized = True
					else:
						raise ValueError('tag %r is blocked.' % tag)
		if not recognized:
			if tag is None and it == grammar.lexicalbyword.end():
				return False, ('no parse: no gold POS tag given '
						'and word %r not in lexicon' % word)
			elif tag is not None and tag not in grammar.toid:
				return False, ('no parse: gold POS tag given '
						'but tag %r not in grammar' % tag)
			return False, 'no parse: all tags for word %r blocked' % word
	return True, ''


cdef inline int process_lexedge(LCFRSItem_fused newitem,
		Prob prob, Prob score, short wordidx,
		Agenda[ItemNo, pair[Prob, Prob]]& agenda, LCFRSChart_fused chart,
		Whitelist whitelist) except -1:
	"""Decide whether to accept a lexical edge ``(POS, word)``.

	:returns: ``True`` when edge is accepted in the chart, ``False`` when
		blocked. When ``False``, ``newitem`` may be reused."""
	cdef bint inagenda, inchart
	cdef ItemNo itemidx
	cdef pair[Prob, Prob] scoreprob
	# avoid generating code for spurious fused type combinations
	if LCFRSItem_fused is SmallChartItem and LCFRSChart_fused is FatLCFRSChart:
		return -1
	elif (LCFRSItem_fused is FatChartItem
			and LCFRSChart_fused is SmallLCFRSChart):
		return -1
	if chart.itemindex.find(newitem) == chart.itemindex.end():
		itemidx = chart.itemindex[newitem] = chart.itemindex.size()
		chart.items.push_back(newitem)
		chart.parseforest.resize(chart.itemindex.size())
		# chart.probs.resize(chart.itemindex.size())
		chart.probs.push_back(INFINITY)
		inagenda = inchart = False
	else:
		itemidx = chart.itemindex[newitem]
		inchart = chart.parseforest[itemidx].size() != 0
		inagenda = agenda.member(itemidx)
	if inagenda:
		return False
		# raise ValueError('lexical edge already in agenda: %s' %
		# 		chart.itemstr(itemidx))
	elif inchart:
		raise ValueError('lexical edge already in chart: %s' %
				chart.itemstr(itemidx))
	# check if we need to prune this item
	elif whitelist is not None and not checkwhitelist(
			newitem, whitelist, False, False):
		return False
	# haven't seen this item before, won't prune
	scoreprob.first = score
	scoreprob.second = prob
	agenda.setitem(itemidx, scoreprob)
	# assert not agenda.empty(), agenda.size()
	chart.addlexedge(itemidx, wordidx)
	return True


cdef inline bint checkwhitelist(LCFRSItem_fused newitem, Whitelist whitelist,
		bint splitprune, bint markorigin):
	"""Return False if item is not on whitelist."""
	cdef uint32_t n, cnt
	cdef Label label
	cdef int a, b
	if whitelist is None:
		return True
	elif splitprune:  # disc. item to be treated as several split items?
		b = 0
		if markorigin:
			cnt = 0
			if whitelist.splitmapping[newitem.label].size() == 0:
				return True
		else:
			if whitelist.mapping[newitem.label] != 0:
				return True
			if LCFRSItem_fused is SmallChartItem:
				COMPONENT.label = whitelist.splitmapping[
						newitem.label][0]
			elif LCFRSItem_fused is FatChartItem:
				FATCOMPONENT.label = whitelist.splitmapping[
						newitem.label][0]
		if LCFRSItem_fused is SmallChartItem:
			a = nextset(newitem.vec, b)
			while a != -1:
				b = nextunset(newitem.vec, a)
				# given a=3, b=6, make bitvector: 1000000 - 1000 = 111000
				COMPONENT.vec = (1UL << b) - (1UL << a)
				if markorigin:
					COMPONENT.label = whitelist.splitmapping[
							newitem.label][cnt]
					cnt += 1
				if whitelist.small[COMPONENT.label].count(COMPONENT) == 0:
					return False
				a = nextset(newitem.vec, b)
		elif LCFRSItem_fused is FatChartItem:
			a = anextset(newitem.vec, b, SLOTS)
			while a != -1:
				b = anextunset(newitem.vec, a, SLOTS)
				# given a=3, b=6, make bitvector: 1000000 - 1000 = 111000
				memset(<void *>FATCOMPONENT.vec, 0, SLOTS * sizeof(uint64_t))
				for n in range(a, b):
					SETBIT(FATCOMPONENT.vec, n)
				if markorigin:
					FATCOMPONENT.label = whitelist.splitmapping[
							newitem.label][cnt]
					cnt += 1
				if whitelist.fat[FATCOMPONENT.label].count(
						FATCOMPONENT) == 0:
					return False
				a = anextset(newitem.vec, b, SLOTS)
	elif whitelist.mapping[newitem.label] != 0:
		label = newitem.label
		newitem.label = whitelist.mapping[label]
		if LCFRSItem_fused is SmallChartItem:
			if whitelist.small[newitem.label].count(newitem) == 0:
				return False
		elif LCFRSItem_fused is FatChartItem:
			if whitelist.fat[newitem.label].count(newitem) == 0:
				return False
		newitem.label = label
	return True


cdef inline void combine_item(LCFRSItem_fused *newitem,
		LCFRSItem_fused *left, LCFRSItem_fused *right):
	if LCFRSItem_fused is SmallChartItem:
		newitem[0].vec = left[0].vec ^ right[0].vec
	elif LCFRSItem_fused is FatChartItem:
		setunion(newitem[0].vec, left[0].vec, right[0].vec, SLOTS)


cdef inline bint concat(ProbRule *rule,
		LCFRSItem_fused *left, LCFRSItem_fused *right):
	"""Test whether two bitvectors combine according to a given rule.

	Ranges should be non-overlapping, continuous when they are concatenated,
	and adhere to the ordering in the yield function. The yield function
	indicates for each span whether it should come from the left or right
	non-terminal (0 meaning left and 1 right), and whether it is contiguous
	with the previous span.

	>>> lvec = 0b0011; rvec = 0b1000
	>>> concat(((0, ), (1, )), lvec, rvec)
	True		# discontinuous, non-overlapping, linearly ordered.
	>>> concat(((0, 1), ), lvec, rvec)
	False		# lvec and rvec are not contiguous
	>>> concat(((1, ), (0, )), lvec, rvec)
	False		# rvec's span should come after lvec's span

	The actual yield functions are encoded in a binary format
	(cf. containers.pyx)::
		((0, 1, 0), (1, 0)) ==> args=0b10010; lengths=0b00101
	NB: note reversal due to the way binary numbers are represented
	the least significant bit (rightmost) corresponds to the lowest
	index in the sentence / constituent (leftmost)."""
	cdef uint64_t lvec, rvec, mask
	cdef uint64_t *alvec
	cdef uint64_t *arvec
	cdef int lpos, rpos, n
	if LCFRSItem_fused is SmallChartItem:
		lvec = left[0].vec
		rvec = right[0].vec
		if lvec & rvec:
			return False
		mask = rvec if testbit(rule.args, 0) else lvec
		for n in range(bitlength(rule.lengths)):
			if testbit(rule.args, n):  # component from right vector
				if rvec & mask == 0:
					return False  # check for expected component
				rvec |= rvec - 1  # trailing 0 bits => 1 bits
				mask = rvec & (~rvec - 1)  # mask of 1 bits up to first 0 bit
			else:  # component from left vector
				if lvec & mask == 0:
					return False  # check for expected component
				lvec |= lvec - 1  # trailing 0 bits => 1 bits
				mask = lvec & (~lvec - 1)  # mask of 1 bits up to first 0 bit
			# zero out component
			lvec &= ~mask
			rvec &= ~mask
			if testbit(rule.lengths, n):  # a gap
				# check that there is a gap in both vectors
				if (lvec ^ rvec) & (mask + 1):
					return False
				# increase mask to cover gap
				# get minimum of trailing zero bits of lvec & rvec
				mask = (~lvec & (lvec - 1)) & (~rvec & (rvec - 1))
			mask += 1  # e.g., 00111 => 01000
		# success if we've reached the end of both left and right vector
		return lvec == rvec == 0
	elif LCFRSItem_fused is FatChartItem:
		alvec = left[0].vec
		arvec = right[0].vec
		for n in range(SLOTS):
			if alvec[n] & arvec[n]:
				return False
		lpos = anextset(alvec, 0, SLOTS)
		rpos = anextset(arvec, 0, SLOTS)
		# this algorithm was adapted from rparse, FastYFComposer.
		for n in range(bitlength(rule.lengths)):
			if testbit(rule.args, n):
				# check if there are any bits left, and
				# if any bits on the right should have gone before
				# ones on this side
				if rpos == -1 or (lpos != -1 and lpos <= rpos):
					return False
				# jump to next gap
				rpos = anextunset(arvec, rpos, SLOTS)
				if lpos != -1 and lpos < rpos:
					return False
				# there should be a gap if and only if
				# this is the last element of this argument
				if testbit(rule.lengths, n):
					if TESTBIT(alvec, rpos):
						return False
				elif not TESTBIT(alvec, rpos):
					return False
				# jump to next argument
				rpos = anextset(arvec, rpos, SLOTS)
			else:  # if bit == 0:
				# vice versa to the above
				if lpos == -1 or (rpos != -1 and rpos <= lpos):
					return False
				lpos = anextunset(alvec, lpos, SLOTS)
				if rpos != -1 and rpos < lpos:
					return False
				if testbit(rule.lengths, n):
					if TESTBIT(arvec, lpos):
						return False
				elif not TESTBIT(arvec, lpos):
					return False
				lpos = anextset(alvec, lpos, SLOTS)
		return lpos == rpos == -1


def testsent(sent, grammar):
	"""Parse sentence with grammar and print 10 best derivations."""
	from math import exp
	from .kbest import lazykbest
	print('len', len(sent.split()), 'sentence:', sent)
	sent = sent.split()
	chart, msg = parse(sent, grammar)
	if len(sent) < 10:
		print(chart)
	if chart:
		print('10 best parse trees (1 expected):')
		derivations = lazykbest(chart, 10)
		for a, p in derivations:
			print(exp(-p), a)
		print()
		assert len(derivations) == 1
		return True
	print(msg)
	return False


def test():
	grammar = Grammar([
		((('S', 'VP_2', 'VMFIN'), ((0, 1, 0), )), 1),
		((('VP_2', 'VP_2', 'VAINF'), ((0, ), (0, 1))), 0.5),
		((('VP_2', 'PROAV', 'VVPP'), ((0, ), (1, ))), 0.5),
		((('PROAV', 'Epsilon'), ('Darueber', )), 1),
		((('VAINF', 'Epsilon'), ('werden', )), 1),
		((('VMFIN', 'Epsilon'), ('muss', )), 1),
		((('VVPP', 'Epsilon'), ('nachgedacht', )), 1)], start='S')
	print(grammar)
	assert testsent('Darueber muss nachgedacht werden', grammar)
	assert testsent('Darueber muss nachgedacht werden werden werden', grammar)
	print('ungrammatical sentence (\'no parse\' expected):')
	assert not testsent('muss Darueber nachgedacht werden', grammar)
	assert testsent('Darueber muss nachgedacht ' + ' '.join(32 * ['werden']),
			grammar)
	assert testsent('Darueber muss nachgedacht ' + ' '.join(64 * ['werden']),
			grammar)

__all__ = ['LCFRSChart', 'SmallLCFRSChart', 'FatLCFRSChart', 'parse']
