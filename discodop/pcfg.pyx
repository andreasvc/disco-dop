"""CKY parser for Probabilistic Context-Free Grammar (PCFG)."""
from __future__ import print_function
from os import unlink
import re
import subprocess
from math import exp, log as pylog
from itertools import count
from collections import defaultdict
import numpy as np
from discodop.tree import Tree
from discodop.plcfrs import DoubleAgenda
from discodop.treebank import TERMINALSRE

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
				self.lensent, self.grammar.nonterminals
				) + self.grammar.toid[self.grammar.start]

	def label(self, item):
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
				self.grammar.tolabel[lhs].decode('ascii'), start, end)


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
		# store parse forest in list instead of dict
		entries = cellidx(self.lensent - 1, self.lensent, self.lensent,
				grammar.nonterminals) + grammar.nonterminals
		self.parseforest = [None] * entries
		self.itemsinorder = []

	def __dealloc__(self):
		if self.probs is not NULL:
			free(self.probs)

	cdef void addedge(self, uint32_t lhs, Idx start, Idx end, Idx mid,
			Rule *rule):
		"""Add new edge to parse forest."""
		cdef Edges edges
		cdef Edge *edge
		cdef size_t block, item = cellidx(
				start, end, self.lensent, self.grammar.nonterminals) + lhs
		if self.parseforest[item] is None:
			edges = Edges()
			self.parseforest[item] = [edges]
			self.itemsinorder.append(item)
		else:
			block = len(<list>self.parseforest[item]) - 1
			edges = self.parseforest[item][block]
			if edges.len == EDGES_SIZE:
				edges = Edges()
				self.parseforest[item].append(edges)
		edge = &(edges.data[edges.len])
		edge.rule = rule
		edge.pos.mid = mid
		edges.len += 1

	cdef void updateprob(self, uint32_t lhs, Idx start, Idx end, double prob):
		"""Update probability for item if better than current one."""
		cdef size_t idx = compactcellidx(
				start, end, self.lensent, self.grammar.nonterminals) + lhs
		if prob < self.probs[idx]:
			self.probs[idx] = prob

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

	cdef getitems(self):
		return [n for n, a in enumerate(self.parseforest) if a is not None]

	cdef list getedges(self, item):
		"""Get edges for item."""
		return self.parseforest[item] if item is not None else []

	cdef bint hasitem(self, size_t item):
		"""Test if item is in chart."""
		return self.parseforest[item] is not None

	def __nonzero__(self):
		"""Return true when the root item is in the chart.

		i.e., test whether sentence has been parsed successfully."""
		return self.parseforest[self.root()] is not None


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
		self.itemsinorder = []

	cdef void addedge(self, uint32_t lhs, Idx start, Idx end, Idx mid,
			Rule *rule):
		"""Add new edge to parse forest."""
		cdef Edges edges
		cdef Edge *edge
		cdef size_t item = cellidx(
				start, end, self.lensent, self.grammar.nonterminals) + lhs
		cdef size_t block
		if item in self.parseforest:
			block = len(<list>self.parseforest[item]) - 1
			edges = self.parseforest[item][block]
			if edges.len == EDGES_SIZE:
				edges = Edges()
				self.parseforest[item].append(edges)
		else:
			edges = Edges()
			self.parseforest[item] = [edges]
			self.itemsinorder.append(item)
		edge = &(edges.data[edges.len])
		edge.rule = rule
		edge.pos.mid = mid
		edges.len += 1

	cdef void updateprob(self, uint32_t lhs, Idx start, Idx end, double prob):
		"""Update probability for item if better than current one."""
		cdef size_t item = cellidx(
				start, end, self.lensent, self.grammar.nonterminals) + lhs
		if item not in self.probs or prob < self.probs[item]:
			self.probs[item] = prob

	cdef double _subtreeprob(self, size_t item):
		"""Get viterbi / inside probability of a subtree headed by `item`."""
		return (PyFloat_AS_DOUBLE(self.probs[item])
				if item in self.probs else INFINITY)

	cdef double subtreeprob(self, item):
		return self._subtreeprob(item)

	cdef bint hasitem(self, size_t item):
		"""Test if item is in chart."""
		return item in self.parseforest


def parse(sent, Grammar grammar, tags=None, start=None, dict whitelist=None):
	"""A CKY parser modeled after Bodenstab's 'fast grammar loop'.

	If ``whitelist`` is given, the loop is filtered by the allowed items.
	The whitelist is of the form: whitelist = {cell: {label: None}};
	cell is a represenattion of a span as used by the CFGChart, label is an
	integer for a non-terminal label; the value of the inner dict is not used.
	The presence of a label means the span with that label will not be pruned.
	"""
	if grammar.maxfanout != 1:
		raise ValueError('Not a PCFG! fanout: %d' % grammar.maxfanout)
	if not grammar.logprob:
		raise ValueError('Expected grammar with log probabilities.')
	if grammar.nonterminals < 20000:
		chart = DenseCFGChart(grammar, sent, start)
		return parse_main(sent, <DenseCFGChart>chart, grammar, tags=tags,
				start=start, whitelist=whitelist)
	else:
		chart = SparseCFGChart(grammar, sent, start)
		return parse_main(sent, <SparseCFGChart>chart, grammar, tags=tags,
				start=start, whitelist=whitelist)


cdef parse_main(sent, CFGChart_fused chart, Grammar grammar, tags=None,
		start=None, dict whitelist=None):
	cdef:
		short [:, :] minleft, maxleft, minright, maxright
		DoubleAgenda unaryagenda = DoubleAgenda()
		dict cellwhitelist = None
		Rule *rule
		short left, right, mid, span, lensent = len(sent)
		short narrowl, narrowr, widel, wider, minmid, maxmid
		double oldscore, prob
		uint32_t n, lhs, rhs1
		size_t cell
	minleft, maxleft, minright, maxright = minmaxmatrices(
			grammar.nonterminals, lensent)
	# assign POS tags
	covered, msg = populatepos(grammar, chart, sent, tags, whitelist,
			minleft, maxleft, minright, maxright)
	if not covered:
		return chart, msg

	for span in range(2, lensent + 1):
		# constituents from left to right
		for left in range(lensent - span + 1):
			right = left + span
			cell = cellidx(left, right, lensent, grammar.nonterminals)
			if whitelist is not None:
				cellwhitelist = <dict>whitelist.get(cell)
			# apply binary rules
			# FIXME: if whitelist is given, loop only over whitelisted labels
			# for cell
			# for lhs in cellwhitelist:
			# only loop over labels which occur on LHS of a phrasal rule.
			for lhs in range(1, grammar.phrasalnonterminals):
				if cellwhitelist is not None and lhs not in cellwhitelist:
					continue
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
							chart.addedge(lhs, left, right, mid, rule)
							chart.updateprob(lhs, left, right, prob)
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
			# FIXME: efficiently fetch labels in current cell: getitems(cell)
			# or: chart.itemsinorder[lastidx:]
			unaryagenda.update_entries([new_DoubleEntry(
						rhs1, chart._subtreeprob(cell + rhs1), 0)
					for rhs1 in range(1, grammar.phrasalnonterminals)
					if chart.hasitem(cell + rhs1)
					and grammar.unary[rhs1].rhs1 == rhs1])
			while unaryagenda.length:
				rhs1 = unaryagenda.popentry().key
				for n in range(grammar.numunary):
					rule = &(grammar.unary[rhs1][n])
					if rule.rhs1 != rhs1:
						break
					elif TESTBIT(grammar.mask, rule.no) or (
							cellwhitelist is not None
							and rule.lhs not in cellwhitelist):
						continue
					lhs = rule.lhs
					prob = rule.prob + chart._subtreeprob(cell + rhs1)
					chart.addedge(lhs, left, right, right, rule)
					if (not chart.hasitem(cell + lhs)
							or prob < chart._subtreeprob(cell + lhs)):
						chart.updateprob(lhs, left, right, prob)
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
		return chart, "no parse " + chart.stats()
	return chart, chart.stats()


def parse_symbolic(sent, Grammar grammar, tags=None, start=None):
	"""Parse sentence without regard for probabilities.

	Currently this calls the normal probabilistic CKY parser producing
	viterbi probabilities, but a more efficient non-probabilistic algorithm
	for producing a parse forest may be possible."""
	return parse(sent, grammar, tags=tags, start=start, whitelist=None)


cdef populatepos(Grammar grammar, CFGChart_fused chart, sent, tags, whitelist,
		short [:, :] minleft, short [:, :] maxleft,
		short [:, :] minright, short [:, :] maxright):
	"""Apply all possible lexical and unary rules on each lexical span.

	:returns: a tuple ``(success, msg)`` where ``success`` is True if a POS tag
	was found for every word in the sentence."""
	cdef:
		DoubleAgenda unaryagenda = DoubleAgenda()
		Rule *rule
		LexicalRule lexrule
		uint32_t n, lhs, rhs1
		short left, right, lensent = len(sent)
	for left, word in enumerate(sent):
		tag = tags[left].encode('ascii') if tags else None
		right = left + 1
		cell = cellidx(left, right, lensent, grammar.nonterminals)
		recognized = False
		for lexrule in grammar.lexicalbyword.get(word, ()):
			assert whitelist is None or cell in whitelist, whitelist.keys()
			if whitelist is not None and lexrule.lhs not in whitelist[cell]:
				continue
			lhs = lexrule.lhs
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if (tag is None or grammar.tolabel[lhs] == tag
					or grammar.tolabel[lhs].startswith(tag + b'@')):
				chart.addedge(lhs, left, right, right, NULL)
				chart.updateprob(lhs, left, right, lexrule.prob)
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
			chart.updateprob(lhs, left, right, 0.0)
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
		# NB: for this agenda, only the probabilities of the edges matter
		while unaryagenda.length:
			rhs1 = unaryagenda.popentry().key
			for n in range(grammar.numunary):
				rule = &(grammar.unary[rhs1][n])
				if rule.rhs1 != rhs1:
					break
				elif TESTBIT(grammar.mask, rule.no) or (
						whitelist is not None
						and rule.lhs not in whitelist[cell]):
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
				chart.updateprob(lhs, left, right, prob)
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


def doinsideoutside(sent, Grammar grammar, inside=None, outside=None,
		tags=None, startid=None):
	if grammar.maxfanout != 1:
		raise('Not a PCFG! fanout = %d' % grammar.maxfanout)
	if grammar.logprob:
		raise('Grammar must not have log probabilities.')
	lensent = len(sent)
	if startid is None:
		startid = grammar.toid[grammar.start]
	if inside is None:
		inside = np.zeros((lensent, lensent + 1,
				grammar.nonterminals), dtype='d')
	else:
		inside[:len(sent), :len(sent) + 1, :] = 0.0
	if outside is None:
		outside = np.zeros((lensent, lensent + 1,
				grammar.nonterminals), dtype='d')
	else:
		outside[:len(sent), :len(sent) + 1, :] = 0.0
	minmaxlr = insidescores(sent, grammar, inside, tags)
	if inside[0, len(sent), startid]:
		outsidescores(grammar, sent, startid, inside, outside, *minmaxlr)
		msg = 'inside prob=%g' % inside[0, len(sent), startid]
		start = (startid, 0, len(sent))
	else:
		start = None
		msg = "no parse"
	return inside, outside, start, msg


def insidescores(sent, Grammar grammar, double [:, :, :] inside, tags=None):
	"""Compute inside scores.

	NB: These are not Viterbi scores, but sums of all derivations headed by a
	certain ``(label, span)``."""
	cdef:
		short left, right, span, lensent = len(sent)
		short narrowl, narrowr, minmid, maxmid
		double prob, ls, rs
		uint32_t n, lhs, rhs1
		bint foundbetter = False
		Rule *rule
		LexicalRule lexrule
		list cell = [{} for _ in grammar.toid]
		DoubleAgenda unaryagenda = DoubleAgenda()
		short [:, :] minleft, maxleft, minright, maxright
		double [:] unaryscores = np.empty(grammar.nonterminals, dtype='d')
	minleft, maxleft, minright, maxright = minmaxmatrices(
			grammar.nonterminals, lensent)
	for left in range(lensent):  # assign POS tags
		tag = tags[left].encode('ascii') if tags else None
		right = left + 1
		for lexrule in grammar.lexicalbyword.get(sent[left], []):
			lhs = lexrule.lhs
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if not tags or (grammar.tolabel[lhs] == tag
					or grammar.tolabel[lhs].startswith(tag + b'@')):
				inside[left, right, lhs] = lexrule.prob
		if not inside.base[left, right].any():
			if tags is not None:
				lhs = grammar.toid[tag]
				if not tags or (grammar.tolabel[lhs] == tag
						or grammar.tolabel[lhs].startswith(tag + b'@')):
					inside[left, right, lhs] = 1.
			else:
				raise ValueError("not covered: %r" % (tag or sent[left]), )
		# unary rules on POS tags (NB: agenda is a min-heap, negate probs)
		unaryagenda.update_entries([new_DoubleEntry(
				rhs1, -inside[left, right, rhs1], 0)
			for rhs1 in range(grammar.nonterminals)
			if inside[left, right, rhs1]
			and grammar.unary[rhs1].rhs1 == rhs1])
		unaryscores[:] = 0.0
		while unaryagenda.length:
			rhs1 = unaryagenda.popentry().key
			for n in range(grammar.numrules):
				rule = &(grammar.unary[rhs1][n])
				if rule.rhs1 != rhs1:
					break
				prob = rule.prob * inside[left, right, rhs1]
				lhs = rule.lhs
				edge = (rule.no, right)
				if edge not in cell[lhs]:
					unaryagenda.setifbetter(lhs, -prob)
					inside[left, right, lhs] += prob
					cell[lhs][edge] = edge
		for a in cell:
			a.clear()
		for lhs in range(grammar.nonterminals):
			if inside[left, right, lhs]:
				# update filter
				if left > minleft[lhs, right]:
					minleft[lhs, right] = left
				if left < maxleft[lhs, right]:
					maxleft[lhs, right] = left
				if right < minright[lhs, left]:
					minright[lhs, left] = right
				if right > maxright[lhs, left]:
					maxright[lhs, left] = right
	for span in range(2, lensent + 1):
		# constituents from left to right
		for left in range(lensent - span + 1):
			right = left + span
			# binary rules
			for n in range(grammar.numrules):
				rule = &(grammar.bylhs[0][n])
				lhs = rule.lhs
				if lhs == grammar.nonterminals:
					break
				elif not rule.rhs2:
					continue
				narrowr = minright[rule.rhs1, left]
				narrowl = minleft[rule.rhs2, right]
				if narrowr >= right or narrowl < narrowr:
					continue
				widel = maxleft[rule.rhs2, right]
				minmid = narrowr if narrowr > widel else widel
				wider = maxright[rule.rhs1, left]
				maxmid = wider if wider < narrowl else narrowl
				# oldscore = inside[left, right, lhs]
				foundbetter = False
				for split in range(minmid, maxmid + 1):
					ls = inside[left, split, rule.rhs1]
					if ls == 0.0:
						continue
					rs = inside[split, right, rule.rhs2]
					if rs == 0.0:
						continue
					foundbetter = True
					inside[left, right, lhs] += rule.prob * ls * rs
					# assert 0.0 < inside[left, right, lhs] <= 1.0, (
					# 	inside[left, right, lhs],
					# 	left, right, grammar.tolabel[lhs])
				if foundbetter:  # and oldscore == 0.0:
					if left > minleft[lhs, right]:
						minleft[lhs, right] = left
					if left < maxleft[lhs, right]:
						maxleft[lhs, right] = left
					if right < minright[lhs, left]:
						minright[lhs, left] = right
					if right > maxright[lhs, left]:
						maxright[lhs, left] = right
			# unary rules on this span
			unaryagenda.update_entries([new_DoubleEntry(
						rhs1, -inside[left, right, rhs1], 0)
					for rhs1 in range(grammar.nonterminals)
					if inside[left, right, rhs1]
					and grammar.unary[rhs1].rhs1 == rhs1])
			unaryscores[:] = 0.0
			while unaryagenda.length:
				rhs1 = unaryagenda.popentry().key
				for n in range(grammar.numrules):
					rule = &(grammar.unary[rhs1][n])
					if rule.rhs1 != rhs1:
						break
					prob = rule.prob * inside[left, right, rhs1]
					lhs = rule.lhs
					edge = (rule.no, right)
					if edge not in cell[lhs]:
						unaryagenda.setifbetter(lhs, -prob)
						inside[left, right, lhs] += prob
						cell[lhs][edge] = edge
			for a in cell:
				a.clear()
			for lhs in range(grammar.nonterminals):
				# update filter
				if inside[left, right, lhs]:
					if left > minleft[lhs, right]:
						minleft[lhs, right] = left
					if left < maxleft[lhs, right]:
						maxleft[lhs, right] = left
					if right < minright[lhs, left]:
						minright[lhs, left] = right
					if right > maxright[lhs, left]:
						maxright[lhs, left] = right
	return minleft, maxleft, minright, maxright


def outsidescores(Grammar grammar, sent, uint32_t start,
		double [:, :, :] inside, double [:, :, :] outside,
		short [:, :] minleft, short [:, :] maxleft,
		short [:, :] minright, short [:, :] maxright):
	cdef:
		short left, right, span, lensent = len(sent)
		short narrowl, narrowr, minmid, maxmid
		double ls, rs, os
		uint32_t n, lhs
		Rule *rule
		DoubleAgenda unaryagenda = DoubleAgenda()
		list cell = [{} for _ in grammar.toid]
	outside[0, lensent, start] = 1.0
	for span in range(lensent, 0, -1):
		for left in range(1 + lensent - span):
			right = left + span
			# unary rules
			unaryagenda.update_entries([new_DoubleEntry(
					lhs, -outside[left, right, lhs], 0)
				for lhs in range(grammar.nonterminals)
				if outside[left, right, lhs]])
			while unaryagenda.length:
				lhs = unaryagenda.popentry().key
				for n in range(grammar.numrules):
					rule = &(grammar.bylhs[lhs][n])
					if rule.lhs != lhs:
						break
					elif rule.rhs2:
						continue
					prob = rule.prob * outside[left, right, lhs]
					edge = (rule.no, right)
					if edge not in cell[lhs]:
						unaryagenda.setifbetter(rule.rhs1, -prob)
						cell[lhs][edge] = edge
						outside[left, right, rule.rhs1] += prob
						# assert 0.0 < outside[left, right, rule.rhs1] <= 1.0, (
						# 		'illegal value: outside[%d, %d, %s] = %g' % (
						# 			left, right, grammar.tolabel[rule.rhs1],
						# 			outside[left, right, rule.rhs1]),
						# 		rule.prob, outside[left, right, lhs],
						# 		grammar.tolabel[rule.lhs])
			for lhs in range(grammar.nonterminals):
				cell[lhs].clear()
			# binary rules
			for n in range(grammar.numrules):
				rule = &(grammar.bylhs[0][n])
				lhs = rule.lhs
				if lhs == grammar.nonterminals:
					break
				elif not rule.rhs2 or outside[left, right, lhs] == 0.0:
					continue
				os = outside[left, right, lhs]
				narrowr = minright[rule.rhs1, left]
				narrowl = minleft[rule.rhs2, right]
				if narrowr >= right or narrowl < narrowr:
					continue
				widel = maxleft[rule.rhs2, right]
				minmid = narrowr if narrowr > widel else widel
				wider = maxright[rule.rhs1, left]
				maxmid = wider if wider < narrowl else narrowl
				for split in range(minmid, maxmid + 1):
					ls = inside[left, split, rule.rhs1]
					if ls == 0.0:
						continue
					rs = inside[split, right, rule.rhs2]
					if rs == 0.0:
						continue
					outside[left, split, rule.rhs1] += rule.prob * rs * os
					outside[split, right, rule.rhs2] += rule.prob * ls * os
					# assert 0.0 < outside[left, split, rule.rhs1] <= 1.0, (
					# 		'illegal value: outside[%d, %d, %s] = %g' % (
					# 			left, split, grammar.tolabel[rule.rhs1],
					# 			outside[left, split, rule.rhs1]),
					# 		rule.prob, rs, os, grammar.tolabel[rule.lhs])
					# assert 0.0 < outside[split, right, rule.rhs2] <= 1.0, (
					# 		'illegal value: outside[%d, %d, %s] = %g' % (
					# 			split, right, grammar.tolabel[rule.rhs2],
					# 			outside[split, right, rule.rhs2]))


def minmaxmatrices(nonterminals, lensent):
	"""Create matrices to track minima and maxima for binary splits."""
	minleft = np.empty((nonterminals, lensent + 1), dtype='int16')
	maxleft = np.empty_like(minleft)
	minleft[...], maxleft[...] = -1, lensent + 1
	minright, maxright = maxleft.copy(), minleft.copy()
	return minleft, maxleft, minright, maxright


def chartmatrix(nonterminals, lensent):
	viterbi = np.empty((nonterminals, lensent, lensent + 1), dtype='d')
	viterbi[...] = np.inf
	return viterbi

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
	from discodop.parser import which
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
	tokens = [word.replace('(', '-LRB-').replace(')', '-RRB-').encode('utf8')
			for word in sent]
	if tags:
		tokens = ['%s@%s' % (tag, token) for tag, token in zip(tags, tokens)]
	# pass empty 'unkwown word file' to disable bitpar's smoothing
	args = ['-y'] if n == 0 else ['-b', str(n)]
	args += ['-s', startlabel, '-vp', '-u', '/dev/null', rulesfile, lexiconfile]
	proc = subprocess.Popen([which('bitpar')] + args,
			shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
			stderr=subprocess.PIPE)
	results, msg = proc.communicate('\n'.join(tokens) + '\n')
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
	cdef Rule *rule
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
		chart.updateprob(lhs, left, right, prob)
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


def pprint_matrix(matrix, sent, tolabel, matrix2=None):
	"""Print a numpy matrix chart; optionally in parallel with another."""
	for span in range(1, len(sent) + 1):
		for left in range(len(sent) - span + 1):
			right = left + span
			if matrix[left, right].any() or (
					matrix2 is not None and matrix2[left, right].any()):
				print('[%d:%d]' % (left, right))
				for lhs in range(len(matrix[left, right])):
					if matrix[left, right, lhs] or (
							matrix2 is not None and matrix2[left, right, lhs]):
						print('%20s\t%8.6g' % (tolabel[lhs].decode('ascii'),
								matrix[left, right, lhs]), end='')
						if matrix2 is not None:
							print('\t%8.6g' % matrix2[left, right, lhs], end='')
						print()


def test():
	from discodop.containers import Grammar
	from discodop.disambiguation import getderivations, marginalize
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
	cfg1 = Grammar([
		((('NP', 'Epsilon'), ('mary', )), 1),
		((('S', 'NP', 'VP'), ((0, 1), )), 1),
		((('VP', 'Epsilon'), ('walks', )), 1)], start='S')
	cfg1.switch(u'default', False)
	i, o, start, _ = doinsideoutside('mary walks'.split(), cfg1)
	assert start
	print(i[0, 2, cfg1.toid[b'S']], o[0, 2, cfg1.toid[b'S']])
	i, o, start, _ = doinsideoutside('walks mary'.split(), cfg1)
	assert not start
	print(i[0, 2, cfg1.toid[b'S']], o[0, 2, cfg1.toid[b'S']])
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
	cfg2.switch(u'default', False)
	sent = 'astronomers saw stars with telescopes'.split()
	inside, outside, _, msg = doinsideoutside(sent, cfg2)
	print(msg)
	pprint_matrix(inside, sent, cfg2.tolabel, outside)
	cfg2.switch(u'default', True)
	chart, msg = parse(sent, cfg2)
	print(msg)
	print(chart)
	derivations, entries = getderivations(chart, 10, True, False, True)
	mpp, _ = marginalize('mpp', derivations, entries, chart)
	for a, p, _ in sorted(mpp, key=itemgetter(1), reverse=True):
		print(p, a)
	# chart1, msg1 = parse_symbolic(sent, cfg2)
	# print(msg, '\n', msg1)

__all__ = ['CFGChart', 'DenseCFGChart', 'SparseCFGChart', 'bitpar_nbest',
		'bitpar_yap_forest', 'chartmatrix', 'doinsideoutside', 'insidescores',
		'minmaxmatrices', 'outsidescores', 'parse', 'parse_bitpar',
		'parse_symbolic', 'pprint_matrix', 'renumber']
