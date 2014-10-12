"""Parser for string-rewriting Linear Context-Free Rewriting Systems.

Expects binarized, epsilon-free, monotone LCFRS grammars."""
from __future__ import print_function
from collections import defaultdict, deque
import logging
import numpy as np
cimport cython
include "constants.pxi"
include "agenda.pxi"

cdef SmallChartItem COMPONENT = new_SmallChartItem(0, 0)
cdef SmallChartItem NONE = new_SmallChartItem(0, 0)
cdef FatChartItem FATNONE = new_FatChartItem(0)
cdef FatChartItem FATCOMPONENT = new_FatChartItem(0)
cdef double INFINITY = float('infinity')

cdef inline bint equalitems(LCFRSItem_fused op1, LCFRSItem_fused op2):
	if LCFRSItem_fused is SmallChartItem:
		return op1.label == op2.label and op1.vec == op2.vec
	return op1.label == op2.label and (
		memcmp(<UChar *>op1.vec, <UChar *>op2.vec, sizeof(op1.vec)) == 0)


cdef class LCFRSChart(Chart):
	"""A chart for LCFRS grammars. An item is a ChartItem object."""
	def __init__(self, Grammar grammar, list sent,
			start=None, logprob=True, viterbi=True):
		self.grammar = grammar
		self.sent = sent
		self.lensent = len(sent)
		self.start = grammar.toid[grammar.start if start is None else start]
		self.logprob = logprob
		self.viterbi = viterbi
		self.probs = grammar.nonterminals * [None]
		self.parseforest = {}
		self.itemsinorder = []

	cdef void addlexedge(self, item, short wordidx):
		"""Add lexical edge."""
		cdef Edges edges
		cdef Edge *edge
		cdef size_t block
		# NB: lexical edges should always be unique, but just in case ...
		if item in self.parseforest:
			block = len(self.parseforest[item]) - 1
			edges = self.parseforest[item][block]
			if edges.len == EDGES_SIZE:
				edges = Edges()
				self.parseforest[item].append(edges)
		else:
			edges = Edges()
			self.parseforest[item] = [edges]
			self.itemsinorder.append(item)
		edge = &(edges.data[edges.len])
		edge.rule = NULL
		edge.pos.mid = wordidx + 1
		edges.len += 1

	cdef void updateprob(self, ChartItem item, double prob):
		cdef dict probs = <dict>self.probs[item.label]
		if probs is None:
			self.probs[item.label] = {item: item}
		elif item in probs:
			item = <ChartItem>probs[item]
			if prob < item.prob:
				item.prob = prob
		else:
			probs[item] = item

	cdef void addprob(self, ChartItem item, double prob):
		# cdef dict probs = <dict>self.probs[item.label]
		# NB: item must be an instance in the chart, otherwise do:
		# item = probs[item]
		item.prob += prob

	cdef double _subtreeprob(self, ChartItem item):
		return item.prob

	cdef double subtreeprob(self, item):
		cdef dict probs = <dict>self.probs[(<ChartItem>item).label]
		return self._subtreeprob(<ChartItem>probs[item])

	def label(self, item):
		return (<ChartItem>item).label

	def itemstr(self, item):
		return '%s[%s]' % (self.grammar.tolabel[self.label(item)
			].decode('ascii'), item.binrepr(self.lensent))


@cython.final
cdef class SmallLCFRSChart(LCFRSChart):
	"""For sentences that fit into a single machine word."""
	def __init__(self, Grammar grammar, list sent,
			start=None, logprob=True, viterbi=True):
		super(SmallLCFRSChart, self).__init__(
				grammar, sent, start, logprob, viterbi)
		self.tmpitem = new_SmallChartItem(0, 0)

	cdef void addedge(self, SmallChartItem item, SmallChartItem left,
			Rule *rule):
		"""Add new edge."""
		cdef Edges edges
		cdef Edge *edge
		cdef size_t block
		if item in self.parseforest:
			block = len(self.parseforest[item]) - 1
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
		edge.pos.lvec = left.vec
		edges.len += 1

	cdef _left(self, item, Edge *edge):
		if edge.rule is NULL:
			return None
		self.tmpitem.label = edge.rule.rhs1
		self.tmpitem.vec = edge.pos.lvec
		return self.probs[self.tmpitem.label][self.tmpitem]

	cdef _right(self, item, Edge *edge):
		if edge.rule is NULL or edge.rule.rhs2 == 0:
			return None
		self.tmpitem.label = edge.rule.rhs2
		self.tmpitem.vec = (<SmallChartItem>item).vec ^ edge.pos.lvec
		return self.probs[self.tmpitem.label][self.tmpitem]

	def indices(self, SmallChartItem item):
		cdef short n
		return [n for n in range(len(self.sent)) if testbit(item.vec, n)]

	cdef copy(self, item):
		return (<SmallChartItem>item).copy()

	def root(self):
		cdef SmallChartItem item = new_SmallChartItem(
				self.grammar.toid[self.grammar.start],
				(1ULL << self.lensent) - 1)
		if self.probs[item.label] is None:
			return item
		return self.probs[item.label].get(item, item)


@cython.final
cdef class FatLCFRSChart(LCFRSChart):
	"""LCFRS chart that supports longer sentences."""
	def __init__(self, Grammar grammar, list sent,
			start=None, logprob=True, viterbi=True):
		super(FatLCFRSChart, self).__init__(
				grammar, sent, start, logprob, viterbi)
		self.tmpitem = new_FatChartItem(0)

	cdef void addedge(self, FatChartItem item, FatChartItem left, Rule *rule):
		"""Add new edge and update viterbi probability."""
		cdef Edges edges
		cdef Edge *edge
		cdef size_t block
		if item in self.parseforest:
			block = len(self.parseforest[item]) - 1
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
		# NB: store pointer; breaks when `left` is garbage collected!
		edge.pos.lvec_fat = left.vec
		edges.len += 1

	cdef _left(self, item, Edge *edge):
		cdef size_t n
		if edge.rule is NULL:
			return None
		self.tmpitem.label = edge.rule.rhs1
		for n in range(SLOTS):
			self.tmpitem.vec[n] = edge.pos.lvec_fat[n]
		return self.probs[self.tmpitem.label][self.tmpitem]

	cdef _right(self, item, Edge *edge):
		cdef size_t n
		if edge.rule is NULL:
			return None
		self.tmpitem.label = edge.rule.rhs2
		for n in range(SLOTS):
			self.tmpitem.vec[n] = (
					<FatChartItem>item).vec[n] ^ edge.pos.lvec_fat[n]
		return self.probs[self.tmpitem.label][self.tmpitem]

	def indices(self, FatChartItem item):
		cdef short n
		return [n for n in range(len(self.sent)) if TESTBIT(item.vec, n)]

	cdef copy(self, item):
		return (<FatChartItem>item).copy()

	def root(self):
		cdef FatChartItem item = CFGtoFatChartItem(
				self.grammar.toid[self.grammar.start], 0, self.lensent)
		if self.probs[item.label] is None:
			return item
		return self.probs[item.label].get(item, item)


def parse(sent, Grammar grammar, tags=None, bint exhaustive=True,
		start=None, list whitelist=None, bint splitprune=False,
		bint markorigin=False, estimates=None, int beamwidth=0):
	"""Parse sentence and produce a chart.

	:param sent: a sequence of tokens
	:param grammar: a ``Grammar`` object.
	:returns: a ``Chart`` object.
	:param tags: optionally, a sequence of gold tags, which will be used
		instead of trying all possible tags.
	:param exhaustive: don't stop at viterbi parse, return a full chart
	:param start: integer corresponding to the start symbol that analyses
		should have, e.g., grammar.toid[b'ROOT']
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
	:param beamwidth: specify the maximum number of items that will be explored
		for each particular span, on a first-come-first-served basis.
		setting to 0 disables this feature. experimental."""
	if len(sent) < sizeof(COMPONENT.vec) * 8:
		chart = SmallLCFRSChart(grammar, list(sent), start)
		return parse_main(<SmallLCFRSChart>chart, <SmallChartItem>chart.root(),
				sent, grammar, tags, exhaustive, start, whitelist, splitprune,
				markorigin, estimates, beamwidth)
	chart = FatLCFRSChart(grammar, list(sent), start)
	return parse_main(<FatLCFRSChart>chart, <FatChartItem>chart.root(),
			sent, grammar, tags, exhaustive, start, whitelist, splitprune,
			markorigin, estimates, beamwidth)


cdef parse_main(LCFRSChart_fused chart, LCFRSItem_fused goal, sent,
		Grammar grammar, tags=None, bint exhaustive=True, start=None,
		list whitelist=None, bint splitprune=False, bint markorigin=False,
		estimates=None, int beamwidth=0):
	cdef:
		DoubleAgenda agenda = DoubleAgenda()  # the agenda
		list probs = chart.probs  # viterbi probabilities for items
		dict beam = <dict>defaultdict(int)  # histogram of spans
		Rule *rule
		LexicalRule lexrule
		LCFRSItem_fused item, newitem
		DoubleEntry entry
		double [:, :, :, :] outside = None  # outside estimates, if provided
		double siblingprob, score
		short wordidx, lensent = len(sent), estimatetype = 0
		int length = 1, left = 0, right = 0, gaps = 0
		UInt lhs
		size_t blocked = 0, maxA = 0, n
		bint recognized
	if start is None:
		start = grammar.toid[grammar.start]
	if estimates is not None:
		estimatetypestr, outside = estimates
		estimatetype = {'SX': SX, 'SXlrgaps': SXlrgaps}[estimatetypestr]
	# newitem will be recycled until it is added to the chart
	if LCFRSItem_fused is SmallChartItem:
		newitem = new_SmallChartItem(0, 0)
	elif LCFRSItem_fused is FatChartItem:
		newitem = new_FatChartItem(0)
	for wordidx, word in enumerate(sent):  # add preterminals to chart
		recognized = False
		if LCFRSItem_fused is SmallChartItem:
			item = new_SmallChartItem(0, wordidx)
		elif LCFRSItem_fused is FatChartItem:
			item = new_FatChartItem(0)
			item.vec[0] = wordidx
		tag = tags[wordidx].encode('ascii') if tags else None
		if estimates is not None:
			left = wordidx
			gaps = 0
			right = lensent - 1 - wordidx
		for lexrule in grammar.lexicalbyword.get(word, ()):
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if (not tags or grammar.tolabel[lexrule.lhs] == tag
					or grammar.tolabel[lexrule.lhs].startswith(tag + b'@')):
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
				# NB: do NOT add length of span to score, so that the scores of
				# POS tags are all strictly smaller than any unaries on them.
				newitem.label = lexrule.lhs
				newitem.prob = lexrule.prob
				if LCFRSItem_fused is SmallChartItem:
					newitem.vec = 1ULL << wordidx
				elif LCFRSItem_fused is FatChartItem:
					memset(<void *>newitem.vec, 0, SLOTS * sizeof(ULong))
					SETBIT(newitem.vec, wordidx)
				if process_lexedge(newitem, score, wordidx,
						agenda, chart, whitelist):
					if LCFRSItem_fused is SmallChartItem:
						newitem = <LCFRSItem_fused>SmallChartItem.__new__(
								SmallChartItem)
					elif LCFRSItem_fused is FatChartItem:
						newitem = <LCFRSItem_fused>FatChartItem.__new__(
								FatChartItem)
					recognized = True
				else:
					blocked += 1
		if not recognized and tags and tag in grammar.toid:
			lhs = grammar.toid[tag]
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
			newitem.prob = 0.0
			if LCFRSItem_fused is SmallChartItem:
				newitem.vec = 1ULL << wordidx
			elif LCFRSItem_fused is FatChartItem:
				memset(<void *>newitem.vec, 0, SLOTS * sizeof(ULong))
				SETBIT(newitem.vec, wordidx)
			# prevent pruning of provided tags => whitelist == None
			if process_lexedge(newitem, 0.0, wordidx,
					agenda, chart, None):
				if LCFRSItem_fused is SmallChartItem:
					newitem = <LCFRSItem_fused>SmallChartItem.__new__(
							SmallChartItem)
				elif LCFRSItem_fused is FatChartItem:
					newitem = <LCFRSItem_fused>FatChartItem.__new__(
							FatChartItem)
				recognized = True
			else:
				raise ValueError('tag %r is blocked.' % tag)
		elif not recognized:
			if tag is None and word not in grammar.lexicalbyword:
				return chart, 'no parse: %r not in lexicon' % word
			elif tag is not None and tag not in grammar.toid:
				return chart, 'no parse: unknown tag %r' % tag
			return chart, 'no parse: all tags for %r blocked' % word
	while agenda.length:  # main parsing loop
		entry = agenda.popentry()
		item = <LCFRSItem_fused>entry.key
		# store viterbi probability; cannot do this when this item is added to
		# the agenda because that would give rise to duplicate edges.
		chart.updateprob(item, item.prob)
		if equalitems(item, goal):
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
						SLOTS * sizeof(ULong))
			if estimates is not None:
				if LCFRSItem_fused is SmallChartItem:
					left = nextset(item.vec, 0)
					gaps = bitlength(item.vec) - length - left
					right = lensent - length - left - gaps
				elif LCFRSItem_fused is FatChartItem:
					left = anextset(item.vec, 0, SLOTS)
					gaps = abitlength(item.vec, SLOTS) - length - left
					right = lensent - length - left - gaps
			if beamwidth:
				lhs = item.label
				item.label = 0
				if beam[item] > beamwidth:
					continue
				beam[item] += 1
				item.label = lhs
			for n in range(grammar.numunary):
				rule = &(grammar.unary[item.label][n])
				if rule.rhs1 != item.label:
					break
				elif TESTBIT(grammar.mask, rule.no):
					continue
				score = newitem.prob = item.prob + rule.prob
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
				if process_edge(newitem, score, rule, item,
						agenda, chart, estimatetype, whitelist,
						splitprune and grammar.fanout[rule.lhs] != 1,
						markorigin):
					if LCFRSItem_fused is SmallChartItem:
						newitem = <LCFRSItem_fused>SmallChartItem.__new__(
								SmallChartItem)
						newitem.vec = item.vec
					elif LCFRSItem_fused is FatChartItem:
						newitem = <LCFRSItem_fused>FatChartItem.__new__(
								FatChartItem)
						memcpy(<void *>newitem.vec, <void *>item.vec,
								SLOTS * sizeof(ULong))
				else:
					blocked += 1
			# binary production, item from agenda is on the right
			for n in range(grammar.numbinary):
				rule = &(grammar.rbinary[item.label][n])
				if rule.rhs2 != item.label:
					break
				elif probs[rule.rhs1] is None:
					continue
				elif TESTBIT(grammar.mask, rule.no):
					continue
				for sib in <dict>probs[rule.rhs1]:
					newitem.label = rule.lhs
					combine_item(newitem, <LCFRSItem_fused>sib, item)
					if beamwidth:
						lhs = item.label
						item.label = 0
						if beam[newitem] > beamwidth:
							item.label = lhs
							continue
						beam[newitem] += 1
						item.label = lhs
					if concat(rule, <LCFRSItem_fused>sib, item):
						siblingprob = (<LCFRSItem_fused>sib).prob
						score = newitem.prob = item.prob + siblingprob + rule.prob
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
						if process_edge(newitem, score,
								rule, <LCFRSItem_fused>sib, agenda, chart,
								estimatetype, whitelist,
								splitprune and grammar.fanout[rule.lhs] != 1,
								markorigin):
							if LCFRSItem_fused is SmallChartItem:
								newitem = <LCFRSItem_fused>SmallChartItem.__new__(
										SmallChartItem)
							elif LCFRSItem_fused is FatChartItem:
								newitem = <LCFRSItem_fused>FatChartItem.__new__(
										FatChartItem)
						else:
							blocked += 1
			# binary production, item from agenda is on the left
			for n in range(grammar.numbinary):
				rule = &(grammar.lbinary[item.label][n])
				if rule.rhs1 != item.label:
					break
				elif probs[rule.rhs2] is None:
					continue
				elif TESTBIT(grammar.mask, rule.no):
					continue
				for sib in <dict>probs[rule.rhs2]:
					newitem.label = rule.lhs
					combine_item(newitem, item, <LCFRSItem_fused>sib)
					if beamwidth:
						lhs = item.label
						item.label = 0
						if beam[newitem] > beamwidth:
							item.label = lhs
							continue
						beam[newitem] += 1
						item.label = lhs
					if concat(rule, item, <LCFRSItem_fused>sib):
						siblingprob = (<LCFRSItem_fused>sib).prob
						score = newitem.prob = item.prob + siblingprob + rule.prob
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
						if process_edge(newitem, score, rule, item,
								agenda, chart, estimatetype, whitelist,
								splitprune and grammar.fanout[rule.lhs] != 1,
								markorigin):
							if LCFRSItem_fused is SmallChartItem:
								newitem = <LCFRSItem_fused>SmallChartItem.__new__(
										SmallChartItem)
							elif LCFRSItem_fused is FatChartItem:
								newitem = <LCFRSItem_fused>FatChartItem.__new__(
										FatChartItem)
						else:
							blocked += 1

		if agenda.length > maxA:
			maxA = agenda.length
	msg = ('agenda max %d, now %d, %s, blocked %d' % (
			maxA, len(agenda), chart.stats(), blocked))
	if not chart:
		return chart, 'no parse ' + msg
	return chart, msg


cdef inline bint process_edge(LCFRSItem_fused newitem,
		double score, Rule *rule, LCFRSItem_fused left,
		DoubleAgenda agenda, LCFRSChart_fused chart, int estimatetype,
		list whitelist, bint splitprune, bint markorigin):
	"""Decide what to do with a newly derived edge.

	:returns: ``True`` when edge is accepted in the chart, ``False`` when
		blocked. When ``False``, ``newitem`` may be reused."""
	cdef UInt a, b, n, cnt, label
	cdef bint inagenda = newitem in agenda.mapping
	cdef bint inchart = newitem in chart.parseforest
	cdef list componentlist = None
	cdef dict componentdict = None
	if not inagenda and not inchart:
		# check if we need to prune this item
		if whitelist is not None and whitelist[newitem.label] is not None:
			if splitprune:  # disc. item to be treated as several split items?
				if markorigin:
					componentlist = <list>(whitelist[newitem.label])
				else:
					componentdict = <dict>(whitelist[newitem.label])
				b = cnt = 0
				if LCFRSItem_fused is SmallChartItem:
					a = nextset(newitem.vec, b)
				elif LCFRSItem_fused is FatChartItem:
					a = anextset(newitem.vec, b, SLOTS)
				while a != -1:
					if LCFRSItem_fused is SmallChartItem:
						b = nextunset(newitem.vec, a)
						# given a=3, b=6, make bitvector: 1000000 - 1000 = 111000
						COMPONENT.vec = (1ULL << b) - (1ULL << a)
					elif LCFRSItem_fused is FatChartItem:
						b = anextunset(newitem.vec, a, SLOTS)
						# given a=3, b=6, make bitvector: 1000000 - 1000 = 111000
						memset(<void *>FATCOMPONENT.vec, 0, SLOTS * sizeof(ULong))
						for n in range(a, b):
							SETBIT(FATCOMPONENT.vec, n)
					if markorigin:
						componentdict = <dict>(componentlist[cnt])
					if LCFRSItem_fused is SmallChartItem:
						if PyDict_Contains(componentdict, COMPONENT) != 1:
							return False
						a = nextset(newitem.vec, b)
					elif LCFRSItem_fused is FatChartItem:
						if PyDict_Contains(componentdict, FATCOMPONENT) != 1:
							return False
						a = anextset(newitem.vec, b, SLOTS)
					cnt += 1
			else:
				label = newitem.label
				newitem.label = 0
				if PyDict_Contains(whitelist[label], newitem) != 1:
					return False
				newitem.label = label
		# haven't seen this item before, won't prune, add to agenda
		agenda.setitem(newitem, score)
	# in agenda (maybe in chart)
	elif inagenda:
		# lower score? => decrease-key in agenda
		agenda.setifbetter(newitem, score)
	# not in agenda => must be in chart
	elif not inagenda and newitem.prob < chart.subtreeprob(newitem):
		# re-add to agenda because we found a better score.
		agenda.setitem(newitem, score)
		if estimatetype != SXlrgaps:
			# This should only happen because of an inconsistent or
			# non-monotonic estimate.
			logging.warning('WARN: re-adding item to agenda already in chart:'
					' %r', newitem)
	# store this edge, regardless of whether the item was new (unary chains)
	chart.addedge(newitem, left, rule)
	return True


cdef inline int process_lexedge(LCFRSItem_fused newitem,
		double score, short wordidx, DoubleAgenda agenda,
		LCFRSChart_fused chart, list whitelist) except -1:
	"""Decide whether to accept a lexical edge ``(POS, word)``.

	:returns: ``True`` when edge is accepted in the chart, ``False`` when
		blocked. When ``False``, ``newitem`` may be reused."""
	cdef UInt label
	cdef bint inagenda = newitem in agenda.mapping
	cdef bint inchart = newitem in chart.parseforest
	if inagenda:
		raise ValueError('lexical edge already in agenda: %s' %
				chart.itemstr(newitem))
	elif inchart:
		raise ValueError('lexical edge already in chart: %s' %
				chart.itemstr(newitem))
	# check if we need to prune this item
	elif whitelist is not None and whitelist[newitem.label] is not None:
		label = newitem.label
		newitem.label = 0
		if PyDict_Contains(whitelist[label], newitem) != 1:
			return False
		newitem.label = label
	# haven't seen this item before, won't prune, add to agenda
	agenda.setitem(newitem, score)
	chart.addlexedge(newitem, wordidx)
	return True


cdef inline void combine_item(LCFRSItem_fused newitem,
		LCFRSItem_fused left, LCFRSItem_fused right):
	if LCFRSItem_fused is SmallChartItem:
		newitem.vec = left.vec ^ right.vec
	elif LCFRSItem_fused is FatChartItem:
		setunion(newitem.vec, left.vec, right.vec, SLOTS)


cdef inline bint concat(Rule *rule,
		LCFRSItem_fused left, LCFRSItem_fused right):
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
	cdef ULLong lvec, rvec, mask
	cdef ULong *alvec
	cdef ULong *arvec
	cdef int lpos, rpos, n
	if LCFRSItem_fused is SmallChartItem:
		lvec = left.vec
		rvec = right.vec
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
		alvec = left.vec
		arvec = right.vec
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


def parse_symbolic(sent, Grammar grammar, tags=None, start=None):
	"""Like parse(), but only compute parse forest, disregard probabilities.

	The agenda is a O(1) queue instead of a O(log n) priority queue."""
	if len(sent) < sizeof(COMPONENT.vec) * 8:
		chart = SmallLCFRSChart(grammar, list(sent))
		return parse_symbolic_main(<SmallLCFRSChart>chart,
				<SmallChartItem>chart.root(), sent, grammar, tags, start)
	chart = FatLCFRSChart(grammar, list(sent))
	return parse_symbolic_main(<FatLCFRSChart>chart,
			<FatChartItem>chart.root(), sent, grammar, tags, start)


def parse_symbolic_main(LCFRSChart_fused chart, LCFRSItem_fused goal,
		sent, Grammar grammar, tags=None, start=None):
	cdef:
		object agenda = deque()  # the agenda
		list items = [deque() for _ in grammar.toid]  # items for each label
		Rule *rule
		LexicalRule lexrule
		LCFRSItem_fused item, sibling, newitem
		UInt i, blocked = 0, maxA = 0
	if start is None:
		start = grammar.toid[grammar.start]
	if LCFRSItem_fused is SmallChartItem:
		newitem = new_SmallChartItem(0, 0)
	elif LCFRSItem_fused is FatChartItem:
		newitem = new_FatChartItem(0)
	for i, word in enumerate(sent):
		recognized = False
		tag = tags[i].encode('ascii') if tags else None
		if LCFRSItem_fused is SmallChartItem:
			item = new_SmallChartItem(0, i)
		elif LCFRSItem_fused is FatChartItem:
			item = new_FatChartItem(0)
			item.vec[0] = i
		for lexrule in grammar.lexicalbyword.get(word, ()):
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if (not tags or grammar.tolabel[lexrule.lhs] == tag
				or grammar.tolabel[lexrule.lhs].startswith(tag + b'@')):
				newitem.label = lexrule.lhs
				if LCFRSItem_fused is SmallChartItem:
					newitem.vec = 1ULL << i
				elif LCFRSItem_fused is FatChartItem:
					SETBIT(newitem.vec, i)
				agenda.append(newitem)
				chart.addedge(newitem, item, NULL)
				items[newitem.label].append(newitem)
				if LCFRSItem_fused is SmallChartItem:
					newitem = <LCFRSItem_fused>SmallChartItem.__new__(
							SmallChartItem)
				elif LCFRSItem_fused is FatChartItem:
					newitem = <LCFRSItem_fused>FatChartItem.__new__(
							FatChartItem)
		if not recognized and tags and tag in grammar.toid:
			newitem.label = grammar.toid[tag]
			if LCFRSItem_fused is SmallChartItem:
				newitem.vec = 1ULL << i
			elif LCFRSItem_fused is FatChartItem:
				SETBIT(newitem.vec, i)
			agenda.append(newitem)
			chart.addedge(newitem, item, NULL)
			items[newitem.label].append(newitem)
			newitem = SmallChartItem.__new__(SmallChartItem)
			recognized = True
		elif not recognized:
			return chart, 'not covered: %r' % (tag or word, )
	while agenda:
		item = agenda.pop()
		if item in chart.parseforest:
			continue
		items[item.label].append(item)
		if not equalitems(item, goal):
			for i in range(grammar.numunary):  # unary
				rule = &(grammar.unary[item.label][i])
				if rule.rhs1 != item.label:
					break
				newitem.label = rule.lhs
				if LCFRSItem_fused is SmallChartItem:
					newitem.vec = item.vec
				elif LCFRSItem_fused is FatChartItem:
					memcpy(<void *>newitem.vec, <void *>item.vec,
							SLOTS * sizeof(ULong))
				chart.addedge(newitem, item, rule)
				if newitem not in chart.parseforest:
					agenda.append(newitem)
					if LCFRSItem_fused is SmallChartItem:
						newitem = <LCFRSItem_fused>SmallChartItem.__new__(
								SmallChartItem)
					elif LCFRSItem_fused is FatChartItem:
						newitem = <LCFRSItem_fused>FatChartItem.__new__(
								FatChartItem)
			for i in range(grammar.numbinary):  # binary, item on right
				rule = &(grammar.rbinary[item.label][i])
				if rule.rhs2 != item.label:
					break
				for sib in items[rule.rhs1]:
					sibling = <LCFRSItem_fused>sib
					if concat(rule, sibling, item):
						newitem.label = rule.lhs
						combine_item(newitem, sibling, item)
						chart.addedge(newitem, sibling, rule)
						if newitem not in chart.parseforest:
							agenda.append(newitem)
							if LCFRSItem_fused is SmallChartItem:
								newitem = (<LCFRSItem_fused>
										SmallChartItem.__new__(SmallChartItem))
							elif LCFRSItem_fused is FatChartItem:
								newitem = (<LCFRSItem_fused>
										FatChartItem.__new__(FatChartItem))
			for i in range(grammar.numbinary):  # binary, item on left
				rule = &(grammar.lbinary[item.label][i])
				if rule.rhs1 != item.label:
					break
				for sib in items[rule.rhs2]:
					sibling = <LCFRSItem_fused>sib
					if concat(rule, item, sibling):
						newitem.label = rule.lhs
						combine_item(newitem, item, sibling)
						chart.addedge(newitem, item, rule)
						if newitem not in chart.parseforest:
							agenda.append(newitem)
							if LCFRSItem_fused is SmallChartItem:
								newitem = (<LCFRSItem_fused>
										SmallChartItem.__new__(SmallChartItem))
							elif LCFRSItem_fused is FatChartItem:
								newitem = (<LCFRSItem_fused>
										FatChartItem.__new__(FatChartItem))
		if agenda.length > maxA:
			maxA = agenda.length
	msg = ('agenda max %d, now %d, %s, blocked %d' % (
			maxA, len(agenda), chart.stats(), blocked))
	if goal not in chart:
		msg = 'no parse ' + msg
	return chart, msg

# def newparser(sent, Grammar grammar, tags=None, start=1,
# 		bint exhaustive=True, list whitelist=None, bint splitprune=False,
# 		bint markorigin=False, estimates=None, int beamwidth=0):
# 	# assign POS tags, unaries on POS tags
# 	for length in range(2, len(sent) + 1):
# 		for leftlength in range(1, length):
# 			for left in chart[leftlength]
# 				for right in chart[length - leftlength]:
# 					# find rules with ... => left right
# 					# can left + right concatenate?
#
# cdef inline set candidateitems(Rule rule, ChartItem item, dict chart,
# 		bint left=True):
# 	"""Find all compatible siblings for an item given a rule.
#
# 	Uses lookup tables with set of items for every start/end point of
# 	components in items."""
# 	cdef size_t n
# 	cdef short pos = nextset(item.vec, 0), prevpos, x, y
# 	cdef set candidates = None, temp
# 	for n in range(bitlength(rule.lengths)):
# 		if (testbit(rule.args, n) == 0) == left: # the given item
# 			if pos == -1:
# 				return False
# 			prevpos = nextunset(item.vec, pos)
# 			pos = nextset(item.vec, prevpos)
# 		else: # the other item for which to find candidates
# 			if n and testbit(rule.lengths, n - 1): # start gap?
# 				temp = set()
# 				if testbit(rule.lengths, n): # & end gap?
# 					for x in range(prevpos + 1, pos):
# 						for y in range(x + 1, pos):
# 							temp.update(chart.bystart[x] & chart.byend[y])
# 				else:
# 					for x in range(prevpos + 1, pos):
# 						temp.update(chart.bystart[x] & chart.byend[pos])
# 				if candidates is None:
# 					candidates = set(temp)
# 				else:
# 					candidates &= temp
# 			elif testbit(rule.lengths, n): # end gap?
# 				temp = set()
# 				for x in range(prevpos + 1, pos):
# 					temp.update(chart.bystart[prevpos] & chart.byend[x])
# 				if candidates is None:
# 					candidates = set(temp)
# 				else:
# 					candidates &= temp
# 			else: # no gaps
# 				if candidates is None:
# 					candidates = chart.bystart[prevpos] & chart.byend[pos]
# 				else:
# 					candidates &= chart.bystart[prevpos]
# 					candidates &= chart.byend[pos]
# 		if not candidates:
# 			break
# 	return candidates if pos == -1 else set()


def do(sent, grammar):
	"""Parse sentence with grammar and print 10 best derivations."""
	from math import exp
	from kbest import lazykbest
	print('len', len(sent.split()), 'sentence:', sent)
	sent = sent.split()
	chart, _ = parse(sent, grammar)
	if len(sent) < 10:
		print(chart)
	if chart:
		print('10 best parse trees:')
		for a, p in lazykbest(chart, 10)[0]:
			print(exp(-p), a)
		print()
		return True
	print('no parse')
	return False


def test():
	grammar = Grammar([
		((('S', 'VP_2', 'VMFIN'), ((0, 1, 0), )), 1),
		((('VP_2', 'VP_2', 'VAINF'), ((0, ), (0, 1))), 0.5),
		((('VP_2', 'PROAV', 'VVPP'), ((0, ), (1, ))), 0.5),
		((('PROAV', 'Epsilon'), ('Daruber', )), 1),
		((('VAINF', 'Epsilon'), ('werden', )), 1),
		((('VMFIN', 'Epsilon'), ('muss', )), 1),
		((('VVPP', 'Epsilon'), ('nachgedacht', )), 1)], start='S')
	print(grammar)
	assert do('Daruber muss nachgedacht werden', grammar)
	assert do('Daruber muss nachgedacht werden werden werden', grammar)
	print('ungrammatical sentence (\'no parse\' expected):')
	assert not do('muss Daruber nachgedacht werden', grammar)
	assert do('Daruber muss nachgedacht ' + ' '.join(32 * ['werden']), grammar)
	assert do('Daruber muss nachgedacht ' + ' '.join(64 * ['werden']), grammar)

__all__ = ['Agenda', 'DoubleAgenda', 'FatLCFRSChart', 'LCFRSChart',
		'SmallLCFRSChart', 'getparent', 'merge', 'parse', 'parse_symbolic']
