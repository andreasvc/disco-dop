""" Probabilistic CKY parser for monotone, string-rewriting
Linear Context-Free Rewriting Systems. """
from __future__ import print_function
from math import exp
from collections import defaultdict, deque
import logging
import numpy as np
cimport cython
include "constants.pxi"

DEF SX = 1
DEF SXlrgaps = 2

cdef SmallChartItem COMPONENT = new_SmallChartItem(0, 0)
cdef SmallChartItem NONE = new_SmallChartItem(0, 0)
cdef FatChartItem FATNONE = new_FatChartItem(0)
cdef FatChartItem FATCOMPONENT = new_FatChartItem(0)
cdef double INFINITY = float('infinity')

include "agenda.pxi"

cdef class LCFRSChart(Chart):
	""" item is a ChartItem object. """
	def __init__(self, Grammar grammar, list sent,
			start=None, logprob=True, viterbi=True):
		self.grammar = grammar
		self.sent = sent
		self.lensent = len(sent)
		self.start = grammar.toid[grammar.start if start is None else start]
		self.logprob = logprob
		self.viterbi = viterbi
		self.probs = [{} for _ in range(grammar.nonterminals)]
		self.parseforest = {}
		self.itemsinorder = []

	cdef void addlexedge(self, item, short wordidx):
		""" Add lexical edge. """
		cdef Edges edges
		cdef Edge *edge
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
		edge = &(edges.data[edges.len])
		edge.rule = NULL
		edge.pos.mid = wordidx + 1
		edges.len += 1

	cdef void updateprob(self, ChartItem item, double prob):
		cdef dict probs = <dict>self.probs[item.label]
		if item not in probs or prob < PyFloat_AS_DOUBLE(probs[item]):
			probs[item] = prob

	cdef void addprob(self, ChartItem item, double prob):
		cdef dict probs = <dict>self.probs[item.label]
		probs[item] += prob

	cdef double _subtreeprob(self, ChartItem item):
		cdef dict probs = <dict>self.probs[item.label]
		return PyFloat_AS_DOUBLE(probs[item]) if item in probs else INFINITY

	cdef double subtreeprob(self, item):
		return self._subtreeprob(<ChartItem>item)

	def label(self, item):
		return (<ChartItem>item).label

	def itemstr(self, item):
		return '%s[%s]' % (self.grammar.tolabel[self.label(item)
			].decode('ascii'), item.binrepr(self.lensent))


cdef class SmallLCFRSChart(LCFRSChart):
	def __init__(self, Grammar grammar, list sent,
			start=None, logprob=True, viterbi=True):
		super(SmallLCFRSChart, self).__init__(
				grammar, sent, start, logprob, viterbi)
		self.tmpleft = new_SmallChartItem(0, 0)
		self.tmpright = new_SmallChartItem(0, 0)

	cdef void addedge(self, SmallChartItem item, SmallChartItem left,
			Rule *rule):
		""" Add new edge. """
		cdef Edges edges
		cdef Edge *edge
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
		edge = &(edges.data[edges.len])
		edge.rule = rule
		edge.pos.lvec = left.vec
		edges.len += 1

	cdef _left(self, item, Edge *edge):
		if edge.rule is NULL:
			return None
		self.tmpleft.label = edge.rule.rhs1
		self.tmpleft.vec = edge.pos.lvec
		return self.tmpleft  # we don't even need to return this ...

	cdef _right(self, item, Edge *edge):
		if edge.rule is NULL or edge.rule.rhs2 == 0:
			return None
		self.tmpright.label = edge.rule.rhs2
		self.tmpright.vec = (<SmallChartItem>item).vec ^ edge.pos.lvec
		return self.tmpright

	cdef copy(self, item):
		return (<SmallChartItem>item).copy()

	def root(self):
		return new_SmallChartItem(
				self.grammar.toid[self.grammar.start],
				(1ULL << self.lensent) - 1)


cdef class FatLCFRSChart(LCFRSChart):
	def __init__(self, Grammar grammar, list sent,
			start=None, logprob=True, viterbi=True):
		super(FatLCFRSChart, self).__init__(
				grammar, sent, start, logprob, viterbi)
		self.tmpleft = new_FatChartItem(0)
		self.tmpright = new_FatChartItem(0)

	cdef void addedge(self, FatChartItem item, FatChartItem left, Rule *rule):
		""" Add new edge and update viterbi probability. """
		cdef Edges edges
		cdef Edge *edge
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
		edge = &(edges.data[edges.len])
		edge.rule = rule
		# NB: store pointer; breaks when `left` is garbage collected!
		edge.pos.lvec_fat = left.vec
		edges.len += 1

	cdef _left(self, item, Edge *edge):
		cdef size_t n
		if edge.rule is NULL:
			return None
		self.tmpleft.label = edge.rule.rhs1
		for n in range(SLOTS):
			self.tmpleft.vec[n] = edge.pos.lvec_fat[n]
		return self.tmpleft

	cdef _right(self, item, Edge *edge):
		cdef size_t n
		if edge.rule is NULL:
			return None
		self.tmpright.label = edge.rule.rhs2
		for n in range(SLOTS):
			self.tmpright.vec[n] = (
					<FatChartItem>item).vec[n] ^ edge.pos.lvec_fat[n]
		return self.tmpright

	cdef copy(self, item):
		return (<FatChartItem>item).copy()

	def root(self):
		return CFGtoFatChartItem(
				self.grammar.toid[self.grammar.start], 0, self.lensent)


def parse(sent, Grammar grammar, tags=None, bint exhaustive=True,
		start=None, list whitelist=None, bint splitprune=False,
		bint markorigin=False, estimates=None, int beamwidth=0):
	""" Parse sentence, a list of tokens, optionally with gold tags, and
	produce a chart, either exhaustive or up until the viterbi parse.

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
		setting to 0 disables this feature. experimental.
	"""
	if len(sent) < sizeof(COMPONENT.vec) * 8:
		chart = SmallLCFRSChart(grammar, list(sent), start)
		return parse_main(<SmallLCFRSChart>chart, <SmallChartItem>chart.root(),
				sent, grammar, tags, exhaustive, start, whitelist, splitprune,
				markorigin, estimates, beamwidth)
	else:
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
		Entry entry
		double [:, :, :, :] outside = None  # outside estimates, if provided
		double itemprob, newitemprob, siblingprob, score
		short wordidx, lensent = len(sent), estimatetype = 0
		int length = 0, left = 0, right = 0, gaps = 0
		UInt lhs
		size_t blocked = 0, maxA = 0, n
		bint recognized
		#unicode word
		#bytes tag
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
			length = 1
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
					if score > 300.0:
						continue
				elif estimatetype == SXlrgaps:
					score += outside[
							lexrule.lhs, length, left + right, gaps]
					if score > 300.0:
						continue
				newitem.label = lexrule.lhs
				if LCFRSItem_fused is SmallChartItem:
					newitem.vec = 1ULL << wordidx
				elif LCFRSItem_fused is FatChartItem:
					ulongset(newitem.vec, 0UL, SLOTS)
					SETBIT(newitem.vec, wordidx)
				if process_lexedge(newitem, score, lexrule.prob, wordidx,
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
				if score > 300.0:
					continue
			elif estimatetype == SXlrgaps:
				score += outside[lhs, length, left + right, gaps]
				if score > 300.0:
					continue
			newitem.label = lhs
			if LCFRSItem_fused is SmallChartItem:
				newitem.vec = 1ULL << wordidx
			elif LCFRSItem_fused is FatChartItem:
				ulongset(newitem.vec, 0UL, SLOTS)
				SETBIT(newitem.vec, wordidx)
			# prevent pruning of provided tags => whitelist == None
			if process_lexedge(newitem, 0.0, 0.0, wordidx,
					agenda, chart, None):
				if LCFRSItem_fused is SmallChartItem:
					newitem = <LCFRSItem_fused>SmallChartItem.__new__(
							SmallChartItem)
				elif LCFRSItem_fused is FatChartItem:
					newitem = <LCFRSItem_fused>FatChartItem.__new__(
							FatChartItem)
				recognized = True
			else:
				raise ValueError
		elif not recognized:
			if tag is None and word not in grammar.lexicalbyword:
				return chart, 'no parse: %r not in lexicon' % word
			elif tag is not None and tag not in grammar.toid:
				return chart, 'no parse: unknown tag %r' % tag
			elif whitelist is not None:
				return chart, 'no parse: all tags for %r blocked' % word
			raise ValueError
	while agenda.length:  # main parsing loop
		entry = agenda.popentry()
		item = <LCFRSItem_fused>entry.key
		itemprob = PyFloat_AS_DOUBLE(entry.value)
		if estimates is not None:
			if LCFRSItem_fused is SmallChartItem:
				length = bitcount(item.vec)
				left = nextset(item.vec, 0)
				gaps = bitlength(item.vec) - length - left
				right = lensent - length - left - gaps
			elif LCFRSItem_fused is FatChartItem:
				length = abitcount(item.vec, SLOTS)
				left = anextset(item.vec, 0, SLOTS)
				gaps = abitlength(item.vec, SLOTS) - length - left
				right = lensent - length - left - gaps
			if estimatetype == SX:
				itemprob -= outside[item.label, left, right, 0]
			elif estimatetype == SXlrgaps:
				itemprob -= outside[item.label, length, left + right, gaps]
		chart.updateprob(item, itemprob)
		if item.label == goal.label and item == goal:
			if not exhaustive:
				break
		else:
			# unary
			if estimates is not None:
				if LCFRSItem_fused is SmallChartItem:
					length = bitcount(item.vec)
					left = nextset(item.vec, 0)
					gaps = bitlength(item.vec) - length - left
					right = lensent - length - left - gaps
				elif LCFRSItem_fused is FatChartItem:
					length = abitcount(item.vec, SLOTS)
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
			if LCFRSItem_fused is SmallChartItem:
				newitem.vec = item.vec
			elif LCFRSItem_fused is FatChartItem:
				ulongcpy(newitem.vec, item.vec, SLOTS)
			for n in range(grammar.numrules):
				rule = &(grammar.unary[item.label][n])
				if rule.rhs1 != item.label:
					break
				score = newitemprob = itemprob + rule.prob
				if estimatetype == SX:
					score += outside[rule.lhs, left, right, 0]
					if score > 300.0:
						continue
				elif estimatetype == SXlrgaps:
					score += outside[
							rule.lhs, length, left + right, gaps]
					if score > 300.0:
						continue
				newitem.label = rule.lhs
				if process_edge(newitem, score, newitemprob,
						rule, item, agenda, chart, whitelist,
						splitprune and grammar.fanout[rule.lhs] != 1,
						markorigin):
					if LCFRSItem_fused is SmallChartItem:
						newitem = <LCFRSItem_fused>SmallChartItem.__new__(
								SmallChartItem)
						newitem.vec = item.vec
					elif LCFRSItem_fused is FatChartItem:
						newitem = <LCFRSItem_fused>FatChartItem.__new__(
								FatChartItem)
						ulongcpy(newitem.vec, item.vec, SLOTS)
				else:
					blocked += 1
			# binary production, child is on the right
			for n in range(grammar.numrules):
				rule = &(grammar.rbinary[item.label][n])
				if rule.rhs2 != item.label:
					break
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
						siblingprob = PyFloat_AS_DOUBLE(
								(<dict>probs[rule.rhs1])[sib])
						score = newitemprob = itemprob + siblingprob + rule.prob
						if estimatetype == SX or estimatetype == SXlrgaps:
							if LCFRSItem_fused is SmallChartItem:
								length = bitcount(newitem.vec)
								left = nextset(newitem.vec, 0)
							elif LCFRSItem_fused is FatChartItem:
								length = abitcount(newitem.vec, SLOTS)
								left = anextset(newitem.vec, 0, SLOTS)
						if estimatetype == SX:
							right = lensent - length - left
							score += outside[rule.lhs, left, right, 0]
							if score > 300.0:
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
							if score > 300.0:
								continue
						if process_edge(newitem, score, newitemprob,
								rule, <LCFRSItem_fused>sib, agenda, chart,
								whitelist,
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
			# binary production, child is on the left
			for n in range(grammar.numrules):
				rule = &(grammar.lbinary[item.label][n])
				if rule.rhs1 != item.label:
					break
				#for sib, sibprob in (<dict>probs[rule.rhs2]).iteritems():
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
						siblingprob = PyFloat_AS_DOUBLE(
								(<dict>probs[rule.rhs2])[sib])
						score = newitemprob = itemprob + siblingprob + rule.prob
						if estimatetype == SX or estimatetype == SXlrgaps:
							if LCFRSItem_fused is SmallChartItem:
								length = bitcount(newitem.vec)
								left = nextset(newitem.vec, 0)
							elif LCFRSItem_fused is FatChartItem:
								length = abitcount(newitem.vec, SLOTS)
								left = anextset(newitem.vec, 0, SLOTS)
						if estimatetype == SX:
							right = lensent - length - left
							score += outside[rule.lhs, left, right, 0]
							if score > 300.0:
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
							if score > 300.0:
								continue
						if process_edge(newitem, score, newitemprob,
								rule, item, agenda, chart, whitelist,
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
		double score, double prob, Rule *rule, LCFRSItem_fused left,
		DoubleAgenda agenda, LCFRSChart_fused chart, list whitelist,
		bint splitprune, bint markorigin):
	""" Decide what to do with a newly derived edge.
	:returns: ``True`` when edge is accepted in the chart, ``False`` when
		blocked. When ``False``, ``newitem`` may be reused. """
	cdef UInt a, b, n, cnt, label
	cdef bint inagenda = agenda.contains(newitem)
	cdef bint inchart = newitem in chart.parseforest
	cdef list componentlist = None
	cdef dict componentdict = None
	if not inagenda and not inchart:
		#if score > 300.0:
		#	return False
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
						#given a=3, b=6, make bitvector: 1000000 - 1000 = 111000
						ulongset(FATCOMPONENT.vec, 0UL, SLOTS)
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
		chart.addedge(newitem, left, rule)
	# in agenda (maybe in chart)
	elif inagenda:
		# lower score? => decrease-key in agenda
		agenda.setifbetter(newitem, score)
		chart.addedge(newitem, left, rule)

	# not in agenda => must be in chart
	elif not inagenda and prob < chart._subtreeprob(newitem):
		#re-add to agenda because we found a better score.
		#should not happen without estimates!
		agenda.setitem(newitem, score)
		chart.addedge(newitem, left, rule)
		logging.warning('WARN: updating score in agenda: %r', newitem)
	return True


cdef inline bint process_lexedge(LCFRSItem_fused newitem,
		double score, double prob, short wordidx,
		DoubleAgenda agenda, LCFRSChart_fused chart, list whitelist):
	""" Decide whether to accept a lexical edge (POS, word), which is assumed
	not to be discontinuous.
	:returns: ``True`` when edge is accepted in the chart, ``False`` when
		blocked. When ``False``, ``newitem`` may be reused. """
	cdef UInt label
	cdef bint inagenda = agenda.contains(newitem)
	cdef bint inchart = newitem in chart.parseforest
	if inagenda:
		raise ValueError('lexical edge already in agenda: %s' %
				chart.itemstr(newitem))
	elif inchart:
		raise ValueError('lexical edge already in chart: %s' %
				chart.itemstr(newitem))
	else:
		#if score > 300.0:
		#	return False
		# check if we need to prune this item
		if whitelist is not None and whitelist[newitem.label] is not None:
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
	""" Determine the compatibility of two bitvectors according to the given
	yield function. Ranges should be non-overlapping, continuous when they are
	concatenated, and adhere to the ordering in the yield function.
	The yield function indicates for each span whether it should come from the
	left or right non-terminal (0 meaning left and 1 right), and whether it is
	contiguous with the previous span.

	>>> lvec = 0b0011; rvec = 0b1000
	>>> concat(((0, ), (1, )), lvec, rvec)
	True		# discontinuous, non-overlapping, linearly ordered.
	>>> concat(((0, 1), ), lvec, rvec)
	False		# lvec and rvec are not contiguous
	>>> concat(((1, ), (0, )), lvec, rvec)
	False		# rvec's span should come after lvec's span

	The actual yield functions are encoded in a binary format;
		cf. containers.pyx
		((0, 1, 0), (1, 0)) ==> args=0b10010; lengths=0b00101
		NB: note reversal due to the way binary numbers are represented
		the least significant bit (rightmost) corresponds to the lowest
		index in the sentence / constituent (leftmost). """
	cdef ULLong lvec, rvec, mask
	cdef ULong *alvec, *arvec
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
				mask = rvec & (~rvec - 1)  # everything before first 0 bit => 1 bits
			else:  # component from left vector
				if lvec & mask == 0:
					return False  # check for expected component
				lvec |= lvec - 1  # trailing 0 bits => 1 bits
				mask = lvec & (~lvec - 1)  # everything before first 0 bit => 1 bits
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
		#this algorithm was adapted from rparse, FastYFComposer.
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
	""" Like parse(), but only compute parse forest, disregard probabilities.
	The agenda is a O(1) queue instead of a O(log n) priority queue. """
	if len(sent) < sizeof(COMPONENT.vec) * 8:
		chart = SmallLCFRSChart(grammar, list(sent))
		return parse_symbolic_main(<SmallLCFRSChart>chart,
				<SmallChartItem>chart.root(), sent, grammar, tags, start)
	else:
		chart = FatLCFRSChart(grammar, list(sent))
		return parse_symbolic_main(<FatLCFRSChart>chart,
				<FatChartItem>chart.root(), sent, grammar, tags, start)


def parse_symbolic_main(LCFRSChart_fused chart, LCFRSItem_fused goal,
		sent, Grammar grammar, tags=None, start=None):
	cdef:
		list items = [deque() for _ in grammar.toid]  # items for each label
		object agenda = deque()  # the agenda
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
		chart.itemsinorder.append(item)
		if item.label == goal.label and item == goal:
			pass
		else:
			# unary
			for i in range(grammar.numrules):
				rule = &(grammar.unary[item.label][i])
				if rule.rhs1 != item.label:
					break
				newitem.label = rule.lhs
				if LCFRSItem_fused is SmallChartItem:
					newitem.vec = item.vec
				elif LCFRSItem_fused is FatChartItem:
					ulongcpy(newitem.vec, item.vec, SLOTS)
				chart.addedge(newitem, item, rule)
				if newitem not in chart.parseforest:
					agenda.append(newitem)
					if LCFRSItem_fused is SmallChartItem:
						newitem = <LCFRSItem_fused>SmallChartItem.__new__(
								SmallChartItem)
					elif LCFRSItem_fused is FatChartItem:
						newitem = <LCFRSItem_fused>FatChartItem.__new__(
								FatChartItem)
			# binary right
			for i in range(grammar.numrules):
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
								newitem = <LCFRSItem_fused>SmallChartItem.__new__(
										SmallChartItem)
							elif LCFRSItem_fused is FatChartItem:
								newitem = <LCFRSItem_fused>FatChartItem.__new__(
										FatChartItem)
			# binary left
			for i in range(grammar.numrules):
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
								newitem = <LCFRSItem_fused>SmallChartItem.__new__(
										SmallChartItem)
							elif LCFRSItem_fused is FatChartItem:
								newitem = <LCFRSItem_fused>FatChartItem.__new__(
										FatChartItem)
		if agenda.length > maxA:
			maxA = agenda.length
	msg = ('agenda max %d, now %d, %s, blocked %d' % (
			maxA, len(agenda), chart.stats(), blocked))
	if goal not in chart:
		msg = 'no parse ' + msg
	return chart, msg


def doinsideoutside(Chart chart):
	""" Given a chart containing a parse forest, compute inside and outside
	probabilities. Results will be stored in the chart. """
	assert not chart.grammar.logprob, 'grammar must not be in logprob mode.'
	getinside(chart)
	getoutside(chart)


def getinside(Chart chart):
	cdef size_t n
	cdef Edges edges
	cdef Edge *edge
	# this needs to be bottom up, so need order in which items were added
	# currently separate list, maintained only by parse_symbolic
	# NB: sorting items by length is not enough,
	# unaries have to be in the right order...

	# choices for probs:
	# - normal => underflow (current)
	# - logprobs => loss of precision
	# - normal, scaled => how?

	# packing parse forest:
	# revitems = {item: n for n, item in self.itemsinorder}
	# now self.inside[n] and self.outside[n] can be double arrays.
	#chart.inside = <dict>defaultdict(float)

	# traverse items in bottom-up order
	for item in chart.itemsinorder:
		for edges in chart.getedges(item):
			for n in range(edges.len):
				edge = &(edges.data[n])
				if edge.rule is NULL:
					label = chart.label(item)
					word = chart.sent[chart.lexidx(item, edge)]
					prob = (<LexicalRule>chart.grammar.lexicalbylhs[
							label][word]).prob
				elif edge.rule.rhs2 == 0:
					leftitem = chart._left(item, edge)
					prob = (edge.rule.prob
							* chart._subtreeprob(leftitem))
				else:
					leftitem = chart._left(item, edge)
					rightitem = chart._right(item, edge)
					prob = (edge.rule.prob
							* chart._subtreeprob(leftitem)
							* chart._subtreeprob(rightitem))
				chart.addprob(item, prob)


def getoutside(Chart chart):
	cdef size_t n
	cdef Edges edges
	cdef Edge *edge
	#cdef double sentprob = chart._subtreeprob(chart.root())
	# traverse items in top-down order
	# could use list with idx of item in itemsinorder
	chart.outside = {chart.root(): 1.0}
	for item in reversed(chart.itemsinorder):
		# can we define outside[item] simply as sentprob - inside[item] ?
		# chart.outside[item] = sentprob - chart._subtreeprob(item)
		for edges in chart.getedges(item):
			for n in range(edges.len):
				edge = &(edges.data[n])
				if edge.rule is NULL:
					pass
				elif edge.rule.rhs2 == 0:
					leftitem = chart._left(item, edge)
					chart.outside[leftitem] += (edge.rule.prob
							* chart.outside[item])
				else:
					leftitem = chart._left(item, edge)
					rightitem = chart._right(item, edge)
					chart.outside[leftitem] += (edge.rule.prob
							* chart._subtreeprob(rightitem)
							* chart.outside[item])
					chart.outside[rightitem] += (edge.rule.prob
							* chart._subtreeprob(leftitem)
							* chart.outside[item])


#def newparser(sent, Grammar grammar, tags=None, start=1, bint exhaustive=True,
#		list whitelist=None, bint splitprune=False, bint markorigin=False,
#		estimates=None, int beamwidth=0):
#	# assign POS tags, unaries on POS tags
#	# parse
#	for length in range(2, len(sent) + 1):
#		for leftlength in range(1, length):
#			for left in chart[leftlength]
#				for right in chart[length - leftlength]:
#					# find rules with ... => left right
#					# can left + right concatenate?
#
#cdef inline set candidateitems(Rule rule, ChartItem item, dict chart,
#		bint left=True):
#	""" For a given rule and item, find all potential sibling items which are
#	compatible with them. Uses lookup tables with set of items for every
#	start/end point of components in items. """
#	cdef size_t n
#	cdef short pos = nextset(item.vec, 0), prevpos, x, y
#	cdef set candidates = None, temp
#	for n in range(bitlength(rule.lengths)):
#		if (testbit(rule.args, n) == 0) == left: # the given item
#			if pos == -1:
#				return False
#			prevpos = nextunset(item.vec, pos)
#			pos = nextset(item.vec, prevpos)
#		else: # the other item for which to find candidates
#			if n and testbit(rule.lengths, n - 1): # start gap?
#				temp = set()
#				if testbit(rule.lengths, n): # & end gap?
#					for x in range(prevpos + 1, pos):
#						for y in range(x + 1, pos):
#							temp.update(chart.bystart[x] & chart.byend[y])
#				else:
#					for x in range(prevpos + 1, pos):
#						temp.update(chart.bystart[x] & chart.byend[pos])
#				if candidates is None:
#					candidates = set(temp)
#				else:
#					candidates &= temp
#			elif testbit(rule.lengths, n): # end gap?
#				temp = set()
#				for x in range(prevpos + 1, pos):
#					temp.update(chart.bystart[prevpos] & chart.byend[x])
#				if candidates is None:
#					candidates = set(temp)
#				else:
#					candidates &= temp
#			else: # no gaps
#				if candidates is None:
#					candidates = chart.bystart[prevpos] & chart.byend[pos]
#				else:
#					candidates &= chart.bystart[prevpos]
#					candidates &= chart.byend[pos]
#		if not candidates:
#			break
#	return candidates if pos == -1 else set()


def do(sent, grammar):
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
	print(chart)
	print('no parse')
	return False


def test():
	cdef Grammar grammar
	testagenda()
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
	assert do('Daruber muss nachgedacht werden werden', grammar)
	assert do('Daruber muss nachgedacht werden werden werden', grammar)
	print('ungrammatical sentence:')
	assert not do('muss Daruber nachgedacht werden', grammar)  # no parse
	print('(as expected)\nlong sentence (%d words):' % 35)
	assert do('Daruber muss nachgedacht ' + ' '.join(32 * ['werden']), grammar)
	print('(as expected)\nlong sentence (%d words):' % 67)
	assert do('Daruber muss nachgedacht ' + ' '.join(64 * ['werden']), grammar)
