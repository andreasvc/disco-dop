""" Probabilistic CKY parser for monotone, string-rewriting
Linear Context-Free Rewriting Systems. """
from __future__ import print_function
# python imports
from math import log, exp
from collections import defaultdict, deque
import re
import logging
import sys
import numpy as np
from agenda import EdgeAgenda, Entry
# cython imports
from cpython cimport PyDict_Contains, PyDict_GetItem
cimport numpy as np
from agenda cimport Entry, EdgeAgenda
from containers cimport Grammar, Rule, LexicalRule, LCFRSEdge, ChartItem, \
	SmallChartItem, FatChartItem, new_LCFRSEdge, new_ChartItem, \
	new_FatChartItem, UChar, UInt, ULong, ULLong
from bit cimport nextset, nextunset, bitcount, bitlength, \
	testbit, testbitint, anextset, anextunset, abitcount, abitlength, \
	ulongset, ulongcpy, setunion
cdef extern from "math.h":
	bint isinf(double x)
	bint isfinite(double x)
cdef extern from "macros.h":
	int BITSIZE
	int BITSLOT(int b)
	ULong BITMASK(int b)
	int BITNSLOTS(int nb)
	void SETBIT(ULong a[], int b)
	ULong TESTBIT(ULong a[], int b)
np.import_array()
DEF SX = 1
DEF SXlrgaps = 2
DEF SLOTS = 3
cdef SmallChartItem COMPONENT = new_ChartItem(0, 0)
cdef SmallChartItem NONE = new_ChartItem(0, 0)
cdef FatChartItem FATNONE = new_FatChartItem(0)
cdef FatChartItem FATCOMPONENT = new_FatChartItem(0)


def parse(sent, Grammar grammar, tags=None, bint exhaustive=True, start=1,
		list whitelist=None, bint splitprune=False, bint markorigin=False,
		estimates=None, int beamwidth=0):
	""" parse sentence, a list of tokens, optionally with gold tags, and
	produce a chart, either exhaustive or up until the viterbi parse.
	Other parameters:
	- start: integer corresponding to the start symbol that analyses should
		have, e.g., grammar.toid[b'ROOT']
	- exhaustive: don't stop at viterbi parser, return a full chart
	- whitelist: a whitelist of allowed ChartItems. Anything else is not
		added to the agenda.
	- splitprune: coarse stage used a split-PCFG where discontinuous node
		appear as multiple CFG nodes. Every discontinuous node will result
		in multiple lookups into whitelist to see whether it should be
		allowed on the agenda.
	- markorigin: in combination with splitprune, coarse labels include an
		integer to distinguish components; e.g., CFG nodes NP*0 and NP*1
		map to the discontinuous node NP_2
	- estimates: use context-summary estimates (heuristics, figures of
		merit) to order agenda. should be a tuple with the kind of
		estimates ('SX' or 'SXlrgaps'), and the estimates themselves in a
		4-dimensional numpy matrix. If estimates are not consistent, it is
		no longer guaranteed that the optimal parse will be found.
		experimental.
	- beamwidth: specify the maximum number of items that will be explored
		for each particular span, on a first-come-first-served basis.
		setting to 0 disables this feature. experimental.
	"""
	cdef:
		dict beam = <dict>defaultdict(int)  # histogram of spans
		dict chart = {}  # the full chart
		list viterbi = [{} for _ in grammar.toid]  # the viterbi probabilities
		EdgeAgenda agenda = EdgeAgenda()  # the agenda
		Rule *rule
		LexicalRule lexrule
		Entry entry
		LCFRSEdge edge
		SmallChartItem item, sibling, newitem = new_ChartItem(0, 0)
		SmallChartItem goal = new_ChartItem(start, (1ULL << len(sent)) - 1)
		np.ndarray[np.double_t, ndim=4] outside
		double x = 0.0, y = 0.0, score, inside
		signed int length = 0, left = 0, right = 0, gaps = 0
		signed int lensent = len(sent), estimatetype = 0
		UInt i, blocked = 0, Epsilon = grammar.toid[b'Epsilon']
		ULong maxA = 0
		ULLong vec = 0

	if lensent >= sizeof(vec) * 8:
		return parse_longsent(sent, grammar, tags=tags, start=start,
			exhaustive=exhaustive, whitelist=whitelist, splitprune=splitprune,
			markorigin=markorigin, estimates=estimates)
	if estimates is not None:
		estimatetypestr, outside = estimates
		estimatetype = {"SX": SX, "SXlrgaps": SXlrgaps}[estimatetypestr]

	# scan
	for i, word in enumerate(sent):
		recognized = False
		item = new_ChartItem(Epsilon, i)
		tag = tags[i].encode('ascii') if tags else None
		if estimates is not None:
			length = 1
			left = i
			gaps = 0
			right = lensent - 1 - i
		for lexrule in grammar.lexical.get(word, ()):
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
					score += outside[lexrule.lhs, length, left + right, gaps]
					if score > 300.0:
						continue
				newitem.label = lexrule.lhs
				newitem.vec = 1ULL << i
				tmp = process_edge(newitem, score, lexrule.prob, NULL,
						item, NONE, agenda, chart, viterbi, grammar,
						exhaustive, whitelist, False, markorigin, &blocked)
				#check whether item has not been blocked
				recognized |= newitem is not tmp
				newitem = tmp
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
			tagitem = new_ChartItem(lhs, 1ULL << i)
			agenda[tagitem] = new_LCFRSEdge(score, 0.0, NULL, item, NONE)
			chart[tagitem] = {}
			recognized = True
		elif not recognized:
			return chart, NONE, "not covered: %r" % (tag or word)

	# parsing
	while agenda.length:
		entry = agenda.popentry()
		item = <SmallChartItem>entry.key
		edge = iscore(<LCFRSEdge>entry.value)
		chart[item][edge] = edge
		(<dict>viterbi[item.label])[item] = edge
		if item.label == goal.label and item.vec == goal.vec:
			if not exhaustive:
				break
		else:
			x = edge.inside

			# unary
			if estimates is not None:
				length = bitcount(item.vec)
				left = nextset(item.vec, 0)
				gaps = bitlength(item.vec) - length - left
				right = lensent - length - left - gaps
			if beamwidth:
				if beam[item.vec] > beamwidth:
					continue
				beam[item.vec] += 1
			for i in range(grammar.numrules):
				rule = &(grammar.unary[item.label][i])
				if rule.rhs1 != item.label:
					break
				score = inside = x + rule.prob
				if estimatetype == SX:
					score += outside[rule.lhs, left, right, 0]
					if score > 300.0:
						continue
				elif estimatetype == SXlrgaps:
					score += outside[rule.lhs, length, left + right, gaps]
					if score > 300.0:
						continue
				newitem.label = rule.lhs
				newitem.vec = item.vec
				newitem = process_edge(newitem, score, inside, rule,
						item, NONE, agenda, chart, viterbi, grammar,
						exhaustive, whitelist,
						splitprune and grammar.fanout[rule.lhs] != 1,
						markorigin, &blocked)

			# binary right
			for i in range(grammar.numrules):
				rule = &(grammar.rbinary[item.label][i])
				if rule.rhs2 != item.label:
					break
				for I, e in (<dict>viterbi[rule.rhs1]).items():
					sibling = <SmallChartItem>I
					if beamwidth:
						if beam[sibling.vec ^ item.vec] > beamwidth:
							continue
						beam[sibling.vec ^ item.vec] += 1
					if concat(rule, sibling.vec, item.vec):
						newitem.label = rule.lhs
						newitem.vec = sibling.vec ^ item.vec
						y = (<LCFRSEdge>e).inside
						score = inside = x + y + rule.prob
						if estimatetype == SX:
							length = bitcount(item.vec)
							left = nextset(item.vec, 0)
							right = lensent - length - left
							score += outside[rule.lhs, left, right, 0]
							if score > 300.0:
								continue
						elif estimatetype == SXlrgaps:
							length = bitcount(item.vec)
							left = nextset(item.vec, 0)
							gaps = bitlength(item.vec) - length - left
							right = lensent - length - left - gaps
							score += outside[rule.lhs, length, left + right, gaps]
							if score > 300.0:
								continue
						newitem = process_edge(newitem, score, inside, rule,
								sibling, item, agenda, chart, viterbi, grammar,
								exhaustive, whitelist,
								splitprune and grammar.fanout[rule.lhs] != 1,
								markorigin, &blocked)

			# binary left
			for i in range(grammar.numrules):
				rule = &(grammar.lbinary[item.label][i])
				if rule.rhs1 != item.label:
					break
				for I, e in (<dict>viterbi[rule.rhs2]).items():
					sibling = <SmallChartItem>I
					if beamwidth:
						if beam[item.vec ^ sibling.vec] > beamwidth:
							continue
						beam[item.vec ^ sibling.vec] += 1
					if concat(rule, item.vec, sibling.vec):
						newitem.label = rule.lhs
						newitem.vec = item.vec ^ sibling.vec
						y = (<LCFRSEdge>e).inside
						score = inside = x + y + rule.prob
						if estimatetype == SX:
							length = bitcount(newitem.vec)
							left = nextset(newitem.vec, 0)
							right = lensent - length - left
							score += outside[rule.lhs, left, right, 0]
							if score > 300.0:
								continue
						elif estimatetype == SXlrgaps:
							length = bitcount(newitem.vec)
							left = nextset(newitem.vec, 0)
							gaps = bitlength(newitem.vec) - length - left
							right = lensent - length - left - gaps
							score += outside[rule.lhs, length, left + right, gaps]
							if score > 300.0:
								continue
						newitem = process_edge(newitem, score, inside, rule,
								item, sibling, agenda, chart, viterbi, grammar,
								exhaustive, whitelist,
								splitprune and grammar.fanout[rule.lhs] != 1,
								markorigin, &blocked)

		if agenda.length > maxA:
			maxA = agenda.length
		#if agenda.length % 1000 == 0:
		#	logging.debug(("agenda max %d, now %d, items %d (%d labels), "
		#		"edges %d, blocked %d" % (maxA, len(agenda),
		#		len(filter(None, chart.values())), len(filter(None, viterbi)),
		#		sum(map(len, chart.values())), blocked)))
	msg = ("agenda max %d, now %d, items %d (%d labels), edges %d, blocked %d"
		% (maxA, len(agenda), len(list(filter(None, chart.values()))),
		len(list(filter(None, viterbi))), sum(map(len, chart.values())), blocked))
	if goal in chart:
		return chart, goal, msg
	else:
		return chart, NONE, "no parse " + msg


cdef inline SmallChartItem process_edge(SmallChartItem newitem, double score,
		double inside, Rule *rule, SmallChartItem left, SmallChartItem right,
		EdgeAgenda agenda, dict chart, list viterbi, Grammar grammar, bint
		exhaustive, list whitelist, bint splitprune, bint markorigin,
		UInt *blocked):
	""" Decide what to do with a newly derived edge. """
	cdef UInt a, b, cnt, label
	cdef bint inagenda
	cdef bint inchart
	cdef list componentlist
	cdef dict componentdict
	# put item in `newitem'; if item ends up in chart, newitem will be replaced
	# by a fresh object, otherwise, it can be re-used.
	inagenda = agenda.contains(newitem)
	inchart = PyDict_Contains(chart, newitem) == 1
	if not inagenda and not inchart:
		#if score > 300.0:
		#	blocked[0] += 1
		#	return newitem
		# check if we need to prune this item
		if whitelist is not None and whitelist[newitem.label] is not None:
			# disc. item to be treated as several split items?
			if splitprune:
				if markorigin:
					componentlist = <list>(whitelist[newitem.label])
				else:
					componentdict = <dict>(whitelist[newitem.label])
				COMPONENT.label = 0
				b = cnt = 0
				a = nextset(newitem.vec, b)
				while a != -1:
					b = nextunset(newitem.vec, a)
					# given a=3, b=6, make bitvector: 1000000 - 1000 = 111000
					COMPONENT.vec = (1ULL << b) - (1ULL << a)
					if markorigin:
						componentdict = <dict>(componentlist[cnt])
					if PyDict_Contains(componentdict, COMPONENT) != 1:
						blocked[0] += 1
						return newitem
					a = nextset(newitem.vec, b)
					cnt += 1
			else:
				label = newitem.label
				newitem.label = 0
				if PyDict_Contains(whitelist[label], newitem) != 1:
					#or inside + <double><object>outside > 300.0):
					blocked[0] += 1
					return newitem
				newitem.label = label

		# haven't seen this item before, won't prune, add to agenda
		agenda.setitem(newitem,
				new_LCFRSEdge(score, inside, rule, left, right))
		chart[newitem] = {}
	# in agenda (maybe in chart)
	elif not exhaustive and inagenda:
		agenda.setifbetter(newitem,
				new_LCFRSEdge(score, inside, rule, left, right))
	elif (inagenda and inside < (<LCFRSEdge>(agenda.getitem(newitem))).inside):
		# item has lower score, decrease-key in agenda
		# (add old, suboptimal edge to chart if parsing exhaustively)
		edge = iscore(agenda.replace(newitem,
				new_LCFRSEdge(score, inside, rule, left, right)))
		chart[newitem][edge] = edge
	# not in agenda => must be in chart
	elif (not inagenda and inside <
				(<LCFRSEdge>(<dict>viterbi[newitem.label])[newitem]).inside):
		#re-add to agenda because we found a better score.
		#should not happen without estimates!
		agenda.setitem(newitem,
				new_LCFRSEdge(score, inside, rule, left, right))
		logging.warning("WARN: updating score in agenda: %r", newitem)
	elif exhaustive:
		# suboptimal edge
		edge = iscore(new_LCFRSEdge(score, inside, rule, left, right))
		chart[newitem][edge] = edge
	return SmallChartItem.__new__(SmallChartItem)


cdef inline bint concat(Rule *rule, ULLong lvec, ULLong rvec):
	""" Determine the compatibility of two bitvectors (tuples of spans /
	ranges) according to the given yield function. Ranges should be
	non-overlapping, continuous when they are concatenated, and adhere to the
	ordering in the yield function. The yield function is a tuple of tuples
	with indices indicating from which vector the next span should be taken
	from, with 0 meaning left and 1 right. Note that the least significant bit
	is the lowest index in the vectors, which is the opposite of their normal
	presentation: 0b100 is the last terminal in a three word sentence, 0b001 is
	the first. E.g.,

	>>> lvec = 0b0011; rvec = 0b1000; yieldfunction = ((0, ), (1, ))
	>>> concat(((0, ), (1, )), lvec, rvec)
	True		#discontinuous, non-overlapping, linearly ordered.
	>>> concat(((0, 1), ), lvec, rvec)
	False		#lvec and rvec are not contiguous
	>>> concat(((1, ), (0, )), lvec, rvec)
	False		#rvec's span should come after lvec's span

	update: yield functions are now encoded in a binary format;
		cf. containers.pyx
		( (0, 1, 0), (1, 0) ) ==> args: 0b10010     lengths: 0b00101
		NB: note reversal due to the way binary numbers are represented
		the least significant bit (rightmost) corresponds to the lowest
		index in the sentence / constituent (leftmost). """
	if lvec & rvec:
		return False
	cdef ULLong mask = rvec if testbitint(rule.args, 0) else lvec
	cdef size_t n
	for n in range(bitlength(rule.lengths)):
		if testbitint(rule.args, n):  # component from right vector
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
		if testbitint(rule.lengths, n):  # a gap
			# check that there is a gap in both vectors
			if (lvec ^ rvec) & (mask + 1):
				return False
			# increase mask to cover gap
			# get minimum of trailing zero bits of lvec & rvec
			mask = (~lvec & (lvec - 1)) & (~rvec & (rvec - 1))
		mask += 1  # e.g., 00111 => 01000
	# success if we've reached the end of both left and right vector
	return lvec == rvec == 0


def parse_longsent(sent, Grammar grammar, tags=None, start=1,
		bint exhaustive=True, list whitelist=None, bint splitprune=False, bint
		markorigin=False, estimates=None):
	""" Parse a sentence longer than the machine word size. """
	cdef dict chart = {}  # the full chart
	cdef list viterbi = [{} for _ in grammar.toid]  # the viterbi probabilities
	cdef EdgeAgenda agenda = EdgeAgenda()  # the agenda
	cdef Rule *rule
	cdef LexicalRule lexrule
	cdef Entry entry
	cdef LCFRSEdge edge
	cdef FatChartItem item, sibling, newitem = new_FatChartItem(0)
	cdef FatChartItem goal
	cdef np.ndarray[np.double_t, ndim=4] outside
	cdef double x = 0.0, y = 0.0, score, inside
	cdef signed int length = 0, left = 0, right = 0, gaps = 0
	cdef signed int lensent = len(sent), estimatetype = 0
	cdef UInt i, blocked = 0, Epsilon = grammar.toid[b'Epsilon']
	cdef ULong maxA = 0
	goal = new_FatChartItem(start)
	ulongset(goal.vec, ~0UL, BITNSLOTS(lensent) - 1)
	goal.vec[BITSLOT(lensent)] = BITMASK(lensent) - 1

	if len(sent) >= (sizeof(goal.vec) * 8):
		msg = ("sentence too long (recompile with larger value for SLOTS)."
				"input length: %d. max: %d" % (lensent, sizeof(goal.vec) * 8))
		return chart, FATNONE, msg
	if estimates is not None:
		estimatetypestr, outside = estimates
		estimatetype = {"SX": SX, "SXlrgaps": SXlrgaps}[estimatetypestr]

	# scan
	for i, word in enumerate(sent):
		recognized = False
		tag = tags[i].encode('ascii') if tags else None
		item = new_FatChartItem(Epsilon)
		item.vec[0] = i
		if estimates is not None:
			length = 1
			left = i
			gaps = 0
			right = lensent - 1 - i
		for lexrule in grammar.lexical.get(word, ()):
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
					score += outside[lexrule.lhs, length, left + right, gaps]
					if score > 300.0:
						continue
				newitem.label = lexrule.lhs
				SETBIT(newitem.vec, i)
				tmp = process_fatedge(newitem, score, lexrule.prob, NULL,
						item, FATNONE, agenda, chart, viterbi, grammar,
						exhaustive, whitelist, False, markorigin, &blocked)
				#check whether item has not been blocked
				recognized |= newitem is not tmp
				newitem = tmp
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
			tagitem = new_FatChartItem(lhs)
			SETBIT(tagitem.vec, i)
			agenda[tagitem] = new_LCFRSEdge(score, 0.0, NULL, item, FATNONE)
			chart[tagitem] = {}
			recognized = True
		elif not recognized:
			msg = "not covered: %r" % (tag or word)
			return chart, FATNONE, msg

	# parsing
	while agenda.length:
		entry = agenda.popentry()
		item = <FatChartItem>entry.key
		edge = iscore(<LCFRSEdge>entry.value)
		chart[item][edge] = edge
		(<dict>viterbi[item.label])[item] = edge
		if item.label == goal.label and item.vec == goal.vec:
			if not exhaustive:
				break
		else:
			x = edge.inside

			# unary
			if estimates is not None:
				length = abitcount(item.vec, SLOTS)
				left = anextset(item.vec, 0, SLOTS)
				gaps = abitlength(item.vec, SLOTS) - length - left
				right = lensent - length - left - gaps
			for i in range(grammar.numrules):
				rule = &(grammar.unary[item.label][i])
				if rule.rhs1 != item.label:
					break
				score = inside = x + rule.prob
				if estimatetype == SX:
					score += outside[rule.lhs, left, right, 0]
					if score > 300.0:
						continue
				elif estimatetype == SXlrgaps:
					score += outside[rule.lhs, length, left + right, gaps]
					if score > 300.0:
						continue
				newitem.label = rule.lhs
				ulongcpy(newitem.vec, item.vec, SLOTS)
				newitem = process_fatedge(newitem, score, inside,
						rule, item, FATNONE, agenda, chart, viterbi,
						grammar, exhaustive, whitelist,
						splitprune and grammar.fanout[rule.lhs] != 1,
						markorigin, &blocked)

			# binary right
			for i in range(grammar.numrules):
				rule = &(grammar.rbinary[item.label][i])
				if rule.rhs2 != item.label:
					break
				for I, e in (<dict>viterbi[rule.rhs1]).items():
					sibling = <FatChartItem>I
					if fatconcat(rule, sibling.vec, item.vec):
						newitem.label = rule.lhs
						setunion(newitem.vec, sibling.vec, item.vec, SLOTS)
						y = (<LCFRSEdge>e).inside
						score = inside = x + y + rule.prob
						if estimatetype == SX:
							length = abitcount(item.vec, SLOTS)
							left = anextset(item.vec, 0, SLOTS)
							right = lensent - length - left
							score += outside[rule.lhs, left, right, 0]
							if score > 300.0:
								continue
						elif estimatetype == SXlrgaps:
							length = abitcount(item.vec, SLOTS)
							left = anextset(item.vec, 0, SLOTS)
							gaps = abitlength(item.vec, SLOTS) - length - left
							right = lensent - length - left - gaps
							score += outside[rule.lhs, length, left + right, gaps]
							if score > 300.0:
								continue
						newitem = process_fatedge(newitem, score, inside,
								rule, sibling, item, agenda, chart, viterbi,
								grammar, exhaustive, whitelist,
								splitprune and grammar.fanout[rule.lhs] != 1,
								markorigin, &blocked)

			# binary left
			for i in range(grammar.numrules):
				rule = &(grammar.lbinary[item.label][i])
				if rule.rhs1 != item.label:
					break
				for I, e in (<dict>viterbi[rule.rhs2]).items():
					sibling = <FatChartItem>I
					if fatconcat(rule, item.vec, sibling.vec):
						newitem.label = rule.lhs
						setunion(newitem.vec, item.vec, sibling.vec, SLOTS)
						y = (<LCFRSEdge>e).inside
						score = inside = x + y + rule.prob
						if estimatetype == SX:
							length = abitcount(newitem.vec, SLOTS)
							left = anextset(newitem.vec, 0, SLOTS)
							right = lensent - length - left
							score += outside[rule.lhs, left, right, 0]
							if score > 300.0:
								continue
						elif estimatetype == SXlrgaps:
							length = abitcount(newitem.vec, SLOTS)
							left = anextset(newitem.vec, 0, SLOTS)
							gaps = abitlength(newitem.vec,
									SLOTS) - length - left
							right = lensent - length - left - gaps
							score += outside[rule.lhs, length, left + right, gaps]
							if score > 300.0:
								continue
						newitem = process_fatedge(newitem, score, inside,
								rule, item, sibling, agenda, chart, viterbi,
								grammar, exhaustive, whitelist,
								splitprune and grammar.fanout[rule.lhs] != 1,
								markorigin, &blocked)

		if agenda.length > maxA:
			maxA = agenda.length
	msg = ("agenda max %d, now %d, items %d (%d labels), edges %d, blocked %d"
		% (maxA, len(agenda), len(list(filter(None, chart.values()))),
		len(list(filter(None, viterbi))), sum(map(len, chart.values())), blocked))
	if goal in chart:
		return chart, goal, msg
	else:
		return chart, FATNONE, "no parse " + msg


cdef inline FatChartItem process_fatedge(FatChartItem newitem, double score,
		double inside, Rule *rule, FatChartItem left, FatChartItem right,
		EdgeAgenda agenda, dict chart, list viterbi, Grammar grammar, bint
		exhaustive, list whitelist, bint splitprune, bint markorigin,
		UInt *blocked):
	""" Decide what to do with a newly derived edge. """
	cdef UInt a, b, cnt, i
	cdef bint inagenda
	cdef bint inchart
	cdef list componentlist
	cdef dict componentdict
	# if item ends up in chart, newitem will be replaced
	# by a fresh object, otherwise, it can be re-used.
	inagenda = agenda.contains(newitem)
	inchart = PyDict_Contains(chart, newitem) == 1
	if not inagenda and not inchart:
		#if score > 300.0:
		#	blocked[0] += 1
		#	return newitem
		# check if we need to prune this item
		if whitelist is not None and whitelist[newitem.label] is not None:
			# disc. item to be treated as several split items?
			if splitprune:
				if markorigin:
					componentlist = <list>(whitelist[newitem.label])
				else:
					componentdict = <dict>(whitelist[newitem.label])
				b = cnt = 0
				a = anextset(newitem.vec, b, SLOTS)
				while a != -1:
					b = anextunset(newitem.vec, a, SLOTS)
					#given a=3, b=6, make bitvector: 1000000 - 1000 = 111000
					for i in range(a, b):
						SETBIT(FATCOMPONENT.vec, i)
					if markorigin:
						componentdict = <dict>(componentlist[cnt])
					if PyDict_Contains(componentdict, FATCOMPONENT) != 1:
						blocked[0] += 1
						return newitem
					a = anextset(newitem.vec, b, SLOTS)
					cnt += 1
			else:
				label = newitem.label
				newitem.label = 0
				if PyDict_Contains(whitelist[label], newitem) != 1:
					#or inside + <double><object>outside > 300.0):
					blocked[0] += 1
					return newitem
				newitem.label = label

		# haven't seen this item before, won't prune, add to agenda
		agenda.setitem(newitem,
				new_LCFRSEdge(score, inside, rule, left, right))
		chart[newitem] = {}
	# in agenda (maybe in chart)
	elif not exhaustive and inagenda:
		agenda.setifbetter(newitem,
				new_LCFRSEdge(score, inside, rule, left, right))
	elif (inagenda and inside < (<LCFRSEdge>(agenda.getitem(newitem))).inside):
		# item has lower score, decrease-key in agenda
		# (add old, suboptimal edge to chart if parsing exhaustively)
		edge = iscore(agenda.replace(newitem,
				new_LCFRSEdge(score, inside, rule, left, right)))
		chart[newitem][edge] = edge
	# not in agenda => must be in chart
	elif (not inagenda and inside <
				(<LCFRSEdge>(<dict>viterbi[newitem.label])[newitem]).inside):
		#re-add to agenda because we found a better score.
		#should not happen without estimates!
		agenda.setitem(newitem,
				new_LCFRSEdge(score, inside, rule, left, right))
		logging.warning("WARN: updating score in agenda: %r", newitem)
	elif exhaustive:
		# suboptimal edge
		edge = iscore(new_LCFRSEdge(score, inside, rule, left, right))
		chart[newitem][edge] = edge
	return FatChartItem.__new__(FatChartItem)


cdef inline bint fatconcat(Rule *rule, ULong *lvec, ULong *rvec):
	""" Determine the compatibility of two bitvectors (tuples of spans /
	ranges) according to the given yield function. Ranges should be
	non-overlapping, continuous when they are concatenated, and adhere to the
	ordering in the yield function.
	The yield function is a tuple of tuples with indices indicating from which
	vector the next span should be taken from, with 0 meaning left and 1 right.
	Note that the least significant bit is the lowest index in the vectors,
	which is the opposite of their normal presentation: 0b100 is the last
	terminal in a three word sentence, 0b001 is the first. E.g.,

	>>> lvec = 0b0011; rvec = 0b1000; yieldfunction = ((0, ), (1, ))
	>>> concat(((0, ), (1, )), lvec, rvec)
	True		#discontinuous, non-overlapping, linearly ordered.
	>>> concat(((0, 1), ), lvec, rvec)
	False		#lvec and rvec are not contiguous
	>>> concat(((1, ), (0, )), lvec, rvec)
	False		#rvec's span should come after lvec's span

	update: yield functions are now encoded in a binary format;
		cf. containers.pyx
		( (0, 1, 0), (1, 0) ) ==> args: 0b10010     lengths: 0b00101
		NB: note reversal due to the way binary numbers are represented
		the least significant bit (rightmost) corresponds to the lowest
		index in the sentence / constituent (leftmost). """
	cdef int lpos
	cdef int rpos
	cdef UInt n
	for n in range(SLOTS):
		if lvec[n] & rvec[n]:
			return False
	lpos = anextset(lvec, 0, SLOTS)
	rpos = anextset(rvec, 0, SLOTS)

	#this algorithm was adapted from rparse, FastYFComposer.
	for n in range(bitlength(rule.lengths)):
		if testbitint(rule.args, n):
			# check if there are any bits left, and
			# if any bits on the right should have gone before
			# ones on this side
			if rpos == -1 or (lpos != -1 and lpos <= rpos):
				return False
			# jump to next gap
			rpos = anextunset(rvec, rpos, SLOTS)
			if lpos != -1 and lpos < rpos:
				return False
			# there should be a gap if and only if
			# this is the last element of this argument
			if testbitint(rule.lengths, n):
				if TESTBIT(lvec, rpos):
					return False
			elif not TESTBIT(lvec, rpos):
				return False
			# jump to next argument
			rpos = anextset(rvec, rpos, SLOTS)
		else:  # if bit == 0:
			# vice versa to the above
			if lpos == -1 or (rpos != -1 and rpos <= lpos):
				return False
			lpos = anextunset(lvec, lpos, SLOTS)
			if rpos != -1 and rpos < lpos:
				return False
			if testbitint(rule.lengths, n):
				if TESTBIT(rvec, lpos):
					return False
			elif not TESTBIT(rvec, lpos):
				return False
			lpos = anextset(lvec, lpos, SLOTS)
	# success if we've reached the end of both left and right vector
	return lpos == rpos == -1


def symbolicparse(sent, Grammar grammar, tags=None, start=1,
		bint exhaustive=True, list whitelist=None, bint splitprune=False, bint
		markorigin=False):
	""" parse sentence, a list of tokens, optionally with gold tags, and
	produce a chart, either exhaustive or up until the first parse.
	Non-probabilistic version.
	Other parameters:
	- start: integer corresponding to the start symbol that analyses should
		have, e.g., grammar.toid['ROOT']
	- exhaustive: don't stop at viterbi parser, return a full chart
	- whitelist: a whitelist of allowed ChartItems. Anything else is not
		added to the agenda.
	- splitprune: coarse stage used a split-PCFG where discontinuous node
		appear as multiple CFG nodes. Every discontinuous node will result
		in multiple lookups into whitelist to see whether it should be
		allowed on the agenda.
	- markorigin: in combination with splitprune, coarse labels include an
		integer to distinguish components; e.g., CFG nodes NP*0 and NP*1
		map to the discontinuous node NP_2. """
	cdef:
		dict chart = {}  # the full chart
		list items = [deque() for _ in grammar.toid]  # items for each label
		object agenda = deque()  # the agenda
		Rule *rule
		LexicalRule lexrule
		Entry entry
		LCFRSEdge edge
		SmallChartItem item, sibling, newitem = new_ChartItem(0, 0)
		SmallChartItem goal = new_ChartItem(start, (1ULL << len(sent)) - 1)
		signed int lensent = len(sent)
		UInt i, blocked = 0, Epsilon = grammar.toid[b'Epsilon']
		ULong maxA = 0
		ULLong vec = 0

	if lensent >= sizeof(vec) * 8:
		raise NotImplementedError("sentence too long.")

	# scan
	for i, word in enumerate(sent):
		recognized = False
		tag = tags[i].encode('ascii') if tags else None
		item = new_ChartItem(Epsilon, i)
		for lexrule in grammar.lexical.get(word, ()):
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if (not tags or grammar.tolabel[lexrule.lhs] == tag
				or grammar.tolabel[lexrule.lhs].startswith(tag + b'@')):
				score = lexrule.prob
				newitem.label = lexrule.lhs
				newitem.vec = 1ULL << i
				edge = new_LCFRSEdge(0.0, 0.0, NULL, item, NONE)
				agenda.append(newitem)
				chart[newitem][edge] = edge
				items[newitem.label].append(newitem)
				newitem = SmallChartItem.__new__(SmallChartItem)
		if not recognized and tags and tag in grammar.toid:
			newitem.label = grammar.toid[tag]
			newitem.vec = 1ULL << i
			edge = new_LCFRSEdge(0.0, 0.0, NULL, item, NONE)
			agenda.append(newitem)
			chart[newitem][edge] = edge
			items[newitem.label].append(newitem)
			newitem = SmallChartItem.__new__(SmallChartItem)
			recognized = True
		elif not recognized:
			return chart, NONE, "not covered: %r" % (tag or word)

	# parsing
	while agenda:
		item = agenda.pop()
		if item in chart:
			continue
		items[item.label].append(item)
		if item.label == goal.label and item.vec == goal.vec:
			if not exhaustive:
				break
		else:
			# unary
			for i in range(grammar.numrules):
				rule = &(grammar.unary[item.label][i])
				if rule.rhs1 != item.label:
					break
				newitem.label = rule.lhs
				newitem.vec = item.vec
				if newitem not in chart:
					edge = new_LCFRSEdge(0.0, 0.0, rule, item, NONE)
					agenda.append(newitem)
					chart[newitem][edge] = edge
					newitem = SmallChartItem.__new__(SmallChartItem)

			# binary right
			for i in range(grammar.numrules):
				rule = &(grammar.rbinary[item.label][i])
				if rule.rhs2 != item.label:
					break
				for I in items[rule.rhs1]:
					sibling = <SmallChartItem>I
					if concat(rule, sibling.vec, item.vec):
						newitem.label = rule.lhs
						newitem.vec = sibling.vec ^ item.vec
						if newitem not in chart:
							edge = new_LCFRSEdge(0.0, 0.0, rule, sibling, item)
							agenda.append(newitem)
							chart[newitem][edge] = edge
							newitem = SmallChartItem.__new__(SmallChartItem)

			# binary left
			for i in range(grammar.numrules):
				rule = &(grammar.lbinary[item.label][i])
				if rule.rhs1 != item.label:
					break
				for I in items[rule.rhs2]:
					sibling = <SmallChartItem>I
					if concat(rule, item.vec, sibling.vec):
						newitem.label = rule.lhs
						newitem.vec = item.vec ^ sibling.vec
						if newitem not in chart:
							edge = new_LCFRSEdge(0.0, 0.0, rule, item, sibling)
							agenda.append(newitem)
							chart[newitem][edge] = edge
							newitem = SmallChartItem.__new__(SmallChartItem)

		if agenda.length > maxA:
			maxA = agenda.length
	msg = ("agenda max %d, now %d, items %d, edges %d, blocked %d" % (
			maxA, len(agenda), len(list(filter(None, chart.values()))),
			sum(map(len, chart.values())), blocked))
	if goal in chart:
		return chart, goal, msg
	else:
		return chart, NONE, "no parse " + msg


#def doinsideoutside(dict chart, ChartItem start):
#	cdef dict result = dict.fromkeys(chart)
#	getinside(result, chart, start)
#
#cdef getinside(dict result, chart, start):
#
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
#		if (testbitint(rule.args, n) == 0) == left: # the given item
#			if pos == -1:
#				return False
#			prevpos = nextunset(item.vec, pos)
#			pos = nextset(item.vec, prevpos)
#		else: # the other item for which to find candidates
#			if n and testbitint(rule.lengths, n - 1): # start gap?
#				temp = set()
#				if testbitint(rule.lengths, n): # & end gap?
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
#			elif testbitint(rule.lengths, n): # end gap?
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


cdef inline LCFRSEdge iscore(LCFRSEdge e):
	""" Replace estimate with inside probability """
	e.score = e.inside
	return e


def sortfunc(a):
	if isinstance(a, SmallChartItem):
		return (bitcount((<SmallChartItem>a).vec), (<SmallChartItem>a).vec)
	elif isinstance(a, FatChartItem):
		return (abitcount((<FatChartItem>a).vec, SLOTS),
			anextset((<FatChartItem>a).vec, 0, SLOTS))
	elif isinstance(a, LCFRSEdge):
		return (<LCFRSEdge>a).inside


def pprint_chart(chart, sent, tolabel):
	""" `pretty print' a chart. """
	cdef ChartItem a
	cdef LCFRSEdge edge
	print("chart:")
	for a in sorted(chart, key=sortfunc):
		if chart[a] == []:
			continue
		print("%s[%s] =>" % (tolabel[a.label].decode('ascii'),
				a.binrepr(len(sent))))
		if isinstance(chart[a], float):
			continue
		for edge in sorted(chart[a], key=sortfunc):
			if edge.rule is NULL:
				print("%9.7f  %9.7f " % (exp(-edge.inside), 1), end='')
				print("\t'%s'" % sent[edge.left.lexidx()], end='')
			else:
				print("%9.7f  %9.7f " % (exp(-edge.inside),
						exp(-edge.rule.prob)),
						"%s[%s]" % (tolabel[edge.left.label].decode('ascii'),
						edge.left.binrepr(len(sent))), end='')
				if edge.right:
					print("\t%s[%s]" % (tolabel[edge.right.label].decode('ascii'),
							edge.right.binrepr(len(sent))), end='')
			print()
		print()


def do(sent, grammar):
	from disambiguation import marginalize
	from operator import itemgetter
	print("sentence", sent)
	sent = sent.split()
	chart, start, _ = parse(sent, grammar)
	if len(sent) < sizeof(ULLong):
		pprint_chart(chart, sent, grammar.tolabel)
	if start:
		print("10 best parse trees:")
		mpp, _ = marginalize("mpp", chart, start, grammar, 10)
		for a, p in reversed(sorted(mpp.items(), key=itemgetter(1))):
			print(p, a)
		print()
		return True
	print("no parse")
	return False


def main():
	from containers import Grammar
	cdef Rule rule
	rule.args = 0b1010
	rule.lengths = 0b1010
	assert concat(&rule, 0b100010, 0b1000100)
	assert not concat(&rule, 0b1010, 0b10100)
	rule.args = 0b101
	rule.lengths = 0b101
	assert concat(&rule, 0b10000, 0b100100)
	assert not concat(&rule, 0b1000, 0b10100)
	grammar = Grammar([
		((('S', 'VP2', 'VMFIN'), ((0, 1, 0), )), 1),
		((('VP2', 'VP2', 'VAINF'), ((0, ), (0, 1))), 0.5),
		((('VP2', 'PROAV', 'VVPP'), ((0, ), (1, ))), 0.5),
		((('PROAV', 'Epsilon'), ('Daruber', )), 1),
		((('VAINF', 'Epsilon'), ('werden', )), 1),
		((('VMFIN', 'Epsilon'), ('muss', )), 1),
		((('VVPP', 'Epsilon'), ('nachgedacht', )), 1)], start='S')
	print(grammar)

	assert do("Daruber muss nachgedacht werden", grammar)
	assert do("Daruber muss nachgedacht werden werden", grammar)
	assert do("Daruber muss nachgedacht werden werden werden", grammar)
	print("ungrammatical sentence:")
	assert not do("muss Daruber nachgedacht werden", grammar)  # no parse
	print("(as expected)")
	print("long sentence (%d words):" % 67)
	assert do('Daruber muss nachgedacht ' + ' '.join(64 * ['werden']), grammar)
