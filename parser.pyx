""" Probabilistic CKY parser for monotone, string-rewriting
Linear Context-Free Rewriting Systems. """
# python imports
from math import log, exp
from array import array
from collections import defaultdict
import re, logging, sys
import numpy as np
from agenda import EdgeAgenda, Entry
# cython imports
from cpython cimport PyDict_Contains, PyDict_GetItem
cimport numpy as np
from agenda cimport Entry, EdgeAgenda
from containers cimport ChartItem, Edge, Grammar, Rule, LexicalRule,\
    UChar, UInt, ULong, ULLong, new_ChartItem, new_Edge
from array cimport array
from bit cimport nextset, nextunset, bitcount, bitlength, testbit, testbitint
cdef extern from "math.h":
	bint isinf(double x)
	bint isfinite(double x)
cdef ChartItem NONE = new_ChartItem(0, 0)
np.import_array()

def parse(sent, Grammar grammar, tags=None, start=1, bint exhaustive=False,
			estimates=None, list whitelist=None, bint splitprune=False,
			bint markorigin=False, int beamwidth=0):
	""" parse sentence, a list of tokens, optionally with gold tags, and
	produce a chart, either exhaustive or up until the viterbi parse.
	Other parameters:
		- start: integer corresponding to the start tag in the grammar,
			e.g., grammar.toid['ROOT']
		- exhaustive: don't stop at viterbi parser, return a full chart
		- estimates: use context-summary estimates (heuristics, figures-of-merit)
			to order agenda. If estimates are not consistent, it is no longer
			guaranteed that the optimal parse will be found. experimental.
		- whitelist: a whitelist of allowed ChartItems. Anything else is not
			added to the agenda.
		- splitprune: coarse stage used a split-PCFG where discontinuous node
			appear as multiple CFG nodes. Every discontinuous node will result
			in multiple lookups into whitelist to see whether it should be
			allowed on the agenda.
        - markorigin: in combination with splitprune, coarse labels include an
            integer to distinguish components; e.g., CFG nodes NP*0 and NP*1
            map to the discontinuous node NP_2
		- beamwidth: specify the maximum number of items that will be explored
			for each particular span, on a first-come-first-served basis.
			setting to 0 disables this feature. experimental.
	"""
	cdef str label = ''
	cdef dict beam = <dict>defaultdict(int)			#histogram of spans
	cdef dict chart = {}							#the full chart
	cdef list viterbi = [{} for _ in grammar.toid]	#the viterbi probabilities
	cdef list itempool								#to recycle chart items
	cdef EdgeAgenda agenda = EdgeAgenda()			#the agenda
	cdef size_t i
	cdef Rule rule
	cdef LexicalRule terminal
	cdef Entry entry
	cdef Edge edge
	cdef ChartItem item, sibling
	cdef ChartItem goal = new_ChartItem(start, (1ULL << len(sent)) - 1)
	cdef np.ndarray[np.double_t, ndim=4] outside
	cdef double x = 0.0, y = 0.0, score, inside
	cdef signed int length = 0, left = 0, right = 0, gaps = 0
	cdef signed int lensent = len(sent), maxlen = 0
	cdef UInt blocked = 0, Epsilon = grammar.toid["Epsilon"]
	cdef ULong maxA = 0
	cdef ULLong vec = 0

	assert len(sent) < (sizeof(vec) * 8)
	if estimates is not None:
		outside, maxlen = estimates
		#assert len(grammar.bylhs) == len(outside)
		assert lensent <= maxlen
	itempool = [new_ChartItem(0, 0) for _ in range(1000)]

	# scan
	for i, w in enumerate(sent):
		recognized = False
		item = new_ChartItem(Epsilon, i)
		if estimates is not None:
			length = 1; left = i; gaps = 0; right = lensent - 1 - i
		for terminal in grammar.lexical.get(w, []):
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if (not tags or grammar.tolabel[terminal.lhs] == tags[i]
				or grammar.tolabel[terminal.lhs].startswith(tags[i] + "@")):
				score = terminal.prob
				if estimates is not None:
					score += outside[terminal.lhs, length, left+right, gaps]
				process_edge(terminal.lhs, 1ULL << i, score, terminal.prob,
					terminal.prob, item, NONE, agenda, chart, viterbi,
					itempool, grammar, exhaustive, whitelist, False,
					markorigin, &blocked)
				recognized = True
		if not recognized and tags and tags[i] in grammar.toid:
			lhs = grammar.toid[tags[i]]
			score = 0.0
			if estimates is not None:
				score += outside[lhs, length, left+right, gaps]
			newitem = new_ChartItem(lhs, 1ULL << i)
			agenda[newitem] = new_Edge(score, 0.0, 0.0, item, NONE)
			chart[newitem] = []
			recognized = True
		elif not recognized:
			return chart, NONE, "not covered: '%s'. " % (tags[i] if tags else w)

	# parsing
	while agenda.length:
		entry = agenda.popentry()
		item = <ChartItem>entry.key
		edge = <Edge>entry.value
		chart[item].append(iscore(edge))
		(<dict>viterbi[item.label])[item] = edge
		if item.label == goal.label and item.vec == goal.vec:
			if not exhaustive: break
		else:
			x = edge.inside

			# unary
			if estimates is not None:
				length = bitcount(item.vec); left = nextset(item.vec, 0)
				gaps = bitlength(item.vec) - length - left
				right = lensent - length - left - gaps
			if not beamwidth or beam[item.vec] < beamwidth:
				for i in range(grammar.numrules):
					rule = grammar.unary[item.label][i]
					if rule.rhs1 != item.label: break
					score = inside = x + rule.prob
					if estimates is not None:
						score += outside[rule.lhs, length, left+right, gaps]
					process_edge(rule.lhs, item.vec, score, inside, rule.prob,
						item, NONE, agenda, chart, viterbi, itempool, grammar,
						exhaustive, whitelist,
						splitprune and grammar.fanout[rule.lhs] != 1,
						markorigin, &blocked)

			# binary left
			for i in range(grammar.numrules):
				rule = grammar.lbinary[item.label][i]
				if rule.rhs1 != item.label: break
				for I, e in (<dict>viterbi[rule.rhs2]).iteritems():
					sibling = <ChartItem>I
					if ((not beamwidth or beam[item.vec ^ sibling.vec]
						< beamwidth) and concat(rule, item.vec, sibling.vec)):
						beam[item.vec ^ sibling.vec] += 1
						vec = item.vec ^ sibling.vec
						y = (<Edge>e).inside
						score = inside = x + y + rule.prob
						if estimates is not None:
							length = bitcount(vec); left = nextset(vec, 0)
							gaps = bitlength(vec) - length - left
							right = lensent - length - left - gaps
							score += outside[rule.lhs,length,left+right,gaps]
						process_edge(rule.lhs, vec, score, inside, rule.prob,
								item, sibling, agenda, chart, viterbi,
								itempool, grammar, exhaustive, whitelist,
								splitprune and grammar.fanout[rule.lhs] != 1,
								markorigin, &blocked)

			# binary right
			for i in range(grammar.numrules):
				rule = grammar.rbinary[item.label][i]
				if rule.rhs2 != item.label: break
				for I, e in (<dict>viterbi[rule.rhs1]).iteritems():
					sibling = <ChartItem>I
					if ((not beamwidth or beam[sibling.vec ^ item.vec]
						< beamwidth) and concat(rule, sibling.vec, item.vec)):
						beam[sibling.vec ^ item.vec] += 1
						vec = sibling.vec ^ item.vec
						y = (<Edge>e).inside
						score = inside = x + y + rule.prob
						if estimates is not None:
							length = bitcount(vec); left = nextset(vec, 0)
							gaps = bitlength(vec) - length - left
							right = lensent - length - left - gaps
							score += outside[rule.lhs,length,left+right,gaps]
						process_edge(rule.lhs, vec, score, inside, rule.prob,
								sibling, item, agenda, chart, viterbi,
								itempool, grammar, exhaustive, whitelist,
								splitprune and grammar.fanout[rule.lhs] != 1,
								markorigin, &blocked)

		if agenda.length > maxA: maxA = agenda.length
		#if agenda.length % 1000 == 0:
		#	logging.debug(("agenda max %d, now %d, items %d (%d labels), "
		#		"edges %d, blocked %d" % (maxA, len(agenda),
		#		len(filter(None, chart.values())), len(filter(None, viterbi)),
		#		sum(map(len, chart.values())), blocked)))
	msg = ("agenda max %d, now %d, items %d (%d labels), edges %d, blocked %d. "
		% (maxA, len(agenda), len(filter(None, chart.values())),
		len(filter(None, viterbi)), sum(map(len, chart.values())), blocked))
	if goal in chart: return chart, goal, msg
	else: return chart, NONE, "no parse " + msg

cdef inline process_edge(UInt label, ULLong vec, double score, double inside,
		double prob, ChartItem left, ChartItem right,
		EdgeAgenda agenda, dict chart, list viterbi, list itempool,
		Grammar grammar, bint exhaustive, list whitelist,
		bint splitprune, bint markorigin, UInt *blocked):
	""" Decide what to do with a newly derived edge. """
	cdef ChartItem newitem
	cdef ULLong component
	cdef UInt a, b, cnt
	cdef bint inagenda
	cdef bint inchart
	cdef list componentlist
	cdef dict componentdict
	if len(itempool) == 0:
		itempool = [new_ChartItem(0, 0) for _ in range(10000)]
	newitem = itempool.pop()
	newitem.label = label; newitem.vec = vec
	inagenda = agenda.contains(newitem)
	inchart = PyDict_Contains(chart, newitem) == 1
	if not inagenda and not inchart:
		if whitelist is not None and whitelist[label] is not None:
			# disc. item to be treated as several split items?
			if splitprune:
				if markorigin: componentlist = <list>(whitelist[label])
				else: componentdict = <dict>(whitelist[label])
				b = cnt = 0
				a = nextset(vec, b)
				while a != -1:
					b = nextunset(vec, a)
					#given a=3, b=6, make bitvector: 1000000 - 1000 = 111000
					component = (1ULL << b) - (1ULL << a)
					if markorigin:
						try: componentdict = <dict>(componentlist[cnt])
						except KeyError:
							print grammar.tolabel[label], bin(vec), cnt,
							print bin(component), map(bin, componentlist)
							raise
					if PyDict_Contains(componentdict, component) != 1:
						blocked[0] += 1
						itempool.append(newitem)
						return
					a = nextset(vec, b)
					cnt += 1
			else:
				if PyDict_Contains(whitelist[label], vec) != 1:
					#or inside+<double><object>outside > 300.0):
					blocked[0] += 1
					itempool.append(newitem)
					return
		#elif score > 300.0:
		#	blocked[0] += 1
		#	itempool.append(newitem)
		#	return

		# haven't seen this item before, won't prune, add to agenda
		agenda.setitem(newitem,
				new_Edge(score, inside, prob, left, right))
		chart[newitem] = []
	# in agenda (maybe in chart)
	elif not exhaustive and inagenda:
		agenda.setifbetter(newitem,
				new_Edge(score, inside, prob, left, right))
	elif (inagenda and inside < (<Edge>(agenda.getitem(newitem))).inside):
		# item has lower score, decrease-key in agenda
		# (add old, suboptimal edge to chart if parsing exhaustively)
		chart[newitem].append(iscore(agenda.replace(newitem,
				new_Edge(score, inside, prob, left, right))))
	# not in agenda => must be in chart
	elif (not inagenda and inside <
				(<Edge>(<dict>viterbi[newitem.label])[newitem]).inside):
		#re-add to agenda because we found a better score.
		#should not happen without estimates!
		agenda.setitem(newitem,
				new_Edge(score, inside, prob, left, right))
		logging.warning("WARN: updating score in agenda: %r" % (newitem))
	elif exhaustive:
		# suboptimal edge
		chart[newitem].append(iscore(
				new_Edge(score, inside, prob, left, right)))

cdef inline bint concat(Rule rule, ULLong lvec, ULLong rvec):
	"""
	Determine the compatibility of two bitvectors (tuples of spans / ranges)
	according to the given yield function. Ranges should be non-overlapping,
	continuous when they are concatenated, and adhere to the ordering in the
	yield function.
	The yield function is a tuple of tuples with indices indicating from which
	vector the next span should be taken from, with 0 meaning left and 1 right.
	Note that the least significant bit is the lowest index in the vectors,
	which is the opposite of their normal presentation: 0b100 is the last
	terminal in a three word sentence, 0b001 is the first. E.g.,

	>>> lvec = 0b0011; rvec = 0b1000; yieldfunction = ((0,), (1,))
	>>> concat(((0,), (1,)), lvec, rvec)
	True		#discontinuous, non-overlapping, linearly ordered.
	>>> concat(((0, 1),), lvec, rvec)
	False		#lvec and rvec are not contiguous
	>>> concat(((1,), (0,)), lvec, rvec)
	False		#rvec's span should come after lvec's span

	update: yield functions are now encoded in a binary format; cf. containers.pyx
		( (0, 1, 0), (1, 0) ) ==> args: 0b10010     lengths: 0b00101
		NB: note reversal due to the way binary numbers are represented
		the least significant bit (rightmost) corresponds to the lowest
		index in the sentence / constituent (leftmost).
	"""
	if lvec & rvec: return False
	cdef int lpos = nextset(lvec, 0)
	cdef int rpos = nextset(rvec, 0)
	cdef UInt n

	# if there are no gaps in lvec and rvec, and the yield function is the
	# concatenation of two elements, then this should^Wcould be quicker
	# fixme: profile, check correctness
	#if rule.args == rule.lengths == 0b10:
	#	n = nextunset(lvec, lpos)
	#	if (lvec >> n) == (rvec >> nextunset(rvec, rpos)) == 0:
	#		return rpos == n

	#this algorithm was adapted from rparse, FastYFComposer.
	for n in range(bitlength(rule.lengths)):
		if testbitint(rule.args, n):
			# check if there are any bits left, and
			# if any bits on the right should have gone before
			# ones on this side
			if rpos == -1 or (lpos != -1 and lpos <= rpos):
				return False
			# jump to next gap
			rpos = nextunset(rvec, rpos)
			if lpos != -1 and lpos < rpos:
				return False
			# there should be a gap if and only if
			# this is the last element of this argument
			if testbitint(rule.lengths, n):
				if testbit(lvec, rpos):
					return False
			elif not testbit(lvec, rpos):
				return False
			# jump to next argument
			rpos = nextset(rvec, rpos)
		else: #if bit == 0:
			# vice versa to the above
			if lpos == -1 or (rpos != -1 and rpos <= lpos):
				return False
			lpos = nextunset(lvec, lpos)
			if rpos != -1 and rpos < lpos:
				return False
			if testbitint(rule.lengths, n):
				if testbit(rvec, lpos):
					return False
			elif not testbit(rvec, lpos):
				return False
			lpos = nextset(lvec, lpos)
	# success if we've reached the end of both left and right vector
	return lpos == rpos == -1

def cfgparse(list sent, Grammar grammar, start=1, tags=None):
	""" A CKY parser modeled after Bodenstab's `fast grammar loop.'
		and the Stanford parser. """
	cdef short left, right, mid, span, lensent = len(sent)
	cdef short narrowr, narrowl, widel, wider, minmid, maxmid
	cdef long numsymbols = len(grammar.toid)
	cdef double oldscore, prob
	cdef size_t i
	cdef UInt lhs, rhs1, Epsilon = grammar.toid["Epsilon"]
	cdef ULLong vec = 0
	cdef bint foundbetter = False
	cdef Rule rule
	cdef LexicalRule terminal
	cdef ChartItem goal = ChartItem(start, (1ULL << len(sent)) - 1)
	cdef dict chart = {}						#the full chart
	cdef set applied = set()
	# the viterbi chart is initially filled with infinite log probabilities,
	# cells which are to be blocked contain NaN.
	cdef np.ndarray[np.double_t, ndim=3] viterbi = np.array([np.inf],
		dtype='d').repeat(lensent * (lensent+1) * numsymbols).reshape(
		(numsymbols, lensent, (lensent+1)))
	# matrices for the filter which gives minima and maxima for splits
	cdef np.ndarray[np.int16_t, ndim=2] minsplitleft = np.array([-1],
		dtype='int16').repeat(numsymbols * (lensent + 1)
		).reshape(numsymbols, lensent + 1)
	cdef np.ndarray[np.int16_t, ndim=2] maxsplitleft = np.array([lensent+1],
		dtype='int16').repeat(numsymbols * (lensent + 1)).reshape(
		numsymbols, lensent + 1)
	cdef np.ndarray[np.int16_t, ndim=2] minsplitright = np.array([lensent + 1],
		dtype='int16').repeat(numsymbols * (lensent + 1)
		).reshape(numsymbols, lensent + 1)
	cdef np.ndarray[np.int16_t, ndim=2] maxsplitright = np.array([-1],
		dtype='int16').repeat(numsymbols * (lensent + 1)).reshape(
		numsymbols, lensent + 1)

	assert len(sent) < (sizeof(vec) * 8), ("sentence too long. "
			"length: %d. limit: %d." % (len(sent), sizeof(vec) * 8))
	i = 0
	while grammar.unary[0][i].rhs1 != grammar.nonterminals: i += 1

	# assign POS tags
	#print 1, # == span
	for i, w in enumerate(sent):
		left = i; right = i + 1
		recognized = False
		for terminal in grammar.lexical.get(w, []):
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if (not tags or grammar.tolabel[terminal.lhs] == tags[i]
				or grammar.tolabel[terminal.lhs].startswith(tags[i] + '@')):
				item = new_ChartItem(terminal.lhs, 1ULL << i)
				sibling = new_ChartItem(Epsilon, i)
				x = viterbi[terminal.lhs, left, right] = terminal.prob
				chart[item] = [new_Edge(x, x, x, sibling, NONE)]
				# update filter
				if left > minsplitleft[terminal.lhs, right]:
					minsplitleft[terminal.lhs, right] = left
				if left < maxsplitleft[terminal.lhs, right]:
					maxsplitleft[terminal.lhs, right] = left
				if right < minsplitright[terminal.lhs, left]:
					minsplitright[terminal.lhs, left] = right
				if right > maxsplitright[terminal.lhs, left]:
					maxsplitright[terminal.lhs, left] = right
				recognized = True
		if not recognized and tags and tags[i] in grammar.toid:
			lhs = grammar.toid[tags[i]]
			item = new_ChartItem(lhs, 1ULL << i)
			sibling = new_ChartItem(Epsilon, i)
			viterbi[lhs, left, right] = 0.0
			chart[item] = [new_Edge(0.0, 0.0, 0.0, sibling, NONE)]
			# update filter
			if left > minsplitleft[lhs, right]:
				minsplitleft[lhs, right] = left
			if left < maxsplitleft[lhs, right]:
				maxsplitleft[lhs, right] = left
			if right < minsplitright[lhs, left]:
				minsplitright[lhs, left] = right
			if right > maxsplitright[lhs, left]:
				maxsplitright[lhs, left] = right
			recognized = True
		elif not recognized:
			logging.error("not covered: '%s'. " % (tags[i] if tags else w))
			return chart, NONE

		# unary rules on the span of this POS tag
		applied.clear()
		for rhs1 in range(grammar.nonterminals):
			if not isfinite(viterbi[rhs1, left, right]): continue
			for i in grammar.unaryclosure[rhs1]:
				rule = grammar.unary[0][i]
				if not isfinite(viterbi[rule.rhs1, left, right]): continue
				elif i in applied: continue
				else: applied.add(i)
				prob = rule.prob + viterbi[rule.rhs1, left, right]
				if isfinite(viterbi[rule.lhs, left, right]):
					chart[new_ChartItem(rule.lhs, (1ULL << right)
						- (1ULL << left))].append(
						new_Edge(prob, prob, rule.prob,
						new_ChartItem(rule.rhs1,
						(1ULL << right) - (1ULL << left)),
						NONE))
				else:
					chart[new_ChartItem(rule.lhs, (1ULL << right)
						- (1ULL << left))] = [
						new_Edge(prob, prob, rule.prob,
						new_ChartItem(rule.rhs1,
						(1ULL << right) - (1ULL << left)),
						NONE)]
				if prob < viterbi[rule.lhs, left, right]:
					viterbi[rule.lhs, left, right] = prob
					# update filter
					if left > minsplitleft[rule.lhs, right]:
						minsplitleft[rule.lhs, right] = left
					if left < maxsplitleft[rule.lhs, right]:
						maxsplitleft[rule.lhs, right] = left
					if right < minsplitright[rule.lhs, left]:
						minsplitright[rule.lhs, left] = right
					if right > maxsplitright[rule.lhs, left]:
						maxsplitright[rule.lhs, left] = right

	for span in range(2, lensent + 1):
		# print span,
		sys.stdout.flush()

		# constituents from left to right
		for left in range(0, lensent - span + 1):
			right = left + span
			# binary rules
			for i in range(grammar.numrules):
				rule = grammar.bylhs[0][i]
				if rule.lhs == grammar.nonterminals: break
				elif not rule.rhs2: continue
				#if not (np.isfinite(viterbi[rule.rhs1,left,left+1:right]).any()
				#and np.isfinite(viterbi[rule.rhs2,left:right-1,right]).any()):
				#	continue

				narrowr = minsplitright[rule.rhs1, left]
				if narrowr >= right: continue
				narrowl = minsplitleft[rule.rhs2, right]
				if narrowl < narrowr: continue
				widel = maxsplitleft[rule.rhs2, right]
				minmid = narrowr if narrowr > widel else widel
				wider = maxsplitright[rule.rhs1, left]
				maxmid = 1 + (wider if wider < narrowl else narrowl)
				oldscore = viterbi[rule.lhs, left, right]
				foundbetter = False
				for mid in range(minmid, maxmid):
					if (isfinite(viterbi[rule.rhs1, left, mid])
						and isfinite(viterbi[rule.rhs2, mid, right])):
						prob = (rule.prob + viterbi[rule.rhs1, left, mid]
								+ viterbi[rule.rhs2, mid, right])
						if isfinite(viterbi[rule.lhs, left, right]):
							chart[new_ChartItem(rule.lhs, (1ULL << right)
								- (1ULL << left))].append(
								new_Edge(prob, prob, rule.prob,
								new_ChartItem(rule.rhs1,
									(1ULL << mid) - (1ULL << left)),
								new_ChartItem(rule.rhs2,
									(1ULL << right) - (1ULL << mid))))
						else:
							chart[new_ChartItem(rule.lhs, (1ULL << right)
								- (1ULL << left))] = [
								new_Edge(prob, prob, rule.prob,
								new_ChartItem(rule.rhs1,
									(1ULL << mid) - (1ULL << left)),
								new_ChartItem(rule.rhs2,
									(1ULL << right) - (1ULL << mid)))]
						if prob < viterbi[rule.lhs, left, right]:
							foundbetter = True
							viterbi[rule.lhs, left, right] = prob
				# update filter
				if foundbetter and isinf(oldscore):
					if left > minsplitleft[rule.lhs, right]:
						minsplitleft[rule.lhs, right] = left
					if left < maxsplitleft[rule.lhs, right]:
						maxsplitleft[rule.lhs, right] = left
					if right < minsplitright[rule.lhs, left]:
						minsplitright[rule.lhs, left] = right
					if right > maxsplitright[rule.lhs, left]:
						maxsplitright[rule.lhs, left] = right

			# unary rules on this span
			applied.clear()
			for rhs1 in range(grammar.nonterminals):
				if not isfinite(viterbi[rhs1, left, right]): continue
				for i in grammar.unaryclosure[rhs1]:
					rule = grammar.unary[0][i]
					if not isfinite(viterbi[rule.rhs1, left, right]): continue
					elif i in applied: continue
					else: applied.add(i)
					prob = rule.prob + viterbi[rule.rhs1, left, right]
					if isfinite(viterbi[rule.lhs, left, right]):
						chart[new_ChartItem(rule.lhs, (1ULL << right)
							- (1ULL << left))].append(
							new_Edge(prob, prob, rule.prob,
							new_ChartItem(rule.rhs1,
							(1ULL << right) - (1ULL << left)),
							NONE))
					else:
						chart[new_ChartItem(rule.lhs, (1ULL << right)
							- (1ULL << left))] = [
							new_Edge(prob, prob, rule.prob,
							new_ChartItem(rule.rhs1,
							(1ULL << right) - (1ULL << left)),
							NONE)]
					if prob < viterbi[rule.lhs, left, right]:
						viterbi[rule.lhs, left, right] = prob
						# update filter
						if left > minsplitleft[rule.lhs, right]:
							minsplitleft[rule.lhs, right] = left
						if left < maxsplitleft[rule.lhs, right]:
							maxsplitleft[rule.lhs, right] = left
						if right < minsplitright[rule.lhs, left]:
							minsplitright[rule.lhs, left] = right
						if right > maxsplitright[rule.lhs, left]:
							maxsplitright[rule.lhs, left] = right
			#for lhs in range(grammar.nonterminals):
			#	best = cur = viterbi[lhs, left, right]
			#	newitem = new_ChartItem(lhs, (1ULL << right) - (1ULL << left))
			#	for i in grammar.unaryclosure[lhs]:
			#		rule = grammar.unary[0][i]
			#		if lhs == rule.rhs1: continue
			#		if not isfinite(viterbi[rule.rhs1, left, right]): continue
			#		prob = rule.prob + viterbi[rule.rhs1, left, right]
			#		if prob < best: best = prob
			#		else:
			#			# a suboptimal edge
			#			chart[newitem].append(
			#				new_Edge(prob, prob, rule.prob,
			#				new_ChartItem(rule.rhs1,
			#				(1ULL << right) - (1ULL << left)),
			#				NONE))
			#	if best < cur:
			#		viterbi[rule.lhs, left, right] = prob
			#		if isfinite(viterbi[rule.lhs, left, right]):
			#			chart[newitem].append(
			#				new_Edge(prob, prob, rule.prob,
			#				new_ChartItem(rule.rhs1,
			#				(1ULL << right) - (1ULL << left)),
			#				NONE))
			#		else:
			#			chart[newitem] = [new_Edge(prob, prob, rule.prob,
			#				new_ChartItem(rule.rhs1,
			#					(1ULL << right) - (1ULL << left)),
			#				NONE)]
			#		# update filter
			#		if left > minsplitleft[rule.lhs, right]:
			#			minsplitleft[rule.lhs, right] = left
			#		if left < maxsplitleft[rule.lhs, right]:
			#			maxsplitleft[rule.lhs, right] = left
			#		if right < minsplitright[rule.lhs, left]:
			#			minsplitright[rule.lhs, left] = right
			#		if right > maxsplitright[rule.lhs, left]:
			#			maxsplitright[rule.lhs, left] = right
	# print
	if goal in chart: return chart, goal
	else: return chart, NONE

cdef inline Edge iscore(Edge e):
	""" Replace estimate with inside probability """
	e.score = e.inside
	return e

def binrepr(ULLong vec, n, cfg=False):
	if cfg:
		start = nextset(vec, 0)
		return "%d-%d" % (start, nextunset(vec, start))
	return bin(vec)[2:].rjust(n, "0")[::-1]

def pprint_chart(chart, sent, tolabel, cfg=False):
	""" `pretty print' a chart. """
	cdef ChartItem a
	cdef Edge edge
	print "chart:"
	for a in sorted(chart, key=lambda a: (bitcount(a.vec), a.vec)):
		if chart[a] == []: continue
		print "%s[%s] =>" % (tolabel[a.label], binrepr(a.vec, len(sent), cfg))
		if isinstance(chart[a], float): continue
		for edge in chart[a]:
			print "%g\t%g" % (exp(-edge.inside), exp(-edge.prob)),
			if edge.left.label:
				print "\t%s[%s]" % (tolabel[edge.left.label],
						binrepr(edge.left.vec, len(sent), cfg)),
			else:
				print "\t", repr(sent[edge.left.vec]),
			if edge.right:
				print "\t%s[%s]" % (tolabel[edge.right.label],
						binrepr(edge.right.vec, len(sent), cfg)),
			print
		print

def do(sent, grammar):
	from disambiguation import marginalize
	from operator import itemgetter
	print "sentence", sent
	chart, start, _ = parse(sent.split(), grammar, start=grammar.toid['S'])
	pprint_chart(chart, sent.split(), grammar.tolabel)
	if start == NONE:
		print "no parse"
		return False
	else:
		print "10 best parse trees:"
		mpp, _ = marginalize(chart, start, grammar.tolabel)
		for a, p in reversed(sorted(mpp.items(), key=itemgetter(1))): print p,a
		print
		return True

def main():
	from containers import Grammar
	grammar = Grammar([
		((('S','VP2','VMFIN'), ((0,1,0),)), 0.0),
		((('VP2','VP2','VAINF'),  ((0,),(0,1))), log(0.5)),
		((('VP2','PROAV','VVPP'), ((0,),(1,))), log(0.5)),
		((('PROAV', 'Epsilon'), ('Daruber',)), 0.0),
		((('VAINF', 'Epsilon'), ('werden',)), 0.0),
		((('VMFIN', 'Epsilon'), ('muss',)), 0.0),
		((('VVPP', 'Epsilon'), ('nachgedacht',)), 0.0)])
	print grammar
	print "Rule structs take", sizeof(Rule), "bytes"

	assert do("Daruber muss nachgedacht werden", grammar)
	assert do("Daruber muss nachgedacht werden werden", grammar)
	assert do("Daruber muss nachgedacht werden werden werden", grammar)
	print "ungrammatical sentence:"
	assert not do("muss Daruber nachgedacht werden", grammar)	#no parse
	print "(as expected)"
	print "long sentence (33 words):"
	assert do("Daruber muss nachgedacht %s" % " ".join(30*["werden"]), grammar)

	cfg = Grammar([
		((('S', 'NP', 'VP'), ((0,1),)), 0.0),
		((('NP', 'Epsilon'), ('mary',)), 0.0),
		((('VP', 'Epsilon'), ('walks',)), 0.0)])
	print "cfg parsing; sentence: mary walks"
	chart, start = cfgparse("mary walks".split(), cfg, start=grammar.toid['S'])
	pprint_chart(chart, "mary walks".split(), cfg.tolabel)
	assert start

if __name__ == '__main__': main()
