""" Probabilistic CKY parser for Simple Range Concatenation Grammars
(equivalent to Linear Context-Free Rewriting Systems)"""
from math import log, exp
from array import array
from collections import defaultdict
import re, logging, sys
import numpy as np
from agenda import EdgeAgenda, Entry
from estimates cimport getoutside
np.import_array()

DEF infinity = float('infinity')
# a regex to strip off arity markers and binarization introduced by the
# fragments of Double DOP (e.g., "NP_2}<...>") or addresses from the DOP
# reduction (e.g., "NP_2@12")
striplabel = re.compile(r"(?:_[0-9]+)?(?:@[0-9]+)?$")

# to avoid overhead of __init__ and __cinit__ constructors
# belongs in containers but putting it here gives
# a better chance of successful inlining
cdef inline ChartItem new_ChartItem(UInt label, ULLong vec):
	cdef ChartItem item = ChartItem.__new__(ChartItem)
	item.label = label; item.vec = vec
	return item

cdef inline Edge new_Edge(double score, double inside, double prob,
	ChartItem left, ChartItem right):
	cdef Edge edge = Edge.__new__(Edge)
	edge.score = score; edge.inside = inside; edge.prob = prob
	edge.left = left; edge.right = right
	return edge


def parse(sent, Grammar grammar, tags=None, start=1, bint exhaustive=False,
			estimate=None, list whitelist=None, dict coarsechart=None,
			Grammar coarsegrammar=None, bint splitprune=False,
			bint markorigin=False, int beamwidth=0):
	""" parse sentence, a list of tokens, optionally with gold tags, and
	produce a chart, either exhaustive or up until the viterbi parse.
	Other parameters:
		- start: integer corresponding to the start tag in the grammar,
			e.g., grammar.toid['ROOT']
		- exhaustive: don't stop at viterbi parser, return a full chart
		- estimate: use context-summary estimates (heuristics) to order agenda.
			if estimates are not consistent, it is no longer guaranteed that
			the optimal parse will be found. experimental.
		- whitelist: a whitelist of allowed ChartItems. Anything else is not
			added to the agenda.
		- coarsechart: the chart from the coarse phase. used when coarse stage
			was continuous while fine stage is discontinuous.
		- coarsegrammar: the coarse grammar being pruned with;
			required for `coarsechart.'
		- splitprune: coarse stage used a split-PCFG where discontinuous node
			appear as multiple CFG nodes. Every discontinuous node will result
			in multiple lookups into coarsechart to see whether it should be
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
	cdef EdgeAgenda agenda = EdgeAgenda()			#the agenda
	cdef size_t i
	cdef Rule rule
	cdef LexicalRule terminal
	cdef Entry entry
	cdef Edge edge, newedge
	cdef ChartItem item, sibling, NONE = new_ChartItem(0, 0)
	cdef ChartItem goal = new_ChartItem(start, (1ULL << len(sent)) - 1)
	cdef np.ndarray[np.double_t, ndim=4] outside
	cdef bint doestimate = bool(estimate), prunenow = False, split = False
	cdef bint doprune = bool(whitelist) or bool(coarsechart)
	cdef double x = 0.0, y = 0.0
	cdef signed int length = 0, left = 0, right = 0, gaps = 0
	cdef signed int lensent = len(sent), maxlen = 0
	cdef UInt blocked = 0, Epsilon = grammar.toid["Epsilon"]
	cdef ULong maxA = 0
	cdef ULLong vec = 0

	assert len(sent) < (sizeof(vec) * 8)
	if splitprune: assert coarsegrammar is not None
	if doestimate:
		outside, maxlen = estimate
		#assert len(grammar.bylhs) == len(outside)
		assert lensent <= maxlen

	# scan
	for i, w in enumerate(sent):
		recognized = False
		for terminal in grammar.lexical.get(w, []):
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if (not tags or grammar.tolabel[terminal.lhs] == tags[i]
					or grammar.tolabel[terminal.lhs].startswith(tags[i] + "@")):
				item = new_ChartItem(terminal.lhs, 1ULL << i)
				sibling = new_ChartItem(Epsilon, i)
				x = terminal.prob
				if doestimate:
					y = getoutside(outside, maxlen, lensent,
						item.label, item.vec)
				process_edge(item, new_Edge(x + y, x, x, sibling, NONE),
					agenda, chart, viterbi, grammar, exhaustive, doprune,
					whitelist, coarsegrammar, coarsechart, 1, splitprune,
					markorigin, &blocked)
				recognized = True
		if not recognized and tags and tags[i] in grammar.toid:
			item = new_ChartItem(grammar.toid[tags[i]], 1ULL << i)
			sibling = new_ChartItem(Epsilon, i)
			if doestimate:
				y = getoutside(outside, maxlen, lensent, item.label, item.vec)
			agenda[item] = new_Edge(y, 0.0, 0.0, sibling, NONE)
			chart[item] = []
			recognized = True
		elif not recognized:
			return chart, NONE, "not covered: %s" % (tags[i] if tags else w)

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
			if doestimate:
				length = bitcount(item.vec); left = nextset(item.vec, 0)
				gaps = bitlength(item.vec) - length - left
				right = lensent - length - left - gaps
			if not beamwidth or beam[item.vec] < beamwidth:
				for i in range(grammar.numrules):
					rule = grammar.unary[item.label][i]
					if rule.rhs1 != item.label: break
					if doestimate:
						newedge = new_Edge(
									outside[rule.lhs, length, left+right, gaps]
									+ x + rule.prob, x + rule.prob,
									rule.prob, item, NONE)
					else:
						newedge = new_Edge(x + rule.prob, x + rule.prob,
											rule.prob, item, NONE)
					process_edge(new_ChartItem(rule.lhs, item.vec), newedge,
						agenda, chart, viterbi, grammar, exhaustive, doprune,
						whitelist, coarsegrammar, coarsechart, rule.fanout,
						splitprune, markorigin, &blocked)

			# binary left
			for i in range(grammar.numrules):
				rule = grammar.lbinary[item.label][i]
				if rule.rhs1 != item.label: break
				for I, e in (<dict>viterbi[rule.rhs2]).iteritems():
					sibling = <ChartItem>I
					if ((not beamwidth or beam[item.vec ^ sibling.vec]
						< beamwidth) and concat(rule, item.vec, sibling.vec)):
						beam[item.vec ^ sibling.vec] += 1
						y = (<Edge>e).inside
						vec = item.vec ^ sibling.vec
						if doestimate:
							length = bitcount(vec); left = nextset(vec, 0)
							gaps = bitlength(vec) - length - left
							right = lensent - length - left - gaps
							newedge = new_Edge(
									outside[rule.lhs, length, left+right, gaps]
									+x+y+rule.prob, x+y+rule.prob,
									rule.prob, item, sibling)
						else:
							newedge = new_Edge(x+y+rule.prob, x+y+rule.prob,
											rule.prob, item, sibling)
						process_edge(new_ChartItem(rule.lhs, vec), newedge,
							agenda, chart, viterbi, grammar, exhaustive,
							doprune, whitelist, coarsegrammar, coarsechart,
							rule.fanout, splitprune, markorigin,
							&blocked)

			# binary right
			for i in range(grammar.numrules):
				rule = grammar.rbinary[item.label][i]
				if rule.rhs2 != item.label: break
				for I, e in (<dict>viterbi[rule.rhs1]).iteritems():
					sibling = <ChartItem>I
					if ((not beamwidth or beam[sibling.vec ^ item.vec]
						< beamwidth) and concat(rule, sibling.vec, item.vec)):
						beam[sibling.vec ^ item.vec] += 1
						y = (<Edge>e).inside
						vec = sibling.vec ^ item.vec
						if doestimate:
							length = bitcount(vec); left = nextset(vec, 0)
							gaps = bitlength(vec) - length - left
							right = lensent - length - left - gaps
							newedge = new_Edge(
									outside[rule.lhs, length, left+right, gaps]
									+x+y+rule.prob, x+y+rule.prob,
									rule.prob, sibling, item)
						else:
							newedge = new_Edge(x+y+rule.prob, x+y+rule.prob,
											rule.prob, sibling, item)
						process_edge(new_ChartItem(rule.lhs, vec), newedge,
							agenda, chart, viterbi, grammar, exhaustive,
							doprune, whitelist, coarsegrammar, coarsechart,
							rule.fanout, splitprune, markorigin,
							&blocked)

		if agenda.length > maxA: maxA = agenda.length
		#if agenda.length % 1000 == 0:
		#	logging.debug(
		#		"agenda max %d, now %d, items %d (%d labels), edges %d, blocked %d"
		#		% (maxA, len(agenda), len(filter(None, chart.values())),
		#		len(filter(None, viterbi)), sum(map(len, chart.values())), blocked))
	msg = "agenda max %d, now %d, items %d (%d labels), edges %d, blocked %d" % (
		maxA, len(agenda), len(filter(None, chart.values())),
		len(filter(None, viterbi)), sum(map(len, chart.values())), blocked)
	if goal in chart: return chart, goal, msg
	else: return chart, NONE, "no parse " + msg

cdef inline process_edge(ChartItem newitem, Edge newedge,
		EdgeAgenda agenda, dict chart, list viterbi, Grammar grammar,
		bint exhaustive, bint doprune, list whitelist, Grammar coarsegrammar,
		dict coarsechart, UChar fanout, bint splitprune, bint markorigin,
		UInt *blocked):
	""" Decide what to do with a newly derived edge. """
	cdef ULLong component, vec
	cdef UInt a, b, origlabel, cnt
	cdef bint inagenda = agenda.contains(newitem)
	cdef bint inchart = PyDict_Contains(chart, newitem) == 1
	cdef str label
	if not inagenda and not inchart:
		if doprune and newitem.label not in grammar.donotprune:
			# disc. item to be treated as several split items?
			if splitprune and fanout > 1:
				origlabel = newitem.label; vec = newitem.vec
				label = striplabel.sub("*", grammar.tolabel[newitem.label], 1)
				if not markorigin:
					try: newitem.label = coarsegrammar.toid[label]
					except KeyError:
						blocked[0] += 1
						return
				a = b = cnt = 0
				while vec >> b:
					if markorigin:
						try:
							newitem.label = coarsegrammar.toid[label + str(cnt)]
						except KeyError:
							blocked[0] += 1
							return
					a = nextset(vec, b)
					b = nextunset(vec, a)
					#given a=3, b=6. from left to right: 
					#10000 => 01111 => 01111000
					component = (1ULL << b) - (1ULL << a)
					newitem.vec = component
					#if outside==NULL or isinf(<double><object>outside):
					if newitem not in coarsechart:
						blocked[0] += 1
						return
					cnt += 1
				newitem.label = origlabel; newitem.vec = vec
			else:
				if coarsechart is None:
					#outside = whitelist[newitem.label][newitem.vec]
					outside = PyDict_GetItem(whitelist[newitem.label], newitem.vec)
					# need a double cast: before outside can be converted to
					# a double, it needs to be treated as a python float
					if (outside==NULL or isinf(<double>(<object>outside))):
						#or newedge.inside+<double><object>outside > 300.0):
						blocked[0] += 1
						return
				else:
					label = striplabel.sub("", grammar.tolabel[newitem.label], 1)
					origlabel = newitem.label
					newitem.label = coarsegrammar.toid[label]
					if newitem not in coarsechart:
						blocked[0] += 1
						return
					newitem.label = origlabel
		#elif newedge.score > 300.0:
		#	blocked[0] += 1
		#	return

		# haven't seen this item before, won't prune, add to agenda
		agenda.setitem(newitem, newedge)
		chart[newitem] = []
	# in agenda (maybe in chart)
	elif not exhaustive and inagenda:
		agenda.setifbetter(newitem, newedge)
	elif (inagenda
		and newedge.inside < (<Edge>(agenda.getitem(newitem))).inside):
		# item has lower score, decrease-key in agenda
		# (add old, suboptimal edge to chart if parsing exhaustively)
		chart[newitem].append(iscore(agenda.replace(newitem, newedge)))
	# not in agenda => must be in chart
	elif (not inagenda and newedge.inside <
				(<Edge>(<dict>viterbi[newitem.label])[newitem]).inside):
		#re-add to agenda because we found a better score
		agenda.setitem(newitem, newedge)
		logging.warning("WARN: re-adding item to agenda: %r" % (newitem))
	elif exhaustive:
		# suboptimal edge
		chart[newitem].append(iscore(newedge))

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

	update: yield functions are now encoded in binary arrays:
		( (0, 1, 0), (1, 0) ) ==> array('H', [0b010, 0b01])
							and lengths: array('B', [3, 2])
		NB: note reversal due to the way binary numbers are represented
		the least significant bit (rightmost) corresponds to the lowest
		index in the sentence / constituent (leftmost).
	"""
	if lvec & rvec: return False
	cdef int lpos = nextset(lvec, 0)
	cdef int rpos = nextset(rvec, 0)
	cdef UInt n

	# if there are no gaps in lvec and rvec, and the yieldfunction is the
	# concatenation of two elements, then this should^Wcould be quicker
	#if rule._lengths[0] == 2 and rule.args.length == 1:
	#	if (lvec >> nextunset(lvec, lpos) == 0
	#		and rvec >> nextunset(rvec, rpos) == 0):
	#		if rule.args[0] == 0b10:
	#			return bitminmax(lvec, rvec)
	#		elif rule.args[0] == 0b01:
	#			return bitminmax(rvec, lvec)

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
	cdef long numsymbols = len(grammar.toid), lhs
	cdef double oldscore, prob
	cdef size_t i
	cdef UInt Epsilon = grammar.toid["Epsilon"]
	cdef ULLong vec = 0
	cdef bint foundnew = False, foundbetter = False
	cdef Rule rule
	cdef LexicalRule terminal
	cdef ChartItem NONE = ChartItem(0, 0)
	cdef ChartItem goal = ChartItem(start, (1ULL << len(sent)) - 1)
	cdef dict chart = {}						#the full chart
	cdef set unaryrules, candidates
	# the viterbi chart is initially filled with infinite log probabilities,
	# cells which are to be blocked contain NaN.
	cdef np.ndarray[np.double_t, ndim=3] viterbi
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
	viterbi = np.array([np.inf],
		dtype='d').repeat(lensent * (lensent+1) * numsymbols).reshape(
		(numsymbols, lensent, (lensent+1)))

	assert len(sent) < (sizeof(vec) * 8), ("sentence too long. "
			"length: %d. limit: %d." % (len(sent), sizeof(vec) * 8))
	for i in range(grammar.numrules):
		if grammar.unary[0][i].rhs1 == grammar.nonterminals:
			unaryrules = set(range(i))

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
			logging.error("not covered: %s" % (tags[i] if tags else w))
			return chart, NONE

		# unary rules on the span of this POS tag
		# only use each rules once (no cycles)
		# keep going until no new items can be derived.
		foundnew = True
		candidates = set(unaryrules)
		while foundnew:
			foundnew = False
			for i in set(candidates):
				rule = grammar.unary[0][i]
				if rule.rhs1 == grammar.nonterminals: break
				elif isfinite(viterbi[rule.rhs1, left, right]):
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
					candidates.discard(i)
					foundnew = True
					break

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
			# only use each rules once (no cycles)
			# keep going until no new items can be derived.
			foundnew = True
			candidates = set(unaryrules)
			while foundnew:
				foundnew = False
				for i in candidates:
					rule = grammar.unary[0][i]
					if rule.rhs1 == grammar.nonterminals: break
					elif isfinite(viterbi[rule.rhs1, left, right]):
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
						candidates.discard(i)
						foundnew = True
						break
	# print
	if goal in chart: return chart, goal
	else: return chart, NONE

cdef inline Edge iscore(Edge e):
	""" Replace estimate with inside probability """
	e.score = e.inside
	return e

def binrepr(ChartItem a, sent, cfg=False):
	if cfg:
		start = nextset(a.vec, 0)
		return "%d-%d" % (start, nextunset(a.vec, start))
	return bin(a.vec)[2:].rjust(len(sent), "0")[::-1]

def pprint_chart(chart, sent, tolabel, cfg=False):
	""" `pretty print' a chart. """
	cdef ChartItem a
	cdef Edge edge
	print "chart:"
	for a in sorted(chart, key=lambda a: (bitcount(a.vec), a.vec)):
		if chart[a] == []: continue
		print "%s[%s] =>" % (tolabel[a.label], binrepr(a, sent, cfg))
		if isinstance(chart[a], float): continue
		for edge in chart[a]:
			print "%g\t%g" % (exp(-edge.inside), exp(-edge.prob)),
			if edge.left.label:
				print "\t%s[%s]" % (tolabel[edge.left.label],
						binrepr(edge.left, sent, cfg)),
			else:
				print "\t", repr(sent[edge.left.vec]),
			if edge.right:
				print "\t%s[%s]" % (tolabel[edge.right.label],
						binrepr(edge.right, sent, cfg)),
			print
		print

def do(sent, grammar):
	from disambiguation import marginalize
	from operator import itemgetter
	print "sentence", sent
	chart, start, _ = parse(sent.split(), grammar, start=grammar.toid['S'])
	pprint_chart(chart, sent.split(), grammar.tolabel)
	if start == new_ChartItem(0, 0):
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
		((('PROAV', 'Epsilon'), ('Daruber', ())), 0.0),
		((('VAINF', 'Epsilon'), ('werden', ())), 0.0),
		((('VMFIN', 'Epsilon'), ('muss', ())), 0.0),
		((('VVPP', 'Epsilon'), ('nachgedacht', ())), 0.0)])
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
		((('NP', 'Epsilon'), ('mary', ())), 0.0),
		((('VP', 'Epsilon'), ('walks', ())), 0.0)])
	print "cfg parsing; sentence: mary walks"
	chart, start = cfgparse("mary walks".split(), cfg, start=grammar.toid['S'])
	pprint_chart(chart, "mary walks".split(), cfg.tolabel)
	assert start

if __name__ == '__main__': main()
