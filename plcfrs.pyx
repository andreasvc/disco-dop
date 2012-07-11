""" Probabilistic CKY parser for Simple Range Concatenation Grammars
(equivalent to Linear Context-Free Rewriting Systems)"""
from math import log, exp
from array import array
from collections import defaultdict
import re, gc, logging, sys
import numpy as np
from agenda import EdgeAgenda, Entry
from estimates cimport getoutside
from containers cimport yfrepr #FIXME temp
#from containers import ChartItem, Edge, Rule, LexicalRule
np.import_array()

DEF infinity = float('infinity')

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

def parse(sent, Grammar grammar, tags=None, start=None, bint exhaustive=False,
			estimate=None, list prunelist=None, dict prunetoid=None,
			dict coarsechart=None, bint splitprune=False, bint markorigin=False,
			bint neverblockmarkovized=False, bint neverblockdiscontinuous=False,
			int beamwidth=0):
	""" parse sentence, a list of tokens, optionally with gold tags, and
	produce a chart, either exhaustive or up until the viterbi parse.
	Other parameters:
		- start: integer corresponding to the start tag in the grammar,
			e.g., grammar.toid['ROOT']
		- exhaustive: don't stop at viterbi parser, return a full chart
		- estimate: use context-summary estimates (heuristics) to order agenda.
			if estimates are not consistent, it is no longer guaranteed that
			the optimal parse will be found. experimental.
		- prunelist: a whitelist of allowed ChartItems. Anything else is not
			added to the agenda.
		- prunetoid: the mapping of string labels to integers, e.g. 'NP' -> 3
		- coarsechart: the chart from the coarse phase. used when coarse stage
			was continuous while fine stage is discontinuous.
		- splitprune: coarse stage used a split-PCFG where discontinuous node
			appear as multiple CFG nodes. Every discontinuous node will result
			in multiple lookups into coarsechart to see whether it should be
			allowed on the agenda.
		- markorigin: in combination with splitprune, coarse labels include an
			integer to distinguish components; e.g., CFG nodes NP*0 and NP*1
			map to the discontinuous node NP_2
		- neverblockmarkovized: do not block nodes introduced by binarization;
			useful if coarse and fine stages employ different kinds of
			markovization; e.g., NP and VP may be blocked, but not NP|<DT-NN>.
		- neverblockdiscontinuous: same as above but for discontinuous
			nodes X_n where X is a label and n is a fanout > 1.
		- beamwidth: specify the maximum number of items that will be explored
			for each particular span, on a first-come-first-served basis.
			setting to 0 disables this feature. experimental.
	"""
	cdef dict lexical = grammar.lexical
	cdef dict toid = grammar.toid
	cdef dict tolabel = grammar.tolabel
	cdef dict beam = <dict>defaultdict(int)		#table of bit vectors to counts
	cdef dict chart = {}						#the full chart
	cdef list viterbi = [{} for _ in toid]		#the viterbi probabilities
	cdef EdgeAgenda agenda = EdgeAgenda()		#the agenda
	cdef size_t i
	cdef Entry entry
	cdef Edge edge, newedge
	cdef ChartItem NONE = new_ChartItem(0, 0)
	cdef ChartItem item, sibling, goal = NONE
	cdef LexicalRule terminal
	cdef Rule rule
	cdef np.ndarray[np.double_t, ndim=4] outside
	cdef bint doestimate = bool(estimate), prunenow, split
	cdef bint doprune = bool(prunelist) or bool(coarsechart)
	cdef signed int length = 0, left = 0, right = 0, gaps = 0
	cdef signed int lensent = len(sent), maxlen = 0
	cdef UInt blocked = 0, Epsilon = toid["Epsilon"]
	cdef ULLong vec = 0
	cdef unsigned long maxA = 0
	cdef double x, y = 0.0
	cdef str label = ''

	if start == None: start = toid["ROOT"]
	assert len(sent) < (sizeof(vec) * 8)
	if splitprune: assert prunetoid is not None
	else: coarsechart = None # ??? FIXME
	vec = (1ULL << len(sent)) - 1
	goal = new_ChartItem(start, vec)

	if doestimate:
		outside, maxlen = estimate
		#assert len(grammar.bylhs) == len(outside)
		assert lensent <= maxlen

	gc.disable() #is this actually beneficial?
	# maybe we can disable refcounting altogether?

	# scan
	for i, w in enumerate(sent):
		recognized = False
		for terminal in lexical.get(w, []):
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if (not tags or tags[i] == tolabel[terminal.lhs].split("@")[0]
					or tolabel[terminal.lhs].startswith("#%s" % tags[i])):
				item = new_ChartItem(terminal.lhs, 1ULL << i)
				sibling = new_ChartItem(Epsilon, i)
				x = terminal.prob
				if doestimate:
					y = getoutside(outside, maxlen, lensent,
						item.label, item.vec)
				agenda[item] = new_Edge(x + y, x, x, sibling, NONE)
				chart[item] = []
				recognized = True
		if not recognized and tags and tags[i] in toid:
			item = new_ChartItem(toid[tags[i]], 1ULL << i)
			sibling = new_ChartItem(Epsilon, i)
			if doestimate:
				y = getoutside(outside, maxlen, lensent, item.label, item.vec)
			agenda[item] = new_Edge(y, 0.0, 0.0, sibling, NONE)
			chart[item] = []
			recognized = True
		elif not recognized:
			logging.error("not covered: %s" % (tags[i] if tags else w))
			return chart, NONE

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
					prunenow = doprune
					label = ''
					split = False
					if prunenow and neverblockdiscontinuous:
						prunenow = (rule.fanout == 1)
					if prunenow and neverblockmarkovized:
						prunenow = "|" not in tolabel[rule.lhs]
					if prunenow and splitprune:
						label = tolabel[rule.lhs]
						if rule.fanout > 1:
							label = label[:label.rindex("_")]
							split = True
						else:
							left = label.find("@")
							if left != -1: label = label[:left]
					process_edge(new_ChartItem(rule.lhs, item.vec), newedge,
						agenda, chart, viterbi, exhaustive, prunenow,
						prunelist, prunetoid, coarsechart, label, split,
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
						prunenow = doprune
						label = ''
						split = False
						if prunenow and neverblockdiscontinuous:
							prunenow = (rule.fanout == 1)
						if prunenow and neverblockmarkovized:
							prunenow = "|" not in tolabel[rule.lhs]
						if prunenow and splitprune:
							label = tolabel[rule.lhs]
							if rule.fanout > 1:
								label = label[:label.rindex("_")]
								split = True
							else:
								left = label.find("@")
								if left != -1: label = label[:left]
						process_edge(new_ChartItem(rule.lhs, vec), newedge,
							agenda, chart, viterbi, exhaustive, prunenow,
							prunelist, prunetoid, coarsechart, label, split,
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
						prunenow = doprune
						label = ''
						split = False
						if prunenow and neverblockdiscontinuous:
							prunenow = (rule.fanout == 1)
						if prunenow and neverblockmarkovized:
							prunenow = "|" not in tolabel[rule.lhs]
						if prunenow and splitprune:
							label = tolabel[rule.lhs]
							if rule.fanout > 1:
								label = label[:label.rindex("_")]
								split = True
							else:
								left = label.find("@")
								if left != -1: label = label[:left]
						process_edge(new_ChartItem(rule.lhs, vec), newedge,
							agenda, chart, viterbi, exhaustive, prunenow,
							prunelist, prunetoid, coarsechart, label, split,
							markorigin, &blocked)

		if agenda.length > maxA: maxA = agenda.length
		#if agenda.length % 1000 == 0:
		#	logging.debug(
		#		"agenda max %d, now %d, items %d (%d labels), edges %d, blocked %d"
		#		% (maxA, len(agenda), len(filter(None, chart.values())),
		#		len(filter(None, viterbi)), sum(map(len, chart.values())), blocked))
	logging.debug(
		"agenda max %d, now %d, items %d (%d labels), edges %d, blocked %d"
		% (maxA, len(agenda), len(filter(None, chart.values())),
		len(filter(None, viterbi)), sum(map(len, chart.values())), blocked))
	gc.enable()
	if goal in chart: return chart, goal
	else: return chart, NONE

cdef inline void process_edge(ChartItem newitem, Edge newedge,
		EdgeAgenda agenda, dict chart, list viterbi, bint exhaustive,
		bint doprune, list prunelist, dict prunetoid, dict coarsechart,
		str label, bint split, bint markorigin, UInt *blocked) except *:
	""" Decide what to do with a newly derived edge. """
	cdef ULLong component, vec
	cdef UInt a = 0, b = 0, splitlabel = 0, origlabel
	cdef int cnt = 0
	cdef bint inagenda = agenda.contains(newitem)
	cdef bint inchart = PyDict_Contains(chart, newitem) == 1
	#not in agenda or chart
	if not (inagenda or inchart):
		if doprune:
			if split: #do we treat discontinuous items as several split items?
				origlabel = newitem.label; vec = newitem.vec
				if not markorigin:
					try: splitlabel = prunetoid[label + "*"]
					except KeyError:
						blocked[0] += 1
						return
				while vec >> b:
					if markorigin:
						try: splitlabel = prunetoid["%s*%d" % (label, cnt)]
						except KeyError:
							blocked[0] += 1
							return
					a = nextset(vec, b)
					b = nextunset(vec, a)
					#given a=3, b=6. from left to right: 
					#10000 => 01111 => 01111000
					component = (1ULL << b) - (1ULL << a)
					newitem.label = splitlabel; newitem.vec = component
					#if outside==NULL or isinf(<double><object>outside):
					if newitem not in coarsechart:
						blocked[0] += 1
						return
					cnt += 1
				newitem.label = origlabel; newitem.vec = vec
			else:
				if coarsechart is None:
					#outside = prunelist[newitem.label][newitem.vec]
					outside = PyDict_GetItem(prunelist[newitem.label], newitem.vec)
					# need a double cast: before outside can be converted to
					# a double, it needs to be treated as a python float
					if (outside==NULL or isinf(<double>(<object>outside))):
						#or newedge.inside+<double><object>outside > 300.0):
						blocked[0] += 1
						return
				else:
					origlabel = newitem.label
					newitem.label = prunetoid[label]
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

def cfgparse(list sent, Grammar grammar, start=None, tags=None):
	""" A CKY parser modeled after Bodenstab's `fast grammar loop.'
		and the Stanford parser. """
	cdef short left, right, mid, span, lensent = len(sent)
	cdef short narrowr, narrowl, widel, wider, minmid, maxmid
	cdef long numsymbols = len(grammar.toid), lhs
	cdef double oldscore, prob
	cdef size_t i
	cdef bint foundbetter = False
	cdef Rule rule
	cdef LexicalRule terminal
	cdef ChartItem NONE = new_ChartItem(0, 0)
	#cdef list chart = [[{} for _ in range(lensent+1)] for _ in range(lensent)]
	cdef dict chart = {}						#the full chart
	cdef UInt Epsilon = grammar.toid["Epsilon"]
	cdef ULLong vec = 0
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

	if start == None: start = grammar.toid["ROOT"]
	assert len(sent) < (sizeof(vec) * 8)
	vec = (1ULL << len(sent)) - 1
	goal = new_ChartItem(start, vec)

	# assign POS tags
	print 1, # == span
	for i, w in enumerate(sent):
		left = i; right = i + 1
		recognized = False
		for terminal in grammar.lexical.get(w, []):
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if (not tags
				or tags[i] == grammar.tolabel[terminal.lhs].rsplit("@", 1)[0]):
				item = new_ChartItem(terminal.lhs, 1ULL << i)
				sibling = new_ChartItem(Epsilon, i)
				x = terminal.prob
				viterbi[terminal.lhs, left, right] = terminal.prob
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

		# unary rules on POS tags 
		for i in range(grammar.numrules):
			rule = grammar.unary[0][i]
			if rule.rhs1 == grammar.nonterminals: break
			if isfinite(viterbi[rule.rhs1, left, right]):
				prob = rule.prob + viterbi[rule.rhs1, left, right]
				if isfinite(viterbi[rule.lhs, left, right]):
					chart[new_ChartItem(rule.lhs, 1ULL << left)].append(
						new_Edge(prob, prob, rule.prob,
						new_ChartItem(rule.rhs1, 1ULL << left), NONE))
				else:
					chart[new_ChartItem(rule.lhs, 1ULL << left)] = [
						new_Edge(prob, prob, rule.prob,
						new_ChartItem(rule.rhs1, 1ULL << left), NONE)]
				if (prob < viterbi[rule.lhs, left, right]):
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
		print span,
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

			# unary rules
			for _ in range(2):	#add up to 2 levels of unary nodes
				for i in range(grammar.numrules):
					rule = grammar.unary[0][i]
					if rule.rhs1 == grammar.nonterminals: break
					if isfinite(viterbi[rule.rhs1, left, right]):
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

	print
	if goal in chart: return chart, goal
	else: return chart, NONE

cdef inline Edge iscore(Edge e):
	""" Replace estimate with inside probability """
	e.score = e.inside
	return e

def binrepr(ChartItem a, sent):
	return bin(a.vec)[2:].rjust(len(sent), "0")[::-1]

def pprint_chart(chart, sent, tolabel):
	""" `pretty print' a chart. """
	cdef ChartItem a
	cdef Edge edge
	print "chart:"
	for a in sorted(chart, key=lambda a: (bitcount(a.vec), a.vec)):
		if not chart[a]: continue
		print "%s[%s] =>" % (tolabel[a.label], binrepr(a, sent))
		for edge in chart[a]:
			print "%g\t%g" % (exp(-edge.inside), exp(-edge.prob)),
			if edge.left.label:
				print "\t%s[%s]" % (tolabel[edge.left.label],
									binrepr(edge.left, sent)),
			else:
				print "\t", repr(sent[edge.left.vec]),
			if edge.right:
				print "\t%s[%s]" % (tolabel[edge.right.label],
									binrepr(edge.right, sent)),
			print
		print

def do(sent, grammar):
	from disambiguation import marginalize
	from operator import itemgetter
	print "sentence", sent
	chart, start = parse(sent.split(), grammar, start=grammar.toid['S'])
	pprint_chart(chart, sent.split(), grammar.tolabel)
	if start == new_ChartItem(0, 0):
		print "no parse"
		return False
	else:
		print "10 best parse trees:"
		mpp = marginalize(chart, start, grammar.tolabel)
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
