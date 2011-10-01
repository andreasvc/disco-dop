""" Probabilistic CKY parser for Simple Range Concatenation Grammars
(equivalent to Linear Context-Free Rewriting Systems)"""
from math import log, exp
from array import array
from collections import defaultdict
import re, gc, logging
import numpy as np
from agenda import EdgeAgenda, Entry
from estimates cimport getoutside
from containers import ChartItem, Edge, Rule, Terminal
np.import_array()

DEF infinity = float('infinity')

# to avoid overhead of __init__ and __cinit__ constructors
# belongs in containers but putting it here gives
# a better chance of successful inlining
cdef inline ChartItem new_ChartItem(unsigned int label, unsigned long long vec):
	cdef ChartItem item = ChartItem.__new__(ChartItem)
	item.label = label; item.vec = vec
	return item

cdef inline Edge new_Edge(double score, double inside, double prob,
	ChartItem left, ChartItem right):
	cdef Edge edge = Edge.__new__(Edge)
	edge.score = score; edge.inside = inside; edge.prob = prob
	edge.left = left; edge.right = right
	return edge

def parse(sent, grammar, tags=None, start=None, bint exhaustive=False,
			estimate=None, list prunelist=None, dict prunetoid=None,
			bint splitprune=False, bint markorigin=False,
			bint neverblockmarkovized=False, bint neverblockdiscontinuous=False,
			int beamwidth=0):
	""" parse sentence, a list of tokens, optionally with gold tags, and
	produce a chart, either exhaustive or up until the viterbi parse
	"""
	cdef array[unsigned char] arity = grammar.arity
	cdef list unary = grammar.unary
	cdef list lbinary = grammar.lbinary
	cdef list rbinary = grammar.rbinary
	cdef dict lexical = grammar.lexical
	cdef dict toid = grammar.toid
	cdef dict tolabel = grammar.tolabel
	cdef dict beam = <dict>defaultdict(int)		#table of bit vectors to counts
	cdef dict chart = {}						#the full chart
	cdef list viterbi = [{} for _ in toid]		#the viterbi probabilities
	cdef EdgeAgenda agenda = EdgeAgenda()		#the agenda
	cdef Py_ssize_t i
	cdef Entry entry
	cdef Edge edge, newedge
	cdef ChartItem NONE = new_ChartItem(0, 0)
	cdef ChartItem item, sibling, newitem, goal = NONE
	cdef Terminal terminal
	cdef Rule rule
	cdef np.ndarray[np.double_t, ndim=4] outside
	cdef bint doestimate = bool(estimate), doprune = bool(prunelist), prunenow
	cdef signed int length = 0, left = 0, right = 0, gaps = 0
	cdef signed int lensent = len(sent), maxlen = 0
	cdef unsigned int newlabel, blocked = 0, splitlabel = 0
	cdef unsigned long long vec = 0
	cdef unsigned long maxA = 0
	cdef double x, y
	cdef str label = ''

	if start == None: start = toid["ROOT"]
	assert len(sent) < (sizeof(vec) * 8)
	vec = (1UL << len(sent)) - 1
	goal = new_ChartItem(start, vec)

	if doestimate:
		outside, maxlen = estimate
		assert len(grammar.bylhs) == len(outside)
		assert lensent <= maxlen

	gc.disable() #does this actually do anything with Cython?

	# scan
	Epsilon = toid["Epsilon"]
	for i, w in enumerate(sent):
		recognized = False
		for terminal in lexical.get(w, []):
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if not tags or tags[i] == tolabel[terminal.lhs].split("@")[0]:
				item = new_ChartItem(terminal.lhs, 1UL << i)
				sibling = new_ChartItem(Epsilon, i)
				x = terminal.prob
				y = getoutside(outside, maxlen, lensent,
							item.label, item.vec) if doestimate else 0.0
				agenda[item] = new_Edge(x + y, x, x, sibling, NONE)
				chart[item] = []
				recognized = True
		if not recognized and tags and tags[i] in toid:
			item = new_ChartItem(toid[tags[i]], 1UL << i)
			sibling = new_ChartItem(Epsilon, i)
			y = getoutside(outside, maxlen, lensent,
						item.label, item.vec) if doestimate else 0.0
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
		append(chart[item], iscore(edge))
		(<dict>(list_getitem(viterbi, item.label)))[item] = edge
		if item.label == goal.label and item.vec == goal.vec:
			if not exhaustive: break
		else:
			x = edge.inside

			# unary
			if doestimate:
				length = bitcount(item.vec); left = nextset(item.vec, 0)
				gaps = bitlength(item.vec) - length - left
				right = lensent - length - left - gaps
			l = <list>list_getitem(unary, item.label)
			if not beamwidth or beam[item.vec] < beamwidth:
				for i in range(list_getsize(l)):
					rule = <Rule>list_getitem(l, i)
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
					if prunenow and neverblockdiscontinuous:
						prunenow = arity[rule.lhs] == 1
					if prunenow and neverblockmarkovized:
						prunenow = "|" not in tolabel[rule.lhs]
					if prunenow and splitprune:
						if arity[rule.lhs] > 1:
							label = tolabel[rule.lhs]
							label = label[:label.rindex("_")]
					process_edge(new_ChartItem(rule.lhs, item.vec), newedge,
						agenda, chart, viterbi, exhaustive, prunenow,
						prunelist, prunetoid, label, markorigin, &blocked)

			# binary left
			l = <list>list_getitem(lbinary, item.label)
			for i in range(list_getsize(l)):
				rule = <Rule>list_getitem(l, i)
				for I, e in (<dict>(list_getitem(viterbi, rule.rhs2))).iteritems():
					sibling = <ChartItem>I
					if ((not beamwidth or beam[item.vec^sibling.vec] < beamwidth)
						and concat(rule, item.vec, sibling.vec)):
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
						if prunenow and neverblockdiscontinuous:
							prunenow = arity[rule.lhs] == 1
						if prunenow and neverblockmarkovized:
							prunenow = "|" not in tolabel[rule.lhs]
						if prunenow and splitprune:
							if arity[rule.lhs] > 1:
								label = tolabel[rule.lhs]
								label = label[:label.rindex("_")]
						process_edge(new_ChartItem(rule.lhs, vec), newedge,
							agenda, chart, viterbi, exhaustive, prunenow,
							prunelist, prunetoid, label, markorigin, &blocked)

			# binary right
			l = <list>list_getitem(rbinary, item.label)
			for i in range(list_getsize(l)):
				rule = <Rule>list_getitem(l, i)
				for I, e in (<dict>(list_getitem(viterbi, rule.rhs1))).iteritems():
					sibling = <ChartItem>I
					if ((not beamwidth or beam[sibling.vec^item.vec] < beamwidth)
						and concat(rule, sibling.vec, item.vec)):
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
						if prunenow and neverblockdiscontinuous:
							prunenow = arity[rule.lhs] == 1
						if prunenow and neverblockmarkovized:
							prunenow = "|" not in tolabel[rule.lhs]
						if prunenow and splitprune:
							if arity[rule.lhs] > 1:
								label = tolabel[rule.lhs]
								label = label[:label.rindex("_")]
						process_edge(new_ChartItem(rule.lhs, vec), newedge,
							agenda, chart, viterbi, exhaustive, prunenow,
							prunelist, prunetoid, label, markorigin, &blocked)

		if agenda.length > maxA: maxA = agenda.length
	logging.debug("agenda max %d, now %d, items %d (%d labels), edges %d, blocked %d" % (maxA, len(agenda), len(filter(None, chart.values())), len(filter(None, viterbi)), sum(map(len, chart.values())), blocked))
	gc.enable()
	if goal in chart: return chart, goal
	else: return chart, NONE

cdef inline void process_edge(ChartItem newitem, Edge newedge,
		EdgeAgenda agenda, dict chart, list viterbi, bint exhaustive,
		bint doprune, list prunelist, dict prunetoid, str label,
		bint markorigin, unsigned int *blocked): # except *:
	""" Decide what to do with a newly derived edge. """
	cdef unsigned long long component
	cdef unsigned int a, b = 0
	cdef int cnt = 0, splitlabel = 0
	cdef Edge e
	cdef bint inagenda = agenda.contains(newitem)
	cdef bint inchart = dict_contains(chart, newitem) == 1
	#not in agenda or chart
	if not (inagenda or inchart):
		if doprune:
			if label: #does the prune list contain split labels?
				if not markorigin:
					try: splitlabel = prunetoid[label + "*"]
					except KeyError:
						blocked[0] += 1
						return
				while newitem.vec >> b:
					if markorigin:
						try: splitlabel = prunetoid["%s*%d" % (label, cnt)]
						except KeyError:
							blocked[0] += 1
							return
					a = nextset(newitem.vec, b)
					b = nextunset(newitem.vec, a) - 1
					#given a=3, b=6. from left to right: 
					#10000 => 01111 => 01111000
					component = (1UL << b) - 1UL << a
					outside = dict_getitem(<object>list_getitem(prunelist,
												splitlabel), component)
					if outside==NULL or isinf(<double><object>outside):
						blocked[0] += 1
						return
					cnt += 1
			else:
				outside = dict_getitem(<object>list_getitem(prunelist,
											newitem.label), newitem.vec)
				# need a double cast: before outside can be converted to
				# a double, it needs to be treated as a python float
				if (outside==NULL or isinf(<double>(<object>outside))):
					#or newedge.inside+<double><object>outside > 300.0):
					blocked[0] += 1
					return

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
		append(chart[newitem], iscore(agenda.replace(newitem, newedge)))
	# not in agenda => must be in chart
	elif (not inagenda and newedge.inside <
				(<Edge>(<dict>viterbi[newitem.label])[newitem]).inside):
		#re-add to agenda because we found a better score
		agenda.setitem(newitem, newedge)
		logging.warning("WARN: re-adding item to agenda: %r" % (newitem))
	elif exhaustive:
		# suboptimal edge
		chart[newitem].append(iscore(newedge))

cdef inline bint concat(Rule rule, unsigned long long lvec, unsigned long long rvec):
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

	# if there are no gaps in lvec and rvec, and the yieldfunction is the
	# concatenation of two elements, then this should be quicker
	if False and rule._lengths[0] == 2 and rule.args.length == 1:
		if (lvec >> nextunset(lvec, lpos) == 0
			and rvec >> nextunset(rvec, rpos) == 0):
			if rule._args[0] == 0b10:
				return bitminmax(lvec, rvec)
			elif rule._args[0] == 0b01:
				return bitminmax(rvec, lvec)

	#this algorithm was adapted from rparse, FastYFComposer.
	cdef unsigned int n, x
	cdef unsigned short m
	for x in range(rule.args.length):
		m = rule._lengths[x] - 1
		for n in range(m + 1):
			if testbitint(rule._args[x], n):
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
				if n == m:
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
				if n == m:
					if testbit(rvec, lpos):
						return False
				elif not testbit(rvec, lpos):
					return False
				lpos = nextset(lvec, lpos)
	# success if we've reached the end of both left and right vector
	return lpos == rpos == -1

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
	for a in sorted(chart, key=lambda a: bitcount(a.vec)):
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
	else:
		print "10 best parse trees:"
		mpp = marginalize(chart, start, grammar.tolabel)
		for a, p in reversed(sorted(mpp.items(), key=itemgetter(1))): print p,a
		print

def main():
	from grammar import splitgrammar
	grammar = splitgrammar([
		((('S','VP2','VMFIN'), ((0,1,0),)), 0),
		((('VP2','VP2','VAINF'),  ((0,),(0,1))), log(0.5)),
		((('VP2','PROAV','VVPP'), ((0,),(1,))), log(0.5)),
		((('PROAV', 'Epsilon'), ('Daruber', ())), 0.0),
		((('VAINF', 'Epsilon'), ('werden', ())), 0.0),
		((('VMFIN', 'Epsilon'), ('muss', ())), 0.0),
		((('VVPP', 'Epsilon'), ('nachgedacht', ())), 0.0)])

	do("Daruber muss nachgedacht werden", grammar)
	do("Daruber muss nachgedacht werden werden", grammar)
	do("Daruber muss nachgedacht werden werden werden", grammar)
	print "ungrammatical sentence:"
	do("muss Daruber nachgedacht werden", grammar)	#no parse
	print "(as expected)"
	print "long sentence (33 words):"
	do("Daruber muss nachgedacht %s" % " ".join(30*["werden"]), grammar)

if __name__ == '__main__': main()
