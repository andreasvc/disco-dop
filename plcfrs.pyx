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
cdef inline ChartItem new_ChartItem(unsigned int label, unsigned long vec):
	cdef ChartItem item = ChartItem.__new__(ChartItem)
	item.label = label; item.vec = vec
	#item._hash = hash((label, vec))
	# this is the hash function used for tuples, apparently
	item._hash = (<unsigned long>1000003
		* ((<unsigned long>1000003 * <unsigned long>0x345678)
		^ label)) ^ (vec & ((1 << 15) - 1) + (vec >> 15))
	if item._hash == -1: item._hash = -2
	return item

cdef inline Edge new_Edge(double score, double inside, double prob,
	ChartItem left, ChartItem right):
	cdef Edge edge = Edge.__new__(Edge)
	edge.score = score; edge.inside = inside; edge.prob = prob
	edge.left = left; edge.right = right
	#hash((inside, prob, left, right))
	# this is the hash function used for tuples, apparently
	edge._hash = (<unsigned long>1000003 * ((<unsigned long>1000003 *
				((<unsigned long>1000003 * ((<unsigned long>1000003 *
											<unsigned long>0x345678)
				^ <long>inside)) ^ <long>prob)) ^ left._hash)) ^ right._hash
	if edge._hash == -1: edge._hash = -2
	return edge

def parse(sent, grammar, tags=None, start=None, bint exhaustive=False,
			estimate=None, list prunelist=None, dict prunetoid=None,
			dict coarsechart=None, bint splitprune=False,
			bint neverblockmarkovized=False, bint neverblockdiscontinuous=False,
			int beamwidth=0):
	""" parse sentence, a list of tokens, optionally with gold tags, and
	produce a chart, either exhaustive or up until the viterbi parse
	"""
	cdef list unary = grammar.unary
	cdef list lbinary = grammar.lbinary
	cdef list rbinary = grammar.rbinary
	cdef dict lexical = <dict>grammar.lexical
	cdef dict toid = <dict>grammar.toid
	cdef dict tolabel = <dict>grammar.tolabel
	cdef array[unsigned char] arity = grammar.arity
	cdef list rules
	cdef dict C = {}							#the full chart
	cdef list Cx = [{} for _ in toid]			#the viterbi probabilities
	cdef dict k
	cdef dict beam = <dict>defaultdict(int)
	cdef signed int length = 0, left = 0, right = 0, gaps = 0
	cdef signed int lensent = len(sent), maxlen = 0
	cdef unsigned int label, newlabel, blocked = 0, splitlabel = 0
	cdef unsigned long vec, maxA = 0
	cdef double x, y, z
	cdef bint doprune = bool(prunelist), prunenow, doestimate = bool(estimate)
	cdef Py_ssize_t i
	cdef EdgeAgenda A = EdgeAgenda()				#the agenda
	cdef Entry entry
	cdef Edge edge, newedge
	cdef ChartItem NONE = new_ChartItem(0, 0)
	cdef ChartItem Ih, I1h, newitem, goal = NONE
	cdef Terminal terminal
	cdef Rule rule
	cdef np.ndarray[np.double_t, ndim=4] outside

	if start == None: start = toid["ROOT"]
	assert len(sent) < (sizeof(vec) * 8)
	vec = (1UL << len(sent)) - 1
	goal = new_ChartItem(start, vec)

	if doestimate:
		outside, maxlen = estimate
		assert len(grammar.bylhs) == len(outside)
		assert lensent <= maxlen

	gc.disable() #does this actually do anything with Cython's Boehm allocator?

	# scan
	Epsilon = toid["Epsilon"]
	for i, w in enumerate(sent):
		recognized = False
		for terminal in lexical.get(w, []):
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if not tags or tags[i] == tolabel[terminal.lhs].split("@")[0]:
				Ih = new_ChartItem(terminal.lhs, 1UL << i)
				I1h = new_ChartItem(Epsilon, i)
				z = terminal.prob
				y = getoutside(outside, maxlen, lensent,
							Ih.label, Ih.vec) if doestimate else 0.0
				A[Ih] = new_Edge(z + y, z, z, I1h, NONE)
				C[Ih] = []
				recognized = True
		if not recognized and tags and tags[i] in toid:
			Ih = new_ChartItem(toid[tags[i]], 1UL << i)
			I1h = new_ChartItem(Epsilon, i)
			y = getoutside(outside, maxlen, lensent,
						Ih.label, Ih.vec) if doestimate else 0.0
			A[Ih] = new_Edge(y, 0.0, 0.0, I1h, NONE)
			C[Ih] = []
			recognized = True
			continue
		elif not recognized:
			logging.error("not covered: %s" % (tags[i] if tags else w))
			return C, NONE

	# parsing
	while A.length:
		entry = A.popentry()
		Ih = <ChartItem>entry.key
		edge = <Edge>entry.value
		append(C[Ih], iscore(edge))
		(<dict>(list_getitem(Cx, Ih.label)))[Ih] = edge
		if Ih.label == goal.label and Ih.vec == goal.vec:
			if not exhaustive: break
		else:
			x = edge.inside

			# unary
			if doestimate:
				length = bitcount(Ih.vec); left = nextset(Ih.vec, 0)
				gaps = <int>bitlength(Ih.vec) - length - left
				right = lensent - length - left - gaps
			l = <list>list_getitem(unary, Ih.label)
			if not beamwidth or beam[Ih.vec] < beamwidth:
				for i in range(list_getsize(l)):
					rule = <Rule>list_getitem(l, i)
					if doestimate:
						newedge = new_Edge(
									outside[rule.lhs, length, left+right, gaps]
									+ x + rule.prob, x + rule.prob,
									rule.prob, Ih, NONE)
					else:
						newedge = new_Edge(x + rule.prob, x + rule.prob,
											rule.prob, Ih, NONE)
					prunenow = doprune
					if prunenow and neverblockdiscontinuous:
						prunenow = arity[rule.lhs] > 1
					if prunenow and neverblockmarkovized:
						prunenow = "|" not in tolabel[rule.lhs]
					if prunenow and splitprune:
						if arity[rule.lhs] > 1:
							splitlabel = prunetoid[tolabel[
										rule.lhs].split("_")[0] + "*"]
						else: splitlabel = 0
					process_edge(new_ChartItem(rule.lhs, Ih.vec), newedge,
						A, C, Cx, prunenow, prunelist, splitlabel,
							coarsechart, lensent, &blocked)

			# binary left
			l = <list>list_getitem(lbinary, Ih.label)
			for i in range(list_getsize(l)):
				rule = <Rule>list_getitem(l, i)
				for I, e in (<dict>(list_getitem(Cx, rule.rhs2))).iteritems():
					I1h = <ChartItem>I
					if ((not beamwidth or beam[Ih.vec^I1h.vec] < beamwidth) and
						concat(rule, Ih.vec, I1h.vec)):
						beam[Ih.vec ^ I1h.vec] += 1
						y = (<Edge>e).inside
						vec = Ih.vec ^ I1h.vec
						if doestimate:
							length = bitcount(vec); left = nextset(vec, 0)
							gaps = <int>bitlength(vec) - length - left
							right = lensent - length - left - gaps
							newedge = new_Edge(
									outside[rule.lhs, length, left+right, gaps]
									+x+y+rule.prob, x+y+rule.prob,
									rule.prob, Ih, I1h)
						else:
							newedge = new_Edge(x+y+rule.prob, x+y+rule.prob,
											rule.prob, Ih, I1h)
						prunenow = doprune
						if prunenow and neverblockdiscontinuous:
							prunenow = arity[rule.lhs] > 1
						if prunenow and neverblockmarkovized:
							prunenow = "|" not in tolabel[rule.lhs]
						if prunenow and splitprune:
							if arity[rule.lhs] > 1:
								splitlabel = prunetoid[tolabel[
											rule.lhs].split("_")[0] + "*"]
						else: splitlabel = 0
						process_edge(new_ChartItem(rule.lhs, vec), newedge,
							A, C, Cx, prunenow, prunelist, splitlabel,
								coarsechart, lensent, &blocked)

			# binary right
			l = <list>list_getitem(rbinary, Ih.label)
			for i in range(list_getsize(l)):
				rule = <Rule>list_getitem(l, i)
				for I, e in (<dict>(list_getitem(Cx, rule.rhs1))).iteritems():
					I1h = <ChartItem>I
					if ((not beamwidth or beam[I1h.vec^Ih.vec] < beamwidth) and
						concat(rule, I1h.vec, Ih.vec)):
						beam[I1h.vec ^ Ih.vec] += 1
						y = (<Edge>e).inside
						vec = I1h.vec ^ Ih.vec
						if doestimate:
							length = bitcount(vec); left = nextset(vec, 0)
							gaps = <int>bitlength(vec) - length - left
							right = lensent - length - left - gaps
							newedge = new_Edge(
									outside[rule.lhs, length, left+right, gaps]
									+x+y+rule.prob, x+y+rule.prob,
									rule.prob, I1h, Ih)
						else:
							newedge = new_Edge(x+y+rule.prob, x+y+rule.prob,
											rule.prob, I1h, Ih)
						prunenow = doprune
						if prunenow and neverblockdiscontinuous:
							prunenow = arity[rule.lhs] > 1
						if prunenow and neverblockmarkovized:
							prunenow = "|" not in tolabel[rule.lhs]
						if prunenow and splitprune:
							if arity[rule.lhs] > 1:
								splitlabel = prunetoid[tolabel[
											rule.lhs].split("_")[0] + "*"]
						else: splitlabel = 0
						process_edge(new_ChartItem(rule.lhs, vec), newedge,
							A, C, Cx, prunenow, prunelist, splitlabel,
								coarsechart, lensent, &blocked)

		if A.length > maxA: maxA = A.length
	logging.debug("max agenda size %d, now %d, chart items %d (%d labels), edges %d, blocked %d" % (maxA, len(A), len(C), len(filter(None, Cx)), sum(map(len, C.values())), blocked))
	gc.enable()
	if goal in C: return C, goal
	else: return C, NONE

cdef inline void process_edge(ChartItem newitem, Edge newedge, EdgeAgenda A,
		dict C, list Cx, bint doprune, list prunelist, int splitlabel,
		dict coarsechart, unsigned int lensent, unsigned int *blocked) except *:
	""" Decide what to do with a newly derived edge. """
	cdef unsigned long component
	cdef int a, b = 0
	cdef Edge e
	#not in A or C
	if not (A.contains(newitem) or dict_contains(C, newitem) == 1):
		if doprune:
			outside = dict_getitem(<object>list_getitem(prunelist,
											newitem.label), newitem.vec)
			if (outside==NULL or isinf(<double>(<object>outside))):
				#or newedge.inside+<double><object>outside > 300.0):
				blocked[0] += 1
				return
			if splitlabel:
				while newitem.vec >> b:
					a = nextset(newitem.vec, b)
					b = nextunset(newitem.vec, a) - 1
					component = (1UL << b) - 1 << a
					outside = dict_getitem(<object>list_getitem(coarsechart,
												splitlabel), component)
					if outside==NULL or isinf(<double><object>outside):
						blocked[0] += 1
						return

		elif newedge.score > 300.0:
			blocked[0] += 1
			return
		# haven't seen this item before, won't prune, add to agenda
		A.setitem(newitem, newedge)
		C[newitem] = []
	# in A (maybe in C)
	elif A.contains(newitem):
		# item has lower score, update agenda (and add old edge to chart)
		if newedge.inside < (<Edge>(A.getitem(newitem))).inside:
			append(C[newitem], iscore(A.replace(newitem, newedge)))
		#worse score, only add to chart
		else:
			C[newitem].append(iscore(newedge))
	# not in A, but is in C
	else:
		C[newitem].append(iscore(newedge))

cdef inline bint concat(Rule rule, unsigned long lvec, unsigned long rvec):
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
	cdef ChartItem a
	cdef Edge edge
	print "chart:"
	for n, a in sorted((bitcount(a.vec), a) for a in chart):
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
	from disambiguation import mostprobableparse
	from operator import itemgetter
	print "sentence", sent
	chart, start = parse(sent.split(), grammar, start=grammar.toid['S'])
	pprint_chart(chart, sent.split(), grammar.tolabel)
	if start == new_ChartItem(0, 0):
		print "no parse"
	else:
		print "10 best parse trees:"
		mpp = mostprobableparse(chart, start, grammar.tolabel)
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

if __name__ == '__main__': main()
