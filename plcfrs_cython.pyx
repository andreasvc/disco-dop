""" Probabilistic CKY parser for Simple Range Concatenation Grammars
(equivalent to Linear Context-Free Rewriting Systems)"""
from cpython cimport PyObject,\
					PyList_Append as append,\
					PyList_GET_ITEM as list_getitem,\
					PyList_GET_SIZE as list_getsize,\
					PyDict_Contains as dict_contains,\
					PyDict_GetItem as dict_getitem
from math import log, exp, fsum, isinf
from random import choice, randrange
from operator import itemgetter
from itertools import islice, count
from collections import defaultdict, deque
from array import array
import re, gc
from nltk import FreqDist, Tree
from grammar import enumchart, induce_srcg
from containers import ChartItem, Edge, Rule, Terminal
from agenda import heapdict, Entry
import numpy as np
np.import_array()

DEF infinity = float('infinity')

# to avoid overhead of __init__ and __cinit__ constructors
# belongs in containers but putting it here gives a better chance of successful
# inlining
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
	estimate=None, dict prune=None, dict prunetoid={}):
	""" parse sentence, a list of tokens, optionally with gold tags, and
	produce a chart, either exhaustive or up until the viterbi parse
	"""
	cdef list unary = grammar.unary
	cdef list lbinary = grammar.lbinary
	cdef list rbinary = grammar.rbinary
	cdef dict lexical = <dict>grammar.lexical
	cdef dict toid = <dict>grammar.toid
	cdef dict tolabel = <dict>grammar.tolabel
	cdef list rules, l, prunelist = range(len(toid))
	cdef dict C = {}							#the full chart
	cdef list Cx = [{} for _ in toid]			#the viterbi probabilities
	cdef dict kbestoutside
	cdef unsigned int maxA = 0, length = 0, left = 0, right = 0, gaps = 0
	cdef unsigned int lensent = len(sent), blocked = 0, maxlen = 0
	cdef unsigned int label, newlabel
	cdef unsigned long vec
	cdef double x, y, z
	cdef Py_ssize_t i
	cdef bint doprune = False, doestimate = bool(estimate)
	cdef heapdict A = heapdict()				#the agenda
	cdef Entry entry
	cdef Edge edge, newedge
	cdef ChartItem Ih, I1h, goal, newitem, NONE = new_ChartItem(0, 0)
	cdef Terminal terminal
	cdef Rule rule
	cdef np.ndarray[np.double_t, ndim=4] outside

	if start == None: start = toid["ROOT"]
	if doestimate:
		outside, maxlen = estimate
		assert len(grammar.bylhs) == len(outside)
		assert lensent <= maxlen
	goal = new_ChartItem(start, (1 << len(sent)) - 1)

	if prune:
		doprune = True
		kbestoutside = kbest_outside(prune, goal, 100)
		l = [{} for a in prunetoid]
		for Ih in prune:
			l[Ih.label][Ih.vec] = kbestoutside.get(Ih, infinity)
		for a, label in toid.iteritems():
			newlabel = prunetoid[a.split("@")[0]]
			prunelist[label] = l[newlabel]
		print 'pruning with %d nonterminals, %d items' % (
				len(filter(None, prunelist)), len(prune))
				# ?? sum(len(filter(lambda x: x < infinity, ll.values()) for ll in l))
	gc.disable()

	# scan
	Epsilon = toid["Epsilon"]
	for i, w in enumerate(sent):
		recognized = False
		for terminal in lexical.get(w, []):
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction, 
			# and give probability of 1
			if not tags or tags[i] == tolabel[terminal.lhs].split("@")[0]:
				Ih = new_ChartItem(terminal.lhs, 1 << i)
				I1h = new_ChartItem(Epsilon, i)
				#z = 0.0 if tags else terminal.prob
				z = terminal.prob
				if doestimate:
					A[Ih] = new_Edge(getoutside(outside, maxlen, lensent,
								terminal.lhs, 1 << i) + z, z, z, I1h, NONE)
				else:
					A[Ih] = new_Edge(z, z, z, I1h, NONE)
				C[Ih] = []
				recognized = True
		if not recognized and tags and tags[i] in toid:
			Ih = new_ChartItem(toid[tags[i]], 1 << i)
			I1h = new_ChartItem(Epsilon, i)
			if doestimate:
				A[Ih] = new_Edge(getoutside(outside, maxlen, lensent,
							terminal.lhs, 1 << i), 0.0, 0.0, I1h, NONE)
			else:
				A[Ih] = new_Edge(0.0, 0.0, 0.0, I1h, NONE)
			C[Ih] = []
			recognized = True
			continue
		elif not recognized:
			print "not covered:", tags[i] if tags else w
			return C, NONE

	# parsing
	while A.length:
		entry = A.popentry()
		Ih = <ChartItem>entry.key
		edge = <Edge>entry.value
		append(C[Ih], iscore(edge))
		#assert Ih not in Cx[Ih.label]
		(<dict>(list_getitem(Cx, Ih.label)))[Ih] = edge
		if Ih.label == goal.label and Ih.vec == goal.vec:
			if not exhaustive: break
		else:
			x = edge.inside

			# unary
			if doestimate:
				vec = Ih.vec; length = bitcount(vec); left = nextset(vec, 0)
				gaps = bitlength(vec) - length - left
				right = lensent - length - left - gaps
			l = <list>list_getitem(unary, Ih.label)
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
				process_edge(new_ChartItem(rule.lhs, Ih.vec), newedge,
								A, C, Cx, doprune, prunelist, lensent, &blocked)
			
			# binary left
			l = <list>list_getitem(lbinary, Ih.label)
			for i in range(list_getsize(l)):
				rule = <Rule>list_getitem(l, i)
				for I, e in (<dict>(list_getitem(Cx, rule.rhs2))).iteritems():
					I1h = <ChartItem>I
					if concat(rule, Ih.vec, I1h.vec):
						y = (<Edge>e).inside
						vec = Ih.vec ^ I1h.vec
						if doestimate:
							length = bitcount(vec); left = nextset(vec, 0)
							gaps = bitlength(vec) - length - left
							right = lensent - length - left - gaps
							newedge = new_Edge(
									outside[rule.lhs, length, left+right, gaps]
									+x+y+rule.prob, x+y+rule.prob,
									rule.prob, Ih, I1h)
						else:
							newedge = new_Edge(x+y+rule.prob, x+y+rule.prob,
											rule.prob, Ih, I1h)
						process_edge(new_ChartItem(rule.lhs, vec), newedge,
								A, C, Cx, doprune, prunelist, lensent, &blocked)

			# binary right
			l = <list>list_getitem(rbinary, Ih.label)
			for i in range(list_getsize(l)):
				rule = <Rule>list_getitem(l, i)
				for I, e in (<dict>(list_getitem(Cx, rule.rhs1))).iteritems():
					I1h = <ChartItem>I
					if concat(rule, I1h.vec, Ih.vec):
						y = (<Edge>e).inside
						vec = I1h.vec ^ Ih.vec
						if doestimate:
							length = bitcount(vec); left = nextset(vec, 0)
							gaps = bitlength(vec) - length - left
							right = lensent - length - left - gaps
							newedge = new_Edge(
									outside[rule.lhs, length, left+right, gaps]
									+x+y+rule.prob, x+y+rule.prob,
									rule.prob, I1h, Ih)
						else:
							newedge = new_Edge(x+y+rule.prob, x+y+rule.prob,
											rule.prob, I1h, Ih)
						process_edge(new_ChartItem(rule.lhs, vec), newedge,
								A, C, Cx, doprune, prunelist, lensent, &blocked)

		maxA = max(maxA, len(A))
	print "max agenda size %d, now %d, chart items %d (%d), edges %d, blocked %d" % (
				maxA, len(A), len(C), len(filter(None, Cx)),
					sum(map(len, C.values())), blocked),
	gc.enable()
	if goal in C: return C, goal
	else: return C, NONE

cdef inline void process_edge(ChartItem newitem, Edge newedge, heapdict A,
		dict C, list Cx, bint doprune, list prunelist,
		unsigned int lensent, unsigned int *blocked):
	""" Decide what to do with a newly derived edge. """
	cdef Edge e
	#not in A or C
	if not (A.contains(newitem) or dict_contains(C, newitem) == 1):
		if doprune:
			outside = dict_getitem(<object>list_getitem(prunelist,
											newitem.label), newitem.vec)
			if outside==NULL or newedge.inside+<double><object>outside > 300.0:
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
			#e = <Edge>A[newitem]
			#append(C[newitem], new_Edge(e.inside, e.inside, e.prob, e.left, e.right))
			#e.inside = newedge.inside
			#e.prob = newedge.prob
			#e.left = newedge.left
			#e.right = newedge.right
		#worse score, only add to chart
		else:
			C[newitem].append(iscore(newedge))
	# not in A, but is in C
	else:
		C[newitem].append(iscore(newedge))
		#Cx[newitem.label][newitem] = min(Cx[newitem.label][newitem], newedge.inside)
		#if newedge.inside < <double>(<dict>(Cx[newitem.label])[newitem]):
		#	(<dict>Cx[newitem.label])[newitem] = newedge.inside

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
		( (0, 1, 0), (1, 0) ) ==> array('B', [0b010, 0b01])
							and lengths: array('B', [3, 2])
		NB: note reversal due to the way binary numbers are represented
	"""
	if lvec & rvec: return False
	cdef int lpos = nextset(lvec, 0)
	cdef int rpos = nextset(rvec, 0)
	# if there are no gaps in lvec and rvec, and the yieldfunction is the
	# concatenation of two elements, then this should be quicker
	if False and (lvec >> nextunset(lvec, lpos) == 0
		and rvec >> nextunset(rvec, rpos) == 0):
		if rule._lengths[0] == 2 and rule.args.length == 1:
			if rule._args[0] == 0b10:
				return bitminmax(lvec, rvec)
			elif rule._args[0] == 0b01:
				return bitminmax(rvec, lvec)
		#else:
		#	return False
	#this algorithm taken from rparse, FastYFComposer.
	cdef unsigned int n, x
	cdef unsigned char arg, m
	for x in range(rule.args.length):
		m = rule._lengths[x] - 1
		for n in range(m + 1):
			if testbitshort(rule._args[x], n):
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
				#jump to next argument
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
			#else: raise ValueError("non-binary element in yieldfunction")
	if lpos != -1 or rpos != -1:
		return False
	# everything looks all right
	return True

cdef dict kbest_outside(dict chart, ChartItem start, int k):
	D = {}
	outside = { start : 0.0 }
	lazykthbest(start, k, k, D, {}, chart, set())
	for (e, j), rootedge in D[start]:
		getitems(e, j, rootedge, D, chart, outside)
	return outside

cdef void getitems(Edge e, tuple j, Edge rootedge, dict D, dict chart, dict outside):
	""" Traverse a derivation e,j, noting outside costs relative to its root edge
	"""
	if e.left in chart:
		if e.left in D: (ee, jj), ee2 = D[e.left][j[0]]
		elif j[0] == 0: jj = (0, 0); ee = ee2 = min(chart[e.left])
		else: raise ValueError
		if e.left not in outside:
			outside[e.left] = rootedge.inside - (<Edge>ee2).inside
		getitems(<Edge>ee, jj, rootedge, D, chart, outside)
	if e.right.label:
		if e.right in D: (ee, jj), ee2 = D[e.right][j[1]]
		elif j[1] == 0: jj = (0, 0); ee = ee2 = min(chart[e.right])
		else: raise ValueError
		if e.right not in outside:
			outside[e.right] = rootedge.inside - (<Edge>ee2).inside
		getitems(<Edge>ee, jj, rootedge, D, chart, outside)

cdef inline Edge iscore(Edge e):
	e.score = e.inside
	return e

def filterchart(chart, start):
	# remove all entries that do not contribute to a complete derivation headed
	# by "start"
	chart2 = {}
	filter_subtree(start, <dict>chart, chart2)
	return chart2

cdef void filter_subtree(ChartItem start, dict chart, dict chart2):
	cdef Edge edge
	cdef ChartItem item
	chart2[start] = chart[start]
	for edge in chart[start]:
		item = (<Edge>edge).left
		if item.label and item not in chart2:
			filter_subtree((<Edge>edge).left, chart, chart2)
		item = (<Edge>edge).right
		if item.label and item not in chart2:
			filter_subtree(edge.right, chart, chart2)

cpdef list getviterbi(dict chart, ChartItem start, dict mem):
	""" recompute the proper viterbi probabilities in a top-down fashion,
		and sort chart entries according to these probabilities
		removes zero-probability edges (infinity with log-probabilities)
		also tracks items visited, for pruning purposes (keys of mem).

		FIXME: not working yet. it seems to be more efficient to do
		the viterbi thing during parsing -- the log n overhead of
		the priority queue is small
	"""
	cdef list probs, bestprops
	cdef double prob, bestprob
	cdef int n
	probs = []
	bestprob = 999999 #float('infinity')
	try: assert len(chart[start])
	except: print "empty", start
	# loop backwards because we want to remove items in-place without
	# invalidating remaining indices.
	for n, (ip, p, rhs) in zip(count(), chart[start])[::-1]:
		probs[:] = [p]
		for a in rhs:
			# only recurse for nonterminals (=nonzero ids)
			if a.label and a in chart:
				if a in mem: result = mem[a]
				else: result = getviterbi(chart, a, mem)
				if not isinstance(result, list):
					print "trouble", start, '->', a
				probs.extend(result)
		prob = fsum(probs)
		if prob < bestprob:
			bestprob = prob
			bestprobs = probs[:]
		# prune or update probability
		if isinf(prob): del chart[start][n]
		else: chart[start][n] = (prob, p, rhs)
	if len(chart[start]):
		chart[start].sort(key=itemgetter(0))
		assert fsum(bestprobs) == chart[start][0][0]
	else:
		bestprobs = [float('infinity')]
	mem[start] = bestprobs
	return bestprobs

cdef samplechart(dict chart, ChartItem start, dict tolabel):
	cdef ChartItem child
	cdef Edge edge = choice(chart[start])
	if edge.left.label == 0: # == "Epsilon":
		return "(%s %d)" % (tolabel[start.label], edge.left.vec), edge.prob
	children = [samplechart(chart, child, tolabel)
				for child in (edge.left, edge.right) if child.label]
	tree = "(%s %s)" % (tolabel[start.label],
							" ".join([a for a,b in children]))
	return tree, fsum([edge.prob] + [b for a,b in children])

def mostprobablederivation(chart, start, tolabel):
	return (getmpd(chart, start, tolabel),
				(<Edge>(min(chart[start]))).inside)

cdef getmpd(dict chart, ChartItem start, dict tolabel):
	cdef Edge edge = <Edge>(min(chart[start]))
	if edge.right.label: #binary
		return "(%s %s %s)" % (tolabel[start.label],
					getmpd(chart, edge.left, tolabel),
					getmpd(chart, edge.right, tolabel))
	else: #unary or terminal
		return "(%s %s)" % (tolabel[start.label],
					getmpd(chart, edge.left, tolabel) if edge.left.label
									else str(edge.left.vec))

removeids = re.compile("@[0-9]+")
def mostprobableparse(chart, start, tolabel, n=10, sample=False, both=False, shortest=False, secondarymodel=None):
	""" sum over n random/best derivations from chart,
		return a dictionary mapping parsetrees to probabilities """
	print "sample =", sample or both, "kbest =", (not sample) or both,
	if both:
		derivations = set(samplechart(chart, start, tolabel) for x in range(n*100))
		derivations.discard(None)
		derivations.update(lazykbest(chart, start, n, tolabel))
		#derivations.update(islice(enumchart(chart, start, tolabel, n), n))
	elif sample:
		#filtercycles(chart, start, set(), set())
		derivations = set(samplechart(chart, start, tolabel) for x in range(n))
		derivations.discard(None)
		#calculate real parse probabilities according to Goodman's claimed
		#method?
	else:
		#mem = {}
		#filtercycles(chart, start, set(), set())
		#getviterbi(chart, start, mem)
		#for a in set(chart.keys()) - set(mem.keys()): del chart[a]
		#print "pruned chart keys", len(chart), "/ values", sum(map(len, chart.values()))
		#for a in chart: chart[a].sort(key=itemgetter(0))
		#derivations = list(islice(enumchart(chart, start, tolabel, n), n))
		#fixme: set shouldn't be necessary
		derivations = set(lazykbest(chart, start, n, tolabel))
		#print len(derivations)
		#print "enumchart:", len(list(islice(enumchart(chart, start, tolabel), n)))
		#assert(len(list(islice(enumchart(chart, start), n))) == len(set((a.freeze(),b) for a,b in islice(enumchart(chart, start), n))))
	if shortest:
		derivations = [(a,b) for a, b in derivations if b == derivations[0][1]]
	#cdef dict parsetrees = <dict>defaultdict(float)
	cdef dict parsetrees = <dict>defaultdict(list)
	cdef double prob, maxprob
	cdef int m = 0
	#cdef str deriv, tree
	for deriv, prob in derivations:
		m += 1
		#parsetrees[removeids.sub("", deriv)] += exp(-prob)
		#restore linear precedence (disabled, seems to make no difference)
		#parsetree = Tree(removeids.sub("", deriv))
		#for a in list(parsetree.subtrees())[::-1]:
		#	a.sort(key=lambda x: x.leaves())
		#parsetrees[parsetree.pprint(margin=999)].append(-prob)
		if shortest:
			tree = Tree(removeids.sub("", deriv))
			sent = sorted(tree.leaves(), key=int)
			rules = induce_srcg([tree], [sent], arity_marks=False)
			prob = -fsum([secondarymodel[r] for r, w in rules
											if r[0][1] != 'Epsilon'])
		parsetrees[removeids.sub("", deriv)].append(-prob)
	# Adding probabilities in log space
	# http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point-numbers
	# https://facwiki.cs.byu.edu/nlp/index.php/Log_Domain_Computations
	for parsetree in parsetrees:
		maxprob = max(parsetrees[parsetree])
		#foo = sum(map(exp, parsetrees[parsetree]))
		parsetrees[parsetree] = exp(fsum([maxprob, log(fsum([exp(prob - maxprob)
										for prob in parsetrees[parsetree]]))]))
		#assert log(foo) == parsetrees[parsetree]
	print "(%d derivations, %d parsetrees)" % (m, len(parsetrees))
	return parsetrees

def binrepr(a, sent):
	return bin((<ChartItem>a).vec)[2:].rjust(len(sent), "0")[::-1]

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
	from grammar import newsplitgrammar
	grammar = newsplitgrammar([
		((('S','VP2','VMFIN'), ((0,1,0),)), 0),
		((('VP2','VP2','VAINF'),  ((0,),(0,1))), log(0.5)),
		((('VP2','PROAV','VVPP'), ((0,),(1,))), log(0.5)),
		((('VP2','VP2'), ((0,),(1,))), log(0.1)),
		((('PROAV', 'Epsilon'), ('Daruber', ())), 0.0),
		((('VAINF', 'Epsilon'), ('werden', ())), 0.0),
		((('VMFIN', 'Epsilon'), ('muss', ())), 0.0),
		((('VVPP', 'Epsilon'), ('nachgedacht', ())), 0.0)])

	do("Daruber muss nachgedacht werden", grammar)
	do("Daruber muss nachgedacht werden werden", grammar)
	do("Daruber muss nachgedacht werden werden werden", grammar)
	do("muss Daruber nachgedacht werden", grammar)	#no parse

if __name__ == '__main__': main()
