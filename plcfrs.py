# probabilistic CKY parser for Simple Range Concatenation Grammars
# (equivalent to Linear Context-Free Rewriting Systems)
import re
from math import log, exp, fsum
from random import choice
from itertools import count, groupby
from operator import itemgetter
from collections import defaultdict
from kbest import lazykbest
from containers import ChartItem, Edge
from agenda import heapdict
try:
	import cython
	assert cython.compiled
except:
	print "plcfrs in non-cython mode"
	from bit import *
NONE = ChartItem(0, 0)

def parse(sent, grammar, tags=[], start=1, exhaustive=False,
		estimate=(), prune={}, prunetoid={}):
	""" parse sentence, a list of tokens, optionally with gold tags, and
	produce a chart, either exhaustive or up until the viterbi parse
	"""
	unary = grammar.unary
	lbinary = grammar.lbinary
	rbinary = grammar.rbinary
	lexical = dict(grammar.lexical)
	toid = dict(grammar.toid)
	tolabel = dict(grammar.tolabel)
	if start is None: start = toid['ROOT']
	vec = 1
	vec = (vec << len(sent)) - 1
	goal = ChartItem(start, vec)
	m = maxA = 0
	C = {}
	Cx = [{} for _ in toid]
	A = heapdict()

	# scan
	Epsilon = toid["Epsilon"]
	for i, w in enumerate(sent):
		recognized = False
		for terminal in lexical.get(w, []):
			if not tags or tags[i] == tolabel[terminal.lhs].split("@")[0]:
				Ih = ChartItem(terminal.lhs, 1 << i)
				I = ChartItem(Epsilon, i)
				#z = 0.0 if tags else terminal.prob
				z = terminal.prob
				A[Ih] = Edge(z, z, z, I, NONE)
				C[Ih] = []
				recognized = True
		if not recognized and tags and tags[i] in toid:
			Ih = ChartItem(toid[tags[i]], 1 << i)
			I = ChartItem(Epsilon, i)
			A[Ih] = Edge(0.0, 0.0, 0.0, I, NONE)
			C[Ih] = []
			recognized = True
			continue
		elif not recognized:
			print "not covered:", tags[i] if tags else w
			return C, NONE
	
	# parsing
	while A:
		Ih, edge = A.popitem()
		C[Ih].append(edge)
		assert Ih not in Cx[Ih.label]
		Cx[Ih.label][Ih] = edge

		if Ih == goal:
			if exhaustive: continue
			else: break
		for I1h, edge in deduced_from(Ih, edge.inside, Cx,
									unary, lbinary, rbinary):
			if I1h not in C and I1h not in A:
				# haven't seen this item before, add to agenda
				A[I1h] = edge
				C[I1h] = []
			elif I1h in A and edge < A[I1h]:
				# either item has lower score, update agenda,
				# or extend chart
				C[I1h].append(A[I1h])
				A[I1h] = edge
			else:
				C[I1h].append(edge)

		maxA = max(maxA, len(A))
	print "max agenda size", maxA, "/ chart items", len(C), "/ edges", sum(map(len, C.values())),
	return (C, goal) if goal in C else (C, NONE)

def deduced_from(Ih, x, Cx, unary, lbinary, rbinary):
	I, Ir = Ih.label, Ih.vec
	result = []
	for rule in unary[I]:
		result.append((ChartItem(rule.lhs, Ir),
			Edge(x+rule.prob, x+rule.prob, rule.prob, Ih, NONE)))
	for rule in lbinary[I]:
		for I1h, edge in Cx[rule.rhs2].iteritems():
			if concat(rule, Ir, I1h.vec):
				result.append((ChartItem(rule.lhs, Ir ^ I1h.vec),
					Edge(x+edge.inside+rule.prob, x+edge.inside+rule.prob,
							rule.prob, Ih, I1h)))
	for rule in rbinary[I]:
		for I1h, edge in Cx[rule.rhs1].iteritems():
			if concat(rule, I1h.vec, Ir):
				result.append((ChartItem(rule.lhs, I1h.vec ^ Ir),
					Edge(x+edge.inside+rule.prob, x+edge.inside+rule.prob,
							rule.prob, I1h, Ih)))
	return result

def concat(rule, lvec, rvec):
	if lvec & rvec: return False
	lpos = nextset(lvec, 0)
	rpos = nextset(rvec, 0)
	# if there are no gaps in lvec and rvec, and the yieldfunction is the
	# concatenation of two elements, then this should be quicker
	if False and (lvec >> nextunset(lvec, lpos) == 0
		and rvec >> nextunset(rvec, rpos) == 0):
		if rule.lengths[0] == 2 and rule.args.length == 1:
			if rule.args[0] == 0b10:
				return bitminmax(lvec, rvec)
			elif rule.args[0] == 0b01:
				return bitminmax(rvec, lvec)
		#else:
		#	return False
	#this algorithm taken from rparse FastYFComposer.
	for x in range(len(rule.args)):
		m = rule.lengths[x] - 1
		for n in range(m + 1):
			if testbitint(rule.args[x], n):
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

def kbest_outside(chart, start, k):
	D = {}
	outside = { start : 0.0 }
	lazykthbest(start, k, k, D, {}, chart, set())
	for (e, j), rootedge in D[start]:
		getitems(e, j, rootedge, D, chart, outside)
	return outside

def getitems(e, j, rootedge, D, chart, outside):
	""" Traverse a derivation e,j, noting outside costs relative to its root edge
	"""
	if e.left in chart:
		if e.left in D: (ee, jj), ee2 = D[e.left][j[0]]
		elif j[0] == 0: jj = (0, 0); ee = ee2 = min(chart[e.left])
		else: raise ValueError
		if e.left not in outside:
			outside[e.left] = rootedge.inside - ee2.inside
		getitems(ee, jj, rootedge, D, chart, outside)
	if e.right.label:
		if e.right in D: (ee, jj), ee2 = D[e.right][j[1]]
		elif j[1] == 0: jj = (0, 0); ee = ee2 = min(chart[e.right])
		else: raise ValueError
		if e.right not in outside:
			outside[e.right] = rootedge.inside - ee2.inside
		getitems(ee, jj, rootedge, D, chart, outside)

def iscore(e):
	e.score = e.inside
	return e

def filterchart(chart, start):
	# remove all entries that do not contribute to a complete derivation headed
	# by "start"
	chart2 = {}
	filter_subtree(start, chart, chart2)
	return chart2

def filter_subtree(start, chart, chart2):
	chart2[start] = chart[start]
	for edge in chart[start]:
		item = edge.left
		if item.label and item not in chart2:
			filter_subtree(edge.left, chart, chart2)
		item = edge.right
		if item.label and item not in chart2:
			filter_subtree(edge.right, chart, chart2)

def getviterbi(chart, start, mem):
	""" recompute the proper viterbi probabilities in a top-down fashion,
		and sort chart entries according to these probabilities
		removes zero-probability edges (infinity with log-probabilities)
		also tracks items visited, for pruning purposes (keys of mem).

		FIXME: not working yet. it seems to be more efficient to do
		the viterbi thing during parsing -- the log n overhead of
		the priority queue is small
	"""
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

def binrepr(a, sent):
	return bin(a.vec)[2:].rjust(len(sent), "0")[::-1]

def pprint_chart(chart, sent, tolabel):
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
	print "sentence", sent
	chart, start = parse(sent.split(), grammar, [], grammar.toid['S'], False, None, None, None)
	pprint_chart(chart, sent.split(), grammar.tolabel)
	if start == ChartItem(0, 0):
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
