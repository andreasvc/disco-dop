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
from agenda import EdgeAgenda
try:
	import cython
	assert cython.compiled
except:
	print "plcfrs in non-cython mode"
	from bit import *
NONE = ChartItem(0, 0)

def parse(sent, grammar, tags=[], start=1, exhaustive=False,
		estimate=(), prunelist=None, prunetoid=None):
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
	A = EdgeAgenda()

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

def iscore(e):
	e.score = e.inside
	return e

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
	from grammar import splitgrammar
	grammar = splitgrammar([
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
