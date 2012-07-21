""" Implementation of LR estimate (Kallmeyer & Maier 2010).
Ported almost directly from rparse (except for sign reversal of log probs). """
from math import exp
from containers cimport ChartItem, Grammar, Rule, UInt, ULLong, new_ChartItem
from agenda cimport Agenda, Entry
from array cimport array
cimport numpy as np
from bit cimport nextset, nextunset, bitcount, bitlength, testbit, testbitint
import numpy as np
np.import_array()

cdef extern from "math.h":
	bint isnan(double x)
	bint isfinite(double x)

cdef class Item:
	cdef public long _hash
	cdef public int state, length, lr, gaps
	def __hash__(self):
		return self._hash
	def __cmp__(self, other):
		if (isinstance(other, Item) and self._hash == other._hash
			and self.state == other.state and self.length == other.length
			and self.lr == other.lr and self.gaps == other.gaps):
			return 0
		return 1
	def __repr__(self):
		return "state %4d, len %2d, lr %2d, gaps %2d" % (
			self.state, self.length, self.lr, self.gaps)

cdef inline new_Item(state, length, lr, gaps):
	item = Item.__new__(Item)
	item.state = state; item.length = length; item.lr = lr; item.gaps = gaps
	item._hash = state * 1021 + length * 571 + lr * 311 + gaps
	return item

cdef inline double getoutside(np.ndarray[np.double_t, ndim=4] outside,
		UInt maxlen, UInt slen, UInt label, ULLong vec):
	""" Query for outside estimate. NB: if this would be used, it should be in
	a .pxd with `inline.' However, passing the numpy array is slow. """
	cdef UInt length = bitcount(vec)
	cdef UInt left = nextset(vec, 0)
	cdef UInt gaps = bitlength(vec) - length - left
	cdef UInt right = slen - length - left - gaps
	cdef UInt lr = left + right
	if slen > maxlen or length + lr + gaps > maxlen:
		return 0.0
	return outside[label, length, lr, gaps]

def simpleinside(Grammar grammar, UInt maxlen,
		np.ndarray[np.double_t, ndim=2] insidescores):
	""" Compute simple inside estimate in bottom-up fashion.
	Here vec is actually the length (number of terminals in the yield of
	the constituent)
	insidescores is a 4-dimensional matrix initialized with NaN to indicate
	values that have yet to be computed. """
	cdef ChartItem I
	cdef Entry entry
	cdef Rule rule
	cdef Agenda agenda = Agenda()
	cdef np.double_t x
	cdef size_t i
	cdef ULLong vec

	for i in range(1, grammar.nonterminals):
		#this is supposed cover all and only preterminals
		if grammar.bylhs[i][0].lhs != i:
			agenda[new_ChartItem(i, 1)] = 0.0

	while agenda.length:
		entry = agenda.popentry()
		I = entry.key
		x = entry.value

		# This comparison catches the case when insidescores has a higher
		# value than x, but also when it is NaN, because all comparisons
		# against NaN are false.
		#if not x >= insidescores[I.label, I.vec]:
		# Mory explicitly:
		if (isnan(insidescores[I.label, I.vec])
			or x < insidescores[I.label, I.vec]):
			insidescores[I.label, I.vec] = x

		for i in range(grammar.nonterminals):
			rule = grammar.unary[I.label][i]
			if rule.rhs1 != I.label: break
			elif isnan(insidescores[rule.lhs, I.vec]):
				agenda.setifbetter(
						new_ChartItem(rule.lhs, I.vec), rule.prob + x)

		for i in range(grammar.nonterminals):
			rule = grammar.lbinary[I.label][i]
			if rule.rhs1 != I.label: break
			for vec in range(1, maxlen - I.vec + 1):
				if (isfinite(insidescores[rule.rhs2, vec])
					and isnan(insidescores[rule.lhs, I.vec + vec])):
					agenda.setifbetter(new_ChartItem(rule.lhs, I.vec + vec),
						rule.prob + x + insidescores[rule.rhs2, vec])

		for i in range(grammar.nonterminals):
			rule = grammar.rbinary[I.label][i]
			if rule.rhs2 != I.label: break
			for vec in range(1, maxlen - I.vec + 1):
				if (isfinite(insidescores[rule.rhs1, vec])
					and isnan(insidescores[rule.lhs, vec + I.vec])):
					agenda.setifbetter(new_ChartItem(rule.lhs, vec + I.vec),
						rule.prob + insidescores[rule.rhs1, vec] + x)

	# anything not reached so far gets probability zero:
	insidescores[np.isnan(insidescores)] = np.inf

def outsidelr(Grammar grammar, np.ndarray[np.double_t, ndim=2] insidescores,
		UInt maxlen, UInt goal, np.ndarray[np.double_t, ndim=4] outside):
	""" Compute the outside SX simple LR estimate in top down fashion. """
	cdef Agenda agenda = Agenda()
	cdef np.double_t current, score
	cdef Entry entry
	cdef Item newitem, I
	cdef Rule rule
	cdef double x, insidescore
	cdef size_t i
	cdef int m, n, totlen, addgaps, addleft, addright, leftarity, rightarity
	cdef int lenA, lenB, lr, ga
	cdef bint stopaddleft, stopaddright
	for n in range(1, maxlen + 1):
		agenda[new_Item(goal, n, 0, 0)] = 0.0
		outside[goal, n, 0, 0] = 0.0
	print "initialized"

	while agenda.length:
		entry = agenda.popentry()
		I = entry.key
		x = entry.value
		if agenda.length % 1000 == 0:
			print "agenda size: %dk top: %r, %g %s" % (
				agenda.length / 1000, I, exp(-x), grammar.tolabel[I.state])
		totlen = I.length + I.lr + I.gaps
		for i in range(grammar.numrules):
			rule = grammar.bylhs[I.state][i]
			if rule.lhs != I.state: break
			# X -> A
			if rule.rhs2 == 0:
				score = rule.prob + x
				if score < outside[rule.rhs1, I.length, I.lr, I.gaps]:
					agenda.setitem(
						new_Item(rule.rhs1, I.length, I.lr, I.gaps), score)
					outside[rule.rhs1, I.length, I.lr, I.gaps] = score
				continue
			# X -> A B
			addgaps = addright = 0
			stopaddright = False
			for n in range(bitlength(rule.lengths) - 1, -1, -1):
				if (not stopaddright
					and not testbitint(rule.args, n)):
					stopaddright = True
				if testbitint(rule.args, n):
					if not stopaddright:
						addright += 1
					else:
						addgaps += 1

			leftarity = rightarity = 1
			if grammar.bylhs[rule.rhs1][0].lhs == rule.rhs1:
				leftarity = grammar.fanout[rule.rhs1]
			if grammar.bylhs[rule.rhs2][0].lhs == rule.rhs2:
				leftarity = grammar.fanout[rule.rhs2]
			# binary-left (A is left)
			for lenA in range(leftarity, I.length - rightarity + 1):
				lenB = I.length - lenA
				insidescore = insidescores[rule.rhs2, lenB]
				for lr in range(I.lr, I.lr + lenB + 1):
					if addright == 0 and lr != I.lr: continue
					for ga in range(leftarity - 1, totlen + 1):
						if (lenA + lr + ga == I.length + I.lr + I.gaps
							and ga >= addgaps):
							score = rule.prob + x + insidescore
							if lenA+lr+ga > maxlen: current = 0.0
							else: current = outside[rule.rhs1, lenA, lr, ga]
							if score < current:
								agenda.setitem(
									new_Item(rule.rhs1, lenA, lr, ga), score)
								outside[rule.rhs1, lenA, lr, ga] = score

			# X -> B A
			addgaps = addleft = addright = 0
			stopaddleft = False
			for n in range(bitlength(rule.lengths)):
				if not stopaddleft and testbitint(rule.args, n):
					stopaddleft = True
				if not testbitint(rule.args, n):
					if stopaddleft:
						addgaps += 1
					else:
						addleft += 1

			stopaddright = False
			for n in range(bitlength(rule.lengths) - 1, -1, -1):
				if (not stopaddright
					and testbitint(rule.args, n)):
					stopaddright = True
				if not testbitint(rule.args, n):
					if not stopaddright:
						addright += 1
			addgaps -= addright

			# binary-right (A is right)
			for lenA in range(rightarity, I.length - leftarity + 1):
				lenB = I.length - lenA
				insidescore = insidescores[rule.rhs1, lenB]
				for lr in range(I.lr, I.lr + lenB + 1):
					for ga in range(rightarity - 1, totlen + 1):
						if (lenA + lr + ga == I.length + I.lr + I.gaps
							and ga >= addgaps):
							score = rule.prob + insidescore + x
							if lenA+lr+ga > maxlen: current = 0.0
							else: current = outside[rule.rhs2, lenA, lr, ga]
							if score < current:
								agenda.setitem(
									new_Item(rule.rhs2, lenA, lr, ga), score)
								outside[rule.rhs2, lenA, lr, ga] = score

def inside(Grammar grammar, UInt maxlen, dict insidescores):
	""" Compute inside estimate in bottom-up fashion, with
	full bit vectors (not used)."""
	cdef ChartItem I
	cdef Entry entry
	cdef size_t i
	infinity = float('infinity')
	agenda = Agenda()

	for i in range(1, grammar.nonterminals):
		#this is supposed cover all and only preterminals
		if grammar.bylhs[i][0].lhs != i:
			agenda[new_ChartItem(i, 1)] = 0.0

	while agenda.length:
		entry = agenda.popentry()
		I = entry.key
		x = entry.value
		if I.label not in insidescores: insidescores[I.label] =  {}
		if x < insidescores[I.label].get(I.vec, infinity):
			insidescores[I.label][I.vec] = x

		for i in range(grammar.nonterminals):
			rule = grammar.unary[I.label][i]
			if rule.rhs1 != I.label: break
			elif (rule.lhs not in insidescores
				or I.vec not in insidescores[rule.lhs]):
				agenda.setifbetter(
					new_ChartItem(rule.lhs, I.vec), rule.prob + x)

		for i in range(grammar.nonterminals):
			rule = grammar.lbinary[I.label][i]
			if rule.rhs1 != I.label: break
			elif rule.rhs2 not in insidescores: continue
			for vec in insidescores[rule.rhs2]:
				left = insideconcat(I.vec, vec, rule, grammar, maxlen)
				if left and (rule.lhs not in insidescores
					or left not in insidescores[rule.lhs]):
					agenda.setifbetter(new_ChartItem(rule.lhs, left),
						rule.prob + x + insidescores[rule.rhs2][vec])

		for i in range(grammar.nonterminals):
			rule = grammar.rbinary[I.label][i]
			if rule.rhs2 != I.label: break
			elif rule.rhs1 not in insidescores: continue
			for vec in insidescores[rule.rhs1]:
				right = insideconcat(vec, I.vec, rule, grammar, maxlen)
				if right and (rule.lhs not in insidescores
					or right not in insidescores[rule.lhs]):
					agenda.setifbetter(new_ChartItem(rule.lhs, right),
						rule.prob + insidescores[rule.rhs1][vec] + x)

	return insidescores

cdef inline ULLong insideconcat(ULLong a, ULLong b, Rule rule, Grammar grammar,
		UInt maxlen):
	if grammar.fanout[rule.lhs] + bitcount(a) + bitcount(b) > maxlen + 1:
		return 0
	result = resultpos = l = r = 0
	for x in range(bitlength(rule.lengths)):
		if testbitint(rule.args, x) == 0:
			subarg = nextunset(a, l) - l
			result |= (1 << subarg) - 1 << resultpos
			resultpos += subarg
			l = subarg + 1
		else:
			subarg = nextunset(b, r) - r
			result |= (1 << subarg) - 1 << resultpos
			resultpos += subarg
			r = subarg + 1
		if testbitint(rule.lengths, x):
			resultpos += 1
			result &= ~(1 << resultpos)
	return result

cpdef getestimates(Grammar grammar, UInt maxlen, UInt goal):
	insidescores = np.array([np.NAN], dtype='d').repeat(
				grammar.nonterminals * (maxlen+1)).reshape(
				(grammar.nonterminals, (maxlen+1)))
	outside = np.array([np.inf], dtype='d').repeat(
				grammar.nonterminals * (maxlen+1) * (maxlen+1) * (maxlen+1)
				).reshape((grammar.nonterminals, maxlen+1, maxlen+1, maxlen+1))
	print "getting inside"
	simpleinside(grammar, maxlen, insidescores)
	print "getting outside"
	outsidelr(grammar, insidescores, maxlen, goal, outside)
	return outside

cpdef testestimates(Grammar grammar, UInt maxlen, UInt goal):
	print "getting inside"
	insidescores = inside(grammar, maxlen, {})
	for a in insidescores:
		for b in insidescores[a]:
			assert 0 <= a < grammar.nonterminals
			assert 0 <= bitlength(b) <= maxlen
			#print a,b
			#print "%s[%d] =" % (grammar.tolabel[a], b), exp(insidescores[a][b])
	print len(insidescores) * sum(map(len, insidescores.values())), '\n'
	insidescores = np.array([np.NAN], dtype='d').repeat(
				grammar.nonterminals * (maxlen+1)).reshape(
				(grammar.nonterminals, (maxlen+1)))
	simpleinside(grammar, maxlen, insidescores)
	print "inside"
	for an, a in enumerate(insidescores):
		for bn, b in enumerate(a):
			if b < np.inf:
				print grammar.tolabel[an], "len", bn, "=", exp(-b)
	#print insidescores
	#for a in range(maxlen):
	#	print grammar.tolabel[goal], "len", a, "=", exp(-insidescores[goal, a])

	print "getting outside"
	outside = np.array([np.inf], dtype='d').repeat(
				grammar.nonterminals * (maxlen+1) * (maxlen+1) * (maxlen+1)
				).reshape((grammar.nonterminals, maxlen+1, maxlen+1, maxlen+1))
	outsidelr(grammar, insidescores, maxlen, goal, outside)
	#print outside
	cnt = 0
	for an, a in enumerate(outside):
		for bn, b in enumerate(a):
			for cn, c in enumerate(b):
				for dn, d in enumerate(c):
					if d < np.inf:
						print grammar.tolabel[an], "length", bn, "lr", cn,
						print "gaps", dn, "=", exp(-d)
						cnt += 1
	print cnt
	print "done"
	return outside

def main():
	from treebank import NegraCorpusReader
	from grammar import induce_plcfrs
	from parser import parse, pprint_chart
	from containers import Grammar
	from treetransforms import addfanoutmarkers
	from nltk import Tree
	corpus = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1")
	trees = list(corpus.parsed_sents())
	for a in trees: a.chomsky_normal_form(vertMarkov=1, horzMarkov=1)
	map(addfanoutmarkers, trees)
	grammar = Grammar(induce_plcfrs(trees, corpus.sents()))
	trees = [Tree.parse("(ROOT (A (a 0) (b 1)))", parse_leaf=int),
			Tree.parse("(ROOT (a 0) (B (c 2) (b 1)))", parse_leaf=int),
			Tree.parse("(ROOT (a 0) (B (c 2) (b 1)))", parse_leaf=int),
			Tree.parse("(ROOT (C (b 0) (a 1)) (c 2))", parse_leaf=int),
			Tree.parse("(ROOT (C (b 0) (a 1)) (c 2))", parse_leaf=int),
			]
	sents =[["a","b"],
			["a","c","b"],
			["a","c","b"],
			["b","a","c"],
			["b","a","a"]]
	print "treebank:"
	for a in trees: print a
	print "\ngrammar:"
	grammar = induce_plcfrs(trees, sents)
	for (r,yf),w in sorted(grammar):
		print r[0], "-->", " ".join(r[1:]), yf, exp(w)
	grammar = Grammar(grammar)
	testestimates(grammar, 4, grammar.toid["ROOT"])
	outside = getestimates(grammar, 4, grammar.toid["ROOT"])
	sent = ["a","b","c"]
	print "\nwithout estimates"
	chart, start, _ = parse(sent, grammar, estimates=None)
	pprint_chart(chart, sent, grammar.tolabel)
	print "\nwith estimates"
	chart, start, _ = parse(sent, grammar, estimates=(outside, 4))
	pprint_chart(chart, sent, grammar.tolabel)

if __name__ == '__main__': main()
