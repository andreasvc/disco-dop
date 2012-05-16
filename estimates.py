"""
Implementation of LR estimate (Kallmeyer & Maier 2010).
Ported almost directly from rparse (except for sign reversal of log probs).
"""
from agenda import Agenda
from containers import ChartItem, Rule
from collections import defaultdict
from math import exp
import numpy as np
try:
	import cython
	assert cython.compiled
except:
	from bit import *
	from numpy import * # to import isfinite and isnan
	exec "new_ChartItem = ChartItem" in globals()
else:
	np.import_array()

class Item(object):
	__slots__ = ("state", "length", "lr", "gaps", "_hash")
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

def new_Item(state, length, lr, gaps):
	item = Item.__new__(Item)
	item.state = state; item.length = length; item.lr = lr; item.gaps = gaps
	item._hash = state * 1021 + length * 571 + lr * 311 + gaps
	return item

def getoutside(outside, maxlen, slen, label, vec):
	""" Used during parsing to query for outside estimates """
	length = bitcount(vec)
	left = nextset(vec, 0)
	gaps = bitlength(vec) - length - left
	right = slen - length - left - gaps
	lr = left + right
	if slen > maxlen or length + lr + gaps > maxlen:
		return 0.0
	return outside[label, length, lr, gaps]

def simpleinside(grammar, maxlen, insidescores):
	""" Compute simple inside estimate in bottom-up fashion.
	Here vec is actually the length (number of terminals in the yield of
	the constituent)
	insidescores is a 4-dimensional matrix initialized with NaN to indicate
	values that have yet to be computed. """
	lexical, unary = grammar.lexical, grammar.unary
	lbinary, rbinary = grammar.lbinary, grammar.rbinary
	infinity = np.inf
	agenda = Agenda()
	nil = ChartItem(0, 0)

	for n, rules in enumerate(grammar.bylhs):
		if n == 0: continue
		#this is supposed cover all and only preterminals
		elif rules == []:
			agenda[new_ChartItem(n, 1)] = 0.0

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

		rules = unary[I.label]
		for rule in rules:
			if isnan(insidescores[rule.lhs, I.vec]):
				agenda.setifbetter(
						new_ChartItem(rule.lhs, I.vec), rule.prob + x)

		rules = lbinary[I.label]
		for rule in rules:
			for vec in range(1, maxlen - I.vec + 1):
				if (isfinite(insidescores[rule.rhs2, vec])
					and isnan(insidescores[rule.lhs, I.vec + vec])):
					agenda.setifbetter(new_ChartItem(rule.lhs, I.vec + vec),
						rule.prob + x + insidescores[rule.rhs2, vec])

		rules = rbinary[I.label]
		for rule in rules:
			for vec in range(1, maxlen - I.vec + 1):
				if (isfinite(insidescores[rule.rhs1, vec])
					and isnan(insidescores[rule.lhs, vec + I.vec])):
					agenda.setifbetter(new_ChartItem(rule.lhs, vec + I.vec),
						rule.prob + insidescores[rule.rhs1, vec] + x)

	# anything not reached so far gets probability zero:
	insidescores[np.isnan(insidescores)] = infinity

def outsidelr(grammar, insidescores, maxlen, goal, arity, outside):
	""" Compute the outside SX simple LR estimate in top down fashion. """
	bylhs = grammar.bylhs
	infinity = np.inf
	agenda = Agenda()
	nil = new_ChartItem(0, 0)
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
		rules = bylhs[I.state]
		for rule in rules:
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
			for m in range(arity[rule.lhs] - 1, -1, -1):
				for n in range(rule._lengths[m] - 1, -1, -1):
					if (not stopaddright
						and not testbitint(rule._args[m], n)):
						stopaddright = True
					if testbitint(rule._args[m], n):
						if not stopaddright:
							addright += 1
						else:
							addgaps += 1

			leftarity = arity[rule.rhs1]
			rightarity = arity[rule.rhs2]
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
			for m in range(arity[rule.lhs]):
				for n in range(rule._lengths[m]):
					if not stopaddleft and testbitint(rule._args[m], n):
						stopaddleft = True
					if not testbitint(rule._args[m], n):
						if stopaddleft:
							addgaps += 1
						else:
							addleft += 1

			stopaddright = False
			for m in range(arity[rule.lhs] - 1, -1, -1):
				for n in range(rule._lengths[m] - 1, -1, -1):
					if (not stopaddright
						and testbitint(rule._args[m], n)):
						stopaddright = True
					if not testbitint(rule._args[m], n):
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

def inside(grammar, maxlen, insidescores):
	""" Compute inside estimate in bottom-up fashion, with
	full bit vectors (not used)."""
	lexical, unary = grammar.lexical, grammar.unary
	lbinary, rbinary = grammar.lbinary, grammar.rbinary
	infinity = float('infinity')
	agenda = Agenda()
	nil = ChartItem(0, 0)

	for n, rules in enumerate(grammar.bylhs):
		if rules == []:
			agenda[new_ChartItem(n, 1)] = 0.0

	while agenda.length:
		entry = agenda.popentry()
		I = entry.key
		x = entry.value
		if I.label not in insidescores: insidescores[I.label] =  {}
		if x < insidescores[I.label].get(I.vec, infinity):
			insidescores[I.label][I.vec] = x

		for rule in unary[I.label]:
			if (rule.lhs not in insidescores
				or I.vec not in insidescores[rule.lhs]):
				agenda.setifbetter(
					new_ChartItem(rule.lhs, I.vec), rule.prob + x)

		for rule in lbinary[I.label]:
			if rule.rhs2 not in insidescores: continue
			for vec in insidescores[rule.rhs2]:
				left = insideconcat(I.vec, vec, rule, maxlen)
				if left and (rule.lhs not in insidescores
					or left not in insidescores[rule.lhs]):
					agenda.setifbetter(new_ChartItem(rule.lhs, left),
						rule.prob + x + insidescores[rule.rhs2][vec])

		for rule in rbinary[I.label]:
			if rule.rhs1 not in insidescores: continue
			for vec in insidescores[rule.rhs1]:
				right = insideconcat(vec, I.vec, rule, maxlen)
				if right and (rule.lhs not in insidescores
					or right not in insidescores[rule.lhs]):
					agenda.setifbetter(new_ChartItem(rule.lhs, right),
						rule.prob + insidescores[rule.rhs1][vec] + x)

	return insidescores

def insideconcat(a, b, rule, maxlen):
	if len(rule.args) + bitcount(a) + bitcount(b) > maxlen + 1:
		return 0
	result = resultpos = l = r = 0
	for n, arg in zip(rule.lengths, rule.args):
		for x in range(n):
			if testbitint(arg, x) == 0:
				subarg = nextunset(a, l) - l
				result |= (1 << subarg) - 1 << resultpos
				resultpos += subarg
				l = subarg + 1
			else:
				subarg = nextunset(b, r) - r
				result |= (1 << subarg) - 1 << resultpos
				resultpos += subarg
				r = subarg + 1
		resultpos += 1
		result &= ~(1 << resultpos)
	return result

def getestimates(grammar, maxlen, goal):
	try: assert cython.compiled; print "estimates: running cython"
	except: print "estimates: not cython"
	insidescores = np.array([np.NAN], dtype='d').repeat(
				len(grammar.bylhs) * (maxlen+1)).reshape(
				(len(grammar.bylhs), (maxlen+1)))
	outside = np.array([np.inf], dtype='d').repeat(
				len(grammar.bylhs) * (maxlen+1) * (maxlen+1) * (maxlen+1)
				).reshape((len(grammar.bylhs), maxlen+1, maxlen+1, maxlen+1))
	print "getting inside"
	simpleinside(grammar, maxlen, insidescores)
	print "getting outside"
	outsidelr(grammar, insidescores, maxlen, goal, grammar.arity, outside)
	return outside

def testestimates(grammar, maxlen, goal):
	infinity = np.inf
	print "getting inside"
	insidescores = inside(grammar, maxlen, {})
	for a in insidescores:
		for b in insidescores[a]:
			assert 0 <= a < len(grammar.bylhs)
			assert 0 <= bitlength(b) <= maxlen
			#print a,b
			#print "%s[%d] =" % (grammar.tolabel[a], b), exp(insidescores[a][b])
	print len(insidescores) * sum(map(len, insidescores.values())), '\n'
	insidescores = np.array([np.NAN], dtype='d').repeat(
				len(grammar.bylhs) * (maxlen+1)).reshape(
				(len(grammar.bylhs), (maxlen+1)))
	simpleinside(grammar, maxlen, insidescores)
	print "inside"
	for an, a in enumerate(insidescores):
		for bn, b in enumerate(a):
			if b < infinity:
				print grammar.tolabel[an], "len", bn, "=", exp(-b)
	#print insidescores
	#for a in range(maxlen):
	#	print grammar.tolabel[goal], "len", a, "=", exp(-insidescores[goal, a])

	print "getting outside"
	outside = np.array([np.inf], dtype='d').repeat(
				len(grammar.bylhs) * (maxlen+1) * (maxlen+1) * (maxlen+1)
				).reshape((len(grammar.bylhs), maxlen+1, maxlen+1, maxlen+1))
	outsidelr(grammar, insidescores, maxlen, goal, grammar.arity, outside)
	#print outside
	cnt = 0
	for an, a in enumerate(outside):
		for bn, b in enumerate(a):
			for cn, c in enumerate(b):
				for dn, d in enumerate(c):
					if d < infinity:
						print grammar.tolabel[an], "length", bn, "lr", cn,
						print "gaps", dn, "=", exp(-d)
						cnt += 1
	print cnt
	print "done"
	return outside

def main():
	from treebank import NegraCorpusReader
	from grammar import induce_srcg, dop_srcg_rules, splitgrammar
	from plcfrs import parse, pprint_chart
	from nltk import Tree
	corpus = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1")
	trees = list(corpus.parsed_sents())
	for a in trees: a.chomsky_normal_form(vertMarkov=1, horzMarkov=1)
	grammar = splitgrammar(induce_srcg(trees, corpus.sents()))
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
	grammar = induce_srcg(trees, sents)
	for (r,yf),w in sorted(grammar):
		print r[0], "-->", " ".join(r[1:]), yf, exp(w)
	grammar = splitgrammar(grammar)
	testestimates(grammar, 4, grammar.toid["ROOT"])
	outside = getestimates(grammar, 4, grammar.toid["ROOT"])
	sent = ["a","b","c"]
	print "\nwithout estimates"
	chart, start = parse(sent, grammar, estimate=None)
	pprint_chart(chart, sent, grammar.tolabel)
	print "\nwith estimates"
	chart, start = parse(sent, grammar, estimate=(outside, 4))
	pprint_chart(chart, sent, grammar.tolabel)

if __name__ == '__main__': main()
