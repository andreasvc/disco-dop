#cython: boundscheck=False
"""
Implementation of LR estimate (Kallmeyer & Maier 2010).
Ported almost directly from rparse.
"""
from cpq import heapdict
from containers import ChartItem, Edge, Rule, Terminal
from collections import defaultdict
from math import exp
import numpy as np
try:
	import cython
	assert cython.compiled
except:
	from bit import *
	# NB the array stuff for yield functions is not compatible with plain
	# python, remove instances of '._B' and '._H' to make it work
	# e.g., rule.args._H[n] => rule.args[n]
	# also, replace new_ChartItem() and new_Edge with ChartItem() and Edge()
else:
	np.import_array()

class Item(object):
	__slots__ = ("state", "length", "lr", "gaps", "_hash")
	def __hash__(self):
		return self._hash
	def __repr__(self):
		return "%s len=%d lr=%d gaps=%d" % (self.state, self.length,
												self.lr, self.gaps)

def new_Item(state, length, lr, gaps):
	item = Item.__new__(Item)
	item.state = state; item.length = length; item.lr = lr; item.gaps = gaps
	item._hash = state * 1021 + length * 571 + lr * 311 + gaps
	return item

def getestimates(grammar, maxlen, goal):
	print "getting inside"
	insidescores = simpleinside(grammar, maxlen)
	print "getting outside"
	outside = outsidelr(grammar, insidescores, maxlen, goal)
	return outside

def testestimates(grammar, maxlen, goal):
	print "getting inside"
	insidescores = inside(grammar, maxlen)
	insidescores = simpleinside(grammar, maxlen)
	for a in insidescores:
		for b in insidescores[a]:
			print a,b
			assert 0 <= a < len(grammar.bylhs)
			assert 0 <= b <= maxlen
			print "%s[%d] =" % (grammar.tolabel[a], b), exp(insidescores[a][b])
	print 
	print len(insidescores) * sum(map(len, insidescores.values()))
	print "getting outside"
	outside = outsidelr(grammar, insidescores, maxlen, goal)
	infinity = float('infinity')
	cnt = 0
	for an, a in (): # enumerate(outside):
		for bn, b in enumerate(a):
			for cn, c in enumerate(b):
				for dn, d in enumerate(c):
					if d < infinity:
						print grammar.tolabel[an], bn, cn, dn, exp(-d)
						cnt += 1
	#print cnt
	print "done"
	return outside

def getoutside(outside, maxlen, slen, label, vec):
	""" Used during parsing to query for outside estimates """
	if slen > maxlen:
		return 0.0
	length = bitcount(vec)
	left = nextset(vec, 0)
	gaps = slen - length - left
	right = slen - length - left - gaps
	lr = left + right
	if length+lr+gaps <= maxlen:
		return outside[label, length, lr, gaps]
	else:
		return 0.0

def inside(grammar, maxlen):
	#infinity = float('infinity')
	#insidescores = defaultdict(lambda: defaultdict(lambda: infinity))
	insidescores = {}
	return doinside(grammar, maxlen, insideconcat, insidescores)

def simpleinside(grammar, maxlen):
	""" Here vec is actually the length (number of terminals in the yield of
	the constituent) """
	#infinity = float('infinity')
	#insidescores = defaultdict(lambda: defaultdict(lambda: infinity))
	insidescores = {}
	return doinside(grammar, maxlen, simpleconcat, insidescores)

def doinside(grammar, maxlen, concat, insidescores):
	lexical, unary = grammar.lexical, grammar.unary
	lbinary, rbinary = grammar.lbinary, grammar.rbinary
	agenda = heapdict()
	nil = ChartItem(0, 0)
	for tags in lexical.values():
		for rule in tags:
			agenda[new_ChartItem(rule.lhs, 1)] = new_Edge(0.0, 0.0, nil, nil)
	while agenda.length:
		entry = agenda.popentry()
		I = entry.key
		x = entry.value.inside
		if I.label not in insidescores: insidescores[I.label] =  {}
		if insidescores[I.label].get(I.vec, 0.0) < x:
			insidescores[I.label][I.vec] = x
		
		for rule in unary[I.label]:
			if (rule.lhs not in insidescores
				or I.vec not in insidescores[rule.lhs]):
				agenda.setitem(new_ChartItem(rule.lhs, I.vec),
						new_Edge(rule.prob + x, 0.0, nil, nil))
		for rule in lbinary[I.label]:
			if rule.rhs2 not in insidescores: continue
			for vec in insidescores[rule.rhs2]:
				left = concat(I.vec, vec, rule, maxlen)
				if left and (rule.lhs not in insidescores
					or left not in insidescores[rule.lhs]):
					agenda.setitem(new_ChartItem(rule.lhs, left),
						new_Edge(x + rule.prob + insidescores[rule.rhs2][vec],
									0.0, nil, nil))
		for rule in rbinary[I.label]:
			if rule.rhs1 not in insidescores: continue
			for vec in insidescores[rule.rhs1]:
				right = concat(vec, I.vec, rule, maxlen)
				if right and (rule.lhs not in insidescores
					or right not in insidescores[rule.lhs]):
					agenda.setitem(new_ChartItem(rule.lhs, right),
						new_Edge(x + rule.prob + insidescores[rule.rhs1][vec],
									0.0, nil, nil))

	return insidescores

def simpleconcat(a, b, ignored, maxlen):
	return a+b if a+b <= maxlen else 0

def insideconcat(a, b, rule, maxlen):
	if len(rule.args) + bitcount(a) + bitcount(b) > maxlen + 1:
		return
	result = resultpos = l = r = 0
	for n, arg in zip(rule.lengths, rule.args):
		for x in range(n):
			if testbitshort(arg, x) == 0:
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

def twodim_dict_to_array(d, a):
	for n, d2 in d.iteritems():
		for m, val in d2.iteritems():
			a[n, m] = val
	return
		
def outsidelr(grammar, insidescores, maxlen, goal):
	try: assert cython.compiled; print "estimates: running cython"
	except: print "estimates: not cython"
	infinity = float('infinity')
	outside = np.array([infinity], dtype='d').repeat(
				len(grammar.bylhs) * (maxlen+1) * (maxlen+1) * (maxlen+1)
				).reshape((len(grammar.bylhs), maxlen+1, maxlen+1, maxlen+1))
	npinsidescores = np.array([infinity], dtype='d').repeat(
				len(grammar.bylhs) * (maxlen+1)).reshape(
				(len(grammar.bylhs), (maxlen+1)))
	twodim_dict_to_array(insidescores, npinsidescores)
	computeoutsidelr(grammar, npinsidescores, maxlen, goal, outside)
	return outside

def computeoutsidelr(grammar, insidescores, maxlen, goal, outside):
	bylhs = grammar.bylhs
	agenda = heapdict()
	nil = new_ChartItem(0, 0)
	for a in range(maxlen):
		newitem = new_Item(goal, a + 1, 0, 0)
		agenda[newitem] = new_Edge(0.0, 0.0, nil, nil)
		outside[goal, a + 1, 0, 0] = 0.0
	print "initialized"
	while agenda.length:
		entry = agenda.popentry()
		I = entry.key
		x = entry.value.inside
		if x == outside[I.state, I.length, I.lr, I.gaps]:
			totlen = I.length + I.lr + I.gaps
			rules = bylhs[I.state]
			for r in rules:
				if isinstance(r, Terminal): continue
				rule = r
				# X -> A
				if rule.rhs2 == 0:
					if rule.rhs1 != 0:
						newitem = new_Item(rule.rhs1, I.length, I.lr, I.gaps)
						score = x + rule.prob
						if outside[rule.rhs1, I.length, I.lr, I.gaps] > score:
							agenda.setitem(newitem, new_Edge(score, 0.0, nil, nil))
							outside[rule.rhs1, I.length, I.lr, I.gaps] = score
				else:
					lstate = rule.rhs1
					rstate = rule.rhs2
					fanout = len(rule.args)
					# X -> A B
					addgaps = addright = 0
					stopaddright = False
					for m in range(fanout - 1, -1, -1):
						arg = rule.args._H[m]
						for a in range(rule.lengths._B[m] - 1, -1, -1):
							if testbitshort(arg, a) == 0:
								stopaddright = True
							else:
								if not stopaddright:
									addright += 1
								else:
									addgaps += 1
					rightarity = sum(bitcount(rule.args._H[n])
											for n in range(fanout))
					leftarity = sum(rule.lengths._B[n] for n in range(fanout))
					leftarity -= rightarity
					# binary-left (A is left)
					for lenA in range(leftarity, I.length - rightarity + 1):
						lenB = I.length - lenA
						insidescore = insidescores[rstate, lenB]
						for lr in range(I.lr, I.lr + lenB + 1):
							if addright == 0 and lr != I.lr: continue
							for ga in range(leftarity - 1, totlen+1):
								if lenA + lr + ga == I.length + I.lr + I.gaps and ga >= addgaps:
									newitem = new_Item(lstate, lenA, lr, ga)
									score = x + insidescore + rule.prob
									if outside[lstate, lenA, lr, ga] > score:
										agenda.setitem(newitem,
											new_Edge(score, 0.0, nil, nil))
										outside[lstate, lenA, lr, ga] = score

					# X -> B A
					addgaps = addleft = 0
					stopaddleft = False
					for m in range(fanout):
						arg = rule.args._H[m]
						for a in range(rule.lengths._B[m]):
							if testbitshort(arg, a):
								stopaddleft = True
							else:
								if stopaddleft:
									addgaps += 1
								else:
									addleft += 1

					addright = 0
					stopaddright = False
					for m in range(fanout -1, -1, -1):
						arg = rule.args._H[m]
						for a in range(rule.lengths._B[m] - 1, -1, -1):
							if testbitshort(arg, a):
								stopaddright = True
							else:
								if not stopaddright:
									addright += 1
					addgaps -= addright
					
					# binary-right (A is right)
					for lenA in range(rightarity, I.length - leftarity + 1):
						lenB = I.length - lenA
						insidescore = insidescores[lstate, lenB]
						for lr in range(I.lr, I.lr + lenB + 1):
							for ga in range(rightarity - 1, totlen+1):
								if lenA + lr + ga == I.length + I.lr + I.gaps and ga >= addgaps:
									newitem = new_Item(rstate, lenA, lr, ga)
									score = x + insidescore + rule.prob
									if outside[rstate, lenA, lr, ga] > score:
										agenda.setitem(newitem,
											new_Edge(score, 0.0, nil, nil))
										outside[rstate, lenA, lr, ga] = score
	pass

def main():
	from negra import NegraCorpusReader
	from grammar import induce_srcg, dop_srcg_rules, newsplitgrammar
	from nltk import Tree
	corpus = NegraCorpusReader(".", "sample2\.export", encoding="iso-8859-1")
	trees = list(corpus.parsed_sents())
	for a in trees: a.chomsky_normal_form(vertMarkov=1, horzMarkov=1)
	grammar = newsplitgrammar(dop_srcg_rules(trees, corpus.sents()))
	testestimates(grammar, 30, grammar.toid["ROOT"])
	#tree = Tree("(S (VP (VP (PROAV 0) (VVPP 2)) (VAINF 3)) (VMFIN 1))")
	#tree.chomsky_normal_form()
	#sent = "Daruber muss nachgedacht werden".split()
	#grammar = splitgrammar(dop_srcg_rules([tree]*30, [sent]*30))
	#testestimates(grammar, 6, grammar.toid["S"])

if __name__ == '__main__': main()
