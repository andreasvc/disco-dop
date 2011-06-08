#cython: boundscheck=False
"""
Implementation of LR estimate (Kallmeyer & Maier 2010).
Ported almost directly from rparse.
"""
from heapdict import heapdict
from containers import ChartItem, Rule, Terminal
from collections import defaultdict
from math import exp
try:
	import cython
	assert cython.compiled
except:
	exec "from bit import *" in globals()

class Item(object):
	__slots__ = ("state", "length", "lr", "gaps", "_hash")
	def __init__(self, state, length, lr, gaps):
		self.state, self.length, self.lr, self.gaps = state, length, lr, gaps
		self._hash = hash((state, length, lr, gaps))
	def __hash__(self):
		return self._hash
	def __repr__(self):
		return "%s len=%d lr=%d gaps=%d" % (self.state, self.length, self.lr, self.gaps)

def getestimates(grammar, maxlen, goal):
	print "getting inside"
	insidescores = simpleinside(grammar, maxlen)
	print "getting outside"
	outside = outsidelr(grammar, insidescores, maxlen, goal)
	return outside

def testestimates(grammar, maxlen, goal):
	#insidescores = inside(grammar, maxlen)
	#for a in insidescores:
	#	for b in insidescores[a]:
	#		print "%s[%s] =" % (a, bin(b)[2:]), exp(insidescores[a][b])
	#x=len(insidescores) * sum(map(len, insidescores.values()))
	#print
	print "getting inside"
	insidescores = simpleinside(grammar, maxlen)
	insidescores = inside(grammar, maxlen)
	for a in insidescores:
		for b in insidescores[a]:
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
	if slen > maxlen: return 0.0
	length = bitcount(vec)
	left = nextset(vec, 0)
	gaps = slen - length - left
	right = slen - length - left - gaps
	lr = left + right
	if length+lr+gaps <= maxlen: return outside[label][length][lr][gaps]
	else: return 0.0

def inside(grammar, maxlen):
	insidescores = defaultdict(lambda: defaultdict(lambda: float('infinity')))
	return doinside(grammar, maxlen, insideconcat, insidescores)

def simpleinside(grammar, maxlen):
	""" Here vec is actually the length (number of terminals in the yield of the constituent) """
	insidescores = defaultdict(lambda: defaultdict(lambda: float('infinity')))
	return doinside(grammar, maxlen, simpleconcat, insidescores)

def doinside(grammar, maxlen, concat, insidescores):
	lexical, unary, lbinary, rbinary = grammar.lexical, grammar.unary, grammar.lbinary, grammar.rbinary
	agenda = heapdict()
	for tags in lexical.values():
		for rule in tags:
			agenda[ChartItem(rule.lhs, 1)] = 0.0
	while agenda:
		I, x = agenda.popitem()
		if I.vec not in insidescores[I.label] or insidescores[I.label][I.vec] < x:
			insidescores[I.label][I.vec] = x
		
		results = []
		for rule in unary[I.label]:
			results.append((rule.lhs, I.vec, rule.prob + insidescores[rule.rhs1][I.vec]))
		for rule in lbinary[I.label]:
			for vec in insidescores[rule.rhs2]:
				left = concat(I.vec, vec, rule, maxlen)
				if left:
					results.append((rule.lhs, left,
						x + rule.prob + insidescores[rule.rhs2][vec]))
		for rule in rbinary[I.label]:
			for vec in insidescores[rule.rhs1]:
				right = concat(vec, I.vec, rule, maxlen)
				if right:
					results.append((rule.lhs, right,
							x + rule.prob + insidescores[rule.rhs1][vec]))

		for label, vec, score in results:
			if label not in insidescores or vec not in insidescores[label]:
				agenda[ChartItem(label, vec)] = score
	return insidescores

def simpleconcat(a, b, ignored, maxlen):
	return a+b if a+b <= maxlen else 0

def insideconcat(a, b, rule, maxlen):
	#if not (a and b and yieldfunction)  ???
	if len(rule.args) + bitcount(a) + bitcount(b) > maxlen + 1:
		return
	result = resultpos = l = r = 0
	for n, arg in zip(rule.lengths, rule.args):
		for x in range(n):
			if testbitc(arg, x) == 0:
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

def outsidelr(grammar, insidescores, maxlen, goal):
	try: assert cython.compiled; print "estimates: running cython"
	except: print "estimates: not cython"
	bylhs = grammar.bylhs
	agenda = heapdict()
	infinity = float('infinity')
	# this should become a numpy array if that is advantageous:
	outside = [[[[infinity] * (maxlen+1) for b in range(maxlen - c + 1)] for c in range(maxlen+1)] for lhs in bylhs]
	for a in range(1, maxlen+1):
		newitem = Item(goal, a, 0, 0)
		agenda[newitem] = 0.0
		outside[goal][a][0][0] = 0.0
	print "initialized"
	while agenda:
		I, x = agenda.popitem()
		if x == outside[I.state][I.length][I.lr][I.gaps]:
			totlen = I.length + I.lr + I.gaps
			for r in bylhs[I.state]:
				if isinstance(r, Terminal): continue
				rule = r
				# X -> A
				if rule.rhs2 == 0:
					if rule.rhs1 != 0:
						newitem = Item(rule.rhs1, I.length, I.lr, I.gaps)
						score = x + rule.prob
						if outside[rule.rhs1][I.length][I.lr][I.gaps] > score:
							agenda[newitem] = score
							outside[rule.rhs1][I.length][I.lr][I.gaps] = score
				else:
					lstate = rule.rhs1
					rstate = rule.rhs2
					# X -> A B
					addgaps = addright = 0
					stopaddright = False
					for n, arg in zip(rule.lengths, rule.args)[::-1]:
						for a in range(n - 1, -1, -1):
							if testbitc(arg, a) == 0:
								stopaddright = True
							else:
								if not stopaddright:
									addright += 1
								else:
									addgaps += 1
					rightarity = sum(bitcount(arg) for arg in rule.args)
					leftarity = sum(rule.lengths) - rightarity
					# binary-left (A is left)
					for lenA in range(leftarity, I.length - rightarity + 1):
						lenB = I.length - lenA
						insidescore = insidescores[rstate][lenB]
						for lr in range(I.lr, I.lr + lenB + 1):
							if addright == 0 and lr != I.lr: continue
							for ga in range(leftarity - 1, totlen+1):
								if lenA + lr + ga == I.length + I.lr + I.gaps and ga >= addgaps:
									newitem = Item(lstate, lenA, lr, ga)
									score = x + insidescore + rule.prob
									#print lstate, lenA, lr, ga
									#print len(outside), len(outside[0]), len(outside[0][0]), len(outside[0][0][0])
									if outside[lstate][lenA][lr][ga] > score:
										agenda[newitem] = score
										outside[lstate][lenA][lr][ga] = score

					# X -> B A
					addgaps = addleft = 0
					stopaddleft = False
					for n, arg in zip(rule.lengths, rule.args):
						for a in range(n):
							if testbitc(arg, a):
								stopaddleft = True
							else:
								if stopaddleft:
									addgaps += 1
								else:
									addleft += 1

					addright = 0
					stopaddright = False
					for n, arg in zip(rule.lengths, rule.args)[::-1]:
						for a in range(n - 1, -1, -1):
							if testbitc(arg, a):
								stopaddright = True
							else:
								if not stopaddright:
									addright += 1
					addgaps -= addright
					
					# binary-right (A is right)
					for lenA in range(rightarity, I.length - leftarity + 1):
						lenB = I.length - lenA
						insidescore = insidescores[lstate][lenB]
						for lr in range(I.lr, I.lr + lenB + 1):
							for ga in range(rightarity - 1, totlen+1):
								if lenA + lr + ga == I.length + I.lr + I.gaps and ga >= addgaps:
									newitem = Item(rstate, lenA, lr, ga)
									score = x + insidescore + rule.prob
									if outside[rstate][lenA][lr][ga] > score:
										agenda[newitem] = score
										outside[rstate][lenA][lr][ga] = score
		#else: print I,x, outside[I.state][I.length][I.lr][I.gaps]

	return outside

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
