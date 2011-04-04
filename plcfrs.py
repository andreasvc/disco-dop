# probabilistic CKY parser for Simple Range Concatenation Grammars
# (equivalent to Linear Context-Free Rewriting Systems)
from rcgrules import enumchart
from dopg import removeids
from nltk import FreqDist
from heapdict import heapdict
#from pyjudy import JudyLObjObj
from bitarray import bitarray
from math import log, e, floor
from random import choice
from itertools import chain, islice
from pprint import pprint
from collections import defaultdict
from operator import or_
import re
#try:
#	import pyximport
#	pyximport.install()
#except: pass
#from bit import *
#try:
#	import psyco
#	psyco.full()
#except: pass

class ChartItem:
	__slots__ = ("label", "vec", "_hash")
	def __init__(self, label, vec):
		self.label = label
		self.vec = vec
		self._hash = hash((self.label, self.vec))
	def __hash__(self):
		return self._hash
	def __eq__(self, other):
		return self.label == other.label and self.vec == other.vec
	def __getitem__(self, n):
		if n == 0: return self.label
		if n == 1: return self.vec
	def __repr__(self):
		#would need bitlen for proper padding
		return "%s[%s]" % (self.label, bin(self.vec)[2:][::-1]) 

def parse(sent, grammar, tags=None, start=None, viterbi=False, n=1):
	""" parse sentence, a list of tokens, using grammar, a dictionary
	mapping rules to probabilities. """
	unary, lbinary, rbinary, bylhs, toid, tolabel = grammar
	if start == None: start = toid['S']
	goal = ChartItem(start, (1 << len(sent)) - 1)
	m = maxA = 0
	A, C, Cx = heapdict(), defaultdict(list), defaultdict(dict)
	#C = JudyLObjObj()
	#from guppy import hpy; h = hpy(); hn = 0
	#h.heap().stat.dump("/tmp/hstat%d" % hn); hn+=1

	# scan
	Epsilon = toid["Epsilon"]
	for i,w in enumerate(sent):
		recognized = False
		for rule, z in unary[Epsilon]:
			if w in rule[1]:
				Ih = ChartItem(rule[0][0], 1 << i)
				I = (ChartItem(Epsilon, i),)
				A[Ih] = ((z, z), I)
				recognized = True
		if not recognized and tags:
			Ih = ChartItem(tags[i], 1 << i)
			I = (ChartItem(Epsilon, i),)
			A[Ih] = ((0, 0), I)
			recognized = True
			continue
		elif not recognized:
			print "not covered:", w
	lensent = len(sent)
	# parsing
	while A:
		Ih, xI = A.popitem()
		#when heapdict is not available:
		#Ih, (x, I) = min(A.items(), key=lambda x:x[1]); del A[Ih]
		#C[Ih] = I, x
		(y, p), b = xI
		C[Ih].append((b, -p))
		Cx[Ih.label][Ih] = y
		if Ih == goal:
			m += 1	#problem: this is not viterbi n-best.
			#goal = Ih
			if viterbi and n==m: break
		else:
			for I1h, yI1 in deduced_from(Ih, xI[0][0], Cx, unary, lbinary, rbinary, lensent):
				# explicit get to avoid inserting spurious keys in defaultdict
				if I1h not in Cx.get(I1h.label, {}) and I1h not in A:
					A[I1h] = yI1
				elif I1h in A and yI1[0][0] > A[I1h][0][0]: 
					A[I1h] = yI1
				else:
					(y, p), b = yI1
					C[I1h].insert(0, (b, -p))
		maxA = max(maxA, len(A))
		#pass #h.heap().stat.dump("/tmp/hstat%d" % hn); hn+=1
		##print h.iso(A,C,Cx).referents | h.iso(A, C, Cx)
	print "max agenda size", maxA, "/ chart keys", len(C), "/ values", sum(map(len, C.values()))
	#h.pb(*("/tmp/hstat%d" % a for a in range(hn)))
	return (C, goal) if goal in C else ({}, ())

def deduced_from(Ih, x, Cx, unary, lbinary, rbinary, bitlen):
	I, Ir = Ih.label, Ih.vec
	result = []
	for rule, z in unary[I]:
		result.append((ChartItem(rule[0][0], Ir), ((x+z,z), (Ih,))))
	for rule, z in lbinary[I]:
		for I1h, y in Cx[rule[0][2]].items():
			if concat(rule[1], Ir, I1h.vec, bitlen):
				result.append((ChartItem(rule[0][0], Ir ^ I1h.vec), ((x+y+z, z), (Ih, I1h))))
	for rule, z in rbinary[I]:
		for I1h, y in Cx[rule[0][1]].items():
			if concat(rule[1], I1h.vec, Ir, bitlen):
				result.append((ChartItem(rule[0][0], I1h.vec ^ Ir), ((x+y+z, z), (I1h, Ih))))
	return result

def concat(yieldfunction, lvec, rvec, bitlen):
	if lvec & rvec: return False
	if len(yieldfunction) == 1 and len(yieldfunction[0]) == 2:
		if yieldfunction[0][0] == 0 and yieldfunction[0][1] == 1:
			return bitminmax(lvec, rvec)
		elif yieldfunction[0][0] == 1 and yieldfunction[0][1] == 0:
			return bitminmax(rvec, lvec)
		else: raise ValueError("non-binary element in yieldfunction")
	#this algorithm taken from rparse FastYFComposer.
	lpos = nextset(lvec, 0)
	rpos = nextset(rvec, 0)
	for arg in yieldfunction:
		m = len(arg) - 1
		for n, b in enumerate(arg):
			if b == 0:
				# check if there are any bits left, and
				# if any bits on the right should have gone before
				# ones on this side
				if lpos == -1 or (rpos != -1 and rpos <= lpos):
					return False
				# jump to next gap
				lpos = nextunset(lvec, lpos)
				# there should be a gap if and only if
				# this is the last element of this argument
				if rpos != -1 and rpos < lpos: return False
				if n == m:
					if testbit(rvec, lpos): return False
				elif not testbit(rvec, lpos): return False
				#jump to next argument
				lpos = nextset(lvec, lpos)
			elif b == 1:
				# vice versa to the above
				if rpos == -1 or (lpos != -1 and lpos <= rpos):
					return False
				rpos = nextunset(rvec, rpos)
				if lpos != -1 and lpos < rpos: return False
				if n == m:
					if testbit(lvec, rpos): return False
				elif not testbit(lvec, rpos): return False
				rpos = nextset(rvec, rpos)
			else: raise ValueError("non-binary element in yieldfunction")
	if lpos != -1 or rpos != -1:
		return False
	# everything looks all right
	return True

# bit operations adapted from http://wiki.python.org/moin/BitManipulation
def nextset2(a, pos, bitlen):
	result = 0
	bitlen -= pos
	a >>= pos
	a = a & -a
	while a >> result and result < bitlen:
		result += 1
	return pos + result - 1 if result < bitlen else -1

def nextunset2(a, pos, bitlen):
	result = 0
	bitlen -= pos
	a >>= pos
	a = ~a & -~a
	while a >> result and result < bitlen:
		result += 1
	return pos + result - 1

def nextset(a, pos):
	result = pos
	while (not (a >> result) & 1) and a >> result:
		result += 1
	return result if a >> result else -1

def nextunset(a, pos):
	result = pos
	while (a >> result) & 1:
		result += 1
	return result

def testbit(a, offset):
	return a & (1 << offset)

def bitcount(a):
	count = 0
	while a:
		a &= a - 1
		count += 1
	return count

def bitminmax(a, b):
	"""test if the least and most significant bits of a and b are 
	consecutive. we shift a and b until they meet in the middle (return true)
	or collide (return false)"""
	b = (b & -b)
	while a and b:
		a >>= 1
		b >>= 1
	return b == 1

def filterchart(chart, start):
	# remove all entries that do not contribute to a complete derivation
	def filter_subtree(start, chart, chart2):
		if isinstance(start, int) or chart2[start]: return True
		else: chart2[start] = [(x,p) for x,p in chart[start] if all(filter_subtree(a, chart, chart2) for a in x)]
		return chart2[start] != []
	chart2 = defaultdict(list)
	filter_subtree(start, chart, chart2)
	return chart2

def samplechart(chart, start, lolabel):
	entry, p = choice(chart[start])
	if len(entry) == 1 and entry[0] not in chart: # entry[0][0] == "Epsilon":
		return "(%s %d)" % (tolabel[start.label], entry[0][1]), p
	children = [samplechart(chart, a, tolabel) for a in entry]
	tree = "(%s %s)" % (tolabel[start.label], " ".join([a for a,b in children]))
	#tree = "(%s_%s %s)" % (start[0], "_".join(repr(a) for a in start[1:]), " ".join([a for a,b in children]))
	return tree, p+sum(b for a,b in children)
	
def mostprobableparse(chart, start, tolabel, n=100, sample=False):
		""" sum over n random derivations from chart,
			return a FreqDist of parse trees, with .max() being the MPP"""
		print "sample =", sample,
		if sample:
			for a,b in chart.items():
				if not len(b): print "spurious chart entry", a
			derivations = set(samplechart(chart, start, tolabel) for x in range(n))
			derivations.discard(None)
			#todo: calculate real parse probabilities
		else:
			#chart = filterchart(chart, start)
			#for a in chart: chart[a].sort(key=lambda x: x[1], reverse=True)
			derivations = islice(enumchart(chart, start, tolabel), n)
		parsetrees = defaultdict(float)
		m = 0
		for a,prob in derivations:
			# if necessary, we could do the addition in log space
			parsetrees[re.sub("@[0-9]+","",a)] += e**prob
			m += 1
		print "(%d derivations)" % m
		return parsetrees

def pprint_chart(chart, sent, tolabel):
	print "chart:"
	for a in sorted(chart, key=lambda x: bitcount(x[1])):
		print tolabel[a[0]], bin(a[1]), "=>"
		for b,p in chart[a]:
			for c in b:
				if tolabel[c[0]] == "Epsilon":
					print "\t", repr(sent[b[0][1]]),
				else:
					print "\t", tolabel[c[0]], bin(c[1]),
			print e**p
		print

def do(sent):
	print "sentence", sent
	chart, start = parse(sent.split(), grammar)
	pprint_chart(chart, sent.split(), grammar[5])
	if chart:
		for a, p in mostprobableparse(chart, start, grammar[5], n=1000).items():
			print p, a
	else: print "no parse"
	print

if __name__ == '__main__':
	from rcgrules import splitgrammar
	grammar = splitgrammar(
		[((('S','VP2','VMFIN'),    ((0,1,0),)),  0),
		((('VP2','VP2','VAINF'),  ((0,),(0,1))), log(0.5)),
		((('VP2','PROAV','VVPP'), ((0,),(1,))), log(0.5)),
		((('PROAV', 'Epsilon'), ('Daruber', ())), 0.0),
		((('VAINF', 'Epsilon'), ('werden', ())), 0.0),
		((('VMFIN', 'Epsilon'), ('muss', ())), 0.0),
		((('VVPP', 'Epsilon'), ('nachgedacht', ())), 0.0)])
	do("Daruber muss nachgedacht werden")
	do("Daruber muss nachgedacht werden werden")
	do("Daruber muss nachgedacht werden werden werden")
	do("muss Daruber nachgedacht werden")	#no parse
