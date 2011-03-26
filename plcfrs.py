# probabilistic CKY parser for Simple Range Concatenation Grammars
# (equivalent to Linear Context-Free Rewriting Systems)
from rcgrules import enumchart
from nltk import FreqDist
from bitarray import bitarray
from heapdict import heapdict
#from pyjudy import JudyLObjObj
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
myintern = {}
class MyBitArray(bitarray):
	# this should go into a patch of bitarray
	def __hash__(self):
		return hash(self.tostring())

def parse(sent, grammar, start="S", viterbi=False, n=1):
	""" parse sentence, a list of tokens, using grammar, a dictionary
	mapping rules to probabilities. """
	unary, binary = defaultdict(list), defaultdict(list)
	# negate the log probabilities because the heap is a min-heap
	for r,w in grammar:
		if len(r[0]) == 2: unary[r[0][1]].append((r, -w))
		elif len(r[0]) == 3: binary[(r[0][1], r[0][2])].append((r, -w))
	vec = bitarray(len(sent))
	vec.setall(True)
	goal = (start, vec)
	m = maxA = 0
	A, C, Cx = heapdict(), defaultdict(set), {}
	#C = JudyLObjObj()
	#from guppy import hpy; h = hpy(); hn = 0
	#h.heap().stat.dump("/tmp/hstat%d" % hn); hn+=1

	# scan
	for i,w in enumerate(sent):
		recognized = False
		for rule, z in unary["Epsilon"]:
			if w in rule[1]:
				vec = MyBitArray(len(sent))
				vec.setall(False)
				vec[i] = True
				Ih = interntuple(rule[0][0], vec)
				I = (("Epsilon", (i,)),)
				A[Ih] = ((z, z), I)
				recognized = True
		if not recognized: print "not covered:", w

	# parsing
	while A:
		Ih, xI = A.popitem()
		#when heapdict is not available:
		#Ih, (x, I) = min(A.items(), key=lambda x:x[1]); del A[Ih]
		#C[Ih] = I, x
		(foo, p), b = xI
		C[Ih].add((b, -p))
		Cx[Ih] = xI
		if Ih == goal:
			m += 1	#problem: this is not viterbi n-best.
			goal = Ih
			if viterbi and n==m: break
		else:
			for I1h, yI1 in deduced_from(Ih, xI[0][0], Cx, unary, binary):
				if I1h not in C and I1h not in A:
					A[I1h] = yI1
				elif I1h in A:
					if yI1[0][0] > A[I1h][0][0]: A[I1h] = yI1
				else:
					(y, p), b = yI1
					C[I1h].add((b, -p))
		maxA = max(maxA, len(A))
		#pass #h.heap().stat.dump("/tmp/hstat%d" % hn); hn+=1
		##print h.iso(A,C,Cx).referents | h.iso(A, C, Cx)
	print "max agenda size", maxA, "/ chart items", len(C)
	#h.pb(*("/tmp/hstat%d" % a for a in range(hn)))
	return (C, goal) if goal in C else ({}, ())

def deduced_from(Ih, x, C, unary, binary):
	I, Ir = Ih
	result = []
	for rule, z in unary[I]:
		for a,b in zip(rule[1][1], Ir): a.append(b)
		left = concat(rule[1][0])
		if left: result.append((interntuple(rule[0][0], left), ((x+z,z,z), (Ih,))))
	for key in C.keys():
		#detect overlap in ranges
		I1, I1r = key
		#if Ir & I1r: continue
		y = C[key][0][0]
		for rule, z in binary[(I, I1)]:
			left = concat(rule[1], Ir, I1r)
			if left: 
				result.append((interntuple(rule[0][0], left), ((x+y+z,z), (Ih, key))))
		for rule, z in binary[(I1, I)]:
			right = concat(rule[1], I1r, Ir)
			if right: 
				result.append((interntuple(rule[0][0], right), ((x+y+z,z), (key, Ih))))
	return result

def concat(yieldfunction, lvec, rvec):
	#this algorithm taken from rparse FastYFComposer.
	lpos = lvec.index(True)
	rpos = rvec.index(True)
	for arg in yieldfunction:
		m = len(arg) - 1
		for n,b in enumerate(arg):
			if b == 0:
				# check if there are any bits left, and
				# if any bits on the right should have gone before
				# ones on this side
				if lpos == -1 or (rpos != -1 and rpos <= lpos):
					return None
				# jump to next gap
				try: lpos += lvec[lpos:].index(False)
				except: lpos = -1
				# there should be a gap if and only if
				# this is the last element of this argument
				if rpos != -1 and rpos < lpos: return None
				if n == m:
					if rvec[lpos]: return None
				elif not rvec[lpos]: return None
				#jump to next argument
				try: lpos += lvec[lpos:].index(True)
				except: lpos = -1
			elif b == 1:
				# vice versa to the above
				if rpos == -1 or (lpos != -1 and lpos <= rpos):
					return None
				try: rpos += rvec[rpos:].index(False)
				except: rpos = -1
				if lpos != -1 and lpos < rpos: return None
				if n == m:
					if lvec[rpos]: return None
				elif not lvec[rpos]: return None
				try: rpos += rvec[rpos:].index(True) 
				except: rpos = -1
			else: raise
	if lpos != -1 or rpos != -1:
		return None
	# finally, return composed vector
	return lvec | rvec

def interntuple(*a):
	#like intern but for tuples: return a canonical reference so that tuples are never stored twice. 
	#doesn't seem to make any difference, unfortunately.
	# note the *. wrong: interntuple([0,1]) correct interntuple(0,1)
	return a
	#return myintern.setdefault(a, a)

# adapted from http://wiki.python.org/moin/BitManipulation
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

def samplechart(chart, start):
	entry, p = choice(chart[start])
	if len(entry) == 1 and entry[0][0] == "Epsilon":
		return "(%s %d)" % (start[0], entry[0][1][0]), p
	children = [samplechart(chart, a) for a in entry]
	tree = "(%s %s)" % (start[0], " ".join([a for a,b in children]))
	#tree = "(%s_%s %s)" % (start[0], "_".join(repr(a) for a in start[1:]), " ".join([a for a,b in children]))
	return tree, p+sum(b for a,b in children)
	
def mostprobableparse(chart, start, n=100, sample=False):
		""" sum over n random derivations from chart,
			return a FreqDist of parse trees, with .max() being the MPP"""
		print "sample =", sample,
		if sample:
			for a,b in chart.items():
				if not len(b): print "spurious chart entry", a
			derivations = set(samplechart(chart, start) for x in range(n))
			derivations.discard(None)
			#todo: calculate real parse probabilities
		else:
			#chart = filterchart(chart, start)
			#for a in chart: chart[a].sort(key=lambda x: x[1], reverse=True)
			derivations = islice(enumchart(chart, start), n)
		parsetrees = FreqDist()
		m = 0
		for n,(a,prob) in enumerate(derivations):
			parsetrees.inc(re.sub(r"@[0-9]+", "", a), e**prob)
			m+=1
		print "(%d derivations)" % m
		return parsetrees

def pprint_chart(chart, sent):
	print "chart:"
	for a in sorted(chart, key=lambda x: x[1].count()):
		print "%s[%s]" % (a[0], a[1].to01()), "=>"
		for b,p in chart[a]:
			for c in b:
				if c[0] == "Epsilon":
					print "\t", repr(sent[b[0][1][0]]),
				else:
					print "\t", "%s[%s]" % (c[0], c[1].to01()),
			print p
		print

def do(sent):
	print "sentence", sent
	chart, start = parse(sent.split(), grammar)
	pprint_chart(chart, sent.split())
	if chart:
		for a, p in mostprobableparse(chart, start, n=1000).items():
			print p, a
	else: print "no parse"
	print

if __name__ == '__main__':
	grammar = [((('S','VP2','VMFIN'),    ((0,1,0),)),  0),
		((('VP2','VP2','VAINF'),  ((0,),(0,1))), log(0.5)),
		((('VP2','PROAV','VVPP'), ((0,),(1,))), log(0.5)),
		((('PROAV', 'Epsilon'), ('Daruber', ())), 0.0),
		((('VAINF', 'Epsilon'), ('werden', ())), 0.0),
		((('VMFIN', 'Epsilon'), ('muss', ())), 0.0),
		((('VVPP', 'Epsilon'), ('nachgedacht', ())), 0.0)]

	do("Daruber muss nachgedacht werden")
	do("Daruber muss nachgedacht werden werden")
	do("Daruber muss nachgedacht werden werden werden")
	do("muss Daruber nachgedacht werden")	#no parse
