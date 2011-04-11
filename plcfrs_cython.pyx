# probabilistic CKY parser for Simple Range Concatenation Grammars
# (equivalent to Linear Context-Free Rewriting Systems)
from rcgrules import enumchart
from dopg import removeids
from nltk import FreqDist, Tree
from heapdict import heapdict
#from pyjudy import JudyLObjObj
from bitarray import bitarray
from math import log, e, exp, floor
from random import choice, randrange
from itertools import chain, islice
from collections import defaultdict, deque
from operator import or_
import re
#try:
#	import pyximport
#	pyximport.install()
#except: pass
#from bit import *

cdef extern from "bit.h":
	int nextset(unsigned long vec, int pos)
	int nextunset(unsigned long vec, int pos)
	bint bitminmax(unsigned long a, unsigned long b)

cdef class ChartItem:
	cdef public int label
	cdef public unsigned long vec
	cdef int _hash
	def __init__(self, label, vec):
		self.label = label
		self.vec = vec
		self._hash = hash((self.label, self.vec))
	def __hash__(ChartItem self):
		return self._hash
	def __richcmp__(ChartItem self, ChartItem other, int op):
		if op == 0: return self.label < other.label or self.vec < other.vec
		elif op == 1: return self.label <= other.label or self.vec <= other.vec
		elif op == 2: return self.label == other.label and self.vec == other.vec
		elif op == 3: return self.label != other.label or self.vec != other.vec
		elif op == 4: return self.label > other.label or self.vec > other.vec
		elif op == 5: return self.label >= other.label or self.vec >= other.vec
	def __getitem__(ChartItem self, int n):
		if n == 0: return self.label
		elif n == 1: return self.vec
	def __repr__(ChartItem self):
		#would need bitlen for proper padding
		return "%s[%s]" % (self.label, bin(self.vec)[2:][::-1]) 

def parse(sent, grammar, tags=None, start=None, bint viterbi=False, int n=1, estimate=lambda a, b: 0.0):
	""" parse sentence, a list of tokens, using grammar, a dictionary
	mapping rules to probabilities. """
	cdef dict unary = <dict>grammar.unary
	cdef dict lbinary = <dict>grammar.lbinary
	cdef dict rbinary = <dict>grammar.rbinary
	cdef dict lexical = <dict>grammar.lexical
	cdef dict toid = <dict>grammar.toid
	cdef dict tolabel = <dict>grammar.tolabel
	cdef ChartItem Ih, I1h, goal
	if start == None: start = toid["ROOT"]
	goal = ChartItem(start, (1 << len(sent)) - 1)
	cdef int m = 0, maxA = 0
	A = heapdict() if viterbi else {}
	cdef dict C = <dict>defaultdict(deque)
	cdef dict Cx = <dict>defaultdict(dict)
	#C = JudyLObjObj()
	#from guppy import hpy; h = hpy(); hn = 0
	#h.heap().stat.dump("/tmp/hstat%d" % hn); hn+=1

	# scan
	Epsilon = toid["Epsilon"]
	for i,w in enumerate(sent):
		recognized = False
		for (rule,yf), z in lexical[w]:
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction, 
			# and give probability of 1
			if not tags or tags[i] == tolabel[rule[0]].split("@")[0]:
				Ih = ChartItem(rule[0], 1 << i)
				I = (ChartItem(Epsilon, i),)
				z = 0 if tags else z
				A[Ih] = (z, z, z, I)
				recognized = True
		if not recognized and tags and tags[i] in toid:
				Ih = ChartItem(toid[tags[i]], 1 << i)
				I = (ChartItem(Epsilon, i),)
				A[Ih] = (0, 0, 0, I)
				recognized = True
				continue
		elif not recognized:
			print "not covered:", tags[i] if tags else w
			return {}, ()
	cdef int lensent = len(sent)
	cdef double y, p, iscore, oscore
	cdef tuple scores, rhs
	# parsing
	while A:
		Ih, xI = A.popitem()
		#when heapdict is not available:
		#Ih, (x, I) = min(A.items(), key=lambda x:x[1]); del A[Ih]
		oscore, iscore, p, rhs = xI
		C[Ih].append((rhs, -p))
		Cx[Ih.label][Ih] = iscore
		if Ih == goal:
			m += 1	#problem: this is not viterbi n-best.
			#goal = Ih
			if viterbi and n==m: break
		else:
			for I1h, scores in deduced_from(Ih, iscore, Cx, unary, lbinary, rbinary, lensent, estimate):
				# I1h = new ChartItem that has been derived.
				# scores: oscore, iscore, p, rhs
				# oscore = estimate of total score (outside estimate + inside score up till now)
				# iscore = inside score, p = rule probability, rhs = backpointers to 1 or 2 ChartItems 
				# that led to this item
				# explicit get to avoid inserting spurious keys
				if I1h not in Cx.get(I1h.label, {}) and I1h not in A:
					A[I1h] = scores
				elif I1h in A and scores[0] < A[I1h][0]:
					A[I1h] = scores
				else: #if not viterbi:
					oscore, iscore, p, rhs = scores
					C[I1h].appendleft((rhs, -p))
		maxA = max(maxA, len(A))
		#pass #h.heap().stat.dump("/tmp/hstat%d" % hn); hn+=1
		##print h.iso(A,C,Cx).referents | h.iso(A, C, Cx)
	print "max agenda size", maxA, "/ chart keys", len(C), "/ values", sum(map(len, C.values())),
	#h.pb(*("/tmp/hstat%d" % a for a in range(hn)))
	#pprint_chart(C, sent, tolabel)
	return (C, goal) if goal in C else ({}, ())

cdef inline list deduced_from(ChartItem Ih, double x, dict Cx, dict unary, dict lbinary, dict rbinary, int bitlen, estimate):
	cdef double z, y
	cdef int I = Ih.label
	cdef unsigned long Ir = Ih.vec
	cdef ChartItem I1h
	cdef list result = []
	cdef tuple rule, yf
	for (rule, yf), z in unary[I]:
		result.append((ChartItem(rule[0], Ir), (estimate(rule[0], Ir)+x+z, x+z, z, (Ih,))))
	for (rule, yf), z in lbinary[I]:
		for I1h, y in Cx[rule[2]].items():
			if concat(yf, Ir, I1h.vec, bitlen):
				result.append((ChartItem(rule[0], Ir ^ I1h.vec), (estimate(rule[0], Ir ^ I1h.vec)+x+y+z, x+y+z, z, (Ih, I1h))))
	for (rule, yf), z in rbinary[I]:
		for I1h, y in Cx[rule[1]].items():
			if concat(yf, I1h.vec, Ir, bitlen):
				result.append((ChartItem(rule[0], I1h.vec ^ Ir), (estimate(rule[0], I1h.vec ^ Ir)+x+y+z, x+y+z, z, (I1h, Ih))))
	return result

cdef inline bint concat(tuple yieldfunction, unsigned long lvec, unsigned long rvec, int bitlen):
	if lvec & rvec: return False
	if len(yieldfunction) == 1 and len(yieldfunction[0]) == 2:
		if yieldfunction[0][0] == 0 and yieldfunction[0][1] == 1:
			return bitminmax(lvec, rvec)
		elif yieldfunction[0][0] == 1 and yieldfunction[0][1] == 0:
			return bitminmax(rvec, lvec)
		else: raise ValueError("non-binary element in yieldfunction")
	#this algorithm taken from rparse, FastYFComposer.
	cdef int lpos = nextset(lvec, 0)
	cdef int rpos = nextset(rvec, 0)
	cdef int n, m, b
	cdef tuple arg
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
cdef inline int nextset1(unsigned long a, int pos):
	cdef int result = pos
	while (not (a >> result) & 1) and a >> result:
		result += 1
	return result if a >> result else -1

cdef inline int nextunset1(unsigned long a, int pos):
	cdef int result = pos
	while (a >> result) & 1:
		result += 1
	return result

cdef inline bint testbit(unsigned long a, int offset):
	return a & (1 << offset)

def bitcount(unsigned long a):
	cdef int count = 0
	while a:
		a &= a - 1
		count += 1
	return count

cdef inline bint bitminmax1(unsigned long a, unsigned long b):
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

def filterchart2(chart, start, visited):
	chart[start] = [(a,b) for a,b in chart[start] if not visited & set(a)]
	for a,p in chart[start]:
		for b in a: 
			filterchart2(chart, b, visited | set(a))

def samplechart(chart, ChartItem start, dict tolabel, set visited):
	visited.add(start)
	eligible = range(len(chart[start]))
	while eligible:
		# pick a random index, pop it and look up the corresponding entry
		entry, p = chart[start][eligible.pop(randrange(len(eligible)))]
		if entry[0] not in visited: break
	else: return #no way out
	if entry[0].label == 0: # == "Epsilon":
		return "(%s %d)" % (tolabel[start.label], entry[0].vec), p
	children = [samplechart(chart, a, tolabel, visited if len(entry)==1 else set()) for a in entry]
	if None in children: return
	tree = "(%s %s)" % (tolabel[start.label], " ".join([a for a,b in children]))
	return tree, p+sum(b for a,b in children)

def mostprobableparse(chart, start, tolabel, n=100, sample=False, both=False):
		""" sum over n random derivations from chart,
			return a FreqDist of parse trees, with .max() being the MPP"""
		print "sample =", sample,
		if both:
			derivations = set(samplechart(chart, start, tolabel, set()) for x in range(n))
			derivations.discard(None)
			derivations.update(islice(enumchart(chart, start, tolabel), n))
		elif sample:
			for a,b in chart.items():
				if not len(b): print "spurious chart entry", a
			#filterchart2(chart, start, set([]))
			derivations = set(samplechart(chart, start, tolabel, set()) for x in range(n))
			derivations.discard(None)
			#calculate real parse probabilities according to Goodman's claimed method?
		else:
			#chart = filterchart(chart, start)
			#for a in chart: chart[a].sort(key=lambda x: x[1], reverse=True)
			derivations = islice(enumchart(chart, start, tolabel), n)
			#assert(len(list(islice(enumchart(chart, start), n))) == len(set((a.freeze(),b) for a,b in islice(enumchart(chart, start), n))))
		parsetrees = defaultdict(float)
		cdef double prob, prevprob
		cdef int m = 0
		for a,prob in derivations:
			m += 1
			parsetrees[re.sub("@[0-9]+","",a)] += exp(prob)
			#tree = re.sub("@[0-9]+","",a)
			#prevprob = parsetrees[tree]
			#if prob > prevprob:
			#	prevprob, prob = prob, prevprob
			# http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point-numbers
			#parsetrees[tree] = prevprob + log(1.0 + exp(prob - prevprob))
		print "(%d derivations)" % m
		return parsetrees

def pprint_chart(chart, sent, tolabel):
	print "chart:"
	for a in sorted(chart, key=lambda x: bitcount(x[1])):
		if len(chart[a][0][0]) != 1: continue
		print "%s[%s] =>" % (tolabel[a.label], ("0" * len(sent) + bin(a.vec)[2:])[::-1][:len(sent)])
		for b,p in chart[a]:
			for c in b:
				if tolabel[c[0]] == "Epsilon":
					print "\t", repr(sent[b[0][1]]),
				else:
					print "\t%s[%s]" % (tolabel[c.label], ("0" * len(sent) + bin(c.vec)[2:])[::-1][:len(sent)]),
			print "\t",e**p
		print

def do(sent, grammar):
	print "sentence", sent
	chart, start = parse(sent.split(), grammar, start=grammar.toid['S'])
	pprint_chart(chart, sent.split(), grammar.tolabel)
	if chart:
		for a, p in mostprobableparse(chart, start, grammar.tolabel, n=1000, sample=False).items():
			print p, a
		for a, p in mostprobableparse(chart, start, grammar.tolabel, n=1000, sample=True).items():
			print p, a
	else: print "no parse"
	print

def main():
	from rcgrules import splitgrammar
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
	do("muss Daruber nachgedacht werden", grammar)	#no parse

if __name__ == '__main__': main()
