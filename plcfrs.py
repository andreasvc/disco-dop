# probabilistic CKY parser for Simple Range Concatenation Grammars
# (equivalent to Linear Context-Free Rewriting Systems)
try:
	import cython
	assert cython.compiled
except:
	print "plcfrs in non-cython mode"
	from bit import *

class ChartItem:
	__slots__ = ("label", "vec", "_hash")
	def __init__(self, label, vec):
		self.label = label
		self.vec = vec
		self._hash = hash((self.label, self.vec))
	def __hash__(self):
		return self._hash
	def __richcmp__(self, other, op):
		if op == 0: return self.label < other.label or self.vec < other.vec
		elif op == 1: return self.label <= other.label or self.vec <= other.vec
		elif op == 2: return self.label == other.label and self.vec == other.vec
		elif op == 3: return self.label != other.label or self.vec != other.vec
		elif op == 4: return self.label > other.label or self.vec > other.vec
		elif op == 5: return self.label >= other.label or self.vec >= other.vec
	def __cmp__(self, other):
		if self.label == other.label and self.vec == other.vec: return 0
		elif self.label < other.label or (self.label == other.label and self.vec < other.vec): return -1
		return 1
	def __getitem__(self, n):
		if n == 0: return self.label
		elif n == 1: return self.vec
	def __repr__(self):
		#would need bitlen for proper padding
		return "%s[%s]" % (self.label, bin(self.vec)[2:][::-1])

from rcgrules import enumchart
from kbest import lazykbest
from nltk import FreqDist
from heapdict import heapdict
from math import log, exp
from random import choice
from itertools import chain, islice
from collections import defaultdict
import re

def parse(sent, grammar, tags=None, start=None, viterbi=False, n=1, estimate=None):
	""" parse sentence, a list of tokens, using grammar, a dictionary
	mapping rules to probabilities. """
	unary = grammar.unary
	lbinary = grammar.lbinary
	rbinary = grammar.rbinary
	lexical = dict(grammar.lexical)
	toid = dict(grammar.toid)
	tolabel = dict(grammar.tolabel)
	if start is None: start = toid['S']
	goal = ChartItem(start, (1 << len(sent)) - 1)
	m = maxA = 0
	C, Cx = {}, {}
	A = heapdict() if viterbi else {}

	# scan
	Epsilon = toid["Epsilon"]
	for i,w in enumerate(sent):
		recognized = False
		for (rule,yf), z in lexical.get(w, []):
			if not tags or tags[i] == tolabel[rule[0]].split("@")[0]:
				Ih = ChartItem(rule[0], 1 << i)
				I = ChartItem(Epsilon, i)
				# if gold tags were provided, give them probability of 1
				A[Ih] = (0.0 if tags else z, 0.0 if tags else z, (I,))
				recognized = True
		if not recognized and tags and tags[i] in toid:
			Ih = ChartItem(toid[tags[i]], 1 << i)
			I = ChartItem(Epsilon, i)
			A[Ih] = (0.0, 0.0, (I,))
			recognized = True
			continue
		elif not recognized:
			print "not covered:", tags[i] if tags else w
	lensent = len(sent)
	# parsing
	while A:
		Ih, xI = A.popitem()
		#when heapdict is not available:
		#Ih, (x, I) = min(A.items(), key=lambda x:x[1]); del A[Ih]
		#C[Ih] = I, x
		iscore, p, rhs = xI
		C.setdefault(Ih, []).append(xI)
		Cx.setdefault(Ih.label, {})[Ih] = iscore
		if Ih == goal:
			m += 1
			if viterbi and n==m: break
		else:
			for I1h, scores in deduced_from(Ih, iscore, Cx, unary, lbinary, rbinary):
				if I1h not in Cx.get(I1h.label, {}) and I1h not in A:
					A[I1h] = scores
				elif I1h in A and scores[0] < A[I1h][0]:
					A[I1h] = scores
				else:
					rhs = scores[2]
					C.setdefault(I1h, []).append(scores)
		maxA = max(maxA, len(A))
	print "max agenda size", maxA, "/ chart keys", len(C), "/ values", sum(map(len, C.values()))
	return (C, goal) if goal in C else ({}, ())

def deduced_from(Ih, x, Cx, unary, lbinary, rbinary):
	I, Ir = Ih.label, Ih.vec
	result = []
	for (rule, yf), z in unary[I]:
		result.append((ChartItem(rule[0], Ir), (x+z, z, (Ih,))))
	for (rule, yf), z in lbinary[I]:
		for I1h, y in Cx.get(rule[2], {}).items():
			if concat(yf, Ir, I1h.vec):
				result.append((ChartItem(rule[0], Ir ^ I1h.vec), (x+y+z, z, (Ih, I1h))))
	for (rule, yf), z in rbinary[I]:
		for I1h, y in Cx.get(rule[1], {}).items():
			if concat(yf, I1h.vec, Ir):
				result.append((ChartItem(rule[0], I1h.vec ^ Ir), (x+y+z, z, (I1h, Ih))))
	return result

def concat(yieldfunction, lvec, rvec):
	if lvec & rvec: return False
	lpos = nextset(lvec, 0)
	rpos = nextset(rvec, 0)
	# if there are no gaps in lvec and rvec, and the yieldfunction is the
	# concatenation of two elements, then this should be quicker
	if (lvec >> nextunset(lvec, lpos) == 0 and rvec >> nextunset(rvec, rpos) == 0):
		if yieldfunction == ((0, 1),):
			return bitminmax(lvec, rvec)
		elif yieldfunction == ((1, 0),):
			return bitminmax(rvec, lvec)
	#this algorithm taken from rparse FastYFComposer.
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

def filterchart(chart, start):
	# remove all entries that do not contribute to a complete derivation
	def filter_subtree(start, chart, chart2):
		if isinstance(start, int) or chart2[start]: return True
		else: chart2[start] = [(x,p) for x,p in chart[start] if all(filter_subtree(a, chart, chart2) for a in x)]
		return chart2[start] != []
	chart2 = defaultdict(list)
	filter_subtree(start, chart, chart2)
	return chart2

def samplechart(chart, start, tolabel):
	iscore, p, entry = choice(chart[start])
	if len(entry) == 1 and entry[0][0] == 0: # Epsilon
		return "(%s %d)" % (tolabel[start.label], entry[0][1]), p
	children = [samplechart(chart, a, tolabel) for a in entry]
	tree = "(%s %s)" % (tolabel[start.label], " ".join([a for a,b in children]))
	return tree, p+sum(b for a,b in children)
	
removeids = re.compile("@[0-9]+")
def mostprobableparse(chart, start, tolabel, n=100, sample=False, both=False):
	""" sum over n random derivations from chart,
		return a FreqDist of parse trees, with .max() being the MPP"""
	print "sample =", sample or both, "kbest =", (not sample) or both,
	if both:
		derivations = set(samplechart(chart, start, tolabel) for x in range(n*100))
		derivations.discard(None)
		derivations.update(lazykbest(chart, start, n, tolabel))
	elif sample:
		for a,b in chart.items():
			if not len(b): print "spurious chart entry", a
		derivations = set(samplechart(chart, start, tolabel) for x in range(n))
		derivations.discard(None)
		#calculate real parse probabilities according to Goodman's claimed method?
	else:
		#derivations = islice(enumchart(chart, start, tolabel), n)
		derivations = lazykbest(chart, start, n, tolabel)
	m = 0
	parsetrees = defaultdict(list)
	for deriv, prob in derivations:
		m += 1
		parsetrees[removeids.sub("", deriv)].append(-prob)
	for parsetree in parsetrees:
		maxprob = max(parsetrees[parsetree])
		#foo = sum(map(exp, parsetrees[parsetree]))
		parsetrees[parsetree] = exp(maxprob + log(sum(exp(prob - maxprob) for prob in parsetrees[parsetree])))
		#assert foo == parsetrees[parsetree]
	print "(%d derivations, %d parsetrees)" % (m, len(parsetrees))
	return parsetrees

def pprint_chart(chart, sent, tolabel):
	print "chart:"
	for a in sorted(chart, key=lambda x: bitcount(x[1])):
		print "%s[%s] =>" % (tolabel[a[0]], ("0" * len(sent) + bin(a[1])[2:])[::-1][:len(sent)])
		for ip,p,b in chart[a]:
			for c in b:
				if tolabel[c[0]] == "Epsilon":
					print "\t", repr(sent[b[0][1]]),
				else:
					print "\t%s[%s]" % (tolabel[c[0]], ("0" * len(sent) + bin(c[1])[2:])[::-1][:len(sent)]),
			print "\t",exp(-p)
		print

def do(sent, grammar):
	print "sentence", sent
	chart, start = parse(sent.split(), grammar)
	pprint_chart(chart, sent.split(), grammar.tolabel)
	if chart:
		for a, p in mostprobableparse(chart, start, grammar.tolabel, n=1000).items():
			print p, a
	else: print "no parse"
	print

def main():
	from rcgrules import splitgrammar
	try: print "compiled", cython.compiled
	except: print "compiled", False
	grammar = splitgrammar(
		[((('S','VP2','VMFIN'),    ((0,1,0),)),  0),
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
