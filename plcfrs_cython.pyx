# probabilistic CKY parser for Simple Range Concatenation Grammars
# (equivalent to Linear Context-Free Rewriting Systems)
from rcgrules import enumchart
from kbest import lazykbest
from nltk import FreqDist, Tree
from heapdict import heapdict
from math import log, exp
from random import choice, randrange
from itertools import chain, islice
from collections import defaultdict, deque
import re

cdef extern from "bit.h":
	int nextset(unsigned long vec, int pos)
	int nextunset(unsigned long vec, int pos)
	int bitcount(unsigned long vec)
	bint testbit(unsigned long vec, unsigned long pos)
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

def parse(sent, grammar, tags=None, start=None, bint viterbi=False, int n=1, estimate=None):
	""" parse sentence, a list of tokens, using grammar, a dictionary
	mapping rules to probabilities. """
	cdef list unary = grammar.unary
	cdef list lbinary = grammar.lbinary
	cdef list rbinary = grammar.rbinary
	cdef dict lexical = <dict>grammar.lexical
	cdef dict toid = <dict>grammar.toid
	cdef dict tolabel = <dict>grammar.tolabel
	cdef ChartItem Ih, I1h, goal
	if start == None: start = toid["ROOT"]
	goal = ChartItem(start, (1 << len(sent)) - 1)
	cdef int m = 0, maxA = 0
	A = heapdict() if viterbi else {}
	cdef dict C = <dict>defaultdict(list)
	cdef dict Cx = <dict>defaultdict(dict)

	# scan
	Epsilon = toid["Epsilon"]
	for i,w in enumerate(sent):
		recognized = False
		for (rule,yf), z in lexical.get(w, []):
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
		Ih, (oscore, iscore, p, rhs) = A.popitem()
		#when heapdict is not available:
		#Ih, (x, I) = min(A.items(), key=lambda x:x[1]); del A[Ih]
		C[Ih].append((iscore, p, rhs))
		Cx[Ih.label][Ih] = iscore
		if Ih == goal:
			m += 1
			if viterbi and n==m: break
		else:
			for I1h, scores in deduced_from(Ih, iscore, Cx, unary, lbinary, rbinary, estimate):
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
					C[I1h].append((iscore, p, rhs))
		maxA = max(maxA, len(A))
	print "max agenda size", maxA, "/ chart keys", len(C), "/ values", sum(map(len, C.values())),
	return (C, goal) if goal in C else ({}, ())

cdef inline list deduced_from(ChartItem Ih, double x, dict Cx, list unary, list lbinary, list rbinary, estimate):
	cdef double z, y
	cdef int I = Ih.label
	cdef unsigned long Ir = Ih.vec
	cdef ChartItem I1h
	cdef list result = []
	cdef tuple rule, yf
	for (rule, yf), z in <list>unary[I]:
		result.append((ChartItem(rule[0], Ir), (estimate(rule[0], Ir) if estimate else 0.0 +x+z, x+z, z, (Ih,))))
	for (rule, yf), z in <list>lbinary[I]:
		for I1h, y in Cx[rule[2]].items():
			if concat(yf, Ir, I1h.vec):
				result.append((ChartItem(rule[0], Ir ^ I1h.vec), (estimate(rule[0], Ir ^ I1h.vec) if estimate else 0.0 +x+y+z, x+y+z, z, (Ih, I1h))))
	for (rule, yf), z in <list>rbinary[I]:
		for I1h, y in Cx[rule[1]].items():
			if concat(yf, I1h.vec, Ir):
				result.append((ChartItem(rule[0], I1h.vec ^ Ir), (estimate(rule[0], I1h.vec ^ Ir) if estimate else 0.0 +x+y+z, x+y+z, z, (I1h, Ih))))
	return result

cdef inline bint concat(tuple yieldfunction, unsigned long lvec, unsigned long rvec):
	if lvec & rvec: return False
	cdef int lpos = nextset(lvec, 0)
	cdef int rpos = nextset(rvec, 0)
	# if there are no gaps in lvec and rvec, and the yieldfunction is the
	# concatenation of two elements, then this should be quicker
	if (lvec >> nextunset(lvec, lpos) == 0 and rvec >> nextunset(rvec, rpos) == 0):
		if yieldfunction == ((0, 1),):
			return bitminmax(lvec, rvec)
		elif yieldfunction == ((1, 0),):
			return bitminmax(rvec, lvec)
	#this algorithm taken from rparse, FastYFComposer.
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

cdef samplechart(dict chart, ChartItem start, dict tolabel): #set visited
	#visited.add(start)
	#eligible = range(len(chart[start]))
	#while eligible:
	#	# pick a random index, pop it and look up the corresponding entry
	#	entry, p = chart[start][eligible.pop(randrange(len(eligible)))]
	#	if entry[0] not in visited: break
	#else: return #no way out
	iscore, p, entry = choice(chart[start])
	if entry[0].label == 0: # == "Epsilon":
		return "(%s %d)" % (tolabel[start.label], entry[0].vec), p
	#children = [samplechart(chart, a, tolabel, visited if len(entry)==1 else set()) for a in entry]
	children = [samplechart(chart, a, tolabel) for a in entry]
	#if None in children: return
	tree = "(%s %s)" % (tolabel[start.label], " ".join([a for a,b in children]))
	return tree, p+sum(b for a,b in children)

def mostprobableparse(chart, start, tolabel, n=100, sample=False, both=False):
		""" sum over n random derivations from chart,
			return a FreqDist of parse trees, with .max() being the MPP"""
		print "sample =", sample or both, "kbest =", (not sample) or both,
		if both:
			derivations = set(samplechart(<dict>chart, start, tolabel) for x in range(n*100))
			derivations.discard(None)
			derivations.update(lazykbest(chart, start, n, tolabel))
			#derivations.update(islice(enumchart(chart, start, tolabel, n), n))
		elif sample:
			for a,b in chart.items():
				if not len(b): print "spurious chart entry", a
			#filterchart2(chart, start, set([]))
			derivations = set(samplechart(<dict>chart, start, tolabel) for x in range(n))
			derivations.discard(None)
			#calculate real parse probabilities according to Goodman's claimed method?
		else:
			#chart = filterchart(chart, start)
			#derivations = list(islice(enumchart(chart, start, tolabel, n), n))
			derivations = lazykbest(chart, start, n, tolabel)
			#print len(derivations)
			#print "enumchart:", len(list(islice(enumchart(chart, start, tolabel), n)))
			#assert(len(list(islice(enumchart(chart, start), n))) == len(set((a.freeze(),b) for a,b in islice(enumchart(chart, start), n))))
		#cdef dict parsetrees = <dict>defaultdict(float)
		cdef dict parsetrees = <dict>defaultdict(list)
		cdef double prob, maxprob
		cdef int m = 0
		#cdef str deriv, tree
		removeids = re.compile("@[0-9]+")
		parens = re.compile("\$\(")
		for deriv, prob in derivations:
			m += 1
			#parsetrees[removeids.sub("", deriv)] += exp(-prob)
			parsetrees[removeids.sub("", deriv)].append(-prob)
			#restore linear precedence (disabled, seems to make no difference)
			#parsetree = Tree(parens.sub("$[", removeids.sub("", deriv)))
			#for a in parsetree.subtrees():
			#	if len(a) > 1: a.sort(key=lambda x: x.leaves())
			#	elif a.node == "$[": a.node = "$("
			#parsetrees[parsetree.pprint(margin=999)].append(-prob)
		# Adding probabilities in log space
		# http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point-numbers
		for parsetree in parsetrees:
			maxprob = max(parsetrees[parsetree])
			parsetrees[parsetree] = exp(maxprob + log(sum(exp(prob - maxprob) for prob in parsetrees[parsetree])))
		print "(%d derivations, %d parsetrees)" % (m, len(parsetrees))
		return parsetrees

def pprint_chart(chart, sent, tolabel):
	print "chart:"
	for a in sorted(chart, key=lambda x: bitcount(x[1])):
		#if len(chart[a][0][0]) != 1: continue #only print unary for debugging
		print "%s[%s] =>" % (tolabel[a.label], bin(a.vec)[2:].rjust(len(sent), "0")[::-1])
		for ip,p,b in chart[a]:
			for c in b:
				if tolabel[c[0]] == "Epsilon":
					print "\t", repr(sent[b[0][1]]),
				else:
					print "\t%s[%s]" % (tolabel[c.label], bin(c.vec)[2:].rjust(len(sent), "0")[::-1]),
			print "\t", exp(-p)
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
