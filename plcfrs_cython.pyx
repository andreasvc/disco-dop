# probabilistic CKY parser for Simple Range Concatenation Grammars
# (equivalent to Linear Context-Free Rewriting Systems)
from rcgrules import enumchart, induce_srcg
from kbest import lazykbest
from nltk import FreqDist, Tree
from heapdict cimport heapdict
from math import log, exp, fsum, isinf
from random import choice, randrange
from operator import itemgetter
from itertools import islice, count
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
		if op == 2: return self.label == other.label and self.vec == other.vec
		elif op == 3: return self.label != other.label or self.vec != other.vec
		elif op == 5: return self.label >= other.label or self.vec >= other.vec
		elif op == 1: return self.label <= other.label or self.vec <= other.vec
		elif op == 0: return self.label < other.label or self.vec < other.vec
		elif op == 4: return self.label > other.label or self.vec > other.vec
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
				# 	oscore = estimate of total score
				#			(outside estimate + inside score up till now)
				# 	iscore = inside score,
				# 	p = rule probability,
				# 	rhs = backpointers to 1 or 2 ChartItems that led here (I1h)
				# explicit get to avoid inserting spurious keys
				if I1h not in Cx.get(I1h.label, {}) and I1h not in A:
					# haven't seen this item before, add to agenda
					A[I1h] = scores
				elif I1h in A:
					if scores[0] < A[I1h][0]:
						# item has lower score, update agenda
						C[I1h].append(A[I1h][1:])
						A[I1h] = scores
					else:
						C[I1h].append(scores[1:])
				else: #if not viterbi:
					C[I1h].append(scores[1:])
					if iscore < Cx[I1h.label][I1h]: Cx[I1h.label][I1h] = iscore
		maxA = max(maxA, len(A))
	print "max agenda size", maxA, "/ chart keys", len(C), "/ values", sum(map(len, C.values())),
	return (C, goal) if goal in C else ({}, ())

cdef inline list deduced_from(ChartItem Ih, double x, dict Cx, list unary, list lbinary, list rbinary, estimate):
	""" Produce a list of all ChartItems that can be deduced using the
	existing items in Cx and the grammar rules. """
	cdef double z, y
	cdef int I = Ih.label, left, sibling
	cdef unsigned long Ir = Ih.vec
	cdef ChartItem I1h
	cdef list result = []
	cdef tuple rule, yf
	for (rule, yf), z in <list>unary[I]:
		result.append((ChartItem(rule[0], Ir), ((estimate(rule[0], Ir)
						if estimate else 0.0) + x + z, x + z, z, (Ih,))))
	for (rule, yf), z in <list>lbinary[I]:
		left = rule[0]
		sibling = rule[2]
		for I1h in Cx[rule[2]]:
			if concat(yf, Ir, I1h.vec):
				y = Cx[rule[2]][I1h]
				result.append((ChartItem(left, Ir ^ I1h.vec),
							((estimate(left, Ir ^ I1h.vec) if estimate
							else 0.0) + x + y + z, x + y + z, z, (Ih, I1h))))
	for (rule, yf), z in <list>rbinary[I]:
		left = rule[0]
		sibling = rule[1]
		for I1h in Cx[sibling]:
			if concat(yf, I1h.vec, Ir):
				y = Cx[sibling][I1h]
				result.append((ChartItem(left, I1h.vec ^ Ir),
							((estimate(left, I1h.vec ^ Ir) if estimate
							else 0.0) + x + y + z, x + y + z, z, (I1h, Ih))))
	return result

cdef inline bint concat(tuple yieldfunction, unsigned long lvec, unsigned long rvec):
	"""
	Determine the compatibility of two bitvectors (tuples of spans / ranges)
	according to the given yield function. Ranges should be non-overlapping,
	continuous when they are concatenated, and adhere to the ordering in the
	yield function.
	The yield function is a tuple of tuples with indices indicating from which
	vector the next span should be taken from, with 0 meaning left and 1 right.
	Note that the least significant bit is the lowest index in the vectors,
	which is the opposite of their normal presentation: 0b100 is the last
	terminal in a three word sentence, 0b001 is the first. E.g.,

	>>> lvec = 0b0011; rvec = 0b1000; yieldfunction = ((0,), (1,))
	>>> concat(((0,), (1,)), lvec, rvec)
	True		#discontinuous, non-overlapping, linearly ordered.
	>>> concat(((0, 1),), lvec, rvec)
	False		#lvec and rvec are not contiguous
	>>> concat(((1,), (0,)), lvec, rvec)
	False		#rvec's span should come after lvec's span
	"""
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
	# remove all entries that do not contribute to a complete derivation headed
	# by "start"
	def filter_subtree(start, chart, chart2):
		if isinstance(start, int) or chart2[start]: return True
		else: chart2[start] = [(x,p) for x,p in chart[start] if all(filter_subtree(a, chart, chart2) for a in x)]
		return chart2[start] != []
	chart2 = defaultdict(list)
	filter_subtree(start, chart, chart2)
	return chart2

def filtercycles(chart, start, visited, current):
	""" remove @#$%! cycles from chart
	FIXME: not working correctly yet. Only zero cost cycles seem to be a
	problem, though. """
	visited.add(start)
	chart[start] = [(ip, p, rhs) for ip, p, rhs in chart[start]
		if current.isdisjoint(rhs)]
	for n, (ip, p, rhs) in zip(count(), chart[start])[::-1]:
		for a in rhs:
			if a in chart and a not in visited:
				filtercycles(chart, a, visited, current.union([start, a]))
			#if a not in chart:
			#	del chart[start][n]
			#	break
	if not len(chart[start]): del chart[start]
	#return not len(chart[start]):

cpdef list getviterbi(dict chart, ChartItem start, dict mem):
	""" recompute the proper viterbi probabilities in a top-down fashion,
		and sort chart entries according to these probabilities
		removes zero-probability edges (infinity with log-probabilities)
		also tracks items visited, for pruning purposes (keys of mem).

		FIXME: not working yet. it seems to be more efficient to do
		the viterbi thing during parsing -- the log n overhead of
		the priority queue is small
	"""
	cdef list probs, bestprops
	cdef double prob, bestprob
	cdef int n
	probs = []
	bestprob = 999999 #float('infinity')
	try: assert len(chart[start])
	except: print "empty", start
	# loop backwards because we want to remove items in-place without
	# invalidating remaining indices.
	for n, (ip, p, rhs) in zip(count(), chart[start])[::-1]:
		probs[:] = [p]
		for a in rhs:
			# only recurse for nonterminals (=nonzero ids)
			if a.label and a in chart:
				if a in mem: result = mem[a]
				else: result = getviterbi(chart, a, mem)
				if not isinstance(result, list):
					print "trouble", start, '->', a
				probs.extend(result)
		prob = fsum(probs)
		if prob < bestprob:
			bestprob = prob
			bestprobs = probs[:]
		# prune or update probability
		if isinf(prob): del chart[start][n]
		else: chart[start][n] = (prob, p, rhs)
	if len(chart[start]):
		chart[start].sort(key=itemgetter(0))
		assert fsum(bestprobs) == chart[start][0][0]
	else:
		bestprobs = [float('infinity')]
	mem[start] = bestprobs
	return bestprobs

cdef samplechart(dict chart, ChartItem start, dict tolabel):
	iscore, p, entry = choice(chart[start])
	if entry[0].label == 0: # == "Epsilon":
		return "(%s %d)" % (tolabel[start.label], entry[0].vec), p
	children = [samplechart(chart, a, tolabel) for a in entry]
	tree = "(%s %s)" % (tolabel[start.label],
							" ".join([a for a,b in children]))
	return tree, fsum([p] + [b for a,b in children])

removeids = re.compile("@[0-9]+")
def mostprobableparse(chart, start, tolabel, n=100, sample=False, both=False, shortest=False, secondarymodel=None):
	""" sum over n random/best derivations from chart,
		return a dictionary mapping parsetrees to probabilities """
	print "sample =", sample or both, "kbest =", (not sample) or both,
	if both:
		derivations = set(samplechart(<dict>chart, start, tolabel) for x in range(n*100))
		derivations.discard(None)
		derivations.update(lazykbest(<dict>chart, start, n, tolabel))
		#derivations.update(islice(enumchart(chart, start, tolabel, n), n))
	elif sample:
		#filtercycles(chart, start, set(), set())
		derivations = set(samplechart(<dict>chart, start, tolabel) for x in range(n))
		derivations.discard(None)
		#calculate real parse probabilities according to Goodman's claimed
		#method?
	else:
		#mem = {}
		#filtercycles(chart, start, set(), set())
		#getviterbi(<dict>chart, start, mem)
		#for a in set(chart.keys()) - set(mem.keys()): del chart[a]
		#print "pruned chart keys", len(chart), "/ values", sum(map(len, chart.values()))
		for a in chart: chart[a].sort(key=itemgetter(0))
		#derivations = list(islice(enumchart(chart, start, tolabel, n), n))
		#fixme: set shouldn't be necessary
		derivations = set(lazykbest(dict(chart), start, n, tolabel))
		#print len(derivations)
		#print "enumchart:", len(list(islice(enumchart(chart, start, tolabel), n)))
		#assert(len(list(islice(enumchart(chart, start), n))) == len(set((a.freeze(),b) for a,b in islice(enumchart(chart, start), n))))
	if shortest:
		derivations = [(a,b) for a, b in derivations if b == derivations[0][1]]
	#cdef dict parsetrees = <dict>defaultdict(float)
	cdef dict parsetrees = <dict>defaultdict(list)
	cdef double prob, maxprob
	cdef int m = 0
	#cdef str deriv, tree
	for deriv, prob in derivations:
		m += 1
		#parsetrees[removeids.sub("", deriv)] += exp(-prob)
		#restore linear precedence (disabled, seems to make no difference)
		#parsetree = Tree(removeids.sub("", deriv))
		#for a in list(parsetree.subtrees())[::-1]:
		#	a.sort(key=lambda x: x.leaves())
		#parsetrees[parsetree.pprint(margin=999)].append(-prob)
		if shortest:
			tree = Tree(removeids.sub("", deriv))
			sent = sorted(tree.leaves(), key=int)
			rules = induce_srcg([tree], [sent], arity_marks=False)
			prob = -fsum([secondarymodel[r] for r, w in rules
											if r[0][1] != 'Epsilon'])
		parsetrees[removeids.sub("", deriv)].append(-prob)
	# Adding probabilities in log space
	# http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point-numbers
	# https://facwiki.cs.byu.edu/nlp/index.php/Log_Domain_Computations
	for parsetree in parsetrees:
		maxprob = max(parsetrees[parsetree])
		#foo = sum(map(exp, parsetrees[parsetree]))
		parsetrees[parsetree] = exp(fsum([maxprob, log(fsum([exp(prob - maxprob)
										for prob in parsetrees[parsetree]]))]))
		#assert log(foo) == parsetrees[parsetree]
	print "(%d derivations, %d parsetrees)" % (m, len(parsetrees))
	return parsetrees

def binrepr(a, sent):
	return bin(a.vec)[2:].rjust(len(sent), "0")[::-1]

def pprint_chart(chart, sent, tolabel):
	print "chart:"
	for a in sorted(chart, key=lambda x: bitcount(x[1])):
		print "%s[%s] =>" % (tolabel[a.label], binrepr(a, sent))
		for ip,p,rhs in chart[a]:
			for c in rhs:
				if tolabel[c[0]] == "Epsilon":
					print "\t", repr(sent[rhs[0][1]]),
				else:
					print "\t%s[%s]" % (tolabel[c.label], binrepr(c, sent)),
			print "\t", exp(-p)
		print

def do(sent, grammar):
	print "sentence", sent
	chart, start = parse(sent.split(), grammar, start=grammar.toid['S'])
	if not chart:
		print "no parse\n"
		return
	mpp = mostprobableparse(chart, start, grammar.tolabel)
	pprint_chart(chart, sent.split(), grammar.tolabel)
	for a, p in mpp.items():
		print p, a, '\n'

def main():
	from rcgrules import splitgrammar
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
