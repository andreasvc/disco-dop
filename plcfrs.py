# probabilistic CKY parser for Simple Range Concatenation Grammars
# (equivalent to Linear Context-Free Rewriting Systems)
from math import log, exp
from random import choice
from itertools import count, groupby
from operator import itemgetter
from collections import defaultdict
import re
from heapdict import heapdict
from kbest import lazykbest
from containers import ChartItem
try:
	import cython
	assert cython.compiled
except:
	print "plcfrs in non-cython mode"
	from bit import *

def parse(sent, grammar, tags=None, start=None, viterbi=False, n=1, estimate=None):
	""" parse sentence, a list of tokens, optionally with gold tags, and
	produce a chart, either exhaustive or up until the viterbi parse
	"""
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
	A = heapdict()

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
	# parsing
	while A:
		Ih, scores = A.popitem()
		iscore, p, rhs = scores
		C.setdefault(Ih, []).append(scores)
		Cx.setdefault(Ih.label, {})[Ih] = iscore

		if Ih == goal:
			m += 1
			if viterbi and n==m: break
		else:
			for I1h, scores in deduced_from(Ih, iscore, Cx, unary, lbinary, rbinary):
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
					# either item has lower score, update agenda,
					# or extend chart
					if scores[0] < A[I1h][0]:
						C.setdefault(I1h, []).append(A[I1h])
						A[I1h] = scores
					else:
						C.setdefault(I1h, []).append(scores)
				else:
					C[I1h].append(scores)
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
				result.append((ChartItem(rule[0], Ir ^ I1h.vec),
								(x + y + z, z, (Ih, I1h))))
	for (rule, yf), z in rbinary[I]:
		for I1h, y in Cx.get(rule[1], {}).items():
			if concat(yf, I1h.vec, Ir):
				result.append((ChartItem(rule[0], I1h.vec ^ Ir),
								(x + y + z, z, (I1h, Ih))))
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

def filter_subtree(start, chart, newchart):
	if isinstance(start, int) or newchart[start]:
		return True
	else:
		temp = [(ip, p, rhs) for ip, p, rhs in chart[start]
				if all(filter_subtree(a, chart, chart2) for a in rhs)]
		if temp: newchart[start] = temp
	return start in newchart

def filterchart(chart, start):
	# remove all entries that do not contribute to a complete derivation headed
	# by "start"
	newchart = {}
	filter_subtree(start, chart, newchart)
	return newchart

def filtercycles(chart, start, visited, current):
	""" remove @#$%! cycles from chart """
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

def filterduplicates(chart):
	for a in chart:
		chart[a] = [min(g, key=itemgetter(0)) for k, g
			in groupby(sorted(chart[a], key=lambda x: hash(x[2])),
				itemgetter(2))]

def getviterbi(chart, start, mem):
	""" recompute the proper viterbi probabilities in a top-down fashion,
		and sort chart entries according to these probabilities
		removes zero-probability edges (infinity with log-probabilities)
		also tracks items visited, for pruning purposes (keys of mem).
	"""
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
		print "%s[%s] =>" % (tolabel[a[0]],
					(bin(a[1])[:1:-1] + "0" * len(sent))[:len(sent)])
		for ip,p,b in chart[a]:
			for c in b:
				if tolabel[c[0]] == "Epsilon":
					print "\t", repr(sent[b[0][1]]),
				else:
					print "\t%s[%s]" % (tolabel[c[0]],
						(bin(c[1])[:1:-1] + "0" * len(sent))[:len(sent)]),
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
	from grammar import splitgrammar
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
	print repr(grammar)
	do("Daruber muss nachgedacht werden", grammar)
	do("Daruber muss nachgedacht werden werden", grammar)
	do("Daruber muss nachgedacht werden werden werden", grammar)
	do("muss Daruber nachgedacht werden", grammar)	#no parse

if __name__ == '__main__': main()
