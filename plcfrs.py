# probabilistic CKY parser for Simple Range Concatenation Grammars
# (equivalent to Linear Context-Free Rewriting Systems)
from rcgrules import fs, enumchart
from nltk import FreqDist 
from heapdict import heapdict
from math import log, e, floor
from random import choice
from itertools import chain, product, islice
from pprint import pprint
from collections import defaultdict
from operator import or_
import re
try:
	import pyximport
	from bit import *
except: pass
try:
	import psyco
	psyco.full()
except: pass

def parse(sent, grammar, start="S", viterbi=False, n=1):
	""" parse sentence, a list of tokens, using grammar, a dictionary
	mapping rules to probabilities. """
	unary, binary = defaultdict(list), defaultdict(list)
	for r,w in grammar:
		if len(r) == 2: unary[r[1][0]].append((r, -w))
		elif len(r) == 3: binary[(r[1][0], r[2][0])].append((r, -w))
	goal = freeze([start, 2**len(sent) - 1])
	epsilon = "Epsilon"
	m = 0
	A, C, Cx = {}, defaultdict(list), defaultdict(list)
	for i,w in enumerate(sent):
		for rule, z in unary[epsilon]:
			if w in rule[0][1][0]:
				A[freeze([[rule[0][0], 2**i], i])] = z
	while A:	
		(I, x) = A.popitem()
		if I[0] not in C or I not in C[I[0]]:
			C[I[0]].append(I)
			Cx[I[0]].append(x)
		if I[0] == goal:
			m += 1
			if viterbi and n==m: break
		else:
			for I1, y in deduced_from(I[0], x, C.keys(), unary, binary):
				if (I1[0] not in C or I1 not in C[I1[0]]) and I1 not in A:
					A[I1] = y
				elif I1 in A:
					if y > A[I1]: A[I1] = y
	for a in C: C[a] = [(b[1:],-p) for b,p in zip(C[a], Cx[a])]
	if goal in C: return C, goal
	else: return {}, ()

def deduced_from(I, x, C, unary, binary):
	result = []
	for rule, z in unary[I[0]]:
		for a,b in zip(rule[1][1:], I[1:]): a.append(b)
		left = concat(rule[0][0], ([a.pop() for a in b] for b in rule[0][1:]))
		if left: result.append(((left, I), z))
	for I1 in C:
		#detect overlap in ranges
		if foldor(I1[1:]) & foldor(I[1:]): continue
		for rule, z in binary[(I[0], I1[0])]:
			for a,b in zip(rule[1][1:], I[1:]): a.append(b)
			for a,b in zip(rule[2][1:], I1[1:]): a.append(b)
			left = concat(rule[0][0], 
				([a.pop() for a in b] for b in rule[0][1:]))
			if left: result.append(((left, I, I1), z))
		for rule, z in binary[(I1[0], I[0])]:
			for a,b in zip(rule[1][1:], I1[1:]): a.append(b)
			for a,b in zip(rule[2][1:], I[1:]): a.append(b)
			right = concat(rule[0][0], 
				([a.pop() for a in b] for b in rule[0][1:]))
			if right: result.append(((right, I1, I), z))
	return result
		
def concat(lhs, node):
	# only concatenate when result will be contiguous
	result = []
	for x in node:
		tmp = x[0]
		for a,b in zip(x, x[1:]):
			if (bitmax(a) + 1) == bitmin(b): tmp |= b
			else: return
		result.append(tmp)
	return (lhs,) + tuple(result)
	#if all(((bitmax(a) + 1) == bitmin(b)) for x in node for a,b in zip(x, x[1:])):
	#	return (lhs,) + tuple(map(foldor, node))

def foldor(s):
	# unrolled version of reduce(or, s) for speed
	if len(s) == 1: return s[0]
	if len(s) == 2: return s[0] | s[1]
	if len(s) == 3: return s[0] | s[1] | s[2]
	if len(s) == 4: return s[0] | s[1] | s[2] | s[3]
	return reduce(or_, s)

# next two taken from http://wiki.python.org/moin/BitManipulation
def bitmax1(int_type):
	return floor(log(int_type, 2))
def bitmax(int_type):
	length = -1
	while (int_type):
		int_type >>= 1
		length += 1
	return(length) 

def bitmin(int_type):
	low = (int_type & -int_type)
	lowBit = -1
	while (low):
		low >>= 1
		lowBit += 1
	return(lowBit)

def freeze(l):
	if isinstance(l, list): return tuple(map(freeze, l))
	else: return l

def filterchart(chart, start):
	def filter_subtree(start, chart, chart2):
		if isinstance(start, int) or chart2[start]: return True
		else: chart2[start] = [(x,p) for x,p in chart[start] if all(filter_subtree(a, chart, chart2) for a in x)]
		return chart2[start] != []
	chart2 = defaultdict(list)
	filter_subtree(start, chart, chart2)
	return chart2

def samplechart(chart, start):
	if chart[start]: entry, p = choice(chart[start])
	else: return #shouldn't happen
	if len(entry) == 1 and isinstance(entry[0], int):
		return "(%s %d)" % (start[0], entry[0]), p
	children = [samplechart(chart, a) for a in entry]
	tree = "(%s %s)" % (start[0], " ".join([a for a,b in children])) 
	return tree, p+sum(b for a,b in children)
	
def mostprobableparse(chart, start, n=100, sample=False):
		""" sum over n random derivations from chart, 
			return a FreqDist of parse trees, with .max() being the MPP"""
		print "sample", sample,
		if sample:
			#chart = filterchart(chart, start)
			for a,b in chart.items():
				if not len(b): print "spurious chart entry", a
			derivations = set(samplechart(chart, start) for x in range(n))
			derivations.discard(None)
		else: 
			for a in chart: chart[a].sort(key=lambda (x,y): y, reverse=True)
			derivations = islice(enumchart(chart, start), n)
		parsetrees = FreqDist()
		m = 0
		for n,(a,prob) in enumerate(derivations):
			parsetrees.inc(re.sub(r"@[0-9]+", "", a), e**prob)
			m+=1
		print "(%d derivations)" % m
		return parsetrees

def do(sent):
	print "sentence", sent
	chart, start = parse(sent.split(), grammar)
	if chart:
		for a, p in mostprobableparse(chart, start, n=1000).items(): 
			print p, a
	else: print "no parse"
	print

if __name__ == '__main__':
	grammar =  [
		(fs("[['S',[?X,?Y,?Z]], ['VP2',?X,?Z], ['VMFIN',?Y]]"),  0),
		(fs("[['VP2',[?X],[?Y,?Z]],['VP2',?X,?Y], ['VAINF',?Z]]"),  log(0.5)),
		(fs("[['VP2',[?X],[?Y]], ['PROAV',?X], ['VVPP',?Y]]"),  log(0.5)),
		(fs("[['PROAV',['Daruber']], [Epsilon]]"),  0),
		(fs("[['VVPP',['nachgedacht']], [Epsilon]]"),  0),
		(fs("[['VMFIN',['muss']], [Epsilon]]"),  0),
		(fs("[['VAINF',['werden']], [Epsilon]]"), 0)
		]

	# a DOP reduction according to Goodman (1996)
	grammar =  [
		(fs("[['S',[?X,?Y,?Z]], ['VP2@3',?X,?Z], ['VMFIN@2',?Y]]"),  log(10/22.)),
		(fs("[['S',[?X,?Y,?Z]], ['VP2@3',?X,?Z], ['VMFIN',?Y]]"),  log(10/22.)),
		(fs("[['S',[?X,?Y,?Z]], ['VP2',?X,?Z], ['VMFIN@2',?Y]]"),  log(1/22.)),
		(fs("[['S',[?X,?Y,?Z]], ['VP2',?X,?Z], ['VMFIN',?Y]]"),  log(1/22.)),
		(fs("[['VP2',[?X],[?Y,?Z]], ['VP2@4',?X,?Y], ['VAINF@7',?Z]]"),  log(4/14.)),
		(fs("[['VP2',[?X],[?Y,?Z]], ['VP2@4',?X,?Y], ['VAINF',?Z]]"),  log(4/14.)),
		(fs("[['VP2',[?X],[?Y,?Z]], ['VP2',?X,?Y], ['VAINF@7',?Z]]"),  log(1/14.)),
		(fs("[['VP2',[?X],[?Y,?Z]], ['VP2',?X,?Y], ['VAINF',?Z]]"),  log(1/14.)),
		(fs("[['VP2@3',[?X],[?Y,?Z]], ['VP2@4',?X,?Y], ['VAINF@7',?Z]]"),  log(4/10.)),
		(fs("[['VP2@3',[?X],[?Y,?Z]], ['VP2@4',?X,?Y], ['VAINF',?Z]]"),  log(4/10.)),
		(fs("[['VP2@3',[?X],[?Y,?Z]], ['VP2',?X,?Y], ['VAINF@7',?Z]]"),  log(1/10.)),
		(fs("[['VP2@3',[?X],[?Y,?Z]], ['VP2',?X,?Y], ['VAINF',?Z]]"),  log(1/10.)),
		(fs("[['VP2',[?X],[?Y]], ['PROAV@5',?X], ['VVPP@6',?Y]]"),  log(1/14.)),
		(fs("[['VP2',[?X],[?Y]], ['PROAV@5',?X], ['VVPP',?Y]]"),  log(1/14.)),
		(fs("[['VP2',[?X],[?Y]], ['PROAV',?X], ['VVPP@6',?Y]]"),  log(1/14.)),
		(fs("[['VP2',[?X],[?Y]], ['PROAV',?X], ['VVPP',?Y]]"),  log(1/14.)),
		(fs("[['VP2@4',[?X],[?Y]], ['PROAV@5',?X], ['VVPP@6',?Y]]"),  log(1/4.)),
		(fs("[['VP2@4',[?X],[?Y]], ['PROAV@5',?X], ['VVPP',?Y]]"),  log(1/4.)),
		(fs("[['VP2@4',[?X],[?Y]], ['PROAV',?X], ['VVPP@6',?Y]]"),  log(1/4.)),
		(fs("[['VP2@4',[?X],[?Y]], ['PROAV',?X], ['VVPP',?Y]]"),  log(1/4.)),
		(fs("[['PROAV',['Daruber']], [Epsilon]]"),  0),
		(fs("[['PROAV@5',['Daruber']], [Epsilon]]"),  0),
		(fs("[['VVPP',['nachgedacht']], [Epsilon]]"),  0),
		(fs("[['VVPP@6',['nachgedacht']], [Epsilon]]"),  0),
		(fs("[['VMFIN',['muss']], [Epsilon]]"),  0),
		(fs("[['VMFIN@2',['muss']], [Epsilon]]"),  0),
		(fs("[['VAINF',['werden']], [Epsilon]]"),  0),
		(fs("[['VAINF@7',['werden']], [Epsilon]]"), 0)
		]

	do("Daruber muss nachgedacht werden")
	do("Daruber muss nachgedacht werden werden")
	do("Daruber muss nachgedacht werden werden werden")
	do("muss Daruber nachgedacht werden")
