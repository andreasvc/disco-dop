# probabilistic CKY parser for Simple Range Concatenation Grammars
# (equivalent to Linear Context-Free Rewriting Systems)
from nltk import FeatStruct, FeatList, featstruct, FreqDist, Tree
from math import log, e
from random import choice
from itertools import chain, product, islice
from pprint import pprint
from collections import defaultdict
import re
try:
	import psyco
	psyco.full()
except: pass

def parse(sent, grammar, start="S", viterbi=False):
	""" parse sentence, a list of tokens, using grammar, a dictionary
	mapping rules to probabilities. """
	unary, binary = defaultdict(list), defaultdict(list)
	for r,w in grammar.items():
		if len(r) == 2: unary[r[1][0]].append((r, w))
		elif len(r) == 3: binary[(r[1][0], r[2][0])].append((r, w))
	goal = fs([start, FeatList(range(len(sent)))])
	#goal = fs([start, FeatList([FeatList(sent)])])
	epsilon = fs("[Epsilon]")
	lhs = fs("[?LHS]")[0]
	def contiguous(s):
		return all(a==b for a,b in zip(s, range(s[0], s[0]+len(s))))
	def concat(node):
		val = [FeatList(chain(*x)) for x in node[0][1:]]
		return fs([FeatList([node[0,0]] + val)] + node[1:])
	def scan(sent):
		for i,w in enumerate(sent):
			for rule, z in unary[epsilon[0]]:
				if w in rule[0][1][0]:
					word = fs("[[%s, [%d]], %d]" % (repr(rule[0][0]), i, i))
					yield word, z
					#yield rule, z
	def deduced_from((I, x), C):
		for rule, z in unary[I[0]]:
			r = concat(rule.unify(FeatList([lhs, I])))
			if all(map(contiguous, r[0][1:])): yield r, z
		for Ic1, y in C.items():
			I1 = Ic1[0]
			#detect overlap in ranges
			if any(a & b for a,b in product(map(set, I1[1:]), map(set, I[1:]))): continue
			for rule, z in binary[(I[0], I1[0])]:
				left = concat(rule.unify(FeatList([lhs, I, I1])))
				if all(map(contiguous, left[0][1:])): yield left, z
			for rule, z in binary[(I1[0], I[0])]:
				right = concat(rule.unify(FeatList([lhs, I1, I])))
				if all(map(contiguous, right[0][1:])): yield right, z
			
	A, C = {}, {}
	A.update(scan(sent))
	while A:	
		I, x = max(A.items(), key=lambda x: x[1])
		del A[I]
		C[I] = x
		if I[0] == goal:
			if viterbi: return C, goal
		else:
			for I1, y in deduced_from((I[0], x), C):
				if I1 not in C and I1 not in A:
					A[I1] = y
				elif I1 in A:
					A[I1] = max(y, A[I1])
	if any(I[0] == goal for I in C): return C, goal
	else: return {}, []

def cartpi(seq):
	""" itertools.product doesn't support infinite sequences! """
	if seq: return ((a,) + b for b in cartpi(seq[1:]) for a in seq[0])
	return ((), )

def enumchart(chart, start):
	"""enumerate trees in chart headed by start in top down fashion. chart is
		a dictionary of FeatStructs and logprobs"""
	def enumchart1(chart, start):
		for a,p in chart[start]:
			if len(a) == 1 and isinstance(a[0], int):
				yield Tree(start[0], a), p
				continue
			for x in cartpi(map(lambda y: enumchart1(chart, y), a)):
				yield Tree(start[0], [z[0] for z in x]), p+x[0][1]+x[1][1]
	chart1 = defaultdict(list)
	for a,p in chart.items(): chart1[a[0]].append((a[1:],p))
	return enumchart1(chart1, start)

def samplechart(chart, start):
	def samplechart1(chart, start):
		entry, p = choice(chart[start])
		if len(entry) == 1 and isinstance(entry[0], int):
			return Tree(str(start[0]), entry), p
		children = [samplechart1(chart, a) for a in entry]
		tree = Tree(start[0], [a for a,b in children]) 
		return tree, p+sum(b for a,b in children)
	chart1 = defaultdict(list)
	for a,p in chart.items(): chart1[a[0]].append((a[1:],p))
	while True: yield samplechart1(chart1, start)
	
def mostprobableparse(chart, start, n=100):
		""" sum over n random derivations from chart, 
			return a FreqDist of parse trees, with .max() being the MPP"""
		l = FreqDist()
		for a,prob in islice(samplechart(chart, start), n):
			l.inc(re.sub(r"@[0-9]+", "", a.pprint(margin=999)), e**prob)
		return l

def do(sent):
	print "sentence", sent
	chart, start = parse(sent.split(), grammar)
	if chart:
		for a, p in mostprobableparse(chart, start).items(): 
			print p, Tree(a)
	else: print "no parse"
	print

def fs(a):
	x = FeatStruct(a)
	x.freeze()
	return x

grammar =  {
	fs("[[S,[?X,?Y,?Z]], [VP2,?X,?Z], [VMFIN,?Y]]"): 0,
	fs("[[VP2,[?X],[?Y,?Z]],[VP2,?X,?Y], [VAINF,?Z]]"): log(0.5),
	fs("[[VP2,[?X],[?Y]], [PROAV,?X], [VVPP,?Y]]"): log(0.5),
	fs("[[PROAV,[Daruber]], [Epsilon]]"): 0,
	fs("[[VVPP,[nachgedacht]], [Epsilon]]"): 0,
	fs("[[VMFIN,[muss]], [Epsilon]]"): 0,
	fs("[[VAINF,[werden]], [Epsilon]]"): 0
	}

# a DOP reduction according to Goodman (1996)
grammar =  {
	fs("[[S,[?X,?Y,?Z]], ['VP2@3',?X,?Z], ['VMFIN@2',?Y]]"): log(10/22.),
	fs("[[S,[?X,?Y,?Z]], ['VP2@3',?X,?Z], [VMFIN,?Y]]"): log(10/22.),
	fs("[[S,[?X,?Y,?Z]], [VP2,?X,?Z], ['VMFIN@2',?Y]]"): log(1/22.),
	fs("[[S,[?X,?Y,?Z]], [VP2,?X,?Z], [VMFIN,?Y]]"): log(1/22.),
	fs("[[VP2,[?X],[?Y,?Z]], ['VP2@4',?X,?Y], ['VAINF@7',?Z]]"): log(4/14.),
	fs("[[VP2,[?X],[?Y,?Z]], ['VP2@4',?X,?Y], [VAINF,?Z]]"): log(4/14.),
	fs("[[VP2,[?X],[?Y,?Z]], [VP2,?X,?Y], ['VAINF@7',?Z]]"): log(1/14.),
	fs("[[VP2,[?X],[?Y,?Z]], [VP2,?X,?Y], [VAINF,?Z]]"): log(1/14.),
	fs("[['VP2@3',[?X],[?Y,?Z]], ['VP2@4',?X,?Y], ['VAINF@7',?Z]]"): log(4/10.),
	fs("[['VP2@3',[?X],[?Y,?Z]], ['VP2@4',?X,?Y], [VAINF,?Z]]"): log(4/10.),
	fs("[['VP2@3',[?X],[?Y,?Z]], [VP2,?X,?Y], ['VAINF@7',?Z]]"): log(1/10.),
	fs("[['VP2@3',[?X],[?Y,?Z]], [VP2,?X,?Y], [VAINF,?Z]]"): log(1/10.),
	fs("[[VP2,[?X],[?Y]], ['PROAV@5',?X], ['VVPP@6',?Y]]"): log(1/14.),
	fs("[[VP2,[?X],[?Y]], ['PROAV@5',?X], [VVPP,?Y]]"): log(1/14.),
	fs("[[VP2,[?X],[?Y]], [PROAV,?X], ['VVPP@6',?Y]]"): log(1/14.),
	fs("[[VP2,[?X],[?Y]], [PROAV,?X], [VVPP,?Y]]"): log(1/14.),
	fs("[['VP2@4',[?X],[?Y]], ['PROAV@5',?X], ['VVPP@6',?Y]]"): log(1/4.),
	fs("[['VP2@4',[?X],[?Y]], ['PROAV@5',?X], [VVPP,?Y]]"): log(1/4.),
	fs("[['VP2@4',[?X],[?Y]], [PROAV,?X], ['VVPP@6',?Y]]"): log(1/4.),
	fs("[['VP2@4',[?X],[?Y]], [PROAV,?X], [VVPP,?Y]]"): log(1/4.),
	fs("[[PROAV,[Daruber]], [Epsilon]]"): 0,
	fs("[['PROAV@5',[Daruber]], [Epsilon]]"): 0,
	fs("[[VVPP,[nachgedacht]], [Epsilon]]"): 0,
	fs("[['VVPP@6',[nachgedacht]], [Epsilon]]"): 0,
	fs("[[VMFIN,[muss]], [Epsilon]]"): 0,
	fs("[['VMFIN@2',[muss]], [Epsilon]]"): 0,
	fs("[[VAINF,[werden]], [Epsilon]]"): 0,
	fs("[['VAINF@7',[werden]], [Epsilon]]"): 0
	}

if __name__ == '__main__':
	do("Daruber muss nachgedacht werden")
	do("Daruber muss nachgedacht werden werden")
	#do("Daruber muss nachgedacht werden werden werden")
	do("muss Daruber nachgedacht werden")
