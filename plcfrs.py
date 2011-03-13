# probabilistic CKY parser for Simple Range Concatenation Grammars
# (equivalent to Linear Context-Free Rewriting Systems)
from nltk import FeatStruct, FeatList, featstruct, FreqDist, Tree
from math import log, e
from itertools import chain, product
import re
try:
	import psyco
	psyco.full()
except: pass

def parse(sent, grammar, start="S", viterbi=False):
	""" parse sentence, a list of tokens, using grammar, a dictionary
	mapping rules to probabilities. """
	unary = [(r,w) for r,w in grammar.items() if len(r) == 2]
	binary = [(r,w) for r,w in grammar.items() if len(r) == 3]
	goal = fs([start, FeatList(range(len(sent)))])
	#goal = fs([start, FeatList([FeatList(sent)])])
	epsilon = fs("[Epsilon]")
	lhs = fs("[?LHS]")[0]
	def concat(node):
		val = [FeatList(chain(*x)) for x in node[0][1:]]
		return fs([FeatList([node[0,0]] + val)] + node[1:])
	def scan(sent):
		for i,w in enumerate(sent):
			for rule, z in unary:
				if rule[1] == epsilon and w in rule[0,1,0]:
					word = fs("[[%s, [%d]], %d]" % (repr(rule[0,0]), i, i))
					yield word, z
					#yield rule, z
	def deduced_from((I, x), C):
		for rule, z in unary:
			if rule[1,0] == I[0]:
				r = rule.unify(FeatList([lhs, I]))
				yield concat(r), z
		for Ic1, y in C.items():
			I1 = Ic1[0]
			#detect overlap
			if any(any(x in b for x in a) for a,b in product(I1[1:], I[1:])): continue
			for rule, z in binary:
				if rule[1,0] == I[0] and rule[2,0] == I1[0]:
					left = FeatList([lhs, I, I1])
					yield concat(rule.unify(left)), z
				elif rule[1,0] == I1[0] and rule[2,0] == I[0]:
					right = FeatList([lhs, I1, I])
					yield concat(rule.unify(right)), z
			
	A, C = {}, {}
	A.update(scan(sent))
	while A:	
		I, x = max(A.items(), key=lambda x: x[1])
		del A[I]
		C[I] = x
		if I[0] == goal:
			if viterbi: return C
		else:
			for I1, y in deduced_from((I[0], x), C):
				if I1 not in C and I1 not in A:
					A[I1] = y
				elif I1 in A:
					A[I1] = max(y, A[I1])
	if any(I[0] == goal for I in C): return C, goal
	else: return {}, []

def enumchart(chart, start):
	"""enumerate trees in chart headed by start in top down fashion. chart is
		a dictionary of FeatStructs and logprobs"""
	for a,p in chart.items():
		if a[0] == start:
			if len(a) == 2 and isinstance(a[1], int):
				yield Tree(str(a[0,0]).replace(' ','_'), map(str, a[1:])), p
				continue
			for x in product(*map(lambda y: enumchart(chart, y), a[1:])):
				yield Tree(str(start[0]).replace(' ','_'), [z[0] for z in x]), p+x[0][1]+x[1][1]

def do(sent):
	print "sentence", sent
	p, start = parse(sent.split(), grammar)
	if p:
		l = FreqDist()
		for n,(a,prob) in enumerate(enumchart(p, start)):
			#print n, prob, a
			l.inc(re.sub(r"@[0-9]+", "", Tree.convert(a).pprint(margin=99999)), e**prob)
		for a in l: print l[a], Tree(a)
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

grammar =  {
	fs("[[S,[?X,?Y,?Z]], ['VP2@3',?X,?Z], ['VMFIN@2',?Y]]"): log(10/22.),
	fs("[[S,[?X,?Y,?Z]], ['VP2@3',?X,?Z], [VMFIN,?Y]]"): log(10/22.),
	fs("[[S,[?X,?Y,?Z]], [VP2,?X,?Z], ['VMFIN@2',?Y]]"): log(1/22.),
	fs("[[S,[?X,?Y,?Z]], [VP2,?X,?Z], [VMFIN,?Y]]"): log(1/22.),
	fs("[[VP2,[?X],[?Y,?Z]], ['VP2@4',?X,?Y], ['VAINF@7',?Z]]"): log(4/14.),
	fs("[[VP2,[?X],[?Y,?Z]], ['VP2@4',?X,?Y], [VAINF,?Z]]"): log(4/14.),
	fs("[[VP2,[?X],[?Y,?Z]], [VP2,?X,?Y], ['VAINF@7',?Z]]"): log(1/14.),
	fs("[[VP2,[?X],[?Y,?Z]], [VP2,?X,?Y], [VAINF,?Z]]"): log(1/14.),
	fs("[['VP2@3',[?X],[?Y,?Z]], ['VP2@4',?X,?Y], ['VAINF@7',?Z]]"): log(0.4),
	fs("[['VP2@3',[?X],[?Y,?Z]], ['VP2@4',?X,?Y], [VAINF,?Z]]"): log(0.4),
	fs("[['VP2@3',[?X],[?Y,?Z]], [VP2,?X,?Y], ['VAINF@7',?Z]]"): log(0.1),
	fs("[['VP2@3',[?X],[?Y,?Z]], [VP2,?X,?Y], [VAINF,?Z]]"): log(0.1),
	fs("[[VP2,[?X],[?Y]], ['PROAV@5',?X], ['VVPP@6',?Y]]"): log(1/14.),
	fs("[[VP2,[?X],[?Y]], ['PROAV@5',?X], [VVPP,?Y]]"): log(1/14.),
	fs("[[VP2,[?X],[?Y]], [PROAV,?X], ['VVPP@6',?Y]]"): log(1/14.),
	fs("[[VP2,[?X],[?Y]], [PROAV,?X], [VVPP,?Y]]"): log(1/14.),
	fs("[['VP2@4',[?X],[?Y]], ['PROAV@5',?X], ['VVPP@6',?Y]]"): log(0.25),
	fs("[['VP2@4',[?X],[?Y]], ['PROAV@5',?X], [VVPP,?Y]]"): log(0.25),
	fs("[['VP2@4',[?X],[?Y]], [PROAV,?X], ['VVPP@6',?Y]]"): log(0.25),
	fs("[['VP2@4',[?X],[?Y]], [PROAV,?X], [VVPP,?Y]]"): log(0.25),
	fs("[[PROAV,[Daruber]], [Epsilon]]"): 0,
	fs("[['PROAV@5',[Daruber]], [Epsilon]]"): 0,
	fs("[[VVPP,[nachgedacht]], [Epsilon]]"): 0,
	fs("[['VVPP@6',[nachgedacht]], [Epsilon]]"): 0,
	fs("[[VMFIN,[muss]], [Epsilon]]"): 0,
	fs("[['VMFIN@2',[muss]], [Epsilon]]"): 0,
	fs("[[VAINF,[werden]], [Epsilon]]"): 0,
	fs("[['VAINF@7',[werden]], [Epsilon]]"): 0
	}

do("Daruber muss nachgedacht werden")
do("Daruber muss nachgedacht werden werden")
#do("Daruber muss nachgedacht werden werden werden")
do("muss Daruber nachgedacht werden")
