from heapdict import heapdict
from plcfrs import ChartItem, bitcount, nextset, nextunset
from collections import defaultdict
from math import e
# Implementation of LR estimate (Kallmeyer & Maier 2010). Ported almost directly from rparse.

class Item:
	__slots__ = ("state", "len", "lr", "gaps", "_hash")
	def __init__(self, state, len, lr, gaps):
		self.state, self.len, self.lr, self.gaps = state, len, lr, gaps
		self._hash = hash((state, len, lr, gaps))
	def __hash__(self):
		return self._hash
	def __repr__(self):
		return "%s len=%d lr=%d gaps=%d" % (self.state, self.len, self.lr, self.gaps)

def getestimates(grammar, maxlen, goal):
	#insidescores = inside(grammar, maxlen)
	#for a in insidescores:
	#	for b in insidescores[a]:
	#		print "%s[%s] =" % (a, bin(b)[2:]), e**-insidescores[a][b]
	#x=len(insidescores) * sum(map(len, insidescores.values()))
	#print
	insidescores = simpleinside(grammar, maxlen)
	for a in insidescores:
		for b in insidescores[a]:
			print "%s[%d] =" % (a, b), e**-insidescores[a][b]
	print 
	print len(insidescores) * sum(map(len, insidescores.values()))
	infinity = float('infinity')
	outside = outsidelr(grammar, insidescores, maxlen, goal)
	#for a in sorted(outside):
	#	if outside[a] < infinity: print a, e**-outside[a]
	#
	print len(outside)
	#return
	cnt = 0
	for a in outside:
		for bn, b in enumerate(outside[a]):
			for cn, c in enumerate(b):
				for dn, d in enumerate(c):
					if d < infinity: print a,bn,cn,dn, e**-d; cnt += 1
	print cnt

def inside(grammar, maxlen):
	return doinside(grammar, maxlen, insideconcat)

def simpleinside(grammar, maxlen):
	""" Here vec is actually the length (number of terminals in the yield of the constituent) """
	return doinside(grammar, maxlen, simpleconcat)

def doinside(grammar, maxlen, concat):
	unary, lbinary, rbinary, bylhs = grammar
	agenda = heapdict()
	insidescores = defaultdict(lambda: defaultdict(lambda: float('infinity')))
	#insidescores = defaultdict(dict)
	for rule,z in unary['Epsilon']:
		agenda[ChartItem(rule[0][0], 1)] = 0.0
	while agenda:
		I,x = agenda.popitem()
		if I.vec not in insidescores[I.label] or insidescores[I.label][I.vec] < x:
			insidescores[I.label][I.vec] = x
		
		results = []
		for rule, y in unary[I.label]:
			results.append((rule[0][0], I.vec, y+insidescores[rule[0][0]]))
		for rule, y in lbinary[I.label]:
			for vec in insidescores[rule[0][2]]:
				left = concat(I.vec, vec, rule[1], maxlen)
				if left: results.append((rule[0][0], left, x+y+insidescores[rule[0][2]][vec]))
		for rule, y in rbinary[I.label]:
			for vec in insidescores[rule[0][1]]:
				right = concat(vec, I.vec, rule[1], maxlen)
				if right: results.append((rule[0][0], right, x+y+insidescores[rule[0][1]][vec]))

		for label, vec, score in results:
			if label not in insidescores or vec not in insidescores[label]:
				agenda[ChartItem(label, vec)] = score
	return insidescores

def simpleconcat(a, b, ignored, maxlen):
	return a+b if a+b <= maxlen else 0

def insideconcat(a, b, yieldfunction, maxlen):
	if not (a and b and yieldfunction) or len(yieldfunction) + bitcount(a) + bitcount(b) > maxlen + 1:
		return
	result = resultpos = l = r = 0
	for arg in yieldfunction:
		for x in arg:
			if x == 0:
				subarg = nextunset(a, l) - l
				result |= (1 << subarg) - 1 << resultpos
				resultpos += subarg
				l = subarg + 1
			elif x == 1:
				subarg = nextunset(b, r) - r
				result |= (1 << subarg) - 1 << resultpos
				resultpos += subarg
				r = subarg + 1
			else: raise ValueError("non-binary value in yield function")
		resultpos += 1
		result &= ~(1 << resultpos)
	return result

def getoutside(I, outside, maxlen):
	if I.len+I.lr+I.gaps <= maxlen: return outside[I]
	else: return 0.0

def outsidelr(grammar, insidescores, maxlen, goal):
	u,l,r,bylhs = grammar
	agenda = heapdict()
	infinity = float('infinity')
	outside = dict((lhs, [[[infinity] * (maxlen+1) for b in range(maxlen - c + 1)] for c in range(maxlen+1)]) for lhs in bylhs)
	for a in range(1, maxlen+1):
		newitem = Item(goal, a, 0, 0)
		agenda[newitem] = 0.0
		outside[goal][a][0][0] = 0.0
	while agenda:
		I, x = agenda.popitem()
		if x == outside[I.state][I.len][I.lr][I.gaps]:
			totlen = I.len + I.lr + I.gaps
			for rule, y in bylhs[I.state]:
				# X -> A
				if len(rule[0]) == 2:
					if rule[0][1] != "Epsilon":
						newitem = Item(rule[0][1], I.len, I.lr, I.gaps)
						score = x + y
						if outside[rule[0][1]][I.len][I.lr][I.gaps] > score:
							agenda[newitem] = score
							outside[rule[0][1]][I.len][I.lr][I.gaps] = score
				else:
					lstate = rule[0][1]
					rstate = rule[0][2]
					yieldfunction = rule[1]
					# X -> A B
					addgaps = addright = 0
					stopaddright = False
					for arg in yieldfunction[::-1]:
						for a in arg[::-1]:
							if a == 0:
								stopaddright = True
							elif a == 1:
								if not stopaddright:
									addright += 1
								else:
									addgaps += 1
							else: raise ValueError("strange value in yield function")

					leftarity = sum(arg.count(0) for arg in yieldfunction)
					rightarity = sum(arg.count(1) for arg in yieldfunction)
					# binary-left (A is left)
					for lenA in range(leftarity, I.len - rightarity + 1):
						lenB = I.len - lenA
						insidescore = insidescores[rstate][lenB]
						for lr in range(I.lr, I.lr + lenB + 1):
							if addright == 0 and lr != I.lr: continue
							for ga in range(leftarity - 1, totlen+1):
								if lenA + lr + ga == I.len + I.lr + I.gaps and ga >= addgaps:
									newitem = Item(lstate, lenA, lr, ga)
									score = x + insidescore + y
									if outside[lstate][lenA][lr][ga] > score:
										agenda[newitem] = score
										outside[lstate][lenA][lr][ga] = score

					# X -> B A
					addgaps = addleft = 0
					stopaddleft = False
					for arg in yieldfunction:
						for a in arg:
							if a == 1:
								stopaddleft = True
							elif a == 0:
								if stopaddleft:
									addgaps += 1
								else:
									addleft += 1
							else: raise ValueError("strange value in yield function")

					addright = 0
					stopaddright = False
					for arg in yieldfunction[::-1]:
						for a in arg[::-1]:
							if a == 1:
								stopaddright = True
							elif a == 0:
								if not stopaddright:
									addright += 1
							else: raise ValueError("strange value in yield function")
					addgaps -= addright
					
					# binary-right (A is right)
					for lenA in range(rightarity, I.len - leftarity + 1):
						lenB = I.len - lenA
						insidescore = insidescores[lstate][lenB]
						for lr in range(I.lr, I.lr + lenB + 1):
							for ga in range(rightarity - 1, totlen+1):
								if lenA + lr + ga == I.len + I.lr + I.gaps and ga >= addgaps:
									newitem = Item(rstate, lenA, lr, ga)
									score = x + insidescore + y
									if outside[rstate][lenA][lr][ga] > score:
										agenda[newitem] = score
										outside[rstate][lenA][lr][ga] = score
		#else: print I,x, outside[I.state][I.len][I.lr][I.gaps]

	return outside

def main():
	from negra import NegraCorpusReader
	from rcgrules import induce_srcg, dop_srcg_rules, splitgrammar
	from nltk import Tree
	corpus = NegraCorpusReader(".", "sample2\.export")
	grammar = splitgrammar(dop_srcg_rules(corpus.parsed_sents(), corpus.sents()))
	getestimates(grammar, 30, "ROOT")
	#tree = Tree("(S (VP (VP (PROAV 0) (VVPP 2)) (VAINF 3)) (VMFIN 1))")
	#tree.chomsky_normal_form()
	#sent = "Daruber muss nachgedacht werden".split()
	#grammar = splitgrammar(dop_srcg_rules([tree], [sent]))
	#getestimates(grammar, 6, "S")

if __name__ == '__main__': main()
