""" generate random sentences with an LCFRS. """
import codecs
from collections import namedtuple
from array import array
from math import exp, log
from random import random
from containers import Rule, LexicalRule

def gen(grammar, start=None, verbose=False):
	""" generate a random sentence """
	if start is None: start = grammar.toid['ROOT']
	if not grammar.bylhs[start]:
		terminal = chooserule(grammar.lexicalbylhs[start])
		return ( terminal.prob, [[terminal.word]] )
	rule = chooserule(grammar.bylhs[start])
	if not rule.rhs2:
		p1, l1 = gen(grammar, rule.rhs1, verbose)
		return (p1+rule.prob, l1)
	return compose(rule, gen(grammar, rule.rhs1, verbose),
				gen(grammar, rule.rhs2, verbose), verbose)

def compose(rule, (p1, l1), (p2, l2), verbose):
	result = []
	if verbose: print "[%g] %s + %s =" % (exp(-(rule.prob+p1+p2)), l1, l2,),
	for n,a in enumerate(rule.lengths):
		arg = []
		for b in range(a):
			if (rule.args[n] >> b) & 1:
				arg += l2.pop(0)
			else:
				arg += l1.pop(0)
		result.append(arg)
	if verbose: print result
	return (rule.prob+p1+p2, result)

def chooserule(rules, normalize=False):
	""" given a list of objects with probabilities,
	choose one according to that distribution."""
	position = random()
	if normalize: position *= sum(a.prob for a in rules)
	for r in rules:
		position -= exp(-r.prob)
		if position < 0: return r
	raise ValueError

def splitgrammar(grammar, lexicon):
	""" split the grammar into various lookup tables, mapping nonterminal
	labels to numeric identifiers. Also negates log-probabilities to
	accommodate min-heaps.
	Can only represent ordered SRCG rules (monotone LCFRS).
	This version represent rules in dedicated Rule objects, """
	Grammar = namedtuple("Grammar", "unary lbinary rbinary lexical bylhs lexicalbylhs toid tolabel arity".split())
	# get a list of all nonterminals; make sure Epsilon and ROOT are first, and assign them unique IDs
	nonterminals = list(enumerate(["Epsilon", "ROOT"]
		+ sorted(set(str(nt) for (rule, yf), weight in grammar for nt in rule)
			- set(["Epsilon", "ROOT"]))))
	toid = dict((lhs, n) for n, lhs in nonterminals)
	tolabel = dict((n, lhs) for n, lhs in nonterminals)
	bylhs = [[] for _ in nonterminals]
	unary = [[] for _ in nonterminals]
	lbinary = [[] for _ in nonterminals]
	rbinary = [[] for _ in nonterminals]
	arity = array('B', [0] * len(nonterminals))
	lexical = {}
	lexicalbylhs = {}
	# remove sign from log probabilities because the heap we use is a min-heap
	for (tag, word), w in lexicon:
		t = LexicalRule(toid[tag[0]], toid[tag[1]], 0, word, abs(w))
		assert arity[t.lhs] in (0, 1)
		arity[t.lhs] = 1
		lexical.setdefault(word, []).append(t)
		lexicalbylhs.setdefault(t.lhs, []).append(t)
	for (rule, yf), w in grammar:
		if len(rule) == 2 and toid[rule[1]] == 0:
			r = LexicalRule(toid[rule[0]], toid[rule[1]], 0, unicode(yf[0]), abs(w))
			assert arity[r.lhs] in (0, 1)
			arity[r.lhs] = 1
		else:
			args, lengths = yfarray(yf)
			assert yf == arraytoyf(args, lengths) # an unbinarized rule causes an error here
			if len(rule) == 2 and w == 0.0: w += 0.00000001
			r = Rule(toid[rule[0]], toid[rule[1]],
				toid[rule[2]] if len(rule) == 3 else 0, args, lengths, abs(w))
			if arity[r.lhs] == 0:
				arity[r.lhs] = len(args)
			assert arity[r.lhs] == len(args)
		if len(rule) == 2:
			unary[r.rhs1].append(r)
			bylhs[r.lhs].append(r)
		elif len(rule) == 3:
			lbinary[r.rhs1].append(r)
			rbinary[r.rhs2].append(r)
			bylhs[r.lhs].append(r)
		else: raise ValueError("grammar not binarized: %s" % repr(r))
	#assert 0 not in arity[1:]
	return Grammar(unary, lbinary, rbinary, lexical, bylhs, lexicalbylhs, toid, tolabel, arity)

def yfarray(yf):
	""" convert a yield function represented as a 2D sequence to an array
	object. """
	# I for 32 bits (int), H for 16 bits (short), B for 8 bits (char)
	# obviously, all related static declarations should match these types
	lentype = 'H'; lensize = 8 * array(lentype).itemsize
	vectype = 'I'; vecsize = 8 * array(vectype).itemsize
	assert len(yf) <= lensize
	assert all(len(a) <= vecsize for a in yf)
	initializer = [sum(2**n*b for n, b in enumerate(a)) for a in yf]
	return array(vectype, initializer), array(lentype, map(len, yf))

def arraytoyf(args, lengths):
	return tuple(tuple(1 if a & (1 << m) else 0 for m in range(n))
							for n, a in zip(lengths, args))

def read_srcg_grammar(rules, lexicon, encoding='utf-8'):
	""" Reads a grammar as produced by write_srcg_grammar. """
	rules = (a[:-1].split('\t') for a in codecs.open(rules, encoding=encoding))
	lexicon = (a[:-1].split('\t') for a in codecs.open(lexicon,
		encoding=encoding))
	rules = [((tuple(a[:-2]), tuple(tuple(map(int, b))
			for b in a[-2].split(","))), float(a[-1])) for a in rules]
	lexicon = [((tuple(a[:-2]), (a[-2])), float(a[-1])) for a in lexicon]
	return rules, lexicon

def test():
	rules = [
		((('S','VP2','VMFIN'),    ((0,1,0),)),   log(1.0)),
		((('VP2','VP2','VAINF'),  ((0,),(0,1))), log(0.5)),
		((('VP2','PROAV','VVPP'), ((0,),(1,))),  log(0.5)),
		((('VP2','VP2'),          ((0,),(0,))),  log(0.1))]
	lexicon = [
		((('PROAV', 'Epsilon'), u'Darueber'),     0.0),
		((('VAINF', 'Epsilon'), u'werden'),      0.0),
		((('VMFIN', 'Epsilon'), u'muss'),        0.0),
		((('VVPP', 'Epsilon'),  u'nachgedacht'), 0.0)]
	grammar = splitgrammar(rules, lexicon)
	p, sent = gen(grammar, start=grammar.toid['S'], verbose=True)
	print " ".join(sent.pop())

def main():
	rules, lexicon = read_srcg_grammar("rules.srcg", "lexicon.srcg")
	grammar = splitgrammar(rules, lexicon)
	for a in range(20):
		p, sent = gen(grammar)
		print "[%g] %s" % (exp(-p), " ".join(sent.pop()))

if __name__ == '__main__':
	import sys
	if "--test" in sys.argv: test()
	else: main()
