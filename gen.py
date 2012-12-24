""" Generate random sentences with an LCFRS. Reads grammar from a
text file. """
import codecs, gzip, sys
from collections import namedtuple, defaultdict
from fractions import Fraction
from math import exp, log
from array import array
from random import random
from grammar import read_lcfrs_grammar, read_bitpar_grammar

USAGE = """
Generate random sentences with a PLCFRS or PCFG.
Reads grammar from a text file in LCFRS or bitpar format.
usage: %s rules lexicon | --test

Grammar is assumed to be in UTF-8; may be gzip'ed (.gz extension).
""" % sys.argv[0]

Grammar = namedtuple("Grammar", ('unary', 'lbinary', 'rbinary', 'lexical',
		'bylhs', 'lexicalbylhs', 'toid', 'tolabel', 'fanout'))
Rule = namedtuple("Rule",
		('lhs', 'rhs1', 'rhs2', 'args', 'lengths', 'prob'))
LexicalRule = namedtuple("LexicalRule",
		('lhs', 'rhs1', 'rhs2', 'word', 'prob'))

def gen(grammar, start=None, verbose=False):
	""" generate a random sentence in top-down fashion. """
	if start is None:
		start = grammar.toid['ROOT']
	if not grammar.bylhs[start]:
		terminal = chooserule(grammar.lexicalbylhs[start])
		return (terminal.prob, [[terminal.word]])
	rule = chooserule(grammar.bylhs[start])
	if not rule.rhs2:
		p1, l1 = gen(grammar, rule.rhs1, verbose)
		return (p1 + rule.prob, l1)
	return compose(rule, gen(grammar, rule.rhs1, verbose),
				gen(grammar, rule.rhs2, verbose), verbose)

def compose(rule, (p1, l1), (p2, l2), verbose):
	""" Combine the results of two generated non-terminals into a single
	non-terminal, as specified by the given rule. """
	result = []
	if verbose:
		print "[%g] %s + %s =" % (exp(-(rule.prob+p1+p2)), l1, l2),
	for n, a in enumerate(rule.lengths):
		arg = []
		for b in range(a):
			if (rule.args[n] >> b) & 1:
				arg += l2.pop(0)
			else:
				arg += l1.pop(0)
		result.append(arg)
	if verbose:
		print result
	return (rule.prob + p1 + p2, result)

def chooserule(rules, normalize=False):
	""" given a list of objects with probabilities,
	choose one according to that distribution."""
	position = random()
	if normalize:
		position *= sum(a.prob for a in rules)
	for r in rules:
		position -= exp(-r.prob)
		if position < 0:
			return r
	raise ValueError

def splitgrammar(rules):
	""" split the grammar into various lookup tables, mapping nonterminal
	labels to numeric identifiers. Also turns probabilities into negative
	log-probabilities. Can only represent binary, monotone LCFRS rules. """
	# get a list of all nonterminals; make sure Epsilon and ROOT are first,
	# and assign them unique IDs
	nonterminals = list(enumerate(["Epsilon", "ROOT"]
		+ sorted(set(str(nt) for (rule, yf), weight in rules for nt in rule)
			- set(["Epsilon", "ROOT"]))))
	grammar = Grammar(
			toid=dict((lhs, n) for n, lhs in nonterminals),
			tolabel=dict((n, lhs) for n, lhs in nonterminals),
			bylhs=[[] for _ in nonterminals],
			unary=[[] for _ in nonterminals],
			lbinary=[[] for _ in nonterminals],
			rbinary=[[] for _ in nonterminals],
			fanout=array('B', [0] * len(nonterminals)),
			lexical={},
			lexicalbylhs={})
	for (rule, yf), w in rules:
		if rule[1] == 'Epsilon':
			word = yf[0]
			t = LexicalRule(grammar.toid[rule[0]], 0, 0, word, abs(log(w)))
			assert grammar.fanout[t.lhs] in (0, 1)
			grammar.fanout[t.lhs] = 1
			grammar.lexical.setdefault(word, []).append(t)
			grammar.lexicalbylhs.setdefault(t.lhs, []).append(t)
			continue
		if len(rule) == 2 and grammar.toid[rule[1]] == 0:
			r = LexicalRule(grammar.toid[rule[0]], grammar.toid[rule[1]], 0,
					unicode(yf[0]), abs(log(w)))
			assert grammar.fanout[r.lhs] in (0, 1)
			grammar.fanout[r.lhs] = 1
		else:
			args, lengths = yfarray(yf)
			# an unbinarized rule causes an error here
			assert yf == arraytoyf(args, lengths)
			if len(rule) == 2 and w == 1:
				w -= 0.00000001
			r = Rule(grammar.toid[rule[0]], grammar.toid[rule[1]],
				grammar.toid[rule[2]] if len(rule) == 3 else 0, args, lengths,
						abs(log(w)))
			if grammar.fanout[r.lhs] == 0:
				grammar.fanout[r.lhs] = len(args)
			assert grammar.fanout[r.lhs] == len(args)
		if len(rule) == 2:
			grammar.unary[r.rhs1].append(r)
			grammar.bylhs[r.lhs].append(r)
		elif len(rule) == 3:
			grammar.lbinary[r.rhs1].append(r)
			grammar.rbinary[r.rhs2].append(r)
			grammar.bylhs[r.lhs].append(r)
		else:
			raise ValueError("grammar not binarized: %s" % repr(r))
	#assert 0 not in grammar.fanout[1:]
	return grammar

def yfarray(yf):
	""" convert a yield function represented as a 2D sequence to an array
	object. """
	# I for 32 bits (int), H for 16 bits (short), B for 8 bits (char)
	# obviously, all related static declarations should match these types
	lentype = 'H'
	vectype = 'I'
	lensize = 8 * array(lentype).itemsize
	vecsize = 8 * array(vectype).itemsize
	assert len(yf) <= lensize
	assert all(len(a) <= vecsize for a in yf)
	initializer = [sum(2**n*b for n, b in enumerate(a)) for a in yf]
	return array(vectype, initializer), array(lentype, map(len, yf))

def arraytoyf(args, lengths):
	""" Inverse of yfarray(). """
	return tuple(tuple(1 if a & (1 << m) else 0 for m in range(n))
							for n, a in zip(lengths, args))

def test():
	""" Demonstration on an example grammar. """
	rules = [
		((('S', 'VP2', 'VMFIN'), ((0, 1, 0), )),  1.0),
		((('VP2', 'VP2', 'VAINF'), ((0, ), (0, 1))), 0.5),
		((('VP2', 'PROAV', 'VVPP'), ((0, ), (1, ))), 0.5),
		((('VP2', 'VP2'), ((0, ), (0, ))), 0.1),
		((('PROAV', 'Epsilon'), (u'Darueber', )), 1),
		((('VAINF', 'Epsilon'), (u'werden', )), 1),
		((('VMFIN', 'Epsilon'), (u'muss', )), 1),
		((('VVPP', 'Epsilon'), (u'nachgedacht', )), 1)]
	grammar = splitgrammar(rules)
	_, sent = gen(grammar, start=grammar.toid['S'], verbose=True)
	print " ".join(sent.pop())

def main():
	""" Load a grammar from a text file and generate 20 sentences. """
	if len(sys.argv) != 3:
		print USAGE
		return
	rules = (gzip.open if sys.argv[1].endswith(".gz") else open)(sys.argv[1])
	lexicon = codecs.getreader('utf-8')((gzip.open
			if sys.argv[2].endswith(".gz") else open)(sys.argv[2]))
	try:
		grammar = read_lcfrs_grammar(rules, lexicon)
	except ValueError as err:
		print err
		grammar = read_bitpar_grammar(rules, lexicon)
	grammar = splitgrammar(grammar)
	for _ in range(20):
		p, sent = gen(grammar)
		print "[%g] %s" % (exp(-p), " ".join(sent.pop()))

if __name__ == '__main__':
	if "--test" in sys.argv:
		test()
	else:
		main()
