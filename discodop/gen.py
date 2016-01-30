"""Generate random sentences with an LCFRS.

Reads grammar from a text file."""
from __future__ import division, print_function, absolute_import
import sys
import gzip
import codecs
from collections import namedtuple, defaultdict
from array import array
from random import random

SHORTUSAGE = '''Generate random sentences with a PLCFRS or PCFG.
Reads grammar from a text file in PLCFRS or bitpar format.
Usage: %(cmd)s [--verbose] <rules> <lexicon>
or: %(cmd)s --test

Grammar is assumed to be in UTF-8; may be gzip'ed (.gz extension).
''' % dict(cmd=sys.argv[0])

Grammar = namedtuple("Grammar", ('numrules', 'unary', 'lbinary', 'rbinary',
		'bylhs', 'lexicalbyword', 'lexicalbylhs', 'toid', 'tolabel', 'fanout'))
Rule = namedtuple("Rule",
		('lhs', 'rhs1', 'rhs2', 'args', 'lengths', 'prob', 'no'))
LexicalRule = namedtuple("LexicalRule",
		('lhs', 'rhs1', 'rhs2', 'word', 'prob', 'no'))


def gen(grammar, start=1, discount=0.75, prodcounts=None, verbose=False):
	"""Generate a random sentence in top-down fashion.

	:param discount: a factor between 0 and 1.0; 1.0 means no discount, lower
		values introduce increasingly larger discount for repeated rules.

	Cf. http://eli.thegreenplace.net/2010/01/28/generating-random-sentences\
-from-a-context-free-grammar/"""
	if prodcounts is None:
		prodcounts = [1] * grammar.numrules
	if not grammar.bylhs[start]:
		terminal = chooserule(grammar.lexicalbylhs[start], discount, prodcounts)
		return (terminal.prob, [[terminal.word]])
	rule = chooserule(grammar.bylhs[start], discount, prodcounts)
	prodcounts[rule.no] += 1
	p1, l1 = gen(grammar, rule.rhs1, discount, prodcounts, verbose)
	assert l1
	if rule.rhs2:
		p2, l2 = gen(grammar, rule.rhs2, discount, prodcounts, verbose)
		assert l2
	prodcounts[rule.no] -= 1
	if verbose:
		print('%s => %s %r [p=%s]' % (
				grammar.tolabel[rule.lhs],
				' '.join((grammar.tolabel[rule.rhs1],
				grammar.tolabel[rule.rhs2] if rule.rhs2 else '')),
				arraytoyf(rule.args, rule.lengths), rule.prob))
	if rule.rhs2:
		return compose(rule, (p1, l1), (p2, l2), verbose)
	return (p1 * rule.prob, l1)


def chooserule(rules, discount, prodcounts):
	"""Given a list of weighted rules, choose one following the distribution."""
	weights = [rule.prob * discount ** prodcounts[rule.no] for rule in rules]
	position = random() * sum(weights)
	for r, w in zip(rules, weights):
		position -= w
		if position < 0:
			return r
	raise ValueError('random selection out of range.')


def compose(rule, left, right, verbose):
	"""Use rule to compose two non-terminals into a new non-terminal."""
	(p1, l1), (p2, l2) = left, right
	result = []
	if verbose:
		print("[%s] %s + %s = " % (rule.prob * p1 * p2, l1, l2), end='')
	for n, a in enumerate(rule.lengths):
		arg = []
		for b in range(a):
			if (rule.args[n] >> b) & 1:
				arg += l2.pop(0)
			else:
				arg += l1.pop(0)
		result.append(arg)
	if verbose:
		print(result)
	return (rule.prob * p1 * p2, result)


def parsefrac(a):
	"""Parse a string of a fraction into a float ('1/2' => 0.5).

	Substitute for creating Fraction objects (which is slow)."""
	n = a.find('/')
	if n == -1:
		return float(a)
	return float(a[:n]) / float(a[n + 1:])


def read_lcfrs_grammar(rules, lexicon):
	"""Read a grammar produced by grammar.writegrammar from two file objects."""
	rules = (a.strip().split('\t') for a in rules)
	grammar = [((tuple(a[:-2]), tuple(tuple(map(int, b))
			for b in a[-2].split(","))), parsefrac(a[-1])) for a in rules]
	# one word per line, word (tag weight)+
	grammar += [(((t, 'Epsilon'), (lexentry[0],)), parsefrac(p))
			for lexentry in (a.strip().split() for a in lexicon)
			for t, p in zip(lexentry[1::2], lexentry[2::2])]
	return grammar


def read_bitpar_grammar(rules, lexicon):
	"""Read a bitpar grammar given two file objects.

	Must be a binarized grammar. Integer frequencies will be converted to exact
	relative frequencies; otherwise weights are kept as-is."""
	grammar = []
	integralweights = True
	ntfd = defaultdict(int)
	for a in rules:
		a = a.split()
		p, rule = float(a[0]), a[1:]
		if integralweights:
			ip = int(p)
			if p == ip:
				p = ip
			else:
				integralweights = False
		ntfd[rule[0]] += p
		if len(rule) == 2:
			grammar.append(((tuple(rule), ((0,),)), p))
		elif len(rule) == 3:
			grammar.append(((tuple(rule), ((0, 1),)), p))
		else:
			raise ValueError("grammar is not binarized")
	for a in lexicon:
		a = a.split()
		word = a[0]
		tags, weights = a[1::2], a[2::2]
		weights = map(float, weights)
		if integralweights:
			if all(int(w) == w for w in weights):
				weights = map(int, weights)
			else:
				integralweights = False
		tags = zip(tags, weights)
		for t, p in tags:
			ntfd[t] += p
		grammar.extend((((t, 'Epsilon'), (word,)), p) for t, p in tags)
	if integralweights:
		return [(rule, p / ntfd[rule[0][0]]) for rule, p in grammar]
	return grammar


def splitgrammar(rules):
	"""Split a grammar into various lookup tables.

	Also maps nonterminal labels to numeric identifiers, and turns
	probabilities into negative log-probabilities. Can only represent binary,
	monotone LCFRS rules."""
	# get a list of all nonterminals; make sure Epsilon and ROOT are first,
	# and assign them unique IDs
	nonterminals = list(enumerate(["Epsilon", "ROOT"]
		+ sorted(set(str(nt) for (rule, yf), weight in rules for nt in rule)
			- {"Epsilon", "ROOT"})))
	grammar = Grammar(
			toid=dict((lhs, n) for n, lhs in nonterminals),
			tolabel=dict((n, lhs) for n, lhs in nonterminals),
			bylhs=[[] for _ in nonterminals],
			unary=[[] for _ in nonterminals],
			lbinary=[[] for _ in nonterminals],
			rbinary=[[] for _ in nonterminals],
			fanout=array('B', [0] * len(nonterminals)),
			lexicalbyword={},
			lexicalbylhs={},
			numrules=len(rules))
	for n, ((rule, yf), w) in enumerate(rules):
		if rule[1] == 'Epsilon':
			word = yf[0]
			t = LexicalRule(grammar.toid[rule[0]], 0, 0, word, w, n)
			assert grammar.fanout[t.lhs] in (0, 1)
			grammar.fanout[t.lhs] = 1
			grammar.lexicalbyword.setdefault(word, []).append(t)
			grammar.lexicalbylhs.setdefault(t.lhs, []).append(t)
			continue
		else:
			args, lengths = yfarray(yf)
			assert yf == arraytoyf(args, lengths), "rule not binarized?"
			if len(rule) == 2 and w == 1:
				w -= 0.00000001
			r = Rule(grammar.toid[rule[0]], grammar.toid[rule[1]],
					grammar.toid[rule[2]] if len(rule) == 3 else 0, args,
					lengths, w, n)
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
	# assert 0 not in grammar.fanout[1:]
	return grammar


def yfarray(yf):
	"""Convert yield function represented as 2D sequence to an array object."""
	# I for 32 bits (int), H for 16 bits (short), B for 8 bits (char)
	# obviously, all related static declarations should match these types
	lentype = 'H'
	vectype = 'I'
	lensize = 8 * array(lentype).itemsize
	vecsize = 8 * array(vectype).itemsize
	assert len(yf) <= lensize
	assert all(len(a) <= vecsize for a in yf)
	initializer = [sum(2 ** n * b for n, b in enumerate(a)) for a in yf]
	return array(vectype, initializer), array(lentype, map(len, yf))


def arraytoyf(args, lengths):
	"""Inverse of yfarray()."""
	return tuple(tuple(1 if a & (1 << m) else 0 for m in range(n))
							for n, a in zip(lengths, args))


def test():
	"""Demonstration on an example grammar."""
	rules = [
		((('S', 'VP2', 'VMFIN'), ((0, 1, 0), )), 1),
		((('VP2', 'VP2', 'VAINF'), ((0, ), (0, 1))), 1. / 2),
		((('VP2', 'PROAV', 'VVPP'), ((0, ), (1, ))), 1. / 2),
		((('VP2', 'VP2'), ((0, ), (0, ))), 1. / 10),
		((('PROAV', 'Epsilon'), ('Darueber', )), 1),
		((('VAINF', 'Epsilon'), ('werden', )), 1),
		((('VMFIN', 'Epsilon'), ('muss', )), 1),
		((('VVPP', 'Epsilon'), ('nachgedacht', )), 1)]
	grammar = splitgrammar(rules)
	_, sent = gen(grammar, start=grammar.toid['S'], verbose=True)
	print(' '.join(sent.pop()))


def main():
	"""Load a grammar from a text file and generate 20 sentences."""
	if "--test" in sys.argv:
		test()
		return
	start = "ROOT"
	if "-s" in sys.argv:
		i = sys.argv.index("-s")
		start = sys.argv.pop(i + 1)
		sys.argv[i:i + 2] = []
	verbose = "--verbose" in sys.argv
	if verbose:
		sys.argv.remove('--verbose')
	if len(sys.argv) != 3:
		print("incorrect number of arguments:", sys.argv[1:], file=sys.stderr)
		print(SHORTUSAGE)
		sys.exit(2)
	rules = (gzip.open if sys.argv[1].endswith(".gz") else open)(sys.argv[1])
	lexicon = codecs.getreader('utf-8')((gzip.open
			if sys.argv[2].endswith(".gz") else open)(sys.argv[2]))
	try:
		xgrammar = read_lcfrs_grammar(rules, lexicon)
	except ValueError:
		xgrammar = read_bitpar_grammar(rules, lexicon)
	grammar = splitgrammar(xgrammar)
	for _ in range(20):
		p, sent = gen(grammar, start=grammar.toid[start], verbose=verbose)
		print("[%g] %s" % (p, ' '.join(sent.pop())))


__all__ = ['Grammar', 'LexicalRule', 'Rule', 'gen', 'chooserule', 'compose',
		'parsefrac', 'read_lcfrs_grammar', 'read_bitpar_grammar',
		'splitgrammar', 'yfarray', 'arraytoyf']
