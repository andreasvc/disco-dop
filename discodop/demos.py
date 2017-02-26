"""Examples of various formalisms encoded in LCFRS grammars."""
from __future__ import print_function, absolute_import
from math import exp
from . import treetransforms, plcfrs, kbest
from .tree import Tree
from .grammar import treebankgrammar
from .containers import Grammar


def tree_adjoining_grammar():
	"""Example of a tree-adjoining grammar (TAG) encoded as an LCFRS.

	Taken from: Chen & Vijay-Shanker (IWPT 2000), Automated extraction of TAGs
	from the Penn treebank. http://nlp.cs.nyu.edu/nycnlp/autoextract.ps

	- no epsilon productions
	- non-terminals have identifiers to encode elementary trees of depth > 2.
	"""
	print("Tree-Adjoining Grammars in LCFRS")
	print('''initial trees:
(S (NP ) (VP (V fell)))
(NP (NN prices))
auxiliary trees:
(S (ADVP (RB Later) (S* ))
(VP (ADVP (RB drastically)) (VP* ))''')
	grammar = Grammar([
			((('ADVP#1', 'RB#1'), ((0, ), )), 1),
			((('ADVP#2', 'RB#2'), ((0, ), )), 1),
			((('NP', 'NN#1'), ((0, ), )), 1),
			((('ROOT', 'S'), ((0, ), )), 1),
			((('S', 'NP', 'VP'), ((0, 1), )), 1),
			((('S', 'ADVP#1', 'S'), ((0, 1), )), 1),
			((('VP', 'V#1'), ((0, ), )), 1),
			((('VP', 'ADVP#2', 'VP'), ((0, 1), )), 1),
			((('RB#1', 'Epsilon'), ('Later', )), 1),
			((('NN#1', 'Epsilon'), ('prices', )), 1),
			((('V#1', 'Epsilon'), ('fell', )), 1),
			((('RB#2', 'Epsilon'), ('drastically', )), 1)])
	print(grammar)
	assert parse(grammar, "prices fell".split())
	assert parse(grammar, "prices drastically fell".split())
	assert parse(grammar, "Later prices fell".split())
	assert parse(grammar, "Later prices drastically fell".split())

	# taken from: slides for course Grammar Formalisms, Kallmeyer (2011),
	# Mildly Context-Sensitive Grammar Formalisms:
	# LCFRS: Relations to other Formalisms
	# https://user.phil.hhu.de/~kallmeyer/GrammarFormalisms/4lcfrs-related-formalisms.pdf
	print("the language {d} + {a**n b**m c**m d **n} with n>0, m>=0")
	print('''initial trees:
(S a (S Epsilon) F)
(F d)
auxiliary trees:
(S b S* c)''')
	grammar = Grammar([
			((('ROOT', 'a1'), ((0, ), )), 1),
			((('ROOT', 'a2'), ((0, ), )), 1),
			((('a1', 'a_b', 'a2'), ((0, 1), )), 1),
			((('a1', '_a', 'a2'), ((0, 1), )), 1),
			((('a2', '_d'), ((0, ), )), 1),
			((('a_b', '_a', 'b'), ((0, 1), )), 1),
			((('b', '_b', '_c'), ((0, 1), )), 1),
			((('b', 'b_2', 'b'), ((0, 1, 0), )), 1),
			((('b_2', '_b', '_c'), ((0, ), (1, ))), 1),
			((('_a', 'Epsilon'), ('a', )), 1),
			((('_b', 'Epsilon'), ('b', )), 1),
			((('_c', 'Epsilon'), ('c', )), 1),
			((('_d', 'Epsilon'), ('d', )), 1)])
	print(grammar)
	assert parse(grammar, list("d"))
	assert parse(grammar, list("ad"))
	assert parse(grammar, list("abcd"))
	assert parse(grammar, list("abbccd"))
	print("wrong:")
	assert not parse(grammar, list("abbbccd"))

	# Taken from: Boullier (1998), Generalization of Mildly
	# Context-Sensitive Formalisms. http://aclweb.org/anthology/W98-0105
	# Epsilon replaced with '|', added preterminal rules w/underscores
	print("the language { ww | w in {a,b}* }")
	print('''initial trees:
		(S (A Epsilon))
		auxiliary trees:
		(A a (A A*) a)
		(A b (A A*) b)
		(A (A A*))''')
	grammar = Grammar([
			((('_aa', '_a', '_a'), ((0, ), (1, ))), 1),
			((('_bb', '_b', '_b'), ((0, ), (1, ))), 1),
			((('A', '_aa', 'A'), ((0, 1), (0, 1))), 1),
			((('A', '_bb', 'A'), ((0, 1), (0, 1))), 1),
			((('A', '_aa'), ((0, ), (0, ))), 1),
			((('A', '_bb'), ((0, ), (0, ))), 1),
			((('ROOT', '_|'), ((0, ), )), 1),
			((('ROOT', 'A', '_|'), ((0, 1, 0), )), 1),
			((('_a', 'Epsilon'), ('a', )), 1),
			((('_b', 'Epsilon'), ('b', )), 1),
			((('_|', 'Epsilon'), ('|', )), 1)])
	print(grammar)
	assert parse(grammar, list("a|a"))
	assert parse(grammar, list("ab|ab"))
	assert parse(grammar, list("abaab|abaab"))
	print("wrong:")
	assert not parse(grammar, list("a|b"))
	assert not parse(grammar, list("aa|bb"))


def dependencygrammar():
	"""An example dependency structure encoded in an LCFRS grammar.

	Taken from: Gildea (2010, fig. 4), Optimal Parsing Strategies for Linear
	Context-Free Rewriting Systems. http://aclweb.org/anthology/N10-1118

	- rules have to be binarized
	- lexical rules have to be unary

	These have been dealt with by introducing nodes w/underscores."""
	print("A dependency grammar in an LCFRS:")
	grammar = Grammar([
			((('NMOD', '_A'), ((0, ), )), 1),
			((('NMOD', '_the'), ((0, ), )), 1),
			((('NMOD_hearing', 'NMOD', '_hearing'), ((0, 1), )), 1),
			((('NP', 'NMOD', '_issue'), ((0, 1), )), 1),
			((('PP', '_on', 'NP'), ((0, 1, ), )), 1),
			((('ROOT', 'SBJ', 'is_VC'), ((0, 1, 0, 1), )), 1),
			((('SBJ', 'NMOD_hearing', 'PP'), ((0, ), (1, ))), 1),
			((('TMP', '_today'), ((0, ), )), 1),
			((('VC', '_scheduled', 'TMP'), ((0, ), (1, ))), 1),
			((('is_VC', '_is', 'VC'), ((0, 1), (1, ))), 1),
			((('_A', 'Epsilon'), ('A', )), 1),
			((('_hearing', 'Epsilon'), ('hearing', )), 1),
			((('_is', 'Epsilon'), ('is', )), 1),
			((('_scheduled', 'Epsilon'), ('scheduled', )), 1),
			((('_on', 'Epsilon'), ('on', )), 1),
			((('_the', 'Epsilon'), ('the', )), 1),
			((('_issue', 'Epsilon'), ('issue', )), 1),
			((('_today', 'Epsilon'), ('today', )), 1)])
	print(grammar)
	testsent = "A hearing is scheduled on the issue today".split()
	assert parse(grammar, testsent)


def bitext():
	"""Bitext parsing with a synchronous CFG.

	Translation would require a special decoder (instead of normal k-best
	derivations where the whole sentence is given)."""
	print("bitext parsing with a synchronous CFG")
	trees = [Tree(a) for a in '''\
	(ROOT (S (NP (NNP (John 0) (John 7))) (VP (VB (misses 1) (manque 5))\
		(PP (IN (a` 6)) (NP (NNP (Mary 2) (Mary 4)))))) (SEP (| 3)))
	(ROOT (S (NP (NNP (Mary 0) (Mary 4))) (VP (VB (likes 1) (aimes 5))\
		(NP (DT (la 6)) (NN (pizza 2) (pizza 7))))) (SEP (| 3)))'''.split('\n')]
	sents = [["0"] * len(a.leaves()) for a in trees]
	for a in trees:
		treetransforms.binarize(a)
	compiled_scfg = Grammar(treebankgrammar(trees, sents))
	print("sentences:")
	for tree in trees:
		print(' '.join(w for _, w in sorted(tree.pos())))
	print("treebank:")
	for tree in trees:
		print(tree)
	print(compiled_scfg, "\n")

	print("correct translations:")
	assert parse(compiled_scfg, ["0"] * 7,
			"John likes Mary | John aimes Mary".split())
	assert parse(compiled_scfg, ["0"] * 9,
			"John misses pizza | la pizza manque a` John".split())

	print("incorrect translations:")
	assert not parse(compiled_scfg, ["0"] * 7,
			"John likes Mary | Mary aimes John".split())
	assert not parse(compiled_scfg, ["0"] * 9,
			"John misses pizza | John manque a` la pizza".split())

	# the following SCFG is taken from:
	# http://cdec-decoder.org/index.php?title=SCFG_translation
	# the grammar has been binarized and some new non-terminals had to be
	# introduced because terminals cannot appear in binary rules.
	lexicon = ("|", "ein", "ich", "Haus", "kleines", "grosses", "sah", "fand",
		"small", "little", "big", "large", "house", "shell", "a", "I",
		"saw", "found")
	another_scfg = Grammar([
			((('DT', '_ein', '_a'), ((0, ), (1, ))), 0.5),
			((('JJ', '_kleines', '_small'), ((0, ), (1, ))), 0.1),
			((('JJ', '_kleines', '_little'), ((0, ), (1, ))), 0.9),
			((('JJ', '_grosses', '_big'), ((0, ), (1, ))), 0.8),
			((('JJ', '_grosses', '_large'), ((0, ), (1, ))), 0.2345),
			((('NN_house', '_Haus', '_house'), ((0, ), (1, ))), 1),
			((('NN_shell', '_Haus', '_shell'), ((0, ), (1, ))), 1),
			((('NP', '_ich', '_I'), ((0, ), (1, ), )), 0.6),
			((('NP', 'DT', 'NP|<JJ-NN>'), ((0, 1), (0, 1))), 0.5),
			((('NP|<JJ-NN>', 'JJ', 'NN_house'), ((0, 1), (0, 1))), 0.1),
			((('NP|<JJ-NN>', 'JJ', 'NN_shell'), ((0, 1), (0, 1))), 1.3),
			((('ROOT', 'S', '_|'), ((0, 1, 0), )), 1),
			((('S', 'NP', 'VP'), ((0, 1), (0, 1))), 0.2),
			((('VP', 'V', 'NP'), ((0, 1), (0, 1))), 0.1),
			((('V', '_sah', '_saw'), ((0, ), (1, ))), 0.4),
			((('V', '_fand', '_found'), ((0, ), (1, ))), 0.4)]
			+ [((('_%s' % word, 'Epsilon'), (word, )), 1)
					for word in lexicon])
	print(another_scfg)
	sents = [
		"ich sah ein kleines Haus | I saw a small house".split(),
		"ich sah ein kleines Haus | I saw a little house".split(),
		"ich sah ein kleines Haus | I saw a small shell".split(),
		"ich sah ein kleines Haus | I saw a little shell".split()]
	for sent in sents:
		assert parse(another_scfg, sent), sent


def parse(compiledgrammar, testsent, testtags=None):
	"""Parse a sentence with a grammar."""
	chart, _ = plcfrs.parse(testsent,
		compiledgrammar, tags=testtags, exhaustive=True)
	print("input:", ' '.join("%d:%s" % a
			for a in enumerate(testtags if testtags else testsent)), end=' ')
	if chart:
		print()
		results = kbest.lazykbest(chart, 10)[0]
		for tree, prob in results:
			tree = Tree(tree)
			treetransforms.unbinarize(tree)
			print(exp(-prob), tree)
		print()
		return True
	else:
		print("no parse!\n")
		print(chart)
		return False


def test():
	"""Alias for main()."""
	main()


def main():
	"""Run all examples."""
	dependencygrammar()
	tree_adjoining_grammar()
	bitext()


__all__ = ['tree_adjoining_grammar', 'dependencygrammar', 'bitext']
