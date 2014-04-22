"""Unit tests for discodop modules."""
# pylint: disable=C0111,W0232
from __future__ import print_function
import re
from unittest import TestCase
from itertools import count, islice
from operator import itemgetter
from discodop.tree import Tree, ParentedTree
from discodop.treebank import incrementaltreereader
from discodop.treetransforms import binarize, unbinarize, \
		splitdiscnodes, mergediscnodes, \
		addbitsets, fanout, canonicalize
from discodop.treebanktransforms import punctraise, balancedpunctraise
from discodop.grammar import flatten, UniqueIDs


class Test_treetransforms(object):
	def test_binarize(self):
		treestr = '(S (VP (PDS 0) (ADV 3) (VVINF 4)) (PIS 2) (VMFIN 1))'
		origtree = Tree(treestr)
		tree = Tree(treestr)
		assert str(binarize(tree, horzmarkov=0, tailmarker='')) == (
				'(S (VP (PDS 0) (VP|<> (ADV 3) (VVINF 4))) (S|<> (PIS 2) '
				'(VMFIN 1)))')
		assert unbinarize(tree) == origtree

		assert str(binarize(tree, horzmarkov=1, tailmarker='')) == (
				'(S (VP (PDS 0) (VP|<ADV> (ADV 3) (VVINF 4))) (S|<PIS> '
				'(PIS 2) (VMFIN 1)))')
		assert unbinarize(tree) == origtree

		assert str(binarize(tree, horzmarkov=1, leftmostunary=False,
				rightmostunary=True, tailmarker='')) == (
				'(S (VP (PDS 0) (VP|<ADV> (ADV 3) (VP|<VVINF> (VVINF 4)))) '
				'(S|<PIS> (PIS 2) (S|<VMFIN> (VMFIN 1))))')
		assert unbinarize(tree) == origtree

		assert str(binarize(tree, horzmarkov=1, leftmostunary=True,
		rightmostunary=False, tailmarker='')) == (
				'(S (S|<VP> (VP (VP|<PDS> (PDS 0) (VP|<ADV> (ADV 3) '
				'(VVINF 4)))) (S|<PIS> (PIS 2) (VMFIN 1))))')
		assert unbinarize(tree) == origtree

		assert str(binarize(tree, factor='left', horzmarkov=2, tailmarker='')
				) == ('(S (S|<PIS,VMFIN> (VP (VP|<ADV,VVINF> (PDS 0) (ADV 3)) '
				'(VVINF 4)) (PIS 2)) (VMFIN 1))')
		assert unbinarize(tree) == origtree

		tree = Tree('(S (A 0) (B 1) (C 2) (D 3) (E 4) (F 5))')
		assert str(binarize(tree, tailmarker='', reverse=False)) == (
				'(S (A 0) (S|<B,C,D,E,F> (B 1) (S|<C,D,E,F> (C 2) (S|<D,E,F> '
				'(D 3) (S|<E,F> (E 4) (F 5))))))')

	def test_mergedicsnodes(self):
		tree = Tree.parse('(S (VP (VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4)'
				'(VVPP 5)) (VAINF 6)) (VMFIN 3))', parse_leaf=int)
		assert str(mergediscnodes(splitdiscnodes(tree))) == (
				'(S (VP (VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) (VVPP 5)) '
				'(VAINF 6)) (VMFIN 3))')

		assert str(mergediscnodes(splitdiscnodes(tree, markorigin=True))) == (
				'(S (VP (VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) (VVPP 5)) '
				'(VAINF 6)) (VMFIN 3))')

		tree = Tree.parse('(S (X (A 0) (A 2)) (X (A 1) (A 3)))', parse_leaf=int)
		assert str(mergediscnodes(splitdiscnodes(tree, markorigin=True))) == (
				'(S (X (A 0) (A 2)) (X (A 1) (A 3)))')

		tree = Tree.parse('(S (X (A 0) (A 2)) (X (A 1) (A 3)))', parse_leaf=int)
		assert str(splitdiscnodes(tree, markorigin=True)) == (
				'(S (X*0 (A 0)) (X*0 (A 1)) (X*1 (A 2)) (X*1 (A 3)))')

		tree = Tree.parse('(S (X (A 0) (A 2)) (X (A 1) (A 3)))', parse_leaf=int)
		assert str(mergediscnodes(splitdiscnodes(tree))) == (
				'(S (X (A 0) (A 1) (A 2) (A 3)))')


class Test_treebank(object):
	def test_incrementaltreereader(self):
		data = '''
		(top (smain (noun 0) (verb 1) (inf (verb 5) (inf (np (det 2)
				(adj 3) (noun 4)) (verb 6) (pp (prep 7) (noun 8))))) (punct 9))
		Het had een prachtige dag kunnen zijn in Londen .
		'''
		result = list(incrementaltreereader([data]))
		assert len(result) == 1
		_tree, sent, _rest = result[0]
		assert sent[0] == u'Het', sent[0]
		assert len(sent) == 10

		data = '''
#BOS 0
is	VB	--	--	500
John	NP	--	--	0
rich	JJ	--	--	500
?	?	--	--	0
#500	VP	--	--	0
#EOS 0
		'''
		result = list(incrementaltreereader(data.splitlines()))
		assert len(result) == 1
		_tree, sent, _rest = result[0]
		assert sent[0] == u'is', sent[0]
		assert len(sent) == 4

		data = '''(S (NP Mary) (VP
			(VB is) (JJ rich)) (. .))'''
		result = list(incrementaltreereader(data.splitlines()))
		assert len(result) == 1


class Test_treebanktransforms(object):
	def test_balancedpunctraise(self):
		tree = ParentedTree.parse('(ROOT ($, 3) ($[ 7) ($[ 13) ($, 14) ($, 20)'
				' (S (NP (ART 0) (ADJA 1) (NN 2) (NP (CARD 4) (NN 5) (PP'
				' (APPR 6) (CNP (NN 8) (ADV 9) (ISU ($. 10) ($. 11)'
				' ($. 12))))) (S (PRELS 15) (MPN (NE 16) (NE 17)) (ADJD 18)'
				' (VVFIN 19))) (VVFIN 21) (ADV 22) (NP (ADJA 23) (NN 24)))'
				' ($. 25))', parse_leaf=int)
		sent = ("Die zweite Konzertreihe , sechs Abende mit ' Orgel plus "
				". . . ' , die Hayko Siemens musikalisch leitet , bietet "
				"wieder ungewoehnliche Kombinationen .".split())
		punctraise(tree, sent)
		balancedpunctraise(tree, sent)
		assert max(map(fanout, addbitsets(tree).subtrees())) == 1

		nopunct = Tree.parse('(ROOT (S (NP (ART 0) (ADJA 1) (NN 2) (NP '
				'(CARD 3) (NN 4) (PP (APPR 5) (CNP (NN 6) (ADV 7)))) (S '
				'(PRELS 8) (MPN (NE 9) (NE 10)) (ADJD 11) (VVFIN 12))) '
				'(VVFIN 13) (ADV 14) (NP (ADJA 15) (NN 16))))', parse_leaf=int)
		assert max(map(fanout, addbitsets(nopunct).subtrees())) == 1


class Test_grammar(object):
	def test_flatten(self):
		ids = UniqueIDs()
		sent = [None, ',', None, '.']
		tree = "(ROOT (S_2 0 2) (ROOT|<$,>_2 ($, 1) ($. 3)))"
		assert flatten(tree, sent, ids, {}, True) == (
				[(('ROOT', 'ROOT}<0>', '$.@.'), ((0, 1),)),
				(('ROOT}<0>', 'S_2', '$,@,'), ((0, 1, 0),)),
				(('$,@,', 'Epsilon'), (',',)), (('$.@.', 'Epsilon'), ('.',))],
				'(ROOT {0} (ROOT|<$,>_2 {1} {2}))')

		assert flatten("(NN 0)", ["foo"], ids, {}, True) == (
				[(('NN', 'Epsilon'), ('foo',))], '(NN 0)')

		prods, frag = flatten(r"(S (S|<VP> (S|<NP> (NP (ART 0) (CNP "
				"(CNP|<TRUNC> (TRUNC 1) (CNP|<KON> (KON 2) (CNP|<NN> "
				"(NN 3)))))) (S|<VAFIN> (VAFIN 4))) (VP (VP|<ADV> (ADV 5) "
				"(VP|<NP> (NP (ART 6) (NN 7)) (VP|<NP> (NP_2 8 10) (VP|<VVPP> "
				"(VVPP 9))))))))",
				['Das', 'Garten-', 'und', 'Friedhofsamt', 'hatte', 'kuerzlich',
				'dem', 'Ortsbeirat', None, None, None], ids, {}, True)
		assert prods == [(('S', 'S}<8>_2', 'VVPP'), ((0, 1, 0),)),
				(('S}<8>_2', 'S}<7>', 'NP_2'), ((0, 1), (1,))),
				(('S}<7>', 'S}<6>', 'NN@Ortsbeirat'), ((0, 1),)),
				(('S}<6>', 'S}<5>', 'ART@dem'), ((0, 1),)),
				(('S}<5>', 'S}<4>', 'ADV@kuerzlich'), ((0, 1),)),
				(('S}<4>', 'S}<3>', 'VAFIN@hatte'), ((0, 1),)),
				(('S}<3>', 'S}<2>', 'NN@Friedhofsamt'), ((0, 1),)),
				(('S}<2>', 'S}<1>', 'KON@und'), ((0, 1),)),
				(('S}<1>', 'ART@Das', 'TRUNC@Garten-'), ((0, 1),)),
				(('ART@Das', 'Epsilon'), ('Das',)),
				(('TRUNC@Garten-', 'Epsilon'), ('Garten-',)),
				(('KON@und', 'Epsilon'), ('und',)),
				(('NN@Friedhofsamt', 'Epsilon'), ('Friedhofsamt',)),
				(('VAFIN@hatte', 'Epsilon'), ('hatte',)),
				(('ADV@kuerzlich', 'Epsilon'), ('kuerzlich',)),
				(('ART@dem', 'Epsilon'), ('dem',)),
				(('NN@Ortsbeirat', 'Epsilon'), ('Ortsbeirat',))]
		assert frag == (
				'(S (S|<VP> (S|<NP> (NP {0} (CNP (CNP|<TRUNC> {1} (CNP|<KON> '
				'{2} (CNP|<NN> {3}))))) (S|<VAFIN> {4})) (VP (VP|<ADV> {5} '
				'(VP|<NP> (NP {6} {7}) (VP|<NP> {8} (VP|<VVPP> {9})))))))')

		assert flatten("(S|<VP>_2 (VP_3 (VP|<NP>_3 (NP 0) (VP|<ADV>_2 "
				"(ADV 2) (VP|<VVPP> (VVPP 4))))) (S|<VAFIN> (VAFIN 1)))",
				(None, None, None, None, None), ids, {}, True) == (
				[(('S|<VP>_2', 'S|<VP>_2}<10>', 'VVPP'), ((0,), (1,))),
				(('S|<VP>_2}<10>', 'S|<VP>_2}<9>', 'ADV'), ((0, 1),)),
				(('S|<VP>_2}<9>', 'NP', 'VAFIN'), ((0, 1),))],
				'(S|<VP>_2 (VP_3 (VP|<NP>_3 {0} (VP|<ADV>_2 {2} (VP|<VVPP> '
				'{3})))) (S|<VAFIN> {1}))')


class TestHeap(TestCase):
	testN = 100

	def check_invariants(self, h):
		from discodop.plcfrs import getparent
		for i in range(len(h)):
			if i > 0:
				self.assertTrue(h.heap[getparent(i)].getvalue()
						<= h.heap[i].getvalue())

	def make_data(self):
		from random import random
		from discodop.plcfrs import Agenda
		pairs = [(random(), random()) for _ in range(TestHeap.testN)]
		h = Agenda()
		d = {}
		for k, v in pairs:
			h[k] = v
			d[k] = v

		pairs.sort(key=itemgetter(1), reverse=True)
		return h, pairs, d

	def test_contains(self):
		h, pairs, d = self.make_data()
		h, pairs2, d = self.make_data()
		for k, _ in pairs + pairs2:
			self.assertEqual(k in h, k in d)

	def test_len(self):
		h, _, d = self.make_data()
		self.assertEqual(len(h), len(d))

	def test_popitem(self):
		h, pairs, d = self.make_data()
		while pairs:
			v = h.popitem()
			v2 = pairs.pop(-1)
			self.assertEqual(v, v2)
			d.pop(v[0])
			self.assertEqual(len(h), len(d))
			self.assertTrue(set(h.items()) == set(d.items()))
		self.assertEqual(len(h), 0)

	def test_popitem_ties(self):
		from discodop.plcfrs import Agenda
		h = Agenda()
		for i in range(TestHeap.testN):
			h[i] = 0.
		for i in range(TestHeap.testN):
			_, v = h.popitem()
			self.assertEqual(v, 0.)
			self.check_invariants(h)

	def test_popitem_ties_fifo(self):
		from discodop.plcfrs import Agenda
		h = Agenda()
		for i in range(TestHeap.testN):
			h[i] = 0.
		for i in range(TestHeap.testN):
			k, v = h.popitem()
			self.assertEqual(k, i)
			self.assertEqual(v, 0.)
			self.check_invariants(h)

	def test_peek(self):
		h, pairs, _ = self.make_data()
		while pairs:
			v = h.peekitem()[0]
			h.popitem()
			v2 = pairs.pop(-1)
			self.assertEqual(v, v2[0])
		self.assertEqual(len(h), 0)

	def test_iter(self):
		h, _, d = self.make_data()
		self.assertEqual(list(h), list(d))

	def test_keys(self):
		h, _, d = self.make_data()
		self.assertEqual(sorted(h.keys()), sorted(d.keys()))

	def test_values(self):
		h, _, d = self.make_data()
		self.assertEqual(sorted(h.values()), sorted(d.values()))

	def test_items(self):
		h, _, d = self.make_data()
		self.assertEqual(sorted(h.items()), sorted(d.items()))

	def test_del(self):
		h, pairs, d = self.make_data()
		while pairs:
			k, _ = pairs.pop(len(pairs) // 2)
			del h[k]
			del d[k]
			self.assertEqual(k in h, False)
			self.assertEqual(k in d, False)
			self.assertEqual(len(h), len(d))
			self.assertTrue(set(h.items()) == set(d.items()))
		self.assertEqual(len(h), 0)

	def test_pop(self):
		h, pairs, d = self.make_data()
		while pairs:
			k, v = pairs.pop(-1)
			v2 = h.pop(k)
			v3 = d.pop(k)
			self.assertEqual(v, v2)
			self.assertEqual(v, v3)
			self.assertEqual(len(h), len(d))
			self.assertTrue(set(h.items()) == set(d.items()))
		self.assertEqual(len(h), 0)

	def test_change(self):
		h, pairs, _ = self.make_data()
		k, v = pairs[TestHeap.testN // 2]
		h[k] = 0.5
		pairs[TestHeap.testN // 2] = (k, 0.5)
		pairs.sort(key=itemgetter(1), reverse=True)
		while pairs:
			v = h.popitem()
			v2 = pairs.pop()
			self.assertEqual(v, v2)
		self.assertEqual(len(h), 0)

	def test_init(self):
		from discodop.plcfrs import Agenda
		h, pairs, d = self.make_data()
		h = Agenda(d.items())
		while pairs:
			v = h.popitem()
			v2 = pairs.pop()
			self.assertEqual(v, v2)
			d.pop(v[0])
		self.assertEqual(len(h), len(d))
		self.assertEqual(len(h), 0)

	def test_repr(self):
		h, _pairs, d = self.make_data()
		#self.assertEqual(h, eval(repr(h)))
		tmp = repr(h)  # 'Agenda({....})'
		#strip off class name
		dstr = tmp[tmp.index('(') + 1:tmp.rindex(')')]
		self.assertEqual(d, eval(dstr))


def test_fragments():
	from discodop._fragments import getctrees, \
			fastextractfragments, exactcounts
	treebank = [binarize(Tree(x)) for x in """\
(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (JJ hungry) (NN dog))))
(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (JJ yellow) (NN cat))))
(S (NP (DT The) (JJ little) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN cat)) (VP (VBP ate) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP ate) (NP (DT the) (NN cat))))\
		""".splitlines()]
	sents = [tree.leaves() for tree in treebank]
	for tree in treebank:
		for n, idx in enumerate(tree.treepositions('leaves')):
			tree[idx] = n
	params = getctrees(treebank, sents)
	fragments = fastextractfragments(params['trees1'], params['sents1'],
			0, 0, params['labels'], discontinuous=True, approx=False)
	counts = exactcounts(params['trees1'], params['trees1'],
			list(fragments.values()), fast=True)
	assert len(fragments) == 25
	assert sum(counts) == 100
	for (a, b), c in sorted(zip(fragments, counts), key=repr):
		print("%s\t%d" % (re.sub("[0-9]+", lambda x: b[int(x.group())], a), c))


def test_grammar(debug=False):
	"""Demonstrate grammar extraction."""
	from discodop.grammar import treebankgrammar, dopreduction, doubledop
	from discodop import plcfrs
	from discodop.containers import Grammar
	from discodop.treebank import NegraCorpusReader
	from discodop.treetransforms import addfanoutmarkers, removefanoutmarkers
	from discodop.disambiguation import recoverfragments
	from discodop.kbest import lazykbest
	from math import exp
	corpus = NegraCorpusReader('alpinosample.export', punct='move')
	sents = list(corpus.sents().values())
	trees = [addfanoutmarkers(binarize(a.copy(True), horzmarkov=1))
			for a in list(corpus.trees().values())[:10]]
	print('plcfrs\n', Grammar(treebankgrammar(trees, sents)))
	print('dop reduction')
	grammar = Grammar(dopreduction(trees[:2], sents[:2])[0],
			start=trees[0].label)
	print(grammar)
	_ = grammar.testgrammar()

	grammarx, backtransform, _, _ = doubledop(trees, sents,
			debug=debug, numproc=1)
	print('\ndouble dop grammar')
	grammar = Grammar(grammarx, start=trees[0].label)
	grammar.getmapping(grammar, striplabelre=None,
			neverblockre=re.compile(b'^#[0-9]+|.+}<'),
			splitprune=False, markorigin=False)
	print(grammar)
	assert grammar.testgrammar()[0], "RFE should sum to 1."
	for tree, sent in zip(corpus.trees().values(), sents):
		print("sentence:", ' '.join(a.encode('unicode-escape').decode()
				for a in sent))
		chart, msg = plcfrs.parse(sent, grammar, exhaustive=True)
		print('\n', msg, '\ngold ', tree, '\n', 'double dop', end='')
		if chart:
			mpp, parsetrees = {}, {}
			derivations, _ = lazykbest(chart, 1000, b'}<')
			for d, (t, p) in zip(chart.rankededges[chart.root()], derivations):
				r = Tree(recoverfragments(d.getkey(), chart, backtransform))
				r = str(removefanoutmarkers(unbinarize(r)))
				mpp[r] = mpp.get(r, 0.0) + exp(-p)
				parsetrees.setdefault(r, []).append((t, p))
			print(len(mpp), 'parsetrees', end='')
			print(sum(map(len, parsetrees.values())), 'derivations')
			for t, tp in sorted(mpp.items(), key=itemgetter(1)):
				print(tp, '\n', t, end='')
				print("match:", t == str(tree))
				assert len(set(parsetrees[t])) == len(parsetrees[t])
				if debug:
					for deriv, p in sorted(parsetrees[t], key=itemgetter(1)):
						print(' <= %6g %s' % (exp(-p), deriv))
		else:
			print('no parse\n', chart)
		print()
	tree = Tree.parse("(ROOT (S (F (E (S (C (B (A 0))))))))", parse_leaf=int)
	Grammar(treebankgrammar([tree], [[str(a) for a in range(10)]]))


def test_optimalbinarize():
	"""Verify that all optimal parsing complexities are lower than or
	equal to the complexities of right-to-left binarizations."""
	from discodop.treetransforms import optimalbinarize, complexityfanout
	from discodop.treebank import NegraCorpusReader
	corpus = NegraCorpusReader('alpinosample.export', punct='move')
	total = violations = violationshd = 0
	for n, (tree, sent) in enumerate(zip(list(
			corpus.trees().values())[:-2000], corpus.sents().values())):
		t = addbitsets(tree)
		if all(fanout(x) == 1 for x in t.subtrees()):
			continue
		print(n, tree, '\n', ' '.join(sent))
		total += 1
		optbin = optimalbinarize(tree.copy(True), headdriven=False, h=None, v=1)
		# undo head-ordering to get a normal right-to-left binarization
		normbin = addbitsets(binarize(canonicalize(Tree.convert(tree))))
		if (max(map(complexityfanout, optbin.subtrees()))
				> max(map(complexityfanout, normbin.subtrees()))):
			print('non-hd\n', tree)
			print(max(map(complexityfanout, optbin.subtrees())), optbin)
			print(max(map(complexityfanout, normbin.subtrees())), normbin, '\n')
			violations += 1

		optbin = optimalbinarize(tree.copy(True), headdriven=True, h=1, v=1)
		normbin = addbitsets(binarize(Tree.convert(tree), horzmarkov=1))
		if (max(map(complexityfanout, optbin.subtrees()))
				> max(map(complexityfanout, normbin.subtrees()))):
			print('hd\n', tree)
			print(max(map(complexityfanout, optbin.subtrees())), optbin)
			print(max(map(complexityfanout, normbin.subtrees())), normbin, '\n')
			violationshd += 1
	print('opt. bin. violations normal: %d / %d;  hd: %d / %d' % (
			violations, total, violationshd, total))
	assert violations == violationshd == 0


def test_splitdisc():
	"""Verify that splitting and merging discontinuities gives the same
	trees."""
	from discodop.treebank import NegraCorpusReader
	correct = wrong = 0
	corpus = NegraCorpusReader('alpinosample.export')
	for tree in corpus.trees().values():
		if mergediscnodes(splitdiscnodes(tree)) == tree:
			correct += 1
		else:
			wrong += 1
	total = len(corpus.sents())
	print('disc. split-merge: correct', correct, '=', 100. * correct / total, '%')
	print('disc. split-merge: wrong', wrong, '=', 100. * wrong / total, '%')
	assert wrong == 0


def test_eval():
	"""Simple sanity check; should give 100% score on all metrics."""
	from discodop.treebank import READERS
	from discodop.eval import Evaluator, readparam
	gold = READERS['export']('alpinosample.export')
	parses = READERS['export']('alpinosample.export')
	goldtrees, goldsents, candsents = gold.trees(), gold.sents(), parses.sents()
	evaluator = Evaluator(readparam(None))
	for n, ctree in parses.trees().items():
		evaluator.add(n, goldtrees[n], goldsents[n], ctree, candsents[n])
	evaluator.breakdowns()
	print(evaluator.summary())


def test_punct():
	"""Verify that punctuation movement does not increase fan-out."""
	from discodop.treebank import NegraCorpusReader
	filename = 'alpinosample.export'
	mangledtrees = NegraCorpusReader(filename, punct='move')
	nopunct = list(NegraCorpusReader(filename,
			punct='remove').trees().values())
	originals = list(NegraCorpusReader(filename, headrules=None,
			encoding='iso-8859-1').trees().values())
	phrasal = lambda x: len(x) and isinstance(x[0], Tree)
	for n, mangled, sent, nopunct, original in zip(count(),
			mangledtrees.trees().values(),
			mangledtrees.sents().values(), nopunct, originals):
		print(n, end='')
		for a, b in zip(sorted(addbitsets(mangled).subtrees(phrasal),
				key=lambda n: min(n.leaves())),
				sorted(addbitsets(nopunct).subtrees(phrasal),
				key=lambda n: min(n.leaves()))):
			if fanout(a) != fanout(b):
				print(' '.join(sent))
				print(mangled)
				print(nopunct)
				print(original)
			assert fanout(a) == fanout(b), '%d %d\n%s\n%s' % (
				fanout(a), fanout(b), a, b)
	print()


def test_transforms():
	"""Test reversibility of Tiger transformations."""
	from discodop.treebanktransforms import transform, reversetransform, \
			bracketings
	from discodop.treebank import NegraCorpusReader, handlefunctions
	headrules = None  # 'alpino.headrules'
	n = NegraCorpusReader('alpinosample.export', headrules=headrules)
	nn = NegraCorpusReader('alpinosample.export', headrules=headrules)
	transformations = ('S-RC', 'VP-GF', 'NP')
	trees = [transform(tree, sent, transformations)
			for tree, sent in zip(nn.trees().values(),
				nn.sents().values())]
	print('\ntransformed')
	correct = exact = d = 0
	for a, b, c in islice(zip(n.trees().values(),
			trees, n.sents().values()), 100):
		transformb = reversetransform(b.copy(True), transformations)
		b1 = bracketings(canonicalize(a))
		b2 = bracketings(canonicalize(transformb))
		z = -1  # 825
		if b1 != b2 or d == z:
			precision = len(set(b1) & set(b2)) / len(set(b1))
			recall = len(set(b1) & set(b2)) / len(set(b2))
			if precision != 1.0 or recall != 1.0 or d == z:
				print(d, ' '.join(':'.join((str(n),
					a.encode('unicode-escape'))) for n, a in enumerate(c)))
				print('no match', precision, recall)
				print(len(b1), len(b2), 'gold-transformed', set(b2) - set(b1),
						'transformed-gold', set(b1) - set(b2))
				print(a)
				print(transformb)
				handlefunctions('add', a)
				print(a, '\n', b, '\n\n')
			else:
				correct += 1
		else:
			exact += 1
			correct += 1
		d += 1
	print('matches', correct, '/', d, 100 * correct / d, '%')
	print('exact', exact)
