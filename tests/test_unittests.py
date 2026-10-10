"""Unit tests for discodop modules."""
# pylint: disable=C0111,W0232
from __future__ import division, print_function, absolute_import, \
		unicode_literals
import os
import re
import pickle
from unittest import TestCase
from itertools import count, islice
from operator import itemgetter
from discodop.tree import Tree, ParentedTree, HEAD
from discodop.treebank import incrementaltreereader
from discodop.treetransforms import (binarize, unbinarize, canonicalize,
		splitdiscnodes, mergediscnodes, addbitsets, fanout)
from discodop.punctuation import punctraise, balancedpunctraise
from discodop.grammar import flatten, UniqueIDs
from discodop.fragments import readtreebanks
from discodop.containers import Ctrees, Vocabulary, FixedVocabulary


class Test_treetransforms(object):
	def test_binarize(self):
		treestr = '(S (VP (PDS 0) (ADV 3) (VVINF 4)) (VMFIN 1) (PIS 2))'
		origtree = Tree(treestr)
		tree = Tree(treestr)
		tree[1].type = HEAD  # set VMFIN as head
		assert str(binarize(tree, horzmarkov=0)) == (
				'(S (VP (PDS 0) (VP|<> (ADV 3) (VVINF 4))) (S|<> (VMFIN 1)'
				' (PIS 2)))')
		assert unbinarize(tree) == origtree

		assert str(binarize(tree, horzmarkov=1)) == (
				'(S (VP (PDS 0) (VP|<ADV> (ADV 3) (VVINF 4))) (S|<VMFIN> '
				'(VMFIN 1) (PIS 2)))')
		assert unbinarize(tree) == origtree

		assert str(binarize(tree, horzmarkov=1, leftmostunary=False,
				rightmostunary=True, headoutward=True)) == (
				'(S (VP (PDS 0) (VP|<ADV> (ADV 3) (VP|<VVINF> (VVINF 4)))) '
				'(S|<VMFIN> (S|<VMFIN> (VMFIN 1)) (PIS 2)))')
		assert unbinarize(tree) == origtree

		assert str(binarize(tree, horzmarkov=1, leftmostunary=True,
		rightmostunary=False, headoutward=True)) == (
				'(S (S|<VP> (VP (VP|<PDS> (PDS 0) (VP|<ADV> (ADV 3) '
				'(VVINF 4)))) (S|<VMFIN> (VMFIN 1) (PIS 2))))')
		assert unbinarize(tree) == origtree

		assert str(binarize(tree, factor='left', horzmarkov=2,
				headoutward=True)
				) == ('(S (S|<VMFIN,PIS> (VP (VP|<PDS,ADV> (PDS 0) (ADV 3)) '
				'(VVINF 4)) (VMFIN 1)) (PIS 2))')
		assert unbinarize(tree) == origtree

		tree = Tree('(S (A 0) (B 1) (C 2) (D 3) (E 4) (F 5))')
		assert str(binarize(tree, headoutward=True)) == (
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
		(top (smain (noun 0=Het) (verb 1=had) (inf (verb 5=kunnen)
				(inf (np (det 2=een) (adj 3=prachtige) (noun 4=dag))
				(verb 6=zijn) (pp (prep 7=in) (noun 8=Londen))))) (punct 9=.))
		'''
		result = list(incrementaltreereader([data]))
		assert len(result) == 1
		_tree, sent, _rest = result[0]
		assert sent[0] == u'Het', sent[0]
		assert len(sent) == 10
		assert sent == (
				u'Het had een prachtige dag kunnen zijn in Londen .'.split())

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

	def test_transform(self):
		from discodop.treebanktransforms import transform, reversetransform, \
				bracketings
		from discodop.treebank import NegraCorpusReader
		n = NegraCorpusReader('alpinosample.export')
		for transformations in (
				('FUNC-NODE', ),
				('MORPH-NODE', ),
				('LEMMA-NODE', ),
				('FUNC-NODE', 'MORPH-NODE', 'LEMMA-NODE')):
			nn = NegraCorpusReader('alpinosample.export')
			trees = [transform(tree, sent, transformations)
					for tree, sent in zip(nn.trees().values(),
						nn.sents().values())]
			for sent, a, b in islice(zip(
					n.sents().values(), n.trees().values(), trees), 100):
				before = bracketings(canonicalize(a))
				transformb = reversetransform(
						b.copy(True), sent, transformations)
				after = bracketings(canonicalize(transformb))
				assert before == after, (
						'mismatch with %r\nbefore: %r\nafter: %r' % (
						transformations, before, after))


class Test_grammar(object):
	def test_flatten(self):
		ids = UniqueIDs()
		assert flatten(
				'(ROOT (S_2 0= 2=) (ROOT|<$,>_2 ($, 1=,) ($. 3=.)))',
				ids) == (
				[(('ROOT', 'ROOT}<0>', '$.@.'), ((0, 1),)),
				(('ROOT}<0>', 'S_2', '$,@,'), ((0, 1, 0),)),
				(('$,@,', 'Epsilon'), (',',)), (('$.@.', 'Epsilon'), ('.',))],
				'(ROOT {0} (ROOT|<$,>_2 {1} {2}))')

		assert flatten("(NN 0=foo)", ids) == (
				[(('NN', 'Epsilon'), ('foo',))], '(NN 0)')

		prods, frag = flatten(r"(S (S|<VP> (S|<NP> (NP (ART 0=Das) (CNP "
				"(CNP|<TRUNC> (TRUNC 1=Garten-) (CNP|<KON> (KON 2=und) "
				"(CNP|<NN> (NN 3=Friedhofsamt)))))) (S|<VAFIN> "
				"(VAFIN 4=hatte))) (VP (VP|<ADV> (ADV 5=kuerzlich) "
				"(VP|<NP> (NP (ART 6=dem) (NN 7=Ortsbeirat)) (VP|<NP> "
				"(NP_2 8= 10=) (VP|<VVPP> (VVPP 9=))))))))",
				ids)
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

		assert flatten("(S|<VP>_2 (VP_3 (VP|<NP>_3 (NP 0=) (VP|<ADV>_2 "
				"(ADV 2=) (VP|<VVPP> (VVPP 4=))))) (S|<VAFIN> (VAFIN 1=)))",
				ids) == (
				[(('S|<VP>_2', 'S|<VP>_2}<10>', 'VVPP'), ((0,), (1,))),
				(('S|<VP>_2}<10>', 'S|<VP>_2}<9>', 'ADV'), ((0, 1),)),
				(('S|<VP>_2}<9>', 'NP', 'VAFIN'), ((0, 1),))],
				'(S|<VP>_2 (VP_3 (VP|<NP>_3 {0} (VP|<ADV>_2 {2} (VP|<VVPP> '
				'{3})))) (S|<VAFIN> {1}))')


class TestHeap(TestCase):
	testN = 100

	def make_data(self):
		from random import random
		from discodop.util import PyAgenda
		pairs = [(random(), random()) for _ in range(TestHeap.testN)]
		h = PyAgenda()
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
		from discodop.util import PyAgenda
		h = PyAgenda()
		for i in range(TestHeap.testN):
			h[i] = 0.
		for i in range(TestHeap.testN):
			_, v = h.popitem()
			self.assertEqual(v, 0.)

	def test_popitem_ties_fifo(self):
		from discodop.util import PyAgenda
		h = PyAgenda()
		for i in range(TestHeap.testN):
			h[i] = 0.
		for i in range(TestHeap.testN):
			k, v = h.popitem()
			self.assertEqual(k, i)
			self.assertEqual(v, 0.)

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
		from discodop.util import PyAgenda
		h, pairs, d = self.make_data()
		h = PyAgenda(d.items())
		while pairs:
			v = h.popitem()
			v2 = pairs.pop()
			self.assertEqual(v, v2)
			d.pop(v[0])
		self.assertEqual(len(h), len(d))
		self.assertEqual(len(h), 0)

	def test_init_small(self):
		from discodop.util import PyAgenda
		for data in (
				[(0, 3), (1, 7), (2, 1)],
				[(0, 7), (1, 3), (2, 1)],
				[(0, 7), (1, 3), (2, 7)]):
			h = PyAgenda(data)
			self.assertEqual(
					[h.popitem(), h.popitem(), h.popitem()],
					sorted(data, key=itemgetter(1)))
			self.assertEqual(len(h), 0)

	def test_repr(self):
		h, _pairs, d = self.make_data()
		# self.assertEqual(h, eval(repr(h)))
		tmp = repr(h)  # 'PyAgenda({....})'
		# strip off class name
		dstr = tmp[tmp.index('(') + 1:tmp.rindex(')')]
		self.assertEqual(d, eval(dstr))  # pylint: disable=eval-used

	def test_merge(self):
		from discodop.util import merge
		_h, _pairs, d1 = self.make_data()
		self.assertEqual(
				list(merge(sorted(d1.keys()), sorted(d1.values()))),
				sorted(list(d1.keys()) + list(d1.values())))


def test_fragments():
	from discodop._fragments import getctrees, extractfragments, exactcounts
	treebank = """\
(S (NP (DT 0) (NN 1)) (VP (VBP 2) (NP (DT 3) (JJ 4) (NN 5))))\
	The cat saw the hungry dog
(S (NP (DT 0) (NN 1)) (VP (VBP 2) (NP (DT 3) (NN 4))))\
	The cat saw the dog
(S (NP (DT 0) (NN 1)) (VP (VBP 2) (NP (DT 3) (NN 4))))\
	The mouse saw the cat
(S (NP (DT 0) (NN 1)) (VP (VBP 2) (NP (DT 3) (JJ 4) (NN 5))))\
	The mouse saw the yellow cat
(S (NP (DT 0) (JJ 1) (NN 2)) (VP (VBP 3) (NP (DT 4) (NN 5))))\
	The little mouse saw the cat
(S (NP (DT 0) (NN 1)) (VP (VBP 2) (NP (DT 3) (NN 4))))\
	The cat ate the dog
(S (NP (DT 0) (NN 1)) (VP (VBP 2) (NP (DT 3) (NN 4))))\
	The mouse ate the cat""".splitlines()
	trees = [binarize(Tree(line.split('\t')[0])) for line in treebank]
	sents = [line.split('\t')[1].split() for line in treebank]
	for tree in trees:
		for n, idx in enumerate(tree.treepositions('leaves')):
			tree[idx] = n
	params = getctrees(zip(trees, sents))
	fragments = extractfragments(params['trees1'],
			0, 0, params['vocab'], disc=True, approx=False)
	counts = exactcounts(
			list(fragments.values()), params['trees1'], params['trees1'])
	assert len(fragments) == 25
	assert sum(counts) == 100


def test_allfragments():
	from discodop.fragments import recurringfragments
	model = """\
(DT the)	1
(DT The)	1
(JJ hungry)	1
(NN cat)	1
(NN dog)	1
(NP|<DT.JJ,NN> (JJ hungry) (NN ))	1
(NP|<DT.JJ,NN> (JJ hungry) (NN dog))	1
(NP|<DT.JJ,NN> (JJ ) (NN ))	1
(NP|<DT.JJ,NN> (JJ ) (NN dog))	1
(NP (DT ) (NN ))	1
(NP (DT ) (NN cat))	1
(NP (DT ) (NP|<DT.JJ,NN> ))	1
(NP (DT ) (NP|<DT.JJ,NN> (JJ hungry) (NN )))	1
(NP (DT ) (NP|<DT.JJ,NN> (JJ hungry) (NN dog)))	1
(NP (DT ) (NP|<DT.JJ,NN> (JJ ) (NN )))	1
(NP (DT ) (NP|<DT.JJ,NN> (JJ ) (NN dog)))	1
(NP (DT The) (NN ))	1
(NP (DT The) (NN cat))	1
(NP (DT the) (NP|<DT.JJ,NN> ))	1
(NP (DT the) (NP|<DT.JJ,NN> (JJ hungry) (NN )))	1
(NP (DT the) (NP|<DT.JJ,NN> (JJ hungry) (NN dog)))	1
(NP (DT the) (NP|<DT.JJ,NN> (JJ ) (NN )))	1
(NP (DT the) (NP|<DT.JJ,NN> (JJ ) (NN dog)))	1
(S (NP (DT ) (NN cat)) (VP ))	1
(S (NP (DT ) (NN cat)) (VP (VBP ) (NP )))	1
(S (NP (DT ) (NN cat)) (VP (VBP ) (NP (DT ) (NP|<DT.JJ,NN> ))))	1
(S (NP (DT ) (NN cat)) (VP (VBP saw) (NP )))	1
(S (NP (DT ) (NN cat)) (VP (VBP saw) (NP (DT ) (NP|<DT.JJ,NN> ))))	1
(S (NP (DT ) (NN )) (VP ))	1
(S (NP (DT ) (NN )) (VP (VBP ) (NP )))	1
(S (NP (DT ) (NN )) (VP (VBP ) (NP (DT ) (NP|<DT.JJ,NN> ))))	1
(S (NP (DT ) (NN )) (VP (VBP saw) (NP )))	1
(S (NP (DT ) (NN )) (VP (VBP saw) (NP (DT ) (NP|<DT.JJ,NN> ))))	1
(S (NP (DT The) (NN cat)) (VP ))	1
(S (NP (DT The) (NN cat)) (VP (VBP ) (NP )))	1
(S (NP (DT The) (NN cat)) (VP (VBP ) (NP (DT ) (NP|<DT.JJ,NN> ))))	1
(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP )))	1
(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT ) (NP|<DT.JJ,NN> ))))	1
(S (NP (DT The) (NN )) (VP ))	1
(S (NP (DT The) (NN )) (VP (VBP ) (NP )))	1
(S (NP (DT The) (NN )) (VP (VBP ) (NP (DT ) (NP|<DT.JJ,NN> ))))	1
(S (NP (DT The) (NN )) (VP (VBP saw) (NP )))	1
(S (NP (DT The) (NN )) (VP (VBP saw) (NP (DT ) (NP|<DT.JJ,NN> ))))	1
(S (NP ) (VP ))	1
(S (NP ) (VP (VBP ) (NP )))	1
(S (NP ) (VP (VBP ) (NP (DT ) (NP|<DT.JJ,NN> ))))	1
(S (NP ) (VP (VBP saw) (NP )))	1
(S (NP ) (VP (VBP saw) (NP (DT ) (NP|<DT.JJ,NN> ))))	1
(VBP saw)	1
(VP (VBP ) (NP ))	1
(VP (VBP ) (NP (DT ) (NP|<DT.JJ,NN> )))	1
(VP (VBP ) (NP (DT ) (NP|<DT.JJ,NN> (JJ ) (NN ))))	1
(VP (VBP ) (NP (DT the) (NP|<DT.JJ,NN> )))	1
(VP (VBP ) (NP (DT the) (NP|<DT.JJ,NN> (JJ ) (NN ))))	1
(VP (VBP saw) (NP ))	1
(VP (VBP saw) (NP (DT ) (NP|<DT.JJ,NN> )))	1
(VP (VBP saw) (NP (DT ) (NP|<DT.JJ,NN> (JJ ) (NN ))))	1
(VP (VBP saw) (NP (DT the) (NP|<DT.JJ,NN> )))	1
(VP (VBP saw) (NP (DT the) (NP|<DT.JJ,NN> (JJ ) (NN ))))	1"""
	model = {a.split('\t')[0]: int(a.split('\t')[1])
			for a in model.splitlines()}
	answers = recurringfragments(
			[Tree('(S (NP (DT 0) (NN 1)) (VP (VBP 2) (NP (DT 3) '
				'(NP|<DT.JJ,NN> (JJ 4) (NN 5)))))')],
			[['The', 'cat', 'saw', 'the', 'hungry', 'dog']],
			disc=False, indices=False, maxdepth=3, maxfrontier=999)
	assert model
	assert answers
	assert answers == model


def test_grammar(debug=False):
	"""Demonstrate grammar extraction."""
	from discodop.grammar import treebankgrammar, dopreduction, doubledop
	from discodop import plcfrs
	from discodop.containers import Grammar
	from discodop.treebank import NegraCorpusReader
	from discodop.treetransforms import addfanoutmarkers
	from discodop.disambiguation import getderivations, marginalize
	corpus = NegraCorpusReader('alpinosample.export', punct='move')
	sents = list(corpus.sents().values())
	trees = [addfanoutmarkers(binarize(a.copy(True), horzmarkov=1))
			for a in list(corpus.trees().values())[:10]]
	if debug:
		print('plcfrs\n', Grammar(treebankgrammar(trees, sents)))
		print('dop reduction')
	grammar = Grammar(dopreduction(trees[:2], sents[:2])[0],
			start=trees[0].label)
	if debug:
		print(grammar)
	_ = grammar.testgrammar()

	grammarx, _backtransform, _, _ = doubledop(trees, sents,
			debug=False, numproc=1)
	if debug:
		print('\ndouble dop grammar')
	grammar = Grammar(grammarx, start=trees[0].label)
	grammar.getmapping(None, striplabelre=None,
			neverblockre=re.compile('^#[0-9]+|.+}<'),
			splitprune=False, markorigin=False)
	if debug:
		print(grammar)
	result, msg = grammar.testgrammar()
	assert result, 'RFE should sum to 1.\n%s' % msg
	for tree, sent in zip(corpus.trees().values(), sents):
		if debug:
			print('sentence:', ' '.join(a.encode('unicode-escape').decode()
					for a in sent))
		chart, msg = plcfrs.parse(sent, grammar, exhaustive=True)
		if debug:
			print('\n', msg, '\ngold ', tree, '\n', 'double dop', end='')
		if chart:
			getderivations(chart, 100)
			_parses, _msg = marginalize('mpp', chart)
		elif debug:
			print('no parse\n', chart)
		if debug:
			print()
	tree = Tree.parse('(ROOT (S (F (E (S (C (B (A 0))))))))', parse_leaf=int)
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
	def phrasal(x):
		return x and isinstance(x[0], Tree)

	from discodop.treebank import NegraCorpusReader
	filename = 'alpinosample.export'
	mangledtrees = NegraCorpusReader(filename, punct='move')
	nopunct = list(NegraCorpusReader(filename,
			punct='remove').trees().values())
	originals = list(NegraCorpusReader(filename, headrules=None,
			encoding='iso-8859-1').trees().values())
	for n, mangled, sent, nopunct, original in zip(count(),
			mangledtrees.trees().values(),
			mangledtrees.sents().values(), nopunct, originals):
		print(n, end='. ')
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
	correct = exact = e = 0
	for a, b, c, d in islice(zip(
			n.trees().values(), n.sents().values(), trees, count()), 100):
		transformc = reversetransform(c.copy(True), b, transformations)
		c1 = bracketings(canonicalize(a))
		c2 = bracketings(canonicalize(transformc))
		z = -1  # 825
		if c1 != c2 or e == z:
			precision = len(set(c1) & set(c2)) / len(set(c1))
			recall = len(set(c1) & set(c2)) / len(set(c2))
			if precision != 1.0 or recall != 1.0 or d == z:
				print(d, ' '.join(':'.join((str(n),
					a.encode('unicode-escape'))) for n, a in enumerate(b)))
				print('no match', precision, recall)
				print(len(c1), len(c2), 'gold-transformed', set(c2) - set(c1),
						'transformed-gold', set(c1) - set(c2))
				print(a)
				print(transformc)
				handlefunctions('add', a)
				print(a, '\n', b, '\n\n')
			else:
				correct += 1
		else:
			exact += 1
			correct += 1
		e += 1
	print('matches', correct, '/', e, 100 * correct / e, '%')
	print('exact', exact)


def test_treedraw():
	"""Draw some trees. Only tests whether no exception occurs."""
	trees = '''(ROOT (S (ADV 0) (VVFIN 1) (NP (PDAT 2) (NN 3)) (PTKNEG 4) \
				(PP (APPRART 5) (NN 6) (NP (ART 7) (ADJA 8) (NN 9)))) ($. 10))
			(S (NP (NN 1) (EX 3)) (VP (VB 0) (JJ 2)))
			(S (VP (PDS 0) (ADV 3) (VVINF 4)) (PIS 2) (VMFIN 1))
			(top (du (comp 0) (smain (noun 1) (verb 2) (inf (verb 8) (inf \
				(adj 3) (pp (prep 4) (np (det 5) (noun 6))) (part 7) (verb 9) \
				(pp (prep 10) (np (det 11) (noun 12) (pp (prep 13) (mwu \
				(noun 14) (noun 15))))))))) (punct 16))
			(top (smain (noun 0) (verb 1) (inf (verb 5) (inf (np (det 2) \
				(adj 3) (noun 4)) (verb 6) (pp (prep 7) (noun 8))))) (punct 9))
			(top (smain (noun 0) (verb 1) (noun 2) (inf (adv 3) (verb 4))) \
				(punct 5))
			(top (punct 5) (du (smain (noun 0) (verb 1) (ppart (np (det 2) \
				(noun 3)) (verb 4))) (conj (sv1 (conj (noun 6) (vg 7) (np \
				(det 8) (noun 9))) (verb 10) (noun 11) (part 12)) (vg 13) \
				(sv1 (verb 14) (ti (comp 19) (inf (np (conj (det 15) (vg 16) \
				(det 17)) (noun 18)) (verb 20)))))) (punct 21))
			(top (punct 10) (punct 16) (punct 18) (smain (np (det 0) (noun 1) \
				(pp (prep 2) (np (det 3) (noun 4)))) (verb 5) (adv 6) (np \
				(noun 7) (noun 8)) (part 9) (np (det 11) (noun 12) (pp \
				(prep 13) (np (det 14) (noun 15)))) (conj (vg 20) (ppres \
				(adj 17) (pp (prep 22) (np (det 23) (adj 24) (noun 25)))) \
				(ppres (adj 19)) (ppres (adj 21)))) (punct 26))
			(top (punct 10) (punct 11) (punct 16) (smain (np (det 0) \
				(noun 1)) (verb 2) (np (det 3) (noun 4)) (adv 5) (du (cp \
				(comp 6) (ssub (noun 7) (verb 8) (inf (verb 9)))) (du \
				(smain (noun 12) (verb 13) (adv 14) (part 15)) (noun 17)))) \
				(punct 18) (punct 19))
			(top (smain (noun 0) (verb 1) (inf (verb 8) (inf (verb 9) (inf \
				(adv 2) (pp (prep 3) (noun 4)) (pp (prep 5) (np (det 6) \
				(noun 7))) (verb 10))))) (punct 11))
			(top (smain (noun 0) (verb 1) (pp (prep 2) (np (det 3) (adj 4) \
				(noun 5) (rel (noun 6) (ssub (noun 7) (verb 10) (ppart \
				(adj 8) (part 9) (verb 11))))))) (punct 12))
			(top (smain (np (det 0) (noun 1)) (verb 2) (ap (adv 3) (num 4) \
				(cp (comp 5) (np (det 6) (adj 7) (noun 8) (rel (noun 9) (ssub \
				(noun 10) (verb 11) (pp (prep 12) (np (det 13) (adj 14) \
				(adj 15) (noun 16))))))))) (punct 17))
			(top (smain (np (det 0) (noun 1)) (verb 2) (adv 3) (pp (prep 4) \
				(np (det 5) (noun 6)) (part 7))) (punct 8))
			(top (punct 7) (conj (smain (noun 0) (verb 1) (np (det 2) \
				(noun 3)) (pp (prep 4) (np (det 5) (noun 6)))) (smain \
				(verb 8) (np (det 9) (num 10) (noun 11)) (part 12)) (vg 13) \
				(smain (verb 14) (noun 15) (pp (prep 16) (np (det 17) \
				(noun 18) (pp (prep 19) (np (det 20) (noun 21))))))) \
				(punct 22))
			(top (smain (np (det 0) (noun 1) (rel (noun 2) (ssub (np (num 3) \
				(noun 4)) (adj 5) (verb 6)))) (verb 7) (ppart (verb 8) (pp \
				(prep 9) (noun 10)))) (punct 11))
			(top (conj (sv1 (np (det 0) (noun 1)) (verb 2) (ppart (verb 3))) \
				(vg 4) (sv1 (verb 5) (pp (prep 6) (np (det 7) (adj 8) \
				(noun 9))))) (punct 10))
			(top (smain (noun 0) (verb 1) (np (det 2) (noun 3)) (inf (adj 4) \
				(verb 5) (cp (comp 6) (ssub (noun 7) (adv 8) (verb 10) (ap \
				(num 9) (cp (comp 11) (np (det 12) (adj 13) (noun 14) (pp \
				(prep 15) (conj (np (det 16) (noun 17)) (vg 18) (np \
				(noun 19))))))))))) (punct 20))
			(top (punct 8) (smain (noun 0) (verb 1) (inf (verb 5) \
				(inf (verb 6) (conj (inf (pp (prep 2) (np (det 3) (noun 4))) \
				(verb 7)) (inf (verb 9)) (vg 10) (inf (verb 11)))))) \
				(punct 12))
			(top (smain (verb 2) (noun 3) (adv 4) (ppart (np (det 0) \
				(noun 1)) (verb 5))) (punct 6))
			(top (conj (smain (np (det 0) (noun 1)) (verb 2) (adj 3) (pp \
				(prep 4) (np (det 5) (noun 6)))) (vg 7) (smain (np (det 8) \
				(noun 9) (pp (prep 10) (np (det 11) (noun 12)))) (verb 13) \
				(pp (prep 14) (np (det 15) (noun 16))))) (punct 17))
			(top (conj (smain (noun 0) (verb 1) (inf (ppart (np (noun 2) \
				(noun 3)) (verb 4)) (verb 5))) (vg 6) (smain (noun 7) \
				(inf (ppart (np (det 8) (noun 9)))))) (punct 10))
			(A (B1 (t 6) (t 13)) (B2 (t 3) (t 7) (t 10))  (B3 (t 1) \
				(t 9) (t 11) (t 14) (t 16)) (B4 (t 0) (t 5) (t 8)))
			(A (B1 6 13) (B2 3 7 10)  (B3 1 \
				9 11 14 16) (B4 0 5 8))
			(VP (VB 0) (PRT 2))
			(VP (VP 0 3) (NP (PRP 1) (NN 2)))
			(ROOT (S (VP_2 (PP (APPR 0) (ART 1) (NN 2) (PP (APPR 3) (ART 4) \
				(ADJA 5) (NN 6))) (ADJD 10) (PP (APPR 11) (NN 12)) (VVPP 13)) \
				(VAFIN 7) (NP (ART 8) (NN 9))) ($. 14))'''
	sents = '''Leider stehen diese Fragen nicht im Vordergrund der \
				augenblicklichen Diskussion .
			is Mary happy there
			das muss man jetzt machen
			Of ze had gewoon met haar vriendinnen rond kunnen slenteren in de \
				buurt van Trafalgar Square .
			Het had een prachtige dag kunnen zijn in Londen .
			Cathy zag hen wild zwaaien .
			Het was een spel geworden , zij en haar vriendinnen kozen iemand \
				uit en probeerden zijn of haar nationaliteit te raden .
			Elk jaar in het hoogseizoen trokken daar massa's toeristen \
				voorbij , hun fototoestel in de aanslag , pratend , gillend \
				en lachend in de vreemdste talen .
			Haar vader stak zijn duim omhoog alsof hij wilde zeggen : " het \
				komt wel goed , joch " .
			Ze hadden languit naast elkaar op de strandstoelen kunnen gaan \
				liggen .
			Het hoorde bij de warme zomerdag die ze ginds achter had gelaten .
			De oprijlaan was niet meer dan een hobbelige zandstrook die zich \
				voortslingerde tussen de hoge grijze boomstammen .
			Haar moeder kleefde bijna tegen het autoraampje aan .
			Ze veegde de tranen uit haar ooghoeken , tilde haar twee koffers \
				op en begaf zich in de richting van het landhuis .
			Het meisje dat vijf keer juist raadde werd getrakteerd op ijs .
			Haar neus werd platgedrukt en leek op een jonge champignon .
			Cathy zag de BMW langzaam verdwijnen tot hij niet meer was dan \
				een zilveren schijnsel tussen de bomen en struiken .
			Ze had met haar moeder kunnen gaan winkelen , zwemmen of \
				terrassen .
			Dat werkwoord had ze zelf uitgevonden .
			De middagzon hing klein tussen de takken en de schaduwen van de \
				wolken drentelden over het gras .
			Zij zou mams rug ingewreven hebben en mam de hare .
			0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
			0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
			0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
			0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
			Mit einer Messe in der Sixtinischen Kapelle ist das Konklave \
				offiziell zu Ende gegangen .'''
	from discodop.tree import DrawTree
	trees = [Tree(a) for a in trees.splitlines()]
	sents = [a.split() for a in sents.splitlines()]
	sents.extend([['Wake', None, 'up'],
		[None, 'your', 'friend', None]])
	for n, (tree, sent) in enumerate(zip(trees, sents)):
		drawtree = DrawTree(tree, sent)
		print('\ntree, sent', n, tree,
				' '.join('...' if a is None else a for a in sent),
				repr(drawtree),
				sep='\n')
		try:
			print(drawtree.text(unicodelines=True, ansi=True), sep='\n')
		except (UnicodeDecodeError, UnicodeEncodeError):
			print(drawtree.text(unicodelines=False, ansi=False), sep='\n')


def test_runexp():
	"""Run ``sample.prm``."""
	from discodop import cli
	if os.path.exists('sample.prm') and os.path.exists('sample/'):
		for path in os.listdir('sample/'):
			os.remove('sample/' + path)
		os.rmdir('sample/')
	cli.runexp(['sample.prm'])


def test_serialization(tmp_path):
	# assumes current working directory is project root
	tb = readtreebanks('alpinosample.export', fmt='export')
	vocab = tb['vocab']
	trees = tb['trees1']
	tmp = str(tmp_path / 'tmp')

	# Vocabulary serialization to file
	vocab.tofile(tmp)
	vocab1 = Vocabulary.fromfile(tmp)
	vocab2 = FixedVocabulary.fromfile(tmp)
	vocab2.makeindex()
	assert vocab.labels == vocab1.labels
	assert vocab.prods == vocab1.prods
	assert vocab.labels == vocab2.labels
	assert vocab.prods == vocab2.prods
	assert vocab.__getstate__() == vocab1.__getstate__()
	assert vocab.__getstate__() == vocab2.__getstate__()

	# Vocabulary pickling
	pickledvocab = pickle.dumps(vocab)
	vocab1 = pickle.loads(pickledvocab)
	assert vocab.labels == vocab1.labels
	assert vocab.prods == vocab1.prods
	assert vocab.__getstate__() == vocab1.__getstate__()

	# FixedVocabulary pickling
	pickledvocab = pickle.dumps(vocab2)
	vocab1 = pickle.loads(pickledvocab)
	assert vocab.labels == vocab1.labels
	assert vocab.prods == vocab1.prods
	assert vocab.__getstate__() == vocab1.__getstate__()

	# Ctrees serialization to file
	trees.tofile(tmp)
	trees1 = Ctrees.fromfile(tmp)
	assert trees.numnodes == trees1.numnodes
	assert trees.numwords == trees1.numwords
	assert trees.len == trees1.len
	assert len(trees.prodindex) == len(trees1.prodindex)
	assert all(a == b for a, b in zip(trees.prodindex, trees1.prodindex))
	# trees1.prodindex is a list, trees1.prodindex is a MultiRoaringBitmap

	trees2 = Ctrees.fromfilemut(tmp)
	assert trees.numnodes == trees2.numnodes
	assert trees.numwords == trees2.numwords
	assert trees.len == trees2.len
	assert len(trees.prodindex) == len(trees2.prodindex)
	assert all(a == b for a, b in zip(trees.prodindex, trees2.prodindex))
	assert trees.__getstate__() == trees2.__getstate__()

	# Ctrees pickling
	pickledtrees = pickle.dumps(trees)
	trees1 = pickle.loads(pickledtrees)
	assert trees.numnodes == trees1.numnodes
	assert trees.numwords == trees1.numwords
	assert trees.len == trees1.len
	assert len(trees.prodindex) == len(trees1.prodindex)
	assert all(a == b for a, b in zip(trees.prodindex, trees1.prodindex))
	assert trees.__getstate__() == trees1.__getstate__()


def test_issue51():
	from discodop.containers import Grammar
	from discodop.plcfrs import parse
	g = Grammar(
			[((('S', 'A'), ((0,),)), 1.0),
			((('A', 'Epsilon'), ('a',)), 1.0)],
			start='S')
	chart, _msg = parse(['b'], g)
	chart.filter()
