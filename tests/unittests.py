import re
from unittest import TestCase
from operator import itemgetter
from discodop.tree import Tree, ParentedTree
from discodop.treetransforms import binarize, unbinarize, \
		splitdiscnodes, mergediscnodes, \
		addbitsets, fanout
from discodop.treebanktransforms import punctraise, balancedpunctraise
from discodop.grammar import flattenbin, UniqueIDs
# FIXME only import for relevant test:
from discodop.plcfrs import Agenda, getparent

class Test_treetransforms:
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

		assert str(binarize(tree, horzmarkov=1, vertmarkov=3, tailmarker='',
				leftmostunary=True, rightmostunary=False, pospa=True)) == (
				'(S (S|<VP> (VP^<S> (VP|<PDS>^<S> (PDS^<VP,S> 0) '
				'(VP|<ADV>^<S> (ADV^<VP,S> 3) (VVINF^<VP,S> 4)))) (S|<PIS> '
				'(PIS^<S> 2) (VMFIN^<S> 1))))')
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


class Test_treebanktransforms:
	def test_balancedpunctraise(self):
		tree = ParentedTree.parse('(ROOT ($, 3) ($[ 7) ($[ 13) ($, 14) ($, 20) '
				'(S (NP (ART 0) (ADJA 1) (NN 2) (NP (CARD 4) (NN 5) (PP '
				'(APPR 6) (CNP (NN 8) (ADV 9) (ISU ($. 10) ($. 11) '
				'($. 12))))) (S (PRELS 15) (MPN (NE 16) (NE 17)) (ADJD 18) '
				'(VVFIN 19))) (VVFIN 21) (ADV 22) (NP (ADJA 23) (NN 24))) '
				'($. 25))', parse_leaf=int)
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


class Test_grammar:
	def test_flatten(self):
		ids = UniqueIDs()
		sent = [None, ',', None, '.']
		tree = "(ROOT (S_2 0 2) (ROOT|<$,>_2 ($, 1) ($. 3)))"
		assert flattenbin(tree, sent, ids, {}) == (
				[(('ROOT', 'ROOT}<0>', '$.@.'), ((0, 1),)),
				(('ROOT}<0>', 'S_2', '$,@,'), ((0, 1, 0),)),
				(('$,@,', 'Epsilon'), (',',)), (('$.@.', 'Epsilon'), ('.',))],
				'(ROOT {0} (ROOT|<$,>_2 {1} {2}))')

		assert flattenbin("(NN 0)", ["foo"], ids, {}) == (
				[(('NN', 'Epsilon'), ('foo',))], '(NN 0)')

		prods, frag = flattenbin(r"(S (S|<VP> (S|<NP> (NP (ART 0) (CNP "
				"(CNP|<TRUNC> (TRUNC 1) (CNP|<KON> (KON 2) (CNP|<NN> "
				"(NN 3)))))) (S|<VAFIN> (VAFIN 4))) (VP (VP|<ADV> (ADV 5) "
				"(VP|<NP> (NP (ART 6) (NN 7)) (VP|<NP> (NP_2 8 10) (VP|<VVPP> "
				"(VVPP 9))))))))",
				['Das', 'Garten-', 'und', 'Friedhofsamt', 'hatte', 'kuerzlich',
				'dem', 'Ortsbeirat', None, None, None], ids, {})
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

		assert flattenbin("(S|<VP>_2 (VP_3 (VP|<NP>_3 (NP 0) (VP|<ADV>_2 "
				"(ADV 2) (VP|<VVPP> (VVPP 4))))) (S|<VAFIN> (VAFIN 1)))",
				(None, None, None, None, None), ids, {}) == (
				[(('S|<VP>_2', 'S|<VP>_2}<10>', 'VVPP'), ((0,), (1,))),
				(('S|<VP>_2}<10>', 'S|<VP>_2}<9>', 'ADV'), ((0, 1),)),
				(('S|<VP>_2}<9>', 'NP', 'VAFIN'), ((0, 1),))],
				'(S|<VP>_2 (VP_3 (VP|<NP>_3 {0} (VP|<ADV>_2 {2} (VP|<VVPP> '
				'{3})))) (S|<VAFIN> {1}))')


class TestHeap(TestCase):
	testN = 100

	def check_invariants(self, h):
		for i in range(len(h)):
			if i > 0:
				self.assertTrue(h.heap[getparent(i)].getvalue()
						<= h.heap[i].getvalue())

	def make_data(self):
		from random import random
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
		h = Agenda()
		for i in range(TestHeap.testN):
			h[i] = 0.
		for i in range(TestHeap.testN):
			_, v = h.popitem()
			self.assertEqual(v, 0.)
			self.check_invariants(h)

	def test_popitem_ties_fifo(self):
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
		h, pairs, d = self.make_data()
		#self.assertEqual(h, eval(repr(h)))
		tmp = repr(h)  # 'Agenda({....})'
		#strip off class name
		dstr = tmp[tmp.index('(') + 1:tmp.rindex(')')]
		self.assertEqual(d, eval(dstr))


def test_fragments():
	from discodop._fragments import getctrees, fastextractfragments, \
			exactcounts
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
	fragments = fastextractfragments(params['trees1'], params['sents1'], 0, 0,
			params['labels'], discontinuous=True, approx=False)
	counts = exactcounts(params['trees1'], params['trees1'],
			list(fragments.values()), fast=True)
	assert len(fragments) == 25
	assert sum(counts) == 100
	for (a, b), c in sorted(zip(fragments, counts), key=repr):
		print("%s\t%d" % (re.sub("[0-9]+", lambda x: b[int(x.group())], a), c))
