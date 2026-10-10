"""Lexical smoothing must survive chart inspection and parse rescoring."""
import re
from math import exp
from types import SimpleNamespace

import numpy as np
import pytest
from discodop.containers import Grammar

from discodop import coarsetofine, disambiguation, pcfg, plcfrs
from discodop.treetransforms import addbitsets


@pytest.fixture
def postagging():
	return SimpleNamespace(method='unknownword', closedclasswords={'the'},
			lexicon={'cat'}, unknownwordfun=lambda word, index, lexicon: '_UNK')


def make_grammar(backtransform=None, suffix='', signatureprob=0.25):
	# VB has no rule for cat, but its signature rule yields the best parse.
	rules = [
		((('ROOT', 'NN'), ((0,),)), 0.000001),
		((('ROOT', 'VB'), ((0,),)), 0.999999),
		((('NN', 'Epsilon'), ('cat',)), 0.5),
		((('NN', 'Epsilon'), ('dog',)), 0.5),
		((('VB', 'Epsilon'), ('_UNK',)), signatureprob),
		((('VB', 'Epsilon'), ('bark',)), 1 - signatureprob),
	]
	if suffix:
		rules = [((tuple(label + suffix if label in ('NN', 'VB') else label
				for label in labels), yf), weight)
				for (labels, yf), weight in rules]
	weights = np.full(len(rules), 0.5)
	# Keep the signature derivation competitive in the auxiliary model too.
	weights[0] = 0.000001
	return Grammar(rules, altweights={'shortest': weights},
			backtransform=backtransform)


def parse(parser, grammar, postagging, word='cat', tags=None):
	chart, msg = parser.parse([word], grammar, postagging=postagging, tags=tags)
	assert chart, msg
	disambiguation.getderivations(chart, 10)
	return chart


@pytest.mark.parametrize('parser', [pcfg, plcfrs])
@pytest.mark.parametrize('word', ['cat', 'Cat'])
def test_chart_and_shortest_scores(parser, word, postagging):
	grammar = make_grammar()
	chart = parse(parser, grammar, postagging, word)
	for logprob in (False, True):
		grammar.switch('default', logprob=logprob)
		description = str(chart)  # Previously raised "no lexical rule".
		assert f"=> {word!r} 0.00025" in description
		assert f"=> {word!r} 0.4995" in description
	for objective in ('mpd', 'mpp', 'shortest'):
		results, _ = disambiguation.marginalize(objective, chart, sent=[word])
		best = max(results, key=lambda result: result[1])
		assert best[0] == '(ROOT (VB 0))'
		score = best[1][1] if objective == 'shortest' else best[1]
		assert score == pytest.approx(0.999999 * 0.25 * 0.001)


@pytest.mark.parametrize('parser', [pcfg, plcfrs])
def test_shortest_after_model_switch(parser, postagging):
	grammar = make_grammar()
	grammar.switch('shortest')
	chart = parse(parser, grammar, postagging)
	grammar.switch('default')
	results, _ = disambiguation.marginalize('shortest', chart, sent=['cat'])
	assert results[0][1][1] == pytest.approx(0.999999 * 0.25 * 0.001)


@pytest.mark.parametrize('parser', [pcfg, plcfrs])
def test_reranking_preserves_smoothed_winner(parser, postagging):
	grammar = make_grammar()
	fine = make_grammar()
	fine.getrulemapping(grammar, re.compile('@[-0-9]+$'))
	chart = parse(parser, grammar, postagging)
	results, _ = disambiguation.marginalize('mpp', chart)
	reranked, _ = disambiguation.doprerank(results, ['cat'], 10,
			grammar, fine, postagging=postagging)
	assert max(reranked, key=lambda result: result[1])[0] == '(ROOT (VB 0))'
	assert {tree: prob for tree, prob, _ in reranked} == pytest.approx(
			{tree: prob for tree, prob, _ in results})


@pytest.mark.parametrize('parser', [pcfg, plcfrs])
@pytest.mark.parametrize('disabled', ['legacy', 'closed', 'gold', 'signature'])
def test_unsmoothed_scores(parser, disabled, postagging):
	grammar = make_grammar()
	word, tags = 'cat', None
	if disabled == 'legacy':
		postagging.closedclasswords = set()
	elif disabled == 'closed':
		postagging.closedclasswords.add('cat')
	elif disabled == 'gold':
		tags = ['NN']
	else:
		word = '_UNK'
	chart = parse(parser, grammar, postagging, word, tags)
	str(chart)
	results, _ = disambiguation.marginalize('shortest', chart, sent=[word])
	expected = 0.999999 * 0.25 if disabled == 'signature' else 0.000001 * 0.5
	assert results[0][1][1] == pytest.approx(expected)
	fine = make_grammar()
	fine.getrulemapping(grammar, re.compile('@[-0-9]+$'))
	assert exp(disambiguation.dopparseprob(addbitsets(results[0][0]),
			[word], grammar, fine, postagging=postagging,
			tags=tags)) == pytest.approx(expected)


def test_sldop_secondary_parse_keeps_smoothing(postagging):
	grammar = make_grammar()
	grammar.getmapping(None, striplabelre=re.compile('@[-0-9]+'))
	chart = parse(plcfrs, grammar, postagging)
	results, msg = disambiguation.marginalize('sl-dop', chart,
			sent=['cat'], k=10, sldop_n=1)
	assert results, msg
	assert results[0][0] == '(ROOT (VB 0))'
	assert grammar.currentmodel == 'default'


@pytest.mark.parametrize('parser', [pcfg, plcfrs])
@pytest.mark.parametrize('logprob', [False, True])
def test_double_dop_fragment_weights(parser, logprob, postagging):
	grammar = make_grammar(backtransform=['(ROOT {0})', '(ROOT {0})'])
	grammar.getmapping(None, striplabelre=re.compile('@[-0-9]+'))
	chart = parse(parser, grammar, postagging)
	grammar.switch('default', logprob=logprob)
	results, _ = disambiguation.marginalize('mpd', chart)
	best = max(results, key=lambda result: result[1])
	assert best[0] == '(ROOT (VB 0))'
	lexical = next(info for fragment, info in best[2] if fragment == '(VB 0=cat)')
	assert float(lexical.split('weight: ')[1]) == pytest.approx(0.25 * 0.001)


@pytest.mark.parametrize('parser', [pcfg, plcfrs])
def test_observed_tag_precedes_signature(parser, postagging):
	rules = [
		((('ROOT', 'NN'), ((0,),)), 1.0),
		((('NN', 'Epsilon'), ('cat',)), 0.75),
		((('NN', 'Epsilon'), ('_UNK',)), 0.25),
	]
	grammar = Grammar(rules)
	chart = parse(parser, grammar, postagging)
	str(chart)
	results, _ = disambiguation.marginalize('shortest', chart, sent=['cat'])
	assert results[0][1][1] == pytest.approx(0.75 * 0.999)
	fine = Grammar(rules)
	fine.getrulemapping(grammar, re.compile('@[-0-9]+$'))
	assert exp(disambiguation.dopparseprob(addbitsets('(ROOT (NN 0))'),
			['cat'], grammar, fine, postagging=postagging)) == pytest.approx(
			0.75 * 0.999)


@pytest.mark.parametrize('parser', [pcfg, plcfrs])
def test_reranking_uses_fine_signature_weights(parser, postagging):
	coarse = make_grammar()
	fine = make_grammar(suffix='@1', signatureprob=0.125)
	fine.getrulemapping(coarse, re.compile('@[-0-9]+$'))
	chart = parse(parser, coarse, postagging)
	results, _ = disambiguation.marginalize('mpp', chart)
	reranked, _ = disambiguation.doprerank(results, ['cat'], 10,
			coarse, fine, postagging=postagging)
	probabilities = {tree: prob for tree, prob, _ in reranked}
	assert probabilities['(ROOT (VB 0))'] == pytest.approx(0.999999 * 0.125 * 0.001)
	assert probabilities['(ROOT (NN 0))'] == pytest.approx(0.000001 * 0.5 * 0.999)


@pytest.mark.parametrize('parser', [pcfg, plcfrs])
def test_posterior_pruning_uses_effective_weights(parser, postagging):
	grammar = make_grammar()
	chart = parse(parser, grammar, postagging)
	items, msg = coarsetofine.posteriorthreshold(chart, 0.5)
	expected = 0.999999 * 0.25 * 0.001 + 0.000001 * 0.5 * 0.999
	assert float(msg.split('sentprob=')[1]) == pytest.approx(expected, rel=1e-5)
	assert len(items) == 2  # ROOT and VB; NN has negligible posterior mass.
	assert grammar.logprob


def test_sparse_chart_smoothing(postagging):
	coarse, fine = make_grammar(), make_grammar()
	chart = parse(pcfg, coarse, postagging)
	fine.getmapping(coarse)
	whitelist, _ = coarsetofine.prunechart(chart, fine, 10,
			splitprune=False, markorigin=False, finecfg=True)
	chart, msg = pcfg.parse(['cat'], fine, whitelist=whitelist,
			postagging=postagging)
	assert chart, msg
	assert isinstance(chart, pcfg.SparseCFGChart)
	str(chart)
	disambiguation.getderivations(chart, 10)
	results, _ = disambiguation.marginalize('shortest', chart, sent=['cat'])
	assert results[0][1][1] == pytest.approx(0.999999 * 0.25 * 0.001)


def test_fat_chart_smoothing(postagging):
	rules = [
		((('ROOT', 'S'), ((0,),)), 1.0),
		((('S', 'VB', 'S'), ((0, 1),)), 0.5),
		((('S', 'VB'), ((0,),)), 0.5),
		((('NN', 'Epsilon'), ('cat',)), 1.0),
		((('VB', 'Epsilon'), ('_UNK',)), 1.0),
	]
	grammar = Grammar(rules)
	sent = ['cat'] * 65
	chart, msg = plcfrs.parse(sent, grammar, postagging=postagging)
	assert chart, msg
	assert isinstance(chart, plcfrs.FatLCFRSChart)
	str(chart)
	disambiguation.getderivations(chart, 1)
	results, _ = disambiguation.marginalize('shortest', chart, sent=sent)
	assert results[0][1][1] == pytest.approx((0.5 * 0.001) ** 65, rel=1e-10, abs=0)


@pytest.mark.parametrize('parser', [pcfg, plcfrs])
@pytest.mark.parametrize('word', ['cat', 'unseen'])
def test_forced_tag_without_lexical_rule(parser, word, postagging):
	grammar = make_grammar()
	chart = parse(parser, grammar, postagging, word, tags=['VB'])
	str(chart)
	results, _ = disambiguation.marginalize('shortest', chart, sent=[word])
	assert results[0][0] == '(ROOT (VB 0))'
	assert results[0][1][1] == pytest.approx(0.999999)


def test_sldop_failure_restores_model(postagging, monkeypatch):
	grammar = make_grammar()
	chart = parse(plcfrs, grammar, postagging)

	def failed_parse(*args, **kwargs):
		assert kwargs['postagging'] is postagging
		return [], 'tree parsing failed', None

	monkeypatch.setattr(disambiguation, 'treeparsing', failed_parse)
	results, _ = disambiguation.marginalize('sl-dop', chart,
			sent=['cat'], k=10, sldop_n=1)
	assert not results
	assert grammar.currentmodel == 'default'
