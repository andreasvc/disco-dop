# -*- coding: UTF-8 -*-
""" Run an experiment given a parameter file. Extracts grammars, does parsing
and evaluation. """
from __future__ import division, print_function
import io
import os
import re
import sys
import time
import gzip
import codecs
import logging
import tempfile
import multiprocessing
from itertools import islice
if sys.version[0] >= '3':
	import pickle
	from itertools import zip_longest  # pylint: disable=E0611
else:
	import cPickle as pickle
	from itertools import izip_longest as zip_longest
from collections import defaultdict, OrderedDict, Counter as multiset
from operator import itemgetter
from subprocess import Popen, PIPE
import numpy as np
from discodop import eval as evalmod
from discodop.tree import Tree
from discodop.treebank import READERS, writetree, treebankfanout
from discodop.treebanktransforms import transform, rrtransform
from discodop.treetransforms import binarize, optimalbinarize, canonicalize, \
		splitdiscnodes, addfanoutmarkers
from discodop.treedraw import DrawTree
from discodop.fragments import getfragments
from discodop.grammar import treebankgrammar, dopreduction, \
		doubledop, grammarinfo, write_lcfrs_grammar
from discodop.containers import Grammar
from discodop.lexicon import getunknownwordmodel, getlexmodel, smoothlexicon, \
		simplesmoothlexicon, replaceraretrainwords, getunknownwordfun
from discodop.parser import DEFAULTSTAGE, readgrammars, Parser, DictObj
from discodop.estimates import getestimates, getpcfgestimates

USAGE = """Usage: %s [--rerun] <parameter file>
If a parameter file is given, an experiment is run. See the file sample.prm for
an example parameter file. To repeat an experiment with an existing grammar,
pass the option --rerun.""" % sys.argv[0]

INTERNALPARAMS = None


def initworker(params):
	""" Set global parameter object """
	global INTERNALPARAMS
	INTERNALPARAMS = params


def startexp(
		stages=(DEFAULTSTAGE, ),  # see above
		corpusfmt='export',  # choices: export, (disc)bracket, alpino, tiger
		corpusdir='.',
		# filenames may include globbing characters '*' and '?'.
		traincorpus='alpinosample.export', trainencoding='utf-8',
		testcorpus='alpinosample.export', testencoding='utf-8',
		testmaxwords=40,
		trainmaxwords=40,
		trainnumsents=2,
		testnumsents=1,  # number of sentences to parse
		skiptrain=True,  # test set starts after training set
		# (useful when they are in the same file)
		skip=0,  # number of sentences to skip from test corpus
		punct=None,  # choices: None, 'move', 'remove', 'root'
		functions=None,  # choices None, 'add', 'remove', 'replace'
		morphology=None,  # choices: None, 'add', 'replace', 'between'
		transformations=None,  # apply treebank transformations
		# postagging: pass None to use tags from treebank.
		postagging=None,
		relationalrealizational=None,  # do not apply RR-transform
		headrules=None,  # rules for finding heads of constituents
		bintype='binarize',  # choices: binarize, optimal, optimalhead
		factor='right',
		revmarkov=True,
		v=1,
		h=2,
		pospa=False,  # when v > 1, add parent annotation to POS tags?
		markhead=False,  # prepend head to siblings
		leftmostunary=True,  # start binarization with unary node
		rightmostunary=True,  # end binarization with unary node
		tailmarker='',  # with headrules, head is last node and can be marked
		fanout_marks_before_bin=False,
		evalparam='proper.prm',  # EVALB-style parameter file
		quiet=False, reallyquiet=False,  # quiet=no per sentence results
		numproc=1,  # increase to use multiple CPUs; None: use all CPUs.
		resultdir='results',
		rerun=False):
	""" Execute an experiment. """
	assert bintype in ('optimal', 'optimalhead', 'binarize')
	if postagging is not None:
		assert set(postagging).issubset({'method', 'model',
				'unknownthreshold', 'openclassthreshold', 'simplelexsmooth'})
		if postagging['method'] == 'unknownword':
			assert postagging['model'] in ('4', '6', 'base')
			assert postagging['unknownthreshold'] >= 1
			assert postagging['openclassthreshold'] >= 0
		else:
			assert postagging['method'] in ('treetagger', 'stanford')

	if rerun:
		assert os.path.exists(resultdir), (
				'Directory %r does not exist.'
				'--rerun requires a directory '
				'with the grammar(s) of a previous experiment.'
				% resultdir)
	else:
		assert not os.path.exists(resultdir), (
			'Directory %r exists.\n'
			'Use --rerun to parse with existing grammar '
			'and overwrite previous results.' % resultdir)
		os.mkdir(resultdir)

	# Log everything, and send it to stderr, in a format with just the message.
	formatstr = '%(message)s'
	if reallyquiet:
		logging.basicConfig(level=logging.WARNING, format=formatstr)
	elif quiet:
		logging.basicConfig(level=logging.INFO, format=formatstr)
	else:
		logging.basicConfig(level=logging.DEBUG, format=formatstr)

	# also log to a file
	fileobj = logging.FileHandler(filename='%s/output.log' % resultdir)
	#fileobj.setLevel(logging.INFO)
	fileobj.setLevel(logging.DEBUG)
	fileobj.setFormatter(logging.Formatter(formatstr))
	logging.getLogger('').addHandler(fileobj)

	corpusreader = READERS[corpusfmt]
	if not rerun:
		corpus = corpusreader(corpusdir, traincorpus, encoding=trainencoding,
				headrules=headrules, headfinal=True, headreverse=False,
				punct=punct, functions=functions, morphology=morphology)
		logging.info('%d sentences in training corpus %s/%s',
				len(corpus.parsed_sents()), corpusdir, traincorpus)
		if isinstance(trainnumsents, float):
			trainnumsents = int(trainnumsents * len(corpus.sents()))
		trees = list(corpus.parsed_sents().values())[:trainnumsents]
		sents = list(corpus.sents().values())[:trainnumsents]
		if transformations:
			trees = [transform(tree, sent, transformations)
					for tree, sent in zip(trees, sents)]
		if relationalrealizational:
			trees = [rrtransform(tree, **relationalrealizational)[0]
					for tree in trees]
		train_tagged_sents = [[(word, tag) for word, (_, tag)
				in zip(sent, sorted(tree.pos()))]
					for tree, sent in zip(trees, sents)]
		blocks = list(corpus.blocks().values())[:trainnumsents]
		assert trees, 'training corpus should be non-empty'
		logging.info('%d training sentences before length restriction',
				len(trees))
		trees, sents, blocks = zip(*[sent for sent in zip(trees, sents, blocks)
			if len(sent[1]) <= trainmaxwords])
		logging.info('%d training sentences after length restriction <= %d',
			len(trees), trainmaxwords)

	testset = corpusreader(corpusdir, testcorpus, encoding=testencoding,
			punct=punct, morphology=morphology, functions=functions)
	gold_sents = testset.tagged_sents()
	test_parsed_sents = testset.parsed_sents()
	if skiptrain:
		skip += trainnumsents
	logging.info('%d sentences in test corpus %s/%s',
			len(testset.parsed_sents()), corpusdir, testcorpus)
	logging.info('%d test sentences before length restriction',
			len(list(gold_sents)[skip:skip + testnumsents]))
	lexmodel = None
	test_tagged_sents = gold_sents
	if postagging and postagging['method'] in ('treetagger', 'stanford'):
		if postagging['method'] == 'treetagger':
			# these two tags are never given by tree-tagger,
			# so collect words whose tag needs to be overriden
			overridetags = ('PTKANT', 'PIDAT')
		elif postagging['method'] == 'stanford':
			overridetags = ('PTKANT', )
		taglex = defaultdict(set)
		for sent in train_tagged_sents:
			for word, tag in sent:
				taglex[word].add(tag)
		overridetagdict = {tag:
			{word for word, tags in taglex.items() if tags == {tag}}
			for tag in overridetags}
		tagmap = {'$(': '$[', 'PAV': 'PROAV'}
		sents_to_tag = OrderedDict((a, b) for a, b
				in islice(gold_sents.items(), skip, skip + testnumsents)
				if len(b) <= testmaxwords),
		test_tagged_sents = externaltagging(postagging['method'],
				postagging['model'], sents_to_tag, overridetagdict, tagmap)
		# give these tags to parser
		usetags = True
	elif postagging and postagging['method'] == 'unknownword' and not rerun:
		postagging['unknownwordfun'] = getunknownwordfun(postagging['model'])
		# get smoothed probalities for lexical productions
		lexresults, msg = getunknownwordmodel(
				train_tagged_sents, postagging['unknownwordfun'],
				postagging['unknownthreshold'],
				postagging['openclassthreshold'])
		logging.info(msg)
		simplelexsmooth = postagging['simplelexsmooth']
		if simplelexsmooth:
			lexmodel = lexresults[2:8]
		else:
			lexmodel, msg = getlexmodel(*lexresults)
			logging.info(msg)
		# NB: knownwords are all words in training set, lexicon is the subset
		# of words that are above the frequency threshold.
		# for training purposes we work with the subset, at test time we exploit
		# the full set of known words from the training set.
		sigs, knownwords, lexicon = lexresults[:3]
		postagging['sigs'], postagging['lexicon'] = sigs, knownwords
		# replace rare train words with signatures
		sents = replaceraretrainwords(train_tagged_sents,
				postagging['unknownwordfun'], lexicon)
		# make sure gold POS tags are not given to parser
		usetags = False
	elif postagging and postagging['method'] == 'unknownword' and rerun:
		usetags = False
	else:
		simplelexsmooth = False
		# give gold POS tags to parser
		usetags = True

	# 0: test sentences as they should be handed to the parser,
	# 1: gold trees for evaluation purposes
	# 2: gold sentence because test sentences may be mangled by unknown word
	#   model
	# 3: blocks from treebank file to reproduce the relevant part of the
	#   original treebank verbatim.
	testset = OrderedDict((a, (test_tagged_sents[a], test_parsed_sents[a],
			gold_sents[a], block)) for a, block
			in islice(testset.blocks().items(), skip, skip + testnumsents)
			if len(test_tagged_sents[a]) <= testmaxwords)
	assert test_tagged_sents, 'test corpus should be non-empty'
	logging.info('%d test sentences after length restriction <= %d',
			len(testset), testmaxwords)

	if rerun:
		trees = []
		sents = []
	toplabels = {tree.label for tree in trees} | {
			test_parsed_sents[n].label for n in testset}
	assert len(toplabels) == 1, 'expected unique ROOT label: %r' % toplabels
	top = toplabels.pop()

	if rerun:
		readgrammars(resultdir, stages, postagging, top)
	else:
		logging.info('read training & test corpus')
		getgrammars(trees, sents, stages, bintype, h, v, factor, tailmarker,
				revmarkov, leftmostunary, rightmostunary, pospa, markhead,
				fanout_marks_before_bin, testmaxwords, resultdir, numproc,
				lexmodel, simplelexsmooth, top, relationalrealizational)
	evalparam = evalmod.readparam(evalparam)
	evalparam['DEBUG'] = -1
	evalparam['CUTOFF_LEN'] = 40
	deletelabel = evalparam.get('DELETE_LABEL', ())
	deleteword = evalparam.get('DELETE_WORD', ())

	begin = time.clock()
	parser = Parser(stages, transformations=transformations,
			tailmarker=tailmarker, postagging=postagging if postagging
			and postagging['method'] == 'unknownword' else None,
			relationalrealizational=relationalrealizational)
	results = doparsing(parser=parser, testset=testset, resultdir=resultdir,
			usetags=usetags, numproc=numproc, deletelabel=deletelabel,
			deleteword=deleteword, corpusfmt=corpusfmt, morphology=morphology)
	if numproc == 1:
		logging.info('time elapsed during parsing: %gs', time.clock() - begin)
	for result in results[0]:
		nsent = len(result.parsetrees)
		header = (' ' + result.name.upper() + ' ').center(35, '=')
		evalsummary = evalmod.doeval(OrderedDict((a, b.copy(True))
				for a, b in test_parsed_sents.items()), gold_sents,
				result.parsetrees, test_tagged_sents if usetags else gold_sents,
				evalparam)
		coverage = 'coverage: %s = %6.2f' % (
				('%d / %d' % (nsent - result.noparse, nsent)).rjust(
				25 if any(len(a) > evalparam['CUTOFF_LEN']
				for a in gold_sents.values()) else 14),
				100.0 * (nsent - result.noparse) / nsent)
		logging.info('\n'.join(('', header, evalsummary, coverage)))
	return top


def getgrammars(trees, sents, stages, bintype, horzmarkov, vertmarkov, factor,
		tailmarker, revmarkov, leftmostunary, rightmostunary, pospa, markhead,
		fanout_marks_before_bin, testmaxwords, resultdir, numproc,
		lexmodel, simplelexsmooth, top, relationalrealizational):
	""" Apply binarization and read off the requested grammars. """
	# fixme: this n should correspond to sentence id
	tbfanout, n = treebankfanout(trees)
	logging.info('treebank fan-out before binarization: %d #%d\n%s\n%s',
			tbfanout, n, trees[n], ' '.join(sents[n]))
	# binarization
	begin = time.clock()
	if fanout_marks_before_bin:
		trees = [addfanoutmarkers(t) for t in trees]
	if bintype == 'binarize':
		bintype += ' %s h=%d v=%d %s' % (factor, horzmarkov, vertmarkov,
			'tailmarker' if tailmarker else '')
		for a in trees:
			binarize(a, factor=factor, tailmarker=tailmarker,
					horzmarkov=horzmarkov, vertmarkov=vertmarkov,
					leftmostunary=leftmostunary, rightmostunary=rightmostunary,
					reverse=revmarkov, pospa=pospa,
					headidx=-1 if markhead else None,
					filterfuncs=(relationalrealizational['ignorefunctions']
						+ (relationalrealizational['adjunctionlabel'], ))
						if relationalrealizational else ())
	elif bintype == 'optimal':
		trees = [Tree.convert(optimalbinarize(tree))
						for n, tree in enumerate(trees)]
	elif bintype == 'optimalhead':
		trees = [Tree.convert(optimalbinarize(tree, headdriven=True,
				h=horzmarkov, v=vertmarkov)) for n, tree in enumerate(trees)]
	trees = [addfanoutmarkers(t) for t in trees]
	logging.info('binarized %s cpu time elapsed: %gs',
						bintype, time.clock() - begin)
	logging.info('binarized treebank fan-out: %d #%d', *treebankfanout(trees))
	trees = [canonicalize(a).freeze() for a in trees]

	for n, stage in enumerate(stages):
		if stage.split:
			traintrees = [binarize(splitdiscnodes(Tree.convert(a),
					stage.markorigin), childchar=':').freeze() for a in trees]
			logging.info('splitted discontinuous nodes')
		else:
			traintrees = trees
		if stage.mode.startswith('pcfg'):
			assert tbfanout == 1 or stage.split
		backtransform = None
		if stage.dop:
			if stage.usedoubledop:
				# find recurring fragments in treebank,
				# as well as depth 1 'cover' fragments
				fragments = getfragments(traintrees, sents, numproc,
						iterate=stage.iterate, complement=stage.complement)
				xgrammar, backtransform, altweights = doubledop(
						traintrees, fragments)
			else:  # DOP reduction
				xgrammar, altweights = dopreduction(
						traintrees, sents, packedgraph=stage.packedgraph)
			nodes = sum(len(list(a.subtrees())) for a in traintrees)
			if lexmodel and simplelexsmooth:
				newrules = simplesmoothlexicon(lexmodel)
				xgrammar.extend(newrules)
				for weights in altweights.values():
					weights.extend(w for _, w in newrules)
			elif lexmodel:
				xgrammar = smoothlexicon(xgrammar, lexmodel)
			msg = grammarinfo(xgrammar)
			rules, lexicon = write_lcfrs_grammar(
					xgrammar, bitpar=stage.mode.startswith('pcfg'))
			grammar = Grammar(rules, lexicon, start=top,
					bitpar=stage.mode.startswith('pcfg'))
			for name in altweights:
				grammar.register(u'%s' % name, altweights[name])
			with gzip.open('%s/%s.rules.gz' % (
					resultdir, stage.name), 'wb') as rulesfile:
				rulesfile.write(rules)
			with codecs.getwriter('utf-8')(gzip.open('%s/%s.lex.gz' % (
					resultdir, stage.name), 'wb')) as lexiconfile:
				lexiconfile.write(lexicon)
			logging.info('DOP model based on %d sentences, %d nodes, '
				'%d nonterminals', len(traintrees), nodes, len(grammar.toid))
			logging.info(msg)
			if stage.estimator != 'dop1':
				grammar.switch(u'%s' % stage.estimator)
			_sumsto1 = grammar.testgrammar()
			if stage.usedoubledop:
				# backtransform keys are line numbers to rules file;
				# to see them together do:
				# $ paste <(zcat dop.rules.gz) <(zcat dop.backtransform.gz)
				with codecs.getwriter('ascii')(gzip.open(
						'%s/%s.backtransform.gz' % (resultdir, stage.name),
						'w')) as out:
					out.writelines('%s\n' % a for a in backtransform)
				if n and stage.prune:
					msg = grammar.getmapping(stages[n - 1].grammar,
						striplabelre=None if stages[n - 1].dop
							else re.compile(b'@.+$'),
						neverblockre=re.compile(b'.+}<'),
						splitprune=stage.splitprune and stages[n - 1].split,
						markorigin=stages[n - 1].markorigin)
				else:
					# recoverfragments() relies on this mapping to identify
					# binarization nodes
					msg = grammar.getmapping(None,
						striplabelre=None,
						neverblockre=re.compile(b'.+}<'),
						splitprune=False, markorigin=False)
				logging.info(msg)
			elif n and stage.prune:  # dop reduction
				msg = grammar.getmapping(stages[n - 1].grammar,
					striplabelre=None if stages[n - 1].dop
						and not stages[n - 1].usedoubledop
						else re.compile(b'@[-0-9]+$'),
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[n - 1].split,
					markorigin=stages[n - 1].markorigin)
				if stage.mode == 'dop-rerank':
					grammar.getrulemapping(stages[n - 1].grammar)
				logging.info(msg)
			# write prob models
			np.savez_compressed('%s/%s.probs.npz' % (resultdir, stage.name),
					**{name: mod for name, mod
						in zip(grammar.modelnames, grammar.models)})
		else:  # not stage.dop
			xgrammar = treebankgrammar(traintrees, sents)
			logging.info('induced %s based on %d sentences',
				('PCFG' if tbfanout == 1 or stage.split else 'PLCFRS'),
				len(traintrees))
			if stage.split or os.path.exists('%s/pcdist.txt' % resultdir):
				logging.info(grammarinfo(xgrammar))
			else:
				logging.info(grammarinfo(xgrammar,
						dump='%s/pcdist.txt' % resultdir))
			if lexmodel and simplelexsmooth:
				newrules = simplesmoothlexicon(lexmodel)
				xgrammar.extend(newrules)
			elif lexmodel:
				xgrammar = smoothlexicon(xgrammar, lexmodel)
			rules, lexicon = write_lcfrs_grammar(
					xgrammar, bitpar=stage.mode.startswith('pcfg'))
			grammar = Grammar(rules, lexicon, start=top,
					bitpar=stage.mode.startswith('pcfg'))
			with gzip.open('%s/%s.rules.gz' % (
					resultdir, stage.name), 'wb') as rulesfile:
				rulesfile.write(rules)
			with codecs.getwriter('utf-8')(gzip.open('%s/%s.lex.gz' % (
					resultdir, stage.name), 'wb')) as lexiconfile:
				lexiconfile.write(lexicon)
			_sumsto1 = grammar.testgrammar()
			if n and stage.prune:
				msg = grammar.getmapping(stages[n - 1].grammar,
					striplabelre=None,
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[n - 1].split,
					markorigin=stages[n - 1].markorigin)
				logging.info(msg)
		logging.info('wrote grammar to %s/%s.{rules,lex%s}.gz', resultdir,
				stage.name, ',backtransform' if stage.usedoubledop else '')

		outside = None
		if stage.getestimates == 'SX':
			assert tbfanout == 1 or stage.split, 'SX estimate requires PCFG.'
			logging.info('computing PCFG estimates')
			begin = time.clock()
			outside = getpcfgestimates(grammar, testmaxwords,
					grammar.toid[trees[0].label])
			logging.info('estimates done. cpu time elapsed: %gs',
					time.clock() - begin)
			np.savez('pcfgoutside.npz', outside=outside)
			logging.info('saved PCFG estimates')
		elif stage.useestimates == 'SX':
			assert tbfanout == 1 or stage.split, 'SX estimate requires PCFG.'
			assert stage.mode != 'pcfg', (
					'estimates require agenda-based parser.')
			outside = np.load('pcfgoutside.npz')['outside']
			logging.info('loaded PCFG estimates')
		if stage.getestimates == 'SXlrgaps':
			logging.info('computing PLCFRS estimates')
			begin = time.clock()
			outside = getestimates(grammar, testmaxwords,
					grammar.toid[trees[0].label])
			logging.info('estimates done. cpu time elapsed: %gs',
						time.clock() - begin)
			np.savez('outside.npz', outside=outside)
			logging.info('saved estimates')
		elif stage.useestimates == 'SXlrgaps':
			outside = np.load('outside.npz')['outside']
			logging.info('loaded PLCFRS estimates')
		stage.update(grammar=grammar, backtransform=backtransform,
				outside=outside)


def doparsing(**kwds):
	""" Parse a set of sentences using worker processes. """
	params = DictObj(usetags=True, numproc=None, tailmarker='',
		category=None, deletelabel=(), deleteword=(), corpusfmt='export')
	params.update(kwds)
	goldbrackets = multiset()
	totaltokens = 0
	results = [DictObj(name=stage.name) for stage in params.parser.stages]
	for result in results:
		result.update(elapsedtime=dict.fromkeys(params.testset),
				parsetrees=dict.fromkeys(params.testset), brackets=multiset(),
				tagscorrect=0, exact=0, noparse=0)
	if params.numproc == 1:
		initworker(params)
		dowork = (worker(a) for a in params.testset.items())
	else:
		pool = multiprocessing.Pool(processes=params.numproc,
				initializer=initworker, initargs=(params,))
		dowork = pool.imap_unordered(worker, params.testset.items())
	logging.info('going to parse %d sentences.', len(params.testset))
	# main parse loop over each sentence in test corpus
	for nsent, data in enumerate(dowork, 1):
		sentid, msg, sentresults = data
		sent, goldtree, goldsent, _ = params.testset[sentid]
		logging.debug('%d/%d (%s). [len=%d] %s\n%s', nsent, len(params.testset),
					sentid, len(sent),
					' '.join(a[0] for a in goldsent), msg)
		evaltree = goldtree.copy(True)
		evalmod.transform(evaltree, [w for w, _ in sent], evaltree.pos(),
				dict(evaltree.pos()), params.deletelabel, params.deleteword,
				{}, {})
		goldb = evalmod.bracketings(evaltree, dellabel=params.deletelabel)
		goldbrackets.update((sentid, (label, span)) for label, span
				in goldb.elements())
		totaltokens += sum(1 for _, t in goldsent if t not in params.deletelabel)
		for n, result in enumerate(sentresults):
			results[n].brackets.update((sentid, (label, span)) for label, span
					in result.candb.elements())
			assert (results[n].parsetrees[sentid] is None
					and results[n].elapsedtime[sentid] is None)
			results[n].parsetrees[sentid] = result.parsetree
			results[n].elapsedtime[sentid] = result.elapsedtime
			if result.noparse:
				results[n].noparse += 1
			if result.exact:
				results[n].exact += 1
			results[n].tagscorrect += sum(1 for (_, a), (_, b)
					in zip(goldsent, sorted(result.parsetree.pos()))
					if b not in params.deletelabel and a == b)
			logging.debug(
				'%s cov %5.2f tag %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f%s',
					result.name.ljust(7),
					100 * (1 - results[n].noparse / nsent),
					100 * (results[n].tagscorrect / totaltokens),
					100 * (results[n].exact / nsent),
					100 * evalmod.precision(goldbrackets, results[n].brackets),
					100 * evalmod.recall(goldbrackets, results[n].brackets),
					100 * evalmod.f_measure(goldbrackets, results[n].brackets),
					('' if n + 1 < len(sentresults) else '\n'))
	if params.numproc != 1:
		pool.terminate()
		pool.join()
		del dowork, pool

	writeresults(results, params)
	return results, goldbrackets


def worker(args):
	""" Parse a sentence using a global Parser object,
	and do incremental evaluation.

	:returns: a string with diagnostic information, as well as a list of
		DictObj with the results for each stage. """
	nsent, (sent, goldtree, _, _) = args
	prm = INTERNALPARAMS
	goldevaltree = goldtree.copy(True)
	gpos = goldevaltree.pos()
	gposdict = dict(gpos)
	evalmod.transform(goldevaltree, [w for w, _ in sent],
			gpos, gposdict, prm.deletelabel, prm.deleteword, {}, {})
	goldb = evalmod.bracketings(goldevaltree, dellabel=prm.deletelabel)
	results, msg = [], ''
	for result in prm.parser.parse([w for w, _ in sent],
			tags=[t for _, t in sent] if prm.usetags else None):
		msg += result.msg
		evaltree = result.parsetree.copy(True)
		evalsent = [w for w, _ in sent]
		cpos = evaltree.pos()
		evalmod.transform(evaltree, evalsent, cpos, gposdict,
				prm.deletelabel, prm.deleteword, {}, {})
		candb = evalmod.bracketings(evaltree, dellabel=prm.deletelabel)
		prec = rec = f1score = 0
		if goldb and candb:
			prec = evalmod.precision(goldb, candb)
			rec = evalmod.recall(goldb, candb)
			f1score = evalmod.f_measure(goldb, candb)
		if f1score == 1.0:
			exact = True
			msg += '\texact match'
		else:
			exact = False
			msg += '\tLP %5.2f LR %5.2f LF %5.2f\n' % (
					100 * prec, 100 * rec, 100 * f1score)
			if (candb - goldb) or (goldb - candb):
				msg += '\t'
			if candb - goldb:
				msg += 'cand-gold=%s ' % evalmod.strbracketings(candb - goldb)
			if goldb - candb:
				msg += 'gold-cand=%s' % evalmod.strbracketings(goldb - candb)
		msg += '\n'
		result.update(dict(candb=candb, exact=exact))
		results.append(result)
	# visualization of last parse tree; highligh matching POS / bracketings
	highlight = [a for a in evaltree.subtrees()
				if evalmod.bracketing(a) in goldb]
	highlight.extend(a for a in evaltree.subtrees()
				if isinstance(a[0], int) and gpos[a[0]] == cpos[a[0]])
	highlight.extend(range(len(cpos)))
	msg += DrawTree(evaltree, evalsent,
			abbr=True, highlight=highlight).text(
				unicodelines=True, ansi=True)
	return (nsent, msg, results)


def writeresults(results, params):
	""" Write parsing results to files in same format as the original corpus.
	(or export if writer not implemented) """
	ext = {'export': 'export', 'bracket': 'mrg',
			'discbracket': 'dbr', 'alpino': 'xml'}
	category = (params.category + '.') if params.category else ''
	if params.corpusfmt in ('alpino', 'tiger'):
		# convert gold corpus because writing these formats is unsupported
		corpusfmt = 'export'
		io.open('%s/%sgold.%s' % (params.resultdir, category, ext[corpusfmt]),
				'w', encoding='utf-8').writelines(
				writetree(goldtree, [w for w, _ in goldsent], n, corpusfmt,
					morphology=params.morphology)
			for n, (_, goldtree, goldsent, _) in params.testset.items())
	else:
		corpusfmt = params.corpusfmt
		io.open('%s/%sgold.%s' % (params.resultdir, category, ext[corpusfmt]),
				'w', encoding='utf-8').writelines(
				a for _, _, _, a in params.testset.values())
	for res in results:
		io.open('%s/%s%s.%s' % (params.resultdir, category, res.name,
				ext[corpusfmt]), 'w', encoding='utf-8').writelines(
					writetree(res.parsetrees[n], [w for w, _ in goldsent], n,
						corpusfmt, morphology=params.morphology)
				for n, (_, _, goldsent, _) in params.testset.items())
	with open('%s/parsetimes.txt' % params.resultdir, 'w') as out:
		out.write('#id\tlen\t%s\n' % '\t'.join(res.name for res in results))
		out.writelines('%s\t%d\t%s\n' % (n, len(params.testset[n][2]),
				'\t'.join(str(res.elapsedtime[n]) for res in results))
				for n in params.testset)
	logging.info('wrote results to %s/%s%s.%s', params.resultdir, category,
			(('{%s}' % ','.join(res.name for res in results))
			if len(results) > 1 else results[0].name),
			ext[corpusfmt])


def oldeval(results, goldbrackets):
	""" Simple evaluation. """
	nsent = len(results[0].parsetrees)
	if nsent:
		for result in results:
			logging.info('%s lp %5.2f lr %5.2f lf %5.2f\n'
					'coverage %d / %d = %5.2f %%  '
					'exact match %d / %d = %5.2f %%\n',
					result.name,
					100 * evalmod.precision(goldbrackets, result.brackets),
					100 * evalmod.recall(goldbrackets, result.brackets),
					100 * evalmod.f_measure(goldbrackets, result.brackets),
					nsent - result.noparse, nsent,
					100 * (nsent - result.noparse) / nsent,
					result.exact, nsent, 100 * result.exact / nsent)


def readtepacoc():
	""" Read the tepacoc test set. """
	tepacocids = set()
	tepacocsents = defaultdict(list)
	cat = 'undefined'
	tepacoc = io.open('../tepacoc.txt', encoding='utf8')
	for line in tepacoc.read().splitlines():
		fields = line.split('\t')  # = [id, '', sent]
		if line.strip() and len(fields) == 3:
			if fields[0].strip():
				# subtract one because our ids are zero-based, tepacoc 1-based
				sentid = int(fields[0]) - 1
				tepacocids.add(sentid)
				tepacocsents[cat].append((sentid, fields[2].split()))
			else:  # new category
				cat = fields[2]
				if cat.startswith('CUC'):
					cat = 'CUC'
		elif fields[0] == 'TuBa':
			break
	return tepacocids, tepacocsents


def parsetepacoc(
		stages=(dict(mode='pcfg', split=True, markorigin=True),
				dict(mode='plcfrs', prune=True, k=10000, splitprune=True),
				dict(mode='plcfrs', prune=True, k=5000, dop=True,
					usedoubledop=True, estimator='dop1', objective='mpp',
					sample=False, kbest=True)),
		trainmaxwords=999, trainnumsents=25005, testmaxwords=999,
		bintype='binarize', h=1, v=1, factor='right', tailmarker='',
		markhead=False, revmarkov=False, pospa=False,
		leftmostunary=True, rightmostunary=True,
		fanout_marks_before_bin=False, transformations=None,
		usetagger='stanford', resultdir='tepacoc', numproc=1):
	""" Parse the tepacoc test set. """
	for stage in stages:
		for key in stage:
			assert key in DEFAULTSTAGE, 'unrecognized option: %r' % key
	stages = [DictObj({k: stage.get(k, v) for k, v in DEFAULTSTAGE.items()})
			for stage in stages]
	os.mkdir(resultdir)
	# Log everything, and send it to stderr, in a format with just the message.
	formatstr = '%(message)s'
	logging.basicConfig(level=logging.DEBUG, format=formatstr)
	# log up to INFO to a results log file
	fileobj = logging.FileHandler(filename='%s/output.log' % resultdir)
	fileobj.setLevel(logging.INFO)
	fileobj.setFormatter(logging.Formatter(formatstr))
	logging.getLogger('').addHandler(fileobj)
	tepacocids, tepacocsents = readtepacoc()
	try:
		(corpus_sents, corpus_taggedsents,
				corpus_trees, corpus_blocks) = pickle.load(
					gzip.open('tiger.pickle.gz', 'rb'))
	except IOError:  # file not found
		corpus = READERS['export']('../tiger/corpus',
				'tiger_release_aug07.export',
				headrules='negra.headrules' if bintype == 'binarize' else None,
				headfinal=True, headreverse=False, punct='move',
				encoding='iso-8859-1')
		corpus_sents = list(corpus.sents().values())
		corpus_taggedsents = list(corpus.tagged_sents().values())
		corpus_trees = list(corpus.parsed_sents().values())
		if transformations:
			corpus_trees = [transform(tree, sent, transformations)
					for tree, sent in zip(corpus_trees, corpus_sents)]
		corpus_blocks = list(corpus.blocks().values())
		pickle.dump((corpus_sents, corpus_taggedsents, corpus_trees,
			corpus_blocks), gzip.open('tiger.pickle.gz', 'wb'), protocol=-1)

	# test sets (one for each category)
	testsets = {}
	allsents = []
	for cat, catsents in tepacocsents.items():
		testset = sents, trees, goldsents, blocks = [], [], [], []
		for n, sent in catsents:
			if sent != corpus_sents[n]:
				logging.error(
						'mismatch. sent %d:\n%r\n%r\n'
						'not in corpus %r\nnot in tepacoc %r',
						n + 1, sent, corpus_sents[n],
						[a for a, b in zip_longest(sent, corpus_sents[n])
							if a and a != b],
						[b for a, b in zip_longest(sent, corpus_sents[n])
							if b and a != b])
			elif len(corpus_sents[n]) <= testmaxwords:
				sents.append(corpus_taggedsents[n])
				trees.append(corpus_trees[n])
				goldsents.append(corpus_taggedsents[n])
				blocks.append(corpus_blocks[n])
		allsents.extend(sents)
		logging.info('category: %s, %d of %d sentences',
				cat, len(testset[0]), len(catsents))
		testsets[cat] = testset
	testsets['baseline'] = zip(*[sent for n, sent in
				enumerate(zip(corpus_taggedsents, corpus_trees,
						corpus_taggedsents, corpus_blocks))
				if len(sent[1]) <= trainmaxwords
				and n not in tepacocids][trainnumsents:trainnumsents + 2000])
	allsents.extend(testsets['baseline'][0])

	if usetagger:
		overridetags = ('PTKANT', 'VAIMP')
		taglex = defaultdict(set)
		for sent in corpus_taggedsents[:trainnumsents]:
			for word, tag in sent:
				taglex[word].add(tag)
		overridetagdict = {tag:
			{word for word, tags in taglex.items()
			if tags == {tag}} for tag in overridetags}
		tagmap = {'$(': '$[', 'PAV': 'PROAV', 'PIDAT': 'PIAT'}
		# the sentences in the list allsents are modified in-place so that
		# the relevant copy in testsets[cat][0] is updated as well.
		externaltagging(usetagger, '', allsents, overridetagdict, tagmap)

	# training set
	trees, sents, blocks = zip(*[sent for n, sent in
				enumerate(zip(corpus_trees, corpus_sents,
							corpus_blocks)) if len(sent[1]) <= trainmaxwords
							and n not in tepacocids][:trainnumsents])
	getgrammars(trees, sents, stages, bintype, h, v, factor, tailmarker,
			revmarkov, leftmostunary, rightmostunary, pospa, markhead,
			fanout_marks_before_bin, testmaxwords, resultdir,
			numproc, None, False, trees[0].label, None)
	del corpus_sents, corpus_taggedsents, corpus_trees, corpus_blocks
	results = {}
	cnt = 0
	parser = Parser(stages, tailmarker=tailmarker,
			transformations=transformations)
	for cat, testset in sorted(testsets.items()):
		if cat == 'baseline':
			continue
		logging.info('category: %s', cat)
		begin = time.clock()
		results[cat] = doparsing(parser=parser, testset=testset,
				resultdir=resultdir, usetags=True, numproc=numproc,
				category=cat)
		cnt += len(testset[0])
		if numproc == 1:
			logging.info('time elapsed during parsing: %g',
					time.clock() - begin)
		#else:  # wall clock time here
	goldbrackets = multiset()
	totresults = [DictObj(name=stage.name) for stage in stages]
	for result in totresults:
		result.elapsedtime = [None] * cnt
		result.parsetrees = [None] * cnt
		result.brackets = multiset()
		result.exact = result.noparse = 0
	goldblocks = []
	goldsents = []
	for cat, res in results.items():
		logging.info('category: %s', cat)
		goldbrackets |= res[2]
		goldblocks.extend(res[3])
		goldsents.extend(res[4])
		for result, totresult in zip(res[0], totresults):
			totresult.exact += result.exact
			totresult.noparse += result.noparse
			totresult.brackets |= result.brackets
			totresult.elapsedtime.extend(result.elapsedtime)
		oldeval(*res)
	logging.info('TOTAL')
	oldeval(totresults, goldbrackets)
	# write TOTAL results file with all tepacoc sentences (not the baseline)
	for stage in stages:
		open('TOTAL.%s.export' % stage.name, 'w').writelines(
				open('%s.%s.export' % (cat, stage.name)).read()
				for cat in list(results) + ['gold'])
	# do baseline separately because it shouldn't count towards the total score
	cat = 'baseline'
	logging.info('category: %s', cat)
	oldeval(*doparsing(parser=parser, testset=testsets[cat],
			resultdir=resultdir, usetags=True, numproc=numproc, category=cat))


def externaltagging(usetagger, model, sents, overridetag, tagmap):
	""" Use an external tool to tag a list of tagged sentences. """
	logging.info("Start tagging.")
	goldtags = [t for sent in sents.values() for _, t in sent]
	if usetagger == "treetagger":  # Tree-tagger
		assert os.path.exists("tree-tagger/bin/tree-tagger"), """\
tree tagger not found. commands to install:
mkdir tree-tagger && cd tree-tagger
wget ftp://ftp.ims.uni-stuttgart.de/pub/corpora/tree-tagger-linux-3.2.tar.gz
tar -xzf tree-tagger-linux-3.2.tar.gz
wget ftp://ftp.ims.uni-stuttgart.de/pub/corpora/tagger-scripts.tar.gz
tar -xzf ftp://ftp.ims.uni-stuttgart.de/pub/corpora/tagger-scripts.tar.gz
mkdir lib && cd lib && wget \
ftp://ftp.ims.uni-stuttgart.de/pub/corpora/german-par-linux-3.2-utf8.bin.gz
gunzip german-par-linux-3.2-utf8.bin.gz"""
		infile, inname = tempfile.mkstemp(text=True)
		with os.fdopen(infile, 'w') as infile:
			for tagsent in sents.values():
				sent = map(itemgetter(0), tagsent)
				infile.write("\n".join(w.encode('utf-8')
					for n, w in enumerate(sent)) + "\n<S>\n")
		filtertags = ''
		if not model:
			model = "tree-tagger/lib/german-par-linux-3.2-utf8.bin"
			filtertags = "| tree-tagger/cmd/filter-german-tags"
		tagger = Popen("tree-tagger/bin/tree-tagger -token -sgml"
				" %s %s %s" % (model, inname, filtertags),
				stdout=PIPE, shell=True)
		tagout = tagger.stdout.read(  # pylint: disable=E1101
				).decode('utf-8').split("<S>")[:-1]
		os.unlink(inname)
		taggedsents = OrderedDict((n, [tagmangle(a, None, overridetag, tagmap)
					for a in tags.splitlines() if a.strip()])
					for n, tags in zip(sents, tagout))
	elif usetagger == "stanford":  # Stanford Tagger
		assert os.path.exists("stanford-postagger-full-2012-07-09"), """\
Stanford tagger not found. Commands to install:
wget http://nlp.stanford.edu/software/stanford-postagger-full-2012-07-09.tgz
tar -xzf stanford-postagger-full-2012-07-09.tgz"""
		infile, inname = tempfile.mkstemp(text=True)
		with os.fdopen(infile, 'w') as infile:
			for tagsent in sents.values():
				sent = map(itemgetter(0), tagsent)
				infile.write(' '.join(w.encode('utf-8')
					for n, w in enumerate(sent)) + "\n")
		if not model:
			model = "models/german-hgc.tagger"
		tagger = Popen(args=(
				"/usr/bin/java -mx2G -classpath stanford-postagger.jar"
				" edu.stanford.nlp.tagger.maxent.MaxentTagger"
				" -tokenize false -encoding utf-8"
				" -model %s -textFile %s" % (model, inname)).split(),
				cwd="stanford-postagger-full-2012-07-09",
				shell=False, stdout=PIPE)
		tagout = tagger.stdout.read(  # pylint: disable=E1101
				).decode('utf-8').splitlines()
		os.unlink(inname)
		taggedsents = OrderedDict((n, [tagmangle(a, "_", overridetag, tagmap)
			for a in tags.split()]) for n, tags in zip(sents, tagout))
	assert len(taggedsents) == len(sents), (
			"mismatch in number of sentences after tagging.")
	for n, tags in taggedsents.items():
		assert len(sents[n]) == len(tags), (
				"mismatch in number of tokens after tagging.\n"
				"before: %r\nafter: %r" % (sents[n], tags))
	newtags = [t for sent in taggedsents.values() for _, t in sent]
	logging.info("Tag accuracy: %5.2f\ngold - cand: %r\ncand - gold %r",
		(100 * evalmod.accuracy(goldtags, newtags)),
		set(goldtags) - set(newtags), set(newtags) - set(goldtags))
	return taggedsents


def tagmangle(a, splitchar, overridetag, tagmap):
	""" Function to filter tags after they are produced by the tagger. """
	word, tag = a.rsplit(splitchar, 1)
	for newtag in overridetag:
		if word in overridetag[newtag]:
			tag = newtag
	return word, tagmap.get(tag, tag)


def readparam(filename):
	""" Parse a parameter file:
		a list of of attribute=value pairs treated as a dict(). """
	paramstr = open(filename).read()
	params = eval("dict(%s)" % paramstr)
	for stage in params['stages']:
		for key in stage:
			assert key in DEFAULTSTAGE, "unrecognized option: %r" % key
	params['stages'] = [DictObj({k: stage.get(k, v)
			for k, v in DEFAULTSTAGE.items()})
				for stage in params['stages']]
	for n, stage in enumerate(params['stages']):
		assert stage.mode in (
				'plcfrs', 'pcfg', 'pcfg-posterior',
				'pcfg-bitpar', 'dop-rerank')
		assert n > 0 or not stage.prune, (
				"need previous stage to prune, but this stage is first.")
		if stage.mode == 'dop-rerank':
			assert stage.prune and not stage.splitprune and stage.k > 1
			assert (stage.dop and not stage.usedoubledop
					and stage.objective == 'mpp')
		if stage.dop:
			assert stage.estimator in ('dop1', 'ewe', 'bon')
			assert stage.objective in ('mpp', 'mpd', 'shortest',
					"sl-dop", "sl-dop-simple")
	return params


def test():
	""" Not implemented. """


def main(argv=None):
	""" Parse command line arguments. """
	try:
		import faulthandler
		faulthandler.enable()
	except ImportError:
		print('run "pip install faulthandler" to get backtraces on segfaults')
	if argv is None:
		argv = sys.argv
	if len(argv) == 1:
		print(USAGE)
	elif '--tepacoc' in argv:
		parsetepacoc()
	else:
		params = readparam(argv[1])
		resultdir = argv[1].rsplit('.', 1)[0]
		top = startexp(resultdir=resultdir, rerun='--rerun' in argv, **params)
		if 'rerun' not in argv:  # copy parameter file to result dir
			open("%s/params.prm" % resultdir, "w").write(
					"top='%s',\n%s" % (top, open(argv[1]).read()))

if __name__ == '__main__':
	main()
