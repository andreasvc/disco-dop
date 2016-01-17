"""Run an experiment given a parameter file.

Does grammar extraction, parsing, and evaluation."""
from __future__ import division, print_function, absolute_import, \
		unicode_literals
import io
import os
import re
import csv
import sys
import json
import gzip
import time
import codecs
import logging
import multiprocessing
from math import log
from collections import defaultdict, Counter
try:
	from cyordereddict import OrderedDict
except ImportError:
	from collections import OrderedDict
if sys.version_info[0] == 2:
	import cPickle as pickle  # pylint: disable=import-error
	from itertools import izip_longest as zip_longest  # pylint: disable=E0611
else:
	import pickle
	from itertools import zip_longest  # pylint: disable=E0611
import numpy as np
from . import eval as evalmod
from . import __version__, treebank, treebanktransforms, treetransforms, \
		grammar, lexicon, parser, estimates
from .treetransforms import binarizetree
from .util import workerfunc
from .containers import Grammar

INTERNALPARAMS = None


def initworker(params):
	"""Set global parameter object."""
	global INTERNALPARAMS
	# this variable is global because we want to pass it to the fork through
	# inheritance from its parent, instead of through serialization.
	INTERNALPARAMS = params


def startexp(
		prm,  # A DictObj with the structure of parser.DEFAULTS
		resultdir='results',
		rerun=False):
	"""Execute an experiment."""
	if rerun:
		if not os.path.exists(resultdir):
			raise ValueError('Directory %r does not exist.\n--rerun requires a'
					' directory with the grammar(s) of a previous experiment.'
					% resultdir)
	else:
		if os.path.exists(resultdir):
			raise ValueError('Directory %r exists.\n'
					'Use --rerun to parse with existing grammar '
					'and overwrite previous results.' % resultdir)
		os.mkdir(resultdir)

	# Log everything, and send it to stderr, in a format with just the message.
	formatstr = '%(message)s'
	if prm.verbosity == 0:
		logging.basicConfig(level=logging.WARNING, format=formatstr)
	elif prm.verbosity == 1:
		logging.basicConfig(level=logging.INFO, format=formatstr)
	elif prm.verbosity == 2:
		logging.basicConfig(level=logging.DEBUG, format=formatstr)
	elif 3 <= prm.verbosity <= 4:
		logging.basicConfig(level=5, format=formatstr)
	else:
		raise ValueError('verbosity should be >= 0 and <= 4. ')

	# also log to a file
	fileobj = logging.FileHandler(filename='%s/output.log' % resultdir)
	fileobj.setLevel(logging.DEBUG)
	fileobj.setFormatter(logging.Formatter(formatstr))
	logging.getLogger('').addHandler(fileobj)
	logging.info('Disco-DOP %s, running on Python %s',
			__version__, sys.version.split()[0])
	if not rerun:
		trees, sents, train_tagged_sents = loadtraincorpus(
				prm.corpusfmt, prm.traincorpus, prm.binarization, prm.punct,
				prm.functions, prm.morphology, prm.removeempty, prm.ensureroot,
				prm.transformations, prm.relationalrealizational)
	elif isinstance(prm.traincorpus.numsents, float):
		raise ValueError('need to specify number of training set sentences, '
				'not fraction, in rerun mode.')

	testsettb = treebank.READERS[prm.corpusfmt](
			prm.testcorpus.path, encoding=prm.testcorpus.encoding,
			headrules=prm.binarization.headrules,
			removeempty=prm.removeempty, morphology=prm.morphology,
			functions=prm.functions, ensureroot=prm.ensureroot)
	if isinstance(prm.testcorpus.numsents, float):
		prm.testcorpus.numsents = int(prm.testcorpus.numsents
				* len(testsettb.blocks()))
	if prm.testcorpus.skiptrain:
		prm.testcorpus.skip += (  # pylint: disable=maybe-no-member
				prm.traincorpus.numsents)  # pylint: disable=maybe-no-member

	test_blocks = OrderedDict()
	test_trees = OrderedDict()
	test_tagged_sents = OrderedDict()
	for n, item in testsettb.itertrees(
			prm.testcorpus.skip,
			prm.testcorpus.skip  # pylint: disable=no-member
			+ prm.testcorpus.numsents):
		if 1 <= len(item.sent) <= prm.testcorpus.maxwords:
			test_blocks[n] = item.block
			test_trees[n] = item.tree
			test_tagged_sents[n] = [(word, tag) for word, (_, tag)
					in zip(item.sent, sorted(item.tree.pos()))]
	logging.info('%d test sentences after length restriction <= %d',
			len(test_trees), prm.testcorpus.maxwords)
	lexmodel = None
	simplelexsmooth = False
	test_tagged_sents_mangled = test_tagged_sents
	if prm.postagging and prm.postagging.method in (
			'treetagger', 'stanford', 'frog'):
		if prm.postagging.method == 'treetagger':
			# these two tags are never given by tree-tagger,
			# so collect words whose tag needs to be overriden
			overridetags = ('PTKANT', 'PIDAT')
		elif prm.postagging.method == 'stanford':
			overridetags = ('PTKANT', )
		elif prm.postagging.method == 'frog':
			overridetags = ()
		taglex = defaultdict(set)
		for sent in train_tagged_sents:
			for word, tag in sent:
				taglex[word].add(tag)
		overridetagdict = {tag:
			{word for word, tags in taglex.items() if tags == {tag}}
			for tag in overridetags}
		tagmap = {'$(': '$[', 'PAV': 'PROAV'}
		test_tagged_sents_mangled = lexicon.externaltagging(
				prm.postagging.method, prm.postagging.model, test_tagged_sents,
				overridetagdict, tagmap)
		if prm.postagging.retag and not rerun:
			logging.info('re-tagging training corpus')
			sents_to_tag = OrderedDict(enumerate(train_tagged_sents))
			train_tagged_sents = lexicon.externaltagging(prm.postagging.method,
					prm.postagging.model, sents_to_tag, overridetagdict,
					tagmap).values()
			for tree, tagged in zip(trees, train_tagged_sents):
				for node in tree.subtrees(
						lambda n: len(n) == 1 and isinstance(n[0], int)):
					node.label = tagged[node[0]][1]
		usetags = True  # give these tags to parser
	elif prm.postagging and prm.postagging.method == 'unknownword':
		if not rerun:
			sents, lexmodel = getposmodel(prm.postagging, train_tagged_sents)
			simplelexsmooth = prm.postagging.simplelexsmooth
		usetags = False  # make sure gold POS tags are not given to parser
	else:
		usetags = True  # give gold POS tags to parser

	# 0: test sentences as they should be handed to the parser,
	# 1: gold trees for evaluation purposes
	# 2: gold sents because test sentences may be mangled by unknown word model
	# 3: blocks from treebank file to reproduce the relevant part of the
	#   original treebank verbatim.
	testset = OrderedDict((n, (
				test_tagged_sents_mangled[n],
				test_trees[n],
				test_tagged_sents[n],
				block))
			for n, block in test_blocks.items())
	if not test_tagged_sents:
		raise ValueError('test corpus (selection) should be non-empty.')

	if rerun:
		trees, sents = [], []
	roots = {t.label for t in trees} | {test_trees[n].label for n in testset}
	if len(roots) != 1:
		raise ValueError('expected unique ROOT label: %r' % roots)
	top = roots.pop()
	funcclassifier = None

	if rerun:
		parser.readgrammars(resultdir, prm.stages, prm.postagging, top)
		if prm.predictfunctions:
			from sklearn.externals import joblib
			funcclassifier = joblib.load('%s/funcclassifier.pickle' % resultdir)
	else:
		logging.info('read training & test corpus')
		if prm.predictfunctions:
			from sklearn.externals import joblib
			from . import functiontags
			logging.info('training function tag classifier')
			funcclassifier, msg = functiontags.trainfunctionclassifier(
					trees, sents, prm.numproc)
			joblib.dump(funcclassifier, '%s/funcclassifier.pickle' % resultdir,
					compress=3)
			logging.info(msg)
		getgrammars(dobinarization(trees, sents, prm.binarization,
					prm.relationalrealizational),
				sents, prm.stages, prm.testcorpus.maxwords, resultdir,
				prm.numproc, lexmodel, simplelexsmooth, top)
	evalparam = evalmod.readparam(prm.evalparam)
	evalparam['DEBUG'] = -1
	evalparam['CUTOFF_LEN'] = 40
	deletelabel = evalparam.get('DELETE_LABEL', ())
	deleteword = evalparam.get('DELETE_WORD', ())

	begin = time.clock()
	theparser = parser.Parser(prm, funcclassifier=funcclassifier)
	results = doparsing(parser=theparser, testset=testset, resultdir=resultdir,
			usetags=usetags, numproc=prm.numproc, deletelabel=deletelabel,
			deleteword=deleteword, corpusfmt=prm.corpusfmt,
			morphology=prm.morphology, evalparam=evalparam)
	if prm.numproc == 1:
		logging.info('time elapsed during parsing: %gs', time.clock() - begin)
	for result in results:
		nsent = len(result.parsetrees)
		overcutoff = any(len(a) > evalparam['CUTOFF_LEN']
				for a in test_tagged_sents.values())
		header = (' ' + result.name.upper() + ' ').center(
				44 if overcutoff else 35, '=')
		evalsummary = result.evaluator.summary()
		coverage = 'coverage: %s = %6.2f' % (
				('%d / %d' % (nsent - result.noparse, nsent)).rjust(
				25 if overcutoff else 14),
				100.0 * (nsent - result.noparse) / nsent)
		logging.info('\n'.join(('', header, evalsummary, coverage)))
	return top


def loadtraincorpus(corpusfmt, traincorpus, binarization, punct, functions,
		morphology, removeempty, ensureroot, transformations,
		relationalrealizational):
	"""Load the training corpus."""
	train = treebank.READERS[corpusfmt](traincorpus.path,
			encoding=traincorpus.encoding, headrules=binarization.headrules,
			removeempty=removeempty, ensureroot=ensureroot, punct=punct,
			functions=functions, morphology=morphology)
	if isinstance(traincorpus.numsents, float):
		traincorpus.numsents = int(traincorpus.numsents * len(train.sents()))
	trainset = [item for _, item in train.itertrees(None, traincorpus.numsents)
			if 1 <= len(item.sent) <= traincorpus.maxwords]
	trees = [item.tree for item in trainset]
	sents = [item.sent for item in trainset]
	logging.info('%d training sentences after length restriction <= %d',
			len(trees), traincorpus.maxwords)
	if not trees:
		raise ValueError('training corpus (selection) should be non-empty.')
	if transformations:
		newtrees, newsents = [], []
		for tree, sent in zip(trees, sents):
			treebanktransforms.transform(tree, sent, transformations)
			if sent:
				newtrees.append(tree)
				newsents.append(sent)
		trees, sents = newtrees, newsents
	if relationalrealizational:
		trees = [treebanktransforms.rrtransform(
				tree, **relationalrealizational)[0] for tree in trees]
	train_tagged_sents = [[(word, tag) for word, (_, tag)
			in zip(sent, sorted(tree.pos()))]
				for tree, sent in zip(trees, sents)]
	return trees, sents, train_tagged_sents


def getposmodel(postagging, train_tagged_sents):
	"""Apply unknown word model to sentences before extracting grammar."""
	postagging.update(unknownwordfun=lexicon.UNKNOWNWORDFUNC[postagging.model])
	# get smoothed probalities for lexical productions
	lexresults, msg = lexicon.getunknownwordmodel(
			train_tagged_sents, postagging.unknownwordfun,
			postagging.unknownthreshold, postagging.openclassthreshold)
	logging.info(msg)
	simplelexsmooth = postagging.simplelexsmooth
	if simplelexsmooth:
		lexmodel = lexresults[2:8]
	else:
		lexmodel, msg = lexicon.getlexmodel(*lexresults)
		logging.info(msg)
	# NB: knownwords are all words in training set, lexicon is the subset
	# of words that are above the frequency threshold.
	# for training purposes we work with the subset, at test time we
	# exploit the full set of known words from the training set.
	sigs, knownwords, lex = lexresults[:3]
	postagging.update(sigs=sigs, lexicon=knownwords)
	# replace rare train words with signatures
	sents = lexicon.replaceraretrainwords(train_tagged_sents,
			postagging.unknownwordfun, lex)
	return sents, lexmodel


def dobinarization(trees, sents, binarization, relationalrealizational):
	"""Apply binarization to treebank."""
	# fixme: this n should correspond to sentence id
	tbfanout, n = treetransforms.treebankfanout(trees)
	logging.info('treebank fan-out before binarization: %d #%d\n%s\n%s',
			tbfanout, n, trees[n], ' '.join(sents[n]))
	# binarization
	begin = time.clock()
	msg = 'binarization: %s' % binarization.method
	if binarization.fanout_marks_before_bin:
		trees = [treetransforms.addfanoutmarkers(t) for t in trees]
	if binarization.method == 'default':
		msg += ' %s h=%d v=%d %s' % (
				binarization.factor, binarization.h, binarization.v,
				'tailmarker' if binarization.tailmarker else '')
	elif binarization.method == 'optimalhead':
		msg += ' h=%d v=%d' % (
				binarization.h, binarization.v)
	if binarization.method is not None:
		trees = [binarizetree(t, binarization, relationalrealizational)
				for t in trees]
	if binarization.markovthreshold:
		msg1 = treetransforms.markovthreshold(trees,
				binarization.markovthreshold,
				binarization.h + binarization.revh - 1,
				max(binarization.v - 1, 1))
		logging.info(msg1)
	trees = [treetransforms.addfanoutmarkers(t) for t in trees]
	logging.info('%s; cpu time elapsed: %gs',
			msg, time.clock() - begin)
	return trees


def getgrammars(trees, sents, stages, testmaxwords, resultdir,
		numproc, lexmodel, simplelexsmooth, top):
	"""Read off the requested grammars."""
	tbfanout, n = treetransforms.treebankfanout(trees)
	logging.info('binarized treebank fan-out: %d #%d', tbfanout, n)
	mappings = [None for _ in stages]
	for n, stage in enumerate(stages):
		traintrees = trees
		stage.mapping = None
		prevn = 0
		if n and stage.prune:
			prevn = [a.name for a in stages].index(stage.prune)
		if stage.split:
			traintrees = [treetransforms.binarize(
					treetransforms.splitdiscnodes(
						tree.copy(True),
						stage.markorigin),
					childchar=':', dot=True, ids=grammar.UniqueIDs())
					for tree in traintrees]
			logging.info('splitted discontinuous nodes')
		if stage.collapse:
			traintrees, mappings[n] = treebanktransforms.collapselabels(
					[tree.copy(True) for tree in traintrees],
					tbmapping=treebanktransforms.MAPPINGS[
						stage.collapse[0]][stage.collapse[1]])
			logging.info('collapsed phrase labels for multilevel '
					'coarse-to-fine parsing to %s level %d',
					*stage.collapse)
		if n and mappings[prevn] is not None:
			# Given original labels A, convert CTF mapping1 A => C,
			# and mapping2 A => B to a mapping B => C.
			mapping1, mapping2 = mappings[prevn], mappings[n]
			if mappings[n] is None:
				stage.mapping = {a: mapping1[a] for a in mapping1}
			else:
				stage.mapping = {mapping2[a]: mapping1[a] for a in mapping2}
		if stage.mode.startswith('pcfg'):
			if tbfanout != 1 and not stage.split:
				raise ValueError('Cannot extract PCFG from treebank '
						'with discontinuities.')
		backtransform = extrarules = None
		if lexmodel and simplelexsmooth:
			extrarules = lexicon.simplesmoothlexicon(lexmodel)
		if stage.mode == 'mc-rerank':
			from . import _fragments
			gram = parser.DictObj(_fragments.getctrees(zip(trees, sents)))
			tree = gram.trees1.extract(0, gram.vocab)
			gram.start = tree[:tree.index(' ')].lstrip('(')
			with gzip.open('%s/%s.train.pickle.gz' % (resultdir, stage.name),
					'wb') as out:
				out.write(pickle.dumps(gram, protocol=-1))
		elif stage.dop:
			if stage.dop in ('doubledop', 'dop1'):
				if stage.dop == 'doubledop':
					(xgrammar, backtransform,
							altweights, fragments) = grammar.doubledop(
							traintrees, sents, binarized=stage.binarized,
							iterate=stage.iterate, complement=stage.complement,
							numproc=numproc, maxdepth=stage.maxdepth,
							maxfrontier=stage.maxfrontier,
							extrarules=extrarules)
				elif stage.dop == 'dop1':
					(xgrammar, backtransform,
							altweights, fragments) = grammar.dop1(
							traintrees, sents, binarized=stage.binarized,
							maxdepth=stage.maxdepth,
							maxfrontier=stage.maxfrontier,
							extrarules=extrarules)
				# dump fragments
				with codecs.getwriter('utf8')(gzip.open('%s/%s.fragments.gz' %
						(resultdir, stage.name), 'w')) as out:
					out.writelines('%s\t%d\n' % (a, len(b))
							for a, b in fragments)
			elif stage.dop == 'reduction':
				xgrammar, altweights = grammar.dopreduction(
						traintrees, sents, packedgraph=stage.packedgraph,
						extrarules=extrarules)
			else:
				raise ValueError('unrecognized DOP model: %r' % stage.dop)
			nodes = sum(len(list(a.subtrees())) for a in traintrees)
			if lexmodel and not simplelexsmooth:  # FIXME: altweights?
				xgrammar = lexicon.smoothlexicon(xgrammar, lexmodel)
			msg = grammar.grammarinfo(xgrammar)
			rules, lex = grammar.writegrammar(
					xgrammar, bitpar=stage.mode.startswith('pcfg-bitpar'))
			with codecs.getwriter('utf8')(gzip.open('%s/%s.rules.gz' % (
					resultdir, stage.name), 'wb')) as rulesfile:
				rulesfile.write(rules)
			with codecs.getwriter('utf8')(gzip.open('%s/%s.lex.gz' % (
					resultdir, stage.name), 'wb')) as lexiconfile:
				lexiconfile.write(lex)
			gram = Grammar(rules, lex, start=top,
					binarized=stage.binarized)
			for name in altweights:
				gram.register('%s' % name, altweights[name])
			logging.info('DOP model based on %d sentences, %d nodes, '
				'%d nonterminals', len(traintrees), nodes, len(gram.toid))
			logging.info(msg)
			if stage.estimator != 'rfe':
				gram.switch('%s' % stage.estimator)
			logging.info(gram.testgrammar()[1])
			if stage.dop in ('doubledop', 'dop1'):
				# backtransform keys are line numbers to rules file;
				# to see them together do:
				# $ paste <(zcat dop.rules.gz) <(zcat dop.backtransform.gz)
				with codecs.getwriter('utf8')(gzip.open(
						'%s/%s.backtransform.gz' % (resultdir, stage.name),
						'wb')) as out:
					out.writelines('%s\n' % a for a in backtransform)
				if n and stage.prune:
					msg = gram.getmapping(stages[prevn].grammar,
							striplabelre=None if stages[prevn].dop
								else re.compile('@.+$'),
							neverblockre=re.compile('.+}<'),
							splitprune=stage.splitprune and stages[prevn].split,
							markorigin=stages[prevn].markorigin,
							mapping=stage.mapping)
				else:
					# recoverfragments() relies on this mapping to identify
					# binarization nodes
					msg = gram.getmapping(None,
							striplabelre=None,
							neverblockre=re.compile('.+}<'),
							splitprune=False, markorigin=False,
							mapping=stage.mapping)
				logging.info(msg)
			elif n and stage.prune:  # dop reduction
				msg = gram.getmapping(stages[prevn].grammar,
						striplabelre=None if stages[prevn].dop
							and stages[prevn].dop not in ('doubledop', 'dop1')
							else re.compile('@[-0-9]+$'),
						neverblockre=re.compile(stage.neverblockre)
							if stage.neverblockre else None,
						splitprune=stage.splitprune and stages[prevn].split,
						markorigin=stages[prevn].markorigin,
						mapping=stage.mapping)
				if stage.mode == 'dop-rerank':
					gram.getrulemapping(
							stages[prevn].grammar, re.compile(r'@[-0-9]+\b'))
				logging.info(msg)
			# write prob models
			np.savez_compressed('%s/%s.probs.npz' % (resultdir, stage.name),
					**{name: mod for name, mod
						in zip(gram.modelnames, gram.models)})
		else:  # not stage.dop
			xgrammar = grammar.treebankgrammar(traintrees, sents,
					extrarules=extrarules)
			logging.info('induced %s based on %d sentences',
				('PCFG' if tbfanout == 1 or stage.split else 'PLCFRS'),
				len(traintrees))
			if stage.split or os.path.exists('%s/pcdist.txt' % resultdir):
				logging.info(grammar.grammarinfo(xgrammar))
			else:
				logging.info(grammar.grammarinfo(xgrammar,
						dump='%s/pcdist.txt' % resultdir))
			if lexmodel and not simplelexsmooth:
				xgrammar = lexicon.smoothlexicon(xgrammar, lexmodel)
			rules, lex = grammar.writegrammar(
					xgrammar, bitpar=stage.mode.startswith('pcfg-bitpar'))
			with codecs.getwriter('utf8')(gzip.open('%s/%s.rules.gz' % (
					resultdir, stage.name), 'wb')) as rulesfile:
				rulesfile.write(rules)
			with codecs.getwriter('utf8')(gzip.open('%s/%s.lex.gz' % (
					resultdir, stage.name), 'wb')) as lexiconfile:
				lexiconfile.write(lex)
			gram = Grammar(rules, lex, start=top)
			logging.info(gram.testgrammar()[1])
			if n and stage.prune:
				msg = gram.getmapping(stages[prevn].grammar,
					striplabelre=None,
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[prevn].split,
					markorigin=stages[prevn].markorigin,
					mapping=stage.mapping)
				logging.info(msg)
		logging.info('wrote grammar to %s/%s.{rules,lex%s}.gz',
				resultdir, stage.name,
				',backtransform' if stage.dop in ('doubledop', 'dop1') else '')

		outside = None
		if stage.estimates in ('SX', 'SXlrgaps'):
			if stage.estimates == 'SX' and tbfanout != 1 and not stage.split:
				raise ValueError('SX estimate requires PCFG.')
			elif stage.mode != 'plcfrs':
				raise ValueError('estimates require parser w/agenda.')
			begin = time.clock()
			logging.info('computing %s estimates', stage.estimates)
			if stage.estimates == 'SX':
				outside = estimates.getpcfgestimates(gram, testmaxwords,
						gram.toid[trees[0].label])
			elif stage.estimates == 'SXlrgaps':
				outside = estimates.getestimates(gram, testmaxwords,
						gram.toid[trees[0].label])
			logging.info('estimates done. cpu time elapsed: %gs',
					time.clock() - begin)
			np.savez_compressed('%s/%s.outside.npz' % (
					resultdir, stage.name), outside=outside)
			logging.info('saved %s estimates', stage.estimates)
		elif stage.estimates:
			raise ValueError('unrecognized value; specify SX or SXlrgaps.')

		stage.update(grammar=gram, backtransform=backtransform,
				outside=outside)

	if any(stage.mapping is not None for stage in stages):
		with codecs.getwriter('utf8')(gzip.open('%s/mapping.json.gz' % (
				resultdir), 'wb')) as mappingfile:
			mappingfile.write(json.dumps([stage.mapping for stage in stages]))


def doparsing(**kwds):
	"""Parse a set of sentences using worker processes."""
	params = parser.DictObj(usetags=True, numproc=None, tailmarker='',
		category=None, deletelabel=(), deleteword=(), corpusfmt='export')
	params.update(kwds)
	results = [parser.DictObj(name=stage.name)
			for stage in params.parser.stages]
	for result in results:
		result.update(
				parsetrees=dict.fromkeys(params.testset),
				sents=dict.fromkeys(params.testset),
				logprob=dict.fromkeys(params.testset, float('nan')),
				frags=dict.fromkeys(params.testset, 0),
				numitems=dict.fromkeys(params.testset, 0),
				golditems=dict.fromkeys(params.testset, 0),
				totalgolditems=dict.fromkeys(params.testset, 0),
				elapsedtime=dict.fromkeys(params.testset),
				evaluator=evalmod.Evaluator(params.evalparam), noparse=0)
	if params.numproc == 1:
		initworker(params)
		dowork = (worker(a) for a in params.testset.items())
	else:
		pool = multiprocessing.Pool(processes=params.numproc,
				initializer=initworker, initargs=(params,))
		dowork = pool.imap_unordered(
				mpworker, params.testset.items())
	logging.info('going to parse %d sentences.', len(params.testset))
	# main parse loop over each sentence in test corpus
	for nsent, data in enumerate(dowork, 1):
		sentid, sent, sentresults = data
		_sent, goldtree, goldsent, _ = params.testset[sentid]
		goldsent = [w for w, _t in goldsent]
		logging.debug('%d/%d (%s). [len=%d] %s\n',
				nsent, len(params.testset), sentid, len(sent),
				' '.join(goldsent))
		for n, result in enumerate(sentresults):
			assert (results[n].parsetrees[sentid] is None
					and results[n].elapsedtime[sentid] is None)
			results[n].parsetrees[sentid] = result.parsetree
			results[n].sents[sentid] = sent
			if isinstance(result.prob, tuple):
				results[n].logprob[sentid] = [log(a) for a in result.prob
						if isinstance(a, float) and 0 < a <= 1][0]
				results[n].frags[sentid] = ([abs(a) for a in result.prob
						if isinstance(a, int)] or [None])[0]
			elif isinstance(result.prob, float):
				try:
					results[n].logprob[sentid] = log(result.prob)
				except ValueError:
					results[n].logprob[sentid] = 300.0
			if result.fragments is not None:
				results[n].frags[sentid] = len(result.fragments)
			results[n].numitems[sentid] = result.numitems
			results[n].golditems[sentid] = result.golditems
			results[n].totalgolditems[sentid] = result.totalgolditems
			results[n].elapsedtime[sentid] = result.elapsedtime
			if result.noparse:
				results[n].noparse += 1

			sentmetrics = results[n].evaluator.add(
					sentid, goldtree.copy(True), goldsent,
					result.parsetree.copy(True), sent)
			msg = result.msg
			scores = sentmetrics.scores()
			msg += '\tPOS %(POS)s ' % scores
			if not scores['FUN'].endswith('nan'):
				msg += 'FUN %(FUN)s ' % scores
			if scores['LF'] == '100.00':
				msg += 'LF exact match'
			else:
				msg += 'LF %(LF)s' % scores
				try:
					msg += '\n\t' + sentmetrics.bracketings()
				except Exception as err:  # pylint: disable=broad-except
					msg += 'PROBLEM bracketings:\n%s\n%s' % (
							result.parsetree, err)
			msg += '\n'
			if n + 1 == len(sentresults):
				try:
					msg += sentmetrics.visualize()
				except Exception as err:  # pylint: disable=broad-except
					msg += 'PROBLEM drawing tree:\n%s\n%s' % (
							sentmetrics.ctree, err)
			logging.debug(msg)
		msg = ''
		for n, result in enumerate(sentresults):
			metrics = results[n].evaluator.acc.scores()
			msg += ('%(name)s cov %(cov)5.2f; tag %(tag)s; %(fun1)s'
					'ex %(ex)s; lp %(lp)s; lr %(lr)s; lf %(lf)s\n' % dict(
					name=result.name.ljust(7),
					cov=100 * (1 - results[n].noparse / nsent),
					fun1='' if metrics['fun'].endswith('nan') else
						('fun %(fun)s; ' % metrics),
					**metrics))
		logging.debug(msg)
	if params.numproc != 1:
		pool.terminate()
		pool.join()
		del dowork, pool

	writeresults(results, params)
	return results


@workerfunc
def mpworker(args):
	"""Multiprocessing wrapper of ``worker``."""
	return worker(args)


def worker(args):
	"""Parse a sentence using global Parser object, and evaluate incrementally.

	:returns: a string with diagnostic information, as well as a list of
		DictObj instances with the results for each stage."""
	nsent, (tagged_sent, goldtree, _, _) = args
	sent = [w for w, _ in tagged_sent]
	prm = INTERNALPARAMS
	results = list(prm.parser.parse(sent,
			tags=[t for _, t in tagged_sent] if prm.usetags else None,
			goldtree=goldtree))  # only used to determine quality of pruning
	return (nsent, sent, results)


def writeresults(results, params):
	"""Write parsing results to files in same format as the original corpus.
	(Or export if writer not implemented)."""
	ext = {'export': 'export', 'bracket': 'mrg',
			'discbracket': 'dbr', 'alpino': 'xml'}
	category = (params.category + '.') if params.category else ''
	if params.corpusfmt in ('alpino', 'tiger'):
		# convert gold corpus because writing these formats is unsupported
		corpusfmt = 'export'
		io.open('%s/%sgold.%s' % (params.resultdir, category, ext[corpusfmt]),
				'w', encoding='utf8').writelines(
				treebank.writetree(goldtree, [w for w, _ in goldsent], n,
					corpusfmt, morphology=params.morphology)
			for n, (_, goldtree, goldsent, _) in params.testset.items())
	else:
		corpusfmt = params.corpusfmt
		io.open('%s/%sgold.%s' % (params.resultdir, category, ext[corpusfmt]),
				'w', encoding='utf8').writelines(
				a for _, _, _, a in params.testset.values())
	for res in results:
		io.open('%s/%s%s.%s' % (params.resultdir, category, res.name,
				ext[corpusfmt]), 'w', encoding='utf8').writelines(
					treebank.writetree(
						res.parsetrees[n], res.sents[n], n, corpusfmt,
						morphology=params.morphology)
				for n in params.testset)

	if sys.version_info[0] == 2:
		fileobj = open('%s/stats.tsv' % params.resultdir, 'wb')
	else:
		fileobj = open('%s/stats.tsv' % params.resultdir, 'w',
				encoding='utf8', newline='')
	with fileobj as out:
		fields = ['sentid', 'len', 'stage', 'elapsedtime', 'logprob', 'frags',
				'numitems', 'golditems', 'totalgolditems']
		writer = csv.writer(out, dialect='excel-tab')
		writer.writerow(fields)
		writer.writerows([n, len(params.testset[n][2]), res.name]
				+ [getattr(res, field)[n] for field in fields[3:]]
				for n in params.testset
					for res in results)

	logging.info('wrote results to %s/%s%s.%s', params.resultdir, category,
			(('{%s}' % ','.join(res.name for res in results))
			if len(results) > 1 else results[0].name),
			ext[corpusfmt])


def oldeval(results, goldbrackets):
	"""Simple evaluation."""
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
	"""Read the tepacoc test set."""
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
				dict(mode='plcfrs', prune=True, k=5000, dop='doubledop',
					estimator='rfe', objective='mpp',
					sample=False, kbest=True)),
		trainmaxwords=999, trainnumsents=25005, testmaxwords=999,
		binarization=parser.DictObj(
			method='default', h=1, v=1, factor='right', tailmarker='',
			headrules='negra.headrules',
			leftmostunary=True, rightmostunary=True,
			markhead=False, fanout_marks_before_bin=False),
		transformations=None, usetagger='stanford', resultdir='tepacoc',
		numproc=1):
	"""Parse the tepacoc test set."""
	for stage in stages:
		for key in stage:
			if key not in parser.DEFAULTSTAGE:
				raise ValueError('unrecognized option: %r' % key)
	stages = [parser.DictObj({k: stage.get(k, v) for k, v
			in parser.DEFAULTSTAGE.items()}) for stage in stages]
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
		corpus = treebank.READERS['export'](
				'../tiger/corpus/tiger_release_aug07.export',
				headrules=binarization.headrules,
				headfinal=True, headreverse=False, punct='move',
				encoding='iso-8859-1')
		corpus_sents = list(corpus.sents().values())
		corpus_taggedsents = list(corpus.tagged_sents().values())
		corpus_trees = list(corpus.trees().values())
		if transformations:
			for tree, sent in zip(corpus_trees, corpus_sents):
				treebanktransforms.transform(tree, sent, transformations)
		corpus_blocks = list(corpus.blocks().values())
		with gzip.open('tiger.pickle.gz', 'wb') as out:
			pickle.dump((corpus_sents, corpus_taggedsents, corpus_trees,
					corpus_blocks), out, protocol=-1)

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
		lexicon.externaltagging(
				usetagger, '', allsents, overridetagdict, tagmap)

	# training set
	trees, sents, blocks = zip(*[sent for n, sent in
				enumerate(zip(corpus_trees, corpus_sents,
							corpus_blocks)) if len(sent[1]) <= trainmaxwords
							and n not in tepacocids][:trainnumsents])
	getgrammars(dobinarization(trees, sents, binarization, False),
			sents, stages, testmaxwords, resultdir,
			numproc, None, False, trees[0].label)
	del corpus_sents, corpus_taggedsents, corpus_trees, corpus_blocks
	results = {}
	cnt = 0
	params = parser.DictObj(parser.DEFAULTS)
	params.update(stages=stages, binarization=binarization,
			transformations=transformations)
	theparser = parser.Parser(params)
	for cat, testset in sorted(testsets.items()):
		if cat == 'baseline':
			continue
		logging.info('category: %s', cat)
		begin = time.clock()
		results[cat] = doparsing(parser=theparser, testset=testset,
				resultdir=resultdir, usetags=True, numproc=numproc,
				category=cat)
		cnt += len(testset[0])
		if numproc == 1:
			logging.info('time elapsed during parsing: %g',
					time.clock() - begin)
		# else:  # wall clock time here
	goldbrackets = Counter()
	totresults = [parser.DictObj(name=stage.name) for stage in stages]
	for result in totresults:
		result.elapsedtime = [None] * cnt
		result.parsetrees = [None] * cnt
		result.brackets = Counter()
		result.exact = result.noparse = 0
	goldblocks = []
	goldsents = []
	# FIXME
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
		with io.open('TOTAL.%s.export' % stage.name,
				'w', encoding='utf8') as tmp:
			tmp.writelines(io.open('%s.%s.export' % (cat, stage.name),
				encoding='utf8').read() for cat in list(results) + ['gold'])
	# do baseline separately because it shouldn't count towards the total score
	cat = 'baseline'
	logging.info('category: %s', cat)
	oldeval(*doparsing(parser=theparser, testset=testsets[cat],
			resultdir=resultdir, usetags=True, numproc=numproc, category=cat))


__all__ = ['initworker', 'startexp', 'loadtraincorpus', 'getposmodel',
		'dobinarization', 'getgrammars', 'doparsing', 'worker', 'writeresults',
		'oldeval', 'readtepacoc', 'parsetepacoc']
