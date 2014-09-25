"""Run an experiment given a parameter file.

Does grammar extraction, parsing, and evaluation."""
from __future__ import division, print_function
import io
import os
import re
import sys
import time
import gzip
import codecs
import logging
import multiprocessing
from math import log
from itertools import islice
from collections import defaultdict, OrderedDict, Counter as multiset
if sys.version[0] >= '3':
	import pickle
	from itertools import zip_longest  # pylint: disable=E0611
else:
	import cPickle as pickle
	from itertools import izip_longest as zip_longest
import numpy as np
from discodop import eval as evalmod
from discodop import treebank, treebanktransforms, treetransforms, \
		grammar, lexicon, parser, estimates
from discodop.tree import Tree, ParentedTree
from discodop.containers import Grammar

USAGE = '''Usage: %s <parameter file> [--rerun]
If a parameter file is given, an experiment is run. See the file sample.prm for
an example parameter file. To repeat an experiment with an existing grammar,
pass the option --rerun. The directory with the name of the parameter file
without extension must exist in the current path; its results will be
overwritten.''' % sys.argv[0]

INTERNALPARAMS = None
DEFAULTS = dict(
	traincorpus=dict(
		# filenames may include globbing characters '*' and '?'.
		path='alpinosample.export',
		encoding='utf-8',
		maxwords=40,  # limit on train set sentences
		numsents=2),  # size of train set (before applying maxwords)
	testcorpus=dict(
		path='alpinosample.export',
		encoding='utf-8',
		maxwords=40,  # test set length limit
		numsents=1,  # size of test set (before length limit)
		skiptrain=True,  # test set starts after training set
		# (useful when they are in the same file)
		skip=0),  # number of sentences to skip from test corpus
	binarization=dict(
		method='default',  # choices: default, optimal, optimalhead
		factor='right',
		headrules=None,  # rules for finding heads of constituents
		v=1,
		h=2,
		markhead=False,  # prepend head to siblings
		leftmostunary=True,  # start binarization with unary node
		rightmostunary=True,  # end binarization with unary node
		tailmarker='',  # with headrules, head is last node and can be marked
		revmarkov=True,  # reverse horizontal markovization
		fanout_marks_before_bin=False))


def initworker(params):
	"""Set global parameter object."""
	global INTERNALPARAMS
	INTERNALPARAMS = params


def startexp(
		stages=(parser.DictObj(parser.DEFAULTSTAGE), ),  # see parser module
		corpusfmt='export',  # choices: export, (disc)bracket, alpino, tiger
		traincorpus=parser.DictObj(DEFAULTS['traincorpus']),
		testcorpus=parser.DictObj(DEFAULTS['testcorpus']),
		binarization=parser.DictObj(DEFAULTS['binarization']),
		removeempty=False,  # whether to remove empty terminals
		punct=None,  # choices: None, 'move', 'remove', 'root'
		functions=None,  # choices None, 'add', 'remove', 'replace'
		morphology=None,  # choices: None, 'add', 'replace', 'between'
		transformations=None,  # apply treebank transformations
		postagging=None,  # postagging: pass None to use tags from treebank.
		relationalrealizational=None,  # do not apply RR-transform
		evalparam='proper.prm',  # EVALB-style parameter file
		verbosity=2,
		numproc=1,  # increase to use multiple CPUs; None: use all CPUs.
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
	if verbosity == 0:
		logging.basicConfig(level=logging.WARNING, format=formatstr)
	elif verbosity == 1:
		logging.basicConfig(level=logging.INFO, format=formatstr)
	elif verbosity == 2:
		logging.basicConfig(level=logging.DEBUG, format=formatstr)
	elif 3 <= verbosity <= 4:
		logging.basicConfig(level=5, format=formatstr)
	else:
		raise ValueError('verbosity should be >= 0 and <= 4. ')

	# also log to a file
	fileobj = logging.FileHandler(filename='%s/output.log' % resultdir)
	fileobj.setLevel(logging.DEBUG)
	fileobj.setFormatter(logging.Formatter(formatstr))
	logging.getLogger('').addHandler(fileobj)

	if not rerun:
		trees, sents, train_tagged_sents = loadtraincorpus(
				corpusfmt, traincorpus, binarization, punct, functions,
				morphology, removeempty, transformations,
				relationalrealizational)

	testset = treebank.READERS[corpusfmt](
			testcorpus.path, encoding=testcorpus.encoding,
			removeempty=removeempty, morphology=morphology,
			functions=functions)
	gold_sents = testset.tagged_sents()
	test_trees = testset.trees()
	if testcorpus.skiptrain:
		testcorpus.skip += traincorpus.numsents  # pylint: disable=E1103
	logging.info('%d sentences in test corpus %s',
			len(test_trees), testcorpus.path)
	logging.info('%d test sentences before length restriction',
			len(list(gold_sents)[testcorpus.skip:  # pylint: disable=E1103
				testcorpus.skip + testcorpus.numsents]))  # pylint: disable=E1103
	lexmodel = None
	simplelexsmooth = False
	test_tagged_sents = gold_sents
	if postagging and postagging.method in ('treetagger', 'stanford', 'frog'):
		if postagging.method == 'treetagger':
			# these two tags are never given by tree-tagger,
			# so collect words whose tag needs to be overriden
			overridetags = ('PTKANT', 'PIDAT')
		elif postagging.method == 'stanford':
			overridetags = ('PTKANT', )
		elif postagging.method == 'frog':
			overridetags = ()
		taglex = defaultdict(set)
		for sent in train_tagged_sents:
			for word, tag in sent:
				taglex[word].add(tag)
		overridetagdict = {tag:
			{word for word, tags in taglex.items() if tags == {tag}}
			for tag in overridetags}
		tagmap = {'$(': '$[', 'PAV': 'PROAV'}
		sents_to_tag = OrderedDict((a, b) for a, b
				in islice(gold_sents.items(),
					testcorpus.skip,  # pylint: disable=E1103
					testcorpus.skip  # pylint: disable=E1103
						+ testcorpus.numsents)
				if len(b) <= testcorpus.maxwords)
		test_tagged_sents = lexicon.externaltagging(postagging.method,
				postagging.model, sents_to_tag, overridetagdict, tagmap)
		if postagging.retag:
			logging.info('re-tagging training corpus')
			sents_to_tag = OrderedDict(enumerate(train_tagged_sents))
			train_tagged_sents = lexicon.externaltagging(postagging.method,
					postagging.model, sents_to_tag, overridetagdict,
					tagmap).values()
			for tree, tagged in zip(trees, train_tagged_sents):
				for node in tree.subtrees(
						lambda n: len(n) == 1 and isinstance(n[0], int)):
					node.label = tagged[node[0]][1]
		usetags = True  # give these tags to parser
	elif postagging and postagging.method == 'unknownword':
		if not rerun:
			sents, lexmodel = getposmodel(postagging, train_tagged_sents)
			simplelexsmooth = postagging.simplelexsmooth
		usetags = False  # make sure gold POS tags are not given to parser
	else:
		usetags = True  # give gold POS tags to parser

	# 0: test sentences as they should be handed to the parser,
	# 1: gold trees for evaluation purposes
	# 2: gold sents because test sentences may be mangled by unknown word model
	# 3: blocks from treebank file to reproduce the relevant part of the
	#   original treebank verbatim.
	testset = OrderedDict((a, (test_tagged_sents[a], test_trees[a],
			gold_sents[a], block)) for a, block
			in islice(testset.blocks().items(),
				testcorpus.skip,  # pylint: disable=E1103
				testcorpus.skip + testcorpus.numsents)  # pylint: disable=E1103
			if a in test_tagged_sents and
				1 <= len(test_tagged_sents[a]) <= testcorpus.maxwords)
	if not test_tagged_sents:
		raise ValueError('test corpus should be non-empty.')
	logging.info('%d test sentences after length restriction <= %d',
			len(testset), testcorpus.maxwords)

	if rerun:
		trees, sents = [], []
	roots = {t.label for t in trees} | {test_trees[n].label for n in testset}
	if len(roots) != 1:
		raise ValueError('expected unique ROOT label: %r' % roots)
	top = roots.pop()

	if rerun:
		parser.readgrammars(resultdir, stages, postagging, top)
	else:
		logging.info('read training & test corpus')
		getgrammars(dobinarization(trees, sents, binarization,
					relationalrealizational),
				sents, stages, testcorpus.maxwords, resultdir, numproc,
				lexmodel, simplelexsmooth, top)
	evalparam = evalmod.readparam(evalparam)
	evalparam['DEBUG'] = -1
	evalparam['CUTOFF_LEN'] = 40
	deletelabel = evalparam.get('DELETE_LABEL', ())
	deleteword = evalparam.get('DELETE_WORD', ())

	begin = time.clock()
	theparser = parser.Parser(stages, transformations=transformations,
			binarization=binarization, postagging=postagging if postagging
				and postagging.method == 'unknownword' else None,
			relationalrealizational=relationalrealizational,
			verbosity=verbosity)
	results = doparsing(parser=theparser, testset=testset, resultdir=resultdir,
			usetags=usetags, numproc=numproc, deletelabel=deletelabel,
			deleteword=deleteword, corpusfmt=corpusfmt, morphology=morphology,
			evalparam=evalparam)
	if numproc == 1:
		logging.info('time elapsed during parsing: %gs', time.clock() - begin)
	for result in results:
		nsent = len(result.parsetrees)
		overcutoff = any(len(a) > evalparam['CUTOFF_LEN']
				for a in gold_sents.values())
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
		morphology, removeempty, transformations, relationalrealizational):
	"""Load the training corpus."""
	train = treebank.READERS[corpusfmt](traincorpus.path,
			encoding=traincorpus.encoding, headrules=binarization.headrules,
			headfinal=True, headreverse=False, removeempty=removeempty,
			punct=punct, functions=functions, morphology=morphology)
	traintrees = train.trees()
	trainsents = train.sents()
	logging.info('%d sentences in training corpus %s',
			len(traintrees), traincorpus.path)
	if isinstance(traincorpus.numsents, float):
		traincorpus.numsents = int(traincorpus.numsents * len(trainsents))
	trees = list(traintrees.values())[:traincorpus.numsents]
	sents = list(trainsents.values())[:traincorpus.numsents]
	if not trees:
		raise ValueError('training corpus should be non-empty.')
	logging.info('%d training sentences before length restriction', len(trees))
	trees, sents = zip(*[sent for sent in zip(trees, sents)
		if 1 <= len(sent[1]) <= traincorpus.maxwords])
	logging.info('%d training sentences after length restriction <= %d',
			len(trees), traincorpus.maxwords)
	if transformations:
		trees = [treebanktransforms.transform(tree, sent, transformations)
				for tree, sent in zip(trees, sents)]
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
	"""Apply binarization."""
	# fixme: this n should correspond to sentence id
	tbfanout, n = treebank.treebankfanout(trees)
	logging.info('treebank fan-out before binarization: %d #%d\n%s\n%s',
			tbfanout, n, trees[n], ' '.join(sents[n]))
	# binarization
	begin = time.clock()
	msg = 'binarization: %s' % binarization.method
	if binarization.fanout_marks_before_bin:
		trees = [treetransforms.addfanoutmarkers(t) for t in trees]
	if binarization.method is None:
		pass
	elif binarization.method == 'default':
		msg += ' %s h=%d v=%d %s' % (
				binarization.factor, binarization.h, binarization.v,
				'tailmarker' if binarization.tailmarker else '')
		for a in trees:
			treetransforms.binarize(a, factor=binarization.factor,
					tailmarker=binarization.tailmarker,
					horzmarkov=binarization.h, vertmarkov=binarization.v,
					leftmostunary=binarization.leftmostunary,
					rightmostunary=binarization.rightmostunary,
					reverse=binarization.revmarkov,
					headidx=-1 if binarization.markhead else None,
					filterfuncs=(relationalrealizational['ignorefunctions']
						+ (relationalrealizational['adjunctionlabel'], ))
						if relationalrealizational else ())
	elif binarization.method == 'optimal':
		trees = [Tree.convert(treetransforms.optimalbinarize(tree))
						for n, tree in enumerate(trees)]
	elif binarization.method == 'optimalhead':
		msg += ' h=%d v=%d' % (
				binarization.h, binarization.v)
		trees = [Tree.convert(treetransforms.optimalbinarize(
				tree, headdriven=True, h=binarization.h, v=binarization.v))
				for n, tree in enumerate(trees)]
	trees = [treetransforms.addfanoutmarkers(t) for t in trees]
	logging.info('%s; cpu time elapsed: %gs',
			msg, time.clock() - begin)
	trees = [treetransforms.canonicalize(a).freeze() for a in trees]
	return trees


def getgrammars(trees, sents, stages, testmaxwords, resultdir,
		numproc, lexmodel, simplelexsmooth, top):
	"""Read off the requested grammars."""
	tbfanout, n = treebank.treebankfanout(trees)
	logging.info('binarized treebank fan-out: %d #%d', tbfanout, n)
	for n, stage in enumerate(stages):
		if stage.split:
			traintrees = [treetransforms.binarize(
					treetransforms.splitdiscnodes(
						Tree.convert(a), stage.markorigin),
					childchar=':', dot=True, ids=grammar.UniqueIDs()).freeze()
					for a in trees]
			logging.info('splitted discontinuous nodes')
		else:
			traintrees = trees
		if stage.mode.startswith('pcfg'):
			if tbfanout != 1 and not stage.split:
				raise ValueError('Cannot extract PCFG from treebank '
						'with discontinuities.')
		backtransform = None
		if stage.dop:
			if stage.usedoubledop:
				(xgrammar, backtransform, altweights, fragments
					) = grammar.doubledop(
						traintrees, sents, binarized=stage.binarized,
						iterate=stage.iterate, complement=stage.complement,
						numproc=numproc)
				# dump fragments
				with codecs.getwriter('utf-8')(gzip.open('%s/%s.fragments.gz' %
						(resultdir, stage.name), 'w')) as out:
					out.writelines('%s\t%d\n' % (treebank.writetree(a, b, 0,
							'bracket' if stage.mode.startswith('pcfg')
							else 'discbracket').rstrip(), sum(c.values()))
							for (a, b), c in fragments.items())
			else:  # DOP reduction
				xgrammar, altweights = grammar.dopreduction(
						traintrees, sents, packedgraph=stage.packedgraph)
			nodes = sum(len(list(a.subtrees())) for a in traintrees)
			if lexmodel and simplelexsmooth:
				newrules = lexicon.simplesmoothlexicon(lexmodel)
				xgrammar.extend(newrules)
				for model, weights in altweights.items():
					if model == u'shortest':
						weights.extend(0.5 for _ in newrules)
					else:
						weights.extend(w1 / w2 for _, (w1, w2) in newrules)
				grammar.sortgrammar(xgrammar, altweights)
			elif lexmodel:
				xgrammar = lexicon.smoothlexicon(xgrammar, lexmodel)
			msg = grammar.grammarinfo(xgrammar)
			rules, lex = grammar.write_lcfrs_grammar(
					xgrammar, bitpar=stage.mode.startswith('pcfg'))
			gram = Grammar(rules, lex, start=top,
					bitpar=stage.mode.startswith('pcfg'),
					binarized=stage.binarized)
			for name in altweights:
				gram.register(u'%s' % name, altweights[name])
			with gzip.open('%s/%s.rules.gz' % (
					resultdir, stage.name), 'wb') as rulesfile:
				rulesfile.write(rules)
			with codecs.getwriter('utf-8')(gzip.open('%s/%s.lex.gz' % (
					resultdir, stage.name), 'wb')) as lexiconfile:
				lexiconfile.write(lex)
			logging.info('DOP model based on %d sentences, %d nodes, '
				'%d nonterminals', len(traintrees), nodes, len(gram.toid))
			logging.info(msg)
			if stage.estimator != 'rfe':
				gram.switch(u'%s' % stage.estimator)
			logging.info(gram.testgrammar()[1])
			if stage.usedoubledop:
				# backtransform keys are line numbers to rules file;
				# to see them together do:
				# $ paste <(zcat dop.rules.gz) <(zcat dop.backtransform.gz)
				with codecs.getwriter('ascii')(gzip.open(
						'%s/%s.backtransform.gz' % (resultdir, stage.name),
						'w')) as out:
					out.writelines('%s\n' % a for a in backtransform)
				if n and stage.prune:
					msg = gram.getmapping(stages[n - 1].grammar,
						striplabelre=None if stages[n - 1].dop
							else re.compile(b'@.+$'),
						neverblockre=re.compile(b'.+}<'),
						splitprune=stage.splitprune and stages[n - 1].split,
						markorigin=stages[n - 1].markorigin)
				else:
					# recoverfragments() relies on this mapping to identify
					# binarization nodes
					msg = gram.getmapping(None,
						striplabelre=None,
						neverblockre=re.compile(b'.+}<'),
						splitprune=False, markorigin=False)
				logging.info(msg)
			elif n and stage.prune:  # dop reduction
				msg = gram.getmapping(stages[n - 1].grammar,
					striplabelre=None if stages[n - 1].dop
						and not stages[n - 1].usedoubledop
						else re.compile(b'@[-0-9]+$'),
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[n - 1].split,
					markorigin=stages[n - 1].markorigin)
				if stage.mode == 'dop-rerank':
					gram.getrulemapping(
							stages[n - 1].grammar, re.compile(br'@[-0-9]+\b'))
				logging.info(msg)
			# write prob models
			np.savez_compressed(  # pylint: disable=no-member
					'%s/%s.probs.npz' % (resultdir, stage.name),
					**{name: mod for name, mod
						in zip(gram.modelnames, gram.models)})
		else:  # not stage.dop
			xgrammar = grammar.treebankgrammar(traintrees, sents)
			logging.info('induced %s based on %d sentences',
				('PCFG' if tbfanout == 1 or stage.split else 'PLCFRS'),
				len(traintrees))
			if stage.split or os.path.exists('%s/pcdist.txt' % resultdir):
				logging.info(grammar.grammarinfo(xgrammar))
			else:
				logging.info(grammar.grammarinfo(xgrammar,
						dump='%s/pcdist.txt' % resultdir))
			if lexmodel and simplelexsmooth:
				newrules = lexicon.simplesmoothlexicon(lexmodel)
				xgrammar.extend(newrules)  # pylint: disable=E1103
			elif lexmodel:
				xgrammar = lexicon.smoothlexicon(xgrammar, lexmodel)
			rules, lex = grammar.write_lcfrs_grammar(
					xgrammar, bitpar=stage.mode.startswith('pcfg'))
			gram = Grammar(rules, lex, start=top,
					bitpar=stage.mode.startswith('pcfg'))
			with gzip.open('%s/%s.rules.gz' % (
					resultdir, stage.name), 'wb') as rulesfile:
				rulesfile.write(rules)
			with codecs.getwriter('utf-8')(gzip.open('%s/%s.lex.gz' % (
					resultdir, stage.name), 'wb')) as lexiconfile:
				lexiconfile.write(lex)
			logging.info(gram.testgrammar()[1])
			if n and stage.prune:
				msg = gram.getmapping(stages[n - 1].grammar,
					striplabelre=None,
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[n - 1].split,
					markorigin=stages[n - 1].markorigin)
				logging.info(msg)
		logging.info('wrote grammar to %s/%s.{rules,lex%s}.gz', resultdir,
				stage.name, ',backtransform' if stage.usedoubledop else '')

		outside = None
		if 'SX' in (stage.getestimates, stage.useestimates):
			if tbfanout != 1 and not stage.split:
				raise ValueError('SX estimate requires PCFG.')
			if stage.mode != 'plcfrs':
				raise ValueError('estimates require parser w/agenda.')
		if stage.getestimates in ('SX', 'SXlrgaps'):
			begin = time.clock()
			logging.info('computing %s estimates', stage.getestimates)
			if stage.getestimates == 'SX':
				outside = estimates.getpcfgestimates(gram, testmaxwords,
						gram.toid[trees[0].label])
			elif stage.getestimates == 'SXlrgaps':
				outside = estimates.getestimates(gram, testmaxwords,
						gram.toid[trees[0].label])
			logging.info('estimates done. cpu time elapsed: %gs',
					time.clock() - begin)
			np.savez_compressed(  # pylint: disable=no-member
					'%s/%s.outside.npz' % (resultdir, stage.name),
					outside=outside)
			logging.info('saved %s estimates', stage.getestimates)
		elif stage.useestimates in ('SX', 'SXlrgaps'):
			outside = np.load(  # pylint: disable=no-member
					'%s/%s.outside.npz' % (resultdir, stage.name))['outside']
			logging.info('loaded %s estimates', stage.useestimates)

		stage.update(grammar=gram, backtransform=backtransform,
				outside=outside)


def doparsing(**kwds):
	"""Parse a set of sentences using worker processes."""
	params = parser.DictObj(usetags=True, numproc=None, tailmarker='',
		category=None, deletelabel=(), deleteword=(), corpusfmt='export')
	params.update(kwds)
	results = [parser.DictObj(name=stage.name)
			for stage in params.parser.stages]
	for result in results:
		result.update(parsetrees=dict.fromkeys(params.testset),
				probs=dict.fromkeys(params.testset, float('nan')),
				frags=dict.fromkeys(params.testset, 0),
				elapsedtime=dict.fromkeys(params.testset),
				evaluator=evalmod.Evaluator(params.evalparam), noparse=0)
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
		sentid, sentresults = data
		sent, goldtree, goldsent, _ = params.testset[sentid]
		goldsent = [w for w, _t in goldsent]
		msg = '%d/%d (%s). [len=%d] %s\n' % (nsent,
				len(params.testset), sentid, len(sent),
				' '.join(goldsent))
		for n, result in enumerate(sentresults):
			assert (results[n].parsetrees[sentid] is None
					and results[n].elapsedtime[sentid] is None)
			results[n].parsetrees[sentid] = result.parsetree
			if isinstance(result.prob, tuple):
				results[n].probs[sentid] = [log(a) for a in result.prob
						if isinstance(a, float)][0]
				results[n].frags[sentid] = [abs(a) for a in result.prob
						if isinstance(a, int)][0]
			elif isinstance(result.prob, float):
				results[n].probs[sentid] = log(result.prob)
			if result.fragments is not None:
				results[n].frags[sentid] = len(result.fragments)
			results[n].elapsedtime[sentid] = result.elapsedtime
			if result.noparse:
				results[n].noparse += 1

			sentmetrics = results[n].evaluator.add(sentid,
					goldtree.copy(True), goldsent,
					ParentedTree.convert(result.parsetree), goldsent)
			msg += result.msg
			if sentmetrics.scores()['LF'] == '100.00':
				msg += '\texact match'
			else:
				msg += '\tLP %(LP)s LR %(LR)s LF %(LF)s' % sentmetrics.scores()
				try:
					msg += '\n\t' + sentmetrics.bracketings()
				except Exception as err:
					msg += 'PROBLEM bracketings:\n%s\n%s' % (
							result.parsetree, err)
			msg += '\n'
			if n + 1 == len(sentresults):
				try:
					msg += sentmetrics.visualize()
				except Exception as err:
					msg += 'PROBLEM drawing tree:\n%s\n%s' % (
							sentmetrics.ctree, err)
		for n, result in enumerate(sentresults):
			metrics = results[n].evaluator.acc.scores()
			msg += ('%(name)s cov %(cov)5.2f tag %(tag)s ex %(ex)s '
					'lp %(lp)s lr %(lr)s lf %(lf)s\n' % dict(
					name=result.name.ljust(7),
					cov=100 * (1 - results[n].noparse / nsent),
					**metrics))
		logging.debug(msg)
	if params.numproc != 1:
		pool.terminate()
		pool.join()
		del dowork, pool

	writeresults(results, params)
	return results


@parser.workerfunc
def worker(args):
	"""Parse a sentence using global Parser object, and evaluate incrementally.

	:returns: a string with diagnostic information, as well as a list of
		DictObj instances with the results for each stage."""
	nsent, (sent, _, _, _) = args
	prm = INTERNALPARAMS
	results = list(prm.parser.parse([w for w, _ in sent],
			tags=[t for _, t in sent] if prm.usetags else None))
	return (nsent, results)


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
				'w', encoding='utf-8').writelines(
				treebank.writetree(goldtree, [w for w, _ in goldsent], n,
					corpusfmt, morphology=params.morphology)
			for n, (_, goldtree, goldsent, _) in params.testset.items())
	else:
		corpusfmt = params.corpusfmt
		io.open('%s/%sgold.%s' % (params.resultdir, category, ext[corpusfmt]),
				'w', encoding='utf-8').writelines(
				a for _, _, _, a in params.testset.values())
	for res in results:
		io.open('%s/%s%s.%s' % (params.resultdir, category, res.name,
				ext[corpusfmt]), 'w', encoding='utf-8').writelines(
					treebank.writetree(res.parsetrees[n],
						[w for w, _ in goldsent], n, corpusfmt,
						morphology=params.morphology)
				for n, (_, _, goldsent, _) in params.testset.items())
	with open('%s/parsetimes.txt' % params.resultdir, 'w') as out:
		out.write('#id\tlen\t%s\n' % '\t'.join(res.name for res in results))
		out.writelines('%s\t%d\t%s\n' % (n, len(params.testset[n][2]),
				'\t'.join(str(res.elapsedtime[n]) for res in results))
				for n in params.testset)
	with open('%s/logprobs.txt' % params.resultdir, 'w') as out:
		out.write('#id\tlen\t%s\n' % '\t'.join(res.name for res in results))
		out.writelines('%s\t%d\t%s\n' % (n, len(params.testset[n][2]),
				'\t'.join(str(res.probs[n]) for res in results))
				for n in params.testset)
	names = [res.name for res, stage in zip(results, params.parser.stages)
			if stage.usedoubledop]
	if names:
		with open('%s/numfrags.txt' % params.resultdir, 'w') as out:
			out.write('#id\tlen\t%s\n' % '\t'.join(names))
			out.writelines('%s\t%d\t%s\n' % (n, len(params.testset[n][2]),
					'\t'.join(str(res.frags[n]) for res, stage
						in zip(results, params.parser.stages)
						if stage.usedoubledop))
					for n in params.testset)
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
				dict(mode='plcfrs', prune=True, k=5000, dop=True,
					usedoubledop=True, estimator='rfe', objective='mpp',
					sample=False, kbest=True)),
		trainmaxwords=999, trainnumsents=25005, testmaxwords=999,
		binarization=parser.DictObj(
			method='default', h=1, v=1, factor='right', tailmarker='',
			headrules='negra.headrules',
			leftmostunary=True, rightmostunary=True,
			markhead=False, revmarkov=False, fanout_marks_before_bin=False),
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
			corpus_trees = [treebanktransforms.transform(
					tree, sent, transformations) for tree, sent
					in zip(corpus_trees, corpus_sents)]
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
	theparser = parser.Parser(stages, binarization=binarization,
			transformations=transformations)
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
	goldbrackets = multiset()
	totresults = [parser.DictObj(name=stage.name) for stage in stages]
	for result in totresults:
		result.elapsedtime = [None] * cnt
		result.parsetrees = [None] * cnt
		result.brackets = multiset()
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
		open('TOTAL.%s.export' % stage.name, 'w').writelines(
				open('%s.%s.export' % (cat, stage.name)).read()
				for cat in list(results) + ['gold'])
	# do baseline separately because it shouldn't count towards the total score
	cat = 'baseline'
	logging.info('category: %s', cat)
	oldeval(*doparsing(parser=theparser, testset=testsets[cat],
			resultdir=resultdir, usetags=True, numproc=numproc, category=cat))


def readparam(filename):
	"""Parse a parameter file.

	The file should contain a list of comma-separated ``attribute=value`` pairs
	and will be read using ``eval('dict(%s)' % open(file).read())``."""
	params = eval('dict(%s)' % open(filename).read())
	for key in DEFAULTS:
		if key not in params:
			raise ValueError('%r not in parameters.' % key)
	for stage in params['stages']:
		for key in stage:
			if key not in parser.DEFAULTSTAGE:
				raise ValueError('unrecognized option: %r' % key)
	params['stages'] = [parser.DictObj({k: stage.get(k, v)
			for k, v in parser.DEFAULTSTAGE.items()})
				for stage in params['stages']]
	for key in DEFAULTS:
		params[key] = parser.DictObj({k: params[key].get(k, v)
				for k, v in DEFAULTS[key].items()})
	for n, stage in enumerate(params['stages']):
		if stage.mode not in (
				'plcfrs', 'pcfg', 'pcfg-posterior',
				'pcfg-bitpar-nbest', 'pcfg-bitpar-forest',
				'dop-rerank'):
			raise ValueError('unrecognized mode argument.')
		if n == 0 and stage.prune:
			raise ValueError('need previous stage to prune, '
					'but this stage is first.')
		if stage.mode == 'dop-rerank':
			assert stage.prune and not stage.splitprune and stage.k > 1
			assert (stage.dop and not stage.usedoubledop
					and stage.objective == 'mpp')
		if stage.dop:
			assert stage.estimator in ('rfe', 'ewe', 'bon')
			assert stage.objective in ('mpp', 'mpd', 'mcc', 'shortest',
					'sl-dop', 'sl-dop-simple')
		assert stage.binarized or stage.mode == 'pcfg-bitpar-nbest', (
				'non-binarized grammar requires mode "pcfg-bitpar-nbest"')
	assert params['binarization'].method in (
			None, 'default', 'optimal', 'optimalhead')
	postagging = params['postagging']
	if postagging is not None:
		assert set(postagging).issubset({'method', 'model', 'retag',
				'unknownthreshold', 'openclassthreshold', 'simplelexsmooth'})
		postagging.setdefault('retag', False)
		postagging = params['postagging'] = parser.DictObj(postagging)
		if postagging.method == 'unknownword':
			assert postagging.model in lexicon.UNKNOWNWORDFUNC
			assert postagging.unknownthreshold >= 1
			assert postagging.openclassthreshold >= 0
		else:
			assert postagging.method in ('treetagger', 'stanford', 'frog')
	return params


def test():
	"""Run ``sample.prm``."""
	if os.path.exists('sample.prm') and os.path.exists('sample/'):
		for path in os.listdir('sample/'):
			os.remove('sample/' + path)
		os.rmdir('sample/')
	main(['runexp.py', 'sample.prm'])


def main(argv=None):
	"""Parse command line arguments."""
	import faulthandler
	faulthandler.enable()
	if argv is None:
		argv = sys.argv
	if len(argv) == 1:
		print(USAGE)
	elif '--tepacoc' in argv:
		parsetepacoc()
	else:
		rerun = '--rerun' in argv
		if rerun:
			argv.remove('--rerun')
		params = readparam(argv[1])
		resultdir = argv[1].rsplit('.', 1)[0]
		top = startexp(resultdir=resultdir, rerun=rerun, **params)
		if not rerun:  # copy parameter file to result dir
			open(os.path.join(resultdir, 'params.prm'), 'w').write(
					"top='%s',\n%s" % (top, open(argv[1]).read()))


__all__ = ['initworker', 'startexp', 'loadtraincorpus', 'getposmodel',
		'dobinarization', 'getgrammars', 'doparsing', 'worker', 'writeresults',
		'oldeval', 'readtepacoc', 'parsetepacoc', 'readparam']

if __name__ == '__main__':
	main()
