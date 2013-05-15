# -*- coding: UTF-8 -*-
""" Run an experiment given a parameter file. Reads of grammars, does parsing
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
from fractions import Fraction
from math import exp
import numpy as np
from .tree import Tree
from .treebank import getreader, fold, writetree, FUNC
from .treetransforms import binarize, unbinarize, optimalbinarize, \
		splitdiscnodes, mergediscnodes, canonicalize, \
		addfanoutmarkers, removefanoutmarkers, addbitsets, fastfanout
from .fragments import getfragments
from .grammar import induce_plcfrs, dopreduction, doubledop, grammarinfo, \
		write_lcfrs_grammar, defaultparse
from .lexicon import getunknownwordmodel, getlexmodel, \
		smoothlexicon, simplesmoothlexicon, replacerarewords, \
		unknownword4, unknownword6, unknownwordbase
from .eval import doeval, readparam, strbracketings, transform, \
		bracketings, precision, recall, f_measure, accuracy
from . import plcfrs, pcfg
from .estimates import getestimates, getpcfgestimates
from .containers import Grammar
from .coarsetofine import prunechart, whitelistfromposteriors
from .disambiguation import marginalize, viterbiderivation

USAGE = """Usage: %s [--rerun] parameter file
If a parameter file is given, an experiment is run. See the file sample.prm for
an example parameter file. To repeat an experiment with an existing grammar,
pass the option --rerun.""" % sys.argv[0]

INTERNALPARAMS = None


def initworker(params):
	""" Set global parameter object """
	global INTERNALPARAMS
	INTERNALPARAMS = params

DEFAULTSTAGE = dict(
		name='stage1',  # identifier, used for filenames
		mode='plcfrs',  # use the agenda-based PLCFRS parser
		prune=False,  # whether to use previous chart to prune this stage
		split=False,  # split disc. nodes VP_2[101] as { VP*[100], VP*[001] }
		splitprune=False,  # VP_2[101] is treated as {VP*[100], VP*[001]} for pruning
		markorigin=False,  # mark origin of split nodes: VP_2 => {VP*1, VP*2}
		k=50,  # no. of coarse pcfg derivations to prune with; k=0 => filter only
		neverblockre=None,  # do not prune nodes with label that match regex
		getestimates=None,  # compute & store estimates
		useestimates=None,  # load & use estimates
		dop=False,  # enable DOP mode (DOP reduction / double DOP)
		packedgraph=False,  # use packed graph encoding for DOP reduction
		usedoubledop=False,  # when False, use DOP reduction instead
		iterate=False,  # for double dop, whether to include fragments of fragments
		complement=False,  # for double dop, whether to include fragments which
				# form the complement of the maximal recurring fragments extracted
		sample=False, kbest=True,
		m=10000,  # number of derivations to sample/enumerate
		estimator="ewe",  # choices: dop1, ewe
		objective="mpp",  # choices: mpp, mpd, shortest, sl-dop[-simple]
			# NB: w/shortest derivation, estimator only affects tie breaking.
		sldop_n=7)


def startexp(
		stages=(
		# see variable 'DEFAULTSTAGE' above
		dict(
			mode='pcfg',  # use the dedicated PCFG parser
			split=True,
			markorigin=True,  # mark origin of split nodes: VP_2 => {VP*1, VP*2}
		),
		dict(
			mode='plcfrs',  # the agenda-based PLCFRS parser
			prune=True,  # whether to use previous chart to prune this stage
			splitprune=True,  # VP_2[101] is treated as { VP*[100], VP*[001] }
			k=50,  # number of coarse pcfg derivations to prune with;
					# k=0 => filter only
		),
		dict(
			mode='plcfrs',  # the agenda-based PLCFRS parser
			prune=True,  # whether to use previous chart to prune this stage
			k=50,		# number of coarse plcfrs derivations to prune with;
					# k=0 => filter only
			dop=True,
			usedoubledop=False,  # when False, use DOP reduction instead
			sample=False, kbest=True,
			m=10000,		# number of derivations to sample/enumerate
			estimator="ewe",  # choices: dop1, ewe
			objective="mpp",  # choices: mpp, mpd, shortest, sl-dop[-simple]
		)),
		corpusfmt="export",  # choices: export, discbracket, bracket
		corpusdir=".",
		# filenames may include globbing characters '*' and '?'.
		traincorpus="sample2.export", trainencoding="iso-8859-1",
		testcorpus="sample2.export", testencoding="iso-8859-1",
		punct=None,  # options: move, remove, root
		functiontags=False,  # whether to add/strip function tags from node labels
		unfolded=False,
		testmaxwords=40,
		trainmaxwords=40,
		trainnumsents=2,
		testnumsents=1,  # number of sentences to parse
		skiptrain=True,  # test set starts after training set
		# (useful when they are in the same file)
		skip=0,  # number of sentences to skip from test corpus
		# postagging: pass None to use tags from treebank.
		postagging=dict(
			method="unknownword",
			# choices: unknownword (assign during parsing),
			# 		treetagger, stanford (external taggers)
			# choices unknownword: 4, 6, base,
			# for treetagger / stanford: [filename of external tagger model]
			model="4",
			# options for unknown word models:
			unknownthreshold=1,  # use probs of rare words for unknown words
			openclassthreshold=50,  # add unseen tags for known words; 0: disable
			simplelexsmooth=True,  # disable sophisticated smoothing
		),
		morphaspos=False,  # use morphological tags as POS tags
		bintype="binarize",  # choices: binarize, optimal, optimalhead
		factor="right",
		revmarkov=True,
		v=1,
		h=2,
		leftmostunary=True,  # start binarization with unary node
		rightmostunary=True,  # end binarization with unary node
		pospa=False,  # when v > 1, add parent annotation to POS tags?
		headrules=None,  # rules for finding heads of constituents
		fanout_marks_before_bin=False,
		tailmarker="",
		evalparam="proper.prm",  # EVALB-style parameter file
		quiet=False, reallyquiet=False,  # quiet=no per sentence results
		numproc=1,  # increase to use multiple CPUs; set to None to use all CPUs.
		resultdir='results',
		rerun=False):
	""" Execute an experiment. """
	assert bintype in ("optimal", "optimalhead", "binarize")
	if postagging is not None:
		assert set(postagging).issubset({"method", "model",
				"unknownthreshold", "openclassthreshold", "simplelexsmooth"})
		if postagging['method'] == "unknownword":
			assert postagging['model'] in ("4", "6", "base")
			assert postagging['unknownthreshold'] >= 1
			assert postagging['openclassthreshold'] >= 0
		else:
			assert postagging['method'] in ("treetagger", "stanford")
	for stage in stages:
		for key in stage:
			assert key in DEFAULTSTAGE, "unrecognized option: %r" % key
	stages = [DictObj({k: stage.get(k, v) for k, v in DEFAULTSTAGE.items()})
			for stage in stages]

	if rerun:
		assert os.path.exists(resultdir), (
				"Directory %r does not exist."
				"--rerun requires a directory "
				"with the grammar of a previous experiment."
				% resultdir)
	else:
		assert not os.path.exists(resultdir), (
			"Directory %r exists.\n"
			"Use --rerun to parse with existing grammar "
			"and overwrite previous results." % resultdir)
		os.mkdir(resultdir)

	# Log everything, and send it to stderr, in a format with just the message.
	formatstr = '%(message)s'
	if reallyquiet:
		logging.basicConfig(level=logging.WARNING, format=formatstr)
	elif quiet:
		logging.basicConfig(level=logging.INFO, format=formatstr)
	else:
		logging.basicConfig(level=logging.DEBUG, format=formatstr)

	# log up to INFO to a results log file
	fileobj = logging.FileHandler(filename='%s/output.log' % resultdir)
	fileobj.setLevel(logging.INFO)
	fileobj.setFormatter(logging.Formatter(formatstr))
	logging.getLogger('').addHandler(fileobj)

	corpusreader = getreader(corpusfmt)
	if not rerun:
		corpus = corpusreader(corpusdir, traincorpus, encoding=trainencoding,
			headrules=headrules, headfinal=True, headreverse=False,
			punct=punct, functiontags=functiontags, dounfold=unfolded,
			morphaspos=morphaspos)
		logging.info("%d sentences in training corpus %s/%s",
				len(corpus.parsed_sents()), corpusdir, traincorpus)
		if isinstance(trainnumsents, float):
			trainnumsents = int(trainnumsents * len(corpus.sents()))
		trees = list(corpus.parsed_sents().values())[:trainnumsents]
		sents = list(corpus.sents().values())[:trainnumsents]
		train_tagged_sents = list(corpus.tagged_sents().values())[:trainnumsents]
		blocks = list(corpus.blocks().values())[:trainnumsents]
		assert trees, "training corpus should be non-empty"
		logging.info("%d training sentences before length restriction",
				len(trees))
		trees, sents, blocks = zip(*[sent for sent in zip(trees, sents, blocks)
			if len(sent[1]) <= trainmaxwords])
		logging.info("%d training sentences after length restriction <= %d",
			len(trees), trainmaxwords)

	testset = corpusreader(corpusdir, testcorpus, encoding=testencoding,
			punct=punct, morphaspos=morphaspos, functiontags=functiontags)
	gold_sents = testset.tagged_sents()
	test_parsed_sents = testset.parsed_sents()
	if skiptrain:
		skip += trainnumsents
	logging.info("%d sentences in test corpus %s/%s",
			len(testset.parsed_sents()), corpusdir, testcorpus)
	logging.info("%d test sentences before length restriction",
			len(list(gold_sents.keys())[skip:skip + testnumsents]))
	lexmodel = None
	if postagging and postagging['method'] in ('treetagger', 'stanford'):
		if postagging['method'] == 'treetagger':
			# these two tags are never given by tree-tagger,
			# so collect words whose tag needs to be overriden
			overridetags = ("PTKANT", "PIDAT")
		elif postagging['method'] == 'stanford':
			overridetags = ("PTKANT", )
		taglex = defaultdict(set)
		for sent in train_tagged_sents:
			for word, tag in sent:
				taglex[word].add(tag)
		overridetagdict = {tag:
			{word for word, tags in taglex.items() if tags == {tag}}
			for tag in overridetags}
		tagmap = {"$(": "$[", "PAV": "PROAV"}
		test_tagged_sents = dotagging(postagging['method'], postagging['model'],
				OrderedDict((a, b) for a, b
				in islice(gold_sents.items(), skip, skip + testnumsents)
				if len(b) <= testmaxwords),
				overridetagdict, tagmap)
		# give these tags to parser
		tags = True
	elif postagging and postagging['method'] == "unknownword":
		# resolve name to function assigning unknown word signatures
		unknownword = {"4": unknownword4,
				"6": unknownword6,
				"base": unknownwordbase}[postagging['model']]
		# get smoothed probalities for lexical productions
		lexresults, msg = getunknownwordmodel(
				train_tagged_sents, unknownword,
				unknownthreshold=postagging['unknownthreshold'],
				openclassthreshold=postagging['openclassthreshold'])
		logging.info(msg)
		simplelexsmooth = postagging['simplelexsmooth']
		if simplelexsmooth:
			lexmodel = lexresults[2:8]
		else:
			lexmodel, msg = getlexmodel(*lexresults)
			logging.info(msg)
		wordclass, knownwords = lexresults[:2]
		# replace rare train words with features
		sents = replacerarewords(train_tagged_sents, unknownword,
				postagging['unknownthreshold'], knownwords, lexresults[4])
		# replace unknown test words with features
		test_tagged_sents = OrderedDict()
		for n, sent in gold_sents.items():
			newsent = []
			for m, (word, _) in enumerate(sent):
				if word in knownwords:
					sig = word
				else:
					sig = unknownword(word, m, knownwords)
					if sig not in wordclass:
						sig = "UNK"
				newsent.append((sig, None))  # tag will not be used, use None.
			test_tagged_sents[n] = newsent
		# make sure gold tags are not given to parser
		tags = False
	else:
		simplelexsmooth = False
		test_tagged_sents = gold_sents
		# give gold POS tags to parser
		tags = True

	# - test sentences as they should be handed to the parser,
	# - gold trees for evaluation purposes
	# - gold sentence because test sentences may be mangled by unknown word
	#   model
	# - blocks from treebank file to reproduce the relevant part of the
	#   original treebank verbatim.
	testset = OrderedDict((a, (test_tagged_sents[a], test_parsed_sents[a],
			gold_sents[a], block)) for a, block
			in islice(testset.blocks().items(), skip, skip + testnumsents)
			if len(test_tagged_sents[a]) <= testmaxwords)
	assert test_tagged_sents, "test corpus should be non-empty"
	logging.info("%d test sentences after length restriction <= %d",
			len(testset), testmaxwords)

	if rerun:
		trees = []
		sents = []
	toplabels = {tree.label for tree in trees} | {
			test_parsed_sents[n].label for n in testset}
	assert len(toplabels) == 1, "expected unique TOP/ROOT label: %r" % toplabels
	top = toplabels.pop()
	if rerun:
		readgrammars(resultdir, stages, top)
	else:
		logging.info("read training & test corpus")
		getgrammars(trees, sents, stages, bintype, h, v, factor, tailmarker,
				revmarkov, leftmostunary, rightmostunary, pospa,
				fanout_marks_before_bin, testmaxwords, resultdir, numproc,
				lexmodel, simplelexsmooth, top)
	evalparam = readparam(evalparam)
	evalparam["DEBUG"] = -1
	evalparam["CUTOFF_LEN"] = 40
	deletelabel = evalparam.get("DELETE_LABEL", ())
	deleteword = evalparam.get("DELETE_WORD", ())

	begin = time.clock()
	results = doparse(stages=stages, unfolded=unfolded, bintype=bintype,
			fanout_marks_before_bin=fanout_marks_before_bin, testset=testset,
			testmaxwords=testmaxwords, testnumsents=testnumsents,
			tags=tags, resultdir=resultdir, numproc=numproc,
			tailmarker=tailmarker, deletelabel=deletelabel,
			deleteword=deleteword, corpusfmt=corpusfmt)
	if numproc == 1:
		logging.info("time elapsed during parsing: %gs", time.clock() - begin)
	for result in results[0]:
		nsent = len(result.parsetrees)
		header = (" " + result.name.upper() + " ").center(35, "=")
		evalsummary = doeval(OrderedDict((a, b.copy(True))
				for a, b in test_parsed_sents.items()), gold_sents,
				result.parsetrees, test_tagged_sents if tags else gold_sents,
				evalparam)
		coverage = "coverage: %s = %6.2f" % (
				("%d / %d" % (nsent - result.noparse, nsent)).rjust(
				25 if any(len(a) > evalparam["CUTOFF_LEN"]
				for a in gold_sents.values()) else 14),
				100.0 * (nsent - result.noparse) / nsent)
		logging.info("\n".join(("", header, evalsummary, coverage)))


def readgrammars(resultdir, stages, top):
	""" Read the grammars from a previous experiment. Must have same parameters.
	"""
	for n, stage in enumerate(stages):
		logging.info("reading: %s", stage.name)
		rules = gzip.open("%s/%s.rules.gz" % (resultdir, stage.name))
		lexicon = codecs.getreader('utf-8')(gzip.open("%s/%s.lex.gz" % (
				resultdir, stage.name)))
		grammar = Grammar(rules.read(), lexicon.read(),
				start=top, bitpar=stage.mode == 'pcfg')
		backtransform = None
		if stage.dop:
			assert stage.objective not in (
					"shortest", "sl-dop", "sl-dop-simple"), ("not supported.")
			assert stage.useestimates is None, "not supported"
			if stage.usedoubledop:
				backtransform = dict(enumerate(
						gzip.open("%s/%s.backtransform.gz" % (resultdir,
						stage.name)).read().splitlines()))
				if n and stage.prune:
					_ = grammar.getmapping(stages[n - 1].grammar,
						striplabelre=re.compile(b'@.+$'),
						neverblockre=re.compile(b'^#[0-9]+|.+}<'),
						splitprune=stage.splitprune and stages[n - 1].split,
						markorigin=stages[n - 1].markorigin)
				else:
					# recoverfragments() relies on this mapping to identify
					# binarization nodes
					_ = grammar.getmapping(None,
						neverblockre=re.compile(b'.+}<'))
			elif n and stage.prune:  # dop reduction
				_ = grammar.getmapping(stages[n - 1].grammar,
					striplabelre=re.compile(b'@[-0-9]+$'),
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[n - 1].split,
					markorigin=stages[n - 1].markorigin)
		else:  # not stage.dop
			if n and stage.prune:
				_ = grammar.getmapping(stages[n - 1].grammar,
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[n - 1].split,
					markorigin=stages[n - 1].markorigin)
		grammar.testgrammar()
		stage.update(grammar=grammar, backtransform=backtransform,
				secondarymodel=None, outside=None)


def getgrammars(trees, sents, stages, bintype, horzmarkov, vertmarkov, factor,
		tailmarker, revmarkov, leftmostunary, rightmostunary, pospa,
		fanout_marks_before_bin, testmaxwords, resultdir, numproc,
		lexmodel, simplelexsmooth, top):
	""" Apply binarization and read off the requested grammars. """
	# fixme: this n should correspond to sentence id
	fanout, n = treebankfanout(trees)
	logging.info("treebank fan-out before binarization: %d #%d\n%s\n%s",
			fanout, n, trees[n], " ".join(sents[n]))
	# binarization
	begin = time.clock()
	if fanout_marks_before_bin:
		trees = [addfanoutmarkers(t) for t in trees]
	if bintype == "binarize":
		bintype += " %s h=%d v=%d %s" % (factor, horzmarkov, vertmarkov,
			"tailmarker" if tailmarker else '')
		for a in trees:
			binarize(a, factor=factor, tailmarker=tailmarker,
					horzmarkov=horzmarkov, vertmarkov=vertmarkov,
					leftmostunary=leftmostunary, rightmostunary=rightmostunary,
					reverse=revmarkov, pospa=pospa)
	elif bintype == "optimal":
		trees = [Tree.convert(optimalbinarize(tree))
						for n, tree in enumerate(trees)]
	elif bintype == "optimalhead":
		trees = [Tree.convert(optimalbinarize(tree, headdriven=True,
				h=horzmarkov, v=vertmarkov)) for n, tree in enumerate(trees)]
	trees = [addfanoutmarkers(t) for t in trees]
	logging.info("binarized %s cpu time elapsed: %gs",
						bintype, time.clock() - begin)
	logging.info("binarized treebank fan-out: %d #%d", *treebankfanout(trees))
	trees = [canonicalize(a).freeze() for a in trees]

	for n, stage in enumerate(stages):
		assert stage.mode in ("plcfrs", "pcfg", "pcfg-posterior")
		if stage.split:
			traintrees = [binarize(splitdiscnodes(Tree.convert(a),
					stage.markorigin), childchar=":").freeze() for a in trees]
			logging.info("splitted discontinuous nodes")
		else:
			traintrees = trees
		assert n > 0 or not stage.prune, (
				"need previous stage to prune, but this stage is first.")
		secondarymodel = backtransform = None
		if stage.dop:
			assert stage.estimator in ("dop1", "ewe")
			assert stage.objective in ("mpp", "mpd", "shortest",
					"sl-dop", "sl-dop-simple")
			if stage.usedoubledop:
				# find recurring fragments in treebank,
				# as well as depth-1 'cover' fragments
				fragments = getfragments(traintrees, sents, numproc,
						iterate=stage.iterate, complement=stage.complement,
						indices=stage.estimator == "ewe")
				xgrammar, backtransform = doubledop(fragments,
						ewe=stage.estimator == "ewe")
				half = Fraction(1, 2)
				if (stage.objective == "shortest"
						or stage.objective.startswith("sl-dop")):
					# any rule corresponding to the introduction of a
					# fragment has a probability of 0.5, else 1.
					shortest = [(r, 1 if ("}" in r[0][0] or "@" in r[0][0])
							else half) for r, _ in xgrammar]
					if stage.objective == "shortest":
						# use RFE for tie breaking of shortest derivations
						# Bod (2000) uses the ranks of subtree frequencies for
						# each root node.
						secondarymodel = dict(xgrammar)
						xgrammar = shortest
					elif stage.objective.startswith("sl-dop"):
						secondarymodel = dict(shortest)
			elif stage.objective == "shortest":  # dopreduction from here on
				# the secondary model is used to resolve ties
				# for the shortest derivation
				# i.e., secondarymodel is probabilistic
				xgrammar, secondarymodel = dopreduction(traintrees, sents,
					ewe=stage.estimator == "ewe", shortestderiv=True)
			elif "sl-dop" in stage.objective:
				# here secondarymodel is non-probabilistic
				xgrammar = dopreduction(traintrees, sents,
						ewe=stage.estimator == "ewe", shortestderiv=False)
				secondarymodel, _ = dopreduction(traintrees, sents,
								ewe=False, shortestderiv=True)
				secondarymodel = Grammar(secondarymodel, start=top)
			else:  # mpp or mpd
				xgrammar = dopreduction(traintrees, sents,
					ewe=(stage.estimator in ("ewe", "sl-dop",
					"sl-dop-simple")), shortestderiv=False,
					packedgraph=stage.packedgraph)
			nodes = sum(len(list(a.subtrees())) for a in traintrees)
			if lexmodel and simplelexsmooth:
				xgrammar = simplesmoothlexicon(xgrammar, lexmodel)
			elif lexmodel:
				xgrammar = smoothlexicon(xgrammar, lexmodel)
			msg = grammarinfo(xgrammar)
			grammar = Grammar(xgrammar, start=top)
			logging.info("DOP model based on %d sentences, %d nodes, "
				"%d nonterminals", len(traintrees), nodes, len(grammar.toid))
			logging.info(msg)
			sumsto1 = grammar.testgrammar()
			if stage.usedoubledop:
				# backtransform keys are line numbers to rules file;
				# to see them together do:
				# $ paste <(zcat dop.rules.gz) <(zcat dop.backtransform.gz)
				with gzip.open("%s/%s.backtransform.gz" % (
						resultdir, stage.name), "w") as out:
					out.writelines("%s\n" % a for a in backtransform.values())
				if n and stage.prune:
					msg = grammar.getmapping(stages[n - 1].grammar,
						striplabelre=re.compile(b'@.+$'),
						neverblockre=re.compile(b'.+}<'),
						# + stage.neverblockre?
						splitprune=stage.splitprune and stages[n - 1].split,
						markorigin=stages[n - 1].markorigin)
				else:
					# recoverfragments() relies on this mapping to identify
					# binarization nodes
					msg = grammar.getmapping(None,
						striplabelre=None,
						neverblockre=re.compile(b'.+}<'),
						# + stage.neverblockre?
						splitprune=False, markorigin=False)
				logging.info(msg)
			elif n and stage.prune:  # dop reduction
				msg = grammar.getmapping(stages[n - 1].grammar,
					striplabelre=re.compile(b'@[-0-9]+$'),
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[n - 1].split,
					markorigin=stages[n - 1].markorigin)
				logging.info(msg)
		else:  # not stage.dop
			xgrammar = induce_plcfrs(traintrees, sents)
			logging.info("induced %s based on %d sentences",
				("PCFG" if fanout == 1 or stage.split else "PLCFRS"),
				len(traintrees))
			if stage.split or os.path.exists("%s/pcdist.txt" % resultdir):
				logging.info(grammarinfo(xgrammar))
			else:
				logging.info(grammarinfo(xgrammar,
						dump="%s/pcdist.txt" % resultdir))
			if lexmodel and simplelexsmooth:
				xgrammar = simplesmoothlexicon(xgrammar, lexmodel)
			elif lexmodel:
				xgrammar = smoothlexicon(xgrammar, lexmodel)
			grammar = Grammar(xgrammar, start=top,
					logprob=stage.mode != "pcfg-posterior")
			sumsto1 = grammar.testgrammar()
			if n and stage.prune:
				msg = grammar.getmapping(stages[n - 1].grammar,
					striplabelre=None,
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[n - 1].split,
					markorigin=stages[n - 1].markorigin)
				logging.info(msg)

		rules = gzip.open("%s/%s.rules.gz" % (resultdir, stage.name), "w")
		lexicon = codecs.getwriter('utf-8')(gzip.open("%s/%s.lex.gz" % (
				resultdir, stage.name), "w"))
		bitpar = fanout == 1 or stage.split
		# when grammar is LCFRS, write rational fractions.
		# when grammar is PCFG, write frequencies if probabilities sum to 1,
		# i.e., in that case probalities can be re-computed as relative
		# frequencies. otherwise, resort to decimal fractions (imprecise).
		write_lcfrs_grammar(xgrammar, rules, lexicon,
				bitpar=bitpar, freqs=bitpar and sumsto1)
		logging.info("wrote grammar to %s/%s.{rules,lex%s}.gz", resultdir,
				stage.name, ",backtransform" if stage.usedoubledop else '')

		outside = None
		if stage.getestimates == 'SX':
			assert fanout == 1 or stage.split, "SX estimate requires PCFG."
			logging.info("computing PCFG estimates")
			begin = time.clock()
			outside = getpcfgestimates(grammar, testmaxwords,
					grammar.toid[trees[0].label])
			logging.info("estimates done. cpu time elapsed: %gs",
					time.clock() - begin)
			np.savez("pcfgoutside.npz", outside=outside)
			logging.info("saved PCFG estimates")
		elif stage.useestimates == 'SX':
			assert fanout == 1 or stage.split, "SX estimate requires PCFG."
			assert stage.mode != 'pcfg', (
				"estimates require agenda-based parser.")
			outside = np.load("pcfgoutside.npz")['outside']
			logging.info("loaded PCFG estimates")
		if stage.getestimates == 'SXlrgaps':
			logging.info("computing PLCFRS estimates")
			begin = time.clock()
			outside = getestimates(grammar, testmaxwords,
					grammar.toid[trees[0].label])
			logging.info("estimates done. cpu time elapsed: %gs",
						time.clock() - begin)
			np.savez("outside.npz", outside=outside)
			logging.info("saved estimates")
		elif stage.useestimates == 'SXlrgaps':
			outside = np.load("outside.npz")['outside']
			logging.info("loaded PLCFRS estimates")
		stage.update(grammar=grammar, backtransform=backtransform,
				secondarymodel=secondarymodel, outside=outside)


def doparse(**kwds):
	""" Parse a set of sentences using worker processes. """
	params = DictObj(tags=True, numproc=None, tailmarker='',
		category=None, deletelabel=(), deleteword=(), corpusfmt="export")
	params.update(kwds)
	goldbrackets = multiset()
	totaltokens = 0
	results = [DictObj(name=stage.name) for stage in params.stages]
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
	logging.info("going to parse %d sentences.", len(params.testset))
	# main parse loop over each sentence in test corpus
	for nsent, data in enumerate(dowork, 1):
		sentid, msg, sentresults = data
		sent, goldtree, goldsent, _ = params.testset[sentid]
		logging.debug("%d/%d (%s). [len=%d] %s\n%s", nsent, len(params.testset),
					sentid, len(sent),
					" ".join(a[0] for a in goldsent), msg)
		evaltree = goldtree.copy(True)
		transform(evaltree, [w for w, _ in sent], evaltree.pos(),
				dict(evaltree.pos()), params.deletelabel, params.deleteword,
				{}, {}, False)
		goldb = bracketings(evaltree, dellabel=params.deletelabel)
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
				"%s cov %5.2f tag %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f%s",
					result.name.ljust(7),
					100 * (1 - results[n].noparse / nsent),
					100 * (results[n].tagscorrect / totaltokens),
					100 * (results[n].exact / nsent),
					100 * precision(goldbrackets, results[n].brackets),
					100 * recall(goldbrackets, results[n].brackets),
					100 * f_measure(goldbrackets, results[n].brackets),
					('' if n + 1 < len(sentresults) else '\n'))
	if params.numproc != 1:
		pool.terminate()
		pool.join()
		del dowork, pool

	writeresults(results, params)
	return results, goldbrackets


def worker(args):
	""" parse a sentence using specified stages (pcfg, plcfrs, dop, ...) """
	nsent, (sent, goldtree, _, _) = args
	prm = INTERNALPARAMS
	evaltree = goldtree.copy(True)
	transform(evaltree, [w for w, _ in sent], evaltree.pos(),
		dict(evaltree.pos()), prm.deletelabel, prm.deleteword, {}, {}, False)
	goldb = bracketings(evaltree, dellabel=prm.deletelabel)
	results = []
	msg = ''
	chart = {}
	start = inside = outside = None
	for n, stage in enumerate(prm.stages):
		begin = time.clock()
		exact = noparse = False
		msg += "%s:\t" % stage.name.upper()
		if not stage.prune or start:
			if n != 0 and stage.prune:
				if prm.stages[n - 1].mode == 'pcfg-posterior':
					(whitelist, sentprob, unfiltered, numitems, numremain
						) = whitelistfromposteriors(inside, outside, start,
							prm.stages[n - 1].grammar, stage.grammar, stage.k,
							stage.splitprune, prm.stages[n - 1].markorigin)
					msg += ("coarse items before pruning=%d; filtered: %d; "
							"pruned: %d; sentprob=%g\n\t" % (
							unfiltered, numitems, numremain, sentprob))
				else:
					whitelist, items = prunechart(chart, start,
						prm.stages[n - 1].grammar, stage.grammar, stage.k,
						stage.splitprune, prm.stages[n - 1].markorigin,
						stage.mode == "pcfg")
					msg += "coarse items before pruning: %d; after: %d\n\t" % (
						(sum(len(a) for x in chart for a in x if a)
						if prm.stages[n - 1].mode == 'pcfg'
						else len(chart)), items)
			else:
				whitelist = None
			if stage.mode == 'pcfg':
				chart, start, msg1 = pcfg.parse([w for w, _ in sent],
						stage.grammar,
						tags=[t for _, t in sent] if prm.tags else None,
						chart=whitelist if stage.prune else None)
			elif stage.mode == 'pcfg-posterior':
				inside, outside, start, msg1 = pcfg.doinsideoutside(
						[w for w, _ in sent],
						stage.grammar,
						tags=[t for _, t in sent] if prm.tags else None)
			elif stage.mode == 'plcfrs':
				chart, start, msg1 = plcfrs.parse([w for w, _ in sent],
						stage.grammar,
						tags=[t for _, t in sent] if prm.tags else None,
						exhaustive=stage.dop or (n + 1 != len(prm.stages)
								and prm.stages[n + 1].prune),
						whitelist=whitelist,
						splitprune=stage.splitprune and prm.stages[n - 1].split,
						markorigin=prm.stages[n - 1].markorigin,
						estimates=(stage.useestimates, stage.outside)
							if stage.useestimates in ('SX', 'SXlrgaps')
							else None)
			else:
				raise ValueError
			msg += "%s\n\t" % msg1
			if (n != 0 and not start and not results[-1].noparse
					and stage.split == prm.stages[n - 1].split):
				logging.error("ERROR: expected successful parse. "
						"sent %s, %s.", nsent, stage.name)
				#raise ValueError("ERROR: expected successful parse. "
				#		"sent %s, %s." % (nsent, stage.name))
		# store & report result
		if start and stage.mode != 'pcfg-posterior':
			if stage.dop:
				begindisamb = time.clock()
				parsetrees, msg1 = marginalize(stage.objective, chart, start,
						stage.grammar, stage.m, sample=stage.sample,
						kbest=stage.kbest, sent=[w for w, _ in sent],
						tags=[t for _, t in sent] if prm.tags else None,
						secondarymodel=stage.secondarymodel,
						sldop_n=stage.sldop_n,
						backtransform=stage.backtransform)
				resultstr, prob = max(parsetrees.items(), key=itemgetter(1))
				msg += "disambiguation: %s, %gs\n\t" % (
						msg1, time.clock() - begindisamb)
				if isinstance(prob, tuple):
					msg += "subtrees = %d, p=%.4e " % (abs(prob[0]), prob[1])
				else:
					msg += "p=%.4e " % prob
			elif not stage.dop:
				resultstr, prob = viterbiderivation(chart, start,
						stage.grammar.tolabel)
				msg += "p=%.4e " % exp(-prob)
			parsetree = Tree.parse(resultstr, parse_leaf=int)
			assert set(parsetree.leaves()) == set(goldtree.leaves())
			if stage.split:
				mergediscnodes(unbinarize(parsetree, childchar=":"))
			saveheads(parsetree, prm.tailmarker)
			unbinarize(parsetree)
			removefanoutmarkers(parsetree)
			if prm.unfolded:
				fold(parsetree)
			evaltree = parsetree.copy(True)
			transform(evaltree, [w for w, _ in sent], evaltree.pos(),
				dict(evaltree.pos()), prm.deletelabel, prm.deleteword,
				{}, {}, False)
			candb = bracketings(evaltree, dellabel=prm.deletelabel)
			if goldb and candb:
				prec = precision(goldb, candb)
				rec = recall(goldb, candb)
				f1score = f_measure(goldb, candb)
			else:
				prec = rec = f1score = 0
			if f1score == 1.0:
				msg += "exact match "
				exact = True
			else:
				msg += "LP %5.2f LR %5.2f LF %5.2f\n" % (
								100 * prec, 100 * rec, 100 * f1score)
				if (candb - goldb) or (goldb - candb):
					msg += '\t'
				if candb - goldb:
					msg += "cand-gold=%s " % strbracketings(candb - goldb)
				if goldb - candb:
					msg += "gold-cand=%s" % strbracketings(goldb - candb)
				msg += '\n\t'  # "%s\n\t" % parsetree
		if not start or stage.mode == 'pcfg-posterior':
			parsetree = defaultparse([(n, t) for n, (w, t) in enumerate(sent)])
			parsetree = Tree.parse("(%s %s)" % (stage.grammar.tolabel[1],
					parsetree), parse_leaf=int)
			evaltree = parsetree.copy(True)
			transform(evaltree, [w for w, _ in sent], evaltree.pos(),
					dict(evaltree.pos()), prm.deletelabel, prm.deleteword,
					{}, {}, False)
			candb = bracketings(evaltree, dellabel=prm.deletelabel)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1score = f_measure(goldb, candb)
			noparse = True
		elapsedtime = time.clock() - begin
		msg += "%.2fs cpu time elapsed\n" % (elapsedtime)
		results.append(DictObj(name=stage.name, candb=candb,
				parsetree=parsetree, noparse=noparse, exact=exact,
				elapsedtime=elapsedtime))
	#msg += "GOLD:   %s" % goldtree.pprint(margin=1000)
	return (nsent, msg, results)


def writeresults(results, params):
	""" Write parsing results to files in same format as the original corpus.
	(or export if writer not implemented) """
	ext = {"export": "export",
			"bracket": "mrg",
			"discbracket": "dbr",
			"alpino": "xml"}
	category = (params.category + ".") if params.category else ""
	if params.corpusfmt == 'alpino':
		corpusfmt = 'export'
		io.open("%s/%sgold.%s" % (params.resultdir, category, ext[corpusfmt]),
				"w", encoding='utf-8').writelines(
				writetree(goldtree, [w for w, _ in goldsent], n, corpusfmt)
			for n, (_, goldtree, goldsent, _) in params.testset.items())
	else:
		corpusfmt = params.corpusfmt
		io.open("%s/%sgold.%s" % (params.resultdir, category, ext[corpusfmt]),
				"w", encoding='utf-8').writelines(
				a for _, _, _, a in params.testset.values())
	for res in results:
		io.open("%s/%s%s.%s" % (params.resultdir, category, res.name,
				ext[corpusfmt]), "w", encoding='utf-8').writelines(
				writetree(res.parsetrees[n], [w for w, _ in goldsent], n,
				corpusfmt) for n, (_, _, goldsent, _) in
				params.testset.items())
	with open("%s/parsetimes.txt" % params.resultdir, "w") as out:
		out.write("#id\tlen\t%s\n" % "\t".join(res.name for res in results))
		out.writelines("%s\t%d\t%s\n" % (n, len(params.testset[n][2]),
				"\t".join(str(res.elapsedtime[n]) for res in results))
				for n in params.testset)
	logging.info("wrote results to %s/%s{%s}.%s", params.resultdir, category,
			",".join(res.name for res in results), ext[corpusfmt])


def oldeval(results, goldbrackets):
	""" Simple evaluation. """
	nsent = len(results[0].parsetrees)
	if nsent == 0:
		return
	for n, result in enumerate(results):
		logging.info("%s lp %5.2f lr %5.2f lf %5.2f\n"
			"coverage %d / %d = %5.2f %%  exact match %d / %d = %5.2f %%\n",
				result.name,
				100 * precision(goldbrackets, result.brackets),
				100 * recall(goldbrackets, result.brackets),
				100 * f_measure(goldbrackets, result.brackets),
				nsent - result.noparse, nsent,
				100 * (nsent - result.noparse) / nsent,
				result.exact, nsent, 100 * result.exact / nsent)


def saveheads(tree, tailmarker):
	""" When a head-outward binarization is used, this function ensures the
	head is known when the tree is converted to export format. """
	for node in tree.subtrees(lambda n: tailmarker in n.label):
		node.source = ['--'] * 6
		node.source[FUNC] = 'HD'


def readtepacoc():
	""" Read the tepacoc test set. """
	tepacocids = set()
	tepacocsents = defaultdict(list)
	cat = "undefined"
	tepacoc = io.open("../tepacoc.txt", encoding="utf8")
	for line in tepacoc.read().splitlines():
		fields = line.split("\t")  # = [id, '', sent]
		if line.strip() and len(fields) == 3:
			if fields[0].strip():
				# subtract one because our ids are zero-based, tepacoc 1-based
				sentid = int(fields[0]) - 1
				tepacocids.add(sentid)
				tepacocsents[cat].append((sentid, fields[2].split()))
			else:  # new category
				cat = fields[2]
				if cat.startswith("CUC"):
					cat = "CUC"
		elif fields[0] == "TuBa":
			break
	return tepacocids, tepacocsents


def parsetepacoc(
		stages=(
				dict(mode='pcfg', split=True, markorigin=True),
				dict(mode='plcfrs', prune=True, k=10000, splitprune=True),
				dict(mode='plcfrs', prune=True, k=5000, dop=True,
					usedoubledop=True, estimator="dop1", objective="mpp",
					sample=False, kbest=True)
		),
		trainmaxwords=999, trainnumsents=25005,
		testmaxwords=999, testnumsents=2000,
		bintype="binarize", h=1, v=1, factor="right", tailmarker='',
		revmarkov=False, leftmostunary=True, rightmostunary=True, pospa=False,
		fanout_marks_before_bin=False, unfolded=False,
		usetagger='stanford', resultdir="tepacoc", numproc=1):
	""" Parse the tepacoc test set. """
	for stage in stages:
		for key in stage:
			assert key in DEFAULTSTAGE, "unrecognized option: %r" % key
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
					gzip.open("tiger.pickle.gz", "rb"))
	except IOError:  # file not found
		corpus = getreader("export")("../tiger/corpus",
				"tiger_release_aug07.export",
				headrules="negra.headrules" if bintype == "binarize" else None,
				headfinal=True, headreverse=False, dounfold=unfolded,
				punct="move", encoding='iso-8859-1')
		corpus_sents = list(corpus.sents().values())
		corpus_taggedsents = list(corpus.tagged_sents().values())
		corpus_trees = list(corpus.parsed_sents().values())
		corpus_blocks = list(corpus.blocks().values())
		pickle.dump((corpus_sents, corpus_taggedsents, corpus_trees,
			corpus_blocks), gzip.open("tiger.pickle.gz", "wb"), protocol=-1)

	# test sets (one for each category)
	testsets = {}
	allsents = []
	for cat, catsents in tepacocsents.items():
		testset = sents, trees, goldsents, blocks = [], [], [], []
		for n, sent in catsents:
			if sent != corpus_sents[n]:
				logging.error(
						"mismatch. sent %d:\n%r\n%r\n"
						"not in corpus %r\nnot in tepacoc %r",
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
		logging.info("category: %s, %d of %d sentences",
				cat, len(testset[0]), len(catsents))
		testsets[cat] = testset
	testsets['baseline'] = zip(*[sent for n, sent in
				enumerate(zip(corpus_taggedsents, corpus_trees,
						corpus_taggedsents, corpus_blocks))
				if len(sent[1]) <= trainmaxwords
				and n not in tepacocids][trainnumsents:trainnumsents + 2000])
	allsents.extend(testsets['baseline'][0])

	if usetagger:
		overridetags = ("PTKANT", "VAIMP")
		taglex = defaultdict(set)
		for sent in corpus_taggedsents[:trainnumsents]:
			for word, tag in sent:
				taglex[word].add(tag)
		overridetagdict = {tag:
			{word for word, tags in taglex.items()
			if tags == {tag}} for tag in overridetags}
		tagmap = {"$(": "$[", "PAV": "PROAV", "PIDAT": "PIAT"}
		# the sentences in the list allsents are modified in-place so that
		# the relevant copy in testsets[cat][0] is updated as well.
		dotagging(usetagger, "", allsents, overridetagdict, tagmap)

	# training set
	trees, sents, blocks = zip(*[sent for n, sent in
				enumerate(zip(corpus_trees, corpus_sents,
							corpus_blocks)) if len(sent[1]) <= trainmaxwords
							and n not in tepacocids][:trainnumsents])
	getgrammars(trees, sents, stages, bintype, h, v, factor,
			tailmarker, revmarkov, leftmostunary, rightmostunary, pospa,
			fanout_marks_before_bin, testmaxwords, resultdir,
			numproc, None, False, trees[0].label)

	del corpus_sents, corpus_taggedsents, corpus_trees, corpus_blocks
	results = {}
	cnt = 0
	for cat, testset in sorted(testsets.items()):
		if cat == 'baseline':
			continue
		logging.info("category: %s", cat)
		begin = time.clock()
		results[cat] = doparse(stages=stages, testset=testset,
				testmaxwords=testmaxwords, testnumsents=testnumsents,
				top=trees[0].label, tags=True, bintype=bintype,
				tailmarker=tailmarker, unfolded=unfolded,
				fanout_marks_before_bin=fanout_marks_before_bin,
				resultdir=resultdir, numproc=numproc, category=cat)
		cnt += len(testset[0])
		if numproc == 1:
			logging.info("time elapsed during parsing: %g",
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
		logging.info("category: %s", cat)
		goldbrackets |= res[2]
		goldblocks.extend(res[3])
		goldsents.extend(res[4])
		for result, totresult in zip(res[0], totresults):
			totresult.exact += result.exact
			totresult.noparse += result.noparse
			totresult.brackets |= result.brackets
			totresult.elapsedtime.extend(result.elapsedtime)
		oldeval(*res)
	logging.info("TOTAL")
	oldeval(totresults, goldbrackets)
	# write TOTAL results file with all tepacoc sentences (not the baseline)
	for stage in stages:
		open("TOTAL.%s.export" % stage.name, "w").writelines(
				open("%s.%s.export" % (cat, stage.name)).read()
				for cat in list(results) + ['gold'])
	# do baseline separately because it shouldn't count towards the total score
	cat = 'baseline'
	logging.info("category: %s", cat)
	oldeval(*doparse(stages=stages, unfolded=unfolded, bintype=bintype,
			fanout_marks_before_bin=fanout_marks_before_bin,
			testset=testsets[cat], testmaxwords=testmaxwords,
			testnumsents=testnumsents, top=trees[0].label, tags=True,
			resultdir=resultdir, numproc=numproc, tailmarker=tailmarker,
			category=cat))


def dotagging(usetagger, model, sents, overridetag, tagmap):
	""" Use an external tool to tag a list of tagged sentences. """
	logging.info("Start tagging.")
	goldtags = [t for sent in sents.values() for _, t in sent]
	if usetagger == "treetagger":  # Tree-tagger
		installation = """tree tagger not found. commands to install:
mkdir tree-tagger && cd tree-tagger
wget ftp://ftp.ims.uni-stuttgart.de/pub/corpora/tree-tagger-linux-3.2.tar.gz
tar -xzf tree-tagger-linux-3.2.tar.gz
wget ftp://ftp.ims.uni-stuttgart.de/pub/corpora/tagger-scripts.tar.gz
tar -xzf ftp://ftp.ims.uni-stuttgart.de/pub/corpora/tagger-scripts.tar.gz
mkdir lib && cd lib && wget \
ftp://ftp.ims.uni-stuttgart.de/pub/corpora/german-par-linux-3.2-utf8.bin.gz
gunzip german-par-linux-3.2-utf8.bin.gz"""
		assert os.path.exists("tree-tagger/bin/tree-tagger"), installation
		infile, inname = tempfile.mkstemp(text=True)
		with os.fdopen(infile, 'w') as infile:
			for tagsent in sents.values():
				sent = map(itemgetter(0), tagsent)
				infile.write("\n".join(wordmangle(w, n, sent)
					for n, w in enumerate(sent)) + "\n<S>\n")
		if not model:
			model = "tree-tagger/lib/german-par-linux-3.2-utf8.bin"
			filtertags = "| tree-tagger/cmd/filter-german-tags"
		else:
			filtertags = ""
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
		install = """Stanford tagger not found. Commands to install:
wget http://nlp.stanford.edu/software/stanford-postagger-full-2012-07-09.tgz
tar -xzf stanford-postagger-full-2012-07-09.tgz"""
		assert os.path.exists("stanford-postagger-full-2012-07-09"), install
		infile, inname = tempfile.mkstemp(text=True)
		with os.fdopen(infile, 'w') as infile:
			for tagsent in sents.values():
				sent = map(itemgetter(0), tagsent)
				infile.write(" ".join(wordmangle(w, n, sent)
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
		(100 * accuracy(goldtags, newtags)),
		set(goldtags) - set(newtags), set(newtags) - set(goldtags))
	return taggedsents

SENTEND = "(\"'!?..."  # ";/-"


def wordmangle(w, n, sent):
	""" Function to filter words before they are sent to the tagger. """
	#if n > 0 and w[0] in string.uppercase and not sent[n - 1] in SENTEND:
	#	return ("%s\tNE\tNN\tFM" % w).encode('utf-8')
	return w.encode('utf-8')


def tagmangle(a, splitchar, overridetag, tagmap):
	""" Function to filter tags after they are produced by the tagger. """
	word, tag = a.rsplit(splitchar, 1)
	for newtag in overridetag:
		if word in overridetag[newtag]:
			tag = newtag
	return word, tagmap.get(tag, tag)


def treebankfanout(trees):
	""" Get maximal fan-out of a list of trees. """
	return max((fastfanout(addbitsets(a)), n) for n, tree in enumerate(trees)
		for a in tree.subtrees(lambda x: len(x) > 1))


class DictObj(object):
	""" A trivial class to wrap a dictionary for reasons of syntactic sugar. """

	def __init__(self, *a, **kw):
		self.__dict__.update(*a, **kw)

	def update(self, *a, **kw):
		""" Update/add more attributes. """
		self.__dict__.update(*a, **kw)

	def __getattr__(self, name):
		""" This is only called when the normal mechanism fails, so in practice
		should never be called. It is only provided to satisfy pylint that it
		is okay not to raise E1101 errors in the client code. """
		raise AttributeError("%r instance has no attribute %r" % (
				self.__class__.__name__, name))

	def __repr__(self):
		return "%s(%s)" % (self.__class__.__name__,
			",\n".join("%s=%r" % a for a in self.__dict__.items()))


def main(argv=None):
	""" Parse command line arguments. """
	try:
		# report backtrace on segfaults &c. pip install faulthandler
		import faulthandler
		faulthandler.enable()
	except ImportError:
		pass
	if argv is None:
		argv = sys.argv
	if len(argv) == 1:
		print(USAGE)
	elif '--tepacoc' in argv:
		parsetepacoc()
	else:
		paramstr = open(argv[1]).read()
		params = eval("dict(%s)" % paramstr)
		params['resultdir'] = argv[1].rsplit(".", 1)[0]
		params['rerun'] = '--rerun' in argv
		startexp(**params)
		if not params['rerun']:  # copy parameter file to result dir
			open("%s/params.prm" % params['resultdir'], "w").write(paramstr)
