# -*- coding: UTF-8 -*-
""" Run an experiment given a parameter file. Reads of grammars, does parsing
and evaluation. """
from __future__ import division, print_function
import os, re, sys, time, gzip, io, codecs, logging, tempfile
if sys.version < '3':
	import cPickle as pickle
	from itertools import islice, izip_longest as zip_longest
else:
	import pickle
	from itertools import islice, zip_longest
import multiprocessing
from collections import defaultdict, OrderedDict, Counter as multiset
from operator import itemgetter
from subprocess import Popen, PIPE
from fractions import Fraction
from math import exp
from tree import Tree
import numpy as np
from treebank import getreader, fold, writetree, FUNC, \
		unknownword4, unknownword6, unknownwordbase, replacerarewords
from treetransforms import binarize, unbinarize, optimalbinarize, \
		splitdiscnodes, mergediscnodes, canonicalize, \
		addfanoutmarkers, removefanoutmarkers, addbitsets, fastfanout
from fragments import getfragments
from grammar import induce_plcfrs, dopreduction, doubledop, \
		grammarinfo, write_lcfrs_grammar, getunknownwordmodel, \
		getlexmodel, smoothlexicon, simplesmoothlexicon
from containers import Grammar
from _parser import parse, cfgparse
from coarsetofine import prunechart
from disambiguation import marginalize, viterbiderivation
from eval import doeval, readparam, strbracketings, transform, \
		bracketings, precision, recall, f_measure, accuracy

USAGE = """Usage: %s --test | [--rerun] parameter file
--test	run tests on all modules
If a parameter file is given, an experiment is run. See the file sample.prm
for an example parameter file. To repeat an experiment with an existing grammar,
pass the option --rerun.""" % sys.argv[0]

class DictObj(object):
	""" A class to wrap a dictionary. """
	def __init__(self, **kw):
		self.__dict__.update(kw)
	def __repr__(self):
		return "%s(%s)" % (self.__class__.__name__,
			",\n".join("%s=%r" % a for a in self.__dict__.items()))


internalparams = None
def initworker(params):
	""" Set global parameter object """
	global internalparams
	internalparams = params

defaultstage = dict(
		name='stage1', # identifier, used for filenames
		mode='plcfrs', # use the agenda-based PLCFRS parser
		prune=False, #whether to use previous chart to prune this stage
		split=False, #split disc. nodes VP_2[101] as { VP*[100], VP*[001] }
		splitprune=False, #VP_2[101] is treated as {VP*[100], VP*[001]} for pruning
		markorigin=False, #mark origin of split nodes: VP_2 => {VP*1, VP*2}
		k=50, #no. of coarse pcfg derivations to prune with; k=0 => filter only
		neverblockre=None, #do not prune nodes with label that match regex
		getestimates=None, #compute & store estimates
		useestimates=None, #load & use estimates
		dop=False, # enable DOP mode (DOP reduction / double DOP)
		packedgraph=False, # use packed graph encoding for DOP reduction
		usedoubledop=False, # when False, use DOP reduction instead
		iterate=False, #for double dop, whether to include fragments of fragments
		complement=False, #for double dop, whether to include fragments which
				#form the complement of the maximal recurring fragments extracted
		sample=False, kbest=True,
		m=10000, #number of derivations to sample/enumerate
		estimator="ewe", # choices: dop1, ewe
		objective="mpp", # choices: mpp, mpd, shortest, sl-dop[-simple]
			# NB: w/shortest derivation, estimator only affects tie breaking.
		sldop_n=7)

def main(
		stages=(), # see variable 'defaults' above
		corpusfmt="export", # choices: export, discbracket, bracket
		corpusdir=".",
		traincorpus="sample2.export", trainencoding="iso-8859-1",
		testcorpus="sample2.export", testencoding="iso-8859-1",
		punct=None, # options: move, remove, root
		functiontags=False, # whether to add/strip function tags from node labels
		unfolded=False,
		testmaxwords=40,
		trainmaxwords=40,
		trainnumsents=2,
		testnumsents=1, # number of sentences to parse
		skiptrain=True, # test set starts after training set
		# (useful when they are in the same file)
		skip=0, # number of sentences to skip from test corpus
		# postagging: pass None to use tags from treebank.
		postagging=dict(
			method="unknownword",
			# choices: unknownword (assign during parsing),
			#		treetagger, stanford (external taggers)
			# choices unknownword: 4, 6, base,
			# for treetagger / stanford: [filename of external tagger model]
			model="4",
			# options for unknown word models:
			unknownthreshold=1, # use probs of rare words for unknown words
			openclassthreshold=50, # add unseen tags for known words. 0 to disable.
			simplelexsmooth=True, # disable sophisticated smoothing
		),
		bintype="binarize", # choices: binarize, optimal, optimalhead
		factor="right",
		revmarkov=True,
		v=1,
		h=2,
		leftmostunary=True, #start binarization with unary node
		rightmostunary=True, #end binarization with unary node
		headrules=None, # rules for finding heads of constituents
		fanout_marks_before_bin=False,
		tailmarker="",
		evalparam="proper.prm", # EVALB-style parameter file
		quiet=False, reallyquiet=False, #quiet=no per sentence results
		numproc=1,	#increase to use multiple CPUs. Set to None to use all CPUs.
		resultdir='results',
		rerun=False):
	""" Main entry point. """
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
			assert key in defaultstage, "unrecognized option: %r" % key
	stages = [DictObj(**{k: stage.get(k, v)
			for k, v in defaultstage.items()}) for stage in stages]

	if rerun:
		assert os.path.exists(resultdir), (
				"Directory %r does not exist."
				"--rerun requires a directory "
				"with the grammar of a previous experiment."
				% resultdir)
	else:
		assert not os.path.exists(resultdir), (
			"Directory %r exists.\n"
			"Use --rerun to parse with existing grammar"
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

	CorpusReader = getreader(corpusfmt)
	if not rerun:
		corpus = CorpusReader(corpusdir, traincorpus, encoding=trainencoding,
			headrules=headrules, headfinal=True, headreverse=False,
			punct=punct, functiontags=functiontags, dounfold=unfolded)
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

	testset = CorpusReader(corpusdir, testcorpus, encoding=testencoding,
			punct=punct)
	gold_sents = testset.tagged_sents()
	test_parsed_sents = testset.parsed_sents()
	if skiptrain:
		skip += trainnumsents
	logging.info("%d sentences in test corpus %s/%s",
			len(testset.parsed_sents()), corpusdir, testcorpus)
	logging.info("%d test sentences before length restriction",
			len(list(gold_sents.keys())[skip:skip+testnumsents]))
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
				OrderedDict((a, b)
				for a, b in islice(gold_sents.items(), skip, skip+testnumsents)
				if len(b) <= testmaxwords),
				overridetagdict, tagmap)
		# give these tags to parser
		tags = True
	elif postagging and postagging['method'] == "unknownword":
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
		replacerarewords(sents, unknownword, postagging['unknownthreshold'])
		# replace unknown test words with features
		test_tagged_sents = OrderedDict()
		for n, sent in gold_sents.items():
			newsent = []
			for m, (word, _) in enumerate(sent):
				if word in knownwords:
					sig = word
					#logging.debug("known word: %s freq=%d", word, knownwords[word])
				else:
					sig = unknownword(word, m, knownwords)
					if sig not in wordclass:
						#logging.debug("UNKNOWN CLASS: %s => %s => UNK", word, sig)
						sig = "UNK"
					else:
						pass
						#logging.debug("unknown word: %s => %s", word, sig)
				newsent.append((sig, None))
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
		readgrammars(resultdir, stages)
		trees = []
		sents = []
	else:
		logging.info("read training & test corpus")

		getgrammars(trees, sents, stages, bintype, h, v, factor, tailmarker,
				revmarkov, leftmostunary, rightmostunary,
				fanout_marks_before_bin, testmaxwords, resultdir, numproc,
				lexmodel, simplelexsmooth)
	toplabels = {tree.label for tree in trees} | {
			test_parsed_sents[n].label for n in testset}
	assert len(toplabels) == 1, "expected unique TOP/ROOT label"
	top = toplabels.pop()
	evalparam = readparam(evalparam)
	evalparam["DEBUG"] = -1
	evalparam["CUTOFF_LEN"] = 40
	deletelabel = evalparam.get("DELETE_LABEL", ())
	deleteword = evalparam.get("DELETE_WORD", ())

	begin = time.clock()
	results = doparse(stages, unfolded, bintype,
			fanout_marks_before_bin, testset, testmaxwords,
			testnumsents, top, tags, resultdir, numproc, tailmarker,
			deletelabel=deletelabel, deleteword=deleteword,
			corpusfmt=corpusfmt)
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

def readgrammars(resultdir, stages):
	""" Read the grammars from a previous experiment. Must have same parameters.
	"""
	from grammar import read_bitpar_grammar, read_lcfrs_grammar
	for n, stage in enumerate(stages):
		logging.info("reading: %s", stage.name)
		rules = gzip.open("%s/%s.rules.gz" % (resultdir, stage.name))
		lexicon = codecs.getreader('utf-8')(gzip.open("%s/%s.lex.gz" % (
				resultdir, stage.name)))
		if stage.mode == 'pcfg':
			grammar = read_bitpar_grammar(rules, lexicon)
		elif stage.mode == 'plcfrs':
			grammar = read_lcfrs_grammar(rules, lexicon)
		else:
			raise ValueError
		logging.info(grammarinfo(grammar))
		grammar = Grammar(grammar)
		stage.backtransform = None
		if stage.dop:
			assert stage.objective not in (
					"shortest", "sl-dop", "sl-dop-simple"), (
					"Shortest derivation parsing not supported.")
			assert stage.useestimates is None, "not supported"
			if stage.usedoubledop:
				stage.backtransform = dict(enumerate(
						gzip.open("%s/%s.backtransform.gz" % (resultdir,
						stage.name)).read().splitlines()))
				if n and stage.prune:
					msg = grammar.getmapping(stages[n-1].grammar,
						striplabelre=re.compile("@.+$"),
						neverblockre=re.compile(r'^#[0-9]+|.+}<'),
						# + stage.neverblockre?
						splitprune=stage.splitprune and stages[n-1].split,
						markorigin=stages[n-1].markorigin)
				else:
					# recoverfragments() relies on this mapping to identify
					# binarization nodes
					msg = grammar.getmapping(None,
						striplabelre=None,
						neverblockre=re.compile(r'.+}<'),
						# + stage.neverblockre?
						splitprune=False, markorigin=False)
				logging.info(msg)
			elif n and stage.prune: # dop reduction
				msg = grammar.getmapping(stages[n-1].grammar,
					striplabelre=re.compile("@[-0-9]+$"),
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[n-1].split,
					markorigin=stages[n-1].markorigin)
				logging.info(msg)
		else: # not stage.dop
			if n and stage.prune:
				msg = grammar.getmapping(stages[n-1].grammar,
					striplabelre=None,
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[n-1].split,
					markorigin=stages[n-1].markorigin)
				logging.info(msg)
		stage.grammar = grammar
		stage.secondarymodel = None
		stage.outside = None
		stage.grammar.testgrammar()

def getgrammars(trees, sents, stages, bintype, h, v, factor,
		tailmarker, revmarkov, leftmostunary, rightmostunary,
		fanout_marks_before_bin, testmaxwords, resultdir, numproc,
		lexmodel, simplelexsmooth):
	""" Apply binarization and read off the requested grammars. """
	# fixme: this n should correspond to sentence id
	f, n = treebankfanout(trees)
	logging.info("treebank fan-out before binarization: %d #%d\n%s\n%s", f, n,
			trees[n], " ".join(sents[n]))
	# binarization
	begin = time.clock()
	if fanout_marks_before_bin:
		trees = [addfanoutmarkers(t) for t in trees]
	if bintype == "binarize":
		bintype += " %s h=%d v=%d %s" % (factor, h, v,
			"tailmarker" if tailmarker else '')
		for a in trees:
			binarize(a, factor=factor, vertmarkov=v, horzmarkov=h,
					tailmarker=tailmarker, leftmostunary=leftmostunary,
					rightmostunary=rightmostunary, reverse=revmarkov)
	elif bintype == "optimal":
		trees = [Tree.convert(optimalbinarize(tree))
						for n, tree in enumerate(trees)]
	elif bintype == "optimalhead":
		trees = [Tree.convert(
					optimalbinarize(tree, headdriven=True, h=h, v=v))
						for n, tree in enumerate(trees)]
	trees = [addfanoutmarkers(t) for t in trees]
	logging.info("binarized %s cpu time elapsed: %gs",
						bintype, time.clock() - begin)
	logging.info("binarized treebank fan-out: %d #%d", *treebankfanout(trees))
	trees = [canonicalize(a).freeze() for a in trees]

	#cycledetection()
	if any(stage.split for stage in stages):
		splittrees = [binarize(splitdiscnodes(Tree.convert(a),
				stages[0].markorigin), childchar=":").freeze()
				for a in trees]
		logging.info("splitted discontinuous nodes")
	for n, stage in enumerate(stages):
		assert stage.mode in ("plcfrs", "pcfg")
		if stage.split:
			traintrees = splittrees
		else:
			traintrees = trees
		assert n > 0 or not stage.prune, (
				"need previous stage to prune, but this stage is first.")
		if stage.dop:
			stages[n].backtransform = stages[n].secondarymodel = None
			assert stage.estimator in ("dop1", "ewe")
			assert stage.objective in ("mpp", "mpd", "shortest",
					"sl-dop", "sl-dop-simple")
			if stage.usedoubledop:
				# find recurring fragments in treebank,
				# as well as depth-1 'cover' fragments
				fragments = getfragments(traintrees, sents, numproc,
						iterate=stage.iterate, complement=stage.complement,
						indices=stage.estimator=="ewe")
				xgrammar, backtransform = doubledop(fragments,
						ewe=stage.estimator=="ewe")
				stages[n].backtransform = backtransform
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
						stages[n].secondarymodel = dict(xgrammar)
						xgrammar = shortest
					elif stage.objective.startswith("sl-dop"):
						stages[n].secondarymodel = dict(shortest)
			elif stage.objective == "shortest": # dopreduction from here on
				# the secondary model is used to resolve ties
				# for the shortest derivation
				# i.e., secondarymodel is probabilistic
				xgrammar, secondarymodel = dopreduction(traintrees, sents,
					ewe=stage.estimator=="ewe", shortestderiv=True)
				stages[n].secondarymodel = secondarymodel
			elif "sl-dop" in stage.objective:
				# here secondarymodel is non-probabilistic
				xgrammar = dopreduction(traintrees, sents,
						ewe=stage.estimator=="ewe", shortestderiv=False)
				secondarymodel, _ = dopreduction(traintrees, sents,
								ewe=False, shortestderiv=True)
				stages[n].secondarymodel = Grammar(secondarymodel)
			else: # mpp or mpd
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
			grammar = Grammar(xgrammar)
			logging.info("DOP model based on %d sentences, %d nodes, "
				"%d nonterminals",  len(traintrees), nodes, len(grammar.toid))
			logging.info(msg)
			sumsto1 = grammar.testgrammar()
			if stage.usedoubledop:
				# backtransform keys are line numbers to rules file;
				# to see them together do:
				# $ paste <(zcat dop.rules.gz) <(zcat dop.backtransform.gz)
				gzip.open(resultdir + "/dop.backtransform.gz", "w").writelines(
						"%s\n" % a for a in backtransform.values())
				if n and stage.prune:
					msg = grammar.getmapping(stages[n-1].grammar,
						striplabelre=re.compile("@.+$"),
						neverblockre=re.compile(r'.+}<'),
						# + stage.neverblockre?
						splitprune=stage.splitprune and stages[n-1].split,
						markorigin=stages[n-1].markorigin)
				else:
					# recoverfragments() relies on this mapping to identify
					# binarization nodes
					msg = grammar.getmapping(None,
						striplabelre=None,
						neverblockre=re.compile(r'.+}<'),
						# + stage.neverblockre?
						splitprune=False, markorigin=False)
				logging.info(msg)
			elif n and stage.prune: # dop reduction
				msg = grammar.getmapping(stages[n-1].grammar,
					striplabelre=re.compile("@[-0-9]+$"),
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[n-1].split,
					markorigin=stages[n-1].markorigin)
				logging.info(msg)
		else: # not stage.dop
			xgrammar = induce_plcfrs(traintrees, sents)
			logging.info("induced %s based on %d sentences",
				("PCFG" if f == 1 or stage.split else "PLCFRS"),
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
			grammar = Grammar(xgrammar)
			sumsto1 = grammar.testgrammar()
			if n and stage.prune:
				msg = grammar.getmapping(stages[n-1].grammar,
					striplabelre=None,
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[n-1].split,
					markorigin=stages[n-1].markorigin)
				logging.info(msg)

		stages[n].grammar = grammar
		rules = gzip.open("%s/%s.rules.gz" % (resultdir, stages[n].name), "w")
		lexicon = codecs.getwriter('utf-8')(gzip.open("%s/%s.lex.gz" % (
			resultdir, stages[n].name), "w"))
		bitpar = f == 1 or stage.split
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
			assert f == 1 or stage.split, "SX estimate requires PCFG."
			from estimates import getpcfgestimates
			logging.info("computing PCFG estimates")
			begin = time.clock()
			outside = getpcfgestimates(grammar, testmaxwords,
					grammar.toid[trees[0].label])
			logging.info("estimates done. cpu time elapsed: %gs",
					time.clock() - begin)
			np.savez("pcfgoutside.npz", outside=outside)
			logging.info("saved PCFG estimates")
		elif stage.useestimates == 'SX':
			assert f == 1 or stage.split, "SX estimate requires PCFG."
			assert stage.mode != 'pcfg', (
				"estimates require agenda-based parser.")
			outside = np.load("pcfgoutside.npz")['outside']
			logging.info("loaded PCFG estimates")
		if stage.getestimates == 'SXlrgaps':
			from estimates import getestimates
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
		stages[n].outside = outside

#def doparse(**params):
#	params = DictObj(**params)
def doparse(stages, unfolded, bintype, fanout_marks_before_bin,
		testset, testmaxwords, testnumsents, top, tags=True,
		resultdir="results", numproc=None, tailmarker='', category=None,
		deletelabel=(), deleteword=(), corpusfmt="export"):
	""" Parse a set of sentences using worker processes. """
	params = DictObj(stages=stages, unfolded=unfolded, bintype=bintype,
			fanout_marks_before_bin=fanout_marks_before_bin,
			testset=testset, testmaxwords=testmaxwords,
			testnumsents=testnumsents, top=top, tags=tags,
			resultdir=resultdir, category=category,
			deletelabel=deletelabel, deleteword=deleteword,
			tailmarker=tailmarker)
	goldbrackets = multiset()
	goldblocks = OrderedDict.fromkeys(testset)
	goldsents = OrderedDict.fromkeys(testset)
	totaltokens = 0
	results = [DictObj(name=stage.name) for stage in stages]
	for result in results:
		result.elapsedtime = dict.fromkeys(testset)
		result.parsetrees = dict.fromkeys(testset)
		result.brackets = multiset()
		result.tagscorrect = result.exact = result.noparse = 0
	if numproc == 1:
		initworker(params)
		dowork = (worker(a) for a in testset.items())
	else:
		pool = multiprocessing.Pool(processes=numproc, initializer=initworker,
				initargs=(params,))
		dowork = pool.imap_unordered(worker, testset.items())
	logging.info("going to parse %d sentences.", len(testset))
	# main parse loop over each sentence in test corpus
	for nsent, data in enumerate(dowork, 1):
		sentid, msg, sentresults = data
		sent, goldtree, goldsent, block = testset[sentid]
		logging.debug("%d/%d (%s). [len=%d] %s\n%s", nsent, len(testset),
					sentid, len(sent),
					" ".join(a[0] + "/"+a[1] for a in sent)
					if tags else " ".join(a[0] for a in goldsent),
					msg)
		evaltree = goldtree.copy(True)
		transform(evaltree, [w for w, _ in sent], evaltree.pos(),
				dict(evaltree.pos()), deletelabel, deleteword, {}, {}, False)
		goldb = bracketings(evaltree, dellabel=deletelabel)
		assert goldblocks[sentid] == goldsents[sentid] == None
		goldblocks[sentid] = block
		goldsents[sentid] = goldsent
		goldbrackets.update((sentid, (label, span)) for label, span
				in goldb.elements())
		totaltokens += sum(1 for _, t in goldsent if t not in deletelabel)
		for n, r in enumerate(sentresults):
			results[n].brackets.update((sentid, (label, span)) for label, span
					in r.candb.elements())
			assert (results[n].parsetrees[sentid]
				== results[n].elapsedtime[sentid] == None)
			results[n].parsetrees[sentid] = r.parsetree
			results[n].elapsedtime[sentid] = r.elapsedtime
			if r.noparse:
				results[n].noparse += 1
			if r.exact:
				results[n].exact += 1
			results[n].tagscorrect += sum(1 for (_, a), (_, b)
					in zip(goldsent, sorted(r.parsetree.pos()))
					if b not in deletelabel and a == b)
			logging.debug(
				"%s cov %5.2f tag %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f%s",
					r.name.ljust(7),
					100 * (1 - results[n].noparse/ nsent),
					100 * (results[n].tagscorrect / totaltokens),
					100 * (results[n].exact / nsent),
					100 * precision(goldbrackets, results[n].brackets),
					100 * recall(goldbrackets, results[n].brackets),
					100 * f_measure(goldbrackets, results[n].brackets),
					('' if n + 1 < len(sentresults) else '\n'))
	if numproc != 1:
		pool.terminate()
		pool.join()
		del dowork, pool

	writeresults(results, goldblocks, goldsents, resultdir, category, corpusfmt)
	return results, goldbrackets

def worker(args):
	""" parse a sentence using specified stages (pcfg, plcfrs, dop, ...) """
	nsent, (sent, goldtree, _, _) = args
	d = internalparams
	evaltree = goldtree.copy(True)
	transform(evaltree, [w for w, _ in sent], evaltree.pos(),
			dict(evaltree.pos()), d.deletelabel, d.deleteword, {}, {}, False)
	goldb = bracketings(evaltree, dellabel=d.deletelabel)
	results = []
	msg = ''
	chart = {}
	start = None
	for n, stage in enumerate(d.stages):
		begin = time.clock()
		exact = noparse = False
		msg += "%s:\t" % stage.name.upper()
		if n == 0 or start:
			if n != 0 and stage.prune:
				whitelist, items = prunechart(chart, start,
					d.stages[n-1].grammar, stage.grammar, stage.k,
					stage.splitprune, d.stages[n-1].markorigin,
					stage.mode == "pcfg")
				msg += "coarse items before pruning: %d; after: %d\n\t" % (
					(sum(len(a) for x in chart for a in x if a)
					if d.stages[n-1].mode == 'pcfg' else len(chart)), items)
			else:
				whitelist = None
			if stage.mode == 'pcfg':
				chart, start, msg1 = cfgparse([w for w, _ in sent],
						stage.grammar,
						tags=[t for _, t in sent] if d.tags else None,
						start=stage.grammar.toid[d.top],
						chart=whitelist if stage.prune else None)
			else:
				chart, start, msg1 = parse([w for w, _ in sent],
						stage.grammar,
						tags=[t for _, t in sent] if d.tags else None,
						start=stage.grammar.toid[d.top],
						exhaustive=stage.dop or (
							n+1 != len(d.stages) and d.stages[n+1].prune),
						whitelist=whitelist,
						splitprune=stage.splitprune and d.stages[n-1].split,
						markorigin=d.stages[n-1].markorigin,
						estimates=(stage.useestimates, stage.outside)
							if stage.useestimates in ('SX', 'SXlrgaps')
							else None)
			msg += "%s\n\t" % msg1
			if (n != 0 and not start and not results[-1].noparse
					and stage.split == d.stages[n-1].split):
				#from _parser import pprint_chart
				#pprint_chart(chart,
				#		[w.encode('unicode-escape') for w, _ in sent],
				#		stage.grammar.tolabel)
				logging.error("ERROR: expected successful parse. "
						"sent %s, %s.", nsent, stage.name)
				#raise ValueError("ERROR: expected successful parse. "
				#		"sent %s, %s." % (nsent, stage.name))
		# store & report result
		if start:
			if stage.dop:
				begindisamb = time.clock()
				parsetrees, msg1 = marginalize(stage.objective, chart, start,
						stage.grammar, stage.m, sample=stage.sample,
						kbest=stage.kbest, sent=[w for w, _ in sent],
						tags=[t for _, t in sent] if d.tags else None,
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
			if stage.split:
				mergediscnodes(unbinarize(parsetree, childchar=":"))
			saveheads(parsetree, d.tailmarker)
			unbinarize(parsetree)
			removefanoutmarkers(parsetree)
			if d.unfolded:
				fold(parsetree)
			evaltree = parsetree.copy(True)
			transform(evaltree, [w for w, _ in sent], evaltree.pos(),
					dict(evaltree.pos()), d.deletelabel, d.deleteword,
					{}, {}, False)
			candb = bracketings(evaltree, dellabel=d.deletelabel)
			if goldb and candb:
				prec = precision(goldb, candb)
				rec = recall(goldb, candb)
				f1 = f_measure(goldb, candb)
			else:
				prec = rec = f1 = 0
			if f1 == 1.0:
				msg += "exact match "
				exact = True
			else:
				msg += "LP %5.2f LR %5.2f LF %5.2f\n" % (
								100 * prec, 100 * rec, 100 * f1)
				if candb - goldb:
					msg += "\tcand-gold=%s " % strbracketings(candb - goldb)
				if goldb - candb:
					msg += "gold-cand=%s" % strbracketings(goldb - candb)
				if (candb - goldb) or (goldb - candb):
					msg += '\n'
				msg += "\t%s\n\t" % parsetree.pprint(margin=1000)
		if not start:
			msg += "no parse. "
			parsetree = defaultparse([(n, t) for n, (w, t) in enumerate(sent)])
			parsetree = Tree.parse("(%s %s)" % (d.top, parsetree),
				parse_leaf=int)
			evaltree = parsetree.copy(True)
			transform(evaltree, [w for w, _ in sent], evaltree.pos(),
					dict(evaltree.pos()), d.deletelabel, d.deleteword,
					{}, {}, False)
			candb = bracketings(evaltree, dellabel=d.deletelabel)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			noparse = True
		elapsedtime = time.clock() - begin
		msg += "%.2fs cpu time elapsed\n" % (elapsedtime)
		results.append(DictObj(name=stage.name, candb=candb,
				parsetree=parsetree, noparse=noparse, exact=exact,
				elapsedtime=elapsedtime))
	msg += "GOLD:   %s" % goldtree.pprint(margin=1000)
	return (nsent, msg, results)

def writeresults(results, goldblocks, goldsents, resultdir, category,
		corpusfmt="export"):
	""" Write parsing results to files in same format as the original corpus.
	(or export if writer not implemented) """
	ext = {"export": "export",
			"bracket": "mrg",
			"discbracket": "dbr",
			"alpino": "xml"}
	if corpusfmt == 'alpino':
		golddir = "%s/%s" % (resultdir,
				(".".join(category, "gold") if category else "gold"))
		os.mkdir(golddir)
		preamble = u'<?xml version="1.0" encoding="UTF-8"?>\n'
		for n, block in goldblocks.items():
			thisdir, sentid = os.path.split(n)
			thispath = os.path.join(golddir, thisdir)
			if not os.path.exists(thispath):
				os.mkdir(thispath)
			io.open("%s/%s.%s" % (thispath, sentid, ext[corpusfmt]),
					"wt", encoding='utf-8').write(preamble + block)
		corpusfmt = 'export'
	else:
		io.open("%s/%s.%s" % (resultdir, (".".join(category, "gold")
				if category else "gold"), ext[corpusfmt]), "w", encoding='utf-8'
				).writelines(goldblocks.values())
	for result in results:
		io.open("%s/%s.%s" % (resultdir,
			".".join(category, result.name) if category else result.name,
			ext[corpusfmt]),
			"w", encoding='utf-8').writelines(writetree(
			result.parsetrees[n], [w for w, _ in goldsents[n]], n, corpusfmt)
			for n in goldsents)
	with open("%s/parsetimes.txt" % resultdir, "w") as f:
		f.write("#id\tlen\t%s\n" % "\t".join(result.name for result in results))
		f.writelines(
			"%s\t%d\t%s\n" % (n, len(goldsents[n]),
					"\t".join(str(result.elapsedtime[n]) for result in results))
				for n in goldblocks)
	logging.info("wrote results to %s/%s{%s}.%s",
		resultdir, (category + ".") if category else "",
		",".join(result.name for result in results), ext[corpusfmt])

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
	for node in tree.subtrees(lambda n: "tailmarker" in n.label):
		node.source = ['--'] * 6
		node.source[FUNC] = 'HD'

def defaultparse(wordstags):
	""" a right branching default parse

	>>> defaultparse([('like','X'), ('this','X'), ('example', 'NN'), \
			('here','X')])
	'(NP (X like) (NP (X this) (NP (NN example) (NP (X here) ))))' """
	if wordstags == []:
		return ''
	return "(%s (%s %s) %s)" % ("NP", wordstags[0][1],
			wordstags[0][0], defaultparse(wordstags[1:]))

def readtepacoc():
	""" Read the tepacoc test set. """
	tepacocids = set()
	tepacocsents = defaultdict(list)
	cat = "undefined"
	tepacoc = io.open("../tepacoc.txt", encoding="utf8")
	for line in tepacoc.read().splitlines():
		fields = line.split("\t") # = [id, '', sent]
		if line.strip() and len(fields) == 3:
			if fields[0].strip():
				# subtract one because our ids are zero-based, tepacoc 1-based
				sentid = int(fields[0]) - 1
				tepacocids.add(sentid)
				tepacocsents[cat].append((sentid, fields[2].split()))
			else: # new category
				cat = fields[2]
				if cat.startswith("CUC"):
					cat = "CUC"
		elif fields[0] == "TuBa":
			break
	return tepacocids, tepacocsents

def parsetepacoc(
		stages=(
		dict(mode='pcfg', # use the dedicated PCFG parser
			split=True,
			markorigin=True, #mark origin of split nodes: VP_2 => {VP*1, VP*2}
		),
		dict(mode='plcfrs', # the agenda-based PLCFRS parser
			prune=True,	#whether to use previous chart to prune this stage
			k = 10000, #number of coarse pcfg derivations to prune with;
					#k=0 => filter only.
			splitprune=True,
		),
		dict(mode='plcfrs', # the agenda-based PLCFRS parser
			prune=True,	#whether to use previous chart to prune this stage
			k = 5000, #number of coarse plcfrs derivations to prune with;
					# k=0 => filter only
			dop=True,
			usedoubledop=True,	# when False, use DOP reduction instead
			estimator = "dop1", # choices: dop1, ewe
			objective = "mpp", # choices: mpp, mpd, shortest, sl-dop[-simple]
			# NB: w/shortest derivation, estimator only affects tie breaking.
			sample=False, kbest=True,
			iterate=False, #for double dop, whether to include
					#fragments of fragments
			complement=False, #for double dop, whether to include fragments
					#which form complement of extracted fragments
		)),
		unfolded=False, bintype="binarize", h=1, v=1, factor="right",
		tailmarker='', revmarkov=False,
		leftmostunary=True, rightmostunary=True,
		fanout_marks_before_bin=False,
		trainmaxwords=999, testmaxwords=999, testnumsents=2000,
		usetagger='stanford', resultdir="tepacoc", numproc=1):
	""" Parse the tepacoc test set. """
	trainnumsents = 25005
	for stage in stages:
		for key in stage:
			assert key in defaultstage, "unrecognized option: %r" % key
	stages = [DictObj(**{k: stage.get(k, v)
			for k, v in defaultstage.items()}) for stage in stages]
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
	except IOError: # file not found
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
				logging.error("mismatch. sent %d:\n%r\n%r\n"
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
				and n not in tepacocids][trainnumsents:trainnumsents+2000])
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
			tailmarker, revmarkov, leftmostunary, rightmostunary,
			fanout_marks_before_bin, testmaxwords, resultdir,
			numproc, None, False)

	del corpus_sents, corpus_taggedsents, corpus_trees, corpus_blocks
	results = {}
	cnt = 0
	for cat, testset in sorted(testsets.items()):
		if cat == 'baseline':
			continue
		logging.info("category: %s", cat)
		begin = time.clock()
		results[cat] = doparse(stages, unfolded, bintype,
				fanout_marks_before_bin, testset, testmaxwords, testnumsents,
				trees[0].label, True, resultdir, numproc, tailmarker,
				category=cat)
		cnt += len(testset[0])
		if numproc == 1:
			logging.info("time elapsed during parsing: %g",
					time.clock() - begin)
		#else: # wall clock time here
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
	# write TOTAL results file with all tepacoc sentences (not the baseline)
	writeresults(totresults, goldblocks, goldsents, resultdir, "TOTAL")
	oldeval(totresults, goldbrackets)
	# do baseline separately because it shouldn't count towards the total score
	cat = 'baseline'
	logging.info("category: %s", cat)
	oldeval(*doparse(stages, unfolded, bintype, fanout_marks_before_bin,
			testsets[cat], testmaxwords, testnumsents, trees[0].label,
			True, resultdir, numproc, tailmarker, category=cat))

def cycledetection(trees, sents):
	""" Find trees with cyclic unary productions. """
	seen = set()
	v = set()
	e = {}
	weights = {}
	for tree, sent in zip(trees, sents):
		rules = [(a, b) for a, b in induce_plcfrs([tree], [sent])
				if a not in seen]
		seen.update([a[0] for a in rules])
		for (rule, _), w in rules:
			if len(rule) == 2 and rule[1] != "Epsilon":
				v.add(rule[0])
				e.setdefault(rule[0], set()).add(rule[1])
				weights[rule[0], rule[1]] = abs(w)

	def visit(current, edges, visited):
		""" depth-first cycle detection """
		for a in edges.get(current, set()):
			if a in visited:
				visit.mem.add(current)
				yield visited[visited.index(a):] + [a]
			elif a not in visit.mem:
				for b in visit(a, edges, visited + [a]):
					yield b
	visit.mem = set()
	for a in v:
		for b in visit(a, e, []):
			logging.debug("cycle (cost %5.2f): %s",
				sum(weights[c, d] for c, d in zip(b, b[1:])), " => ".join(b))

def dotagging(usetagger, model, sents, overridetag, tagmap):
	""" Use an external tool to tag a list of tagged sentences. """
	logging.info("Start tagging.")
	goldtags = [t for sent in sents.values() for _, t in sent]
	if usetagger == "treetagger": # Tree-tagger
		installation = """tree tagger not found. commands to install:
mkdir tree-tagger && cd tree-tagger
wget ftp://ftp.ims.uni-stuttgart.de/pub/corpora/tree-tagger-linux-3.2.tar.gz
tar -xzf tree-tagger-linux-3.2.tar.gz
wget ftp://ftp.ims.uni-stuttgart.de/pub/corpora/tagger-scripts.tar.gz
tar -xzf ftp://ftp.ims.uni-stuttgart.de/pub/corpora/tagger-scripts.tar.gz
mkdir lib && cd lib
wget ftp://ftp.ims.uni-stuttgart.de/pub/corpora/german-par-linux-3.2-utf8.bin.gz
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
		tagout = tagger.stdout.read().decode('utf-8').split("<S>")[:-1]
		os.unlink(inname)
		taggedsents = OrderedDict((n, [tagmangle(a, None, overridetag, tagmap)
					for a in tags.splitlines() if a.strip()])
					for n, tags in zip(sents, tagout))
	elif usetagger == "stanford": # Stanford Tagger
		installation = """Stanford tagger not found. Commands to install:
wget http://nlp.stanford.edu/software/stanford-postagger-full-2012-07-09.tgz
tar -xzf stanford-postagger-full-2012-07-09.tgz"""
		assert os.path.exists("stanford-postagger-full-2012-07-09"), installation
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
		tagout = tagger.stdout.read().decode('utf-8').splitlines()
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

sentend = "(\"'!?..." # ";/-"
def wordmangle(w, n, sent):
	""" Function to filter words before they are sent to the tagger. """
	#if n > 0 and w[0] in string.uppercase and not sent[n-1] in sentend:
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

def testmain():
	# Tiger treebank version 2 sample:
	# http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/annotation/sample2.export
	main(
		stages=[dict(
			mode='pcfg', # use the dedicated PCFG parser
			split=True,
			markorigin=True, #mark origin of split nodes: VP_2 => {VP*1, VP*2}
			getestimates=False, #compute & store estimates
			useestimates=False,  #load & use estimates
		),
		dict(
			mode='plcfrs', # the agenda-based PLCFRS parser
			prune=True,	#whether to use previous chart to prune this stage
			splitprune=True, #VP_2[101] is treated as { VP*[100], VP*[001] }
			k=50, #number of coarse pcfg derivations to prune with;
					#k=0 => filter only
			neverblockre=None, #do not prune nodes with label that match regex
			getestimates=False, #compute & store estimates
			useestimates=False,  #load & use estimates
		),
		dict(
			mode='plcfrs', # the agenda-based PLCFRS parser
			prune=True,	#whether to use previous chart to prune this stage
			k=50,		#number of coarse plcfrs derivations to prune with;
					#k=0 => filter only
			dop=True,
			usedoubledop=False,	# when False, use DOP reduction instead
			iterate=False, #for double dop, whether to include
					#fragments of fragments
			complement=False, #for double dop, whether to include fragments
					#that are the complement of the extracted fragments
			sample=False, kbest=True,
			m=10000,		#number of derivations to sample/enumerate
			estimator="ewe", # choices: dop1, ewe
			objective="mpp", # choices: mpp, mpd, shortest, sl-dop[-simple]
			# NB: w/shortest derivation, estimator only affects tie breaking.
			sldop_n=7,
			neverblockre=None, #do not prune nodes with label that match regex
		)],
		corpusdir=".",
		traincorpus="sample2.export",
		testcorpus="sample2.export",
		testencoding="iso-8859-1",
		trainencoding="iso-8859-1",
		punct="move",
		unfolded=False,
		testmaxwords=40,
		trainmaxwords=40,
		trainnumsents=2,
		testnumsents=1, # number of sentences to parse
		skip=0,	# dev set
		#skip=1000, #skip dev set to get test set
		postagging=None,	#default is to use gold tags from treebank.
		bintype="binarize", # choices: binarize, optimal, optimalhead
		factor="right",
		revmarkov=True,
		v=1,
		h=2,
		fanout_marks_before_bin=False,
		tailmarker="",
		quiet=False, reallyquiet=False, #quiet=no per sentence results
		numproc=1,	#increase to use multiple CPUs. Set to None to use all CPUs.
	)

def test():
	""" Run doctests and other tests from all modules. """
	from doctest import testmod, NORMALIZE_WHITESPACE, ELLIPSIS
	import bit, demos, kbest, grammar, treebank, estimates, _fragments, _parser
	import agenda, coarsetofine, treetransforms, disambiguation, eval
	import gen, treedist, tree
	modules = (bit, eval, demos, kbest, _parser, grammar, treebank, estimates,
			_fragments, agenda, coarsetofine, treetransforms, treedist, gen,
			tree, disambiguation)
	results = {}
	for mod in modules:
		print('running doctests of', mod.__file__)
		results[mod] = fail, attempted = testmod(mod, verbose=False,
			optionflags=NORMALIZE_WHITESPACE | ELLIPSIS)
		assert fail == 0, mod.__file__
	if any(not attempted for fail, attempted in results.values()):
		print("no doctests:")
		for mod, (fail, attempted) in results.items():
			if not attempted:
				print(mod.__file__, end='')
		print()
	for mod in modules:
		if hasattr(mod, 'test'):
			mod.test()
		else:
			mod.main()
	#testmain() # test this module (runexp)
	for mod, (fail, attempted) in sorted(results.items(),
			key=itemgetter(1)):
		if attempted:
			print('%s: %d doctests succeeded!' % (mod.__file__, attempted))

if __name__ == '__main__':
	sys.stdout = codecs.getwriter('utf8')(sys.stdout)
	if len(sys.argv) == 1:
		print(USAGE)
	elif '--test' in sys.argv:
		test()
	elif '--tepacoc' in sys.argv:
		parsetepacoc()
	else:
		paramstr = open(sys.argv[1]).read()
		theparams = eval("dict(%s)" % paramstr)
		theparams['resultdir'] = sys.argv[1].rsplit(".", 1)[0]
		theparams['rerun'] = '--rerun' in sys.argv
		main(**theparams)
		# copy parameter file to result dir
		open("%s/params.prm" % theparams['resultdir'], "w").write(paramstr)
