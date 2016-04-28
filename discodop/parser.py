"""Parser object that performs coarse-to-fine and postprocessing.

Additionally, a simple command line interface similar to bitpar."""
from __future__ import division, print_function, absolute_import, \
		unicode_literals
import io
import os
import re
import sys
import gzip
import json
import time
import codecs
import logging
import tempfile
import traceback
import string  # pylint: disable=W0402
import multiprocessing
from math import exp, log
from heapq import nlargest
from getopt import gnu_getopt, GetoptError
from operator import itemgetter
if sys.version_info[0] == 2:
	from itertools import imap as map  # pylint: disable=E0611,W0622
	import cPickle as pickle  # pylint: disable=import-error
else:
	import pickle
import numpy as np
from . import plcfrs, pcfg, disambiguation
from . import grammar, treetransforms, treebanktransforms
from .containers import Grammar, BITPARRE
from .coarsetofine import prunechart
from .tree import ParentedTree, escape, ptbescape
from .eval import alignsent
from .lexicon import replaceraretestwords, UNKNOWNWORDFUNC, UNK
from .treebank import writetree, handlefunctions
from .heads import saveheads, readheadrules
from .punctuation import punctprune, applypunct
from .functiontags import applyfunctionclassifier
from .util import workerfunc, openread
from .treetransforms import binarizetree

SHORTUSAGE = '''
usage: discodop parser [options] <grammar/> [input [output]]
or:    discodop parser --simple [options] <rules> <lexicon> [input [output]]'''

DEFAULTS = dict(
	# two-level keys:
	traincorpus=dict(
		# filenames may include globbing characters '*' and '?'.
		path='alpinosample.export',
		encoding='utf8',
		maxwords=40,  # limit on train set sentences
		numsents=2),  # size of train set (before applying maxwords)
	testcorpus=dict(
		path='alpinosample.export',
		encoding='utf8',
		maxwords=40,  # test set length limit
		numsents=1,  # size of test set (before length limit)
		skiptrain=True,  # test set starts after training set
		# (useful when they are in the same file)
		skip=0),  # number of sentences to skip from test corpus
	binarization=dict(
		method='default',  # choices: default, optimal, optimalhead
		factor='right',
		headrules=None,  # rules for finding heads of constituents
		v=1, h=2, revh=0,
		markhead=False,  # prepend head to siblings
		leftmostunary=False,  # start binarization with unary node
		rightmostunary=False,  # end binarization with unary node
		tailmarker='',  # with headrules, head is last node and can be marked
		markovthreshold=None,
		labelfun=None,
		fanout_marks_before_bin=False),
	# other keys:
		corpusfmt='export',  # choices: export, (disc)bracket, alpino, tiger
		removeempty=False,  # whether to remove empty terminals
		ensureroot=None,  # ensure every tree has a root node with this label
		punct=None,  # choices: None, 'move', 'remove', 'root'
		functions=None,  # choices None, 'add', 'remove', 'replace'
		morphology=None,  # choices: None, 'add', 'replace', 'between'
		transformations=None,  # apply treebank transformations
		postagging=None,  # postagging: pass None to use tags from treebank.
		relationalrealizational=None,  # do not apply RR-transform
		predictfunctions=False,  # use discriminative classifier to add
				# grammatical functions in postprocessing step
		evalparam='proper.prm',  # EVALB-style parameter file
		verbosity=2,
		numproc=1)  # increase to use multiple CPUs; None: use all CPUs.

DEFAULTSTAGE = dict(
		name='stage1',  # identifier, used for filenames
		mode='plcfrs',  # use the agenda-based PLCFRS parser
		prune=False,  # whether to use previous chart to prune this stage
		split=False,  # split disc. nodes VP_2[101] as { VP*[100], VP*[001] }
		splitprune=False,  # treat VP_2[101] as {VP*[100], VP*[001]} for pruning
		markorigin=False,  # mark origin of split nodes: VP_2 => {VP*1, VP*2}
		collapselabels=None,  # options: None, 'head', 'all'.
		k=50,  # no. of coarse pcfg derivations to prune with; k=0: filter only
		dop=None,  # DOP mode: dopreduction, doubledop, dop1
		binarized=True,  # for doubledop, whether to binarize extracted grammar
		# (False requires use of bitpar)
		maxdepth=1,  # for dop1 & doubledop cover fragments,
		# maximum depth of fragments to extract.
		maxfrontier=999,  # for dop1 & doubledop cover fragments,
		# maximum frontier NTs in fragments.
		sample=False, kbest=True,
		m=10,  # number of derivations to sample/enumerate
		estimator='rfe',  # choices: rfe, ewe
		objective='mpp',  # choices: mpp, mpd, shortest, sl-dop[-simple]
			# NB: w/shortest derivation, estimator only affects tie breaking.
		sldop_n=7,  # number of trees to consider when using sl-dop[-simple]
		mcc_labda=1.0,  # weight to assign to recall vs. mistake rate with mcc
		mcc_labels=None,  # optionally, set of labels to optimize for with mcc
		packedgraph=False,  # use packed graph encoding for DOP reduction
		iterate=False,  # for double dop, whether to add fragments of fragments
		complement=False,  # for double dop, whether to include fragments which
			# form the complement of the maximal recurring fragments extracted
		neverblockre=None,  # do not prune nodes with label that match regex
		estimates=None,  # compute, store & use outside estimates
		beam_beta=1.0,  # beam pruning factor, between 0 and 1; 1 to disable.
		beam_delta=40,  # maximum span length to which beam_beta is applied
		collapse=None,  # optionally, collapse phrase labels for multilevel CTF
		)


class DictObj(object):
	"""Trivial class to wrap a dictionary for reasons of syntactic sugar."""

	def __init__(self, *args, **kwds):
		self.__dict__.update(*args, **kwds)

	def update(self, *args, **kwds):
		"""Update/add more attributes."""
		self.__dict__.update(*args, **kwds)

	def __getattr__(self, name):
		"""Dummy function for suppressing pylint E1101 errors."""
		raise AttributeError('%r instance has no attribute %r.\n'
				'Available attributes: %r' % (
				self.__class__.__name__, name, self.__dict__.keys()))

	def __repr__(self):
		return '%s(%s)' % (self.__class__.__name__,
			',\n\t'.join('%s=%r' % a for a in self.__dict__.items()))


PARAMS = DictObj()  # used for multiprocessing when using CLI of this module


class Parser(object):
	"""A coarse-to-fine parser based on a given set of parameters.

	:param prm: A DictObj with parameters as returned by
		:py:func:`parser.readparam()`.
	:param funcclassifier: optionally, a function tag classifier trained by
		:py:func:`functiontags.trainfunctionclassifier`.
	"""

	def __init__(self, prm, funcclassifier=None):
		self.prm = prm
		self.stages = prm.stages
		self.transformations = prm.transformations
		self.binarization = prm.binarization
		self.postagging = prm.postagging
		self.relationalrealizational = prm.relationalrealizational
		self.verbosity = prm.verbosity
		self.funcclassifier = funcclassifier
		for stage in prm.stages:
			model = 'default'
			if stage.dop:
				if stage.objective == 'shortest':
					model = 'shortest'
				elif stage.estimator != 'rfe':
					model = stage.estimator
			if stage.mode != 'mc-rerank':
				stage.grammar.switch(model, logprob=True)
			if prm.verbosity >= 3:
				logging.debug(stage.name)
				logging.debug(stage.grammar)

	def parse(self, sent, tags=None, goldtree=None):
		"""Parse a sentence and perform postprocessing.

		Yields a dictionary from parse trees to probabilities for each stage.

		:param sent: a sequence of tokens.
		:param tags: optionally, a list of POS tags as strings to be given
			to the parser instead of trying all possible tags.
		:param goldtree: if given, will be used to evaluate pruned parse
			forests."""
		if 'PUNCT-PRUNE' in (self.transformations or ()):
			origsent = sent[:]
			punctprune(None, sent)
			if tags:
				newtags = alignsent(sent, origsent, dict(enumerate(tags)))
				tags = [newtags[n] for n, _ in enumerate(sent)]
		if 'PTBbrackets' in (self.transformations or ()):
			sent = xsent = [ptbescape(token) for token in sent]
		else:
			xsent = list(sent)
			sent = [escape(token) for token in sent]
		if self.postagging and self.postagging.method == 'unknownword':
			sent = list(replaceraretestwords(sent,
					self.postagging.unknownwordfun,
					self.postagging.lexicon, self.postagging.sigs))
		if tags is not None:
			tags = list(tags)

		if goldtree is not None:
			# reproduce preprocessing so that gold items can be counted
			goldtree = goldtree.copy(True)
			applypunct(self.prm.punct, goldtree, sent[:])
			if self.prm.transformations:
				treebanktransforms.transform(goldtree, sent,
						self.prm.transformations)
			binarizetree(goldtree, self.prm.binarization,
					self.relationalrealizational)
			treetransforms.addfanoutmarkers(goldtree)

		charts = {}  # stage.name => chart
		prevparsetrees = {}  # stage.name => parsetrees
		chart = lastsuccessfulparse = None
		totalgolditems = 0
		# parse with each coarse-to-fine stage
		for n, stage in enumerate(self.stages):
			begin = time.clock()
			noparse = False
			parsetrees = fragments = None
			golditems = 0
			msg = '%s:\t' % stage.name.upper()
			model = 'default'
			if stage.dop:
				if stage.objective == 'shortest':
					model = 'shortest'
				elif stage.estimator != 'rfe':
					model = stage.estimator
			if stage.mode != 'mc-rerank':
				x = stage.grammar.currentmodel
				stage.grammar.switch(model, logprob=True)
			if stage.mode.startswith('pcfg-bitpar'):
				if (not hasattr(stage, 'rulesfile')
						or x != stage.grammar.currentmodel):
					exportbitpargrammar(stage)
			elif hasattr(stage, 'rulesfile'):
				del stage.rulesfile, stage.lexiconfile
			if not stage.binarized and not stage.mode.startswith('pcfg-bitpar'):
				raise ValueError('non-binarized grammar requires use of bitpar')

			# do parsing; if CTF pruning enabled, require parent stage to
			# be successful.
			if sent and (not stage.prune or charts[stage.prune]):
				prevn = 0
				if stage.prune:
					prevn = [a.name for a in self.stages].index(stage.prune)
				tree = goldtree
				if goldtree is not None and self.stages[prevn].split:
					tree = treetransforms.splitdiscnodes(
							goldtree.copy(True), self.stages[prevn].markorigin)
				if n > 0 and stage.prune and stage.mode not in (
						'dop-rerank', 'mc-rerank'):
					beginprune = time.clock()
					whitelist, items, msg1 = prunechart(
							charts[stage.prune], stage.grammar, stage.k,
							stage.splitprune,
							self.stages[prevn].markorigin,
							stage.mode.startswith('pcfg'),
							self.stages[prevn].mode == 'pcfg-bitpar-nbest')
					# count number of gold bracketings in pruned chart
					if goldtree is not None and items:
						item = next(iter(items))
						for node in tree.subtrees():
							if (charts[stage.prune].toitem(node, item)
									in items):
								fanout = re.search('_([0-9]+)$',
										node.label)
								golditems += (int(fanout.group(1))
										if fanout and not stage.split
										else 1)
						msg1 += (';\n\t%d/%d gold items remain after '
								'pruning' % (golditems, totalgolditems))
					msg += '%s; %gs\n\t' % (msg1, time.clock() - beginprune)
				else:
					whitelist = None
				if not sent:
					pass
				elif stage.mode == 'pcfg':
					chart, msg1 = pcfg.parse(
							sent, stage.grammar, tags=tags,
							whitelist=whitelist if stage.prune else None,
							symbolic=False,
							beam_beta=-log(stage.beam_beta),
							beam_delta=stage.beam_delta)
				elif stage.mode.startswith('pcfg-bitpar'):
					if stage.mode == 'pcfg-bitpar-forest':
						numderivs = 0
					elif (n == len(self.stages) - 1
							or not self.stages[n + 1].prune):
						numderivs = stage.m
					else:  # request 1000 nbest parses for CTF pruning
						numderivs = 1000
					chart, cputime, msg1 = pcfg.parse_bitpar(stage.grammar,
							stage.rulesfile.name, stage.lexiconfile.name,
							sent, numderivs, stage.grammar.start,
							stage.grammar.toid[stage.grammar.start], tags=tags)
					begin -= cputime
				elif stage.mode == 'plcfrs':
					chart, msg1 = plcfrs.parse(
							sent, stage.grammar, tags=tags,
							exhaustive=stage.dop or (
								n + 1 != len(self.stages)
								and self.stages[n + 1].prune),
							whitelist=whitelist,
							splitprune=stage.splitprune
								and self.stages[prevn].split,
							markorigin=self.stages[prevn].markorigin,
							estimates=(stage.estimates, stage.outside)
								if stage.estimates in ('SX', 'SXlrgaps')
								else None,
							symbolic=False,
							beam_beta=-log(stage.beam_beta),
							beam_delta=stage.beam_delta)
				elif stage.mode == 'dop-rerank':
					if prevparsetrees[stage.prune]:
						parsetrees, msg1 = disambiguation.doprerank(
								prevparsetrees[stage.prune], sent, stage.k,
								self.stages[prevn].grammar, stage.grammar)
				elif stage.mode == 'mc-rerank':
					if prevparsetrees[stage.prune]:
						parsetrees, msg1 = disambiguation.mcrerank(
								prevparsetrees[stage.prune], sent, stage.k,
								stage.grammar.trees1, stage.grammar.vocab)
				else:
					raise ValueError('unknown mode specified.')
				msg += '%s\n\t' % msg1
				if (n > 0 and stage.prune and not chart and not noparse
						and stage.split == self.stages[prevn].split):
					logging.error('ERROR: expected successful parse. '
							'sent: %s\nstage %d: %s.',
							' '.join(sent), n, stage.name)
					# raise ValueError('ERROR: expected successful parse. '
					# 		'sent %s, %s.' % (nsent, stage.name))
			numitems = (len(chart.getitems())
					if hasattr(chart, 'getitems') else 0)

			# do disambiguation of resulting parse forest
			if (sent and chart and stage.mode not in ('dop-rerank', 'mc-rerank')
					and not (self.relationalrealizational and stage.split)):
				begindisamb = time.clock()
				if stage.mode == 'pcfg-bitpar-nbest':
					if not stage.kbest or stage.sample:
						raise ValueError('sampling not possible with bitpar '
								'in nbest mode.')
					derivations = chart.rankededges[chart.root()]
					entries = [None] * len(derivations)
				else:
					derivations, entries = disambiguation.getderivations(
							chart, stage.m, kbest=stage.kbest,
							sample=stage.sample,
							derivstrings=stage.dop not in ('doubledop', 'dop1')
									or self.verbosity >= 3
									or stage.objective == 'mcc')
				if self.verbosity >= 3:
					print('sent: %s\nstage: %s' % (' '.join(sent), stage.name))
					print('%d-best derivations:\n%s' % (
						min(stage.m, 100),
						'\n'.join('%d. %s %s' % (n + 1,
							('subtrees=%d' % abs(int(prob / log(0.5))))
							if stage.objective == 'shortest'
							else ('p=%g' % exp(-prob)), deriv)
						for n, (deriv, prob) in enumerate(derivations[:100]))))
					print('sum of probabitilies: %g\n' %
							sum(exp(-prob) for _, prob in derivations[:100]))
				if stage.objective == 'shortest':
					stage.grammar.switch('default' if stage.estimator == 'rfe'
							else stage.estimator, True)
				parsetrees, msg1 = disambiguation.marginalize(
						stage.objective if stage.dop else 'mpd',
						derivations, entries, chart,
						sent=sent, tags=tags,
						backtransform=stage.backtransform,
						k=stage.m, sldop_n=stage.sldop_n,
						mcc_labda=stage.mcc_labda, mcc_labels=stage.mcc_labels,
						bitpar=stage.mode == 'pcfg-bitpar-nbest')
				msg += 'disambiguation: %s, %gs\n\t' % (
						msg1, time.clock() - begindisamb)
				if self.verbosity >= 3:
					besttrees = nlargest(100, parsetrees, key=itemgetter(1))
					print('100-best parse trees:\n%s' % '\n'.join(
							'%d. %s %s' % (n + 1, probstr(prob), treestr)
							for n, (treestr, prob, _) in enumerate(besttrees)))
					print('sum of probabitilies: %g\n' %
							sum((prob[1] if isinstance(prob, tuple) else prob)
								for _, prob, _ in besttrees))
				if not stage.prune and tree is not None:
					item = next(iter(chart.rankededges))
					totalgolditems = sum(1 for node in tree.subtrees())
					golditems = sum(
							1 for node in tree.subtrees()
							if chart.toitem(node, item) in chart)
					msg += ('%d/%d gold items in derivations\n\t' % (
							golditems, totalgolditems))
			if self.verbosity >= 4:
				print('Chart:\n%s' % chart)
			if stage.name in (stage.prune for stage in self.stages):
				charts[stage.name] = chart
				prevparsetrees[stage.name] = parsetrees

			# postprocess, yield result
			if parsetrees:
				try:
					resultstr, prob, fragments = max(
							parsetrees, key=itemgetter(1))
					parsetree, noparse = self.postprocess(resultstr, xsent, n)
					if not all(a for a in parsetree.subtrees()):
						raise ValueError('empty nodes in tree: %s' % parsetree)
					if len(parsetree.leaves()) != len(sent):
						raise ValueError('leaves missing. original tree: %s\n'
							'postprocessed: %r' % (resultstr, parsetree))
				except Exception:  # pylint: disable=W0703
					logging.error("something's amiss. %s", ''.join(
								traceback.format_exception(*sys.exc_info())))
					parsetree, prob, noparse = self.noparse(
							stage, xsent, tags, lastsuccessfulparse)
				else:
					lastsuccessfulparse = parsetree
				msg += probstr(prob) + ' '
			else:
				fragments = None
				parsetree, prob, noparse = self.noparse(
						stage, xsent, tags, lastsuccessfulparse)
			elapsedtime = time.clock() - begin
			msg += '%.2fs cpu time elapsed\n' % (elapsedtime)
			yield DictObj(name=stage.name, parsetree=parsetree, prob=prob,
					parsetrees=parsetrees, fragments=fragments,
					noparse=noparse, elapsedtime=elapsedtime,
					numitems=numitems, golditems=golditems,
					totalgolditems=totalgolditems, msg=msg)
		del charts, prevparsetrees

	def postprocess(self, treestr, sent, stage):
		"""Take parse tree and apply postprocessing."""
		parsetree = ParentedTree(treestr)
		if self.stages[stage].split:
			treetransforms.mergediscnodes(treetransforms.unbinarize(
					parsetree, childchar=':', expandunary=False))
		if self.binarization.tailmarker or self.binarization.headrules:
			saveheads(parsetree, self.binarization.tailmarker)
		treetransforms.unbinarize(parsetree, expandunary=False)
		treetransforms.removefanoutmarkers(parsetree)
		if self.relationalrealizational:
			parsetree = treebanktransforms.rrbacktransform(parsetree,
					self.relationalrealizational['adjunctionlabel'])
		if self.funcclassifier is not None:
			applyfunctionclassifier(self.funcclassifier, parsetree, sent)
		if self.transformations:
			treebanktransforms.reversetransform(parsetree, self.transformations)
		else:
			treetransforms.canonicalize(parsetree)
		return parsetree, False

	def noparse(self, stage, sent, tags, lastsuccessfulparse):
		"""Return parse from previous stage or a dummy parse."""
		# use successful parse from earlier stage if available
		if lastsuccessfulparse is not None:
			parsetree = lastsuccessfulparse.copy(True)
		else:  # Produce a dummy parse for evaluation purposes.
			default = grammar.defaultparse([(n, t) for n, t
					in enumerate(tags or (len(sent) * ['NONE']))])
			parsetree = ParentedTree(
					'(%s %s)' % (stage.grammar.start, default))
		noparse = True
		prob = 1.0
		return parsetree, prob, noparse


def readgrammars(resultdir, stages, postagging=None, top='ROOT'):
	"""Read the grammars from a previous experiment.

	Expects a directory ``resultdir`` which contains the relevant grammars and
	the parameter file ``params.prm``, as produced by ``runexp``."""
	if os.path.exists('%s/mapping.json.gz' % resultdir):
		mappings = json.load(openread('%s/mapping.json.gz' % resultdir))
		for stage, mapping in zip(stages, mappings):
			stage.mapping = mapping
	else:
		for stage in stages:
			stage.mapping = None
	for n, stage in enumerate(stages):
		logging.info('reading: %s', stage.name)
		if stage.mode != 'mc-rerank':
			rules = openread('%s/%s.rules.gz' % (resultdir, stage.name)).read()
			lexicon = openread('%s/%s.lex.gz' % (resultdir, stage.name))
			xgrammar = Grammar(rules, lexicon.read(),
					start=top, binarized=stage.binarized)
		backtransform = outside = None
		prevn = 0
		if n and stage.prune:
			prevn = [a.name for a in stages].index(stage.prune)
		if stage.mode == 'mc-rerank':
			xgrammar = pickle.loads(gzip.open('%s/%s.train.pickle.gz' % (
					resultdir, stage.name), 'rb').read())
		elif stage.dop:
			if stage.estimates is not None:
				raise ValueError('not supported')
			if stage.dop in ('doubledop', 'dop1'):
				backtransform = openread('%s/%s.backtransform.gz' % (
						resultdir, stage.name)).read().splitlines()
				if n and stage.prune:
					_ = xgrammar.getmapping(stages[prevn].grammar,
						striplabelre=re.compile('@.+$'),
						neverblockre=re.compile('^#[0-9]+|.+}<'),
						splitprune=stage.splitprune and stages[prevn].split,
						markorigin=stages[prevn].markorigin,
						mapping=stage.mapping)
				else:
					# recoverfragments() relies on this mapping to identify
					# binarization nodes
					_ = xgrammar.getmapping(None,
						neverblockre=re.compile('.+}<'))
			elif n and stage.prune:  # dop reduction
				_ = xgrammar.getmapping(stages[prevn].grammar,
					striplabelre=re.compile('@[-0-9]+$'),
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[prevn].split,
					markorigin=stages[prevn].markorigin,
					mapping=stage.mapping)
				if stage.mode == 'dop-rerank':
					xgrammar.getrulemapping(
							stages[prevn].grammar, re.compile(r'@[-0-9]+\b'))
			probsfile = '%s/%s.probs.npz' % (resultdir, stage.name)
			if os.path.exists(probsfile):
				probmodels = np.load(probsfile)
				for name in probmodels.files:
					if name != 'default':
						xgrammar.register(name, probmodels[name])
		else:  # not stage.dop
			if n and stage.prune:
				_ = xgrammar.getmapping(stages[prevn].grammar,
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[prevn].split,
					markorigin=stages[prevn].markorigin,
					mapping=stage.mapping)
			if stage.estimates in ('SX', 'SXlrgaps'):
				if stage.estimates == 'SX' and xgrammar.maxfanout != 1:
					raise ValueError('SX estimate requires PCFG.')
				if stage.mode != 'plcfrs':
					raise ValueError('estimates require parser w/agenda.')
				outside = np.load('%s/%s.outside.npz' % (
						resultdir, stage.name))['outside']
				logging.info('loaded %s estimates', stage.estimates)
			elif stage.estimates:
				raise ValueError('unrecognized value; specify SX or SXlrgaps.')

		if stage.mode.startswith('pcfg-bitpar'):
			if xgrammar.maxfanout != 1:
				raise ValueError('bitpar requires a PCFG.')

		if stage.mode != 'mc-rerank':
			_sumsto1, msg = xgrammar.testgrammar()
		logging.info('%s: %s', stage.name, msg)
		stage.update(grammar=xgrammar, backtransform=backtransform,
				outside=outside)
	if postagging and postagging.method == 'unknownword':
		postagging.unknownwordfun = UNKNOWNWORDFUNC[postagging.model]
		postagging.lexicon = {w for w in stages[0].grammar.lexicalbyword
				if not w.startswith(UNK)}
		postagging.sigs = {w for w in stages[0].grammar.lexicalbyword
				if w.startswith(UNK)}


def exportbitpargrammar(stage):
	"""(re-)export bitpar grammar with current weights."""
	if not hasattr(stage, 'rulesfile'):
		stage.rulesfile = tempfile.NamedTemporaryFile()
		stage.lexiconfile = tempfile.NamedTemporaryFile()
	stage.rulesfile.seek(0)
	stage.rulesfile.truncate()
	if not BITPARRE.match(stage.grammar.origrules):
		# convert to bitpar format
		stage.rulesfile.writelines(
				'%g\t%s\n' % (weight, line.rsplit('\t', 2)[0])
				for weight, line in
				zip(stage.grammar.models[stage.grammar.currentmodel],
					stage.grammar.origrules.splitlines()))
	elif stage.grammar.currentmodel != 0:
		# merge current weights
		stage.rulesfile.writelines(
				'%g\t%s\n' % (weight, line.split(None, 1)[1])
				for weight, line in
				zip(stage.grammar.models[stage.grammar.currentmodel],
					stage.grammar.origrules.splitlines()))
	else:
		stage.rulesfile.write(stage.grammar.origrules)
	stage.rulesfile.flush()

	stage.lexiconfile.seek(0)
	stage.lexiconfile.truncate()
	lexicon = escape(stage.grammar.origlexicon)  # escape brackets
	lexiconfile = codecs.getwriter('utf-8')(stage.lexiconfile)
	if stage.grammar.currentmodel == 0:
		lexiconfile.write(lexicon)
	else:
		weights = iter(stage.grammar.models[stage.grammar.currentmodel,
				stage.grammar.numrules:])
		lexiconfile.writelines('%s\t%s\n' % (line.split(None, 1)[0],
				'\t'.join('%s %g' % (tag, next(weights))
					for tag in line.split()[1::2]))
				for line in lexicon.splitlines())
	stage.lexiconfile.flush()


def probstr(prob):
	"""Render probability / number of subtrees as string."""
	if isinstance(prob, tuple):
		return 'subtrees=%d, p=%.4g ' % (abs(prob[0]), prob[1])
	return 'p=%.4g' % prob


def readparam(filename):
	"""Parse a parameter file.

	:param filename: The file should contain a list of comma-separated
		``attribute=value`` pairs and will be read using ``eval('dict(%s)' %
		open(file).read())``.
	:returns: A DictObj."""
	with io.open(filename, encoding='utf8') as fileobj:
		params = eval('dict(%s)' % fileobj.read())  # pylint: disable=eval-used
	for key in DEFAULTS:
		if key not in params:
			if isinstance(DEFAULTS[key], dict):
				raise ValueError('%r not in parameters.' % key)
			else:
				params[key] = DEFAULTS[key]
	for stage in params['stages']:
		for key in stage:
			if key not in DEFAULTSTAGE:
				raise ValueError('unrecognized option: %r' % key)
	params['stages'] = [DictObj({k: stage.get(k, v)
			for k, v in DEFAULTSTAGE.items()})
				for stage in params['stages']]
	for key in DEFAULTS:
		if isinstance(DEFAULTS[key], dict):
			params[key] = DictObj({k: params[key].get(k, v)
					for k, v in DEFAULTS[key].items()})
	for n, stage in enumerate(params['stages']):
		if stage.mode not in (
				'plcfrs', 'pcfg', 'pcfg-bitpar-nbest', 'pcfg-bitpar-forest',
				'dop-rerank', 'mc-rerank'):
			raise ValueError('unrecognized mode argument: %r.' % stage.mode)
		if n == 0 and stage.prune:
			raise ValueError('need previous stage to prune, '
					'but this stage is first.')
		if stage.prune is True:  # for backwards compatibility
			stage.prune = params['stages'][n - 1].name
		if stage.mode == 'dop-rerank':
			assert stage.prune and not stage.splitprune and stage.k > 1
			assert (stage.dop and stage.dop not in ('doubledop', 'dop1')
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
		postagging = params['postagging'] = DictObj(postagging)
		if postagging.method == 'unknownword':
			assert postagging.model in UNKNOWNWORDFUNC
			assert postagging.unknownthreshold >= 1
			assert postagging.openclassthreshold >= 0
		else:
			assert postagging.method in ('treetagger', 'stanford', 'frog')
	if params['transformations']:
		params['transformations'] = treebanktransforms.expandpresets(
				params['transformations'])
	return DictObj(params)


def readinputbitparstyle(infile):
	"""Yields lists of tokens, where '\\n\\n' identifies a sentence break.

	Lazy version of ``infile.read().split('\\n\\n')``."""
	sent = []
	for line in infile:
		line = line.strip()
		if not line:
			yield ' '.join(sent)
			sent = []
		sent.append(line)
	if sent:
		yield ' '.join(sent)


def initworker(parser, printprob, usetags, numparses,
		fmt, morphology):
	"""Load parser for a worker process."""
	headrules = None
	if fmt in ('mst', 'conll'):
		headrules = readheadrules(parser.binarization.headrules)
	PARAMS.update(parser=parser, printprob=printprob,
			usetags=usetags, numparses=numparses, fmt=fmt,
			morphology=morphology, headrules=headrules)


@workerfunc
def mpworker(args):
	"""Parse a single sentence (multiprocessing wrapper)."""
	return worker(args)


def worker(args):
	"""Parse a single sentence."""
	key, line = args
	line = line.strip()
	if not line:
		return '', True, 0, ''
	begin = time.clock()
	sent = line.split(' ')
	tags = None
	if PARAMS.usetags:
		sent, tags = zip(*(a.rsplit('/', 1) for a in sent))
	msg = 'parsing %s: %s' % (key, ' '.join(sent))
	result = list(PARAMS.parser.parse(sent, tags=tags))[-1]
	output = ''
	if result.noparse:
		msg += '\nNo parse for "%s"' % ' '.join(sent)
		if PARAMS.printprob:
			output += 'prob=%.16g\n' % result.prob
		output += writetree(
				result.parsetree, sent,
				key if PARAMS.numparses == 1 else ('%s-1' % key),
				PARAMS.fmt, morphology=PARAMS.morphology,
				comment=('prob=%.16g' % result.prob)
					if PARAMS.printprob else None)
	else:
		tmp = []
		for k, (tree, prob, _) in enumerate(nlargest(
				PARAMS.numparses, result.parsetrees, key=itemgetter(1))):
			tree, _ = PARAMS.parser.postprocess(tree, sent, -1)
			if 'bracket' in PARAMS.fmt:
				handlefunctions('add', tree)
			tmp.append(writetree(
					tree, sent,
					key if PARAMS.numparses == 1 else ('%s-%d' % (key, k)),
					PARAMS.fmt, morphology=PARAMS.morphology,
					comment=('prob=%.16g' % prob)
						if PARAMS.printprob else None))
		output += ''.join(tmp)
	sec = time.clock() - begin
	msg += '\n%g s' % sec
	return output, result.noparse, sec, msg


def doparsing(parser, infile, out, printprob, oneline, usetags, numparses,
		numproc, fmt, morphology, sentid):
	"""Parse sentences from file and write results to file, log to stdout."""
	times = []
	unparsed = 0
	if not oneline:
		infile = readinputbitparstyle(infile)
	if sentid:
		infile = (line.split('|', 1) for line in infile if line.strip())
	else:
		infile = enumerate((line for line in infile if line.strip()), 1)
	if numproc == 1:
		initworker(parser, printprob, usetags, numparses, fmt, morphology)
		mymap, myworker = map, worker
	else:
		pool = multiprocessing.Pool(
				processes=numproc, initializer=initworker,
				initargs=(parser, printprob, usetags, numparses, fmt,
					morphology))
		mymap, myworker = pool.map, mpworker
	for output, noparse, sec, msg in mymap(myworker, infile):
		if output:
			print(msg, file=sys.stderr)
			out.write(output)
			if noparse:
				unparsed += 1
			times.append(sec)
			sys.stderr.flush()
			out.flush()
	print('average time per sentence', sum(times) / len(times),
			'\nunparsed sentences:', unparsed,
			'\nfinished',
			file=sys.stderr)
	out.close()


def main():
	"""Handle command line arguments."""
	flags = 'help prob tags bitpar sentid simple'.split()
	options = flags + 'obj= bt= numproc= fmt= verbosity='.split()
	try:
		opts, args = gnu_getopt(sys.argv[2:], 'hb:s:m:x', options)
	except GetoptError as err:
		print('error:', err, file=sys.stderr)
		print(SHORTUSAGE)
		sys.exit(2)
	if not 1 <= len(args) <= 4:
		print('error: incorrect number of arguments', file=sys.stderr)
		print(SHORTUSAGE)
		sys.exit(2)
	for n, filename in enumerate(args):
		if not os.path.exists(filename):
			raise ValueError('file %d not found: %r' % (n + 1, filename))
	opts = dict(opts)
	numparses = int(opts.get('-b', 1))
	top = opts.get('-s', 'TOP')
	prob = '--prob' in opts
	tags = '--tags' in opts
	oneline = '-x' not in opts
	sentid = '--sentid' in opts
	if '--simple' in opts:
		if not 2 <= len(args) <= 4:
			print('error: incorrect number of arguments', file=sys.stderr)
			print(SHORTUSAGE)
			sys.exit(2)
		rules = openread(args[0]).read()
		lexicon = openread(args[1]).read()
		bitpar = rules[0] in string.digits
		if '--bitpar' in opts:
			if not bitpar:
				raise ValueError('bitpar requires bitpar grammar format.')
			mode = 'pcfg-bitpar-nbest'
		else:
			mode = 'pcfg' if bitpar else 'plcfrs'
		xgrammar = Grammar(rules, lexicon, start=top,
				binarized='--bitpar' not in opts)
		stages = []
		stage = DEFAULTSTAGE.copy()
		backtransform = None
		if opts.get('--bt'):
			backtransform = openread(opts.get('--bt')).read().splitlines()
		stage.update(
				name='grammar',
				mode=mode,
				grammar=xgrammar,
				binarized='--bitpar' not in opts,
				backtransform=backtransform if len(args) < 4 else None,
				m=numparses,
				objective='mpd')
		if '--obj' in opts:
			stage.update(
					dop='reduction' if backtransform is None else 'doubledop',
					objective=opts['--obj'],
					m=int(opts.get('-m', 1)))
		stages.append(DictObj(stage))
		if backtransform:
			_ = stages[-1].grammar.getmapping(None,
				neverblockre=re.compile('.+}<'))
		prm = DictObj(stages=stages, verbosity=int(opts.get('--verbosity', 2)))
		parser = Parser(prm)
		morph = None
		del args[:2]
	else:
		directory = args[0]
		if not os.path.isdir(directory):
			raise ValueError('expected directory produced by "discodop runexp"')
		params = readparam(os.path.join(directory, 'params.prm'))
		params.update(resultdir=directory)
		readgrammars(directory, params.stages, params.postagging,
				top=getattr(params, 'top', top))
		params.update(verbosity=int(opts.get('--verbosity', params.verbosity)))
		parser = Parser(params)
		morph = params.morphology
		del args[:1]
	infile = openread(args[0] if len(args) >= 1 else '-')
	out = (io.open(args[1], 'w', encoding='utf8')
			if len(args) == 2 else sys.stdout)
	doparsing(parser, infile, out, prob, oneline, tags, numparses,
			int(opts.get('--numproc', 1)), opts.get('--fmt', 'discbracket'),
			morph, sentid)


__all__ = ['DictObj', 'Parser', 'doparsing', 'exportbitpargrammar',
		'initworker', 'probstr', 'readgrammars', 'readinputbitparstyle',
		'readparam']
