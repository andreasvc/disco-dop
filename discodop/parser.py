"""Parser object that performs coarse-to-fine and postprocessing.

Additionally, a simple command line interface similar to bitpar."""
import io
import os
import re
import sys
import gzip
import json
import logging
import traceback
import multiprocessing
from math import exp, log
from time import process_time
from heapq import nlargest
from getopt import gnu_getopt, GetoptError
from operator import itemgetter
import pickle
import numpy as np
from . import plcfrs, pcfg, disambiguation
from . import grammar, treetransforms, treebanktransforms
from .containers import Grammar, Vocabulary, Ctrees
from .coarsetofine import prunechart
from .tree import ParentedTree, escape, ptbescape
from .eval import alignsent
from .lexicon import replaceraretestwords, UNKNOWNWORDFUNC, UNK
from .treebank import writetree, handlefunctions
from .heads import saveheads, readheadrules, applyheadrules
from .punctuation import punctprune, applypunct
from .functiontags import applyfunctionclassifier
from .util import workerfunc, openread
from .treetransforms import binarizetree, binarize, splitdiscnodes
from .grammar import UniqueIDs
from .kbest import partitionincompletechart

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
		numsents=2,  # size of train set (before applying maxwords)
		skip=0),  # number of sentences to skip from train corpus
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
		headrules=None,  # filename of rules for finding heads of constituents
		v=1, h=1, revh=0,
		markhead=False,  # prepend head to siblings
		leftmostunary=False,  # start binarization with unary node
		rightmostunary=False,  # end binarization with unary node
		tailmarker='',  # with headrules, head is last node and can be marked
		markovthreshold=None,
		filterlabels=(),
		labelfun=None,
		direction=True,
		dot=False,
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

		# parameters affecting grammar extraction
		split=False,  # split disc. nodes VP_2[101] as { VP*[100], VP*[001] }
		markorigin=False,  # mark origin of split nodes: VP_2 => {VP*1, VP*2}
		dop=None,  # DOP mode: dopreduction, doubledop, dop1, ostag
		maxdepth=1,  # for dop1 & doubledop cover fragments,
		# maximum depth of fragments to extract.
		maxfrontier=999,  # for dop1 & doubledop cover fragments,
		# maximum frontier NTs in fragments.
		packedgraph=False,  # use packed graph encoding for DOP reduction
		estimates=None,  # compute, store & use outside estimates
		collapse=None,  # optionally, collapse phrase labels for multilevel CTF
		neverblockre=None,  # do not prune nodes with label that match regex

		# parameters that can be changed before parsing each sentence:
		mode='plcfrs',  # use the agenda-based PLCFRS parser
		prune=False,  # whether to use previous chart to prune this stage
		k=50,  # no. of coarse pcfg derivations to prune with; k=0: filter only
		m=10,  # number of derivations to enumerate
		estimator='rfe',  # choices: rfe, ewe
		objective='mpp',  # choices: mpp, mpd, shortest, sl-dop[-simple]
			# NB: w/shortest derivation, estimator only affects tie breaking.
		sldop_n=7,  # number of trees to consider when using sl-dop[-simple]
		mcplambda=1.0,  # weight to assign to recall vs. mistake rate with mcp
		mcplabels=None,  # optionally, set of labels to optimize for with mcp
		beam_beta=1.0,  # beam pruning factor, between 0 and 1; 1 to disable.
		beam_delta=40,  # maximum span length to which beam_beta is applied
		# deprecated options
		kbest=True, sample=False, binarized=True,
		iterate=False, complement=False,
		# now automatically inferred:
		splitprune=False,  # treat VP_2[101] as {VP*[100], VP*[001]} for pruning
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

	def __init__(self, prm, funcclassifier=None, loadtrees=False):
		self.prm = prm
		self.stages = prm.stages
		self.transformations = prm.transformations
		self.binarization = prm.binarization
		self.postagging = prm.postagging
		self.relationalrealizational = prm.relationalrealizational
		self.verbosity = prm.verbosity
		self.funcclassifier = funcclassifier
		self.headrules = None
		if prm.binarization and prm.binarization.headrules and os.path.exists(
				prm.binarization.headrules):
			# FIXME: store headrules in grammar? non-essential
			self.headrules = readheadrules(prm.binarization.headrules)
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
				print(stage.name)
				print(stage.grammar)
		self.ctrees = self.newctrees = self.vocab = None
		self.phrasallabels = self.functiontags = self.poslabels = None
		if loadtrees:
			self._loadtrees()

	def _loadtrees(self):
		from .runexp import loadtraincorpus, getposmodel, dobinarization
		from .containers import REMOVESTATESPLITS
		from . import _fragments
		prm = self.prm
		if os.path.exists('%s/train.ct' % prm.resultdir):
			self.vocab = Vocabulary.fromfile(
					'%s/vocab.idx' % prm.resultdir)
			self.ctrees = Ctrees.fromfilemut('%s/train.ct' % prm.resultdir)
		else:
			trees, sents, train_tagged_sents = loadtraincorpus(
						prm.corpusfmt, prm.traincorpus, prm.binarization,
						prm.punct, prm.functions, prm.morphology,
						prm.removeempty, prm.ensureroot,
						prm.transformations, prm.relationalrealizational,
						prm.resultdir)
			if prm.postagging and prm.postagging.method == 'unknownword':
				sents, _ = getposmodel(prm.postagging, train_tagged_sents)
			trees = dobinarization(trees, sents, prm.binarization,
					prm.relationalrealizational)
			result = _fragments.getctrees(zip(trees, sents))
			self.ctrees, self.vocab = result['trees1'], result['vocab']
			self.ctrees.tofile('%s/train.ct' % prm.resultdir)
			self.vocab.tofile('%s/vocab.idx' % prm.resultdir)
		self.newctrees = Ctrees()
		m = 0
		for n, stage in enumerate(prm.stages):
			if not stage.split and not stage.dop:
				m = n
				break
		# NB: assumes labels of first stage have not been collapsed
		# handle POS labels without match like -LRB- or :
		pos = [(a, REMOVESTATESPLITS.match(a))
				for a in prm.stages[m].grammar.getpos()]
		labels = [match for match in (REMOVESTATESPLITS.match(a)
				for a in prm.stages[m].grammar.getlabels())
				if match]
		self.poslabels = {match.group(2) if match else a for a, match in pos}
		self.phrasallabels = {match.group(2) for match in labels
					} - self.poslabels
		self.functiontags = {match.group(3)[1:]
				for match in labels if match.group(3)} | {
				match.group(3)[1:] for a, match in pos
				if match and match.group(3)}
		self.morphtags = {match.group(4)[1:]
				for a, match in pos if match and match.group(4)}
		for stage in prm.stages:
			if stage.dop == 'doubledop':
				# map of fragments to line numbers
				stage.fragments = {line[:line.rindex('\t')]: n
						for n, line in enumerate(gzip.open(
							'%s/%s.fragments.gz' % (prm.resultdir,
							stage.name), 'rt', encoding='utf8'))}
		self.cnt = 0

	def parse(self, sent, tags=None, root=None, goldtree=None,
			require=(), block=()):
		"""Parse a sentence and perform postprocessing.

		Yields a dictionary from parse trees to probabilities for each stage.

		:param sent: a sequence of tokens.
		:param tags: optionally, a list of POS tags as strings to be given
			to the parser instead of trying all possible tags.
		:param root: optionally, specify a non-default root label.
		:param goldtree: if given, will be used to evaluate pruned parse
			forests.
		:param require: optionally, a list of tuples ``(label, indices)``; only
			parse trees containing these labeled spans will be returned.
			For example, ``('NP', [0, 1, 2])``.
		:param block: optionally, a list of tuples ``(label, indices)``;
			these labeled spans will be pruned."""
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
		partialparse = False
		# parse with each coarse-to-fine stage
		for n, stage in enumerate(self.stages):
			begin = process_time()
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
				stage.grammar.switch(model, logprob=True)

			# do parsing; if CTF pruning enabled, require parent stage to
			# be successful.
			splitprune = False
			if sent and (not stage.prune or charts[stage.prune]):
				prevn = 0
				if stage.prune:
					prevn = [a.name for a in self.stages].index(stage.prune)
					if not stage.split and self.stages[prevn].split:
						splitprune = True
				tree = goldtree
				if goldtree is not None and self.stages[prevn].split:
					tree = treetransforms.splitdiscnodes(
							goldtree.copy(True), self.stages[prevn].markorigin)
				if n > 0 and stage.prune and stage.mode not in (
						'dop-rerank', 'mc-rerank'):
					beginprune = process_time()
					whitelist, msg1 = prunechart(
							charts[stage.prune], stage.grammar, stage.k,
							splitprune, self.stages[prevn].markorigin,
							stage.mode.startswith('pcfg'),
							set(require or ()), set(block or ()))
					msg += '%s; %gs\n\t' % (msg1, process_time() - beginprune)
				else:
					whitelist = None
				if not sent:
					pass
				elif stage.mode == 'pcfg':
					chart, msg1 = pcfg.parse(
							sent, stage.grammar, tags=tags, start=root,
							whitelist=whitelist if stage.prune else None,
							beam_beta=-log(stage.beam_beta),
							beam_delta=stage.beam_delta,
							itemsestimate=estimateitems(
								sent, stage.prune, stage.mode, stage.dop),
							postagging=self.postagging)
				elif stage.mode == 'plcfrs':
					chart, msg1 = plcfrs.parse(
							sent, stage.grammar, tags=tags, start=root,
							exhaustive=stage.dop or (
								n + 1 != len(self.stages)
								and self.stages[n + 1].prune),
							whitelist=whitelist,
							splitprune=splitprune,
							markorigin=self.stages[prevn].markorigin,
							estimates=(stage.estimates, stage.outside)
								if stage.estimates in ('SX', 'SXlrgaps')
								else None,
							beam_beta=-log(stage.beam_beta),
							beam_delta=stage.beam_delta,
							itemsestimate=estimateitems(
								sent, stage.prune, stage.mode, stage.dop),
							postagging=self.postagging)
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
					raise ValueError('unknown mode specified: %s' % stage.mode)
				if n > 0 and stage.prune and stage.mode not in (
						'dop-rerank', 'mc-rerank') and goldtree is not None:
					# count number of gold bracketings in pruned chart.
					for node in tree.subtrees():
						# test whether node is part of *whitelist*
						if chart.itemid(node.label, node.leaves(), whitelist):
							fanout = re.search('_([0-9]+)$',
									node.label)
							golditems += (int(fanout.group(1))
									if fanout and not stage.split
									else 1)
					msg1 += (';\n\t%d/%d gold items remain after '
							'pruning' % (golditems, totalgolditems))
				msg += '%s\n\t' % msg1
				if (n > 0 and stage.prune and not chart and not noparse
						and stage.split == self.stages[prevn].split):
					logging.error('ERROR: expected successful parse;\n'
							'sent: %s\nstage %d: %s',
							' '.join(sent), n, stage.name)
					# raise ValueError('ERROR: expected successful parse. '
					# 		'sent %s, %s.' % (nsent, stage.name))
			numitems = chart.numitems() if hasattr(chart, 'numitems') else 0

			if self.verbosity >= 3 and chart:
				print('sent: %s\nstage: %s' % (' '.join(sent), stage.name))
			if self.verbosity >= 4:
				print('chart:\n%s' % chart)
			# do disambiguation of resulting parse forest
			if (sent and chart and stage.mode not in ('dop-rerank', 'mc-rerank')
					and not (self.relationalrealizational and stage.split)):
				begindisamb = process_time()
				disambiguation.getderivations(
						chart, stage.m,
						derivstrings=stage.dop not in ('doubledop', 'dop1')
								or stage.objective == 'mcp'
								or self.verbosity >= 3)
				if self.verbosity >= 3:
					print('%d-best derivations:\n%s' % (
						min(stage.m, 100),
						'\n'.join('%d. %s %s' % (n + 1,
							('subtrees=%d' % abs(int(prob / log(0.5))))
							if stage.objective == 'shortest'
							else ('p=%g' % exp(-prob)), deriv)
						for n, (deriv, prob) in enumerate(
							chart.derivations[:100]))))
					print('sum of probabilities: %g\n' % sum(exp(-prob)
							for _, prob in chart.derivations[:100]))
				if stage.objective == 'shortest':
					stage.grammar.switch('default'
							if stage.estimator == 'rfe'
							else stage.estimator, True)
				parsetrees, msg1 = disambiguation.marginalize(
						stage.objective if stage.dop else 'mpd',
						chart, sent=sent, tags=tags,
						k=stage.m, sldop_n=stage.sldop_n,
						mcplambda=stage.mcplambda,
						mcplabels=stage.mcplabels,
						ostag=stage.dop == 'ostag',
						require=set(require or ()),
						block=set(block or ()))
				msg += 'disambiguation: %s, %gs\n\t' % (
						msg1, process_time() - begindisamb)
				if self.verbosity >= 3:
					besttrees = nlargest(
							100, parsetrees, key=itemgetter(1))
					print('100-best parse trees:\n%s' % '\n'.join(
							'%d. %s %s' % (n + 1, probstr(prob), treestr)
							for n, (treestr, prob, _)
							in enumerate(besttrees)))
					print('sum of probabilities: %g\n' %
							sum((prob[1]
								if isinstance(prob, tuple) else prob)
								for _, prob, _ in besttrees))
				if not stage.prune and tree is not None:
					totalgolditems = sum(1 for node in tree.subtrees())
					golditems = sum(
							1 for node in tree.subtrees()
							if chart.itemid(node.label, node.leaves()))
					msg += ('%d/%d gold items in derivations\n\t' % (
							golditems, totalgolditems))
			elif (sent and not chart
					and stage.mode not in ('dop-rerank', 'mc-rerank')
					and not (self.relationalrealizational and stage.split)
					and not partialparse):  # sentence could not be parsed
				partition = partitionincompletechart(chart, 0, len(sent))
				msg = '%spartition: %s\n' % (
						msg.rstrip('\t'), repr(partition))
				tmp = []
				prob = 1
				for label, a, b in partition:
					if label == 'NOPARSE':
						tmp.append('(NN %d)' % a)
					else:
						parts = list(self.parse(
								sent[a:b],
								tags=tags[a:b] if tags else None,
								root=label))
						part = parts[-1]
						parttree, partprob, _partfrags = max(
								part.parsetrees, key=itemgetter(1))
						prob *= partprob
						parttree = ParentedTree(parttree)
						for node in parttree.subtrees(
								lambda n: isinstance(n[0], int)):
							node[0] += a
						tmp.append(str(parttree))
						if self.verbosity >= 3:
							print('part:', parttree)
						# msg += 'part %s %g\n%s' % ((label, a, b), partprob,
						# 		''.join(part.msg for part in parts))
						msg += 'part %s %g\n%s' % ((label, a, b), partprob,
								part.msg)
				tree = ParentedTree('(ROOT %s)' % ' '.join(tmp))
				from .runexp import dobinarization
				tree = dobinarization([tree], [sent],
						self.prm.binarization,
						self.prm.relationalrealizational, logmsg=False)[0]
				parsetrees = [(str(tree), prob, None)]
				partialparse = True
			if stage.name in (stage.prune for stage in self.stages):
				charts[stage.name] = chart
				prevparsetrees[stage.name] = parsetrees

			# postprocess, yield result
			if parsetrees:
				resultstr = ''
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
					logging.error(
							"something's amiss. %s\n%s", resultstr,
							''.join(traceback.format_exception(
								*sys.exc_info())))
					parsetree, prob, noparse = self.noparse(
							stage, xsent, tags, lastsuccessfulparse, n)
				else:
					lastsuccessfulparse = resultstr
				msg += probstr(prob) + ' '
			else:
				fragments = None
				parsetree, prob, noparse = self.noparse(
						stage, xsent, tags, lastsuccessfulparse, n)
				parsetrees = [(lastsuccessfulparse or str(parsetree),
						prob, None)]
			elapsedtime = process_time() - begin
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
		# when possible, infer head from binarization
		if (self.binarization and self.binarization.headrules
				and self.binarization.rightmostunary):
			saveheads(parsetree, self.binarization.tailmarker)
		treetransforms.unbinarize(parsetree, expandunary=False)
		treetransforms.removefanoutmarkers(parsetree)
		if self.relationalrealizational:
			parsetree = treebanktransforms.rrbacktransform(parsetree,
					self.relationalrealizational['adjunctionlabel'])
		if self.funcclassifier is not None:
			applyfunctionclassifier(self.funcclassifier, parsetree, sent)
		if self.transformations:
			treebanktransforms.reversetransform(
					parsetree, sent, self.transformations)
		else:
			treetransforms.canonicalize(parsetree)
		if self.headrules:
			applyheadrules(parsetree, self.headrules)
		return parsetree, False

	def noparse(self, stage, sent, tags, lastsuccessfulparse, n):
		"""Return parse from previous stage or a dummy parse."""
		# use successful parse from earlier stage if available
		if lastsuccessfulparse is not None:
			parsetree, _ = self.postprocess(lastsuccessfulparse, sent, n)
		else:  # Produce a dummy parse for evaluation purposes.
			default = grammar.defaultparse([(n, t) for n, t
					in enumerate(tags or (len(sent) * ['NN']))])
			parsetree = ParentedTree(
					'(%s %s)' % (stage.grammar.start, default))
			applyheadrules(parsetree, self.headrules)
		noparse = True
		prob = 1.0
		return parsetree, prob, noparse

	def augmentgrammar(self, newtrees, newsents):
		"""Extract grammar rules from trees and merge with current grammar."""
		from .runexp import dobinarization
		from . import _fragments
		if not newtrees:
			return
		prm = self.prm
		newtrees = [a.copy(True) for a in newtrees]  # will modify in-place
		for tree, sent in zip(newtrees, newsents):
			treebanktransforms.transform(tree, sent, prm.transformations)
		# FIXME: re-estimate unknown word model?
		if self.postagging and self.postagging.method == 'unknownword':
			for sent in newsents:  # add new known words
				self.postagging.lexicon.update(sent)
		newtrees = dobinarization(newtrees, newsents, prm.binarization,
				prm.relationalrealizational)
		orignumtrees = self.ctrees.len
		# FIXME: by adding trees at this point, cannot support 2dop+splitdisc
		self.ctrees.addtrees(list(zip(newtrees, newsents)), self.vocab)
		for n, stage in enumerate(prm.stages):
			prevn = 0
			backtransform = None
			traintrees = newtrees
			if n and stage.prune:
				prevn = [a.name for a in prm.stages].index(stage.prune)
			if stage.split:
				traintrees = [binarize(splitdiscnodes(
							tree.copy(True),
							stage.markorigin),
						childchar=':', dot=True, ids=UniqueIDs())
						for tree in traintrees]
			if stage.dop:
				if stage.dop == 'doubledop':
					if self.ctrees is None:
						raise ValueError('original treebank required')
					# get fragments
					fragments = _fragments.extractfragments(
							self.ctrees, orignumtrees, 0, self.vocab,
							self.ctrees, approx=False, disc=True)
					# for fragments already part of the grammar, increment
					# existing rules with occurrences in the new trees.
					fragmentkeysold = [a for a in fragments
							if a in stage.fragments]
					bitsetsold = [fragments[a] for a in fragmentkeysold]
					counts0 = _fragments.exactcountsslice(
							bitsetsold, self.ctrees, self.ctrees, indices=1,
							start=orignumtrees)
					for a, b in zip(fragmentkeysold, counts0):
						stage.grammar.incrementrulecount(
								stage.fragments[a], len(b))
					# for the subset of new fragments, create new grammar rules
					fragmentkeysnew = [a for a in fragments
							if a not in stage.fragments]
					bitsetsnew = [fragments[a] for a in fragmentkeysnew]
					counts1 = _fragments.exactcounts(
							bitsetsnew, self.ctrees, self.ctrees, indices=1)
					combined = dict(zip(fragmentkeysnew, counts1))
					# merge cover fragments
					if stage.maxdepth:
						cover = _fragments.allfragments(
								self.ctrees, self.vocab, stage.maxdepth,
								maxfrontier=stage.maxfrontier, disc=True,
								indices=True, start=orignumtrees)
						for a, b in cover.items():
							if a not in stage.fragments:
								combined[a] = b
							elif a not in fragments:
								stage.grammar.incrementrulecount(
										stage.fragments[a], len(b))
					# it is crucial that auxiliary binarization symbols are
					# unique, so include a sequence number
					ids = grammar.UniqueIDs(prefix='%d_' % self.cnt)
					self.cnt += 1
					# get grammar
					(xgrammar, backtransform, _, newfragments
								) = grammar.dopgrammar(
								traintrees, combined, ids=ids)
				else:
					raise NotImplementedError
			else:
				xgrammar = grammar.treebankgrammar(
						traintrees, newsents, extrarules=None)
			rules, lex = grammar.writegrammar(
					xgrammar, bitpar=stage.grammar.bitpar)
			orignumlabels = stage.grammar.nonterminals
			orignumrules = stage.grammar.numrules
			stage.grammar.addrules(
					rules.encode('utf8'), lex.encode('utf8'), backtransform)
			if stage.dop:
				if stage.dop in ('doubledop', 'dop1'):
					stage.fragments.update((a, orignumrules + n)
							for n, (a, _) in enumerate(newfragments))
					# recoverfragments() relies on this mapping to identify
					# binarization nodes. treeparsing() relies on this as well.
					stage.grammar.getmapping(
							None, neverblockre=re.compile('.+}<'),
							startidx=orignumlabels)
					if n and stage.prune:
						stage.grammar.getmapping(
								prm.stages[prevn].grammar,
								striplabelre=None if prm.stages[prevn].dop
									else re.compile('@.+$'),
								neverblockre=re.compile('.+}<'),
								splitprune=not stage.split
									and prm.stages[prevn].split,
								markorigin=prm.stages[prevn].markorigin,
								mapping=stage.mapping,
								startidx=orignumlabels)
				else:
					raise NotImplementedError
			else:
				if n and stage.prune:
					stage.grammar.getmapping(prm.stages[prevn].grammar,
						striplabelre=None,
						neverblockre=re.compile(stage.neverblockre)
							if stage.neverblockre else None,
						splitprune=not stage.split and prm.stages[prevn].split,
						markorigin=prm.stages[prevn].markorigin,
						mapping=stage.mapping,
						startidx=orignumlabels)


def readgrammars(resultdir, stages, postagging=None,
		transformations=None, top='ROOT', cache=False):
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
	if postagging and postagging.method == 'unknownword':
		if os.path.exists(resultdir + '/closedclasswords.txt'):
			with open(resultdir + '/closedclasswords.txt') as inp:
				postagging.closedclasswords = set(inp.read().splitlines())
		else:
			postagging.closedclasswords = None
	for n, stage in enumerate(stages):
		logging.info('reading: %s', stage.name)
		backtransform = outside = None
		prevn = 0
		if stage.mode != 'mc-rerank':
			rules = '%s/%s.rules.gz' % (resultdir, stage.name)
			lexicon = '%s/%s.lex.gz' % (resultdir, stage.name)
			probsfile = '%s/%s.probs.npz' % (resultdir, stage.name)
			if not os.path.exists(probsfile):
				probsfile = None
			if stage.dop in ('doubledop', 'dop1'):
				backtransform = openread('%s/%s.backtransform.gz' % (
						resultdir, stage.name)).read().splitlines()
			if cache and os.path.exists('%s/%s.g' % (resultdir, stage.name)):
				gram = Grammar.frombinfile('%s/%s.g' % (resultdir, stage.name),
						rules, lexicon, backtransform=backtransform)
			else:
				gram = Grammar(rules, lexicon, start=top, altweights=probsfile,
						backtransform=backtransform)
				if cache:
					gram.tobinfile('%s/%s.g' % (resultdir, stage.name))
		if n and stage.prune:
			prevn = [a.name for a in stages].index(stage.prune)
		if stage.mode == 'mc-rerank':
			gram = pickle.loads(gzip.open('%s/%s.train.pickle.gz' % (
					resultdir, stage.name), 'rb').read())
		elif stage.dop:
			if stage.estimates is not None:
				raise ValueError('not supported')
			if stage.dop in ('doubledop', 'dop1'):
				# recoverfragments() relies on this mapping to identify
				# binarization nodes. treeparsing() relies on this as well.
				_ = gram.getmapping(
						None, neverblockre=re.compile('.+}<'), debug=False)
				if n and stage.prune:
					_ = gram.getmapping(stages[prevn].grammar,
						striplabelre=re.compile('@.+$'),
						neverblockre=re.compile('^#[0-9]+|.+}<'),
						splitprune=not stage.split and stages[prevn].split,
						markorigin=stages[prevn].markorigin,
						mapping=stage.mapping, debug=False)
			else:  # dop reduction
				if n and stage.prune:
					_ = gram.getmapping(stages[prevn].grammar,
						striplabelre=re.compile(r'@[-0-9]+(?:\$\[.*\])?$'),
						neverblockre=re.compile(stage.neverblockre)
							if stage.neverblockre else None,
						splitprune=not stage.split and stages[prevn].split,
						markorigin=stages[prevn].markorigin,
						mapping=stage.mapping, debug=False)
					if stage.mode == 'dop-rerank':
						gram.getrulemapping(stages[prevn].grammar,
								re.compile(r'@[-0-9]+\b'))
				if stage.objective == 'sl-dop':  # needed for treeparsing()
					_ = gram.getmapping(
							None, striplabelre=re.compile(r'@[-0-9]+\b'),
							debug=False)
					# only need rulemapping for dop reduction,
					# defaults to 1-1 mapping otherwise.
					gram.getrulemapping(gram, re.compile(r'@[-0-9]+\b'))
		else:  # not stage.dop
			if n and stage.prune:
				_ = gram.getmapping(stages[prevn].grammar,
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=not stage.split and stages[prevn].split,
					markorigin=stages[prevn].markorigin,
					mapping=stage.mapping, debug=False)
			if stage.estimates in ('SX', 'SXlrgaps'):
				if stage.estimates == 'SX' and gram.maxfanout != 1:
					raise ValueError('SX estimate requires PCFG.')
				if stage.mode != 'plcfrs':
					raise ValueError('estimates require parser w/agenda.')
				outside = np.load('%s/%s.outside.npz' % (
						resultdir, stage.name))['outside']
				logging.info('loaded %s estimates', stage.estimates)
			elif stage.estimates:
				raise ValueError('unrecognized value; specify SX or SXlrgaps.')

		if stage.mode != 'mc-rerank':
			_sumsto1, msg = gram.testgrammar()
		logging.info('%s: %s', stage.name, msg)
		stage.update(grammar=gram, outside=outside)
	if postagging and postagging.method == 'unknownword':
		postagging.unknownwordfun = UNKNOWNWORDFUNC[postagging.model]
		postagging.lexicon = {w for w in stages[0].grammar.getwords()
				if not w.startswith(UNK)}
		postagging.sigs = {w for w in stages[0].grammar.getwords()
				if w.startswith(UNK)}
	if transformations and 'ftbundocompounds' in transformations:
		treebanktransforms.getftbcompounds(
				None, None, resultdir + '/compounds.txt')


def probstr(prob):
	"""Render probability / number of subtrees as string."""
	if isinstance(prob, tuple):
		return 'subtrees=%d, p=%.4g ' % (abs(prob[0]), prob[1])
	return 'p=%.4g' % prob


def estimateitems(sent, prune, mode, dop):
	"""Estimate number of chart items needed for a given sentence.

	The result is used to pre-allocate the chart; an over- or underestimate
	will only affect memory allocation efficiency. These constants were Based
	on a regression with Tiger parsing experiments."""
	beta = 600
	if prune:
		if dop:
			beta = 10 if mode == 'pcfg' else 20
		else:
			beta = 2
	return beta * len(sent) ** 2


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
		for key in ('sample', 'iterate', 'complement'):
			if stage.get(key):
				raise ValueError('option %r no longer supported' % key)
		if not stage.get('binarized', True):
			raise ValueError('option \'binarized\' no longer supported')
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
				'plcfrs', 'pcfg', 'dop-rerank', 'mc-rerank'):
			raise ValueError('unrecognized mode argument: %r.' % stage.mode)
		if n == 0 and stage.prune:
			raise ValueError('need previous stage to prune, '
					'but this stage is first.')
		if stage.prune is True:  # for backwards compatibility
			stage.prune = params['stages'][n - 1].name
		if stage.mode == 'dop-rerank':
			assert stage.prune and stage.k > 1
			# and stage.split == params['stages'][prevn].split)
			assert (stage.dop and stage.dop not in ('doubledop', 'dop1')
					and stage.objective == 'mpp')
		if stage.dop:
			assert stage.estimator in ('rfe', 'ewe', 'bon')
			assert stage.objective in ('mpp', 'mpd', 'mcp', 'shortest',
					'sl-dop', 'sl-dop-simple')
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
	if (params['binarization'].headrules  # ensure absolute path
			and os.path.split(params['binarization'].headrules)[0] == ''):
		params['binarization'].headrules = os.path.join(
				os.path.dirname(filename), params['binarization'].headrules)
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
	PARAMS.update(parser=parser, printprob=printprob,
			usetags=usetags, numparses=numparses, fmt=fmt,
			morphology=morphology)


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
	begin = process_time()
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
	sec = process_time() - begin
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
	flags = 'help prob tags sentid simple'.split()
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
	for n, filename in enumerate(args[:3 if '--simple' in opts else 2]):
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
		rules, lexicon = args[0], args[1]
		backtransform = None
		if opts.get('--bt'):
			backtransform = openread(opts.get('--bt')).read().splitlines()
		gram = Grammar(rules, lexicon, start=top, backtransform=backtransform)
		mode = 'pcfg' if gram.maxfanout == 1 else 'plcfrs'
		stages = []
		stage = DEFAULTSTAGE.copy()
		stage.update(
				name='grammar',
				mode=mode,
				grammar=gram,
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
				neverblockre=re.compile('.+}<'), debug=False)
		prm = DictObj(stages=stages, verbosity=int(opts.get('--verbosity', 2)),
				transformations=None, binarization=None, postagging=None,
				relationalrealizational=None)
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
				params.transformations, top=getattr(params, 'top', top))
		params.update(verbosity=int(opts.get('--verbosity', params.verbosity)))
		parser = Parser(params)
		morph = params.morphology
		del args[:1]
	with openread(args[0] if len(args) >= 1 else '-') as infile:
		with io.open(args[1] if len(args) == 2 and args[1] != '-'
				else sys.stdout.fileno(), 'w', encoding='utf8') as out:
			doparsing(parser, infile, out, prob, oneline, tags, numparses,
					int(opts.get('--numproc', 1)),
					opts.get('--fmt', 'discbracket'), morph, sentid)


__all__ = ['DictObj', 'Parser', 'doparsing', 'initworker', 'probstr',
		'readgrammars', 'readinputbitparstyle', 'readparam']
