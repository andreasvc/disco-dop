"""Parser object that performs coarse-to-fine and postprocessing.

Additionally, a simple command line interface similar to bitpar."""
from __future__ import division, print_function, absolute_import, \
		unicode_literals
import io
import os
import re
import sys
import json
import time
import codecs
import logging
import tempfile
import traceback
import string  # pylint: disable=W0402
import multiprocessing
if sys.version[0] == '2':
	from itertools import imap as map  # pylint: disable=E0611,W0622
from math import exp, log
from heapq import nlargest
from getopt import gnu_getopt, GetoptError
from operator import itemgetter
from functools import wraps
import numpy as np
from discodop import plcfrs, pcfg
from discodop.grammar import defaultparse
from discodop.containers import Grammar, BITPARRE
from discodop.coarsetofine import prunechart, whitelistfromposteriors
from discodop.disambiguation import getderivations, marginalize, doprerank
from discodop.tree import ParentedTree
from discodop.eval import alignsent
from discodop.lexicon import replaceraretestwords, UNKNOWNWORDFUNC, UNK
from discodop.treebank import WRITERS, writetree, openread
from discodop.treebanktransforms import reversetransform, rrbacktransform
from discodop.heads import saveheads, readheadrules
from discodop.punctuation import punctprune
from discodop.functiontags import applyfunctionclassifier
from discodop.treetransforms import mergediscnodes, unbinarize, \
		removefanoutmarkers, canonicalize

USAGE = '''
usage: %(cmd)s [options] <grammar/> [input [output]]
or:    %(cmd)s --simple [options] <rules> <lexicon> [input [output]]

'grammar/' is a directory with a model produced by "discodop runexp".
When no filename is given, input is read from standard input and the results
are written to standard output. Input should contain one sentence per line
with space-delimited tokens. Output consists of bracketed trees in
selected format. Files must be encoded in UTF-8.

General options:
  -x             Input is one token per line, sentences separated by two
                 newlines (like bitpar).
  -b k           Return the k-best parses instead of just 1.
  --prob         Print probabilities as well as parse trees.
  --tags         Tokens are of the form "word/POS"; give both to parser.
  --fmt=(export|bracket|discbracket|alpino|conll|mst|wordpos)
                 Format of output [default: discbracket].
  --numproc=k    Launch k processes, to exploit multiple cores.
  --simple       Parse with a single grammar and input file; similar interface
                 to bitpar. The files 'rules' and 'lexicon' define a binarized
                 grammar in bitpar or PLCFRS format.
  --verbosity=x  0 <= x <= 4. Same effect as verbosity in parameter file.

Options for simple mode:
  -s x           Use "x" as start symbol instead of default "TOP".
  --bt=file      Apply backtransform table to recover TSG derivations.
  --obj=(mpd|mpp|mrp|mcc|shortest|sl-dop)
                 Objective function to maximize [default: mpd].
  -m x           Use x derivations to approximate objective functions;
                 mpd and shortest require only 1.
  --bitpar       Use bitpar to parse with an unbinarized grammar.
''' % dict(cmd=sys.argv[0], fmt=','.join(WRITERS))

DEFAULTSTAGE = dict(
		name='stage1',  # identifier, used for filenames
		mode='plcfrs',  # use the agenda-based PLCFRS parser
		prune=False,  # whether to use previous chart to prune this stage
		split=False,  # split disc. nodes VP_2[101] as { VP*[100], VP*[001] }
		splitprune=False,  # treat VP_2[101] as {VP*[100], VP*[001]} for pruning
		markorigin=False,  # mark origin of split nodes: VP_2 => {VP*1, VP*2}
		collapselabels=None,  # options: None, 'head', 'all'. TODO: implement.
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

	def __init__(self, *a, **kw):
		self.__dict__.update(*a, **kw)

	def update(self, *a, **kw):
		"""Update/add more attributes."""
		self.__dict__.update(*a, **kw)

	def __getattr__(self, name):
		"""Dummy function for suppressing pylint E1101 errors."""
		raise AttributeError('%r instance has no attribute %r' % (
				self.__class__.__name__, name))

	def __repr__(self):
		return '%s(%s)' % (self.__class__.__name__,
			',\n\t'.join('%s=%r' % a for a in self.__dict__.items()))


PARAMS = DictObj()  # used for multiprocessing when using CLI of this module


def main():
	"""Handle command line arguments."""
	flags = 'help prob tags bitpar simple'.split()
	options = flags + 'obj= bt= numproc= fmt= verbosity='.split()
	try:
		opts, args = gnu_getopt(sys.argv[1:], 'hb:s:m:x', options)
	except GetoptError as err:
		print('error:', err, file=sys.stderr)
		print(USAGE)
		sys.exit(2)
	if '--help' in opts or '-h' in opts:
		print(USAGE)
		return
	if not 1 <= len(args) <= 4:
		print('error: incorrect number of arguments', file=sys.stderr)
		print(USAGE)
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
	if '--simple' in opts:
		if not 2 <= len(args) <= 4:
			print('error: incorrect number of arguments', file=sys.stderr)
			print(USAGE)
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
		grammar = Grammar(rules, lexicon, start=top,
				binarized='--bitpar' not in opts)
		stages = []
		stage = DEFAULTSTAGE.copy()
		backtransform = None
		if opts.get('--bt'):
			backtransform = openread(opts.get('--bt')).read().splitlines()
		stage.update(
				name='grammar',
				mode=mode,
				grammar=grammar,
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
		from discodop.runexp import readparam
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
	infile = openread(args[0] if len(args) >= 1 else sys.stdin.fileno())
	out = (io.open(args[1], 'w', encoding='utf8')
			if len(args) == 2 else sys.stdout)
	doparsing(parser, infile, out, prob, oneline, tags, numparses,
			int(opts.get('--numproc', 1)), opts.get('--fmt', 'discbracket'),
			morph)


def doparsing(parser, infile, out, printprob, oneline, usetags, numparses,
		numproc, fmt, morphology):
	"""Parse sentences from file and write results to file, log to stdout."""
	times = []
	unparsed = 0
	if not oneline:
		infile = readinputbitparstyle(infile)
	infile = (line for line in infile if line.strip())
	if numproc == 1:
		initworker(parser, printprob, usetags, numparses, fmt, morphology)
		mymap = map
	else:
		pool = multiprocessing.Pool(
				processes=numproc, initializer=initworker,
				initargs=(parser, printprob, usetags, numparses, fmt,
					morphology))
		mymap = pool.map
	for output, noparse, sec, msg in mymap(worker, enumerate(infile)):
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


def initworker(parser, printprob, usetags, numparses,
		fmt, morphology):
	"""Load parser for a worker process."""
	headrules = None
	if fmt in ('mst', 'conll'):
		headrules = readheadrules(parser.binarization.headrules)
	PARAMS.update(parser=parser, printprob=printprob,
			usetags=usetags, numparses=numparses, fmt=fmt,
			morphology=morphology, headrules=headrules)


def workerfunc(func):
	"""Wrap a multiprocessing worker function to produce a full traceback."""
	@wraps(func)
	def wrapper(*args, **kwds):
		"""Apply decorated function."""
		try:
			import faulthandler
			faulthandler.enable()  # Dump information on segfault.
		except (ImportError, io.UnsupportedOperation):
			pass
		# NB: only concurrent.futures on Python 3.3+ will exit gracefully.
		try:
			return func(*args, **kwds)
		except Exception:  # pylint: disable=W0703
			# Put traceback as string into an exception and raise that
			raise Exception('in worker process\n%s' %
					''.join(traceback.format_exception(*sys.exc_info())))
	return wrapper


@workerfunc
def worker(args):
	"""Parse a single sentence."""
	n, line = args
	line = line.strip()
	if not line:
		return '', True, 0, ''
	begin = time.clock()
	sent = line.split(' ')
	tags = None
	if PARAMS.usetags:
		sent, tags = zip(*(a.rsplit('/', 1) for a in sent))
	msg = 'parsing %d: %s' % (n, ' '.join(sent))
	result = list(PARAMS.parser.parse(sent, tags=tags))[-1]
	output = ''
	if result.noparse:
		msg += '\nNo parse for "%s"' % ' '.join(sent)
		if PARAMS.printprob:
			output += 'prob=%.16g\n' % result.prob
		output += '%s\t%s\n' % (result.parsetree, ' '.join(sent))
	else:
		output += ''.join(
				writetree(
					PARAMS.parser.postprocess(tree, sent, -1)[0], sent,
					n if PARAMS.numparses == 1 else ('%d-%d' % (n, k)),
					PARAMS.fmt, morphology=PARAMS.morphology,
					comment=('prob=%.16g' % prob) if PARAMS.printprob else None)
				for k, (tree, prob, _) in enumerate(nlargest(
					PARAMS.numparses, result.parsetrees, key=itemgetter(1))))
	sec = time.clock() - begin
	msg += '\n%g s' % sec
	return output, result.noparse, sec, msg


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


class Parser(object):
	"""A coarse-to-fine parser based on a given set of parameters.

	:param prm: A DictObj with parameters as returned by
		:py:func:`runexp.readparam()`.
	:param funcclassifier: optionally, a function tag classifier trained by
		:py:func:`functiontags.trainfunctionclassifier`.
	"""
	def __init__(self, prm, funcclassifier=None):
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
				if (stage.estimator == 'ewe'
						or stage.objective.startswith('sl-dop')):
					model = 'ewe'
				elif stage.estimator == 'bon':
					model = 'bon'
				if stage.objective == 'shortest':
					model = 'shortest'
			stage.grammar.switch(model, logprob=stage.mode != 'pcfg-posterior')
			if prm.verbosity >= 3:
				logging.debug(stage.name)
				logging.debug(stage.grammar)

	def parse(self, sent, tags=None):
		"""Parse a sentence and perform postprocessing.

		Yields a dictionary from parse trees to probabilities for each stage.

		:param sent: a sequence of tokens.
		:param tags: if given, will be given to the parser instead of trying
			all possible tags."""
		if self.transformations and 'PUNCT-PRUNE' in self.transformations:
			origsent = sent[:]
			punctprune(None, sent)
			if tags:
				newtags = alignsent(sent, origsent, dict(enumerate(tags)))
				tags = [newtags[n] for n, _ in enumerate(sent)]
		if self.postagging and self.postagging.method == 'unknownword':
			sent = replaceraretestwords(sent,
					self.postagging.unknownwordfun,
					self.postagging.lexicon, self.postagging.sigs)
		sent = list(sent)
		if tags is not None:
			tags = list(tags)

		# parse with each coarse-to-fine stage
		chart = start = inside = outside = lastsuccessfulparse = None
		for n, stage in enumerate(self.stages):
			begin = time.clock()
			noparse = False
			parsetrees = fragments = None
			msg = '%s:\t' % stage.name.upper()
			model = 'default'
			if stage.dop:
				if (stage.estimator == 'ewe'
						or stage.objective.startswith('sl-dop')):
					model = 'ewe'
				elif stage.estimator == 'bon':
					model = 'bon'
				if stage.objective == 'shortest':
					model = 'shortest'
			x = stage.grammar.currentmodel
			stage.grammar.switch(model, logprob=stage.mode != 'pcfg-posterior')
			if stage.mode.startswith('pcfg-bitpar'):
				if (not hasattr(stage, 'rulesfile')
						or x != stage.grammar.currentmodel):
					exportbitpargrammar(stage)
			elif hasattr(stage, 'rulesfile'):
				del stage.rulesfile, stage.lexiconfile
			if not stage.binarized and not stage.mode.startswith('pcfg-bitpar'):
				raise ValueError('non-binarized grammar requires use of bitpar')

			# do parsing; if CTF pruning enabled, require previous stage to
			# be successful.
			if sent and (not stage.prune or chart):
				if n > 0 and stage.prune and stage.mode != 'dop-rerank':
					beginprune = time.clock()
					if self.stages[n - 1].mode == 'pcfg-posterior':
						whitelist, msg1 = whitelistfromposteriors(
								inside, outside, start,
								self.stages[n - 1].grammar, stage.grammar,
								stage.k, stage.splitprune,
								self.stages[n - 1].markorigin,
								stage.mode.startswith('pcfg'))
					else:
						whitelist, msg1 = prunechart(
								chart, stage.grammar, stage.k,
								stage.splitprune,
								self.stages[n - 1].markorigin,
								stage.mode.startswith('pcfg'),
								self.stages[n - 1].mode == 'pcfg-bitpar-nbest')
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
				elif stage.mode == 'pcfg-posterior':
					inside, outside, start, msg1 = pcfg.doinsideoutside(
							sent, stage.grammar, tags=tags)
					chart = start
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
							sent, numderivs,
							stage.grammar.start,
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
								and self.stages[n - 1].split,
							markorigin=self.stages[n - 1].markorigin,
							estimates=(stage.estimates, stage.outside)
								if stage.estimates in ('SX', 'SXlrgaps')
								else None,
							symbolic=False,
							beam_beta=-log(stage.beam_beta),
							beam_delta=stage.beam_delta)
				elif stage.mode == 'dop-rerank':
					if chart:
						parsetrees = doprerank(chart, sent, stage.k,
								self.stages[n - 1].grammar, stage.grammar)
						msg1 = 're-ranked %d parse trees. ' % len(parsetrees)
				else:
					raise ValueError('unknown mode specified.')
				msg += '%s\n\t' % msg1
				if (n > 0 and not chart and not noparse
						and stage.split == self.stages[n - 1].split):
					logging.error('ERROR: expected successful parse. '
							'sent: %s\nstage: %s.', ' '.join(sent), stage.name)
					# raise ValueError('ERROR: expected successful parse. '
					# 		'sent %s, %s.' % (nsent, stage.name))

			# do disambiguation of resulting parse forest
			if sent and chart and stage.mode not in (
					'pcfg-posterior', 'dop-rerank') and not (
					self.relationalrealizational and stage.split):
				begindisamb = time.clock()
				if stage.mode == 'pcfg-bitpar-nbest':
					if not stage.kbest or stage.sample:
						raise ValueError('sampling not possible with bitpar '
								'in nbest mode.')
					derivations = chart.rankededges[chart.root()]
					entries = [None] * len(derivations)
				else:
					derivations, entries = getderivations(chart, stage.m,
							kbest=stage.kbest, sample=stage.sample,
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
					stage.grammar.switch('ewe' if stage.estimator == 'ewe'
							else 'default', True)
				parsetrees, msg1 = marginalize(
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
			if self.verbosity >= 4:
				print('Chart:\n%s' % chart)

			# postprocess, yield result
			if parsetrees:
				try:
					resultstr, prob, fragments = max(
							parsetrees, key=itemgetter(1))
					parsetree, noparse = self.postprocess(resultstr, sent, n)
					if not all(a for a in parsetree.subtrees()):
						raise ValueError('empty nodes in tree: %s' % parsetree)
					if not len(parsetree.leaves()) == len(sent):
						raise ValueError('leaves missing. original tree: %s\n'
							'postprocessed: %r' % (resultstr, parsetree))
				except Exception:  # pylint: disable=W0703
					logging.error("something's amiss. %s", ''.join(
								traceback.format_exception(*sys.exc_info())))
					parsetree, prob, noparse = self.noparse(
							stage, sent, tags, lastsuccessfulparse)
				else:
					lastsuccessfulparse = parsetree
				msg += probstr(prob) + ' '
			else:
				fragments = None
				parsetree, prob, noparse = self.noparse(
						stage, sent, tags, lastsuccessfulparse)
			elapsedtime = time.clock() - begin
			msg += '%.2fs cpu time elapsed\n' % (elapsedtime)
			yield DictObj(name=stage.name, parsetree=parsetree, prob=prob,
					parsetrees=parsetrees, fragments=fragments,
					noparse=noparse, elapsedtime=elapsedtime, msg=msg)

	def postprocess(self, treestr, sent, stage):
		"""Take parse tree and apply postprocessing."""
		parsetree = ParentedTree(treestr)
		if self.stages[stage].split:
			mergediscnodes(unbinarize(parsetree, childchar=':',
					expandunary=False))
		if self.binarization.tailmarker or self.binarization.headrules:
			saveheads(parsetree, self.binarization.tailmarker)
		unbinarize(parsetree, expandunary=False)
		removefanoutmarkers(parsetree)
		if self.relationalrealizational:
			parsetree = rrbacktransform(parsetree,
					self.relationalrealizational['adjunctionlabel'])
		if self.funcclassifier is not None:
			applyfunctionclassifier(self.funcclassifier, parsetree, sent)
		if self.transformations:
			reversetransform(parsetree, self.transformations)
		else:
			canonicalize(parsetree)
		return parsetree, False

	def noparse(self, stage, sent, tags, lastsuccessfulparse):
		"""Return parse from previous stage or a dummy parse."""
		# use successful parse from earlier stage if available
		if lastsuccessfulparse is not None:
			parsetree = lastsuccessfulparse.copy(True)
		else:  # Produce a dummy parse for evaluation purposes.
			default = defaultparse([(n, t) for n, t
					in enumerate(tags or (len(sent) * ['NONE']))])
			parsetree = ParentedTree('(%s %s)' % (
					stage.grammar.start, default))
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
		rules = openread('%s/%s.rules.gz' % (resultdir, stage.name)).read()
		lexicon = openread('%s/%s.lex.gz' % (resultdir, stage.name))
		grammar = Grammar(rules, lexicon.read(),
				start=top, binarized=stage.binarized)
		backtransform = outside = None
		if stage.dop:
			if stage.estimates is not None:
				raise ValueError('not supported')
			if stage.dop in ('doubledop', 'dop1'):
				backtransform = openread('%s/%s.backtransform.gz' % (
						resultdir, stage.name)).read().splitlines()
				if n and stage.prune:
					_ = grammar.getmapping(stages[n - 1].grammar,
						striplabelre=re.compile('@.+$'),
						neverblockre=re.compile('^#[0-9]+|.+}<'),
						splitprune=stage.splitprune and stages[n - 1].split,
						markorigin=stages[n - 1].markorigin,
						mapping=stage.mapping)
				else:
					# recoverfragments() relies on this mapping to identify
					# binarization nodes
					_ = grammar.getmapping(None,
						neverblockre=re.compile('.+}<'))
			elif n and stage.prune:  # dop reduction
				_ = grammar.getmapping(stages[n - 1].grammar,
					striplabelre=re.compile('@[-0-9]+$'),
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[n - 1].split,
					markorigin=stages[n - 1].markorigin,
					mapping=stage.mapping)
				if stage.mode == 'dop-rerank':
					grammar.getrulemapping(
							stages[n - 1].grammar, re.compile(r'@[-0-9]+\b'))
			probsfile = '%s/%s.probs.npz' % (resultdir, stage.name)
			if os.path.exists(probsfile):
				probmodels = np.load(probsfile)
				for name in probmodels.files:
					if name != 'default':
						grammar.register(name, probmodels[name])
		else:  # not stage.dop
			if n and stage.prune:
				_ = grammar.getmapping(stages[n - 1].grammar,
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[n - 1].split,
					markorigin=stages[n - 1].markorigin,
					mapping=stage.mapping)
			if stage.estimates in ('SX', 'SXlrgaps'):
				if stage.estimates == 'SX' and grammar.maxfanout != 1:
					raise ValueError('SX estimate requires PCFG.')
				if stage.mode != 'plcfrs':
					raise ValueError('estimates require parser w/agenda.')
				outside = np.load('%s/%s.outside.npz' % (
						resultdir, stage.name))['outside']
				logging.info('loaded %s estimates', stage.estimates)
			elif stage.estimates:
				raise ValueError('unrecognized value; specify SX or SXlrgaps.')

		if stage.mode.startswith('pcfg-bitpar'):
			if grammar.maxfanout != 1:
				raise ValueError('bitpar requires a PCFG.')

		_sumsto1, msg = grammar.testgrammar()
		logging.info('%s: %s', stage.name, msg)
		stage.update(grammar=grammar, backtransform=backtransform,
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
	lexicon = stage.grammar.origlexicon.replace(
			'(', '-LRB-').replace(')', '-RRB-')
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


def which(program):
	"""Return first match for program in search path."""
	for path in os.environ.get('PATH', os.defpath).split(":"):
		if path and os.path.exists(os.path.join(path, program)):
			return os.path.join(path, program)
	raise ValueError('%r not found in path; please install it.' % program)


__all__ = ['DictObj', 'Parser', 'doparsing', 'exportbitpargrammar',
		'initworker', 'probstr', 'readgrammars', 'readinputbitparstyle',
		'which', 'worker', 'workerfunc']

if __name__ == '__main__':
	main()
