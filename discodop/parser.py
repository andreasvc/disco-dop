"""Parser object that performs coarse-to-fine and postprocessing.

Additionally, a simple command line interface similar to bitpar."""
from __future__ import print_function
import io
import os
import re
import sys
import time
import gzip
import codecs
import logging
import tempfile
import traceback
import string  # pylint: disable=W0402
import multiprocessing
if sys.version[0] > '2':
	imap = map
else:
	from itertools import imap
from math import exp, log
from heapq import nlargest
from getopt import gnu_getopt, GetoptError
from operator import itemgetter
from functools import wraps
import numpy as np
from discodop import plcfrs, pcfg
from discodop.grammar import defaultparse
from discodop.containers import Grammar
from discodop.coarsetofine import prunechart, whitelistfromposteriors
from discodop.disambiguation import getderivations, marginalize, doprerank
from discodop.tree import Tree
from discodop.lexicon import replaceraretestwords, UNKNOWNWORDFUNC, UNK
from discodop.treebank import WRITERS, writetree
from discodop.treebanktransforms import reversetransform, rrbacktransform, \
		saveheads, NUMBERRE, readheadrules
from discodop.treetransforms import mergediscnodes, unbinarize, \
		removefanoutmarkers

USAGE = '''
usage: %(cmd)s [options] <grammar/> [input [output]]
or:    %(cmd)s --simple [options] <rules> <lexicon> [input [output]]

'grammar/' is a directory with a model produced by "discodop runexp".
When no filename is given, input is read from standard input and the results
are written to standard output. Input should contain one sentence per line
with space-delimited tokens. Output consists of bracketed trees in
selected format. Files must be encoded in UTF-8.

General options:
  -x           Input is one token per line, sentences separated by two
               newlines (like bitpar).
  -b k         Return the k-best parses instead of just 1.
  --prob       Print probabilities as well as parse trees.
  --tags       Tokens are of the form "word/POS"; give both to parser.
  --fmt=[export|bracket|discbracket|alpino|conll|mst|wordpos]
               Format of output [default: discbracket].
  --numproc=k  Launch k processes, to exploit multiple cores.
  --simple     Parse with a single grammar and input file; similar interface
               to bitpar. The files 'rules' and 'lexicon' define a binarized
               grammar in bitpar or PLCFRS format.

Options for simple mode:
  -s x         Use "x" as start symbol instead of default "TOP".
  --bt=file    Apply backtransform table to recover TSG derivations.
  --mpp=k      By default, the output consists of derivations, with the most
               probable derivation (MPD) ranked highest. With a PTSG such as
               DOP, it is possible to aim for the most probable parse (MPP)
               instead, whose probability is the sum of any number of the
               k-best derivations.
  --bitpar     Use bitpar to parse with an unbinarized grammar.
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
		neverblockre=None,  # do not prune nodes with label that match regex
		getestimates=None,  # compute & store estimates
		useestimates=None,  # load & use estimates
		dop=False,  # enable DOP mode (DOP reduction / double DOP)
		packedgraph=False,  # use packed graph encoding for DOP reduction
		usedoubledop=False,  # when False, use DOP reduction instead
		binarized=True,  # for double dop, whether to binarize extracted grammar
			# (False requires use of bitpar)
		iterate=False,  # for double dop, whether to add fragments of fragments
		complement=False,  # for double dop, whether to include fragments which
			# form the complement of the maximal recurring fragments extracted
		sample=False, kbest=True,
		m=10,  # number of derivations to sample/enumerate
		estimator='rfe',  # choices: rfe, ewe
		objective='mpp',  # choices: mpp, mpd, shortest, sl-dop[-simple]
			# NB: w/shortest derivation, estimator only affects tie breaking.
		sldop_n=7,  # number of trees to consider when using sl-dop[-simple]
		mcc_labda=1.0,  # weight to assign to recall vs. mistake rate with mcc
		mcc_labels=None,  # optionally, set of labels to optimize for with mcc
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
	options = 'prob tags bitpar simple mpp= bt= numproc= fmt='.split()
	try:
		opts, args = gnu_getopt(sys.argv[1:], 'b:s:x', options)
		assert 1 <= len(args) <= 4, 'incorrect number of arguments'
	except (GetoptError, AssertionError) as err:
		print(err, USAGE)
		return
	for n, filename in enumerate(args):
		assert os.path.exists(filename), (
				'file %d not found: %r' % (n + 1, filename))
	opts = dict(opts)
	numparses = int(opts.get('-b', 1))
	top = opts.get('-s', 'TOP')
	prob = '--prob' in opts
	tags = '--tags' in opts
	oneline = '-x' not in opts
	if '--simple' in opts:
		assert 2 <= len(args) <= 4, 'incorrect number of arguments'
		rules = (gzip.open if args[0].endswith('.gz') else open)(args[0]).read()
		lexicon = codecs.getreader('utf-8')((gzip.open if args[1].endswith('.gz')
				else open)(args[1])).read()
		bitpar = rules[0] in string.digits
		if '--bitpar' in opts:
			assert bitpar, 'bitpar requires bitpar grammar format.'
			mode = 'pcfg-bitpar-nbest'
		else:
			mode = 'pcfg' if bitpar else 'plcfrs'
		grammar = Grammar(rules, lexicon, start=top, bitpar=bitpar,
				binarized='--bitpar' not in opts)
		stages = []
		stage = DEFAULTSTAGE.copy()
		backtransform = None
		if opts.get('--bt'):
			backtransform = (gzip.open if opts.get('--bt').endswith('.gz')
					else open)(opts.get('--bt')).read().splitlines()
		stage.update(
				name='grammar',
				mode=mode,
				grammar=grammar,
				binarized='--bitpar' not in opts,
				backtransform=backtransform if len(args) < 4 else None,
				m=numparses,
				objective='mpd')
		if '--mpp' in opts:
			stage.update(dop=True, objective='mpp', m=int(opts['--mpp']))
		stages.append(DictObj(stage))
		if backtransform:
			_ = stages[-1].grammar.getmapping(None,
				neverblockre=re.compile(b'.+}<'))
		parser = Parser(stages)
	else:
		from discodop.runexp import readparam
		directory = args[0]
		assert os.path.isdir(directory), (
				'expected directory producted by "discodop runexp".')
		params = readparam(os.path.join(directory, 'params.prm'))
		params['resultdir'] = directory
		stages = params['stages']
		postagging = params['postagging']
		readgrammars(directory, stages, postagging,
				top=params.get('top', top))
		parser = Parser(stages,
				transformations=params.get('transformations'),
				binarization=params['binarization'],
				postagging=postagging if postagging and
				postagging.method == 'unknownword' else None,
				relationalrealizational=params.get('relationalrealizational'),
				verbosity=params.get('verbosity', 2))
	infile = (io.open(args[2], encoding='utf-8')
			if len(args) >= 3 else sys.stdin)
	out = (io.open(args[3], 'w', encoding='utf-8')
			if len(args) == 4 else sys.stdout)
	doparsing(parser, infile, out, prob, oneline, tags, numparses,
			int(opts.get('--numproc', 1)), opts.get('--fmt', 'discbracket'),
			params['morphology'])


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
		mymap = imap
	else:
		pool = multiprocessing.Pool(processes=numproc, initializer=initworker,
				initargs=(parser, printprob, usetags, numparses, fmt,
					morphology))
		mymap = pool.imap
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
		import faulthandler
		faulthandler.enable()  # Dump information on segfault.
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
	lexicon = PARAMS.parser.stages[0].grammar.lexicalbyword
	assert PARAMS.usetags or not set(sent) - set(lexicon), (
			'unknown words and no tags or unknown word model supplied.\n'
			'sentence: %r\nunknown words:%r' % (
			sent, list(set(sent) - set(lexicon))))
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
					PARAMS.parser.postprocess(tree)[0], sent,
					n if PARAMS.numparses == 1 else ('%d-%d' % (n, k)),
					PARAMS.fmt, headrules=PARAMS.headrules,
					morphology=PARAMS.morphology,
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

	:param stages: a list of coarse-to-fine stages containing grammars and
		parameters.
	:param transformations: treebank transformations to reverse on parses.
	:param binarization: settings used for binarization; used for the
		tailmarker attribute which identifies heads in parser output.
	:param postagging: if given, an unknown word model is used to assign POS
		tags during parsing. The model consists of a DictObj with (at least)
		the following attributes:

		- unknownwordfun: function to produces signatures for unknown words.
		- lexicon: the set of known words in the grammar.
		- sigs: the set of word signatures occurring in the grammar.
	:param relationalrealizational: whether to reverse the RR-transform."""
	def __init__(self, stages, transformations=None, postagging=None,
			binarization=DictObj(tailmarker=None),
			relationalrealizational=None, verbosity=2):
		self.stages = stages
		self.transformations = transformations
		self.binarization = binarization
		self.postagging = postagging
		self.relationalrealizational = relationalrealizational
		self.verbosity = verbosity
		for stage in stages:
			if stage.mode.startswith('pcfg-bitpar'):
				exportbitpargrammar(stage)
			model = u'default'
			if stage.dop:
				if (stage.estimator == 'ewe'
						or stage.objective.startswith('sl-dop')):
					model = u'ewe'
				elif stage.estimator == 'bon':
					model = u'bon'
				if stage.objective == 'shortest':
					model = u'shortest'
			stage.grammar.switch(model, logprob=stage.mode != 'pcfg-posterior')
			if verbosity >= 3:
				logging.debug(stage.name)
				logging.debug(stage.grammar)

	def parse(self, sent, tags=None):
		"""Parse a sentence and perform postprocessing.

		Yields a dictionary from parse trees to probabilities for each stage.

		:param sent: a sequence of tokens.
		:param tags: if given, will be given to the parser instead of trying
			all possible tags."""
		if self.postagging:
			if self.transformations and 'FOLD-NUMBERS' in self.transformations:
				sent = ['000' if NUMBERRE.match(a) else a for a in sent]
			sent = replaceraretestwords(sent,
					self.postagging.unknownwordfun,
					self.postagging.lexicon, self.postagging.sigs)
		sent = list(sent)
		if tags is not None:
			tags = list(tags)
		chart = start = inside = outside = lastsuccessfulparse = None
		for n, stage in enumerate(self.stages):
			begin = time.clock()
			noparse = False
			parsetrees = fragments = None
			msg = '%s:\t' % stage.name.upper()
			model = u'default'
			if stage.dop:
				if (stage.estimator == 'ewe'
						or stage.objective.startswith('sl-dop')):
					model = u'ewe'
				elif stage.estimator == 'bon':
					model = u'bon'
				if stage.objective == 'shortest':
					model = u'shortest'
			x = stage.grammar.currentmodel
			stage.grammar.switch(model, logprob=stage.mode != 'pcfg-posterior')
			if stage.mode.startswith('pcfg-bitpar') and (
					not hasattr(stage, 'rulesfile')
					or x != stage.grammar.currentmodel):
				exportbitpargrammar(stage)
			assert stage.binarized or stage.mode.startswith('pcfg-bitpar'), (
					'non-binarized grammar requires use of bitpar')
			if not stage.prune or chart:
				if n != 0 and stage.prune and stage.mode != 'dop-rerank':
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
				if stage.mode == 'pcfg':
					chart, msg1 = pcfg.parse(
							sent, stage.grammar, tags=tags,
							whitelist=whitelist if stage.prune else None)
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
							exhaustive=stage.dop or (n + 1 != len(self.stages)
								and self.stages[n + 1].prune),
							whitelist=whitelist,
							splitprune=stage.splitprune
								and self.stages[n - 1].split,
							markorigin=self.stages[n - 1].markorigin,
							estimates=(stage.useestimates, stage.outside)
								if stage.useestimates in ('SX', 'SXlrgaps')
								else None)
				elif stage.mode == 'dop-rerank':
					if chart:
						parsetrees = doprerank(chart, sent, stage.k,
								self.stages[n - 1].grammar, stage.grammar)
						msg1 = 're-ranked %d parse trees. ' % len(parsetrees)
				else:
					raise ValueError('unknown mode specified.')
				msg += '%s\n\t' % msg1
				if (n != 0 and not chart and not noparse
						and stage.split == self.stages[n - 1].split):
					logging.error('ERROR: expected successful parse. '
							'sent: %s\nstage: %s.', ' '.join(sent), stage.name)
					#raise ValueError('ERROR: expected successful parse. '
					#		'sent %s, %s.' % (nsent, stage.name))
			if chart and stage.mode not in ('pcfg-posterior', 'dop-rerank'
					) and not (self.relationalrealizational and stage.split):
				begindisamb = time.clock()
				if stage.mode == 'pcfg-bitpar-nbest':
					assert stage.kbest and not stage.sample, (
							'sampling not possible with bitpar in nbest mode.')
					derivations = chart.rankededges[chart.root()]
					entries = [None] * len(derivations)
				else:
					derivations, entries = getderivations(chart, stage.m,
							kbest=stage.kbest, sample=stage.sample,
							derivstrings=not stage.usedoubledop
									or self.verbosity >= 3
									or stage.objective == 'mcc')
				if self.verbosity >= 3:
					print('sent: %s\nstage: %s' % (' '.join(sent), stage.name))
					print('100-best derivations:\n%s\n' % '\n'.join(
						'%d. %s %s' % (n + 1,
							('subtrees=%d' % abs(int(prob / log(0.5))))
							if stage.objective == 'shortest'
							else ('p=%g' % exp(-prob)), deriv)
						for n, (deriv, prob) in enumerate(derivations[:100])))
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
					print('100-best parse trees:\n%s\n' % '\n'.join(
							'%d. %s %s' % (n + 1, probstr(prob), treestr)
							for n, (treestr, prob, _) in enumerate(
								nlargest(100, parsetrees,
									key=itemgetter(1)))))
			if parsetrees:
				try:
					resultstr, prob, fragments = max(
							parsetrees, key=itemgetter(1))
					parsetree, noparse = self.postprocess(resultstr, n)
					assert all(a for a in parsetree.subtrees()), (
							'tree has empty nodes: %s' % parsetree)
					assert len(parsetree.leaves()) == len(sent), (
							'leaves missing. original tree: %s\n'
							'postprocessed: %r' % (resultstr, parsetree))
				except Exception as err:  # pylint: disable=W0703
					logging.error("something's amiss. %s", err)
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

	def postprocess(self, treestr, stage=-1):
		"""Take parse tree and apply postprocessing."""
		parsetree = Tree.parse(treestr, parse_leaf=int)
		if self.stages[stage].split:
			mergediscnodes(unbinarize(parsetree, childchar=':',
					expandunary=False))
		saveheads(parsetree, self.binarization.tailmarker)
		unbinarize(parsetree, expandunary=False)
		removefanoutmarkers(parsetree)
		if self.relationalrealizational:
			parsetree = rrbacktransform(parsetree,
					self.relationalrealizational['adjunctionlabel'])
		if self.transformations:
			reversetransform(parsetree, self.transformations)
		return parsetree, False

	def noparse(self, stage, sent, tags, lastsuccessfulparse):
		"""Return parse from previous stage or a dummy parse."""
		# use successful parse from earlier stage if available
		if lastsuccessfulparse is not None:
			parsetree = lastsuccessfulparse.copy(True)
		else:  # Produce a dummy parse for evaluation purposes.
			default = defaultparse([(n, t) for n, t
					in enumerate(tags or (len(sent) * ['NONE']))])
			parsetree = Tree.parse('(%s %s)' % (stage.grammar.start,
					default), parse_leaf=int)
		noparse = True
		prob = 1.0
		return parsetree, prob, noparse


def readgrammars(resultdir, stages, postagging=None, top='ROOT'):
	"""Read the grammars from a previous experiment.

	Expects a directory ``resultdir`` which contains the relevant grammars and
	the parameter file ``params.prm``, as produced by ``runexp``."""
	for n, stage in enumerate(stages):
		logging.info('reading: %s', stage.name)
		rules = gzip.open('%s/%s.rules.gz' % (resultdir, stage.name))
		lexicon = codecs.getreader('utf-8')(gzip.open('%s/%s.lex.gz' % (
				resultdir, stage.name)))
		grammar = Grammar(rules.read(), lexicon.read(),
				start=top, bitpar=stage.mode.startswith('pcfg'),
				binarized=stage.binarized)
		backtransform = None
		if stage.dop:
			assert stage.useestimates is None, 'not supported'
			if stage.usedoubledop:
				backtransform = gzip.open('%s/%s.backtransform.gz' % (
						resultdir, stage.name)).read().splitlines()
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
				if stage.mode == 'dop-rerank':
					grammar.getrulemapping(stages[n - 1].grammar)
			probsfile = '%s/%s.probs.npz' % (resultdir, stage.name)
			if os.path.exists(probsfile):
				probmodels = np.load(probsfile)  # pylint: disable=no-member
				for name in probmodels.files:
					if name != 'default':
						grammar.register(unicode(name), probmodels[name])
		else:  # not stage.dop
			if n and stage.prune:
				_ = grammar.getmapping(stages[n - 1].grammar,
					neverblockre=re.compile(stage.neverblockre)
						if stage.neverblockre else None,
					splitprune=stage.splitprune and stages[n - 1].split,
					markorigin=stages[n - 1].markorigin)
		if stage.mode.startswith('pcfg-bitpar'):
			assert grammar.maxfanout == 1
		_sumsto1, msg = grammar.testgrammar()
		logging.info('%s: %s', stage.name, msg)
		stage.update(grammar=grammar, backtransform=backtransform, outside=None)
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
	if stage.grammar.currentmodel == 0:
		stage.rulesfile.write(stage.grammar.origrules)
	else:
		stage.rulesfile.writelines(
				'%g\t%s\n' % (weight, line.split(None, 1)[1])
				for weight, line in
				zip(stage.grammar.models[stage.grammar.currentmodel],
					stage.grammar.origrules.splitlines()))
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
