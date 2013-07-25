""" Simple command line interface to parse with grammar(s) in text format.  """
from __future__ import print_function
import io
import os
import re
import sys
import time
import gzip
import codecs
import logging
import string  # pylint: disable=W0402
from math import exp
from getopt import gnu_getopt, GetoptError
from operator import itemgetter
from . import plcfrs, pcfg
from .grammar import FORMAT, defaultparse
from .containers import Grammar, DictObj
from .coarsetofine import prunechart, whitelistfromposteriors
from .disambiguation import marginalize, extractfragments
from .tree import Tree
from .lexicon import replaceraretestwords, getunknownwordfun
from .treebank import fold, saveheads
from .treetransforms import mergediscnodes, unbinarize, removefanoutmarkers

USAGE = """
usage: %s [options] rules lexicon [input [output]]
or: %s [options] coarserules coarselexicon finerules finelexicon \
[input [output]]

Grammars need to be binarized, and are in bitpar or PLCFRS format.
When no file is given, output is written to standard output;
when additionally no input is given, it is read from standard input.
Files must be encoded in UTF-8.
Input should contain one token per line, with sentences delimited by two
newlines. Output consists of bracketed trees, with discontinuities indicated
through indices pointing to words in the original sentence.

    Options:
    -b k          Return the k-best parses instead of just 1.
    -s x          Use "x" as start symbol instead of default "TOP".
    --kbestctf k  Use k-best coarse-to-fine;
                  prune items not in k-best derivations (default 50).
    --prob        Print probabilities as well as parse trees.
    --mpd         In coarse-to-fine mode, produce the most probable
                  derivation (MPD) instead of the most probable parse (MPP).

%s
""" % (sys.argv[0], sys.argv[0], FORMAT)

DEFAULTSTAGE = dict(
		name='stage1',  # identifier, used for filenames
		mode='plcfrs',  # use the agenda-based PLCFRS parser
		prune=False,  # whether to use previous chart to prune this stage
		split=False,  # split disc. nodes VP_2[101] as { VP*[100], VP*[001] }
		splitprune=False,  # treat VP_2[101] as {VP*[100], VP*[001]} for pruning
		markorigin=False,  # mark origin of split nodes: VP_2 => {VP*1, VP*2}
		k=50,  # no. of coarse pcfg derivations to prune with; k=0: filter only
		neverblockre=None,  # do not prune nodes with label that match regex
		getestimates=None,  # compute & store estimates
		useestimates=None,  # load & use estimates
		dop=False,  # enable DOP mode (DOP reduction / double DOP)
		packedgraph=False,  # use packed graph encoding for DOP reduction
		usedoubledop=False,  # when False, use DOP reduction instead
		iterate=False,  # for double dop, whether to add fragments of fragments
		complement=False,  # for double dop, whether to include fragments which
			# form the complement of the maximal recurring fragments extracted
		sample=False, kbest=True,
		m=10,  # number of derivations to sample/enumerate
		estimator="ewe",  # choices: dop1, ewe
		objective="mpp",  # choices: mpp, mpd, shortest, sl-dop[-simple]
			# NB: w/shortest derivation, estimator only affects tie breaking.
		sldop_n=7)


def main():
	""" Handle command line arguments. """
	print("PLCFRS parser - Andreas van Cranenburgh", file=sys.stderr)
	options = "kbestctf= prob mpd".split()
	try:
		opts, args = gnu_getopt(sys.argv[1:], "u:b:s:", options)
		assert 2 <= len(args) <= 6, "incorrect number of arguments"
	except (GetoptError, AssertionError) as err:
		print(err, USAGE)
		return
	for n, filename in enumerate(args):
		assert os.path.exists(filename), (
				"file %d not found: %r" % (n + 1, filename))
	opts = dict(opts)
	k = int(opts.get("-b", 1))
	top = opts.get("-s", "TOP")
	prob = "--prob" in opts
	rules = (gzip.open if args[0].endswith(".gz") else open)(args[0]).read()
	lexicon = codecs.getreader('utf-8')((gzip.open if args[1].endswith(".gz")
			else open)(args[1])).read()
	bitpar = rules[0] in string.digits
	coarse = Grammar(rules, lexicon, start=top, bitpar=bitpar)
	stages = []
	infile = (io.open(args[2], encoding='utf-8')
			if len(args) >= 3 else sys.stdin)
	out = (io.open(args[3], "w", encoding='utf-8')
			if len(args) == 4 else sys.stdout)
	stage = DEFAULTSTAGE.copy()
	stage.update(
			name='coarse',
			mode='pcfg' if bitpar else 'plcfrs',
			grammar=coarse,
			secondarymodel=None,
			backtransform=None,
			m=k)
	stages.append(DictObj(stage))
	if 4 <= len(args) <= 6:
		threshold = int(opts.get("--kbestctf", 50))
		rules = (gzip.open if args[2].endswith(".gz") else open)(args[2]).read()
		lexicon = codecs.getreader('utf-8')((gzip.open
				if args[3].endswith(".gz") else open)(args[3])).read()
		# detect bitpar format
		bitpar = rules[0] in string.digits
		fine = Grammar(rules, lexicon, start=top, bitpar=bitpar)
		fine.getmapping(coarse, striplabelre=re.compile(b"@.+$"))
		infile = (io.open(args[4], encoding='utf-8')
				if len(args) >= 5 else sys.stdin)
		out = (io.open(args[5], "w", encoding='utf-8')
				if len(args) == 6 else sys.stdout)
		stage = DEFAULTSTAGE.copy()
		stage.update(
				name='fine',
				mode='pcfg' if bitpar else 'plcfrs',
				grammar=fine,
				secondarymodel=None,
				backtransform=None,
				m=k,
				k=threshold,
				objective='mpd' if '--mpd' in opts else 'mpp')
		stages.append(DictObj(stage))
	doparsing(Parser(stages), infile, out, prob)


def doparsing(parser, infile, out, printprob):
	""" Parse sentences from file and write results to file, log to stdout. """
	times = [time.clock()]
	unparsed = 0
	for n, a in enumerate(infile.read().split("\n\n")):
		if not a.strip():
			continue
		sent = a.splitlines()
		lexicon = parser.stages[0].grammar.lexical
		assert not set(sent) - set(lexicon), (
			"unknown words and no open class tags supplied: %r" % (
			list(set(sent) - set(lexicon))))
		print("parsing %d: %s" % (n, ' '.join(sent)), file=sys.stderr)
		sys.stdout.flush()
		result = list(parser.parse(sent))[-1]
		if result.noparse:
			unparsed += 1
		if printprob:
			out.writelines("vitprob=%.16g\n%s\n" % (exp(-prob), tree)
					for tree, prob in sorted(result.parsetrees.items(),
						key=itemgetter(1)))
		else:
			out.writelines("%s\n" % tree
					for tree in sorted(result.parsetrees,
						key=result.parsetrees.get))
		out.flush()
		times.append(time.clock())
		print(times[-1] - times[-2], "s", file=sys.stderr)
	times = [a - b for a, b in zip(times[1::2], times[::2])]
	print("raw cpu time", time.clock() - times[0],
			"\naverage time per sentence", sum(times) / len(times),
			"\nunparsed sentences:", unparsed,
			"\nfinished",
			file=sys.stderr)
	out.close()


class Parser(object):
	""" An object to parse sentences following parameters given as a sequence
	of coarse-to-fine stages. """
	def __init__(self, stages, transformations=None, tailmarker=None,
			postagging=None):
		""" Parameters:
		stages: a list of coarse-to-fine stages containing grammars and
			parameters.
		transformations: treebank transformations to reverse on parses.
		tailmarker: if heads have been marked with a symbol, use this to
			mark heads in the output.
		postagging: if given, an unknown word model is used, consisting of a
			dictionary with three items:
			- unknownwordfun: function to produces signatures for unknown words.
			- lexicon: the set of known words in the grammar.
			- sigs: the set of word signatures occurring in the grammar. """
		self.stages = stages
		self.transformations = transformations
		self.tailmarker = tailmarker
		self.postagging = postagging

	def parse(self, sent, tags=None):
		""" Parse a sentence and yield a dictionary from parse trees to
		probabilities for each stage.
		tags: if given, will be given to the parser instead of trying all
			possible tags. """
		if self.postagging:
			sent = replaceraretestwords(sent, self.postagging['unknownwordfun'],
					self.postagging['lexicon'], self.postagging['sigs'])
		sent = list(sent)
		if tags is not None:
			tags = list(tags)
		chart = {}
		start = inside = outside = None
		for n, stage in enumerate(self.stages):
			begin = time.clock()
			noparse = False
			parsetrees = fragments = None
			msg = "%s:\t" % stage.name.upper()
			if not stage.prune or start:
				if n != 0 and stage.prune:
					if self.stages[n - 1].mode == 'pcfg-posterior':
						(whitelist, sentprob, unfiltered,
							numitems, numremain) = whitelistfromposteriors(
								inside, outside, start,
								self.stages[n - 1].grammar, stage.grammar,
								stage.k, stage.splitprune,
								self.stages[n - 1].markorigin)
						msg += ("coarse items before pruning=%d; filtered: %d; "
								"pruned: %d; sentprob=%g\n\t" % (
								unfiltered, numitems, numremain, sentprob))
					else:
						whitelist, items = prunechart(
								chart, start, self.stages[n - 1].grammar,
								stage.grammar, stage.k, stage.splitprune,
								self.stages[n - 1].markorigin,
								stage.mode == "pcfg")
						msg += "coarse items before pruning: %d; " % (
								sum(len(a) for x in chart for a in x if a)
								if self.stages[n - 1].mode == 'pcfg'
								else len(chart))
						msg += "after: %d\n\t" % (items)
				else:
					whitelist = None
				if stage.mode == 'pcfg':
					chart, start, msg1 = pcfg.parse(
							sent, stage.grammar, tags=tags,
							chart=whitelist if stage.prune else None)
				elif stage.mode == 'pcfg-symbolic':
					chart, start, msg1 = pcfg.symbolicparse(
							sent, stage.grammar, tags=tags)
				elif stage.mode == 'pcfg-posterior':
					inside, outside, start, msg1 = pcfg.doinsideoutside(
							sent, stage.grammar, tags=tags)
				elif stage.mode == 'plcfrs':
					chart, start, msg1 = plcfrs.parse(sent,
							stage.grammar, tags=tags,
							exhaustive=stage.dop or (n + 1 != len(self.stages)
								and self.stages[n + 1].prune),
							whitelist=whitelist,
							splitprune=stage.splitprune
								and self.stages[n - 1].split,
							markorigin=self.stages[n - 1].markorigin,
							estimates=(stage.useestimates, stage.outside)
								if stage.useestimates in ('SX', 'SXlrgaps')
								else None)
				else:
					raise ValueError
				msg += "%s\n\t" % msg1
				if (n != 0 and not start and not noparse
						and stage.split == self.stages[n - 1].split):
					logging.error("ERROR: expected successful parse. "
							"sent: %s\nstage: %s.", ' '.join(sent), stage.name)
					#raise ValueError("ERROR: expected successful parse. "
					#		"sent %s, %s." % (nsent, stage.name))
			if start and stage.mode != 'pcfg-posterior':
				begindisamb = time.clock()
				parsetrees, msg1 = marginalize(stage.objective,
						chart, start, stage.grammar, stage.m,
						sample=stage.sample, kbest=stage.kbest,
						sent=sent, tags=tags,
						secondarymodel=stage.secondarymodel,
						sldop_n=stage.sldop_n,
						backtransform=stage.backtransform)
				if stage.backtransform is not None and stage.mode == 'plcfrs':
					fragments = extractfragments(
						chart, start, stage.grammar, stage.backtransform)
				resultstr, prob = max(parsetrees.items(), key=itemgetter(1))
				msg += "disambiguation: %s, %gs\n\t" % (
						msg1, time.clock() - begindisamb)
				if isinstance(prob, tuple):
					msg += "subtrees = %d, p=%.4e " % (
							abs(prob[0]), prob[1])
				else:
					msg += "p=%.4e " % prob
				parsetree = Tree.parse(resultstr, parse_leaf=int)
				if stage.split:
					mergediscnodes(unbinarize(parsetree, childchar=":"))
				saveheads(parsetree, self.tailmarker)
				unbinarize(parsetree)
				removefanoutmarkers(parsetree)
				if self.transformations:
					fold(parsetree, self.transformations)
			else:
				parsetree = defaultparse([(n, t)
						for n, t in enumerate(tags or (len(sent) * ['NONE']))])
				parsetree = Tree.parse("(%s %s)" % (stage.grammar.tolabel[1],
						parsetree), parse_leaf=int)
				prob = 0.0
				noparse = True
			elapsedtime = time.clock() - begin
			msg += "%.2fs cpu time elapsed\n" % (elapsedtime)
			yield DictObj(name=stage.name, parsetree=parsetree, prob=prob,
					parsetrees=parsetrees, fragments=fragments,
					noparse=noparse, elapsedtime=elapsedtime, msg=msg)


def readgrammars(resultdir, stages, postagging=None, top='ROOT'):
	""" Read the grammars from a previous experiment. Must have same parameters.
	Expects a directory 'resultdir' which contains the relevant grammars and
	the parameter file 'params.prm', as produced by runexp. """
	for n, stage in enumerate(stages):
		logging.info("reading: %s", stage.name)
		rules = gzip.open("%s/%s.rules.gz" % (resultdir, stage.name))
		lexicon = codecs.getreader('utf-8')(gzip.open("%s/%s.lex.gz" % (
				resultdir, stage.name)))
		grammar = Grammar(rules.read(), lexicon.read(),
				start=top, bitpar=stage.mode.startswith('pcfg'),
				logprob=stage.mode != 'pcfg-posterior')
		backtransform = None
		if stage.dop:
			assert stage.objective not in (
					"shortest", "sl-dop", "sl-dop-simple"), "not supported."
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
	if postagging and postagging['method'] == 'unknownword':
		postagging['unknownwordfun'] = getunknownwordfun(postagging['model'])
		postagging['lexicon'] = {w for w in stages[0].grammar.lexical
				if not w.startswith("UNK")}
		postagging['sigs'] = {w for w in stages[0].grammar.lexical
				if w.startswith("UNK")}


if __name__ == '__main__':
	main()
