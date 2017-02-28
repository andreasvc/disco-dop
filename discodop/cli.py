"""Command-line interfaces to modules."""
from __future__ import division, print_function, absolute_import, \
		unicode_literals
from sys import argv, stdout, stderr, version_info
from sys import exit as sysexit

COMMANDS = {
		'runexp': 'Run experiment: grammar extraction, parsing & evaluation.',
		'fragments': 'Extract recurring fragments from treebanks.',
		'eval': 'Evaluate discontinuous parse trees; similar to EVALB.',
		'treetransforms': 'Apply tree transformations '
			'and convert between formats.',
		'treedraw': 'Visualize (discontinuous) trees.',
		'treesearch': 'Query treebanks.',
		'grammar': 'Read off grammars from treebanks.',
		'parser': 'Simple command line parser.',
		'demos': 'Show some demonstrations of formalisms encoded in LCFRS.',
		'gen': 'Generate sentences from a PLCFRS.',
	}


def main():
	"""Expose command-line interfaces."""
	from os import execlp
	from os.path import basename
	thiscmd = basename(argv[0])
	if len(argv) == 2 and argv[1] in ('-v', '--version'):
		from discodop import __version__
		print(__version__)
	elif len(argv) <= 1 or argv[1] not in dict(COMMANDS):
		print('Usage: %s <command> [arguments]\n' % thiscmd, file=stderr)
		print('Command is one of:', file=stderr)
		for a, b in COMMANDS.items():
			print('   %s  %s' % (a.ljust(15), b))
		print('for additional instructions issue: %s <command> --help'
			% thiscmd, file=stderr)
	elif len(argv) == 3 and argv[2] in ('-h', '--help'):
		# help on subcommand
		execlp('man', 'man', 'discodop-%s' % argv[1])
	else:
		cmd = argv[1]
		# use the CLI defined here, or default to the module's main function.
		try:
			globals()[cmd]()
		except KeyError:
			getattr(__import__('discodop.%s' % cmd,
					fromlist=['main']), 'main')()


def treedraw():
	"""Usage: discodop treedraw [<treebank>...] [options]

If no treebank is given, input is read from standard input; format is detected.
Pipe the output through 'less -R' to preserve the colors."""
	from getopt import gnu_getopt, GetoptError
	from itertools import islice, chain
	from .treebank import READERS, incrementaltreereader
	from .tree import DrawTree, frontier
	from .util import openread

	def processtree(tree, sent):
		"""Produced output for a single tree."""
		if output == 'frontier':
			return frontier(tree, sent)
		dt = DrawTree(tree, sent, abbr='--abbr' in opts)
		if output == 'text' or output == 'html':
			return dt.text(unicodelines=True, ansi=ansi, html=html,
					funcsep=funcsep)
		elif output == 'svg':
			return dt.svg(funcsep=funcsep)
		elif output == 'tikznode':
			return dt.tikznode(funcsep=funcsep) + '\n'
		elif output == 'tikzmatrix':
			return dt.tikzmatrix(funcsep=funcsep) + '\n'
		elif output == 'tikzqtree':
			return dt.tikzqtree() + '\n'
		raise ValueError('unrecognized --output format')

	flags = ('test', 'help', 'abbr', 'plain', 'frontier')
	options = ('fmt=', 'encoding=', 'functions=', 'morphology=', 'numtrees=',
			'output=')
	try:
		opts, args = gnu_getopt(argv[2:], 'hn:', flags + options)
	except GetoptError as err:
		print('error:', err, file=stderr)
		print(treedraw.__doc__)
		sysexit(2)
	opts = dict(opts)
	limit = opts.get('--numtrees', opts.get('-n'))
	limit = int(limit) if limit else None
	output = opts.get('--output', 'text')
	funcsep = ('-' if opts.get('--functions')
		in ('add', 'between') else None)
	ansi = output == 'text' and '--plain' not in opts
	html = output == 'html' and '--plain' not in opts
	if output in ('html', 'svg'):
		print(DrawTree.templates[output][0])  # preamble
	elif output in ('tikznode', 'tikzmatrix', 'tikzqtree'):
		print(DrawTree.templates['latex'][0])  # preamble
	if args and opts.get('--fmt', 'export') != 'auto':
		reader = READERS[opts.get('--fmt', 'export')]
		corpora = []
		for path in args:
			corpus = reader(
					path,
					encoding=opts.get('--encoding', 'utf8'),
					functions=opts.get('--functions'),
					morphology=opts.get('--morphology'))
			corpora.append((corpus.trees(), corpus.sents()))
		numsents = len(corpus.sents())
		print('Viewing:', ' '.join(args))
		for n, sentid in enumerate(islice(corpora[0][0], 0, limit), 1):
			print('%d of %s (sentid=%s; len=%d):' % (
					n, numsents, sentid, len(corpora[0][1][sentid])))
			for trees, sents in corpora:
				tree, sent = trees[sentid], sents[sentid]
				print(processtree(tree, sent))
	else:  # read from stdin + detect format
		encoding = opts.get('--encoding', 'utf8')
		if not args:
			args = ['-']
		stream = chain.from_iterable(
				openread(fname, encoding=encoding)
				for fname in args)
		trees = islice(incrementaltreereader(stream,
				morphology=opts.get('--morphology'),
				functions=opts.get('--functions'),
				othertext=True),
				0, limit)
		try:
			n = 1
			for tree, sent, rest in trees:
				if tree is None:
					print(rest)
					continue
				print('%d. (len=%d):' % (n, len(sent)), end=' ')
				if '--frontier' in opts:
					print('%s %s' % (frontier(tree, sent), rest))
				else:
					print(rest or '')
					print(processtree(tree, sent))
				n += 1
		except (IOError, KeyboardInterrupt):
			pass
	if output in ('html', 'svg'):
		print(DrawTree.templates[output][1])  # postamble
	elif output in ('tikznode', 'tikzmatrix', 'tikzqtree'):
		print(DrawTree.templates['latex'][1])  # postamble


def runexp(args=None):
	"""Usage: discodop runexp <parameter file> [--rerun]

If a parameter file is given, an experiment is run. See the file sample.prm for
an example parameter file. To repeat an experiment with an existing grammar,
pass the option --rerun. The directory with the name of the parameter file
without extension must exist in the current path; its results will be
overwritten."""
	import io
	import os
	from .parser import readparam
	from .runexp import startexp, parsetepacoc
	if args is None:
		args = argv[2:]
	if len(args) == 0:
		print('error: incorrect number of arguments', file=stderr)
		print(runexp.__doc__)
		sysexit(2)
	elif '--tepacoc' in args:
		parsetepacoc()
	else:
		rerun = '--rerun' in args
		if rerun:
			args.remove('--rerun')
		params = readparam(args[0])
		resultdir = args[0].rsplit('.', 1)[0]
		top = startexp(
				params, resultdir=resultdir, rerun=rerun)
		if not rerun:  # copy parameter file to result dir
			paramlines = io.open(args[0], encoding='utf8').readlines()
			if paramlines[0].startswith("top='"):
				paramlines = paramlines[1:]
			outfile = os.path.join(resultdir, 'params.prm')
			with io.open(outfile, 'w', encoding='utf8') as out:
				out.write("top='%s',\n" % top)
				out.writelines(paramlines)


def treetransforms():
	"""Treebank binarization and conversion.
Usage: discodop treetransforms [input [output]] [options]
where input and output are treebanks; standard in/output is used if not given.
"""
	import io
	from getopt import gnu_getopt, GetoptError
	from itertools import islice
	from . import treebank, treebanktransforms
	from .treetransforms import canonicalize, binarize, \
			unbinarize, optimalbinarize, splitdiscnodes, mergediscnodes, \
			introducepreterminals, markovthreshold
	flags = ('binarize optimalbinarize unbinarize splitdisc mergedisc '
			'introducepreterminals renumber sentid removeempty '
			'help markorigin markhead leftunary rightunary '
			'tailmarker direction').split()
	options = ('inputfmt= outputfmt= inputenc= outputenc= slice= ensureroot= '
			'punct= headrules= functions= morphology= lemmas= factor= fmt= '
			'markorigin= maxlen= enc= transforms= markovthreshold= labelfun= '
			'transforms= reversetransforms= ').split()
	try:
		origopts, args = gnu_getopt(argv[2:], 'h:v:H:', flags + options)
		if len(args) > 2:
			raise GetoptError('expected 0, 1, or 2 positional arguments')
	except GetoptError as err:
		print('error:', err, file=stderr)
		print(treetransforms.__doc__)
		sysexit(2)
	opts = dict(origopts)
	if '--fmt' in opts:
		opts['--inputfmt'] = opts['--outputfmt'] = opts['--fmt']
	if '--enc' in opts:
		opts['--inputenc'] = opts['--outputenc'] = opts['--enc']
	if opts.get('--outputfmt', treebank.WRITERS[0]) not in treebank.WRITERS:
		print('error: unrecognized output format: %r\navailable formats: %s'
				% (opts.get('--outputfmt'), ' '.join(treebank.WRITERS)),
				file=stderr)
		sysexit(2)
	infilename = (args[0] if len(args) >= 1 else '-')
	outfilename = (args[1] if len(args) == 2 and args[1] != '-'
			else stdout.fileno())

	# open corpus
	corpus = treebank.READERS[opts.get('--inputfmt', 'export')](
			infilename,
			encoding=opts.get('--inputenc', 'utf8'),
			headrules=opts.get('--headrules'),
			ensureroot=opts.get('--ensureroot'),
			removeempty='--removeempty' in opts,
			punct=opts.get('--punct'),
			functions=opts.get('--functions'),
			morphology=opts.get('--morphology'),
			lemmas=opts.get('--lemmas'))
	start, end = opts.get('--slice', ':').split(':')
	start, end = (int(start) if start else None), (int(end) if end else None)
	trees = corpus.itertrees(start, end)
	if '--maxlen' in opts:
		maxlen = int(opts['--maxlen'])
		trees = ((key, item) for key, item in trees
				if len(item.sent) <= maxlen)
	if '--renumber' in opts:
		trees = (('%8d' % n, item) for n, (_key, item) in enumerate(trees, 1))

	# select transformations
	actions = []
	for key, value in origopts:  # pylint: disable=unused-variable
		if key == '--introducepreterminals':
			actions.append(lambda tree, sent:
					(introducepreterminals(tree, sent), sent))
		if key == '--transforms':
			actions.append(lambda tree, sent, value=value:
					(treebanktransforms.transform(tree, sent,
						treebanktransforms.expandpresets(value.split(','))),
					sent))
		if key in ('--binarize', '--optimalbinarize'):
			if key == '--binarize':
				actions.append(lambda tree, sent:
						(binarize(
							tree,
							opts.get('--factor', 'right'),
							int(opts.get('-h', 999)),
							int(opts.get('-v', 1)),
							revhorzmarkov=int(opts.get('-H', 0)),
							leftmostunary='--leftunary' in opts,
							rightmostunary='--rightunary' in opts,
							tailmarker='$' if '--tailmarker' in opts else '',
							direction='--direction' in opts,
							headoutward='--headrules' in opts,
							markhead='--markhead' in opts,
							labelfun=eval(  # pylint: disable=eval-used
								opts['--labelfun'])
								if '--labelfun' in opts else None),
						sent))
			elif key == '--optimalbinarize':
				actions.append(lambda tree, sent:
						(optimalbinarize(
							tree, '|',
							'--headrules' in opts,
							int(opts.get('-h', 999)),
							int(opts.get('-v', 1))),
						sent))
		if key == '--splitdisc':
			actions.append(lambda tree, sent:
					(splitdiscnodes(tree, '--markorigin' in opts), sent))
		if key == '--mergediscnodes':
			actions.append(lambda tree, sent: (mergediscnodes(tree), sent))
		if key == '--unbinarize':
			actions.append(lambda tree, sent: (unbinarize(tree, sent), sent))
		if key == '--reversetransforms':
			actions.append(lambda tree, sent, value=value:
					(treebanktransforms.reversetransform(tree,
						treebanktransforms.expandpresets(value.split(','))),
					sent))

	# read, transform, & write trees
	if actions:
		def applytransforms(trees):
			"""Apply transforms and yield modified items."""
			for key, item in trees:
				for action in actions:
					item.tree, item.sent = action(item.tree, item.sent)
				yield key, item

		trees = applytransforms(trees)
		if 'binarize' in opts and '--markovthreshold' in opts:
			trees = list(trees)
			h, v = int(opts.get('-h', 999)), int(opts.get('-v', 1))
			revh = int(opts.get('-H', 0))
			markovthreshold([item.tree for _, item in trees],
					int(opts['--markovthreshold']),
					revh + h - 1,
					v - 1 if v > 1 else 1)

	if opts.get('--outputfmt') in ('mst', 'conll'):
		if not opts.get('--headrules'):
			raise ValueError('need head rules for dependency conversion')
	cnt = 0
	if opts.get('--outputfmt') == 'dact':
		import alpinocorpus
		outfile = alpinocorpus.CorpusWriter(outfilename)
		if (not actions and opts.get('--inputfmt') in ('alpino', 'dact')
				and set(opts) <= {'--slice', '--inputfmt', '--outputfmt',
				'--renumber'}):
			for n, (key, block) in islice(enumerate(
					corpus.blocks().items(), 1), start, end):
				outfile.write((('%8d' % n) if '--renumber' in opts
						else key).encode('utf8'), block)
				cnt += 1
		else:
			for key, item in trees:
				outfile.write(key.encode('utf8'), treebank.writetree(
						item.tree, item.sent, key, 'alpino',
						comment=item.comment).encode('utf8'))
				cnt += 1
	else:
		encoding = opts.get('outputenc', 'utf8')
		with io.open(outfilename, 'w', encoding=encoding) as outfile:
			# copy trees verbatim when only taking slice or converting encoding
			if (not actions and opts.get('--inputfmt') == opts.get(
					'--outputfmt') and set(opts) <= {'--slice', '--inputenc',
					'--outputenc', '--inputfmt', '--outputfmt'}):
				for block in islice(corpus.blocks().values(), start, end):
					outfile.write(block)
					cnt += 1
			else:
				if opts.get('--outputfmt', 'export') == 'bracket':
					trees = ((key, canonicalize(item.tree) and item)
							for key, item in trees)
				if opts.get('--outputfmt', 'export') == 'export':
					outfile.write(treebank.EXPORTHEADER)
				fmt = opts.get('--outputfmt', 'export')
				sentid = '--sentid' in opts
				for key, item in trees:
					outfile.write(treebank.writetree(item.tree, item.sent, key,
							fmt, comment=item.comment, sentid=sentid))
					cnt += 1
	print('%s: transformed %d trees' % (args[0] if args else 'stdin', cnt),
			file=stderr)


def grammar():
	"""Read off grammars from treebanks.
Usage: discodop grammar <type> <input> <output> [options]
or: discodop grammar param <parameter-file> <output-directory>
or: discodop grammar info <rules-file>
or: discodop grammar merge (rules|lexicon|fragments) \
<input1> <input2>... <output>"""
	import io
	import os
	import codecs
	import logging
	from gzip import open as gzipopen
	from getopt import gnu_getopt, GetoptError
	from .tree import STRTERMRE
	from .util import openread
	from .treebank import READERS
	from .treetransforms import addfanoutmarkers, canonicalize
	from .grammar import treebankgrammar, dopreduction, doubledop, dop1, \
			compiletsg, writegrammar, grammarinfo, grammarstats, \
			splitweight, merge, sumfrags, sumrules, sumlex, stripweight, \
			addindices
	from .parser import readparam
	from .runexp import loadtraincorpus, getposmodel, dobinarization, \
			getgrammars
	logging.basicConfig(level=logging.DEBUG, format='%(message)s')
	shortoptions = 'hs:'
	options = ('help', 'gzip', 'packed', 'bitpar', 'inputfmt=', 'inputenc=',
			'dopestimator=', 'maxdepth=', 'maxfrontier=', 'numproc=')
	try:
		opts, args = gnu_getopt(argv[2:], shortoptions, options)
		model = args[0]
		if model not in ('info', 'merge'):
			treebankfile, grammarfile = args[1:
					]  # pylint: disable=unbalanced-tuple-unpacking
	except (GetoptError, IndexError, ValueError) as err:
		print('error: %r' % err, file=stderr)
		print(grammar.__doc__)
		sysexit(2)
	opts = dict(opts)
	if model not in ('pcfg', 'plcfrs', 'dopreduction', 'doubledop', 'dop1',
			'ptsg', 'param', 'info', 'merge'):
		raise ValueError('unrecognized model: %r' % model)
	if opts.get('dopestimator', 'rfe') not in ('rfe', 'ewe', 'shortest'):
		raise ValueError('unrecognized estimator: %r' % opts['dopestimator'])

	if model == 'info':
		grammarstats(args[1])
		return
	elif model == 'merge':
		if len(args) < 5:
			raise ValueError('need at least 2 input and 1 output arguments.')
		if args[1] == 'rules':
			merge(args[2:-1], args[-1], sumrules, stripweight)
		elif args[1] == 'lexicon':
			merge(args[2:-1], args[-1], sumlex, lambda x: x.split(None, 1)[0])
		elif args[1] == 'fragments':
			merge(args[2:-1], args[-1], sumfrags, lambda x: x.rsplit('\t', 1)[0])
		return
	elif model == 'param':
		if opts:
			raise ValueError('all options should be set in parameter file.')
		prm = readparam(args[1])
		resultdir = args[2]
		if os.path.exists(resultdir):
			raise ValueError('Directory %r already exists.\n' % resultdir)
		os.mkdir(resultdir)
		trees, sents, train_tagged_sents = loadtraincorpus(
				prm.corpusfmt, prm.traincorpus, prm.binarization, prm.punct,
				prm.functions, prm.morphology, prm.removeempty, prm.ensureroot,
				prm.transformations, prm.relationalrealizational)
		simplelexsmooth = False
		if prm.postagging and prm.postagging.method == 'unknownword':
			sents, lexmodel = getposmodel(prm.postagging, train_tagged_sents)
			simplelexsmooth = prm.postagging.simplelexsmooth
	elif model == 'ptsg':  # read fragments
		xfragments = {frag: splitweight(weight) for frag, weight
				in (line.split('\t') for line in openread(treebankfile,
					encoding=opts.get('--inputenc', 'utf8')))}
		if STRTERMRE.search(next(iter(xfragments))) is not None:
			xfragments = {addindices(frag): splitweight(weight) for frag, weight
					in xfragments.items()}
	else:  # read treebank
		corpus = READERS[opts.get('--inputfmt', 'export')](
				treebankfile,
				encoding=opts.get('--inputenc', 'utf8'))
		trees = list(corpus.trees().values())
		sents = list(corpus.sents().values())
		if not trees:
			raise ValueError('no trees; is --inputfmt correct?')
		for a in trees:
			canonicalize(a)
			addfanoutmarkers(a)

	# read off grammar
	if model in ('pcfg', 'plcfrs'):
		xgrammar = treebankgrammar(trees, sents)
	elif model == 'dopreduction':
		xgrammar, altweights = dopreduction(trees, sents,
				packedgraph='--packed' in opts)
	elif model == 'doubledop':
		xgrammar, backtransform, altweights, _ = doubledop(trees, sents,
				numproc=int(opts.get('--numproc', 1)),
				binarized='--bitpar' not in opts)
	elif model == 'dop1':
		xgrammar, backtransform, altweights, _ = dop1(trees, sents,
				maxdepth=int(opts.get('--maxdepth', 3)),
				maxfrontier=int(opts.get('--maxfrontier', 999)),
				binarized='--bitpar' not in opts)
	elif model == 'ptsg':
		xgrammar, backtransform, altweights = compiletsg(xfragments,
				binarized='--bitpar' not in opts)
	elif model == 'param':
		getgrammars(dobinarization(trees, sents, prm.binarization,
				prm.relationalrealizational),
				sents, prm.stages, prm.testcorpus.maxwords, resultdir,
				prm.numproc, lexmodel, simplelexsmooth, trees[0].label)
		paramfile = os.path.join(resultdir, 'params.prm')
		with openread(args[1]) as inp:
			with io.open(paramfile, 'w', encoding='utf8') as out:
				out.write("top='%s',\n%s" % (trees[0].label, inp.read()))
		return  # grammars have already been written
	if opts.get('--dopestimator', 'rfe') != 'rfe':
		xgrammar = [(rule, w) for (rule, _), w in
				zip(xgrammar, altweights[opts['--dopestimator']])]

	rulesname = grammarfile + '.rules'
	lexiconname = grammarfile + '.lex'
	myopen = open
	if '--gzip' in opts:
		myopen = gzipopen
		rulesname += '.gz'
		lexiconname += '.gz'
	bitpar = model == 'pcfg' or opts.get('--inputfmt') == 'bracket'
	if model == 'ptsg':
		bitpar = STRTERMRE.search(next(iter(xfragments))) is not None
	if '--bitpar' in opts and not bitpar:
		raise ValueError('parsing with an unbinarized grammar requires '
				'a grammar in bitpar format.')

	rules, lexicon = writegrammar(xgrammar, bitpar=bitpar)
	# write output
	with codecs.getwriter('utf8')(myopen(rulesname, 'wb')) as rulesfile:
		rulesfile.write(rules)
	with codecs.getwriter('utf8')(myopen(lexiconname, 'wb')) as lexiconfile:
		lexiconfile.write(lexicon)
	if model in ('doubledop', 'ptsg'):
		backtransformfile = '%s.backtransform%s' % (grammarfile,
			'.gz' if '--gzip' in opts else '')
		with codecs.getwriter('utf8')(myopen(backtransformfile, 'wb')) as bt:
			bt.writelines('%s\n' % a for a in backtransform)
		print('wrote backtransform to', backtransformfile)
	print('wrote grammar to %s and %s.' % (rulesname, lexiconname))
	start = opts.get('-s', next(iter(xgrammar))[0][0][0]
			if model == 'ptsg' else trees[0].label)
	if version_info[0] == 2:
		start = start.decode('utf8')
	if len(xgrammar) < 10000:  # this is very slow so skip with large grammars
		print(grammarinfo(xgrammar))
	try:
		from .containers import Grammar
		print(Grammar(rules, lexicon, binarized='--bitpar' not in opts,
				start=start).testgrammar()[1])
	except (ImportError, AssertionError) as err:
		print(err)


if __name__ == "__main__":
	main()

__all__ = ['treedraw', 'runexp', 'treetransforms', 'grammar', 'main']
