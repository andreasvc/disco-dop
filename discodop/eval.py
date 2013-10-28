""" Evaluation of (discontinuous) parse trees, following EVALB as much as
possible, as well as some alternative evaluation metrics.  """
from __future__ import division, print_function
import io
import sys
from getopt import gnu_getopt, GetoptError
from itertools import count
from collections import defaultdict, Counter as multiset
if sys.version[0] >= '3':
	from itertools import zip_longest  # pylint: disable=E0611
else:
	from itertools import izip_longest as zip_longest

from discodop.tree import Tree
from discodop.treedraw import DrawTree
from discodop.treebank import READERS, readheadrules, dependencies, splitpath
try:
	from discodop.treedist import treedist, newtreedist
except ImportError:
	from discodop.treedist import newtreedist as newtreedist, newtreedist

USAGE = 'Usage: %s <gold> <parses> [param] [options]' % sys.argv[0]
HELP = """\
Evaluation of (discontinuous) parse trees, following EVALB as much as possible.

%s
where gold and parses are files with parse trees, param is an EVALB parameter
file, and options may consist of:

  --cutofflen=n    Overrides the sentence length cutoff of the parameter file.
  --verbose        Print table with per sentence information.
  --debug          Print debug information with per sentence bracketings etc.
  --disconly       Only evaluate bracketings of discontinuous constituents
                   (only affects Parseval measures).
  --goldfmt|--parsesfmt=[*%s]
                   Specify corpus format.
  --goldenc|--parsesenc=[*UTF-8|ISO-8859-1|...]
                   Specify a different encoding than the default UTF-8.
  --ted            Enable tree-edit distance evaluation.
  --headrules=x    Specify file with rules for head assignment of constituents
                   that do not already have a child marked as head; this
                   enables dependency evaluation.
  --functions=x    'remove'=default: strip functions off labels,
                   'leave': leave syntactic labels as is,
                   'add': evaluate both syntactic categories and functions,
                   'replace': only evaluate grammatical functions.
  --morphology=x   'no'=default: only evaluate POS tags,
                   'add': concatenate morphology tags to POS tags,
                   'replace': replace POS tags with morphology tags,
                   'between': add morphological node between POS tag and word.

The parameter file should be encoded in UTF-8 and supports the following
options (in addition to those described in the README of EVALB):
  DISC_ONLY        only consider discontinuous constituents for F-scores.
  TED              when enabled, give tree-edit distance scores; disabled by
                   default as these are slow to compute.
  DEBUG            -1 only print summary table
                   0 additionally, print category / tag breakdowns (default)
                     (after application of cutoff length).
                   1 give per-sentence results ('--verbose')
                   2 give detailed information for each sentence ('--debug')
  MAX_ERROR        this values is ignored, no errors are tolerated.
                   the parameter is accepted to support usage of unmodified
                   EVALB parameter files. """ % (
        USAGE, '|'.join(READERS.keys()))

HEADER = """
   Sentence                 Matched   Brackets            Corr      Tag
  ID Length  Recall  Precis Bracket   gold   test  Words  Tags Accuracy    LA\
""".splitlines()


def main():
	""" Command line interface for evaluation. """
	flags = ('test', 'verbose', 'debug', 'disconly', 'ted')
	options = ('goldenc=', 'parsesenc=', 'goldfmt=', 'parsesfmt=',
			'cutofflen=', 'headrules=', 'functions=', 'morphology=')
	try:
		opts, args = gnu_getopt(sys.argv[1:], '', flags + options)
	except GetoptError as err:
		print('error: %s\n%s' % (err, HELP))
		sys.exit(2)
	else:
		opts = dict(opts)
		if '--test' in opts:
			test()
			return
	try:
		assert 2 <= len(args) <= 3, 'Wrong number of arguments.\n%s' % (
				USAGE if len(args) else HELP)
		goldfile, parsesfile = args[:2]
		param = readparam(args[2] if len(args) == 3 else None)
		param['CUTOFF_LEN'] = int(opts.get('--cutofflen', param['CUTOFF_LEN']))
		param['DISC_ONLY'] = '--disconly' in opts
		param['DEBUG'] = max(param['DEBUG'],
				'--verbose' in opts, 2 * ('--debug' in opts))
		param['TED'] |= '--ted' in opts
		param['DEP'] = '--headrules' in opts
		if '--headrules' in opts:
			param['HEADRULES'] = readheadrules(opts['--headrules'])
	except AssertionError as err:
		print('error: %s' % err)
		sys.exit(2)
	goldreader = READERS[opts.get('--goldfmt', 'export')]
	parsesreader = READERS[opts.get('--parsesfmt', 'export')]
	gold = goldreader(*splitpath(goldfile),
			encoding=opts.get('--goldenc', 'utf-8'),
			functions=opts.get('--functions', 'remove'),
			morphology=opts.get('--morphology'))
	parses = parsesreader(*splitpath(parsesfile),
			encoding=opts.get('--parsesenc', 'utf-8'),
			functions=opts.get('--functions', 'remove'),
			morphology=opts.get('--morphology'))
	print(doeval(gold.parsed_sents(),
			gold.tagged_sents(),
			parses.parsed_sents(),
			parses.tagged_sents(),
			param))


def doeval(gold_trees, gold_sents, cand_trees, cand_sents, param):
	""" Do the actual evaluation on given parse trees and parameters.
	Results are printed to standard output. """
	assert gold_trees, 'no trees in gold file'
	assert cand_trees, 'no trees in parses file'
	keylen = max(len(str(x)) for x in cand_trees)
	if param['DEBUG'] == 1:
		print('Parameters:')
		for a in param:
			print('%s\t%s' % (a, param[a]))
		for a in HEADER:
			print(' ' * (keylen - 4) + a)
		print('', '_' * ((keylen - 5) + len(HEADER[-1])))
	# the suffix '40' is for the length restricted results
	maxlenseen = sentcount = maxlenseen40 = sentcount40 = 0
	goldb = multiset()
	candb = multiset()
	goldb40 = multiset()
	candb40 = multiset()
	goldbcat = defaultdict(multiset)
	candbcat = defaultdict(multiset)
	goldbcat40 = defaultdict(multiset)
	candbcat40 = defaultdict(multiset)
	lascores = []
	lascores40 = []
	golddep = []
	canddep = []
	golddep40 = []
	canddep40 = []
	goldpos = []
	candpos = []
	goldpos40 = []
	candpos40 = []
	exact = exact40 = 0.0
	dicenoms = dicedenoms = dicenoms40 = dicedenoms40 = 0
	for n, ctree in cand_trees.items():
		gtree = gold_trees[n]
		cpos = sorted(ctree.pos())
		gpos = sorted(gtree.pos())
		csent = [w for w, _ in cand_sents[n]]
		gsent = [w for w, _ in gold_sents[n]]
		lencpos = sum(1 for _, b in cpos
			if b not in param['DELETE_LABEL_FOR_LENGTH'])
		lengpos = sum(1 for _, b in gpos
			if b not in param['DELETE_LABEL_FOR_LENGTH'])
		assert lencpos == lengpos, ('sentence length mismatch. '
				'sents:\n%s\n%s' % (' '.join(csent), ' '.join(gsent)))
		# massage the data (in-place modifications)
		transform(ctree, csent, cpos, dict(gpos), param['DELETE_LABEL'],
				param['DELETE_WORD'], param['EQ_LABEL'], param['EQ_WORD'])
		transform(gtree, gsent, gpos, dict(gpos), param['DELETE_LABEL'],
				param['DELETE_WORD'], param['EQ_LABEL'], param['EQ_WORD'])
		#if not gtree or not ctree:
		#	continue
		assert csent == gsent, ('candidate & gold sentences do not match:\n'
			'%r // %r' % (' '.join(csent), ' '.join(gsent)))
		cbrack = bracketings(ctree, param['LABELED'], param['DELETE_LABEL'],
				param['DISC_ONLY'])
		gbrack = bracketings(gtree, param['LABELED'], param['DELETE_LABEL'],
				param['DISC_ONLY'])
		if not param['DISC_ONLY'] or cbrack or gbrack:
			sentcount += 1
		# this is to deal with 'sentences' with only a single punctuation mark.
		if not gpos:
			continue
		if maxlenseen < lencpos:
			maxlenseen = lencpos
		if cbrack == gbrack:
			if not param['DISC_ONLY'] or cbrack or gbrack:
				exact += 1
		candb.update((n, a) for a in cbrack.elements())
		goldb.update((n, a) for a in gbrack.elements())
		for a in gbrack:
			goldbcat[a[0]][(n, a)] += 1
		for a in cbrack:
			candbcat[a[0]][(n, a)] += 1
		goldpos.extend(gpos)
		candpos.extend(cpos)
		lascores.append(leafancestor(gtree, ctree, param['DELETE_LABEL']))
		if param['TED']:
			ted, denom = treedisteval(gtree, ctree,
				includeroot=gtree.label not in param['DELETE_LABEL'])
			dicenoms += ted
			dicedenoms += denom
		if param['DEP']:
			cdep = dependencies(ctree, param['HEADRULES'])
			gdep = dependencies(gtree, param['HEADRULES'])
			canddep.extend(cdep)
			golddep.extend(gdep)
		if lencpos <= param['CUTOFF_LEN']:
			if not param['DISC_ONLY'] or cbrack or gbrack:
				sentcount40 += 1
			if maxlenseen40 < lencpos:
				maxlenseen40 = lencpos
			candb40.update((n, a) for a in cbrack.elements())
			goldb40.update((n, a) for a in gbrack.elements())
			for a in gbrack:
				goldbcat40[a[0]][(n, a)] += 1
			for a in cbrack:
				candbcat40[a[0]][(n, a)] += 1
			if cbrack == gbrack:
				if not param['DISC_ONLY'] or cbrack or gbrack:
					exact40 += 1
			goldpos40.extend(gpos)
			candpos40.extend(cpos)
			if lascores[-1] is not None:
				lascores40.append(lascores[-1])
			if param['TED']:
				dicenoms40 += ted
				dicedenoms40 += denom
			if param['DEP']:
				canddep40.extend(cdep)
				golddep40.extend(gdep)
		assert lascores[-1] != 1 or gbrack == cbrack, (
				'leaf ancestor score 1.0 but no exact match: (bug?)')
		if lascores[-1] is None:
			del lascores[-1]
		if param['DEBUG'] <= 0:
			continue
		if param['DEBUG'] > 1:
			for a in HEADER:
				print(' ' * (keylen - 4) + a)
			print('', '_' * ((keylen - 5) + len(HEADER[-1])))
		print(('%' + str(keylen) +
			's  %5d  %s  %s   %5d  %5d  %5d  %5d  %4d  %s %6.2f%s%s') % (
				n,
				lengpos,
				nozerodiv(lambda: recall(gbrack, cbrack)),
				nozerodiv(lambda: precision(gbrack, cbrack)),
				sum((gbrack & cbrack).values()),
				sum(gbrack.values()),
				sum(cbrack.values()),
				len(gpos),
				sum(1 for a, b in zip(gpos, cpos) if a == b),
				nozerodiv(lambda: accuracy(gpos, cpos)),
				100 * lascores[-1],
				str(ted).rjust(3) if param['TED'] else '',
				nozerodiv(lambda: accuracy(gdep, cdep))
						if param['DEP'] else ''))
		if param['DEBUG'] > 1:
			print('Sentence:', ' '.join(gsent))
			print('Gold tree:\n%s\nCandidate tree:\n%s' % (
					DrawTree(gtree, gsent, abbr=True).text(
						unicodelines=True, ansi=True),
					DrawTree(ctree, csent, abbr=True).text(
						unicodelines=True, ansi=True)))
			print('Gold brackets:      %s\nCandidate brackets: %s' % (
					strbracketings(gbrack), strbracketings(cbrack)))
			print('Matched brackets:      %s\nUnmatched brackets: %s' % (
					strbracketings(gbrack & cbrack),
					strbracketings((cbrack - gbrack) | (gbrack - cbrack))))
			goldpaths = leafancestorpaths(gtree, param['DELETE_LABEL'])
			candpaths = leafancestorpaths(ctree, param['DELETE_LABEL'])
			for leaf in goldpaths:
				print('%6.3g  %s     %s : %s' % (
						pathscore(goldpaths[leaf], candpaths[leaf]),
						gsent[leaf].ljust(15),
						' '.join(goldpaths[leaf][::-1]).rjust(20),
						' '.join(candpaths[leaf][::-1])))
			print('%6.3g  average = leaf-ancestor score' % lascores[-1])
			print('POS: ', ' '.join('%s/%s' % (a[1], b[1])
					for a, b in zip(cpos, gpos)))
			if param['TED']:
				print('Tree-dist: %g / %g = %g' % (
					ted, denom, 1 - ted / denom))
				newtreedist(gtree, ctree, True)
			if param['DEP']:
				print('Sentence:', ' '.join(gsent))
				print('dependencies gold', ' ' * 35, 'cand')
				for (_, a, b), (_, c, d) in zip(gdep, cdep):
					# use original sentences because we don't delete
					# punctuation for dependency evaluation
					print('%15s -> %15s           %15s -> %15s' % (
						gold_sents[n][a - 1][0], gold_sents[n][b - 1][0],
						cand_sents[n][c - 1][0], cand_sents[n][d - 1][0]))
			print()
	breakdowns(param, goldb40, candb40, goldpos40, candpos40, goldbcat40,
			candbcat40, maxlenseen)
	msg = summary(param, goldb, candb, goldpos, candpos, sentcount,
			maxlenseen, exact, lascores, dicenoms, dicedenoms, golddep,
			canddep, goldb40, candb40, goldpos40, candpos40, sentcount40,
			maxlenseen40, exact40, lascores40, dicenoms40, dicedenoms40,
			golddep40, canddep40)
	return msg


def breakdowns(param, goldb, candb, goldpos, candpos, goldbcat, candbcat,
		maxlenseen):
	""" Print breakdowns for the most frequent labels / tags. """
	if param['LABELED'] and param['DEBUG'] != -1:
		print()
		print(' Category Statistics (10 most frequent categories / errors)',
				end='')
		if maxlenseen > param['CUTOFF_LEN']:
			print(' for length <= %d' % param['CUTOFF_LEN'], end='')
		print()
		print('  label  % gold  recall    prec.     F1',
				'          test/gold   count')
		print('_______________________________________',
				'       ____________________')
		gmismatch = {(n, indices): label
					for n, (label, indices) in (goldb - candb)}
		wrong = multiset((label, gmismatch[n, indices])
					for n, (label, indices) in (candb - goldb)
					if (n, indices) in gmismatch)
		freqcats = sorted(set(goldbcat) | set(candbcat),
				key=lambda x: len(goldbcat[x]), reverse=True)
		for cat, mismatch in zip_longest(freqcats[:10], wrong.most_common(10)):
			if cat is None:
				print('                                       ', end='')
			else:
				print('%s  %6.2f  %s  %s  %s' % (
					cat.rjust(7),
					100 * sum(goldbcat[cat].values()) / len(goldb),
					nozerodiv(lambda: recall(goldbcat[cat], candbcat[cat])),
					nozerodiv(lambda: precision(goldbcat[cat], candbcat[cat])),
					nozerodiv(lambda: f_measure(goldbcat[cat], candbcat[cat])),
					), end='')
			if mismatch is not None:
				print('       %s %7d' % (
						'/'.join(mismatch[0]).rjust(12), mismatch[1]), end='')
			print()

		if accuracy(goldpos, candpos) != 1:
			print()
			print(' Tag Statistics (10 most frequent tags / errors)', end='')
			if maxlenseen > param['CUTOFF_LEN']:
				print(' for length <= %d' % param['CUTOFF_LEN'], end='')
			print('\n    tag  % gold  recall   prec.      F1',
					'          test/gold   count')
			print('_______________________________________',
					'       ____________________')
			tags = multiset(tag for _, tag in goldpos)
			wrong = multiset((c, g)
					for (_, g), (_, c) in zip(goldpos, candpos) if g != c)
			for tag, mismatch in zip_longest(tags.most_common(10),
					wrong.most_common(10)):
				if tag is None:
					print(''.rjust(40), end='')
				else:
					goldtag = multiset(n for n, (w, t) in enumerate(goldpos)
							if t == tag[0])
					candtag = multiset(n for n, (w, t) in enumerate(candpos)
							if t == tag[0])
					print('%s  %6.2f  %6.2f  %6.2f  %6.2f' % (
							tag[0].rjust(7),
							100 * len(goldtag) / len(goldpos),
							100 * recall(goldtag, candtag),
							100 * precision(goldtag, candtag),
							100 * f_measure(goldtag, candtag)), end='')
				if mismatch is not None:
					print('       %s %7d' % (
						'/'.join(mismatch[0]).rjust(12), mismatch[1]), end='')
				print()
		print()


def summary(param, goldb, candb, goldpos, candpos, sentcount, maxlenseen,
		exact, lascores, dicenoms, dicedenoms, golddep, canddep,
		goldb40, candb40, goldpos40, candpos40, sentcount40, maxlenseen40,
		exact40, lascores40, dicenoms40, dicedenoms40, golddep40, canddep40):
	""" :returns: overview with scores for all sentences. """
	discbrackets = sum(1 for _, (_, a) in candb.elements()
			if a != tuple(range(min(a), max(a) + 1)))
	gdiscbrackets = sum(1 for _, (_, a) in goldb.elements()
			if a != tuple(range(min(a), max(a) + 1)))

	if maxlenseen <= param['CUTOFF_LEN']:
		msg = ['%s' % ' Summary (ALL) '.center(35, '_'),
			'number of sentences:       %6d' % (sentcount),
			'longest sentence:          %6d' % (maxlenseen)]
		if gdiscbrackets or discbrackets:
			msg.extend(['gold brackets (disc.):     %6d (%d)' % (
						len(goldb), gdiscbrackets),
					'cand. brackets (disc.):    %6d (%d)' % (
						len(candb), discbrackets)])
		else:
			msg.extend(['gold brackets:             %6d' % len(goldb),
				'cand. brackets:            %6d' % len(candb)])
		msg.extend([
				'labeled recall:            %s' % (
					nozerodiv(lambda: recall(goldb, candb))),
				'labeled precision:         %s' % (
					nozerodiv(lambda: precision(goldb, candb))),
				'labeled f-measure:         %s' % (
					nozerodiv(lambda: f_measure(goldb, candb))),
				'exact match:               %s' % (
					nozerodiv(lambda: exact / sentcount)),
				'leaf-ancestor:             %s' % (
					nozerodiv(lambda: mean(lascores)))])
		if param['TED']:
			msg.append('tree-dist (Dice micro avg) %s' % (
					nozerodiv(lambda: 1 - dicenoms / dicedenoms)))
		if param['DEP']:
			msg.append('unlabeled dependencies:    %s' % (
					nozerodiv(lambda: accuracy(golddep, canddep))))
		msg.append('tagging accuracy:          %s' % (
				nozerodiv(lambda: accuracy(goldpos, candpos))))
		return '\n'.join(msg)

	discbrackets40 = sum(1 for _, (_, a) in candb40.elements()
			if a != tuple(range(min(a), max(a) + 1)))
	gdiscbrackets40 = sum(1 for _, (_, a) in goldb40.elements()
			if a != tuple(range(min(a), max(a) + 1)))
	msg = ['%s <= %d ______ ALL' % (
			' Summary '.center(27, '_'), param['CUTOFF_LEN']),
		'number of sentences:       %6d     %6d' % (sentcount40, sentcount),
		'longest sentence:          %6d     %6d' % (maxlenseen40, maxlenseen),
		'gold brackets:             %6d     %6d' % (len(goldb40), len(goldb)),
		'cand. brackets:            %6d     %6d' % (len(candb40), len(candb))]
	if gdiscbrackets or discbrackets:
		msg.extend(['disc. gold brackets:       %6d     %6d' % (
				gdiscbrackets40, gdiscbrackets),
				'disc. cand. brackets:      %6d     %6d' % (
				discbrackets40, discbrackets)])
	msg.extend(['labeled recall:            %s     %s' % (
			nozerodiv(lambda: recall(goldb40, candb40)),
			nozerodiv(lambda: recall(goldb, candb))),
		'labeled precision:         %s     %s' % (
			nozerodiv(lambda: precision(goldb40, candb40)),
			nozerodiv(lambda: precision(goldb, candb))),
		'labeled f-measure:         %s     %s' % (
			nozerodiv(lambda: f_measure(goldb40, candb40)),
			nozerodiv(lambda: f_measure(goldb, candb))),
		'exact match:               %s     %s' % (
			nozerodiv(lambda: exact40 / sentcount40),
			nozerodiv(lambda: exact / sentcount)),
		'leaf-ancestor:             %s     %s' % (
			nozerodiv(lambda: mean(lascores40)),
			nozerodiv(lambda: mean(lascores)))])
	if param['TED']:
		msg.append('tree-dist (Dice micro avg) %s     %s' % (
			nozerodiv(lambda: (1 - dicenoms40 / dicedenoms40)),
			nozerodiv(lambda: (1 - dicenoms / dicedenoms))))
	if param['DEP']:
		msg.append('unlabeled dependencies:    %s     %s  (%d / %d)' % (
				nozerodiv(lambda: accuracy(golddep40, canddep40)),
				nozerodiv(lambda: accuracy(golddep, canddep)),
				len([a for a in zip(golddep, canddep) if a[0] == a[1]]),
				len(golddep)))
	msg.append('tagging accuracy:          %s     %s' % (
			nozerodiv(lambda: accuracy(goldpos40, candpos40)),
			nozerodiv(lambda: accuracy(goldpos, candpos))))
	return '\n'.join(msg)


def readparam(filename):
	""" read an EVALB-style parameter file and return a dictionary. """
	param = defaultdict(list)
	# NB: we ignore MAX_ERROR, we abort immediately on error.
	validkeysonce = ('DEBUG', 'MAX_ERROR', 'CUTOFF_LEN', 'LABELED',
			'DISC_ONLY', 'TED', 'DEP')
	param = {'DEBUG': 0, 'MAX_ERROR': 10, 'CUTOFF_LEN': 40,
				'LABELED': 1, 'DELETE_LABEL_FOR_LENGTH': set(),
				'DELETE_LABEL': set(), 'DELETE_WORD': set(),
				'EQ_LABEL': set(), 'EQ_WORD': set(),
				'DISC_ONLY': 0, 'TED': 0, 'DEP': 0}
	seen = set()
	for a in io.open(filename, encoding='utf8') if filename else ():
		line = a.strip()
		if line and not line.startswith('#'):
			key, val = line.split(None, 1)
			if key in validkeysonce:
				assert key not in seen, 'cannot declare %s twice' % key
				seen.add(key)
				param[key] = int(val)
			elif key in ('DELETE_LABEL', 'DELETE_LABEL_FOR_LENGTH',
					'DELETE_WORD'):
				param[key].add(val)
			elif key in ('EQ_LABEL', 'EQ_WORD'):
				# these are given as undirected pairs (A, B), (B, C), ...
				# to be represented as equivalence classes A => {A, B, C, D}
				try:
					b, c = val.split()
				except ValueError:
					raise ValueError('%s requires two values' % key)
				param[key].add((b, c))
			else:
				raise ValueError('unrecognized parameter key: %s' % key)
	for key in ('EQ_LABEL', 'EQ_WORD'):
		# from arbitrary pairs: [('A', 'B'), ('B', 'C')]
		# to eq classes: {'A': {'A', 'B', 'C'}}
		# to a mapping of all elements to their representative:
		# {'A': 'A', 'B': 'A', 'C': 'A'}
		param[key] = {x: k
				for k, eqclass in transitiveclosure(param[key]).items()
					for x in eqclass}
	return param


def transitiveclosure(eqpairs):
	""" Transitive closure of (undirected) EQ relations with DFS;
	i.e., given a sequence of pairs denoting an equivalence relation,
	produce a dictionary with equivalence classes as values and
	arbitrary members of those classes as keys.

	>>> result = transitiveclosure({('A', 'B'), ('B', 'C')})
	>>> len(result)
	1
	>>> k, v = result.popitem()
	>>> k in ('A', 'B', 'C') and v == {'A', 'B', 'C'}
	True """
	edges = defaultdict(set)
	for a, b in eqpairs:
		edges[a].add(b)
		edges[b].add(a)
	eqclasses = {}
	seen = set()
	for elem in set(edges):
		if elem in seen:
			continue
		eqclasses[elem] = set()
		agenda = edges.pop(elem)
		while agenda:
			eqelem = agenda.pop()
			seen.add(eqelem)
			eqclasses[elem].add(eqelem)
			agenda.update(edges[eqelem] - seen)
	return eqclasses


def transform(tree, sent, pos, gpos, dellabel, delword, eqlabel, eqword):
	""" Apply the transformations according to the parameter file,
	except for deleting the root node, which is a special case because if there
	is more than one child it cannot be deleted.

	:param pos: a list with the contents of tree.pos(); modified in-place.
	:param gpos: a dictionary of the POS tags of the original gold tree, before
		any tags/words have been deleted. """
	leaves = list(range(len(sent)))
	posnodes = []
	for a in reversed(list(tree.subtrees(lambda n: isinstance(n[0], Tree)))):
		for n, b in list(zip(count(), a))[::-1]:
			b.label = eqlabel.get(b.label, b.label)
			if not b:
				a.pop(n)  # remove empty nodes
			elif isinstance(b[0], Tree):
				if b.label in dellabel:
					# replace phrasal node with its children
					# (must remove nodes from b first because ParentedTree)
					bnodes = b[:]
					b[:] = []
					a[n:n + 1] = bnodes
			elif gpos[b[0]] in dellabel or sent[b[0]] in delword:
				# remove pre-terminal entirely, but only look at gold tree,
				# to ensure the sentence lengths stay the same
				leaves.remove(b[0])
				del a[n]
			else:
				posnodes.append(b)
				a.indices = b.indices = b[:]
	# retain words still in tree
	sent[:] = [sent[n] for n in leaves]
	pos[:] = [pos[n] for n in leaves]
	# removed POS tags cause the numbering to be off, restore.
	leafmap = {m: n for n, m in enumerate(leaves)}
	for a in posnodes:
		a[0] = leafmap[a[0]]
		a.indices = [a[0]]
		if sent[a[0]] in eqword:
			sent[a[0]] = eqword[sent[a[0]]]
	# cache spans
	for a in reversed(list(tree.subtrees())):
		indices = []
		for b in a:
			if isinstance(b, Tree):
				indices.extend(b.indices)
			elif isinstance(b, int):
				indices.append(b)
			else:
				raise ValueError('Tree should consist of Tree nodes and '
						'integer indices:\n%r' % b)
		assert len(indices) == len(set(indices)), (
			'duplicate index in tree:\n%s' % tree)
		a.indices = tuple(sorted(indices))


def bracketings(tree, labeled=True, dellabel=(), disconly=False):
	""" :returns: the labeled set of bracketings for a tree

	For each nonterminal node, the set will contain a tuple with the label and
	the set of terminals which it dominates.
	``tree`` must have been processed by ``transform()``.
	The argument ``dellabel`` is only used to exclude the ROOT node from the
	results (because it cannot be deleted by ``transform()`` when non-unary).

	>>> tree = Tree.parse('(S (NP 1) (VP (VB 0) (JJ 2)))', parse_leaf=int)
	>>> transform(tree, tree.leaves(), tree.pos(), dict(tree.pos()), (),
	... (), {}, {})
	>>> sorted(bracketings(tree).items())
	[(('S', (0, 1, 2)), 1), (('VP', (0, 2)), 1)]
	>>> tree = Tree.parse('(S (NP 1) (VP (VB 0) (JJ 2)))', parse_leaf=int)
	>>> transform(tree, tree.leaves(), tree.pos(), dict(tree.pos()), ('VP',),
	... (), {}, {})
	>>> bracketings(tree)
	Counter({('S', (0, 1, 2)): 1})
	>>> tree = Tree.parse('(S (NP 1) (VP (VB 0) (JJ 2)))', parse_leaf=int)
	>>> transform(tree, tree.leaves(), tree.pos(), dict(tree.pos()), ('S',),
	... (), {}, {})
	>>> bracketings(tree, dellabel=('S',))
	Counter({('VP', (0, 2)): 1})
	"""
	return multiset(bracketing(a, labeled) for a in tree.subtrees()
			if a and isinstance(a[0], Tree)  # nonempty, not a preterminal
				and a.label not in dellabel
				and (not disconly or disc(a)))


def bracketing(node, labeled=True):
	""" Generate bracketing ``(label, indices)`` for a given node. """
	return (node.label if labeled else '', node.indices)


def strbracketings(brackets):
	""" :returns: a string with a concise representation of a bracketing.

	>>> strbracketings({('S', (0, 1, 2)), ('VP', (0, 2))})
	'S[0-2], VP[0,2]'
	"""
	if not brackets:
		return '{}'
	return ', '.join('%s[%s]' % (a, ','.join(
		'-'.join(str(y) for y in sorted(set(x)))
		for x in intervals(sorted(b)))) for a, b in sorted(brackets))


def leafancestorpaths(tree, dellabel):
	""" Generate a list of ancestors for each leaf node in a tree. """
	#uses [] to mark components, and () to mark constituent boundaries
	#deleted words/tags should not affect boundary detection
	paths = {a: [] for a in tree.indices}
	# do a top-down level-order traversal
	thislevel = [tree]
	while thislevel:
		nextlevel = []
		for n in thislevel:
			leaves = sorted(n.indices)
			# skip empty nodes and POS tags
			if not leaves or not isinstance(n[0], Tree):
				continue
			first, last = min(leaves), max(leaves)
			# skip root node if it is to be deleted
			if n.label not in dellabel:
				for b in leaves:
					# mark end of constituents / components
					if b + 1 not in leaves:
						if b == last and ')' not in paths[b]:
							paths[b].append(')')
						elif b != last and ']' not in paths[b]:
							paths[b].append(']')
					# add this label to the lineage
					paths[b].append(n.label)
					# mark beginning of constituents / components
					if b - 1 not in leaves:
						if b == first and '(' not in paths[b]:
							paths[b].append('(')
						elif b != first and '[' not in paths[b]:
							paths[b].append('[')
			nextlevel.extend(n)
		thislevel = nextlevel
	return paths


def pathscore(gold, cand):
	""" Get edit distance for two leaf-ancestor paths. """
	return 1.0 - (edit_distance(cand, gold)
					/ max(len(gold) + len(cand), 1))


def leafancestor(goldtree, candtree, dellabel):
	""" Geoffrey Sampson, Anna Babarcz (2003):
	A test of the leaf-ancestor metric for parse accuracy """
	gold = leafancestorpaths(goldtree, dellabel)
	cand = leafancestorpaths(candtree, dellabel)
	return mean([pathscore(gold[leaf], cand[leaf]) for leaf in gold])


def treedisteval(a, b, includeroot=False, debug=False):
	""" Get tree-distance for two trees and compute the Dice normalization. """
	ted = treedist(a, b, debug)
	# Dice denominator
	denom = len(list(a.subtrees()) + list(b.subtrees()))
	# optionally discount ROOT nodes and preterminals
	if not includeroot:
		denom -= 2
	#if not includepreterms:
	#	denom -= len(a.leaves() + b.leaves())
	return ted, denom


# If the goldfile contains n constituents for the same span, and the parsed
# file contains m constituents with that nonterminal, the scorer works as
# follows:
#
# i) If m>n, then the precision is n/m, recall is 100%
#
# ii) If n>m, then the precision is 100%, recall is m/n.
#
# iii) If n==m, recall and precision are both 100%.
def recall(reference, candidate):
	""" Get recall score for two multisets. """
	if not reference:
		return float('nan')
	return sum(min(reference[a], candidate[a])
			for a in reference & candidate) / sum(reference.values())


def precision(reference, candidate):
	""" Get precision score for two multisets. """
	if not candidate:
		return float('nan')
	return sum(min(reference[a], candidate[a])
			for a in reference & candidate) / sum(candidate.values())


def f_measure(reference, candidate, alpha=0.5):
	""" Get F-measure of precision and recall for two multisets.
	The default weight ``alpha=0.5`` corresponds to the F_1-measure. """
	p = precision(reference, candidate)
	r = recall(reference, candidate)
	if p == 0 or r == 0:
		return float('nan')
	return 1.0 / (alpha / p + (1 - alpha) / r)


def accuracy(reference, candidate):
	""" Given a sequence of reference values and a corresponding sequence of
	test values, return the fraction of corresponding values that are equal.
	In particular, return the fraction of indices
	``0<i<=len(test)`` such that ``test[i] == reference[i]``. """
	assert len(reference) == len(candidate), (
		'Sequences must have the same length.')
	return sum(1 for a, b in zip(reference, candidate)
			if a == b) / len(reference)


def harmean(seq):
	""" Compute harmonic mean of a sequence of non-zero numbers. """
	numerator = denominator = 0
	for a in seq:
		if not a:
			return float('nan')
		numerator += 1
		denominator += 1 / a
	if not denominator:
		return float('nan')
	return numerator / denominator


def mean(seq):
	""" Compute arithmetic mean of a sequence. """
	numerator = denominator = 0
	for a in seq:
		numerator += a
		denominator += 1
	if not denominator:
		return float('nan')
	return numerator / denominator


def intervals(seq):
	""" Partition seq into a sequence of intervals corresponding to contiguous
	ranges. An interval is a pair ``(a, b)``, with ``a <= b`` denoting
	terminals ``x`` such that ``a <= x <= b``.

	>>> list(intervals((0, 1, 3, 4, 6, 7, 8)))
	[(0, 1), (3, 4), (6, 8)] """
	start = prev = None
	for a in seq:
		if start is None:
			start = prev = a
		elif a == prev + 1:
			prev = a
		else:
			yield start, prev
			start = prev = a
	if start is not None:
		yield start, prev


def disc(node):
	""" This function evaluates whether a particular node is locally
	discontinuous. The root node will, by definition, be continuous.
	Nodes can be continuous even if some of their children are discontinuous.
	"""
	if not isinstance(node, Tree):
		return False
	start = prev = None
	for a in sorted(node.indices):
		if start is None:
			start = prev = a
		elif a == prev + 1:
			prev = a
		else:
			return True
	return False


def nozerodiv(func):
	""" Convenience function to catch zero division or ``None`` as a result
	from evaluating ``func()``; otherwise format its return value as a
	percentage with two decimals; the result is a 6-character string. """
	try:
		result = func()
	except ZeroDivisionError:
		return ' 0DIV!'
	return '  None' if result is None else '%6.2f' % (100 * result)


def edit_distance(seq1, seq2):
	""" Calculate the Levenshtein edit-distance between two strings. The edit
	distance is the number of characters that need to be substituted, inserted,
	or deleted, to transform seq1 into seq2.  For example, transforming 'rain'
	to 'shine' requires three steps, consisting of two substitutions and one
	insertion: 'rain' -> 'sain' -> 'shin' -> 'shine'.  These operations could
	have been done in other orders, but at least three steps are needed.
	"""
	# set up a 2-D array
	len1 = len(seq1)
	len2 = len(seq2)
	# initialize 2-D array to zero
	lev = [[0] * (len2 + 1) for _ in range(len1 + 1)]
	for i in range(len1 + 1):
		lev[i][0] = i           # column 0: 0,1,2,3,4,...
	for j in range(len2 + 1):
		lev[0][j] = j           # row 0: 0,1,2,3,4,...

	# iterate over the array
	for i in range(len1):
		for j in range(len2):
			a = lev[i][j + 1] + 1               # skip seq1[i]
			b = lev[i][j] + (seq1[i] != seq2[j])  # match seq1[i] with seq2[j]
			c = lev[i + 1][j] + 1               # skip seq2[j]
			lev[i + 1][j + 1] = min(a, b, c)    # pick the cheapest
	return lev[len1][len2]


def test():
	""" Simple sanity check; should give 100% score on all metrics. """
	gold = READERS['export']('.', 'alpinosample.export')
	parses = READERS['export']('.', 'alpinosample.export')
	doeval(gold.parsed_sents(),
			gold.tagged_sents(),
			parses.parsed_sents(),
			parses.tagged_sents(),
			readparam(None))

if __name__ == '__main__':
	main()
