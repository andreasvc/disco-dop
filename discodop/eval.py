"""Evaluation of (discontinuous) parse trees.

Follows EVALB as much as possible, and provides some alternative evaluation
metrics."""
from __future__ import division, print_function
import io
import sys
from getopt import gnu_getopt, GetoptError
from decimal import Decimal, InvalidOperation
from itertools import count
from collections import defaultdict, Counter as multiset
if sys.version[0] >= '3':
	from itertools import zip_longest  # pylint: disable=E0611
else:
	from itertools import izip_longest as zip_longest

from discodop import grammar
from discodop.tree import Tree
from discodop.treedraw import DrawTree
from discodop.treebank import READERS, dependencies
from discodop.treebanktransforms import readheadrules
try:
	from discodop.treedist import treedist, newtreedist
except ImportError:
	from discodop.treedist import newtreedist as newtreedist, newtreedist

USAGE = 'Usage: %s <gold> <parses> [param] [options]' % sys.argv[0]
HELP = '''\
Evaluation of (discontinuous) parse trees, following EVALB as much as possible.

%s
where gold and parses are files with parse trees, param is an EVALB parameter
file, and options may consist of:

  --cutofflen=n    Overrides the sentence length cutoff of the parameter file.
  --verbose        Print table with per sentence information.
  --debug          Print debug information with per sentence bracketings etc.
  --disconly       Only evaluate bracketings of discontinuous constituents
                   (only affects Parseval measures).
  --goldfmt|--parsesfmt=[%s]
                   Specify corpus format [default: export].
  --fmt=[...]      Shorthand for setting both --goldfmt and --parsesfmt.
  --goldenc|--parsesenc=[utf-8|iso-8859-1|...]
                   Specify encoding [default: utf-8].
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
  DELETE_ROOT_PRETERMS
                   when enabled, preterminals directly under the root in
                   gold trees are ignored for scoring purposes.
  DISC_ONLY        only consider discontinuous constituents for F-scores.
  TED              when enabled, give tree-edit distance scores; disabled by
                   default as these are slow to compute.
  DEBUG            -1 only print summary table
                   0 additionally, print category / tag breakdowns (default)
                     (after application of cutoff length).
                   1 give per-sentence results ('--verbose')
                   2 give detailed information for each sentence ('--debug')
  MAX_ERROR        this value is ignored, no errors are tolerated.
                   the parameter is accepted to support usage of unmodified
                   EVALB parameter files. ''' % (USAGE, '|'.join(READERS))

HEADER = '''
   Sentence                 Matched   Brackets            Corr      Tag
  ID Length  Recall  Precis Bracket   gold   cand  Words  Tags Accuracy    LA\
'''.splitlines()


class Evaluator(object):
	"""Incremental evaluator for syntactic trees."""
	def __init__(self, param, keylen=8):
		"""Initialize evaluator object with given parameters."""
		self.param = param
		self.keylen = keylen
		self.acc = EvalAccumulator(param['DISC_ONLY'])
		self.acc40 = None
		if param['CUTOFF_LEN'] is not None:
			self.acc40 = EvalAccumulator(param['DISC_ONLY'])
		if param['DEBUG'] == 1:
			print('Parameters:')
			for a in param:
				print('%s\t%s' % (a, param[a]))
			for a in HEADER:
				print(' ' * (self.keylen - 4) + a)
			print('', '_' * ((self.keylen - 5) + len(HEADER[-1])))

	def add(self, n, gtree, gsent, ctree, csent):
		"""Add a pair of gold and candidate trees to the evaluation.

		:param n: a unique identifier for this sentence.
		:param gtree, ctree: ParentedTree objects (will be modified in-place)
		:param gsent, csent: lists of tokens.
		:returns: a ``TreePairResult`` object."""
		treepair = TreePairResult(n, gtree, gsent, ctree, csent, self.param)
		self.acc.add(treepair)
		if (self.param['CUTOFF_LEN'] is not None
				and treepair.lencpos <= self.param['CUTOFF_LEN']):
			self.acc40.add(treepair)
		if self.param['DEBUG'] > 1:
			treepair.debug()
			for a in HEADER:
				print(' ' * (self.keylen - 4) + a)
			print('', '_' * ((self.keylen - 5) + len(HEADER[-1])))
		if self.param['DEBUG'] >= 1:
			treepair.info('%%%ds  ' % self.keylen)
		return treepair

	def breakdowns(self):
		"""Print breakdowns for the most frequent rules, labels, tags."""
		limit = 10 if self.param['DEBUG'] <= 0 else None
		self.rulebreakdowns(limit)
		self.catbreakdown(limit)
		if accuracy(self.acc.goldpos, self.acc.candpos) != 1:
			self.tagbreakdown(limit)
		print()

	def rulebreakdowns(self, limit=10):
		"""Print breakdowns for the most frequent rule mismatches."""
		acc = self.acc
		# NB: unary nodes not handled properly
		gmismatch = {(n, indices): rule
					for n, indices, rule in acc.goldrule - acc.candrule}
		wrong = multiset((rule, gmismatch[n, indices]) for n, indices, rule
				in acc.candrule - acc.goldrule
				if len(indices) > 1 and (n, indices) in gmismatch)
		print('\n Rewrite rule mismatches (for given span)')
		print('   count   cand / gold rules')
		for (crule, grule), cnt in wrong.most_common(limit):
			print(' %7d  %s' % (cnt, grammar.printrule(*crule)))
			print(' %7s  %s' % (' ', grammar.printrule(*grule)))
		gspans = {(n, indices) for n, indices, _ in acc.goldrule}
		wrong = multiset(rule for n, indices, rule
				in acc.candrule - acc.goldrule
				if len(indices) > 1 and (n, indices) not in gspans)
		print('\n Rewrite rules (span not in gold trees)')
		print('   count   rule in candidate parses')
		for crule, cnt in wrong.most_common(limit):
			print(' %7d  %s' % (cnt, grammar.printrule(*crule)))
		cspans = {(n, indices) for n, indices, _ in acc.candrule}
		wrong = multiset(rule for n, indices, rule
				in acc.goldrule - acc.candrule
				if len(indices) > 1 and (n, indices) not in cspans)
		print('\n Rewrite rules (span missing from candidate parses)')
		print('   count   rule in gold standard set')
		for grule, cnt in wrong.most_common(limit):
			print(' %7d  %s' % (cnt, grammar.printrule(*grule)))

	def catbreakdown(self, limit=10):
		"""Print breakdowns for the most frequent labels."""
		acc = self.acc
		print('\n Attachment errors (correct labeled bracketing, wrong parent)')
		print('  label     cand     gold    count')
		print(' ' + 33 * '_')
		gmismatch = dict(acc.goldbatt - acc.candbatt)
		wrong = multiset((label, cparent, gmismatch[n, label, indices])
					for (n, label, indices), cparent
					in acc.candbatt - acc.goldbatt
					if (n, label, indices) in gmismatch)
		for (cat, gparent, cparent), cnt in wrong.most_common(limit):
			print('%s  %s  %s  %7d' % (cat.rjust(7), gparent.rjust(7),
					cparent.rjust(7), cnt))
		print('\n Category Statistics (%s categories / errors)' % (
				('%d most frequent ' % limit) if limit else 'all'))
		print('  label  % gold  recall    prec.     F1',
				'          cand gold       count')
		print(' ' + 38 * '_' + 7 * ' ' + 24 * '_')
		gmismatch = {(n, indices): label
					for n, (label, indices) in acc.goldb - acc.candb}
		wrong = multiset((label, gmismatch[n, indices])
					for n, (label, indices) in acc.candb - acc.goldb
					if (n, indices) in gmismatch)
		freqcats = sorted(set(acc.goldbcat) | set(acc.candbcat),
				key=lambda x: len(acc.goldbcat[x]), reverse=True)
		for cat, mismatch in zip_longest(freqcats[:limit],
				wrong.most_common(limit)):
			if cat is None:
				print(39 * ' ', end='')
			else:
				print('%s  %6.2f  %s  %s  %s' % (
					cat.rjust(7),
					100 * sum(acc.goldbcat[cat].values()) / len(acc.goldb),
					nozerodiv(lambda: recall(
							acc.goldbcat[cat], acc.candbcat[cat])),
					nozerodiv(lambda: precision(
							acc.goldbcat[cat], acc.candbcat[cat])),
					nozerodiv(lambda: f_measure(
							acc.goldbcat[cat], acc.candbcat[cat])),
					), end='')
			if mismatch is not None:
				print('       %s %7d' % (' '.join((mismatch[0][0].rjust(8),
						mismatch[0][1].ljust(8))), mismatch[1]), end='')
			print()

	def tagbreakdown(self, limit=10):
		"""Print breakdowns for the most frequent tags."""
		acc = self.acc
		print('\n Tag Statistics (%s tags / errors)' % (
			('%d most frequent ' % limit) if limit else 'all'), end='')
		print('\n    tag  % gold  recall   prec.      F1',
				'          cand gold   count')
		print(' ' + 38 * '_' + 7 * ' ' + 20 * '_')
		tags = multiset(tag for _, tag in acc.goldpos)
		wrong = multiset((c, g) for (_, g), (_, c)
				in zip(acc.goldpos, acc.candpos) if g != c)
		for tag, mismatch in zip_longest(tags.most_common(limit),
				wrong.most_common(limit)):
			if tag is None:
				print(''.rjust(40), end='')
			else:
				goldtag = multiset(n for n, (w, t)
						in enumerate(acc.goldpos) if t == tag[0])
				candtag = multiset(n for n, (w, t)
						in enumerate(acc.candpos) if t == tag[0])
				print('%s  %6.2f  %6.2f  %6.2f  %6.2f' % (
						tag[0].rjust(7),
						100 * len(goldtag) / len(acc.goldpos),
						100 * recall(goldtag, candtag),
						100 * precision(goldtag, candtag),
						100 * f_measure(goldtag, candtag)), end='')
			if mismatch is not None:
				print('       %s %7d' % (' '.join((mismatch[0][0].rjust(8),
						mismatch[0][1].ljust(8))).rjust(12), mismatch[1]),
						end='')
			print()

	def summary(self):
		""":returns: a string with an overview of scores for all sentences."""
		acc = self.acc
		acc40 = self.acc40
		discbrackets = sum(1 for _, (_, a) in acc.candb.elements()
				if a != tuple(range(min(a), max(a) + 1)))
		gdiscbrackets = sum(1 for _, (_, a) in acc.goldb.elements()
				if a != tuple(range(min(a), max(a) + 1)))

		if acc.maxlenseen <= self.param['CUTOFF_LEN']:
			msg = ['%s' % ' Summary (ALL) '.center(35, '_'),
				'number of sentences:       %6d' % (acc.sentcount),
				'longest sentence:          %6d' % (acc.maxlenseen)]
			if gdiscbrackets or discbrackets:
				msg.extend(['gold brackets (disc.):     %6d (%d)' % (
							len(acc.goldb), gdiscbrackets),
						'cand. brackets (disc.):    %6d (%d)' % (
							len(acc.candb), discbrackets)])
			else:
				msg.extend(['gold brackets:             %6d' % len(acc.goldb),
					'cand. brackets:            %6d' % len(acc.candb)])
			msg.extend([
					'labeled recall:            %s' % (
						nozerodiv(lambda: recall(acc.goldb, acc.candb))),
					'labeled precision:         %s' % (
						nozerodiv(lambda: precision(acc.goldb, acc.candb))),
					'labeled f-measure:         %s' % (
						nozerodiv(lambda: f_measure(acc.goldb, acc.candb))),
					'exact match:               %s' % (
						nozerodiv(lambda: acc.exact / acc.sentcount)),
					'leaf-ancestor:             %s' % (
						nozerodiv(lambda: mean(acc.lascores)))])
			if self.param['TED']:
				msg.append('tree-dist (Dice micro avg) %s' % (
						nozerodiv(lambda: 1 - acc.dicenoms / acc.dicedenoms)))
			if self.param['DEP']:
				msg.append('unlabeled dependencies:    %s' % (
						nozerodiv(lambda: accuracy(acc.golddep, acc.canddep))))
			msg.append('tagging accuracy:          %s' % (
					nozerodiv(lambda: accuracy(acc.goldpos, acc.candpos))))
			return '\n'.join(msg)

		discbrackets40 = sum(1 for _, (_, a) in acc40.candb.elements()
				if a != tuple(range(min(a), max(a) + 1)))
		gdiscbrackets40 = sum(1 for _, (_, a) in acc40.goldb.elements()
				if a != tuple(range(min(a), max(a) + 1)))
		msg = ['%s <= %d ______ ALL' % (
				' Summary '.center(27, '_'), self.param['CUTOFF_LEN']),
			'number of sentences:       %6d     %6d' % (
					acc40.sentcount, acc.sentcount),
			'longest sentence:          %6d     %6d' % (
					acc40.maxlenseen, acc.maxlenseen),
			'gold brackets:             %6d     %6d' % (
					len(acc40.goldb), len(acc.goldb)),
			'cand. brackets:            %6d     %6d' % (
					len(acc40.candb), len(acc.candb))]
		if gdiscbrackets or discbrackets:
			msg.extend(['disc. gold brackets:       %6d     %6d' % (
					gdiscbrackets40, gdiscbrackets),
					'disc. cand. brackets:      %6d     %6d' % (
					discbrackets40, discbrackets)])
		msg.extend(['labeled recall:            %s     %s' % (
				nozerodiv(lambda: recall(acc40.goldb, acc40.candb)),
				nozerodiv(lambda: recall(acc.goldb, acc.candb))),
			'labeled precision:         %s     %s' % (
				nozerodiv(lambda: precision(acc40.goldb, acc40.candb)),
				nozerodiv(lambda: precision(acc.goldb, acc.candb))),
			'labeled f-measure:         %s     %s' % (
				nozerodiv(lambda: f_measure(acc40.goldb, acc40.candb)),
				nozerodiv(lambda: f_measure(acc.goldb, acc.candb))),
			'exact match:               %s     %s' % (
				nozerodiv(lambda: acc40.exact / acc40.sentcount),
				nozerodiv(lambda: acc.exact / acc.sentcount)),
			'leaf-ancestor:             %s     %s' % (
				nozerodiv(lambda: mean(acc40.lascores)),
				nozerodiv(lambda: mean(acc.lascores)))])
		if self.param['TED']:
			msg.append('tree-dist (Dice micro avg) %s     %s' % (
				nozerodiv(lambda: (1 - acc40.dicenoms / acc40.dicedenoms)),
				nozerodiv(lambda: (1 - acc.dicenoms / acc.dicedenoms))))
		if self.param['DEP']:
			msg.append('unlabeled dependencies:    %s     %s  (%d / %d)' % (
					nozerodiv(lambda: accuracy(acc40.golddep, acc40.canddep)),
					nozerodiv(lambda: accuracy(acc.golddep, acc.canddep)),
					sum(a[0] == a[1] for a in zip(acc.golddep, acc.canddep)),
					len(acc.golddep)))
		msg.append('tagging accuracy:          %s     %s' % (
				nozerodiv(lambda: accuracy(acc40.goldpos, acc40.candpos)),
				nozerodiv(lambda: accuracy(acc.goldpos, acc.candpos))))
		return '\n'.join(msg)


class TreePairResult(object):
	"""Holds the evaluation result of a pair of trees."""
	def __init__(self, n, gtree, gsent, ctree, csent, param):
		"""Construct a pair of gold and candidate trees for evaluation."""
		self.n = n
		self.param = param
		self.gtree, self.ctree = gtree, ctree
		self.csentorig, self.gsentorig = csent, gsent
		self.csent, self.gsent = csent[:], gsent[:]
		self.cpos, self.gpos = sorted(ctree.pos()), sorted(gtree.pos())
		self.lencpos = sum(1 for _, b in self.cpos
				if b not in self.param['DELETE_LABEL_FOR_LENGTH'])
		self.lengpos = sum(1 for _, b in self.gpos
				if b not in self.param['DELETE_LABEL_FOR_LENGTH'])
		assert self.lencpos == self.lengpos, ('sentence length mismatch. '
				'sents:\n%s\n%s' % (' '.join(self.csent), ' '.join(self.gsent)))
		grootpos = {child[0] for child in gtree if isinstance(child[0], int)}
		# massage the data (in-place modifications)
		transform(self.ctree, self.csent, self.cpos, dict(self.gpos),
				self.param, grootpos)
		transform(self.gtree, self.gsent, self.gpos, dict(self.gpos),
				self.param, grootpos)
		#if not gtree or not ctree:
		#	return dict(LP=0, LR=0, LF=0)
		assert self.csent == self.gsent, (
				'candidate & gold sentences do not match:\n'
				'%r // %r' % (' '.join(csent), ' '.join(gsent)))
		self.cbrack = bracketings(ctree, self.param['LABELED'],
				self.param['DELETE_LABEL'], self.param['DISC_ONLY'])
		self.gbrack = bracketings(gtree, self.param['LABELED'],
				self.param['DELETE_LABEL'], self.param['DISC_ONLY'])
		self.lascore = self.ted = self.denom = Decimal('nan')
		self.cdep = self.gdep = ()
		self.pgbrack = self.pcbrack = self.grule = self.crule = ()
		if not self.gpos:
			return  # avoid 'sentences' with only a single punctuation mark.
		self.lascore = leafancestor(self.gtree, self.ctree,
				self.param['DELETE_LABEL'])
		if self.param['TED']:
			self.ted, self.denom = treedisteval(self.gtree, self.ctree,
				includeroot=self.gtree.label not in self.param['DELETE_LABEL'])
		if self.param['DEP']:
			self.cdep = dependencies(self.ctree, self.param['HEADRULES'])
			self.gdep = dependencies(self.gtree, self.param['HEADRULES'])
		assert self.lascore != 1 or self.gbrack == self.cbrack, (
				'leaf ancestor score 1.0 but no exact match: (bug?)')
		self.pgbrack = parentedbracketings(self.gtree, labeled=True,
				dellabel=self.param['DELETE_LABEL'],
				disconly=self.param['DISC_ONLY'])
		self.pcbrack = parentedbracketings(self.ctree, labeled=True,
				dellabel=self.param['DELETE_LABEL'],
				disconly=self.param['DISC_ONLY'])
		self.grule = multiset((node.indices, rule)
				for node, rule in zip(self.gtree.subtrees(),
				grammar.lcfrsproductions(self.gtree, self.gsent)))
		self.crule = multiset((node.indices, rule)
				for node, rule in zip(self.ctree.subtrees(),
				grammar.lcfrsproductions(self.ctree, self.csent)))

	def info(self, fmt='%8s  '):
		"""Print one line with evaluation results."""
		print((fmt + '%5d  %s  %s   %5d  %5d  %5d  %5d  %4d  %s %6.2f%s%s') % (
				self.n, self.lengpos,
				nozerodiv(lambda: recall(self.gbrack, self.cbrack)),
				nozerodiv(lambda: precision(self.gbrack, self.cbrack)),
				sum((self.gbrack & self.cbrack).values()),
				sum(self.gbrack.values()), sum(self.cbrack.values()),
				len(self.gpos),
				sum(1 for a, b in zip(self.gpos, self.cpos) if a == b),
				nozerodiv(lambda: accuracy(self.gpos, self.cpos)),
				100 * self.lascore,
				str(self.ted).rjust(3) if self.param['TED'] else '',
				nozerodiv(lambda: accuracy(self.gdep, self.cdep))
						if self.param['DEP'] else ''))

	def debug(self):
		"""Print detailed information."""
		print('Sentence:', ' '.join(self.gsent))
		print('Gold tree:\n%s\nCandidate tree:\n%s' % (
				self.visualize(gold=True), self.visualize(gold=False)))
		print('Gold brackets:      %s\nCandidate brackets: %s' % (
				strbracketings(self.gbrack), strbracketings(self.cbrack)))
		print('Matched brackets:      %s\nUnmatched brackets: %s' % (
				strbracketings(self.gbrack & self.cbrack),
				strbracketings((self.cbrack - self.gbrack)
					| (self.gbrack - self.cbrack))))
		goldpaths = leafancestorpaths(self.gtree, self.param['DELETE_LABEL'])
		candpaths = leafancestorpaths(self.ctree, self.param['DELETE_LABEL'])
		for leaf in goldpaths:
			print('%6.3g  %s     %s : %s' % (
					pathscore(goldpaths[leaf], candpaths[leaf]),
					self.gsent[leaf].ljust(15),
					' '.join(goldpaths[leaf][::-1]).rjust(20),
					' '.join(candpaths[leaf][::-1])))
		print('%6.3g  average = leaf-ancestor score' % self.lascore)
		print('POS: ', ' '.join('%s/%s' % (a[1], b[1])
				for a, b in zip(self.cpos, self.gpos)))
		if self.param['TED']:
			print('Tree-dist: %g / %g = %g' % (
				self.ted, self.denom, 1 - self.ted / Decimal(self.denom)))
			newtreedist(self.gtree, self.ctree, True)
		if self.param['DEP']:
			print('Sentence:', ' '.join(self.gsent))
			print('dependencies gold', ' ' * 35, 'cand')
			for (_, a, b), (_, c, d) in zip(self.gdep, self.cdep):
				# use original sentences because we don't delete
				# punctuation for dependency evaluation
				print('%15s -> %15s           %15s -> %15s' % (
					self.gsentorig[a - 1][0], self.gsentorig[b - 1][0],
					self.csentorig[c - 1][0], self.csentorig[d - 1][0]))
		print()

	def scores(self):
		"""Return precision, recall, f-measure for sentence pair."""
		return dict(
				LP=nozerodiv(lambda: precision(self.gbrack, self.cbrack)),
				LR=nozerodiv(lambda: recall(self.gbrack, self.cbrack)),
				LF=nozerodiv(lambda: f_measure(self.gbrack, self.cbrack)))

	def bracketings(self):
		"""Return a string representation of bracketing errors."""
		msg = ''
		if self.cbrack - self.gbrack:
			msg += 'cand-gold=%s ' % strbracketings(self.cbrack - self.gbrack)
		if self.gbrack - self.cbrack:
			msg += 'gold-cand=%s' % strbracketings(self.gbrack - self.cbrack)
		return msg

	def visualize(self, gold=False):
		"""Visualize candidate parse, highlight matching POS, bracketings.

		:param gold: by default, the candidate tree is visualized; if True,
			visualize the gold tree instead."""
		tree, brack, pos = self.ctree, self.gbrack, self.gpos
		if gold:
			tree, brack, pos = self.gtree, self.cbrack, self.cpos
		if tree:  # avoid empty trees with just punctuation
			highlight = [a for a in tree.subtrees()
						if bracketing(a) in brack]
			highlight.extend(a for a in tree.subtrees()
						if isinstance(a[0], int)
						and a.label == pos[a[0]])
			highlight.extend(range(len(pos)))
			return DrawTree(tree, self.csent, highlight=highlight
					).text(unicodelines=True, ansi=True)
		return ''


class EvalAccumulator(object):
	"""Collect scores of evaluation."""
	def __init__(self, disconly=False):
		""":param disconly: if True, only collect discontinuous bracketings."""
		self.disconly = disconly
		self.maxlenseen = Decimal(0)
		self.sentcount = Decimal(0)
		self.exact = Decimal(0)
		self.dicenoms, self.dicedenoms = Decimal(0), Decimal(0)
		self.goldb, self.candb = multiset(), multiset()  # all brackets
		self.lascores = []
		self.golddep, self.canddep = [], []
		self.goldpos, self.candpos = [], []
		# extra accounting for breakdowns:
		self.goldbcat = defaultdict(multiset)  # brackets per category
		self.candbcat = defaultdict(multiset)
		self.goldbatt, self.candbatt = set(), set()  # attachments per category
		self.goldrule, self.candrule = multiset(), multiset()

	def add(self, pair):
		"""Add scores from given TreePairResult object."""
		if not self.disconly or pair.cbrack or pair.gbrack:
			self.sentcount += 1
		if self.maxlenseen < pair.lencpos:
			self.maxlenseen = pair.lencpos
		self.candb.update((pair.n, a) for a in pair.cbrack.elements())
		self.goldb.update((pair.n, a) for a in pair.gbrack.elements())
		if pair.cbrack == pair.gbrack:
			if not self.disconly or pair.cbrack or pair.gbrack:
				self.exact += 1
		self.goldpos.extend(pair.gpos)
		self.candpos.extend(pair.cpos)
		if pair.lascore is not None:
			self.lascores.append(pair.lascore)
		if pair.ted is not None:
			self.dicenoms += pair.ted
			self.dicedenoms += pair.denom
		if pair.gdep is not None:
			self.golddep.extend(pair.gdep)
			self.canddep.extend(pair.cdep)
		# extra bookkeeping for breakdowns
		for a in pair.gbrack:
			self.goldbcat[a[0]][(pair.n, a)] += 1
		for a in pair.cbrack:
			self.candbcat[a[0]][(pair.n, a)] += 1
		for (label, indices), parent in pair.pgbrack:
			self.goldbatt.add(((pair.n, label, indices), parent))
		for (label, indices), parent in pair.pcbrack:
			self.candbatt.add(((pair.n, label, indices), parent))
		self.goldrule.update((pair.n, indices, rule)
				for indices, rule in pair.grule)
		self.candrule.update((pair.n, indices, rule)
				for indices, rule in pair.crule)

	def scores(self):
		"""Return a dictionary with running scores for all added sentences."""
		return dict(lr=nozerodiv(lambda: recall(self.goldb, self.candb)),
				lp=nozerodiv(lambda: precision(self.goldb, self.candb)),
				lf=nozerodiv(lambda: f_measure(self.goldb, self.candb)),
				ex=nozerodiv(lambda: self.exact / self.sentcount),
				tag=nozerodiv(lambda: accuracy(self.goldpos, self.candpos)))


def main():
	"""Command line interface for evaluation."""
	flags = {'verbose', 'debug', 'disconly', 'ted'}
	options = {'goldenc=', 'parsesenc=', 'goldfmt=', 'parsesfmt=', 'fmt=',
			'cutofflen=', 'headrules=', 'functions=', 'morphology='}
	try:
		opts, args = gnu_getopt(sys.argv[1:], '', flags | options)
	except GetoptError as err:
		print('error: %s\n%s' % (err, HELP))
		sys.exit(2)
	else:
		opts = dict(opts)
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
	if '--fmt' in opts:
		opts['--goldfmt'] = opts['--parsesfmt'] = opts['--fmt']
	goldreader = READERS[opts.get('--goldfmt', 'export')]
	parsesreader = READERS[opts.get('--parsesfmt', 'export')]
	gold = goldreader(goldfile,
			encoding=opts.get('--goldenc', 'utf-8'),
			functions=opts.get('--functions', 'remove'),
			morphology=opts.get('--morphology'))
	parses = parsesreader(parsesfile,
			encoding=opts.get('--parsesenc', 'utf-8'),
			functions=opts.get('--functions', 'remove'),
			morphology=opts.get('--morphology'))
	goldtrees, goldsents = gold.trees(), gold.sents()
	candtrees, candsents = parses.trees(), parses.sents()
	assert goldtrees, 'no trees in gold file'
	assert candtrees, 'no trees in parses file'
	evaluator = Evaluator(param, max(len(str(x)) for x in candtrees))
	for n, ctree in candtrees.items():
		evaluator.add(n, goldtrees[n], goldsents[n], ctree, candsents[n])
	if param['LABELED'] and param['DEBUG'] != -1:
		evaluator.breakdowns()
	print(evaluator.summary())


def readparam(filename):
	"""Read an EVALB-style parameter file and return a dictionary."""
	param = defaultdict(list)
	# NB: we ignore MAX_ERROR, we abort immediately on error.
	validkeysonce = ('DEBUG', 'MAX_ERROR', 'CUTOFF_LEN', 'LABELED',
			'DISC_ONLY', 'TED', 'DEP', 'DELETE_ROOT_PRETERMS')
	param = {'DEBUG': 0, 'MAX_ERROR': 10, 'CUTOFF_LEN': 40,
				'LABELED': 1, 'DELETE_LABEL_FOR_LENGTH': set(),
				'DELETE_LABEL': set(), 'DELETE_WORD': set(),
				'EQ_LABEL': set(), 'EQ_WORD': set(),
				'DISC_ONLY': 0, 'TED': 0, 'DEP': 0, 'DELETE_ROOT_PRETERMS': 0}
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
	"""Transitive closure of (undirected) EQ relations with DFS.

	Given a sequence of pairs denoting an equivalence relation,
	produce a dictionary with equivalence classes as values and
	arbitrary members of those classes as keys.

	>>> result = transitiveclosure({('A', 'B'), ('B', 'C')})
	>>> len(result)
	1
	>>> k, v = result.popitem()
	>>> k in ('A', 'B', 'C') and v == {'A', 'B', 'C'}
	True"""
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


def transform(tree, sent, pos, gpos, param, grootpos):
	"""Apply the transformations according to the parameter file.

	Does not delete the root node, which is a special case because if there is
	more than one child it cannot be deleted.

	:param pos: a list with the contents of tree.pos(); modified in-place.
	:param gpos: a dictionary of the POS tags of the original gold tree, before
		any tags/words have been deleted.
	:param param: the parameters specifing which labels / words to delete
	:param grootpos: the set of indices with preterminals directly under the
		root node of the gold tree."""
	leaves = list(range(len(sent)))
	posnodes = []
	for a in reversed(list(tree.subtrees(lambda n: isinstance(n[0], Tree)))):
		for n, b in list(zip(count(), a))[::-1]:
			b.label = param['EQ_LABEL'].get(b.label, b.label)
			if not b:
				a.pop(n)  # remove empty nodes
			elif isinstance(b[0], Tree):
				if b.label in param['DELETE_LABEL']:
					# replace phrasal node with its children
					# (must remove nodes from b first because ParentedTree)
					bnodes = b[:]
					b[:] = []
					a[n:n + 1] = bnodes
			elif (gpos[b[0]] in param['DELETE_LABEL']
					or sent[b[0]] in param['DELETE_WORD']
					or (param['DELETE_ROOT_PRETERMS'] and b[0] in grootpos)):
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
		if sent[a[0]] in param['EQ_WORD']:
			sent[a[0]] = param['EQ_WORD'][sent[a[0]]]
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


def parentedbracketings(tree, labeled=True, dellabel=(), disconly=False):
	"""Return the labeled bracketings with parents for a tree.

	:returns:
		multiset with items of the form ``((label, indices), parentlabel)``
	"""
	return multiset((bracketing(a, labeled), getattr(a.parent, 'label', ''))
			for a in tree.subtrees()
			if a and isinstance(a[0], Tree)  # nonempty, not a preterminal
				and a.label not in dellabel
				and (not disconly or disc(a)))


def bracketings(tree, labeled=True, dellabel=(), disconly=False):
	"""Return the labeled set of bracketings for a tree.

	For each nonterminal node, the set will contain a tuple with the label and
	the set of terminals which it dominates.
	``tree`` must have been processed by ``transform()``.
	The argument ``dellabel`` is only used to exclude the ROOT node from the
	results (because it cannot be deleted by ``transform()`` when non-unary).

	>>> tree = Tree.parse('(S (NP 1) (VP (VB 0) (JJ 2)))', parse_leaf=int)
	>>> params = {'DELETE_LABEL': set(), 'DELETE_WORD': set(),
	... 'EQ_LABEL': {}, 'EQ_WORD': {},
	... 'DELETE_ROOT_PRETERMS': 0}
	>>> transform(tree, tree.leaves(), tree.pos(), dict(tree.pos()),
	... params, set())
	>>> sorted(bracketings(tree).items())
	[(('S', (0, 1, 2)), 1), (('VP', (0, 2)), 1)]
	>>> tree = Tree.parse('(S (NP 1) (VP (VB 0) (JJ 2)))', parse_leaf=int)
	>>> params['DELETE_LABEL'] = {'VP'}
	>>> transform(tree, tree.leaves(), tree.pos(), dict(tree.pos()),
	... params, set())
	>>> bracketings(tree)
	Counter({('S', (0, 1, 2)): 1})"""
	return multiset(bracketing(a, labeled) for a in tree.subtrees()
			if a and isinstance(a[0], Tree)  # nonempty, not a preterminal
				and a.label not in dellabel and (not disconly or disc(a)))


def bracketing(node, labeled=True):
	"""Generate bracketing ``(label, indices)`` for a given node."""
	return (node.label if labeled else '', node.indices)


def strbracketings(brackets):
	"""Return a string with a concise representation of a bracketing.

	>>> strbracketings({('S', (0, 1, 2)), ('VP', (0, 2))})
	'S[0-2], VP[0,2]'
	"""
	if not brackets:
		return '{}'
	return ', '.join('%s[%s]' % (a, ','.join(
		'-'.join(str(y) for y in sorted(set(x)))
		for x in intervals(sorted(b)))) for a, b in sorted(brackets))


def leafancestorpaths(tree, dellabel):
	"""Generate a list of ancestors for each leaf node in a tree."""
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
	"""Get edit distance for two leaf-ancestor paths."""
	return 1 - Decimal(editdistance(cand, gold)) / max(len(gold + cand), 1)


def leafancestor(goldtree, candtree, dellabel):
	"""Sampson, Babarcz (2003): A test of the leaf-ancestor metric [...]."""
	gold = leafancestorpaths(goldtree, dellabel)
	cand = leafancestorpaths(candtree, dellabel)
	return mean([pathscore(gold[leaf], cand[leaf]) for leaf in gold])


def treedisteval(a, b, includeroot=False, debug=False):
	"""Get tree-distance for two trees and compute the Dice normalization."""
	ted = treedist(a, b, debug)
	denom = len(list(a.subtrees()) + list(b.subtrees()))  # Dice denominator
	if not includeroot:  # optionally discount ROOT nodes and preterminals
		denom -= 2
	return ted, denom


# If the goldfile contains n constituents for the same span, and the parsed
# file contains m constituents with that nonterminal, the scorer works as
# follows:
#
# i) If m>n, then the precision is n/m, recall is 100%
# ii) If n>m, then the precision is 100%, recall is m/n.
# iii) If n==m, recall and precision are both 100%.
def recall(reference, candidate):
	"""Get recall score for two multisets."""
	if not reference:
		return Decimal('NaN')
	return Decimal(sum(min(reference[a], candidate[a])
			for a in reference & candidate)) / sum(reference.values())


def precision(reference, candidate):
	"""Get precision score for two multisets."""
	if not candidate:
		return Decimal('NaN')
	return Decimal(sum(min(reference[a], candidate[a])
			for a in reference & candidate)) / sum(candidate.values())


def f_measure(reference, candidate, alpha=Decimal(0.5)):
	"""Get F-measure of precision and recall for two multisets.

	The default weight ``alpha=0.5`` corresponds to the F_1-measure."""
	p = precision(reference, candidate)
	r = recall(reference, candidate)
	if p == 0 or r == 0:
		return Decimal('NaN')
	return Decimal(1) / (alpha / p + (1 - alpha) / r)


def accuracy(reference, candidate):
	"""Compute fraction of equivalent pairs in two sequences.

	In particular, return the fraction of indices
	``0<i<=len(test)`` such that ``test[i] == reference[i]``."""
	assert len(reference) == len(candidate), (
		'Sequences must have the same length.')
	return Decimal(sum(1 for a, b in zip(reference, candidate)
			if a == b)) / len(reference)


def harmean(seq):
	"""Compute harmonic mean of a sequence of numbers.

	Returns NaN when ``seq`` contains zero."""
	numerator = denominator = Decimal(0)
	for a in seq:
		if not a:
			return Decimal('NaN')
		numerator += 1
		denominator += Decimal(1) / a
	if not denominator:
		return Decimal('NaN')
	return numerator / denominator


def mean(seq):
	"""Compute arithmetic mean of a sequence.

	Returns NaN when ``seq`` is empty."""
	numerator = denominator = Decimal(0)
	for a in seq:
		numerator += a
		denominator += 1
	if not denominator:
		return Decimal('NaN')
	return numerator / denominator


def intervals(seq):
	"""Return a sequence of intervals corresponding to contiguous ranges.

	``seq`` is a sorted list of integers. An interval is a pair ``(a, b)``,
	with ``a <= b`` denoting indices ``x`` in ``seq``
	such that ``a <= x <= b``.

	>>> list(intervals((0, 1, 3, 4, 6, 7, 8)))
	[(0, 1), (3, 4), (6, 8)]"""
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
	"""Evaluate whether a particular node is locally discontinuous.

	The root node of a complete tree will, by definition, be continuous. Nodes
	can be continuous even if some of their children are discontinuous."""
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
	"""Return ``func()`` as 6-character string but catch zero division."""
	try:
		result = func()
	except (ZeroDivisionError, InvalidOperation):
		return ' 0DIV!'
	return '  None' if result is None else '%6.2f' % (100 * result)


def editdistance(seq1, seq2):
	"""Calculate the Levenshtein edit-distance between two strings.

	The edit distance is the number of characters that need to be substituted,
	inserted, or deleted, to transform seq1 into seq2.  For example,
	transforming 'rain' to 'shine' requires three steps, consisting of two
	substitutions and one insertion: 'rain' -> 'sain' -> 'shin' -> 'shine'.
	These operations could have been done in other orders, but at least three
	steps are needed."""
	# initialize 2-D array to zero
	len1, len2 = len(seq1), len(seq2)
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

if __name__ == '__main__':
	main()
