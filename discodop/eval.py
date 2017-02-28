"""Evaluation of (discontinuous) parse trees.

Designed to behave like the reference implementation EVALB [1] for regular
parse trees, with a natural extension to the discontinuous case. Also provides
additional, alternative parse tree evaluation metrics (leaf ancestor, tree-edit
distance, unlabeled dependencies), as well as facilities for error analysis.

[1] http://nlp.cs.nyu.edu/evalb/"""
# pylint: disable=cell-var-from-loop
from __future__ import division, print_function, absolute_import, \
		unicode_literals
import io
import sys
from getopt import gnu_getopt, GetoptError
from decimal import Decimal, InvalidOperation
from collections import defaultdict, Counter  # == multiset
if sys.version_info[0] == 2:
	from itertools import count, \
			izip_longest as zip_longest  # pylint: disable=no-name-in-module
else:
	from itertools import count, \
			zip_longest  # pylint: disable=no-name-in-module
from . import grammar
from .tree import Tree, DrawTree, isdisc, bitfanout
from .treebank import READERS, dependencies, handlefunctions
from .treetransforms import getbits
from .treebanktransforms import functions
from .treedist import treedist, newtreedist

SHORTUSAGE = 'Usage: discodop eval <gold> <parses> [param] [options]'

HEADER = '''
   Sentence                 Matched   Brackets            Corr   POS
  ID Length  Recall  Precis Bracket   gold   cand  Words  POS  Accur.\
'''.splitlines()


class Evaluator(object):
	"""Incremental evaluator for syntactic trees."""

	def __init__(self, param, keylen=8):
		"""Initialize evaluator object with given parameters.

		:param param: a dictionary of parameters, as read by ``readparam``.
		:param keylen: the length of the longest sentence ID, for padding
			purposes."""
		self.param = param
		self.keylen = keylen
		self.acc = EvalAccumulator(param['DISC_ONLY'])
		self.acc40 = None
		if param['CUTOFF_LEN'] is not None:
			self.acc40 = EvalAccumulator(param['DISC_ONLY'])
		if param['DEBUG'] >= 1:
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
				and treepair.lengpos <= self.param['CUTOFF_LEN']):
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
		if self.acc.candfun:
			self.funcbreakdown(limit)
		try:
			acc = accuracy(self.acc.goldpos, self.acc.candpos)
		except InvalidOperation:
			pass
		else:
			if acc != 1:
				self.tagbreakdown(limit)
		print()

	def rulebreakdowns(self, limit=10):
		"""Print breakdowns for the most frequent rule mismatches."""
		acc = self.acc
		# NB: unary nodes not handled properly
		gmismatch = {(n, indices): rule
					for n, indices, rule in acc.goldrule - acc.candrule}
		wrong = Counter((rule, gmismatch[n, indices]) for n, indices, rule
				in acc.candrule - acc.goldrule
				if pyintbitcount(indices) > 1 and (n, indices) in gmismatch)
		print('\n Rewrite rule mismatches (for given span)')
		print('   count   cand / gold rules')
		for (crule, grule), cnt in wrong.most_common(limit):
			print(' %7d  %s' % (cnt, grammar.printrule(*crule)))
			print(' %7s  %s' % (' ', grammar.printrule(*grule)))
		gspans = {(n, indices) for n, indices, _ in acc.goldrule}
		wrong = Counter(rule for n, indices, rule
				in acc.candrule - acc.goldrule
				if pyintbitcount(indices) > 1 and (n, indices) not in gspans)
		print('\n Rewrite rules (span not in gold trees)')
		print('   count   rule in candidate parses')
		for crule, cnt in wrong.most_common(limit):
			print(' %7d  %s' % (cnt, grammar.printrule(*crule)))
		cspans = {(n, indices) for n, indices, _ in acc.candrule}
		wrong = Counter(rule for n, indices, rule
				in acc.goldrule - acc.candrule
				if pyintbitcount(indices) > 1 and (n, indices) not in cspans)
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
		wrong = Counter((label, cparent, gmismatch[n, label, indices])
					for (n, label, indices), cparent
					in acc.candbatt - acc.goldbatt
					if (n, label, indices) in gmismatch)
		for (cat, gparent, cparent), cnt in wrong.most_common(limit):
			print('%s  %s  %s  %7d' % (cat.rjust(7), gparent.rjust(7),
					cparent.rjust(7), cnt))
		print('\n Category Statistics (%s categories / errors)' % (
				('%d most frequent' % limit) if limit else 'all'))
		print('  label  % gold  recall    prec.     F1',
				'          cand gold       count')
		print(' ' + 38 * '_' + 8 * ' ' + 24 * '_')
		gmismatch = {(n, indices): label
					for n, (label, indices) in acc.goldb - acc.candb}
		wrong = Counter((label, gmismatch[n, indices])
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

	def funcbreakdown(self, limit=10):
		"""Print breakdowns for the most frequent function tags."""
		acc = self.acc
		print('\n Function Tag Statistics (%s tags / errors)' % (
				('%d most frequent' % limit) if limit else 'all'))
		print('  func.  % gold  recall    prec.     F1',
				'          cand gold       count')
		print(' ' + 38 * '_' + 8 * ' ' + 24 * '_')
		gmismatch = {(n, span): tag
					for n, (span, tag) in acc.goldfun - acc.candfun}
		wrong = Counter((tag, gmismatch[n, span])
					for n, (span, tag) in acc.candfun - acc.goldfun
					if (n, span) in gmismatch)
		freqcats = sorted(set(acc.goldbfunc) | set(acc.candbfunc),
				key=lambda x: len(acc.goldbfunc[x]), reverse=True)
		for cat, mismatch in zip_longest(freqcats[:limit],
				wrong.most_common(limit)):
			if cat is None:
				print(39 * ' ', end='')
			else:
				print('%s  %6.2f  %s  %s  %s' % (
					cat.rjust(7),
					100 * sum(acc.goldbfunc[cat].values()) / len(acc.goldfun),
					nozerodiv(lambda: recall(
							acc.goldbfunc[cat], acc.candbfunc[cat])),
					nozerodiv(lambda: precision(
							acc.goldbfunc[cat], acc.candbfunc[cat])),
					nozerodiv(lambda: f_measure(
							acc.goldbfunc[cat], acc.candbfunc[cat])),
					), end='')
			if mismatch is not None:
				print('       %s %7d' % (' '.join((mismatch[0][0].rjust(8),
						mismatch[0][1].ljust(8))), mismatch[1]), end='')
			print()

	def tagbreakdown(self, limit=10):
		"""Print breakdowns for the most frequent tags."""
		acc = self.acc
		print('\n POS Statistics (%s tags / errors)' % (
			('%d most frequent' % limit) if limit else 'all'), end='')
		print('\n    tag  % gold  recall   prec.      F1',
				'          cand gold   count')
		print(' ' + 38 * '_' + 12 * ' ' + 20 * '_')
		tags = Counter(acc.goldpos)
		wrong = Counter((c, g) for c, g
				in zip(acc.candpos, acc.goldpos) if c != g)
		for tag, mismatch in zip_longest(tags.most_common(limit),
				wrong.most_common(limit)):
			if tag is None:
				print(''.rjust(40), end='')
			else:
				# only one tag per index may occur, but multiset is required by
				# metrics
				goldtag = Counter(n for n, t in enumerate(acc.goldpos)
						if t == tag[0])
				candtag = Counter(n for n, t in enumerate(acc.candpos)
						if t == tag[0])
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
				if bitfanout(a) > 1)
		gdiscbrackets = sum(1 for _, (_, a) in acc.goldb.elements()
				if bitfanout(a) > 1)

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
						nozerodiv(lambda: acc.exact / acc.sentcount))])
			if self.param['LA']:
				msg.append('leaf-ancestor:             %s' % (
						nozerodiv(lambda: mean(acc.lascores))))
			if self.param['TED']:
				msg.append('tree-dist (Dice micro avg) %s' % (
						nozerodiv(lambda: 1 - acc.dicenoms / acc.dicedenoms)))
			if self.param['DEP']:
				msg.append('unlabeled dependencies:    %s' % (
						nozerodiv(lambda: accuracy(acc.golddep, acc.canddep))))
			if acc.candfun:
				msg.append('function tags:             %s' %
						nozerodiv(lambda: f_measure(acc.goldfun, acc.candfun)))
			msg.append('pos accuracy:              %s' % (
					nozerodiv(lambda: accuracy(acc.goldpos, acc.candpos))))
			return '\n'.join(msg)

		discbrackets40 = sum(1 for _, (_, a) in acc40.candb.elements()
				if bitfanout(a) > 1)
		gdiscbrackets40 = sum(1 for _, (_, a) in acc40.goldb.elements()
				if bitfanout(a) > 1)
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
				nozerodiv(lambda: acc.exact / acc.sentcount))])
		if self.param['LA']:
			msg.append('leaf-ancestor:             %s     %s' % (
				nozerodiv(lambda: mean(acc40.lascores)),
				nozerodiv(lambda: mean(acc.lascores))))
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
		if acc.candfun:
			msg.append('function tags:             %s     %s' % (
					nozerodiv(lambda: f_measure(acc40.goldfun, acc40.candfun)),
					nozerodiv(lambda: f_measure(acc.goldfun, acc.candfun))))
		msg.append('pos accuracy:              %s     %s' % (
				nozerodiv(lambda: accuracy(acc40.goldpos, acc40.candpos)),
				nozerodiv(lambda: accuracy(acc.goldpos, acc.candpos))))
		return '\n'.join(msg)


class TreePairResult(object):
	"""Holds the evaluation result of a pair of trees."""

	def __init__(self, n, gtree, gsent, ctree, csent, param):
		"""Construct a pair of gold and candidate trees for evaluation."""
		self.n = n
		self.param = param
		self.csentorig, self.gsentorig = csent, gsent
		self.csent, self.gsent = csent[:], gsent[:]
		self.cpos, self.gpos = sorted(ctree.pos()), sorted(gtree.pos())
		self.lengpos = sum(1 for _, b in self.gpos
				if b not in self.param['DELETE_LABEL_FOR_LENGTH'])
		grootpos = {child[0] for child in gtree
				if child and isinstance(child[0], int)}
		# massage the data (in-place modifications)
		self.ctree = transform(ctree, self.csent, self.cpos,
				alignsent(self.csent, self.gsent, dict(self.gpos)),
				self.param, grootpos)
		self.gtree = transform(gtree, self.gsent, self.gpos,
				dict(self.gpos), self.param, grootpos)
		if len(self.csent) != len(self.gsent):
			raise ValueError('sentence length mismatch. sents:\n%s\n%s' % (
					' '.join(self.csent), ' '.join(self.gsent)))
		if self.csent != self.gsent:
			raise ValueError('candidate & gold sentences do not match:\n'
					'%r // %r' % (' '.join(csent), ' '.join(gsent)))
		self.cbrack = bracketings(self.ctree, self.param['LABELED'],
				self.param['DELETE_LABEL'], self.param['DISC_ONLY'])
		self.gbrack = bracketings(self.gtree, self.param['LABELED'],
				self.param['DELETE_LABEL'], self.param['DISC_ONLY'])
		self.lascore = self.ted = self.denom = Decimal('nan')
		self.cdep = self.gdep = ()
		self.pgbrack = Counter()
		self.pcbrack = Counter()
		self.grule = Counter()
		self.crule = Counter()
		# collect the function tags for correct bracketings & POS tags
		self.candfun = Counter((bracketing(a), b)
				for a in self.ctree.subtrees()
					for b in functions(a)
					if bracketing(a) in self.gbrack or (
						a and isinstance(a[0], int)
						and self.gpos[a[0]] == a.label))
		self.goldfun = Counter((bracketing(a), b)
				for a in self.gtree.subtrees()
					for b in functions(a)
					if bracketing(a) in self.cbrack or (
						a and isinstance(a[0], int)
						and self.cpos[a[0]] == a.label))
		if not self.gpos:
			return  # avoid 'sentences' with only punctuation.
		if self.param['LA']:
			self.lascore = leafancestor(self.gtree, self.ctree,
					self.param['DELETE_LABEL'])
		if self.param['TED']:
			self.ted, self.denom = treedisteval(self.gtree, self.ctree,
				includeroot=self.gtree.label not in self.param['DELETE_LABEL'])
		if self.param['DEP']:
			self.cdep = dependencies(self.ctree)
			self.gdep = dependencies(self.gtree)
		assert self.lascore != 1 or self.gbrack == self.cbrack, (
				'leaf ancestor score 1.0 but no exact match: (bug?)')
		self.pgbrack = parentedbracketings(self.gtree, labeled=True,
				dellabel=self.param['DELETE_LABEL'],
				disconly=self.param['DISC_ONLY'])
		self.pcbrack = parentedbracketings(self.ctree, labeled=True,
				dellabel=self.param['DELETE_LABEL'],
				disconly=self.param['DISC_ONLY'])
		self.grule = Counter((node.bitset, rule)
				for node, rule in zip(self.gtree.subtrees(),
				grammar.lcfrsproductions(self.gtree, self.gsent)))
		self.crule = Counter((node.bitset, rule)
				for node, rule in zip(self.ctree.subtrees(),
				grammar.lcfrsproductions(self.ctree, self.csent)))

	def info(self, fmt='%8s  '):
		"""Print one line with evaluation results."""
		print((fmt + '%5d  %s  %s   %5d  %5d  %5d  %5d  %4d  %s %s%s%s%s') % (
				self.n, self.lengpos,
				nozerodiv(lambda: recall(self.gbrack, self.cbrack)),
				nozerodiv(lambda: precision(self.gbrack, self.cbrack)),
				sum((self.gbrack & self.cbrack).values()),
				sum(self.gbrack.values()), sum(self.cbrack.values()),
				len(self.gpos),
				sum(1 for a, b in zip(self.gpos, self.cpos) if a == b),
				nozerodiv(lambda: accuracy(self.gpos, self.cpos)),
				nozerodiv(lambda: f_measure(self.goldfun, self.candfun))
						if self.candfun else '',
				nozerodiv(lambda: 100 * self.lascore)
						if self.param['LA'] else '',
				nozerodiv(lambda: self.ted) if self.param['TED'] else '',
				nozerodiv(lambda: accuracy(self.gdep, self.cdep))
						if self.param['DEP'] else ''))

	def debug(self):
		"""Print detailed information."""
		print('Sentence:', ' '.join(self.gsent))
		print('Gold tree:\n%s\nCandidate tree:\n%s' % (
				self.visualize(gold=True), self.visualize(gold=False)))
		print('Gold brackets:      ', strbracketings(self.gbrack))
		print('Candidate brackets: ', strbracketings(self.cbrack))
		print('Matched brackets:   ',
				strbracketings(self.gbrack & self.cbrack))
		print('Unmatched brackets: ', strbracketings(
				(self.cbrack - self.gbrack) | (self.gbrack - self.cbrack)))
		goldpaths = leafancestorpaths(self.gtree, self.param['DELETE_LABEL'])
		candpaths = leafancestorpaths(self.ctree, self.param['DELETE_LABEL'])
		if self.candfun:
			print('Function tags')
			print('gold:               ', strbracketings(
					(a, b) for (_, b), a in self.goldfun))
			print('candidate:          ', strbracketings(
					(a, b) for (_, b), a in self.candfun))
			print('matched:            ', strbracketings(
					(a, b) for (_, b), a in self.candfun & self.goldfun))
			print('unmatched:          ', strbracketings(
					(a, b) for (_, b), a in (self.candfun - self.goldfun)
					| (self.goldfun - self.candfun)))

		print('%15s %8s %8s | %10s %36s : %s' % (
				'word', 'gold POS', 'cand POS',
				'path score', 'gold path', 'cand path'))
		for leaf in goldpaths:
			print('%15s %8s %8s   %6.3g %40s : %s' % (
					self.gsent[leaf],
					self.gpos[leaf],
					self.cpos[leaf],
					pathscore(goldpaths[leaf], candpaths[leaf]),
					' '.join(goldpaths[leaf][::-1]),
					' '.join(candpaths[leaf][::-1])))
		if self.param['LA']:
			print('leaf-ancestor score: %6.3g' % self.lascore)
		if self.param['TED']:
			print('Tree-dist: %g / %g = %g' % (
				self.ted, self.denom, 1 - self.ted / Decimal(self.denom)))
			newtreedist(self.gtree, self.ctree, True)
		if self.param['DEP']:
			print('Sentence:', ' '.join(self.gsent))
			print('dependencies gold', ' ' * 35, 'cand')
			for (a, _, b), (c, _, d) in zip(self.gdep, self.cdep):
				# use original sentences because we don't delete
				# punctuation for dependency evaluation
				print('%15s -> %15s           %15s -> %15s' % (
					self.gsentorig[a - 1][0], self.gsentorig[b - 1][0],
					self.csentorig[c - 1][0], self.csentorig[d - 1][0]))
		print()

	def scores(self):
		"""Return precision, recall, f-measure for sentence pair."""
		return dict(LP=nozerodiv(lambda: precision(self.gbrack, self.cbrack)),
				LR=nozerodiv(lambda: recall(self.gbrack, self.cbrack)),
				LF=nozerodiv(lambda: f_measure(self.gbrack, self.cbrack)),
				POS=nozerodiv(lambda: accuracy(self.gpos, self.cpos)),
				FUN=nozerodiv(lambda: f_measure(self.goldfun, self.candfun)))

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
		if not tree:  # avoid empty trees with just punctuation
			return ''
		if self.candfun:
			tree = tree.copy(True)
		highlight = list(tree.subtrees(lambda n: bracketing(n) in brack))
		highlight.extend(tree.subtrees(lambda n: n and isinstance(n[0], int)
				and n.label == pos[n[0]]))
		highlight.extend(range(len(pos)))
		highlightfunc = ()
		if self.candfun:
			highlightfunc = [a for a in tree.subtrees()
					if all((bracketing(a), b) in self.candfun & self.goldfun
						for b in functions(a))]
			handlefunctions('add', tree)
		return DrawTree(tree, self.csent, highlight=highlight,
				highlightfunc=highlightfunc).text(
				unicodelines=True, ansi=True,
				funcsep='-' if self.candfun else None)


class EvalAccumulator(object):
	"""Collect scores of evaluation."""

	def __init__(self, disconly=False):
		""":param disconly: if True, only collect discontinuous bracketings."""
		self.disconly = disconly
		self.maxlenseen, self.sentcount = Decimal(0), Decimal(0)
		self.exact = Decimal(0)
		self.dicenoms, self.dicedenoms = Decimal(0), Decimal(0)
		self.goldb, self.candb = Counter(), Counter()  # all brackets
		self.goldfun, self.candfun = Counter(), Counter()
		self.lascores = []
		self.golddep, self.canddep = [], []
		self.goldpos, self.candpos = [], []
		# extra accounting for breakdowns:
		self.goldbcat = defaultdict(Counter)  # brackets per category
		self.candbcat = defaultdict(Counter)
		self.goldbfunc = defaultdict(Counter)  # brackets by function tag
		self.candbfunc = defaultdict(Counter)
		self.goldbatt, self.candbatt = set(), set()  # attachments per category
		self.goldrule, self.candrule = Counter(), Counter()

	def add(self, pair):
		"""Add scores from given TreePairResult object."""
		if not self.disconly or pair.cbrack or pair.gbrack:
			self.sentcount += 1
		if self.maxlenseen < pair.lengpos:
			self.maxlenseen = pair.lengpos
		self.candb.update((pair.n, a) for a in pair.cbrack.elements())
		self.goldb.update((pair.n, a) for a in pair.gbrack.elements())
		if pair.cbrack == pair.gbrack:
			if not self.disconly or pair.cbrack or pair.gbrack:
				self.exact += 1
		self.goldpos.extend(pair.gpos)
		self.candpos.extend(pair.cpos)
		self.goldfun.update((pair.n, a) for a in pair.goldfun.elements())
		self.candfun.update((pair.n, a) for a in pair.candfun.elements())
		if pair.lascore is not None:
			self.lascores.append(pair.lascore)
		if pair.ted is not None:
			self.dicenoms += pair.ted
			self.dicedenoms += pair.denom
		if pair.gdep is not None:
			self.golddep.extend(pair.gdep)
			self.canddep.extend(pair.cdep)
		# extra bookkeeping for breakdowns
		for a, n in pair.gbrack.items():
			self.goldbcat[a[0]][(pair.n, a)] += n
		for a, n in pair.cbrack.items():
			self.candbcat[a[0]][(pair.n, a)] += n
		for a, n in pair.goldfun.items():
			self.goldbfunc[a[1]][(pair.n, a)] += n
		for a, n in pair.candfun.items():
			self.candbfunc[a[1]][(pair.n, a)] += n
		for (label, indices), parent in pair.pgbrack:
			self.goldbatt.add(((pair.n, label, indices), parent))
		for (label, indices), parent in pair.pcbrack:
			self.candbatt.add(((pair.n, label, indices), parent))
		self.goldrule.update((pair.n, indices, rule)
				for indices, rule in pair.grule.elements())
		self.candrule.update((pair.n, indices, rule)
				for indices, rule in pair.crule.elements())

	def scores(self):
		"""Return a dictionary with running scores for all added sentences."""
		return dict(lr=nozerodiv(lambda: recall(self.goldb, self.candb)),
				lp=nozerodiv(lambda: precision(self.goldb, self.candb)),
				lf=nozerodiv(lambda: f_measure(self.goldb, self.candb)),
				ex=nozerodiv(lambda: self.exact / self.sentcount),
				tag=nozerodiv(lambda: accuracy(self.goldpos, self.candpos)),
				fun=nozerodiv(lambda: f_measure(self.goldfun, self.candfun)))


def main():
	"""Command line interface for evaluation."""
	flags = {'help', 'verbose', 'debug', 'disconly', 'ted', 'la'}
	options = {'goldenc=', 'parsesenc=', 'goldfmt=', 'parsesfmt=', 'fmt=',
			'cutofflen=', 'headrules=', 'functions=', 'morphology='}
	try:
		opts, args = gnu_getopt(sys.argv[2:], 'h', flags | options)
	except GetoptError as err:
		print('error:', err, file=sys.stderr)
		print(SHORTUSAGE)
		sys.exit(2)
	opts = dict(opts)
	if len(args) < 2 or len(args) > 3:
		print('error: Wrong number of arguments.', file=sys.stderr)
		print(SHORTUSAGE)
		sys.exit(2)
	goldfile, parsesfile = args[:2]
	param = readparam(args[2] if len(args) == 3 else None)
	param['CUTOFF_LEN'] = int(opts.get('--cutofflen', param['CUTOFF_LEN']))
	param['DISC_ONLY'] = '--disconly' in opts
	param['DEBUG'] = max(param['DEBUG'],
			'--verbose' in opts, 2 * ('--debug' in opts))
	param['TED'] |= '--ted' in opts
	param['LA'] |= '--la' in opts
	param['DEP'] = '--headrules' in opts
	if '--fmt' in opts:
		opts['--goldfmt'] = opts['--parsesfmt'] = opts['--fmt']
	goldreader = READERS[opts.get('--goldfmt', 'export')]
	parsesreader = READERS[opts.get('--parsesfmt', 'export')]
	gold = goldreader(goldfile,
			encoding=opts.get('--goldenc', 'utf8'),
			functions=opts.get('--functions', 'remove'),
			morphology=opts.get('--morphology'),
			headrules=opts.get('--headrules'))
	parses = parsesreader(parsesfile,
			encoding=opts.get('--parsesenc', 'utf8'),
			functions=opts.get('--functions', 'remove'),
			morphology=opts.get('--morphology'),
			headrules=opts.get('--headrules'))
	goldtrees, goldsents = gold.trees(), gold.sents()
	candtrees, candsents = parses.trees(), parses.sents()
	if not goldtrees:
		raise ValueError('no trees in gold file')
	if not candtrees:
		raise ValueError('no trees in parses file')
	if param['DEBUG'] >= 2:
		print('gold:', goldfile)
		print('parses:', parsesfile, '\n')
	evaluator = Evaluator(param, max(len(str(key)) for key in candtrees))
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
			'DISC_ONLY', 'LA', 'TED', 'DEP', 'DELETE_ROOT_PRETERMS')
	param = {'DEBUG': 0, 'MAX_ERROR': 10, 'CUTOFF_LEN': 40,
				'LABELED': 1, 'DELETE_LABEL_FOR_LENGTH': set(),
				'DELETE_LABEL': set(), 'DELETE_WORD': set(),
				'EQ_LABEL': set(), 'EQ_WORD': set(),
				'DISC_ONLY': 0, 'LA': 0, 'TED': 0, 'DEP': 0,
				'DELETE_ROOT_PRETERMS': 0}
	seen = set()
	for a in io.open(filename, encoding='utf8') if filename else ():
		line = a.strip()
		if line and not line.startswith('#'):
			key, val = line.split(None, 1)
			if key in validkeysonce:
				if key in seen:
					raise ValueError('cannot declare %s twice' % key)
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


def alignsent(csent, gsent, gpos):
	"""Map tokens of ``csent`` onto those of ``gsent``, and translate indices.

	:returns: a copy of ``gpos`` with indices of ``csent`` as keys,
		but tags from ``gpos``.

	>>> gpos = {0: "``", 1: 'RB', 2: '.', 3: "''"}
	>>> alignsent(['No'], ['``', 'No', '.', "''"], gpos) == {0: 'RB'}
	True"""
	n = m = 0
	result = {}
	while n < len(csent) and m < len(gsent):
		if csent[n] == gsent[m]:
			result[n] = gpos[m]
			n += 1
			m += 1
		else:
			m += 1
	return result


def transform(tree, sent, pos, gpos, param, grootpos):
	"""Apply the transformations according to the parameter file.

	Does not delete the root node, which is a special case because if there is
	more than one child it cannot be deleted.

	:param pos: a list with the contents of tree.pos(); modified in-place.
	:param gpos: a dictionary of the POS tags of the original gold tree, before
		any tags/words have been deleted.
	:param param: the parameters specifying which labels / words to delete
	:param grootpos: the set of indices with preterminals directly under the
		root node of the gold tree.
	:returns: an immutable, transformed copy of ``tree``."""
	leaves = list(range(len(sent)))
	posnodes = []
	for a in reversed(list(tree.subtrees(
			lambda n: n and isinstance(n[0], Tree)))):
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
	# retain words still in tree
	sent[:] = [sent[n] for n in leaves]
	# drop indices from POS tags
	pos[:] = [pos[n][1] for n in leaves]
	# removed POS tags cause the numbering to be off, restore.
	leafmap = {m: n for n, m in enumerate(leaves)}
	for a in posnodes:
		a[0] = leafmap[a[0]]
		if sent[a[0]] in param['EQ_WORD']:
			sent[a[0]] = param['EQ_WORD'][sent[a[0]]]
	return tree.freeze()


def parentedbracketings(tree, labeled=True, dellabel=(), disconly=False):
	"""Return the labeled bracketings with parents for a tree.

	:returns:
		multiset with items of the form ``((label, indices), parentlabel)``
	"""
	return Counter((bracketing(a, labeled), getattr(a.parent, 'label', ''))
			for a in tree.subtrees()
			if a and isinstance(a[0], Tree)  # nonempty, not a preterminal
				and a.label not in dellabel
				and (not disconly or isdisc(a)))


def bracketings(tree, labeled=True, dellabel=(), disconly=False):
	"""Return the labeled set of bracketings for a tree.

	For each nonterminal node, the set will contain a tuple with the label and
	the set of terminals which it dominates.
	``tree`` must have been processed by ``transform()``.
	The argument ``dellabel`` is only used to exclude the ROOT node from the
	results (because it cannot be deleted by ``transform()`` when non-unary).

	>>> tree = Tree('(S (NP 1) (VP (VB 0) (JJ 2)))')
	>>> params = {'DELETE_LABEL': set(), 'DELETE_WORD': set(),
	... 'EQ_LABEL': {}, 'EQ_WORD': {},
	... 'DELETE_ROOT_PRETERMS': 0}
	>>> tree = transform(tree, tree.leaves(), tree.pos(), dict(tree.pos()),
	... params, set())
	>>> for (label, span), cnt in sorted(bracketings(tree).items()):
	...		print(label, bin(span), cnt)
	S 0b111 1
	VP 0b101 1
	>>> tree = Tree('(S (NP 1) (VP (VB 0) (JJ 2)))')
	>>> params['DELETE_LABEL'] = {'VP'}
	>>> tree = transform(tree, tree.leaves(), tree.pos(), dict(tree.pos()),
	... params, set())
	>>> for (label, span), cnt in sorted(bracketings(tree).items()):
	...		print(label, bin(span), cnt)
	S 0b111 1"""
	return Counter(bracketing(a, labeled) for a in tree.subtrees()
			if a and isinstance(a[0], Tree)  # nonempty, not a preterminal
				and a.label not in dellabel and (not disconly or isdisc(a)))


def bracketing(node, labeled=True):
	"""Generate bracketing ``(label, indices)`` for a given node."""
	return (node.label if labeled else '', node.bitset)


def strbracketings(brackets):
	"""Return a string with a concise representation of a bracketing.

	>>> print(strbracketings({('S', 0b111), ('VP', 0b101)}))
	S[0-2], VP[0,2]
	"""
	if not brackets:
		return '{}'
	return ', '.join('%s[%s]' % (a, ','.join(
		'-'.join('%d' % y for y in sorted(set(x)))
		for x in intervals(b))) for a, b in sorted(brackets))


def leafancestorpaths(tree, dellabel):
	"""Generate a list of ancestors for each leaf node in a tree."""
	# uses [] to mark components, and () to mark constituent boundaries
	# deleted words/tags should not affect boundary detection
	paths = {a: [] for a in getbits(tree.bitset)}
	# do a top-down level-order traversal
	thislevel = [tree]
	while thislevel:
		nextlevel = []
		for n in thislevel:
			leaves = list(getbits(n.bitset))
			# skip empty nodes and POS tags
			if not leaves or (not n or not isinstance(n[0], Tree)):
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
	"""Sampson, Babarcz (2002): A test of the leaf-ancestor metric [...].

	http://www.lrec-conf.org/proceedings/lrec2002/pdf/ws20.pdf p. 27;
	2003 journal paper: https://doi.org/10.1017/S1351324903003243"""
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
	if len(reference) != len(candidate):
		raise ValueError('Sequences must have the same length.')
	return Decimal(sum(a == b for a, b in zip(reference, candidate))
			) / len(reference)


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


def intervals(bitset):
	"""Return a sequence of intervals corresponding to contiguous ranges.

	``seq`` is an integer representing a bitvector. An interval is a pair
	``(a, b)``, with ``a <= b`` denoting a contiguous range of one bits ``x``
	in ``seq`` such that ``a <= x <= b``.

	>>> list(intervals(0b111011011))  # NB: read from right to left
	[(0, 1), (3, 4), (6, 8)]"""
	start = prev = None
	for a in getbits(bitset):
		if start is None:
			start = prev = a
		elif a == prev + 1:
			prev = a
		else:
			yield start, prev
			start = prev = a
	if start is not None:
		yield start, prev


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


def pyintbitcount(a):
	"""Return number of set bits (1s) in a Python integer.

	>>> pyintbitcount(0b0011101)
	4"""
	cnt = 0
	while a:
		a &= a - 1
		cnt += 1
	return cnt


__all__ = ['Evaluator', 'TreePairResult', 'EvalAccumulator', 'main',
		'readparam', 'transitiveclosure', 'alignsent', 'transform',
		'parentedbracketings', 'bracketings', 'bracketing', 'strbracketings',
		'leafancestorpaths', 'pathscore', 'leafancestor', 'treedisteval',
		'recall', 'precision', 'f_measure', 'accuracy', 'harmean', 'mean',
		'intervals', 'nozerodiv', 'editdistance', 'pyintbitcount']
