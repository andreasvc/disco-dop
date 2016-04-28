""" Objects for grammars and grammar rules. """
import re
import logging
import numpy as np
from .tree import escape, unescape

# This regex should match exactly the set of valid yield functions,
# i.e., comma-separated strings of alternating occurrences from the set {0,1},
YFBINARY = re.compile(
		r'^(?:0|1|1?(?:01)+|0?(?:10)+)(?:,(?:0|1|1?(?:01)+|0?(?:10)+))*$')
YFUNARYRE = re.compile(r'^0(?:,0)*$')
# Match when non-integral weights are present
LCFRS_NONINT = re.compile('\t[0-9]+[./][0-9]+\n')
BITPAR_NONINT = re.compile('(?:^|\n)[0-9]+\.[0-9]+[ \t]')
LEXICON_NONINT = re.compile('[ \t][0-9]+[./][0-9]+[ \t\n]')
# Detect rule format of bitpar
BITPARRE = re.compile(r'^[-.e0-9]+\b')

# comparison functions for sorting rules on LHS/RHS labels.
cdef int cmp0(const void *p1, const void *p2) nogil:
	cdef ProbRule *a = <ProbRule *>p1
	cdef ProbRule *b = <ProbRule *>p2
	if a.lhs == b.lhs:
		return (a.no > b.no) - (a.no < b.no)
	return (a.lhs > b.lhs) - (a.lhs < b.lhs)
cdef int cmp1(const void *p1, const void *p2) nogil:
	cdef ProbRule *a = <ProbRule *>p1
	cdef ProbRule *b = <ProbRule *>p2
	if a.rhs1 == b.rhs1:
		return (a.prob < b.prob) - (a.prob > b.prob)
	return (a.rhs1 > b.rhs1) - (a.rhs1 < b.rhs1)
cdef int cmp2(const void *p1, const void *p2) nogil:
	cdef ProbRule *a = <ProbRule *>p1
	cdef ProbRule *b = <ProbRule *>p2
	if a.rhs2 == b.rhs2:
		return (a.prob < b.prob) - (a.prob > b.prob)
	return (a.rhs2 > b.rhs2) - (a.rhs2 < b.rhs2)


@cython.final
cdef class Grammar:
	"""A grammar object which stores rules compactly, indexed in various ways.

	:param rule_tuples_or_str: either a sequence of tuples containing both
		phrasal & lexical rules, or a string containing the phrasal
		rules in text format; in the latter case ``lexicon`` should be given.
		The text format allows for more efficient loading and is used
		internally.
	:param start: a string identifying the unique start symbol of this grammar,
		which will be used by default when parsing with this grammar
	:param binarized: whether to require a binarized grammar;
		a non-binarized grammar can only be used by bitpar.

	By default the grammar is in logprob mode;
	invoke ``grammar.switch('default', logprob=False)`` to switch.
	If the grammar only contains integral weights (frequencies), they will
	be normalized into relative frequencies; if the grammar contains any
	non-integral weights, weights will be left unchanged."""
	def __cinit__(self):
		self.fanout = self.unary = self.mapping = self.splitmapping = NULL

	def __init__(self, rule_tuples_or_str, lexicon=None, start='ROOT',
			binarized=True):
		cdef LexicalRule lexrule
		cdef double [:] weights
		cdef int n
		self.mapping = self.splitmapping = self.bylhs = NULL
		self.start = start
		self.binarized = binarized
		self.numunary = self.numbinary = self.currentmodel = 0
		self.modelnames = ['default']
		self.logprob = False

		if rule_tuples_or_str and isinstance(rule_tuples_or_str, str):
			if not isinstance(lexicon, str):
				raise ValueError('expected lexicon argument.')
			self.origrules = rule_tuples_or_str
			self.origlexicon = lexicon
			self.bitpar = BITPARRE.match(self.origrules)
		elif rule_tuples_or_str and isinstance(
				rule_tuples_or_str[0], tuple):
			# convert tuples to strings with text format
			from .grammar import writegrammar
			self.origrules, self.origlexicon = writegrammar(
					rule_tuples_or_str, bitpar=False)
			self.bitpar = False
		else:
			raise ValueError(
					'expected non-empty sequence of tuples or unicode string.'
					'got: %r' % type(rule_tuples_or_str))

		# collect non-terminal labels; count number of rules in each category
		# for allocation purposes.
		rulelines = self.origrules.splitlines()
		fanoutdict = self._countrules(rulelines)
		self._convertlexicon(fanoutdict)
		self.tolabel = sorted(self.toid, key=self.toid.get)
		self.nonterminals = len(self.toid)
		self._allocate()
		self._convertrules(rulelines, fanoutdict)
		for n in range(self.nonterminals):
			self.fanout[n] = fanoutdict[self.tolabel[n]]
		del rulelines, fanoutdict
		# index & filter phrasal rules in different ways
		self._indexrules(self.bylhs, 0, 0)
		self._indexrules(self.unary, 1, 2)
		self._indexrules(self.lbinary, 1, 3)
		self._indexrules(self.rbinary, 2, 3)
		# indexing requires sorting; this map gives the new index
		# given an original rule number (useful with the rulestr method).
		for n in range(self.numrules):
			self.revmap[self.bylhs[0][n].no] = n
		# if the grammar only contains integral values (frequencies),
		# normalize them into relative frequencies.
		nonint = BITPAR_NONINT if self.bitpar else LCFRS_NONINT
		if not nonint.search(self.origrules):
			# 	or LEXICON_NONINT.search(self.origlexicon)):
			self._normalize()
		# store 'default' weights
		weights = self.models[0]
		for n in range(self.numrules):
			weights[self.bylhs[0][n].no] = self.bylhs[0][n].prob
		for n, lexrule in enumerate(self.lexical, self.numrules):
			weights[n] = lexrule.prob
		self.switch(u'default', True)  # enable log probabilities

	@cython.wraparound(True)
	def _countrules(self, list rulelines):
		"""Count unary & binary rules; make a canonical list of all
		non-terminal labels and assign them unique IDs."""
		cdef int numother = 0
		Epsilon = 'Epsilon'
		# Epsilon gets ID 0, only occurs implicitly in RHS of lexical rules.
		self.toid = {Epsilon: 0}
		fanoutdict = {Epsilon: 0}  # temporary mapping of labels to fan-outs
		for line in rulelines:
			if not line.strip():
				continue
			fields = line.split()
			if self.bitpar:
				rule = fields[1:]
				yf = '0' if len(rule) == 2 else '01'
			else:
				rule = fields[:-2]
				yf = fields[-2]
			if Epsilon in rule:
				raise ValueError('Epsilon symbol may only occur '
						'in RHS of lexical rules.')
			if self.start in rule[1:]:
				raise ValueError('Start symbol should only occur on LHS.')
			if len(rule) == 2:
				if not YFUNARYRE.match(yf):
					raise ValueError('yield function refers to non-existent '
							'second non-terminal: %r\t%r' % (rule, yf))
				self.numunary += 1
			elif len(rule) == 3:
				if not YFBINARY.match(yf):
					raise ValueError('illegal yield function: %s' % yf)
				if '0' not in yf or '1' not in yf:
					raise ValueError('mismatch between non-terminals and '
							'yield function: %r\t%r' % (rule, yf))
				self.numbinary += 1
			elif self.binarized:
				raise ValueError('grammar not binarized:\n%s' % line)
			else:
				numother += 1
			if rule[0] not in self.toid:
				self.toid[rule[0]] = len(self.toid)
				fanout = yf.count(',') + 1
				fanoutdict[rule[0]] = fanout
				if fanout > self.maxfanout:
					self.maxfanout = fanout

		if self.start not in self.toid:
			raise ValueError('Start symbol %r not in set of non-terminal '
					'labels extracted from grammar rules.' % self.start)
		self.numrules = self.numunary + self.numbinary + numother
		self.phrasalnonterminals = len(self.toid)
		if not self.numrules:
			raise ValueError('no rules found')
		return fanoutdict

	def _convertlexicon(self, fanoutdict):
		""" Make objects for lexical rules. """
		cdef int x
		cdef double w
		self.lexical = []
		self.lexicalbyword = {}
		self.lexicalbylhs = {}
		for line in self.origlexicon.splitlines():
			if not line.strip():
				continue
			x = line.index('\t')
			word = escape(line[:x])
			fields = line[x + 1:].split()
			if word in self.lexicalbyword:
				raise ValueError('word %r appears more than once '
						'in lexicon file' % unescape(word))
			self.lexicalbyword[word] = []
			for tag, weight in zip(fields[::2], fields[1::2]):
				if tag not in self.toid:
					self.toid[tag] = len(self.toid)
					fanoutdict[tag] = 1
					# disabled because we add ids for labels on the fly:
					# logging.warning('POS tag %r for word %r '
					# 		'not used in any phrasal rule', tag, word)
					# continue
				else:
					if fanoutdict[tag] != 1:
						raise ValueError('POS tag %r has fan-out %d, '
								'may only be 1.' % (fanoutdict[tag], tag))
				w = convertweight(weight.encode('ascii'))
				if w <= 0:
					raise ValueError('weights should be positive '
							'and non-zero:\n%r' % line)
				lexrule = LexicalRule(self.toid[tag], word, w)
				if lexrule.lhs not in self.lexicalbylhs:
					self.lexicalbylhs[lexrule.lhs] = {}
				self.lexical.append(lexrule)
				self.lexicalbyword[word].append(lexrule)
				self.lexicalbylhs[lexrule.lhs][word] = lexrule
			if not (self.lexical and self.lexicalbyword and self.lexicalbylhs):
				raise ValueError('no lexical rules found.')

	def _allocate(self):
		"""Allocate memory to store rules."""
		# store all non-lexical rules in a contiguous array
		# the other arrays will contain pointers to relevant parts thereof
		# (indexed on lhs, rhs1, and rhs2 of rules)
		self.bylhs = <ProbRule **>malloc(sizeof(ProbRule *)
				* self.nonterminals * 4)
		if self.bylhs is NULL:
			raise MemoryError('allocation error')
		self.bylhs[0] = NULL
		self.unary = &(self.bylhs[1 * self.nonterminals])
		self.lbinary = &(self.bylhs[2 * self.nonterminals])
		self.rbinary = &(self.bylhs[3 * self.nonterminals])
		# allocate the actual contiguous array that will contain the rules
		# (plus sentinels)
		self.bylhs[0] = <ProbRule *>malloc(sizeof(ProbRule) *
			(self.numrules + (2 * self.numbinary) + self.numunary + 4))
		if self.bylhs[0] is NULL:
			raise MemoryError('allocation error')
		self.unary[0] = &(self.bylhs[0][self.numrules + 1])
		self.lbinary[0] = &(self.unary[0][self.numunary + 1])
		self.rbinary[0] = &(self.lbinary[0][self.numbinary + 1])
		self.fanout = <uint8_t *>malloc(sizeof(uint8_t) * self.nonterminals)
		if self.fanout is NULL:
			raise MemoryError('allocation error')
		self.models = np.empty(
				(1, self.numrules + len(self.lexical)), dtype='d')
		self.mask = <uint64_t *>malloc(
				BITNSLOTS(self.numrules) * sizeof(uint64_t))
		if self.mask is NULL:
			raise MemoryError('allocation error')
		self.setmask(None)
		self.revmap = <uint32_t *>malloc(self.numrules * sizeof(uint32_t))
		if self.revmap is NULL:
			raise MemoryError('allocation error')

	@cython.wraparound(True)
	cdef _convertrules(Grammar self, list rulelines, dict fanoutdict):
		"""Auxiliary function to create Grammar objects. Copies grammar
		rules from a text file to a contiguous array of structs."""
		cdef uint32_t n = 0, m
		cdef double w
		cdef ProbRule *cur
		self.rulenos = {}
		for line in rulelines:
			if not line.strip():
				continue
			fields = line.split()
			if self.bitpar:
				rule = fields[1:]
				# NB: this is wrong when len(rule) > 10
				yf = ''.join(map(str, range(len(rule) - 1)))
				weight = fields[0]
			else:
				rule = fields[:-2]
				yf = fields[-2]
				weight = fields[-1]
			# check whether RHS labels have been seen as LHS and check fanout
			for m, nt in enumerate(rule):
				if nt not in self.toid:
					raise ValueError('symbol %r has not been seen as LHS '
						'in any rule: %s' % (nt, line))
				if self.binarized:
					fanout = yf.count(',01'[m]) + (m == 0)
					if fanoutdict[nt] != fanout:
						raise ValueError("conflicting fanouts for symbol "
							"'%s'.\nprevious: %d; this non-terminal: %d.\n"
							"yf: %s; rule: %s" % (
							nt, fanoutdict[nt], fanout, yf, line))
			w = convertweight(weight.encode('ascii'))
			if w <= 0:
				raise ValueError('weights should be positive and non-zero:\n%r'
						% line)
			# n is the rule index in the array, and will be the ID for the rule
			cur = &(self.bylhs[0][n])
			cur.no = n
			cur.lhs = self.toid[rule[0]]
			cur.rhs1 = 0 if len(rule) > 3 else self.toid[rule[1]]
			cur.rhs2 = 0 if len(rule) == 2 else self.toid[rule[2]]
			cur.prob = w
			cur.lengths = cur.args = m = 0
			for a in yf:
				if a == ',':
					cur.lengths |= 1 << (m - 1)
					continue
				elif a == '1':
					cur.args += 1 << m
				elif a != '0' and self.binarized:
					raise ValueError('expected: %r; got: %r' % ('0', a))
				m += 1
			cur.lengths |= 1 << (m - 1)
			if self.binarized and m >= (8 * sizeof(cur.args)):
				raise ValueError('Parsing complexity (%d) too high (max %d).\n'
						'Rule: %s' % (m, (8 * sizeof(cur.args)), line))
			self.rulenos[yf + ' ' + ' '.join(rule)] = n
			n += 1
		assert n == self.numrules, (n, self.numrules)

	def _normalize(self):
		"""Optionally normalize frequencies to relative frequencies.
		Should be run during initialization."""
		cdef double mass = 0
		cdef uint32_t n = 0, lhs
		cdef LexicalRule lexrule
		for lhs in range(self.nonterminals):
			mass = 0
			n = 0
			while self.bylhs[lhs][n].lhs == lhs:
				mass += self.bylhs[lhs][n].prob
				n += 1
			for lexrule in self.lexicalbylhs.get(lhs, {}).values():
				mass += lexrule.prob
			n = 0
			while self.bylhs[lhs][n].lhs == lhs:
				self.bylhs[lhs][n].prob /= mass
				n += 1
			for lexrule in self.lexicalbylhs.get(lhs, {}).values():
				lexrule.prob /= mass

	cdef _indexrules(Grammar self, ProbRule **dest, int idx, int filterlen):
		"""Auxiliary function to create Grammar objects. Copies certain
		grammar rules and sorts them on the given index.
		Resulting array is ordered by lhs, rhs1, or rhs2 depending on the value
		of `idx` (0, 1, or 2); filterlen can be 0, 2, or 3 to get all, only
		unary, or only binary rules, respectively.
		A separate array has a pointer for each non-terminal into this array;
		e.g.: dest[NP][0] == the first rule with an NP in the idx position."""
		cdef uint32_t prev = self.nonterminals, idxlabel = 0, n, m = 0
		cdef ProbRule *cur
		# need to set dest even when there are no rules for that idx
		for n in range(1, self.nonterminals):
			dest[n] = dest[0]
		if dest is self.bylhs:
			m = self.numrules
		else:
			for n in range(self.numrules):
				if ((filterlen == 2 and self.bylhs[0][n].rhs2 == 0)
						or (filterlen == 3 and self.bylhs[0][n].rhs1
						and self.bylhs[0][n].rhs2)):
					# copy this rule
					dest[0][m] = self.bylhs[0][n]
					assert dest[0][m].no < self.numrules
					m += 1
		if filterlen == 2:
			assert m == self.numunary, (m, self.numunary)
		elif filterlen == 3:
			assert m == self.numbinary, (m, self.numbinary)
		# sort rules by idx (NB: qsort is not stable, use appropriate cmp func)
		if idx == 0:
			qsort(dest[0], m, sizeof(ProbRule), &cmp0)
		elif idx == 1:
			qsort(dest[0], m, sizeof(ProbRule), &cmp1)
		elif idx == 2:
			qsort(dest[0], m, sizeof(ProbRule), &cmp2)
		# make index: dest[NP] points to first rule with NP in index position
		for n in range(m):
			cur = &(dest[0][n])
			if idx == 0:
				idxlabel = cur.lhs
			elif idx == 1:
				idxlabel = cur.rhs1
			elif idx == 2:
				idxlabel = cur.rhs2
			if idxlabel != prev:
				dest[idxlabel] = cur
			prev = idxlabel
			assert cur.no < self.numrules
		# sentinel rule
		dest[0][m].lhs = dest[0][m].rhs1 = dest[0][m].rhs2 = self.nonterminals

	def register(self, name, weights):
		"""Register a probabilistic model given a name and a sequence of
		floats ``weights``, with weights in the same order as
		``self.origrules`` and ``self.origlexicon`` (which is an arbitrary
		order except that tags for each word are clustered together)."""
		cdef int n, m = len(self.modelnames)
		cdef double [:] tmp
		assert len(self.modelnames) <= 255, (
				'256 probabilistic models should be enough for anyone.')
		if name in self.modelnames:
			raise ValueError('model %r already exists' % name)
		if len(weights) != self.numrules + len(self.lexical):
			raise ValueError('length mismatch: %d grammar rules, '
					'%d weights given.' % (
					self.numrules + len(self.lexical), len(weights)))
		self.models.resize(m + 1, self.numrules + len(self.lexical))
		self.modelnames.append(name)
		tmp = self.models[m]
		for n in range(self.numrules + len(self.lexical)):
			tmp[n] = weights[n]

	def switch(self, str name, bint logprob=True):
		"""Switch to a different probabilistic model;
		use u'default' to swith back to model given during initialization."""
		cdef int n, m = self.modelnames.index(name)
		cdef double [:] tmp
		cdef LexicalRule lexrule
		if self.currentmodel == m and self.logprob == logprob:
			return
		tmp = -np.log(self.models[m]) if logprob else self.models[m]
		for n in range(self.numrules):
			self.bylhs[0][n].prob = tmp[self.bylhs[0][n].no]
		for n in range(self.numbinary):
			self.lbinary[0][n].prob = tmp[self.lbinary[0][n].no]
			self.rbinary[0][n].prob = tmp[self.rbinary[0][n].no]
		for n in range(self.numunary):
			self.unary[0][n].prob = tmp[self.unary[0][n].no]
		for n, lexrule in enumerate(self.lexical, self.numrules):
			lexrule.prob = tmp[n]
		self.logprob = logprob
		self.currentmodel = m

	def setmask(self, seq):
		"""Given a sequence of rule numbers, store a mask so that any phrasal
		rules not in the sequence are deactivated. If sequence is None, the
		mask is cleared."""
		cdef int n
		# zero-bit = not blocked or out of range; 1-bit = blocked.
		if seq is None:
			memset(<void *>self.mask, 0,
					BITNSLOTS(self.numrules) * sizeof(uint64_t))
			return
		memset(<void *>self.mask, 255,
				BITNSLOTS(self.numrules) * sizeof(uint64_t))
		for n in seq:
			CLEARBIT(self.mask, n)
		# clear out-of-range bits: 000011111 <-- 1-bits up to numrules.
		self.mask[BITSLOT(self.numrules)] = BITMASK(self.numrules) - 1UL

	def buildchainvec(self):
		"""Build a boolean matrix representing the unary (chain) rules."""
		cdef uint32_t n
		cdef ProbRule *rule
		self.chainvec = <uint64_t *>calloc(self.nonterminals
				* BITNSLOTS(self.nonterminals), sizeof(uint64_t))
		if self.chainvec is NULL:
			raise MemoryError('allocation error')
		for n in range(self.numunary):
			rule = self.unary[n]
			SETBIT(self.chainvec, rule.rhs1 * self.nonterminals + rule.lhs)

	def testgrammar(self, epsilon=np.finfo(np.double).eps):  # machine epsilon
		"""Test whether all left-hand sides sum to 1 +/-epsilon for the
		currently selected weights."""
		cdef ProbRule *rule
		cdef LexicalRule lexrule
		cdef uint32_t n, maxlabel = 0
		cdef list weights = [[] for _ in self.toid]
		cdef double [:] tmp = self.models[self.currentmodel, :]
		# We could be strict about separating POS tags and phrasal categories,
		# but Negra contains at least one tag (--) used for both.
		for n in range(self.numrules):
			rule = &(self.bylhs[0][n])
			weights[rule.lhs].append(tmp[rule.no])
		for n, lexrule in enumerate(self.lexical, self.numrules):
			weights[lexrule.lhs].append(tmp[n])
		maxdiff = epsilon
		for lhs, lhsweights in enumerate(weights[1:], 1):
			mass = fsum(lhsweights)
			if abs(mass - 1.0) > maxdiff:
				maxdiff = abs(mass - 1.0)
				maxlabel = lhs
		if maxdiff > epsilon:
			msg = ('Weights do not sum to 1 +/- %g.\n'
					'Largest difference with rules for LHS \'%s\': '
					'sum = %g; diff = %g' % (
					epsilon, self.tolabel[maxlabel],
					fsum(weights[maxlabel]), maxdiff))
			return False, msg
		return True, 'All left hand sides sum to 1 +/- epsilon=%s' % epsilon

	def getmapping(Grammar self, Grammar coarse, striplabelre=None,
			neverblockre=None, bint splitprune=False, bint markorigin=False,
			dict mapping=None):
		"""Construct mapping of this grammar's non-terminal labels to another.

		:param coarse: the grammar to which this grammar's labels will be
			mapped. May be ``None``; useful when ``neverblockre`` needs to be
			applied.
		:param striplabelre: if not None, a compiled regex used to form
			the coarse label for a given fine label. This regex is applied
			with a substitution to the empty string.
		:param neverblockre: labels that match this regex will never be pruned.
			Also used to identify auxiliary labels of Double-DOP grammars.

			- use ``|<`` to ignore nodes introduced by binarization;
				useful if coarse and fine stages employ different kinds of
				markovization; e.g., ``NP`` and ``VP`` may be blocked,
				but not ``NP|<DT-NN>``.
			- ``_[0-9]+`` to ignore discontinuous nodes ``X_n`` where ``X`` is
				a label and *n* is a fanout.

		:param mapping: a dictionary with strings of fine labels mapped to
			coarse labels. striplabelre, if given, is applied first.

		The regexes should be compiled objects, i.e., ``re.compile(regex)``,
		or ``None`` to leave labels unchanged.
		"""
		cdef int n, m, components = 0
		cdef set seen = {0}
		if coarse is None:
			coarse = self
		if self.mapping is not NULL:
			free(self.mapping)
		self.mapping = <uint32_t *>malloc(sizeof(uint32_t) * self.nonterminals)
		if splitprune and markorigin:
			if self.splitmapping is not NULL:
				if self.splitmapping[0] is not NULL:
					free(self.splitmapping[0])
				free(self.splitmapping)
			self.splitmapping = <uint32_t **>malloc(sizeof(uint32_t *)
					* self.nonterminals)
			for n in range(self.nonterminals):
				self.splitmapping[n] = NULL
			self.splitmapping[0] = <uint32_t *>malloc(sizeof(uint32_t) *
				sum([self.fanout[n] for n in range(self.nonterminals)
					if self.fanout[n] > 1]))
		for n in range(self.nonterminals):
			if not neverblockre or neverblockre.search(self.tolabel[n]) is None:
				strlabel = self.tolabel[n]
				if striplabelre is not None:
					strlabel = striplabelre.sub('', strlabel, 1)
				if mapping is not None:
					strlabel = mapping[strlabel]
				if self.fanout[n] > 1 and splitprune:
					strlabel += '*'
				if self.fanout[n] > 1 and splitprune and markorigin:
					self.mapping[n] = self.nonterminals  # sentinel value
					self.splitmapping[n] = &(self.splitmapping[0][components])
					components += self.fanout[n]
					for m in range(self.fanout[n]):
						self.splitmapping[n][m] = coarse.toid[
								strlabel + str(m)]
						seen.add(self.splitmapping[n][m])
				else:
					self.mapping[n] = coarse.toid[strlabel]
			else:
				self.mapping[n] = 0
		if seen == set(range(coarse.nonterminals)):
			msg = 'label sets are equal'
		else:
			# NB: ALL fine symbols are mapped to some coarse symbol;
			# we only check if all coarse symbols have received a mapping.
			l = sorted([coarse.tolabel[a] for a in
					set(range(coarse.nonterminals)) - seen])
			diff = ', '.join(l[:10]) + (', ...' if len(l) > 10 else '')
			if coarse.nonterminals > self.nonterminals:
				msg = ('grammar is not a superset of coarse grammar:\n'
						'coarse labels without mapping: { %s }' % diff)
			elif coarse.nonterminals < self.nonterminals:
				msg = 'grammar is a proper superset of coarse grammar.'
			else:
				msg = ('equal number of nodes, but not equivalent:\n'
						'coarse labels without mapping: { %s }' % diff)
		return msg

	def getrulemapping(Grammar self, Grammar coarse, striplabelre):
		"""Produce a mapping of coarse rules to sets of fine rules.

		A coarse rule for a given fine rule is found by applying the regex
		``striplabelre`` to labels. NB: this regex is applied to strings with
		multiple non-terminal labels at once, it should not match on the end of
		string ``$``. The mapping uses the rule numbers (``rule.no``) derived
		from the original order of the rules when the Grammar object was
		created; e.g., ``self.rulemapping[12] == [34, 56, 78, ...]``
		where 12 refers to a rule in the given coarse grammar, and the other
		IDs to rules in this grammar."""
		cdef int n
		cdef ProbRule *rule
		self.rulemapping = [[] for _ in range(coarse.numrules)]
		for n in range(self.numrules):
			rule = &(self.bylhs[0][n])
			key = '%s %s %s' % (self.yfstr(rule[0]),
					self.tolabel[rule.lhs], self.tolabel[rule.rhs1])
			if rule.rhs2:
				key += ' ' + self.tolabel[rule.rhs2]
			key = striplabelre.sub('', key)
			self.rulemapping[coarse.rulenos[key]].append(rule.no)

	cpdef rulestr(self, int n):
		"""Return a string representation of a specific rule in this grammar."""
		cdef ProbRule rule
		if not 0 <= n < self.numrules:
			raise ValueError('Out of range: %s' % n)
		rule = self.bylhs[0][n]
		left = '%.2f %s => %s%s' % (
			exp(-rule.prob) if self.logprob else rule.prob,
			self.tolabel[rule.lhs],
			self.tolabel[rule.rhs1],
			' %s' % self.tolabel[rule.rhs2]
				if rule.rhs2 else '')
		return '%s %s [%d]' % (left.ljust(40), self.yfstr(rule), rule.no)

	cdef yfstr(self, ProbRule rule):
		cdef int n, m = 0
		cdef str result = ''
		for n in range(8 * sizeof(rule.args)):
			result += '1' if (rule.args >> n) & 1 else '0'
			if (rule.lengths >> n) & 1:
				m += 1
				if m == self.fanout[rule.lhs]:
					return result
				else:
					result += ','
		raise ValueError('illegal yield function expected %d components.\n'
				'args: %s; lengths: %s' % (self.fanout[rule.lhs],
				bin(rule.args), bin(rule.lengths)))

	def __str__(self):
		rules = '\n'.join(filter(None,
			[self.rulestr(n) for n in range(self.numrules)]))
		lexical = '\n'.join(['%.2f %s => %s' % (
				exp(-lexrule.prob) if self.logprob else lexrule.prob,
				self.tolabel[lexrule.lhs],
				lexrule.word.encode('unicode-escape').decode('ascii'))
			for word in sorted(self.lexicalbyword)
			for lexrule in sorted(self.lexicalbyword[word],
			key=lambda lexrule: (<LexicalRule>lexrule).lhs)])
		labels = ', '.join(['%s=%d' % (a, b)
				for a, b in sorted(self.toid.items())])
		return 'rules:\n%s\nlexicon:\n%s\nlabels:\n%s' % (
				rules, lexical, labels)

	def __repr__(self):
		return '%s(\n%s,\n%s\n)' % (self.__class__.__name__,
				self.origrules, self.origlexicon)

	def __reduce__(self):
		"""Helper function for pickling."""
		return (Grammar, (self.origrules, self.origlexicon,
				self.start, self.binarized))

	def __dealloc__(self):
		if self.bylhs is NULL:
			return
		free(self.bylhs[0])
		free(self.bylhs)
		free(self.fanout)
		free(self.mask)
		free(self.revmap)
		if self.chainvec is not NULL:
			free(self.chainvec)
		if self.mapping is not NULL:
			free(self.mapping)
		if self.splitmapping is not NULL:
			free(self.splitmapping[0])
			free(self.splitmapping)
		self.bylhs = self.fanout = self.mask = self.revmap = NULL
		self.chainvec = self.mapping = self.splitmapping = NULL


cdef inline double convertweight(const char *weight):
	"""Convert weight to double; weight may be a fraction '1/2',
	decimal float '0.5' or hex float '0x1.0p-1'. Returns 0 on error."""
	cdef char *endptr = NULL
	cdef double w = strtod(weight, &endptr)
	if endptr[0] == b'/':
		w /= strtod(&endptr[1], NULL)
	elif endptr[0]:
		return 0
	return w
