import codecs, logging
from math import exp
from collections import Set, Iterable
from functools import partial
from nltk import Tree

DEF SLOTS = 2
#maxbitveclen = sizeof(ULLong) * 8
maxbitveclen = SLOTS * sizeof(ULong) * 8

cdef class Grammar:
	def __cinit__(self):
		self.fanout = self.unary = self.mapping = self.splitmapping = NULL
	def __init__(self, grammar):
		""" Turn a sequence of grammar rules into various lookup tables,
		mapping nonterminal labels to numeric identifiers. Also negates
		log-probabilities to accommodate min-heaps. Can only represent monotone
		LCFRS rules; i.e., the components of the yield that are covered by a
		non-terminal are ordered from left to right. """
		self.origrules = frozenset(grammar)
		# get a list of all nonterminals; make sure Epsilon and ROOT are first,
		# and assign them unique IDs
		# convert them to ASCII strings.
		nonterminals = list(enumerate(["Epsilon", "ROOT"]
			+ sorted(set(str(nt) for (rule, _), _ in grammar for nt in rule)
				- set(["Epsilon", "ROOT"]))))
		self.nonterminals = len(nonterminals)
		self.numrules = sum([1 for (r, _), _ in grammar if r[1] != 'Epsilon'])
		self.toid = dict((lhs, n) for n, lhs in nonterminals)
		self.tolabel = dict((n, lhs) for n, lhs in nonterminals)
		self.lexical = {}
		self.lexicalbylhs = {}
		self.mapping = self.splitmapping = NULL


		# the strategy is to lay out all non-lexical rules in a contiguous array
		# these arrays will contain pointers to relevant parts thereof
		# (one index per nonterminal)
		self.unary = <Rule **>malloc(sizeof(Rule *) * self.nonterminals * 4)
		assert self.unary is not NULL
		self.unary[0] = NULL
		self.lbinary = &(self.unary[1 * self.nonterminals])
		self.rbinary = &(self.unary[2 * self.nonterminals])
		self.bylhs = &(self.unary[3 * self.nonterminals])
		self.fanout = <UChar *>malloc(sizeof(UChar) * self.nonterminals)
		for n in range(self.nonterminals): self.fanout[n] = 0

		# count number of rules in each category for allocation purposes
		unary_len = binary_len = 0
		for (rule, yf), w in grammar:
			if len(rule) == 2:
				if rule[1] != 'Epsilon':
					assert all(b == 0 for a in yf for b in a), (
						"yield function refers to non-existent second "
						"non-terminal: %r\t%r" % (rule, yf))
					unary_len += 1
			elif len(rule) == 3:
				assert all(b == 0 or b == 1 for a in yf for b in a), (
					"grammar must be binarized")
				assert any(b == 0 for a in yf for b in a), (
					"mismatch between non-terminals "
					"and yield function: %r\t%r" % (rule, yf))
				assert any(b == 1 for a in yf for b in a), (
					"mismatch between non-terminals "
					"and yield function: %r\t%r" % (rule, yf))
				binary_len += 1
			else:
				raise ValueError("grammar not binarized: %s" % repr((rule,yf,w)))
			if self.fanout[self.toid[rule[0]]] == 0:
				self.fanout[self.toid[rule[0]]] = len(yf)
			else:
				assert self.fanout[self.toid[rule[0]]] == len(yf), (
					"conflicting fanouts for symbol '%s'.\n"
					"previous: %d; this non-terminal: %d.\nrule: %r" % (
					rule[0], self.fanout[self.toid[rule[0]]], len(yf), rule))
		#'\n'.join(repr(r) for r in grammar if r[0][0][0] == rule[0])
		bylhs_len = unary_len + binary_len
		# allocate the actual contiguous array that will contain the rules
		# (plus sentinels)
		self.unary[0] = <Rule *>malloc(sizeof(Rule) *
			(unary_len + bylhs_len + (2 * binary_len) + 4))
		assert self.unary is not NULL
		self.lbinary[0] = &(self.unary[0][unary_len + 1])
		self.rbinary[0] = &(self.lbinary[0][binary_len + 1])
		self.bylhs[0] = &(self.rbinary[0][binary_len + 1])

		# convert rules and copy to structs / cdef class
		# remove sign from log probabilities because we use a min-heap
		for (rule, yf), w in grammar:
			if len(rule) == 2 and self.toid[rule[1]] == 0:
				lr = LexicalRule(self.toid[rule[0]], self.toid[rule[1]], 0,
					unicode(yf[0]), abs(w))
				# lexical productions (mis)use the field for the yield function
				# to store the word
				self.lexical.setdefault(yf[0], []).append(lr)
				self.lexicalbylhs.setdefault(lr.lhs, []).append(lr)
		copyrules(grammar, self.unary,   1, 2, self.toid, self.nonterminals)
		copyrules(grammar, self.lbinary, 1, 3, self.toid, self.nonterminals)
		copyrules(grammar, self.rbinary, 2, 3, self.toid, self.nonterminals)
		copyrules(grammar, self.bylhs,   0, 0, self.toid, self.nonterminals)
	def testgrammar(self, epsilon=0.01):
		""" report whether all left-hand sides sum to 1 +/-epsilon. """
		#We could be strict about separating POS tags and phrasal categories,
		#but Negra contains at least one tag (--) used for both.
		cdef size_t lhs, n
		cdef LexicalRule term
		sums = {}
		for lhs from 1 <= lhs < self.nonterminals:
			n = 0
			while self.bylhs[lhs][n].lhs == lhs:
				sums[lhs] = sums.get(lhs, 0.0) + exp(-self.bylhs[lhs][n].prob)
				n += 1
		for terminals in self.lexical.itervalues():
			for term in terminals:
				sums[term.lhs] = sums.get(term.lhs, 0.0) + exp(-term.prob)
		for lhs, mass in sums.iteritems():
			if abs(mass - 1.0) > epsilon:
				#logging.info("rules with %s:\n%s" % (
				#	self.tolabel[lhs], self.rulesrepr(lhs)))
				logging.error("Does not sum to 1 (+/- %g): %s; sums to %g" % (
					epsilon, self.tolabel[lhs], mass))
				return False
		logging.info("All left hand sides sum to 1")
		return True
	def getunaryclosure(self):
		""" FIXME: closure should be related to probabilities as well.
		Also, there appears to be an infinite loop here. """
		cdef size_t i = 0, n
		closure = [set() for n in range(self.nonterminals)]
		candidates = [set() for n in range(self.nonterminals)]
		self.unaryclosure = [[] for n in range(self.nonterminals)]
		while self.unary[0][i].lhs != self.nonterminals:
			candidates[self.unary[0][i].rhs1].add(i)
			i += 1
		for n in range(self.nonterminals):
			while candidates[n]:
				i = candidates[n].pop()
				m = self.unary[0][i].lhs
				if i not in closure[n]:
					self.unaryclosure[n].append(i)
					closure[n].add(i)
				for x in self.unaryclosure[m]:
					if x not in closure[n]: self.unaryclosure[n].append(x)
				closure[n] |= closure[m]
				candidates[n] |= candidates[m]
				candidates[n] -= closure[n]
	def printclosure(self):
		if self.unaryclosure is None:
			print "not computed."
			return
		for m, a in enumerate(self.unaryclosure):
			print '%s[%d] ' % (self.tolabel[m], m),
			for n in a:
				print "%s <= %s  " % (self.tolabel[self.unary[0][n].rhs1],
						self.tolabel[self.unary[0][n].lhs]),
			print
	cpdef getmapping(Grammar self, striplabelre, neverblockre, Grammar coarse,
			bint splitprune, bint markorigin, bint debug=False):
		""" Construct a mapping of fine non-terminal IDs to coarse non-terminal
		IDS, by applying a regex to the labels, used for coarse-to-fine
		parsing. A secondary regex is for items that should never be pruned.
		The regexes should be compiled objects, i.e., re.compile(regex),
		or None to leave labels unchanged.
        - use "|<" to ignore nodes introduced by binarization;
            useful if coarse and fine stages employ different kinds of
            markovization; e.g., NP and VP may be blocked, but not NP|<DT-NN>.
        - "_[0-9]+" to ignore nodes X_n where X is a label and n is a fanout.
		"""
		cdef int n, m, components = 0
		assert coarse is not None
		if self.mapping is not NULL: free(self.mapping)
		self.mapping = <UInt *>malloc(sizeof(UInt) * self.nonterminals)
		if splitprune and markorigin:
			if self.splitmapping is not NULL:
				if self.splitmapping[0] is not NULL: free(self.splitmapping[0])
				free(self.splitmapping)
			self.splitmapping = <UInt **>malloc(sizeof(UInt *) * self.nonterminals)
			for n in range(self.nonterminals): self.splitmapping[n] = NULL
			self.splitmapping[0] = <UInt *>malloc(sizeof(UInt) *
				sum([self.fanout[n] for n in range(self.nonterminals)
					if self.fanout[n] > 1]))
		for n in range(self.nonterminals):
			if not neverblockre or neverblockre.search(self.tolabel[n]) is None:
				strlabel = self.tolabel[n]
				if striplabelre is not None:
					strlabel = striplabelre.sub("", strlabel, 1)
				if self.fanout[n] == 1 or not splitprune:
					self.mapping[n] = coarse.toid[strlabel]
				else:
					strlabel += "*"
					if markorigin:
						self.splitmapping[n] = &(
								self.splitmapping[0][components])
						components += self.fanout[n]
						for m in range(self.fanout[n]):
							self.splitmapping[n][m] = coarse.toid[
								strlabel + str(m)]
					else:
						self.mapping[n] = coarse.toid[strlabel]
			else:
				self.mapping[n] = 0
		if debug:
			for n in range(self.nonterminals):
				if self.mapping[n]:
					print "%s[%d] =>" % (self.tolabel[n], self.fanout[n]),
					if self.fanout[n] == 1 or not (splitprune and markorigin):
						print coarse.tolabel[self.mapping[n]],
					elif self.fanout[n] > 1:
						for m in range(self.fanout[n]):
							print coarse.tolabel[self.splitmapping[n][m]],
					print
			print dict(striplabelre=striplabelre.pattern,
					neverblockre=neverblockre.pattern,
					splitprune=splitprune, markorigin=markorigin)
	def write_lcfrs_grammar(self, rules, lexicon):
		""" Writes the grammar into a simple text file format, to the file
		objects given as arguments. Fields are separated by tabs. Components of
		the yield function are comma-separated. Weights are expressed as
		hexadecimal negative logprobs. TODO: store & print rational numbers instead
		E.g.
		rules: S    NP  VP  010 0x1.9c041f7ed8d33p+1
			VP_2    VB  NP  0,1 0x1.434b1382efeb8p+1
			NP	NN	0	0.3
		lexicon: NN Epsilon Haus    0.3
		"""
		cdef size_t n = 0
		cdef Rule rule = self.bylhs[0][n]
		cdef LexicalRule term
		while rule.lhs < self.nonterminals:
			pyfloat = rule.prob
			rules.write("%s\t%s%s\t%s\t%g\n" % (self.tolabel[rule.lhs],
				self.tolabel[rule.rhs1],
				('\t'+self.tolabel[rule.rhs2] if rule.rhs2 else ''),
				self.yfrepr(rule), exp(-pyfloat))) #pyfloat.hex()))
			n += 1
			rule = self.bylhs[0][n]
		for word in self.lexical:
			for term in self.lexical[word]:
				pyfloat = term.prob
				lexicon.write("%s\t%s\t%g\n" % (
					self.tolabel[term.lhs], word, exp(-pyfloat))) #pyfloat.hex()))
	def write_bitpar_grammar(self, rules, lexicon):
		""" write grammar as a bitpar grammar to files specified by rules and
		lexicon. As `frequencies' we give the probabilities in the grammar,
		which is supported by bitpar. """
		cdef size_t n = 0
		cdef Rule rule = self.bylhs[0][n]
		cdef LexicalRule term
		while rule.lhs < self.nonterminals:
			assert self.fanout[rule.lhs] == 1, ("can only export CFG rules.\n"
					"rule: %s" % (self.rulerepr(rule)))
			rules.write("%g\t%s\t%s%s\n" % (exp(-rule.prob),
				self.tolabel[rule.lhs], self.tolabel[rule.rhs1],
				("\t" + self.tolabel[rule.rhs2]) if rule.rhs2 else ''))
			n += 1
			rule = self.bylhs[0][n]
		for word in self.lexical:
			lexicon.write(word)
			for term in self.lexical[word]:
				lexicon.write("\t%s %g" % (self.tolabel[term.lhs],
					exp(-term.prob)))
			lexicon.write("\n")
	cdef str rulerepr(self, Rule rule):
		left = "%.2f %s => %s%s" % (
			exp(-rule.prob),
			self.tolabel[rule.lhs],
			self.tolabel[rule.rhs1],
			"  %s" % self.tolabel[rule.rhs2]
				if rule.rhs2 else "")
		return left.ljust(40) + self.yfrepr(rule)
	cdef str yfrepr(self, Rule rule):
		cdef int n, m = 0
		cdef str result = ""
		for n in range(8 * sizeof(rule.args)):
			result += "1" if (rule.args >> n) & 1 else "0"
			if (rule.lengths >> n) & 1:
				m += 1
				if m == self.fanout[rule.lhs]: return result
				else: result += ","
		raise ValueError("expected %d components" % self.fanout[rule.lhs])
	def rulesrepr(self, lhs):
		cdef int n = 0
		result = []
		while self.bylhs[lhs][n].lhs == lhs:
			result.append(self.rulerepr(self.bylhs[lhs][n]))
			n += 1
		return "\n".join(result)
	def __repr__(self):
		return "%s(%r)" % (self.__class__.__name__, self.origrules)
	def __str__(self):
		cdef LexicalRule lr
		rules = "\n".join(filter(None,
			[self.rulesrepr(lhs) for lhs in range(1, self.nonterminals)]))
		lexical = "\n".join(["%.2f %s => %s" % (exp(-lr.prob),
				self.tolabel[lr.lhs], lr.word.encode('unicode-escape'))
			for word in sorted(self.lexical) for lr in self.lexical[word]])
		return "rules:\n%s\nlexicon:\n%s\nlabels:\n%s" % (rules, lexical,
			", ".join("%s=%d" % a for a in sorted(self.toid.iteritems())))
	def __reduce__(self):
		return (Grammar, (self.origrules,))
	def __dealloc__(Grammar self):
		if self.unary is not NULL:
			if self.unary[0] is not NULL:
				free(self.unary[0])
				self.unary[0] = NULL
			free(self.unary)
			self.unary = NULL
		if self.fanout is not NULL:
			free(self.fanout)
			self.fanout = NULL
		if self.mapping is not NULL:
			free(self.mapping)
			self.mapping = NULL
		if self.splitmapping is not NULL:
			free(self.splitmapping[0])
			free(self.splitmapping)
			self.splitmapping = NULL

def myitemget(idx, x):
	""" Given a grammar rule 'x', return the non-terminal in position 'idx'. """
	if idx < len(x[0][0]): return x[0][0][idx]
	return 0

cdef copyrules(grammar, Rule **dest, idx, filterlen, toid, nonterminals):
	""" Auxiliary function to create Grammar objects. Copies certain grammar
	rules from the list in `grammar` to an array of structs. Grammar rules are
	placed in a contiguous array, sorted order by lhs, rhs1 or rhs2 depending
	on the value of `idx'. A separate array has a pointer for each non-terminal
	into this array; e.g.: dest[NP][0] == the first rule with an NP in the idx
	position. """
	cdef UInt prev = 0
	cdef size_t n = 0	# rule number
	cdef size_t m		# bit index in yield function
	cdef Rule *cur
	cdef dict rulenos = dict([(rule, m) for m, (rule, w) in enumerate(grammar)])
	filteredgrammar = [rule for rule in grammar
			if rule[0][0][1] != 'Epsilon'
			and (not filterlen or len(rule[0][0]) == filterlen)]
	sortedgrammar = sorted(filteredgrammar, key=partial(myitemget, idx))
	#need to set dest even when there are no rules for that idx
	for m in range(nonterminals): dest[m] = dest[0]
	for (rule, yf), w in sortedgrammar:
		cur = &(dest[0][n])
		cur.no = rulenos[rule, yf]
		cur.lhs  = toid[rule[0]]
		cur.rhs1 = toid[rule[1]]
		cur.rhs2 = toid[rule[2]] if len(rule) == 3 else 0
		#cur.fanout = len(yf)
		cur.prob = abs(w)
		cur.lengths = cur.args = m = 0
		for a in yf:
			for b in a: #component:
				if b == 1:
					cur.args += 1 << m
				m += 1
			cur.lengths |= 1 << (m - 1)
		assert m < (8 * sizeof(cur.args)), (m, (8 * sizeof(cur.args)))
		# if this is the first rule with this non-terminal,
		# add it to the index
		if n and toid[rule[idx]] != prev:
			dest[toid[rule[idx]]] = cur
		prev = toid[rule[idx]]
		n += 1
	# sentinel rule
	dest[0][n].lhs = dest[0][n].rhs1 = dest[0][n].rhs2 = nonterminals

cdef class SmallChartItem:
	""" Item with word sized bitvector """
	def __init__(SmallChartItem self, label, vec):
		self.label = label
		self.vec = vec
	def __hash__(SmallChartItem self):
		# juxtapose bits of label and vec, rotating vec if > 33 words
		return self.label ^ (self.vec << 31UL) ^ (self.vec >> 31UL)
	def __richcmp__(SmallChartItem self, SmallChartItem other, int op):
		if   op == 2: return self.label == other.label and self.vec == other.vec
		elif op == 3: return self.label != other.label or self.vec != other.vec
		elif op == 5: return self.label >= other.label or self.vec >= other.vec
		elif op == 1: return self.label <= other.label or self.vec <= other.vec
		elif op == 0: return self.label < other.label or self.vec < other.vec
		elif op == 4: return self.label > other.label or self.vec > other.vec
	def __nonzero__(SmallChartItem self):
		return self.label != 0 and self.vec != 0
	def __repr__(self):
		return "%s(%d, %s)" % (self.__class__.__name__,
				self.label, bin(self.vec))
	def lexidx(self):
		assert self.label == 0
		return self.vec
	def copy(SmallChartItem self):
		return SmallChartItem(self.label, self.vec)

cdef class FatChartItem:
	""" Item with fixed-with bitvector. """
	def __hash__(self):
		# juxtapose bits of label and first 32 bits of vec
		cdef long _hash
		cdef size_t n
		_hash = self.label ^ (self.vec[0] << 31UL) ^ (self.vec[0] >> 31UL)
		# add remaining bits
		for n in range(sizeof(self.vec[0]), sizeof(self.vec)):
			_hash *= 33 ^ (<UChar *>self.vec)[n]
		return _hash
	def __richcmp__(FatChartItem self, FatChartItem other, int op):
		cdef int cmp = memcmp(<UChar *>self.vec, <UChar *>other.vec,
			sizeof(self.vec))
		cdef bint labelmatch = self.label == other.label
		if   op == 2: return labelmatch and cmp == 0
		elif op == 3: return not labelmatch or cmp != 0
		elif op == 5: return self.label >= other.label or (labelmatch and cmp >= 0)
		elif op == 1: return self.label <= other.label or (labelmatch and cmp <= 0)
		elif op == 0: return self.label < other.label or (labelmatch and cmp < 0)
		elif op == 4: return self.label > other.label or (labelmatch and cmp > 0)
	def __nonzero__(self):
		cdef int n
		if self.label:
			for n in range(SLOTS):
				if self.vec[n]: return True
		return False
	def __repr__(self):
		return "%s(%d, %s)" % (self.__class__.__name__,
			self.label, binrepr(self.vec))
	def lexidx(self):
		assert self.label == 0
		return self.vec[0]
	def copy(FatChartItem self):
		cdef FatChartItem a = FatChartItem(self.label)
		for n in range(SLOTS):
			a.vec[n] = self.vec[n]
		return a

cdef class CFGChartItem:
	""" Item for CFG parsing; span is denoted with start and end indices. """
	def __hash__(self):
		cdef long _hash
		# juxtapose bits of label and indices of span
		_hash = self.label
		_hash ^= <ULong>self.start << 32UL
		_hash ^= <ULong>self.end << 40UL
		return _hash
	def __richcmp__(CFGChartItem self, CFGChartItem other, int op):
		cdef bint labelmatch = self.label == other.label
		if   op == 2: return (labelmatch and self.start == other.start
				and self.end == other.end)
		elif op == 3: return (not labelmatch or self.start != other.start
				or self.end != other.end)
		elif op == 5: return self.label >= other.label or (labelmatch
			and (self.start >= other.start or (
			self.start == other.start and self.end >= other.end)))
		elif op == 1: return self.label <= other.label or (labelmatch
			and (self.start <= other.start or (
			self.start == other.start and self.end <= other.end)))
		elif op == 0: return self.label < other.label or (labelmatch
			and (self.start < other.start or (
			self.start == other.start and self.end < other.end)))
		elif op == 4: return self.label > other.label or (labelmatch
			and (self.start > other.start or (
			self.start == other.start and self.end > other.end)))
	def __nonzero__(self):
		return self.label and self.end
	def __repr__(self):
		return "%s(%d, %d, %d)" % (self.__class__.__name__,
				self.label, self.start, self.end)
	def lexidx(self):
		assert self.label == 0
		return self.start
	def copy(CFGChartItem self):
		return new_CFGChartItem(self.label, self.start, self.end)

cdef SmallChartItem CFGtoSmallChartItem(UInt label, UChar start, UChar end):
	return new_ChartItem(label, (1ULL << end) - (1ULL << start))

cdef FatChartItem CFGtoFatChartItem(UInt label, UChar start, UChar end):
	cdef FatChartItem fci = new_FatChartItem(label)
	if BITSLOT(start) == BITSLOT(end):
		fci.vec[BITSLOT(start)] = (1ULL << end) - (1ULL << start)
	else:
		fci.vec[BITSLOT(start)] = ~0UL << (start % BITSIZE)
		for n in range(BITSLOT(start) + 1, BITSLOT(end)): fci.vec[n] = ~0UL
		fci.vec[BITSLOT(end)] = BITMASK(end) - 1
	return fci

cdef binrepr(ULong *vec):
	cdef int m, n = SLOTS - 1
	cdef str result
	while n and vec[n] == 0: n -= 1
	result = bin(vec[n])
	for m in range(n - 1, -1, -1):
		result += bin(vec[m])[2:].rjust(BITSIZE, '0')
	return result

#cdef class NewChartItem:
#	""" Item with arbitrary length bitvector. Not used. """
#	def __init__(self, label):
#		self.label = label
#	def __hash__(self):
#		cdef size_t n
#		cdef long _hash = 5381
#		for n in range(7 * sizeof(ULong)):
#			_hash *= 33 ^ (<char *>self.vecptr)[n]
#		return _hash
#	def __richcmp__(NewChartItem self, NewChartItem other, int op):
#		#if op == 2: return self.label == other.label and self.vec == other.vec
#		#elif op == 3: return self.label != other.label or self.vec != other.vec
#		#elif op == 5: return self.label >= other.label or self.vec >= other.vec
#		#elif op == 1: return self.label <= other.label or self.vec <= other.vec
#		#elif op == 0: return self.label < other.label or self.vec < other.vec
#		#elif op == 4: return self.label > other.label or self.vec > other.vec
#		raise NotImplemented
#	def __nonzero__(self):
#		raise NotImplemented
#		#return self.label != 0 and self.vec != 0
#	def __repr__(self):
#		raise NotImplemented
#		#return "ChartItem(%d, %s)" % (self.label, bin(self.vec))

cdef class LCFRSEdge:
	def __init__(LCFRSEdge self, score, inside, prob, ruleno, left, right):
		self.score = score; self.inside = inside; self.prob = prob
		self.ruleno = ruleno; self.left = left; self.right = right
		cdef long h
		#self._hash = hash((inside, prob, left, right))
		# this is the hash function used for tuples, apparently
		h = (1000003UL * 0x345678UL) ^ <long>self.inside
		h = (1000003UL * h) ^ <long>self.prob
		h = (1000003UL * h) ^ <long>self.left.__hash__()
		h = (1000003UL * h) ^ <long>self.right.__hash__()
		self._hash = h
	def __hash__(LCFRSEdge self):
		return self._hash
	def __richcmp__(LCFRSEdge self, other, int op):
		# the ordering only depends on the estimate / inside score
		if op == 0: return self.score < (<LCFRSEdge>other).score
		elif op == 1: return self.score <= (<LCFRSEdge>other).score
		# (in)equality compares all elements
		# boolean trick: equality and inequality in one expression i.e., the
		# equality between the two boolean expressions acts as biconditional
		elif op == 2 or op == 3:
			return (op == 2) == (
				self.score == (<LCFRSEdge>other).score
				and self.inside == (<LCFRSEdge>other).inside
				and self.prob == (<LCFRSEdge>other).prob
				and self.left == (<LCFRSEdge>other).left
				and self.right == (<LCFRSEdge>other).right)
		elif op == 4: return self.score > other.score
		elif op == 5: return self.score >= other.score
		elif op == 1: return self.score <= other.score
		elif op == 0: return self.score < other.score
	def __repr__(self):
		return "%s(%g, %g, %g, %r, %r)" % (self.__class__.__name__,
				self.score, self.inside, self.prob, self.left, self.right)
	def copy(self):
		return LCFRSEdge(self.score, self.inside, self.prob, self.ruleno,
				self.left.copy(), self.right.copy())

cdef class CFGEdge:
	def __init__(self): raise NotImplemented
	#def __init__(CFGEdge self, inside, Rule *rule, mid):
	#	self.inside = inside; self.rule = rule; self.mid = mid
	def __hash__(CFGEdge self):
		cdef long h
		# this is the hash function used for tuples, apparently
		h = (1000003UL * 0x345678UL) ^ <long>self.inside
		h = (1000003UL * h) ^ <long>self.rule
		h = (1000003UL * h) ^ <long>self.mid
		return h
	def __richcmp__(CFGEdge self, CFGEdge other, int op):
		# the ordering only depends on the inside score
		if op == 0: return self.inside < other.inside
		elif op == 1: return self.inside <= other.inside
		# (in)equality compares all elements
		# boolean trick: equality and inequality in one expression i.e., the
		# equality between the two boolean expressions acts as biconditional
		elif op == 2 or op == 3:
			return (op == 2) == (
				self.rule is other.rule
				and self.inside == other.inside
				and self.mid == other.mid)
		elif op == 4: return self.inside > other.inside
		elif op == 5: return self.inside >= other.inside
		elif op == 1: return self.inside <= other.inside
		elif op == 0: return self.inside < other.inside
	def __repr__(self):
		return "%s(%g, 0x%x, %r)" % (self.__class__.__name__,
				self.inside, <long>self.rule, self.mid)

cdef class RankedEdge:
	def __cinit__(self, ChartItem head, LCFRSEdge edge, int j1, int j2):
		self.head = head; self.edge = edge
		self.left = j1; self.right = j2
		cdef long h
		#h = hash((head, edge, j1, j2))
		h = (1000003UL * 0x345678UL) ^ hash(self.head)
		h = (1000003UL * h) ^ hash(self.edge)
		h = (1000003UL * h) ^ self.left
		h = (1000003UL * h) ^ self.right
		self._hash = h
	def __hash__(self):
		return self._hash
	def __richcmp__(RankedEdge self, RankedEdge other, int op):
		if op == 2 or op == 3:
			return (op == 2) == (
				self.left == other.left
				and self.right == other.right
				and self.head == other.head
				and self.edge == other.edge)
		else:
			raise NotImplemented
	def __repr__(self):
		return "%s(%r, %r, %d, %d)" % (self.__class__.__name__,
			self.head, self.edge, self.left, self.right)

cdef class RankedCFGEdge:
	def __cinit__(self, UInt label, UChar start, UChar end, Edge edge,
			int j1, int j2):
		self.label = label; self.start = start; self.end = end
		self.edge = edge; self.left = j1; self.right = j2
		cdef long h
		h = hash((label, start, end, edge, j1, j2))
		#h = (1000003UL * 0x345678UL) ^ hash(self.head)
		#h = (1000003UL * h) ^ hash(self.edge)
		#h = (1000003UL * h) ^ self.left
		#h = (1000003UL * h) ^ self.right
		self._hash = h
	def __hash__(self):
		return self._hash
	def __richcmp__(RankedCFGEdge self, RankedCFGEdge other, int op):
		if op == 2 or op == 3:
			return (op == 2) == (
				self.left == other.left
				and self.right == other.right
				and self.label == other.label
				and self.start == other.start
				and self.end == other.end
				and self.edge == other.edge)
		else:
			raise NotImplemented
	def __repr__(self):
		return "%s(%r, %r, %r, %r, %d, %d)" % (self.__class__.__name__,
			self.label, self.start, self.end, self.edge, self.left, self.right)

cdef class LexicalRule:
	def __init__(self, lhs, rhs1, rhs2, word, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.word = word; self.prob = prob
	def __repr__(self):
		return "%s%r" % (self.__class__.__name__,
				(self.lhs, self.rhs1, self.rhs2, self.word, self.prob))

cdef class Ctrees:
	"""auxiliary class to be able to pass around collections of trees in
	Python"""
	def __cinit__(self):
		self.data = NULL
	def __init__(self, list trees=None, dict labels=None,
		dict prods=None):
		self.len=0; self.max=0; self.maxnodes = 0; self.nodesleft = 0
		if trees is None: return
		else: assert labels is not None and prods is not None
		self.alloc(len(trees), sum(map(len, trees)))
		for tree in trees: self.add(tree, labels, prods)
	cpdef alloc(self, int numtrees, long numnodes):
		""" Initialize an array of trees of nodes structs. """
		self.max = numtrees
		self.data = <NodeArray *>malloc(numtrees * sizeof(NodeArray))
		assert self.data is not NULL
		self.data[0].nodes = <Node *>malloc(numnodes * sizeof(Node))
		assert self.data[0].nodes is not NULL
		self.nodes = self.nodesleft = numnodes
	cdef realloc(self, int len):
		""" Increase size of array (handy with incremental binarization) """
		#other options: new alloc: fragmentation (maybe not so bad)
		#memory pool: idem
		cdef size_t n
		cdef Node *new = NULL
		self.nodes += (self.max - self.len) * len #estimate
		new = <Node *>realloc(self.data[0].nodes, self.nodes * sizeof(Node))
		assert new is not NULL
		if new != self.data[0].nodes: # need to update all previous pointers
			self.data[0].nodes = new
			for n from 1 <= n < self.len:
				# derive pointer from previous tree offset by its size
				self.data[n].nodes = &(
					self.data[n-1].nodes)[self.data[n-1].len]
	cpdef add(self, list tree, dict labels, dict prods):
		""" Trees can be incrementally added to the node array; useful
		when dealing with large numbers of NLTK trees (say 100,000)."""
		assert self.len < self.max, ("either no space left (len >= max) or "
			"alloc() has not been called (max=0). max = %d" % self.max)
		if self.nodesleft < len(tree): self.realloc(len(tree))
		self.data[self.len].len = len(tree)
		if self.len: # derive pointer from previous tree offset by its size
			self.data[self.len].nodes = &(
				self.data[self.len-1].nodes)[self.data[self.len-1].len]
		copynodes(tree, labels, prods, self.data[self.len].nodes)
		self.data[self.len].root = tree[0].root
		self.len += 1
		self.nodesleft -= len(tree)
		self.maxnodes = max(self.maxnodes, len(tree))
	def __dealloc__(Ctrees self):
		if self.data is not NULL:
			if self.data[0].nodes is not NULL: free(self.data[0].nodes)
			free(self.data)
	def __len__(self): return self.len

cdef inline copynodes(tree, dict labels, dict prods, Node *result):
	""" Convert NLTK tree to an array of Node structs. """
	cdef int n
	for n, a in enumerate(tree):
		if isinstance(a, Tree):
			assert 1 <= len(a) <= 2, (
				"trees must be non-empty and binarized:\n%s\n%s" % (a, tree[0]))
			result[n].label = labels.get(a.node, -2)
			if len(a.prod) == 1: result[n].prod = -2 #fixme: correct for LCFRS?
			else: result[n].prod = prods.get(a.prod, -2)
			if hasattr(a[0], 'idx'): result[n].left = a[0].idx
			else: result[n].left = -1
			if len(a) == 2 and hasattr(a[1], 'idx'):
				result[n].right = a[1].idx
			else: result[n].right = -1
		elif isinstance(a, Terminal):
			result[n].label = a.node
			result[n].prod = result[n].left = result[n].right = -1
		else: assert isinstance(a, Tree) or isinstance(a, Terminal)

class Terminal:
	"""auxiliary class to be able to add indices to terminal nodes of NLTK
	trees"""
	def __init__(self, node): self.prod = self.node = node
	def __repr__(self): return repr(self.node)
	def __hash__(self): return hash(self.node)
	def __iter__(self): return iter(())
	def __len__(self): return 0
	def __index__(self): return self.node
	def __getitem__(self, val):
		if isinstance(val, slice): return ()
		else: raise IndexError("A terminal has zero children.")

cdef class FrozenArray:
	""" A wrapper around a Python array, with hash value and comparison
	operators. When used as key in a dictionary or in a set, make sure
	it is not mutated, because objects with a __hash__ method are expected to
	be immutable. """
	def __hash__(self):
		cdef size_t n
		cdef long _hash = 5381
		for n in range(len(self.obj)):
			_hash *= 33 ^ self.obj.data.as_uchars[n]
		return _hash
	def __richcmp__(FrozenArray self, FrozenArray other, int op):
		cdef int cmp = -1
		if (self.obj.ob_descr.itemsize == other.obj.ob_descr.itemsize
				and len(self.obj) == len(other.obj)):
			cmp = memcmp(self.obj.data.as_uchars, other.obj.data.as_uchars,
				len(self.obj) * self.obj.ob_descr.itemsize)
		if op == 2: return cmp == 0
		elif op == 3: return cmp != 0
		elif op == 0: return cmp < 0
		elif op == 4: return cmp > 0
		elif op == 1: return cmp <= 0
		return cmp >= 0

cdef class CBitset:
	""" auxiliary class to be able to pass around bitsets in Python.
	the memory for the bitset itself is not managed by this class. """
	def __cinit__(self, UChar slots):
		self.slots = slots
	def __hash__(self):
		cdef size_t n
		cdef long _hash = 5381
		for n in range(self.slots * sizeof(ULong)):
			_hash *= 33 ^ (<char *>self.data)[n]
		return _hash
	def __richcmp__(CBitset self, CBitset other, int op):
		# value comparisons
		cdef int cmp = memcmp(<void *>self.data, <void *>other.data,
					self.slots)
		if op == 2: return cmp == 0
		elif op == 3: return cmp != 0
		elif op == 0: return cmp < 0
		elif op == 4: return cmp > 0
		elif op == 1: return cmp <= 0
		return cmp >= 0

	cdef int bitcount(self):
		""" number of set bits in variable length bitvector """
		cdef int a, result = __builtin_popcountl(self.data[0])
		for a from 1 <= a < self.slots:
			result += __builtin_popcountl(self.data[a])
		return result

	cdef int nextset(self, UInt pos):
		""" return next set bit starting from pos, -1 if there is none. """
		cdef UInt a = BITSLOT(pos), offset = pos % BITSIZE
		if self.data[a] >> offset:
			return pos + __builtin_ctzl(self.data[a] >> offset)
		for a in range(a + 1, self.slots):
			if self.data[a]: return a * BITSIZE + __builtin_ctzl(self.data[a])
		return -1

	cdef int nextunset(self, UInt pos):
		""" return next unset bit starting from pos. """
		cdef UInt a = BITSLOT(pos), offset = pos % BITSIZE
		if ~(self.data[a] >> offset):
			return pos + __builtin_ctzl(~(self.data[a] >> offset))
		a += 1
		while self.data[a] == ~0UL: a += 1
		return a * BITSIZE + __builtin_ctzl(~(self.data[a]))

	cdef void setunion(self, CBitset src):
		""" dest gets the union of dest and src; both operands must have at
		least `slots' slots. """
		cdef int a
		for a in range(self.slots): self.data[a] |= src.data[a]

	cdef bint superset(self, CBitset op):
		""" test whether `op' is a superset of this bitset; i.e., whether
		all bits of this bitset are in op. """
		cdef int a
		for a in range(self.slots):
			if self.data[a] != (self.data[a] & op.data[a]): return False
		return True

	cdef bint subset(self, CBitset op):
		""" test whether `op' is a subset of this bitset; i.e., whether
		all bits of op are in this bitset. """
		cdef int a
		for a in range(self.slots):
			if (self.data[a] & op.data[a]) != op.data[a]: return False
		return True

	cdef bint disjunct(self, CBitset op):
		""" test whether `op' is disjunct from this bitset; i.e., whether
		no bits of op are in this bitset & vice versa. """
		cdef int a
		for a in range(self.slots):
			if (self.data[a] & op.data[a]): return False
		return True

cdef class MemoryPool:
	"""A memory pool that allocates chunks of poolsize, up to limit times.
	Memory is automatically freed when object is deallocated. """
	def __cinit__(self, int poolsize, int limit):
		cdef int x
		self.poolsize = poolsize
		self.limit = limit
		self.n = 0
		self.pool = <void **>malloc(limit * sizeof(void *))
		assert self.pool is not NULL
		for x in range(limit): self.pool[x] = NULL
		self.cur = self.pool[0] = <ULong *>malloc(self.poolsize)
		assert self.cur is not NULL
		self.leftinpool = self.poolsize
	cdef void *malloc(self, int size):
		cdef void *ptr
		if size > self.poolsize: return NULL
		elif self.leftinpool < size:
			self.n += 1
			assert self.n < self.limit
			if self.pool[self.n] is NULL:
				self.pool[self.n] = <ULong *>malloc(self.poolsize)
			self.cur = self.pool[self.n]
			assert self.cur is not NULL
			self.leftinpool = self.poolsize
		ptr = self.cur
		self.cur = &((<char *>self.cur)[size])
		self.leftinpool -= size
		return ptr
	cdef void reset(self):
		self.n = 0
		self.cur = self.pool[0]
		self.leftinpool = self.poolsize
	def __dealloc__(MemoryPool self):
		cdef int x
		for x in range(self.n+1): free(self.pool[x])
		free(self.pool)

class OrderedSet(Set):
	""" A frozen, ordered set which maintains a regular list/tuple and set. """
	def __init__(self, iterable=None):
		if iterable:
			self.seq = tuple(iterable)
			self.theset = frozenset(self.seq)
		else:
			self.seq = ()
			self.theset = frozenset()
	def __hash__(self):
		return hash(self.theset)
	def __contains__(self, value):
		return value in self.theset
	def __len__(self):
		return len(self.theset)
	def __iter__(self):
		return iter(self.seq)
	def __getitem__(self, n):
		return self.seq[n]
	def __reversed__(self):
		return reversed(self.seq)
	def __repr__(self):
		if not self.seq:
			return '%s()' % (self.__class__.__name__,)
		return '%s(%r)' % (self.__class__.__name__, self.seq)
	def __eq__(self, other):
		#if isinstance(other, (OrderedSet, Sequence)):
		#	return len(self) == len(other) and list(self) == list(other)
		# equality is defined _without_ regard for order
		return self.theset == set(other)
	def __and__(self, other):
		""" maintain the order of the left operand. """
		if not isinstance(other, Iterable):
			return NotImplemented
		return self._from_iterable(value for value in self if value in other)
