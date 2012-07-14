import codecs
from math import exp
from collections import Set, Iterable
from functools import partial
from nltk import Tree

cdef class Grammar:
	def __init__(self, grammar):
		""" split the grammar into various lookup tables, mapping nonterminal
		labels to numeric identifiers. Also negates log-probabilities to
		accommodate min-heaps.
		Can only represent ordered SRCG rules (monotone LCFRS). """
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
		self.donotprune = frozenset()

		# the strategy is to lay out all non-lexical rules in a contiguous array
		# these arrays will contain pointers to relevant parts thereof
		# (one index per nonterminal)
		self.unary = <Rule **>malloc(sizeof(Rule *) * self.nonterminals * 4)
		assert self.unary is not NULL
		self.lbinary = &(self.unary[1 * self.nonterminals])
		self.rbinary = &(self.unary[2 * self.nonterminals])
		self.bylhs = &(self.unary[3 * self.nonterminals])

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
			else: raise ValueError(
				"grammar not binarized: %s" % repr((rule,yf,w)))
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
		copyrules(grammar, self.unary, 1,
			lambda rule: len(rule) == 2 and rule[1] != 'Epsilon',
			self.toid, self.nonterminals)
		copyrules(grammar, self.lbinary, 1,
			lambda rule: len(rule) == 3, self.toid, self.nonterminals)
		copyrules(grammar, self.rbinary, 2,
			lambda rule: len(rule) == 3, self.toid, self.nonterminals)
		copyrules(grammar, self.bylhs, 0,
			lambda rule: rule[1] != 'Epsilon', self.toid, self.nonterminals)
	def testgrammar(self, epsilon=0.01):
		""" report whether all left-hand sides sum to 1 +/-epsilon. """
		#We could be strict about separating POS tags and phrasal categories,
		#but Negra contains at least one tag (--) used for both.
		cdef size_t lhs
		sums = {}
		for lhs from 1 <= lhs < self.nonterminals:
			for n in range(self.numrules):
				if self.bylhs[lhs][n].lhs != lhs: break
				sums[lhs] = sums.get(lhs, 0.0) + exp(-self.bylhs[lhs][n].prob)
		for terminals in self.lexical.itervalues():
			for term in terminals:
				sums[term.lhs] = sums.get(term.lhs, 0.0) + exp(-term.prob)
		for lhs, mass in sums.iteritems():
			if abs(mass - 1.0) > epsilon:
				# fixme: use logging here?
				print "rules with %s:\n%s" % (
					self.tolabel[lhs], self.rulerepr(lhs))
				print "Does not sum to 1:",
				print self.tolabel[lhs], mass
				return False
		print "All left hand sides sum to 1"
		return True
	def getdonotprune(self, neverblockre):
		""" Construct a list of non-terminal IDs that match a regex. Items in
		this list will not be pruned during coarse-to-fine parsing.
		The regex will be applied with .match(), so it should match from the
		beginning of the label.
        - e.g., ".+|<" to ignore nodes introduced by binarization;
            useful if coarse and fine stages employ different kinds of
            markovization; e.g., NP and VP may be blocked, but not NP|<DT-NN>.
        - ".+_[0-9]+" to ignore nodes X_n where X is a label and n is a fanout > 1.
		"""
		self.donotprune = frozenset([n for n in range(self.nonterminals)
			if neverblockre.match(self.tolabel[n]) is not None])
	def write_srcg_grammar(self, rules, lexicon):
		""" Writes the grammar into a simple text file format.
		Fields are separated by tabs. Components of the yield function are
		comma-separated. Weights are expressed as hexadecimal negative logprobs.
		E.g.
		rules: S    NP  VP  010 0.5
			VP_2    VB  NP  0,1 0.4
			NP	NN	0	0.3
		lexicon: NN Epsilon Haus    0.3
		"""
		with open(rules, "w") as rules:
			for n in range(self.numrules):
				if self.bylhs[0][n].lhs == self.nonterminals: break
				pyfloat = self.bylhs[0][n].prob
				rules.write("%s\t%s%s\t%s\t%s\n" % (
					self.tolabel[self.bylhs[0][n].lhs],
					self.tolabel[self.bylhs[0][n].rhs1],
					('\t'+self.tolabel[self.bylhs[0][n].rhs2]
						if self.bylhs[0][n].rhs2 else ''),
					yfrepr(self.bylhs[0][n]),
					pyfloat.hex()))
		with codecs.open(lexicon, "w", encoding='utf-8') as lexicon:
			for word in self.lexical:
				for term in self.lexical[word]:
					pyfloat = term.prob
					lexicon.write("%s\t%s\t%s\n" % (
						self.tolabel[term.lhs], word, pyfloat.hex()))
		lexicon.close()
	def write_bitpar_grammar(self, rules, lexicon, encoding='utf-8'):
		""" write grammar as a bitpar grammar to files specified by rules and
		lexicon. As `frequencies' we give the probabilities in the grammar,
		which is supported by bitpar. """
		with open(rules, "w") as rules:
			for n in range(self.numrules):
				if self.bylhs[0][n].lhs == self.nonterminals: break
				assert self.bylhs[0][n].fanout == 1, "can only export CFG rules"
				rules.write("%f\t%s\t%s%s\n" % (
						exp(-self.bylhs[0][n].prob),
						self.tolabel[self.bylhs[0][n].lhs],
						self.tolabel[self.bylhs[0][n].rhs1],
						("\t" + self.tolabel[self.bylhs[0][n].rhs2])
							if self.bylhs[0][n].rhs2 else ''))
		with codecs.open(lexicon, "w", encoding=encoding) as lexicon:
			for word in self.lexical:
				lexicon.write(word)
				for term in self.lexical[word]:
					lexicon.write("\t%s %f" % (self.tolabel[term.lhs],
						exp(-term.prob)))
				lexicon.write("\n")
	def rulerepr(self, lhs):
		result = []
		for n in range(self.numrules):
			if self.bylhs[lhs][n].lhs != lhs: break
			result.append("%.2f %s => %s%s (%s)" % (
				exp(-self.bylhs[lhs][n].prob),
				self.tolabel[lhs],
				self.tolabel[self.bylhs[lhs][n].rhs1],
				" %s" % self.tolabel[self.bylhs[lhs][n].rhs2]
					if self.bylhs[lhs][n].rhs2 else "",
				yfrepr(self.bylhs[lhs][n])))
		return "\n".join(result)
	def __repr__(self):
		rules = "\n".join(filter(None,
			[self.rulerepr(lhs) for lhs in range(1, self.nonterminals)]))
		lexical = "\n".join(["%.2f %s => %s" % (
			exp(-lr.prob), self.tolabel[lr.lhs], lr.word.encode('unicode-escape'))
				for word in sorted(self.lexical)
					for lr in self.lexical[word]])
		return "rules:\n%s\nlexicon:\n%s\nlabels=%r" % (rules, lexical,
			self.toid)
	def __reduce__(self):
		return (Grammar, (self.origrules,))
	def __dealloc__(self):
		pass #FIXME
		#free(self.unary[0]); self.unary[0] = NULL
		#free(self.unary); self.unary = NULL

def myitemget(idx, x):
	""" Given a grammar rule 'x', return the non-terminal in position 'idx'. """
	if idx < len(x[0][0]): return x[0][0][idx]
	return 0

cdef copyrules(grammar, Rule **dest, idx, cond, toid, nonterminals):
	""" Auxiliary function to create Grammar objects. Copies certain grammar
	rules from the list in `grammar` to an array of structs. Grammar rules are
	placed in a contiguous array, sorted order by lhs, rhs1 or rhs2
	A separate array has a pointer for each non-terminal into this array;
	e.g.: dest[NP][0] == the first rule with an NP in the idx position.
	"""
	cdef UInt prev = 0
	cdef size_t n = 0	# rule number
	cdef size_t m		# bit index in yield function
	cdef Rule *cur
	sortedgrammar = sorted(grammar, key=partial(myitemget, idx))
	filteredgrammar = [rule for rule in sortedgrammar if cond(rule[0][0])]
	#need to set dest even when there are no rules for that idx
	for m in range(nonterminals): dest[m] = dest[0]
	for (rule, yf), w in filteredgrammar:
		cur = &(dest[0][n])
		cur.lhs  = toid[rule[0]]
		cur.rhs1 = toid[rule[1]]
		cur.rhs2 = toid[rule[2]] if len(rule) == 3 else 0
		cur.fanout = len(yf)
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

cdef str yfrepr(Rule rule):
	cdef int n, m = 0
	cdef str result = ""
	for n in range(8 * sizeof(rule.args)):
		result += "1" if (rule.args >> n) & 1 else "0"
		if (rule.lengths >> n) & 1:
			m += 1
			if m == rule.fanout: return result
			else: result += ","
	raise ValueError("expected %d components" % rule.fanout)

cdef class FatChartItem:
	def __init__(self, label):
		self.label = label
		#?? self.vec = vec
	def __hash__(self):
		cdef size_t n
		cdef long _hash = 5381
		for n in range(7 * sizeof(ULong)):
			_hash *= 33 ^ (<char *>self.vec)[n]
		return _hash
	def __richcmp__(FatChartItem self, FatChartItem other, int op):
		if op == 2: return self.label == other.label and self.vec == other.vec
		elif op == 3: return self.label != other.label or self.vec != other.vec
		elif op == 5: return self.label >= other.label or self.vec >= other.vec
		elif op == 1: return self.label <= other.label or self.vec <= other.vec
		elif op == 0: return self.label < other.label or self.vec < other.vec
		elif op == 4: return self.label > other.label or self.vec > other.vec
	def __nonzero__(self):
		raise NotImplemented
		#return self.label != 0 and self.vec != 0
	def __repr__(self):
		raise NotImplemented
		#return "ChartItem(%d, %s)" % (self.label, bin(self.vec))

cdef class ChartItem:
	def __init__(self, label, vec):
		self.label = label
		self.vec = vec
	def __hash__(self):
		cdef long h
		# juxtapose bits of label and vec, rotating vec if > 33 words
		return self.label ^ (self.vec << 31UL) ^ (self.vec >> 31UL)
	def __richcmp__(ChartItem self, ChartItem other, int op):
		if op == 2: return self.label == other.label and self.vec == other.vec
		elif op == 3: return self.label != other.label or self.vec != other.vec
		elif op == 5: return self.label >= other.label or self.vec >= other.vec
		elif op == 1: return self.label <= other.label or self.vec <= other.vec
		elif op == 0: return self.label < other.label or self.vec < other.vec
		elif op == 4: return self.label > other.label or self.vec > other.vec
	def __nonzero__(self):
		return self.label != 0 and self.vec != 0
	def __repr__(self):
		return "ChartItem(%d, %s)" % (self.label, bin(self.vec))

cdef class Edge:
	def __init__(self, score, inside, prob, left, right):
		self.score = score; self.inside = inside; self.prob = prob
		self.left = left; self.right = right
	def __hash__(self):
		cdef long h
		#self._hash = hash((inside, prob, left, right))
		# this is the hash function used for tuples, apparently
		h = (1000003UL * 0x345678UL) ^ <long>self.inside
		h = (1000003UL * h) ^ <long>self.prob
		h = (1000003UL * h) ^ (<ChartItem>self.left).vec
		h = (1000003UL * h) ^ (<ChartItem>self.left).label
		h = (1000003UL * h) ^ (<ChartItem>self.right).vec
		h = (1000003UL * h) ^ (<ChartItem>self.right).label
		return h
	def __richcmp__(Edge self, other, int op):
		# the ordering only depends on the estimate / inside score
		if op == 0: return self.score < (<Edge>other).score
		elif op == 1: return self.score <= (<Edge>other).score
		# (in)equality compares all elements
		# boolean trick: equality and inequality in one expression i.e., the
		# equality between the two boolean expressions acts as biconditional
		elif op == 2 or op == 3:
			return (op == 2) == (
				self.score == (<Edge>other).score
				and self.inside == (<Edge>other).inside
				and self.prob == (<Edge>other).prob
				and self.left == (<Edge>other).left
				and self.right == (<Edge>other).right)
		elif op == 4: return self.score > other.score
		elif op == 5: return self.score >= other.score
	def __repr__(self):
		return "Edge(%g, %g, %g, %r, %r)" % (
				self.score, self.inside, self.prob, self.left, self.right)

cdef class RankedEdge:
	def __cinit__(self, ChartItem head, Edge edge, int j1, int j2):
		self.head = head; self.edge = edge
		self.left = j1; self.right = j2
	def __hash__(self):
		cdef long h
		#h = hash((head, edge, j1, j2))
		h = (1000003UL * 0x345678UL) ^ hash(self.head)
		h = (1000003UL * h) ^ hash(self.edge)
		h = (1000003UL * h) ^ self.left
		h = (1000003UL * h) ^ self.right
		return h
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
		return "RankedEdge(%r, %r, %d, %d)" % (
			self.head, self.edge, self.left, self.right)

cdef class LexicalRule:
	def __init__(self, lhs, rhs1, rhs2, word, prob):
		self.lhs = lhs; self.rhs1 = rhs1; self.rhs2 = rhs2
		self.word = word; self.prob = prob
	def __repr__(self):
		return repr((self.lhs, self.rhs1, self.rhs2, self.word, self.prob))

cdef class Ctrees:
	"""auxiliary class to be able to pass around collections of trees in
	Python"""
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
	def __dealloc__(self):
		free(self.data[0].nodes); self.data[0].nodes = NULL
		free(self.data); self.data = NULL
	def __len__(self): return self.len

cdef inline copynodes(tree, dict labels, dict prods, Node *result):
	""" Convert NLTK tree to an array of Node structs. """
	cdef int n
	for n, a in enumerate(tree):
		if isinstance(a, Tree):
			assert 1 <= len(a) <= 2, "trees must be non-empty and binarized:\n%s" % a
			result[n].label = labels.get(a.node, -2)
			if len(a.prod) == 1: result[n].prod = -2
			else: result[n].prod = prods.get(a.prod, -2)
			result[n].left = a[0].idx
			if len(a) == 2:
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

cdef class FrozenArray:
	""" A wrapper around a Python unsigned long array, with hash value and
	comparison operators."""
	def __init__(self, array data):
		self.data = data
	def __hash__(self):
		cdef size_t n
		cdef long _hash = 5381
		for n in range(self.data.length):
			_hash *= 33 ^ self.data._B[n]
		return _hash
	def __richcmp__(FrozenArray self, FrozenArray other, int op):
		cdef int cmp = memcmp(self.data._B, other.data._B,
			self.data.length*sizeof(ULong))
		if op == 2: return cmp == 0
		elif op == 3: return cmp != 0
		elif op == 0: return cmp < 0
		elif op == 4: return cmp > 0
		elif op == 1: return cmp <= 0
		return cmp >= 0

cdef inline FrozenArray new_FrozenArray(array data):
	cdef FrozenArray item = FrozenArray.__new__(FrozenArray)
	item.data = data
	return item

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
	def __dealloc__(self):
		cdef int x
		for x in range(self.n+1):
			free(self.pool[x])
			self.pool[x] = NULL
		free(self.pool)
		self.pool = NULL

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

# some helper functions that only serve to bridge cython & python code
cpdef inline UInt getlabel(ChartItem a):
	return a.label
cpdef inline ULLong getvec(ChartItem a):
	return a.vec
cpdef inline double getscore(Edge a):
	return a.score
cpdef inline dict dictcast(d):
	return <dict>d
cpdef inline ChartItem itemcast(i):
	return <ChartItem>i
cpdef inline Edge edgecast(e):
	return <Edge>e

maxbitveclen = sizeof(ULLong) * 8
