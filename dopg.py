"""DOP1 implementation. Andreas van Cranenburgh <andreas@unstable.nl>
"""
from nltk import Production, WeightedProduction, WeightedGrammar, FreqDist, \
		Tree, ImmutableTree, Nonterminal, InsideChartParser, ProbabilisticTree
from collections import defaultdict
from itertools import chain, count
from operator import mul
#from math import log #do something with logprobs instead?
try:
	from itertools import product
except ImportError:
	# for pre 2.7 versions
	def product(*seq):
		if seq: return (b + (a,) for b in product(*seq[:-1]) for a in seq[-1])
		return ((), )

class GoodmanDOP:
	def __init__(self, treebank, rootsymbol='S', wrap=False, cnf=True,
				cleanup=True, normalize=False, extratags=(),
				parser=InsideChartParser, **parseroptions):
		""" initialize a DOP model given a treebank. uses the Goodman
		reduction of a STSG to a PCFG.  after initialization,
		self.parser will contain an InsideChartParser.

		>>> tree = Tree("(S (NP mary) (VP walks))")
		>>> d = GoodmanDOP([tree])
		>>> print d.grammar
		    Grammar with 8 productions (start state = S)
			NP -> 'mary' [1.0]
			NP@1 -> 'mary' [1.0]
			S -> NP VP [0.25]
			S -> NP VP@2 [0.25]
			S -> NP@1 VP [0.25]
			S -> NP@1 VP@2 [0.25]
			VP -> 'walks' [1.0]
			VP@2 -> 'walks' [1.0]
		>>> print d.parser.parse("mary walks".split())
		(S (NP mary) (VP@2 walks)) (p=0.25)		
		
		@param treebank: a list of Tree objects. Caveat lector:
			terminals may not have (non-terminals as) siblings.
		@param wrap: boolean specifying whether to add the start symbol
			to each tree
		@param normalize: whether to normalize frequencies
		@param parser: a class which will be instantiated with the DOP 
			model as its grammar. Supports BitParChartParser.
		
		instance variables:
		- self.grammar a WeightedGrammar containing the PCFG reduction
		- self.fcfg a list of strings containing the PCFG reduction 
		  with frequencies instead of probabilities
		- self.parser an InsideChartParser object
		- self.exemplars dictionary of known parse trees (memoization)"""
		from bitpar import BitParChartParser
		nonterminalfd, subtreefd, cfg = FreqDist(), FreqDist(), FreqDist()
		ids = count(1)
		self.exemplars = {}
		if wrap:
			# wrap trees in a common root symbol (eg. for morphology)
			treebank = [Tree(rootsymbol, [a]) for a in treebank]
		if cnf:
			#CNF conversion is destructive
			treebank = list(treebank)
			for a in treebank:
				a.chomsky_normal_form() #todo: sibling annotation necessary?

		# add unique IDs to nodes
		utreebank = [(tree, decorate_with_ids(tree, ids)) for tree in treebank]

		# count node frequencies
		for tree, utree in utreebank:
			nodefreq(tree, utree, subtreefd, nonterminalfd)

		if isinstance(parser, BitParChartParser):
			lexicon = set(x for a, b in utreebank for x in a.pos() + b.pos())
			# this takes the most time, produce CFG rules:
			cfg = FreqDist(chain(*(self.goodman(tree, utree)
								for tree, utree in utreebank)))
			cfg.update("%s\t%s" % (t, w) for w, t in extratags
								if w not in lexicon)
			lexicon.update(a for a in extratags if a not in lexicon)
			# annotate rules with frequencies
			self.fcfg = frequencies(cfg, subtreefd, nonterminalfd, normalize)
			self.parser = BitParChartParser(self.fcfg, lexicon, rootsymbol,
									cleanup=cleanup, **parseroptions)
		else:
			cfg = FreqDist(chain(*(self.goodman(tree, utree, False)
							for tree, utree in utreebank)))
			probs = probabilities(cfg, subtreefd, nonterminalfd)
			#for a in probs: print a
			self.grammar = WeightedGrammar(Nonterminal(rootsymbol), probs)
			self.parser = InsideChartParser(self.grammar)
			
		#stuff for self.mccparse
		#the highest id
		#self.addresses = ids.next()
		#a list of interior + exterior nodes, 
		#ie., non-terminals with and without ids
		#self.nonterminals = nonterminalfd.keys()
		#a mapping of ids to nonterminals without their IDs
		#self.nonterminal = dict(a.split("@")[::-1] for a in 
		#	nonterminalfd.keys() if "@" in a)

		#clean up
		del cfg, nonterminalfd

	def goodman(self, tree, utree, bitparfmt=True):
		""" given a parsetree from a treebank, yield a goodman
		reduction of eight rules per node (in the case of a binary tree).

		>>> tree = Tree("(S (NP mary) (VP walks))")
		>>> d = GoodmanDOP([tree])
		>>> utree = decorate_with_ids(tree, count(1))
		>>> sorted(d.goodman(tree, utree, False))
		[(NP, ('mary',)), (NP@1, ('mary',)), (S, (NP, VP)), (S, (NP, VP@2)),
		(S, (NP@1, VP)), (S, (NP@1, VP@2)), (VP, ('walks',)),
		(VP@2, ('walks',))]
		"""
		# linear: nr of nodes
		sep = "\t"
		for p, up in zip(tree.productions(), utree.productions()):
			if len(p.rhs()) == 0: raise ValueError
			if len(p.rhs()) == 1:
				if not isinstance(p.rhs()[0], Nonterminal): rhs = (p.rhs(), )
				else: rhs = (p.rhs(), up.rhs())
			#else: rhs = product(*zip(p.rhs(), up.rhs()))
			else:
				if all(isinstance(a, Nonterminal) for a in up.rhs()):
					rhs = set(product(*zip(p.rhs(), up.rhs())))
				else: rhs = product(*zip(p.rhs(), up.rhs()))

			# constant factor: 8
			#for l, r in product(*((p.lhs(), up.lhs()), rhs)):
			for l, r in product(set((p.lhs(), up.lhs())), rhs):
				#yield Production(l, r)
				if bitparfmt:
					yield "%s%s%s" % (l, sep, sep.join(map(unicode, r)))
				else:
					yield l, r
				# yield a delayed computation that also gives the frequencies
				# given a distribution of nonterminals
				#yield (lambda fd: WeightedProduction(l, r, prob= 
				#	reduce(mul, map(lambda z: '@' in z and
				#	fd[z] or 1, r)) / float(fd[l])))
	
	def parse(self, sent):
		"""most probable derivation (not very good)."""
		return self.parser.parse(sent)

	def mostprobableparse(self, sent, sample=None):
		"""warning: this problem is NP-complete. using an unsorted
		chart parser avoids unnecessary sorting (since we need all
		derivations anyway).
		
		@param sent: a sequence of terminals
		@param sample: None or int; if int then sample that many parses"""
		p = FreqDist()
		for a in self.parser.nbest_parse(sent, sample):
			p.inc(removeids(a).freeze(), a.prob())
		if p.max():
			return ProbabilisticTree(p.max().node, p.max(), prob=p[p.max()])
		else: raise ValueError("no parse")

	def mostconstituentscorrect(self, sent):
		""" not working yet. almost verbatim translation of Goodman's (1996)
		most constituents correct parsing algorithm, except for python's
		zero-based indexing. needs to be modified to return the actual parse
		tree. expects a pcfg in the form of a dictionary from productions to
		probabilities """ 
		def g(s, t, x):
			def f(s, t, x):
				return self.pcfg[Production(rootsymbol,
					sent[1:s] + [x] + sent[s+1:])]
			def e(s, t, x):
				return self.pcfg[Production(x, sent[s:t+1])]
			return f(s, t, x) * e(s, t, x ) / e(1, n, rootsymbol)

		sumx = defaultdict(int) #zero
		maxc = defaultdict(int) #zero
		for length in range(2, len(sent)+1):
			for s in range(1, len(sent) + length):
				t = s + length - 1
				for x in self.nonterminals:
					sumx[x] = g(s, t, x)
				for k in range(self.addresses):
					#ordered dictionary here
					x = self.nonterminal[k]
					sumx[x] += g(s, t, "%s@%d" % (x, k))
				max_x = max(sumx[x] for x in self.nonterminals)
				#for x in self.nonterminals:
				#	max_x = argmax(sumx, x) #???
				best_split = max(maxc[(s,r)] + maxc[(r+1,t)]
									for r in range(s, t))
				#for r in range(s, t):
				#	best_split = max(maxc[(s,r)] + maxc[(r+1,t)])
				maxc[(s,t)] = sumx[max_x] + best_split
		
		return maxc[(1, len(sent) + 1)]

def decorate_with_ids(tree, ids, include_preterminals=True):
	""" add unique identifiers to each internal non-terminal of a tree.

	>>> tree = Tree("(S (NP (DT the) (N dog)) (VP walks))")
	>>> decorate_with_ids(tree, count(1))
	Tree('S', [Tree('NP@1', [Tree('DT@2', ['the']), Tree('N@3', ['dog'])]), 
			Tree('VP@4', ['walks'])])

		@param ids: an iterator yielding a stream of IDs"""
	utree = tree.copy(True)
	#skip root node
	for a in utree:
		if not isinstance(a, Tree): continue
		for b in a.subtrees():
			#skip word boundary markers
			if a.node == "_": continue
			if include_preterminals or any(isinstance(c, Tree) for c in b):
				b.node = "%s@%d" % (b.node, ids.next())
	return utree

def nodefreq(tree, utree, subtreefd, nonterminalfd):
	"""count frequencies of nodes and calculate the number of
	subtrees headed by each node. updates "subtreefd" and "nonterminalfd"
	as a side effect. Expects a normal tree and a tree with IDs.

	>>> fd = FreqDist()
	>>> tree = Tree("(S (NP mary) (VP walks))")
	>>> d = GoodmanDOP([tree])
	>>> utree = decorate_with_ids(tree, count(1))
	>>> nodefreq(tree, utree, fd, FreqDist())
	4
	>>> fd.items()
	[('S', 4), ('NP', 1), ('NP@1', 1), ('VP', 1), ('VP@2', 1)]

		@param nonterminalfd: the FreqDist to store the counts in."""
	if not isinstance(tree, Tree):
		raise ValueError
	if len(tree) == 0:
		# this error occurs when a node has zero children,
		# e.g., (TOP (wrong))
		raise ValueError
	if any(isinstance(a, Tree) for a in tree):
		n = reduce(mul,
			(nodefreq(x, ux, subtreefd, nonterminalfd) + 1 for x, ux
			in zip(tree, utree)))
		subtreefd.inc(tree.node, count=n)
		nonterminalfd.inc(tree.node, count=1)
		# only add counts when utree.node is actually an interior node,
		# e.g., root node receives no ID so shouldn't be counted twice
		if utree.node != tree.node:
			subtreefd.inc(utree.node, count=n)
		return n
	else:
		subtreefd.inc(tree.node, count=1)
		nonterminalfd.inc(tree.node, count=1)
		if utree.node != tree.node:
			subtreefd.inc(utree.node, count=1)
		return 1

def probabilities(cfg, fd, nonterminalfd):
	"""merge cfg and frequency distribution into a pcfg with the right
	probabilities.

		@param cfg: a list of Productions
		@param fd: number of subtrees headed by each node
		@param nonterminalfd: a FreqDist of (non)terminals (with and
		without IDs)""" 
	#return [a(nonterminalfd) for a in cfg)
	def prob(l, r):
		return reduce(mul, map((lambda z: '@' in str(z)
			and fd[unicode(z)] or 1), r)) / float(fd[unicode(l)])
	# format expected by mccparse()
	#self.pcfg = dict((Production(l, r), (reduce(mul,
	#	map((lambda z: '@' in (type(z) == Nonterminal and z.symbol() or z)
	#	and nonterminalfd[z] or 1), r)) / nonterminalfd[l]))
	#	for l, r in set(cfg))

	# merge identical rules:
	#return [WeightedProduction(rule[0], rule[1:], prob=freq*prob(rule[0], rule[1:])) for rule, freq in ((rule.split('\t'), freq) for rule,freq in cfg.items())]
	
	return [WeightedProduction(l, r, prob=freq*prob(l, r))
						for (l,r),freq in cfg.items()]
	# do not merge identical rules
	#return [WeightedProduction(l, r, prob=prob(l, r)) for l, r in cfg]

def frequencies(cfg, fd, nonterminalfd, normalize=False):
	"""merge cfg and frequency distribution into a list of weighted 
	productions with frequencies as weights (as expected by bitpar).

		@param cfg: a list of Productions
		@param fd: number of subtrees headed by each node
		@param nonterminalfd: a FreqDist of (non)terminals (with and
		without IDs)""" 
	if normalize:
		# normalize by assigning equal weight to each node
		return ((rule, freq * reduce(mul,
			map((lambda z: fd[unicode(z)] if '@' in unicode(z) else 1),
			rule.split('\t')[1:]))
			/ (1 if '@' in rule.split('\t')[0]
			else float(nonterminalfd[rule.split('\t')[0]])))
		for rule, freq in cfg.items())
	return ((rule, freq * reduce(mul,
		map((lambda z: fd[unicode(z)] if '@' in unicode(z) else 1),
		rule.split('\t')[1:])))
		for rule, freq in cfg.items())

def removeids(tree):
	""" remove unique IDs introduced by the Goodman reduction """
	for a in tree.subtrees(lambda t: '@' in t.node):
		a.node = a.node.rsplit('@', 1)[0]
	return tree

#NB: the following code is equivalent to nltk.Tree.productions,
# except for accepting unicode
def productions(tree):
	"""
	Generate the productions that correspond to the non-terminal nodes of the
	tree.  For each subtree of the form (P: C1 C2 ... Cn) this produces a
	production of the form P -> C1 C2 ... Cn.
		@rtype: list of C{Production}s
	"""
	def _child_names(tree):
		names = []
		for child in tree:
			if isinstance(child, Tree):
				names.append(Nonterminal(child.node))
			else:
				names.append(child)
		return names

	if not isinstance(tree.node, basestring):
		raise TypeError, 'Productions can only be generated from trees having \
node labels that are strings'

	prods = [Production(Nonterminal(tree.node), _child_names(tree))]
	for child in tree:
		if isinstance(child, Tree):
			prods += productions(child)
	return prods
				
def main():
	""" a basic REPL for testing """
	corpus = """(S (NP John) (VP (V likes) (NP Mary)))
(S (NP Peter) (VP (V hates) (NP Susan)))
(S (NP Harry) (VP (V eats) (NP pizza)))
(S (NP Hermione) (VP (V eats)))""".splitlines()
	corpus = """(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (JJ hungry) (NN dog))))
(S (NP (DT The) (JJ little) (NN mouse)) (VP (VBP ate) (NP (DT the) (NN cat))))""".splitlines()
	#corpus = """(S (NP mary) (VP walks) (AP quickly))""".splitlines()
	#(S (NP Harry) (VP (V likes) (NP Susan) (ADVP (RB very) (RB much))))
	corpus = [Tree(a) for a in corpus]
	#d = GoodmanDOP(corpus, rootsymbol='S')
	from bitpar import BitParChartParser
	d = GoodmanDOP(corpus, rootsymbol='TOP', wrap='TOP',
						parser=BitParChartParser)
	#d = GoodmanDOP(corpus, rootsymbol='TOP', wrap='TOP')
	#print d.grammar
	print "corpus"
	for a in corpus: print a
	w = "foo!"
	while w:
		print "sentence:",
		w = raw_input().split()
		try:
			p = FreqDist()
			for n, a in enumerate(d.parser.nbest_parse(w)):
				if n > 1000: break
				print a
				p.inc(ImmutableTree.convert(removeids(a)), a.prob())
			#for b, a in sorted((b,a) for (a,b) in p.items()):
			#	print a, b
			print
			print 'best', p.max(), p[p.max()]
			#print d.parse(w)
		except Exception: # as e:
			print "error", #e

if __name__ == '__main__':
	import doctest
	# do doctests, but don't be pedantic about whitespace (I suspect it is the
	# militant anti-tab faction who are behind this obnoxious default)
	fail, attempted = doctest.testmod(verbose=False,
	optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)
	if attempted and not fail:
		print "%d doctests succeeded!" % attempted
	main()
