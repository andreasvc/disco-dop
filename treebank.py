# -*- coding: UTF-8 -*-
""" Read and write treebanks. """
import os, re, sys, codecs
from glob import glob
from itertools import count, repeat
from collections import OrderedDict, deque, Counter as multiset
from operator import itemgetter
from tree import ParentedTree, Tree

WORD, LEMMA, TAG, MORPH, FUNC, PARENT = range(6)
TERMINALSRE = re.compile(r" ([^ )]+)\)")
EXPORTNONTERMINAL = re.compile(r"^#[0-9]+$")

class NegraCorpusReader(object):
	""" Read a corpus in the Negra export format. """
	def __init__(self, root, fileids, encoding="utf-8", headrules=None,
			headfinal=False, headreverse=False, dounfold=False,
			functiontags=False, punct=None):
		""" headrules: if given, read rules for assigning heads and apply them
				by ordering constituents according to their heads
			headfinal: whether to put the head in final or in frontal position
			headreverse: the head is made final/frontal by reversing everything
			before or after the head. When true, the side on which the head is
				will be the reversed side.
			dounfold: whether to apply corpus transformations
			functiontags: whether to add function tags to node labels e.g. NP+OA
			punct: one of ...
				None: leave punctuation as is.
				'move': move punctuation to appropriate constituents
						using heuristics.
				'remove': eliminate punctuation.
				'restore': attach punctuation directly to root
						(as in original Negra/Tiger treebanks).
			"""
		self.reverse = headreverse
		self.headfinal = headfinal
		self.unfold = dounfold
		self.functiontags = functiontags
		self.punct = punct
		self.headrules = readheadrules(headrules) if headrules else {}
		self._encoding = encoding
		self._sents_cache = None
		self._tagged_sents_cache = None
		self._parsed_sents_cache = None
		self._filenames = glob(os.path.join(root, fileids))
		assert punct in (None, "move", "remove", "restore")
		assert self._filenames, (
				"no files matched pattern %s" % os.path.join(root, fileids))
		self._block_cache = self._read_blocks()
	def parsed_sents(self):
		""" Return a dictionary of parse trees. """
		if not self._parsed_sents_cache:
			self._parsed_sents_cache = OrderedDict((a, self._parse(b))
					for a, b in self._block_cache.iteritems())
		return self._parsed_sents_cache
	def tagged_sents(self):
		""" Return a dictionary of tagged sentences,
		each sentence being a list of (word, tag) tuples. """
		if not self._tagged_sents_cache:
			self._tagged_sents_cache = OrderedDict((a, self._tag(b))
					for a, b in self._block_cache.iteritems())
		return self._tagged_sents_cache
	def sents(self):
		""" Return a dictionary of sentences, each sentence being a list of
		words. """
		if not self._sents_cache:
			self._sents_cache = OrderedDict((a, self._word(b))
					for a, b in self._block_cache.iteritems())
		return self._sents_cache
	def blocks(self, includetransformations=False):
		""" Return a list of strings containing the raw representation of
		trees in the treebank, verbatim or with transformations applied."""
		if includetransformations:
			return OrderedDict((x[2], export(*x)) for x in zip(
				self.parsed_sents().values(), self.sents().values(),
				self.sents().keys(), repeat("export")))
		return OrderedDict((a, "#BOS %s\n%s\n#EOS %s\n" % (a,
				"\n".join("\t".join(c) for c in b), a))
				for a, b in self._block_cache.iteritems())
	def _read_blocks(self):
		def sixelements(a):
			""" take a line and add dummy lemma if that field is not present """
			if "%%" in a:
				a[a.index("%%"):] = []
			lena = len(a)
			if lena == 5:
				return a[:1] + [''] + a[1:]
			elif lena >= 6:
				return a[:6] # skip secondary edges
			else:
				raise ValueError("expected at lest 5 columns: %r" % a)
		result = OrderedDict()
		started = False
		for filename in self._filenames:
			for line in codecs.open(filename, encoding=self._encoding):
				if line.startswith("#BOS"):
					assert not started, ("beginning of sentence marker while "
							"previous one still open: %s" % line)
					started = True
					sentid = line.strip().split()[1]
					lines = []
				elif line.startswith("#EOS"):
					assert started, "end of sentence marker while none started"
					thissentid = line.strip().split()[1]
					assert sentid == thissentid, ("unexpected sentence id: "
							"start=%s, end=%s" % (sentid, thissentid))
					started = False
					assert sentid not in result, (
							"duplicate sentence ID: %s" % sentid)
					result[sentid] = lines
				elif started:
					lines.append(sixelements(line.split()))
		return result
	def _parse(self, s):
		def getchildren(parent, children):
			results = []
			for n, a in children[parent]:
				# n is the index in the block to record word indices
				if EXPORTNONTERMINAL.match(a[WORD]):
					results.append(ParentedTree(a[TAG], getchildren(a[WORD][1:],
						children)))
					results[-1].source = a
				else: #terminal
					results.append(ParentedTree(a[TAG].replace("$(", "$["), [n]))
					results[-1].source = a
			return results
		children = {}
		for n, a in enumerate(s):
			children.setdefault(a[PARENT], []).append((n, a))
		result = ParentedTree("ROOT", getchildren("0", children))
		sent = self._word(s, orig=True)
		# roughly order constituents by order in sentence
		for a in reversed(list(result.subtrees(lambda x: len(x) > 1))):
			a.sort(key=leaves)
		if self.punct == "remove":
			punctremove(result, sent)
		elif self.punct == "move":
			punctraise(result, sent)
			balancedpunctraise(result, sent)
			# restore order
			for a in reversed(list(result.subtrees(lambda x: len(x) > 1))):
				a.sort(key=leaves)
		elif self.punct == "restore":
			punctrestore(result, sent)
		if self.unfold:
			result = unfold(result)
		if self.headrules:
			for node in result.subtrees(lambda n: isinstance(n[0], Tree)):
				sethead(headfinder(node, self.headrules))
				headorder(node, self.headfinal, self.reverse)
		if self.functiontags:
			addfunctions(result)
		new = Tree.convert(result)
		for a, b in zip(new.subtrees(), result.subtrees()):
			if hasattr(b, "source"):
				a.source = b.source
		return new
	def _word(self, s, orig=False):
		if orig or self.punct != "remove":
			return [a[WORD] for a in s if not EXPORTNONTERMINAL.match(a[WORD])]
		return [a[WORD] for a in s if not EXPORTNONTERMINAL.match(a[WORD])
			and not a[TAG].startswith("$")]
	def _tag(self, s):
		if self.punct == "remove":
			return [(a[WORD], a[TAG].replace("$(", "$[")) for a in s
				if not EXPORTNONTERMINAL.match(a[WORD])
				and not a[TAG].startswith("$")]
		return [(a[WORD], a[TAG].replace("$(", "$[")) for a in s
				if not EXPORTNONTERMINAL.match(a[WORD])]

class DiscBracketCorpusReader(object):
	""" A corpus reader where the phrase-structure is represented by a tree in
	bracket notation, where the leaves are indices pointing to words in a
	separately represented sentence; e.g.:
	(S (NP 1) (VP (VB 0) (JJ 2)) (? 3))	is John rich ?
	Note that the tree and the sentence are separated by a tab, while the words
	in the sentence are separated by spaces. There is one tree plus sentence
	per line.
	Compared to Negra's export format, this format lacks morphology, lemmas and
	functional edges. On the other hand, it is very close to the internal
	representation employed here, so it is quick to load.
	"""
	def __init__(self, root, fileids, encoding="utf-8", headrules=None,
			headfinal=False, headreverse=False, dounfold=False,
			functiontags=False, punct=None):
		""" headrules: if given, read rules for assigning heads and apply them
				by ordering constituents according to their heads
			headfinal: whether to put the head in final or in frontal position
			headreverse: the head is made final/frontal by reversing everything
			before or after the head. When true, the side on which the head is
				will be the reversed side.
			dounfold: whether to apply corpus transformations
			functiontags: ignored
			punct: one of ...
				None: leave punctuation as is.
				'move': move punctuation to appropriate constituents
						using heuristics.
				'remove': eliminate punctuation.
				'restore': attach punctuation directly to root
						(as in original Negra/Tiger treebanks). """
		self.reverse = headreverse
		self.headfinal = headfinal
		self.unfold = dounfold
		self.functiontags = functiontags
		self.punct = punct
		self.headrules = readheadrules(headrules) if headrules else {}
		self._encoding = encoding
		self._filenames = glob(os.path.join(root, fileids))
		self._parsed_sents_cache = None
		assert punct in (None, "move", "remove", "restore")
	def sents(self):
		return OrderedDict((n, self._word(line))
				for n, line in self.blocks().iteritems())
	def tagged_sents(self):
		# for each line, zip its words & tags together in a list.
		return OrderedDict((n,
				zip(self._word(line), map(itemgetter(1),
				sorted(Tree.parse(line.split("\t", 1)[0],
					parse_leaf=int).pos()))))
				for n, line in self.blocks().iteritems())
	def parsed_sents(self):
		if not self._parsed_sents_cache:
			self._parsed_sents_cache = OrderedDict(enumerate(
					map(self._parse, self.blocks().itervalues()), 1))
		return self._parsed_sents_cache
	def blocks(self):
		""" Return a list of strings containing the raw representation of
		trees in the treebank, with any transformations applied."""
		return OrderedDict(enumerate(filter(None,
			(line for filename in self._filenames
			for line in codecs.open(filename, encoding=self._encoding))), 1))
	def _parse(self, s):
		result = Tree.parse(s.split("\t", 1)[0], parse_leaf=int)
		sent = self._word(s, orig=True)
		# roughly order constituents by order in sentence
		for a in reversed(list(result.subtrees(lambda x: len(x) > 1))):
			a.sort(key=leaves)
		if self.punct == "remove":
			punctremove(result, sent)
		elif self.punct == "move":
			punctraise(result, sent)
			balancedpunctraise(result, sent)
			# restore order
			for a in reversed(list(result.subtrees(lambda x: len(x) > 1))):
				a.sort(key=leaves)
		elif self.punct == "restore":
			punctrestore(result, sent)
		if self.unfold:
			result = unfold(result)
		if self.headrules:
			for node in result.subtrees(lambda n: isinstance(n[0], Tree)):
				sethead(headfinder(node, self.headrules))
				headorder(node, self.headfinal, self.reverse)
		return result
	def _word(self, s, orig=False):
		sent = s.split("\t", 1)[1].rstrip("\n\r").split(" ")
		if orig or self.punct != "remove":
			return sent
		return [a for a in sent if a not in '.,:;\'"()?!-']

class BracketCorpusReader(object):
	""" A standard corpus reader where the phrase-structure is represented by a
	tree in bracket notation; e.g.:
	(S (NP John) (VP (VB is) (JJ rich)) (. .))
	"""
	def __init__(self, root, fileids, encoding="utf-8", headrules=None,
	headfinal=False, headreverse=False, dounfold=False, functiontags=False,
	punct=None):
		""" headrules: if given, read rules for assigning heads and apply them
				by marking constituents with '^' according to their heads
			headfinal: ignored (for compatibility with the other corpus readers)
			headreverse: ignored
			before or after the head. When true, the side on which the head is
				will be the reversed side.
			dounfold: whether to apply corpus transformations
			functiontags: whether to leaves function tags on node labels (True)
				e.g. NP-SBJ, or strip them away (False).
			punct: one of ...
				None: leave punctuation as is.
				'move': move punctuation to appropriate constituents
						using heuristics.
				'remove': eliminate punctuation.
				'restore': attach punctuation directly to root
						(as in original Negra/Tiger treebanks). """
		self.unfold = dounfold
		self.functiontags = functiontags
		self.punct = punct
		self.headrules = readheadrules(headrules) if headrules else {}
		self._encoding = encoding
		self._filenames = glob(os.path.join(root, fileids))
		self._parsed_sents_cache = None
		self._sents_cache = None
		assert punct in (None, "move", "remove", "restore")
	def sents(self):
		if not self._sents_cache:
			self._sents_cache = OrderedDict((n, self._word(tree))
				for n, tree in self.blocks().iteritems())
		return self._sents_cache
	def tagged_sents(self):
		return OrderedDict((n, [(a, b) for a, (_, b) in zip(sent, tree.pos())])
			for (n, tree), sent
			in zip(self.parsed_sents().iteritems(), self.sents().values()))
	def parsed_sents(self):
		if not self._parsed_sents_cache:
			self._parsed_sents_cache = OrderedDict(enumerate(
					map(self._parse, self.blocks().itervalues()), 1))
		return self._parsed_sents_cache
	def blocks(self):
		""" Return a list of strings containing the raw representation of
		trees in the treebank. """ #, with any transformations applied."""
		return OrderedDict(enumerate(filter(None,
			(line for filename in self._filenames
			for line in codecs.open(filename, encoding=self._encoding))), 1))
	def _parse(self, s):
		c = count()
		result = Tree.parse(s, parse_leaf=lambda _: c.next())
		sent = self._word(s, orig=True)
		if self.punct == "remove":
			punctremove(result, sent)
		elif self.punct == "move":
			punctraise(result, sent)
			balancedpunctraise(result, sent)
			# restore order
			for a in reversed(list(result.subtrees(lambda x: len(x) > 1))):
				a.sort(key=leaves)
		elif self.punct == "restore":
			punctrestore(result, sent)
		if self.unfold:
			result = unfold(result)
		if self.headrules:
			for node in result.subtrees(lambda n: isinstance(n[0], Tree)):
				headmark(headfinder(node, self.headrules))
		if not self.functiontags:
			stripfunctions(result)
		return result
	def _word(self, s, orig=False):
		sent = TERMINALSRE.findall(s)
		if orig or self.punct != "remove":
			return sent
		return [a for a in sent if a not in '...,:;\'"()?!-']

def getreader(fmt):
	""" Return the appropriate corpus reader class given a format string. """
	if fmt == 'export':
		return NegraCorpusReader
	elif fmt == 'discbracket':
		return DiscBracketCorpusReader
	elif fmt == 'bracket':
		return BracketCorpusReader
	else:
		raise ValueError("unrecognized format: %r" % fmt)

indexre = re.compile("\b[0-9]+\b")
def export(tree, sent, n, fmt, headrules=None):
	""" Convert a tree with indices as leafs and a sentence with the
	corresponding non-terminals to a single string in the given format.
	Formats are bracket, discbracket, and Negra's export format,
	as well unlabelled dependency conversion into mst or conll format
	(requires head rules). Lemmas, functions, and morphology information will
	be empty unless nodes contain a 'source' attribute with such information.
	"""
	if fmt == "bracket":
		return indexre.sub(lambda x: sent[int(x.group())], "%s\n" % tree)
	elif fmt == "discbracket":
		return "%s\t%s\n" % (tree, " ".join(sent))
	elif fmt == "export":
		result = []
		if n is not None:
			result.append("#BOS %s" % n)
		indices = tree.treepositions('leaves')
		wordsandpreterminals = indices + [a[:-1] for a in indices]
		phrasalnodes = [a for a in tree.treepositions()
			if a not in wordsandpreterminals and a != ()]
		phrasalnodes.sort(key=len, reverse=True)
		wordids = {tree[a]: a for a in indices}
		assert len(sent) == len(indices) == len(wordids)
		for i, word in enumerate(sent):
			idx = wordids[i]
			a = tree[idx[:-1]]
			result.append("\t".join((word,
					a.label.replace("$[","$("),
					a.source[MORPH] if hasattr(a, "source") else "--",
					a.source[FUNC] if hasattr(a, "source") else "--",
					str(500+phrasalnodes.index(idx[:-2])
						if len(idx) > 2 else 0))))
		for idx in phrasalnodes:
			a = tree[idx]
			result.append("\t".join(("#%d" % (500 + phrasalnodes.index(idx)),
					a.label,
					a.source[MORPH] if hasattr(a, "source") else "--",
					a.source[FUNC] if hasattr(a, "source") else "--",
					str(500+phrasalnodes.index(idx[:-1])
						if len(idx) > 1 else 0))))
		if n is not None:
			result.append("#EOS %s" % n)
		return "%s\n" % "\n".join(result)
	elif fmt in ("conll", "mst"):
		assert headrules, "dependency conversion requires head rules."
		deps = dependencies(tree, headrules)
		if fmt == "mst": # MST parser can read this format
			# fourth line with function tags is left empty.
			return "\n".join((
				"\t".join(word for word in sent),
				"\t".join(tag for _, tag in sorted(tree.pos())),
				"\t".join(str(head) for _, _, head in deps))) + "\n\n"
		elif fmt == "conll":
			return "\n".join("%d\t%s\t_\t%s\t%s\t_\t%d\t%s\t_\t_" % (
				n, word, tag, tag, head, rel)
				for word, (_, tag), (rel, n, head)
				in zip(sent, sorted(tree.pos()), deps)) + "\n\n"
	else:
		raise ValueError("unrecognized format: %r" % fmt)

def addfunctions(tree):
	""" Add function tags to phrasal labels e.g., 'VP' => 'VP-HD'. """
	# FIXME: functions on pos tags should be optional.
	for a in tree.subtrees():
		if a.source[FUNC].split("-")[0]:
			a.label += "+%s" % a.source[FUNC].split("-")[0]

def stripfunctions(tree):
	""" Remove function tags from phrasal labels e.g., 'VP-HD' => 'VP' """
	for a in tree.subtrees():
		x = a.label.find("-")
		y = a.label.find("=")
		if x >= 1:
			a.label = a.label[:x]
		if y >= 0:
			a.label = a.label[:y]
		if a.label[0] != "-":
			a.label = a.label.split("-")[0].split("=")[0]

def readheadrules(filename):
	""" Read a file containing heuristic rules for head assigment.
	The file containing head assignment rules for negra is part of rparse,
	under src/de/tuebingen/rparse/treebank/constituent/negra/
	Example line: "s right-to-left vmfin vafin vaimp", which means
	traverse siblings of an S constituent from right to left, the first child
	with a label of vmfin, vafin, or vaimp will be marked as head.
	"""
	headrules = {}
	for line in open(filename):
		line = line.strip().upper()
		if line and not line[0].startswith("%") and len(line.split()) > 2:
			label, lr, heads = line.split(None, 2)
			headrules.setdefault(label, []).append((lr, heads.split()))
	if "VROOT" in headrules:
		headrules["ROOT"] = headrules["VROOT"]
	return headrules

def headfinder(tree, headrules):
	""" use head finding rules to select one child of tree as head. """
	candidates = [a for a in tree if hasattr(a, "source")
			and "HD" in a.source[FUNC].split("-")]
	if candidates:
		return candidates[0]
	for lr, heads in headrules.get(tree.label, []):
		if lr == "LEFT-TO-RIGHT":
			children = tree
		elif lr == "RIGHT-TO-LEFT":
			children = tree[::-1]
		else:
			raise ValueError
		for head in heads:
			for child in children:
				if isinstance(child, Tree) and child.label == head:
					return child
	# default head is initial nonterminal
	for child in tree:
		if isinstance(child, Tree):
			return child

def sethead(child):
	""" mark node as head in an auxiliary field. """
	child.source = getattr(child, "source", 6 * [''])
	if "-HD" in child.source[FUNC]:
		pass
	elif not child.source[FUNC] or child.source[FUNC] == "--":
		child.source[FUNC] = "HD"
	else:
		child.source[FUNC] += "-HD"

def headmark(tree):
	""" add marker to label of head node. """
	head = [a for a in tree if hasattr(a, "source")
			and "HD" in a.source[FUNC].split("-")]
	if not head:
		return
	head[-1].label += "-HD"

def headorder(tree, headfinal, reverse):
	""" change order of constituents based on head (identified with
	function tag). """
	head = [n for n, a in enumerate(tree)
		if hasattr(a, "source") and "HD" in a.source[FUNC].split("-")]
	if not head:
		return
	headidx = head.pop()
	# everything until the head is reversed and prepended to the rest,
	# leaving the head as the first element
	nodes = tree[:]
	tree[:] = []
	if headfinal:
		if reverse:
			# head final, reverse rhs: A B C^ D E => A B E D C^
			tree[:] = nodes[:headidx] + nodes[headidx:][::-1]
		else:
			# head final, reverse lhs:  A B C^ D E => E D A B C^
			tree[:] = nodes[headidx+1:][::-1] + nodes[:headidx+1]
	else:
		if reverse:
			# head first, reverse lhs: A B C^ D E => C^ B A D E
			tree[:] = nodes[:headidx+1][::-1] + nodes[headidx+1:]
		else:
			# head first, reverse rhs: A B C^ D E => C^ D E B A
			tree[:] = nodes[headidx:] + nodes[:headidx][::-1]

def dependencies(root, headrules):
	""" Lin (1995): A Dependency-based Method
	for Evaluating Broad-Coverage Parser """
	deps = []
	deps.append(("ROOT", makedep(root, deps, headrules), 0))
	return sorted(deps)

def makedep(root, deps, headrules):
	""" Traverse a tree marking heads and extracting dependencies. """
	if not isinstance(root[0], Tree):
		return root[0] + 1
	headchild = headfinder(root, headrules)
	lexhead = makedep(headchild, deps, headrules)
	for child in root:
		if child is headchild:
			continue
		lexheadofchild = makedep(child, deps, headrules)
		deps.append(("NONE", lexheadofchild, lexhead))
	return lexhead

def getgeneralizations():
	# generalizations suggested by SyntaxGeneralizer of TigerAPI
	# however, instead of renaming, we introduce unary productions
	# POS tags
	tonp = "NN NNE PNC PRF PDS PIS PPER PPOS PRELS PWS".split()
	#topp = "PROAV PWAV".split()  # ??
	#toap = "ADJA PDAT PIAT PPOSAT PRELAT PWAT PRELS".split()
	#toavp = "ADJD ADV".split()

	tagtoconst = {}
	for label in tonp:
		tagtoconst[label] = "NP"
	#for label in toap:
	#	tagtoconst[label] = "AP"
	#for label in toavp:
	#	tagtoconst[label] = "AVP"

	# phrasal categories
	tonp = "CNP NM PN".split()
	#topp = "CPP".split()
	#toap = "MTA CAP".split()
	#toavp = "AA CAVP".split()
	unaryconst = {}
	for label in tonp:
		unaryconst[label] = "NP"
	return tagtoconst, unaryconst

def function(tree):
	if hasattr(tree, "source"):
		return tree.source[FUNC].split("-")[0]
	else:
		return ''

def ishead(tree):
	if hasattr(tree, "source"):
		return "HD" in tree.source[FUNC]
	else:
		return False

def rindex(l, v):
	""" Like list.index(), but go from right to left. """
	return len(l) - 1 - l[::-1].index(v)

def labels(tree):
	return [a.label for a in tree if isinstance(a, Tree)]

def unfold(tree):
	""" Unfold redundancies and perform other transformations introducing
	more hierarchy in the phrase-structure annotation, based on
	grammatical functions and syntactic categories.
	"""
	# for debugging:
	#original = tree.copy(Tree)
	#current = tree
	def pop(a):
		try:
			return a.parent.pop(a.parent_index)
		except AttributeError:
			return a
	tagtoconst, unaryconst = getgeneralizations()

	# un-flatten PPs
	addtopp = "AC".split()
	for pp in tree.subtrees(lambda n: n.label == "PP"):
		ac = [a for a in pp if function(a) in addtopp]
		# anything before an initial preposition goes to the PP (modifiers,
		# punctuation), otherwise it goes to the NP; mutatis mutandis for
		# postpositions
		functions = map(function, pp)
		if "AC" in functions and "NK" in functions:
			if functions.index("AC") < functions.index("NK"):
				ac[:0] = pp[:functions.index("AC")]
			if rindex(functions, "AC") > rindex(functions, "NK"):
				ac += pp[rindex(functions, "AC")+1:]
		#else:
		#	print "PP but no AC or NK", " ".join(functions)
		nk = [a for a in pp if a not in ac]
		# introduce a PP unless there is already an NP in the PP (annotation
		# mistake), or there is a PN and we want to avoid a cylic unary of
		# NP -> PN -> NP
		if ac and nk and (len(nk) > 1 or nk[0].label not in "NP PN".split()):
			pp[:] = []
			pp[:] = ac + [ParentedTree("NP", nk)]

	# introduce DPs
	#determiners = set("ART PDS PDAT PIS PIAT PPOSAT PRELS PRELAT "
	#	"PWS PWAT PWAV".split())
	determiners = {"ART"}
	for np in list(tree.subtrees(lambda n: n.label == "NP")):
		if np[0].label in determiners:
			np.label = "DP"
			if len(np) > 1 and np[1].label != "PN":
				np1 = np[1:]
				np[1:] = []
				np[1:] = [ParentedTree("NP", np1)]

	# VP category split based on head
	for vp in tree.subtrees(lambda n: n.label == "VP"):
		hd = [x for x in vp if ishead(x)].pop()
		vp.label += "-" + hd.label

	# introduce finite VP at S level, collect objects and modifiers
	# introduce new S level for discourse markers
	newlevel = "DM".split()
	addtovp = "HD AC DA MO NG OA OA2 OC OG PD VO SVP".split()
	def finitevp(s):
		if any(x.label.startswith("V") and x.label.endswith("FIN")
			for x in s if isinstance(x, Tree)):
			vp = ParentedTree("VP", [pop(a) for a in s if function(a) in addtovp])
			# introduce a VP unless it would lead to a unary VP -> VP production
			if len(vp) != 1 or vp[0].label != "VP":
				s[:] = [pop(a) for a in s if function(a) not in addtovp] + [vp]
	# relative clause => S becomes SRC
	for s in tree.subtrees(lambda n: n.label == "S" and function(n) == "RC"):
		s.label = "SRC"
	toplevel_s = []
	if "S" in labels(tree):
		toplevel_s = [a for a in tree if a.label == "S"]
		for s in toplevel_s:
			while function(s[0]) in newlevel:
				s[:] = [s[0], ParentedTree("S", s[1:])]
				s = s[1]
				toplevel_s = [s]
	elif "CS" in labels(tree):
		cs = tree[labels(tree).index("CS")]
		toplevel_s = [a for a in cs if a.label == "S"]
	map(finitevp, toplevel_s)

	# introduce POS tag for particle verbs
	for a in tree.subtrees(lambda n: "SVP" in map(function, n)):
		svp = [x for x in a if function(x) == "SVP"].pop()
		#apparently there can be a _verb_ particle without a verb.
		#headlines? annotation mistake?
		if any(map(ishead, a)):
			hd = [x for x in a if ishead(x)].pop()
			if hd.label != a.label:
				particleverb = ParentedTree(hd.label, [hd, svp])
				a[:] = [particleverb if ishead(x) else x
									for x in a if function(x) != "SVP"]

	# introduce SBAR level
	sbarfunc = "CP".split()
	# in the annotation, complementizers belong to the first S
	# in S conjunctions, even when they appear to apply to the whole
	# conjunction.
	for s in list(tree.subtrees(lambda n: n.label == "S"
			and function(n[0]) in sbarfunc and len(n) > 1)):
		s.label = "SBAR"
		s[:] = [s[0], ParentedTree("S", s[1:])]

	# introduce nested structures for modifiers
	# (iterated adjunction instead of sister adjunction)
	#adjunctable = set("NP".split()) # PP AP VP
	#for a in list(tree.subtrees(lambda n: n.label in adjunctable
	#		and any(function(x) == "MO" for x in n))):
	#	modifiers = [x for x in a if function(x) == "MO"]
	#	if min(n for n, x in enumerate(a) if function(x) =="MO") == 0:
	#		modifiers[:] = modifiers[::-1]
	#	while modifiers:
	#		modifier = modifiers.pop()
	#		a[:] = [ParentedTree(a.label, [x for x in a if x != modifier]), modifier]
	#		a = a[0]

	# restore linear precedence ordering
	for a in tree.subtrees(lambda n: len(n) > 1):
		a.sort(key=lambda n: n.leaves())
	# introduce phrasal projections for single tokens (currently only adds NPs)
	for a in tree.treepositions("leaves"):
		tag = tree[a[:-1]]   # e.g. NN
		const = tree[a[:-2]] # e.g. S
		if tag.label in tagtoconst and const.label != tagtoconst[tag.label]:
			tag[:] = [ParentedTree(tag.label, [tag[0]])]	# NN -> NN -> word
			tag.label = tagtoconst[tag.label]		# NP -> NN -> word
	return tree

def fold(tree):
	""" Undo the transformations performed by unfold. Do not apply twice
	(might remove VPs which shouldn't be). """
	tagtoconst, unaryconst = getgeneralizations()

	# restore linear precedence ordering
	for a in tree.subtrees(lambda n: len(n) > 1):
		a.sort(key=lambda n: n.leaves())
	#original = tree.copy(True)
	#current = tree

	# remove DPs
	for dp in tree.subtrees(lambda n: n.label == "DP"):
		dp.label = "NP"
		if len(dp) > 1 and dp[1].label == "NP":
			#dp1 = dp[1][:]
			#dp[1][:] = []
			#dp[1:] = dp1
			dp[1][:], dp[1:] = [], dp[1][:]
	# flatten adjunctions
	#nkonly = set("PDAT CAP PPOSS PPOSAT ADJA FM PRF NM NN NE "
	#	"PIAT PRELS PN TRUNC CH CNP PWAT PDS VP CS CARD ART PWS PPER".split())
	#probably_nk = set("AP PIS".split()) | nkonly
	#for np in tree.subtrees(lambda n: len(n) == 2
	#								and n.label == "NP"
	#								and [x.label for x in n].count("NP") == 1
	#								and not set(labels(n)) & probably_nk):
	#	np.sort(key=lambda n: n.label == "NP")
	#	np[:] = np[:1] + np[1][:]

	# flatten PPs
	for pp in tree.subtrees(lambda n: n.label == "PP"):
		if "NP" in labels(pp) and "NN" not in labels(pp):
			#ensure NP is in last position
			pp.sort(key=lambda n: n.label == "NP")
			pp[-1][:], pp[-1:] = [], pp[-1][:]
	# SRC => S, VP-* => VP
	for s in tree.subtrees(lambda n: n.label == "SRC"):
		s.label = "S"
	for vp in tree.subtrees(lambda n: n.label.startswith("VP-")):
		vp.label = "VP"

	# merge extra S level
	for sbar in list(tree.subtrees(lambda n: n.label == "SBAR"
		or (n.label == "S" and len(n) == 2 and labels(n) == ["PTKANT", "S"]))):
		sbar.label = "S"
		if sbar[0].label == "S":
			sbar[:] = sbar[1:] + sbar[0][:]
		else:
			sbar[:] = sbar[:1] + sbar[1][:]

	# merge finite VP with S level
	def mergevp(s):
		for vp in (n for n, a in enumerate(s) if a.label == "VP"):
			if any(a.label.endswith("FIN") for a in s[vp]):
				s[vp][:], s[vp:vp+1] = [], s[vp][:]
	#if any(a.label == "S" for a in tree):
	#	map(mergevp, [a for a in tree if a.label == "S"])
	#elif any(a.label == "CS" for a in tree):
	#	map(mergevp, [s for cs in tree for s in cs if cs.label == "CS"
	#		and s.label == "S"])
	for s in tree.subtrees(lambda n: n.label == "S"):
		mergevp(s)

	# remove constituents for particle verbs
	# get the grandfather of each verb particle
	hasparticle = lambda n: any("PTKVZ" in (x.label
			for x in m if isinstance(x, Tree)) for m in n if isinstance(m, Tree))
	for a in list(tree.subtrees(hasparticle)):
		for n, b in enumerate(a):
			if (len(b) == 2 and b.label.startswith("V")
				and "PTKVZ" in (c.label for c in b if isinstance(c, Tree))
				and any(c.label == b.label for c in b)):
				a[n:n+1] = b[:]

	# remove phrasal projections for single tokens
	for a in tree.treepositions("leaves"):
		tag = tree[a[:-1]]    # NN
		const = tree[a[:-2]]  # NP
		parent = tree[a[:-3]] # PP
		if (len(const) == 1 and tag.label in tagtoconst
				and const.label == tagtoconst[tag.label]):
			parent[a[-3]] = const.pop(0)
			del const
	# restore linear precedence ordering
	for a in tree.subtrees(lambda n: len(n) > 1):
		a.sort(key=lambda n: n.leaves())
	return tree

def bracketings(tree):
	""" Labelled bracketings of a tree. """
	return [(a.label, tuple(sorted(a.leaves())))
		for a in tree.subtrees(lambda t: t and isinstance(t[0], Tree))]

def labelfunc(tree):
	""" Add grammatical function to phrasal label of all nodes in tree. """
	for a in tree.subtrees():
		a.label += "-" + function(a)
	return tree

punctre = re.compile(r"[-.,'\"`:()\[\]/?!]|\.\.\.|''|``")
def punctremove(tree, sent):
	for a in reversed(tree.treepositions("leaves")):
		if tree[a[:-1]].label.startswith("$") or punctre.match(sent[tree[a]]):
			# remove this punctuation node and any empty ancestors
			for n in range(1, len(a)):
				del tree[a[:-n]]
				if tree[a[:-(n + 1)]]:
					break
	#renumber
	oldleaves = sorted(tree.leaves())
	newleaves = {a: n for n, a in enumerate(oldleaves)}
	for a in tree.treepositions("leaves"):
		tree[a] = newleaves[tree[a]]
	assert sorted(tree.leaves()) == range(len(tree.leaves())), tree


def punctrestore(tree, sent):
	""" Move punctuation directly under the ROOT node, as in the
	original Negra/Tiger treebanks. """
	punct = []
	for a in reversed(tree.treepositions("leaves")):
		if tree[a[:-1]].label.startswith("$") or punctre.match(sent[tree[a]]):
			# store punctuation node
			punct.append(tree[a[:-1]])
			# remove this punctuation node and any empty ancestors
			for n in range(1, len(a)):
				del tree[a[:-n]]
				if tree[a[:-(n + 1)]]:
					break
	tree.extend(punct)

def renumber(tree):
	""" Renumber indices after terminals have been removed. """
	indices = sorted(tree.leaves())
	newleaves = {}
	shift = 0
	for n, a in enumerate(indices):
		if a > 1 and a - 1 not in indices:
			shift += 1
		newleaves[a] = n + shift
	for a in tree.treepositions("leaves"):
		tree[a] = newleaves[tree[a]]

def punctlower(tree, sent):
	""" Find suitable constituent for punctuation marks and add it there;
	removal at previous location is up to the caller.  Based on rparse code.
	Initial candidate is the root node."""
	def lower(node, candidate):
		num = node.leaves()[0]
		for i, child in enumerate(sorted(candidate, key=lambda x: x.leaves())):
			if not isinstance(child[0], Tree):
				continue
			termdom = child.leaves()
			if num < min(termdom):
				print "moving", node, "under", candidate.label
				candidate.insert(i + 1, node)
				break
			elif num < max(termdom):
				lower(node, child)
				break
	for a in tree.treepositions("leaves"):
		if tree[a[:-1]].label.startswith("$") or punctre.match(sent[tree[a]]):
			b = tree[a[:-1]]
			del tree[a[:-1]]
			lower(b, tree)

def punctraise(tree, sent):
	""" Trees in the Negra corpus have punctuation attached to the root;
	i.e., it is not part of the phrase-structure.  This function attaches
	punctuation nodes (that is, a POS tag with punctuation terminal) to an
	appropriate constituent. """
	punct = list(tree.subtrees(lambda n: n.label.startswith("$")
			or (isinstance(n[0], int) and punctre.match(sent[n[0]]))))
	while punct:
		node = punct.pop()
		# dedicated punctation node
		if all(a.label.startswith("$") for a in node.parent): # ??
			continue
		node.parent.pop(node.parent_index)
		phrasalnode = lambda x: len(x) and isinstance(x[0], Tree)
		for candidate in tree.subtrees(phrasalnode):
			# add punctuation mark next to biggest constituent which it borders
			#if any(node[0] - 1 in borders(sorted(a.leaves())) for a in candidate):
			#if any(node[0] - 1 == max(leaves(a)) for a in candidate):
			if any(node[0] + 1 == min(leaves(a)) for a in candidate):
				candidate.append(node)
				break
		else:
			tree.append(node)

def balancedpunctraise(tree, sent):
	""" Move balanced punctuation marks " ' - ( ) [ ] together in the same
	constituent. Based on rparse code.

	>>> tree = ParentedTree.parse("(ROOT ($, 3) ($[ 7) ($[ 13) ($, 14) ($, 20)"
	... " (S (NP (ART 0) (ADJA 1) (NN 2) (NP (CARD 4) (NN 5) (PP (APPR 6) "
	... "(CNP (NN 8) (ADV 9) (ISU ($. 10) ($. 11) ($. 12))))) (S (PRELS 15) "
	... "(MPN (NE 16) (NE 17)) (ADJD 18) (VVFIN 19))) (VVFIN 21) (ADV 22) "
	... "(NP (ADJA 23) (NN 24))) ($. 25))", parse_leaf=int)
	>>> sent = ("Die zweite Konzertreihe , sechs Abende mit ' Orgel plus "
	... ". . . ' , die Hayko Siemens musikalisch leitet , bietet wieder "
	... "ungewoehnliche Kombinationen .".split())
	>>> punctraise(tree, sent)
	>>> balancedpunctraise(tree, sent)
	>>> from treetransforms import slowfanout
	>>> print max(map(slowfanout, tree.subtrees()))
	1
	>>> nopunct = Tree.parse("(ROOT (S (NP (ART 0) (ADJA 1) (NN 2) (NP (CARD 3)"
	... " (NN 4) (PP (APPR 5) (CNP (NN 6) (ADV 7)))) (S (PRELS 8) (MPN (NE 9) "
	... "(NE 10)) (ADJD 11) (VVFIN 12))) (VVFIN 13) (ADV 14) (NP (ADJA 15) "
	... "(NN 16))))", parse_leaf=int)
	>>> print max(map(slowfanout, nopunct.subtrees()))
	1
	"""
	match = {'"': '"', '[': ']', '(': ')', "-": "-", "'": "'"}
	punctmap = {}
	termparent = dict(zip(leaves(tree), preterminals(tree)))
	#assert isinstance(tree, ParentedTree)
	for preterminal in sorted(termparent.itervalues(), key=itemgetter(0)):
		terminal = preterminal[0]
		if (not preterminal.label.startswith("$")
				and not punctre.match(sent[terminal])):
			continue
		if sent[terminal] in punctmap:
			right = terminal
			left = punctmap[sent[right]]
			rightparent = preterminal.parent
			leftparent = termparent[left].parent
			if max(leaves(leftparent)) == right - 1:
				node = termparent[right]
				leftparent.append(node.parent.pop(node.parent_index))
			elif min(leaves(rightparent)) == left + 1:
				node = termparent[left]
				rightparent.insert(0, node.parent.pop(node.parent_index))
			if sent[right] in punctmap:
				del punctmap[sent[right]]
		elif sent[terminal] in match:
			punctmap[match[sent[terminal]]] = terminal

def leaves(node):
	""" Return the leaves of the tree. Non-recursive version. """
	queue, theleaves = deque(node), []
	while queue:
		node = queue.popleft()
		if isinstance(node, Tree):
			queue.extend(node)
		else:
			theleaves.append(node)
	return theleaves

def preterminals(node):
	""" Return the preterminal nodes of the tree. Non-recursive version. """
	queue, preterms = deque(node), []
	while queue:
		node = queue.popleft()
		if all(isinstance(a, Tree) for a in node):
			queue.extend(node)
		else:
			preterms.append(node)
	return preterms

### Unknown word models ###
def replacerarewords(sents, unknownword, unknownthreshold):
	""" Replace all terminals that occur less than unknownthreshold times
	with a signature of features as returned by unknownword().
	Returns list of words that exceed the threshold (i.e., the lexicon). """
	words = multiset(word for sent in sents for word in sent)
	lexicon = {a for a, b in words.iteritems() if b > unknownthreshold}
	for sent in sents:
		for n, word in enumerate(sent):
			if words[word] <= unknownthreshold:
				sent[n] = unknownword(word, n, lexicon)
	return lexicon

hasdigit = re.compile(r"\d", re.UNICODE)
hasnondigit = re.compile(r"\D", re.UNICODE)
#NB: includes hypen, em-dash, en-dash, &c
hasdash = re.compile(r"[-\u2010-\u2015]", re.UNICODE)
haslower = re.compile(u"[a-zçéàìùâêîôûëïüÿœæ]", re.UNICODE)
hasupper = re.compile(u"[a-zçéàìùâêîôûëïüÿœæ]", re.UNICODE)
hasletter = re.compile(u"[A-Za-zçéàìùâêîôûëïüÿœæÇÉÀÌÙÂÊÎÔÛËÏÜŸŒÆ]", re.UNICODE)
def unknownword6(word, loc, lexicon):
	""" Model 6 of the Stanford parser (for WSJ treebank). """
	wlen = len(word)
	numcaps = 0
	sig = "UNK"
	numcaps = len(hasupper.findall(word))
	lowered = word.lower()
	if numcaps > 1:
		sig += "-CAPS"
	elif numcaps > 0:
		if loc == 0:
			sig += "-INITC"
			if lowered in lexicon:
				sig += "-KNOWNLC"
		else:
			sig += "-CAP"
	elif haslower.search(word):
		sig += "-LC"
	if hasdigit.search(word):
		sig += "-NUM"
	if hasdash.search(word):
		sig += "-DASH"
	if lowered.endswith("s") and wlen >= 3:
		if lowered[-2] not in "siu":
			sig += "-s"
	elif wlen >= 5 and not hasdash.search(word) and not (
			hasdigit.search(word) and numcaps > 0):
		suffixes = ("ed", "ing", "ion", "er", "est", "ly", "ity", "y", "al")
		for a in suffixes:
			if lowered.endswith(a):
				sig += "-%s" % a
	return sig

def unknownword4(word, loc, lexicon):
	""" Model 4 of the Stanford parser. Relatively language agnostic. """
	# Cf. http://en.wikipedia.org/wiki/French_alphabet
	lower = u"abcdefghijklmnopqrstuvwxyzçéàìùâêîôûëïüÿœæ"
	upper = u"ABCDEFGHIJKLMNOPQRSTUVWXYZÇÉÀÌÙÂÊÎÔÛËÏÜŸŒÆ"
	lowerupper = lower + upper
	sig = "UNK"

	# letters
	if word[0] in upper:
		if not haslower:
			sig += "-AC"
		elif loc == 0:
			sig += "-SC"
		else:
			sig += "-C"
	elif haslower.search(word):
		sig += "-L"
	elif hasletter.search(word):
		sig += "-U"
	else:
		sig += "-S" # no letter

	# digits
	if hasdigit.search(word):
		if hasnondigit.search(word):
			sig += "-n"
		else:
			sig += "-N"

	# punctuation
	if "-" in word:
		sig += "-H"
	if "." in word:
		sig += "-P"
	if "," in word:
		sig += "-C"
	if len(word) > 3:
		if word[-1] in lowerupper:
			sig += "-%s" % word[-1].lower()
	return sig

nounsuffix = re.compile(u"(ier|ière|ité|ion|ison|isme|ysme|iste|esse|eur|euse"
		u"|ence|eau|erie|ng|ette|age|ade|ance|ude|ogue|aphe|ate|duc|anthe|archie"
		u"|coque|érèse|ergie|ogie|lithe|mètre|métrie|odie|pathie|phie|phone"
		u"|phore|onyme|thèque|scope|some|pole|ôme|chromie|pie)s?$")
adjsuffix = re.compile(u"(iste|ième|uple|issime|aire|esque|atoire|ale|al|able"
		u"|ible|atif|ique|if|ive|eux|aise|ent|ois|oise|ante|el|elle|ente|oire"
		u"|ain|aine)s?$")
possibleplural = re.compile(u"(s|ux)$")
verbsuffix = re.compile(u"(ir|er|re|ez|ont|ent|ant|ais|ait|ra|era|eras|é|és|ées"
		u"|isse|it)$")
advsuffix = re.compile(u"(iment|ement|emment|amment)$")
haspunc = re.compile(u"([\u0021-\u002F\u003A-\u0040\u005B\u005C\u005D"
		u"\u005E-\u0060\u007B-\u007E\u00A1-\u00BF\u2010-\u2027\u2030-\u205E"
		u"\u20A0-\u20B5])+")
ispunc = re.compile(u"([\u0021-\u002F\u003A-\u0040\u005B\u005C\u005D"
		u"\u005E-\u0060\u007B-\u007E\u00A1-\u00BF\u2010-\u2027\u2030-\u205E"
		u"\u20A0-\u20B5])+$")

def unknownwordftb(word, loc, lexicon):
	""" Model 2 for French of the Stanford parser. """
	# Cf. http://en.wikipedia.org/wiki/French_alphabet
	upper = u"ABCDEFGHIJKLMNOPQRSTUVWXYZÇÉÀÌÙÂÊÎÔÛËÏÜŸŒÆ"
	sig = "UNK"
	# unicode hack.
	if isinstance(word, str):
		word = word.decode("utf-8")

	if advsuffix.search(word):
		sig += "-ADV"
	elif verbsuffix.search(word):
		sig += "-VB"
	elif nounsuffix.search(word):
		sig += "-NN"

	if adjsuffix.search(word):
		sig += "-ADV"
	if hasdigit.search(word):
		sig += "-NUM"
	if possibleplural.search(word):
		sig += "-PL"

	if ispunc.search(word):
		sig += "-ISPUNC"
	elif haspunc.search(word):
		sig += "-HASPUNC"

	if loc > 0 and len(word) > 0 and word[0] in upper:
		sig += "-UP"

	return sig


def puncttest():
	""" Verify that punctuation movement does not increase fan-out. """
	from treetransforms import slowfanout as fanout
	filename = 'sample2.export' #'negraproc.export'
	mangledtrees = NegraCorpusReader(".", filename, headrules=None,
			encoding="iso-8859-1", punct="move")
	nopunct = NegraCorpusReader(".", filename, headrules=None,
			encoding="iso-8859-1", punct="remove").parsed_sents().values()
	originals = NegraCorpusReader(".", filename, headrules=None,
			encoding="iso-8859-1").parsed_sents().values()
	phrasal = lambda x: len(x) and isinstance(x[0], Tree)
	for n, mangled, sent, nopunct, original in zip(count(),
			mangledtrees.parsed_sents().values(), mangledtrees.sents().values(),
			nopunct, originals):
		print n,
		for a, b in zip(sorted(mangled.subtrees(phrasal),
			key=lambda n: min(leaves(n))),
			sorted(nopunct.subtrees(phrasal), key=lambda n: min(leaves(n)))):
			if fanout(a) != fanout(b):
				print " ".join(sent)
				print mangled
				print nopunct
				print original
			assert fanout(a) == fanout(b), "%d %d\n%s\n%s" % (
				fanout(a), fanout(b), a, b)

def main():
	"""" Test whether the Tiger transformations (fold / unfold) are
	reversible. """
	from treetransforms import canonicalize
	# this fixes utf-8 output when piped through e.g. less
	# won't help if the locale is not actually utf-8, of course
	sys.stdout = codecs.getwriter('utf8')(sys.stdout)
	headrules = "negra.headrules"
	n = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1",
			headrules=headrules)
	nn = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1",
			headrules=headrules, dounfold=True)
	print "\nunfolded"
	correct = exact = d = 0
	nk = set()
	mo = set()
	fnk = multiset()
	fmo = multiset()
	for a, b, c in zip(n.parsed_sents().values()[:100],
			nn.parsed_sents().values()[:100], n.sents().values()[:100]):
		#if len(c) > 15:
		#	continue
		for x in a.subtrees(lambda n: n.label == "NP"):
			nk.update(y.label for y in x if function(y) == "NK")
			mo.update(y.label for y in x if function(y) == "MO")
			fnk.update(y.label for y in x if function(y) == "NK")
			fmo.update(y.label for y in x if function(y) == "MO")
		foldb = fold(b.copy(True))
		b1 = bracketings(canonicalize(a))
		b2 = bracketings(canonicalize(foldb))
		z = -1 #825
		if b1 != b2 or d == z:
			precision = len(set(b1) & set(b2)) / float(len(set(b1)))
			recall = len(set(b1) & set(b2)) / float(len(set(b2)))
			if precision != 1.0 or recall != 1.0 or d == z:
				print d, " ".join(":".join((str(n),
					a.encode('unicode-escape'))) for n, a in enumerate(c))
				print "no match", precision, recall
				print len(b1), len(b2), "gold-fold", set(b2) - set(b1),
				print "fold-gold", set(b1) - set(b2)
				print labelfunc(a)
				print foldb
				print b
			else:
				correct += 1
		else:
			exact += 1
			correct += 1
		d += 1
	print "matches", correct, "/", d, 100 * correct / float(d), "%"
	print "exact", exact
	print
	print "nk & mo", " ".join(nk & mo)
	print "nk - mo", " ".join(nk - mo)
	print "mo - nk", " ".join(mo - nk)
	for x in nk & mo:
		print x, "as nk", fnk[x], "as mo", fmo[x]
	puncttest()

if __name__ == '__main__':
	main()
