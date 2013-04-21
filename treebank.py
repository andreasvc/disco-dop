# -*- coding: UTF-8 -*-
""" Read and write treebanks. """
from __future__ import division, print_function, unicode_literals
import io, os, re
import xml.etree.cElementTree as ElementTree
from glob import glob
from itertools import count, repeat, islice
from collections import OrderedDict, Counter as multiset
from operator import itemgetter
from tree import Tree, ParentedTree

FIELDS = tuple(range(8))
WORD, LEMMA, TAG, MORPH, FUNC, PARENT, SECEDGETAG, SECEDGEPARENT = FIELDS
POSRE = re.compile(r"\(([^() ]+) [^ ()]+\)")
TERMINALSRE = re.compile(r" ([^ ()]+)\)")
EXPORTNONTERMINAL = re.compile(r"^#([0-9]+)$")

class CorpusReader(object):
	""" Abstract corpus reader. """
	def __init__(self, root, fileids, encoding='utf-8', headrules=None,
				headfinal=True, headreverse=False, markheads=False, punct=None,
				dounfold=False, functiontags=None, morphaspos=False):
		""" headrules: if given, read rules for assigning heads and apply them
				by ordering constituents according to their heads
			headfinal: whether to put the head in final or in frontal position
			headreverse: the head is made final/frontal by reversing everything
				before or after the head. When true, the side on which the head
				is will be the reversed side.
			markheads: add '^' to phrasal label of heads.
			dounfold: whether to apply corpus transformations
			functiontags: if True, add function tags to node labels;
					if False, strip them away if present.
			punct: one of ...
				None: leave punctuation as is.
				'move': move punctuation to appropriate constituents
						using heuristics.
				'remove': eliminate punctuation.
				'root': attach punctuation directly to root
						(as in original Negra/Tiger treebanks).
			morphaspos: use morphological tags as POS tags.
			""" # idea: put morphology tag between POS and word
			# (NN (N[soort,mv,basis] problemen))
		self.reverse = headreverse
		self.headfinal = headfinal
		self.markheads = markheads
		self.unfold = dounfold
		self.functiontags = functiontags
		self.punct = punct
		self.morphaspos = morphaspos
		self.headrules = readheadrules(headrules) if headrules else {}
		self._encoding = encoding
		if fileids == '':
			fileids = '*'
		self._filenames = sorted(glob(os.path.join(root, fileids)), key=numbase)
		assert punct in (None, "move", "remove", "root")
		assert self._filenames, (
				"no files matched pattern %s" % os.path.join(root, fileids))
		self._sents_cache = None
		self._tagged_sents_cache = None
		self._parsed_sents_cache = None
		self._block_cache = self._read_blocks()
	def parsed_sents(self):
		""" Return an ordered dictionary of parse trees
		(Tree objects with integer indices as leaves). """
		if not self._parsed_sents_cache:
			self._parsed_sents_cache = OrderedDict((a, self._parse(b))
					for a, b in self._block_cache.items())
		return self._parsed_sents_cache
	def sents(self):
		""" Return an ordered dictionary of sentences,
		each sentence being a list of words. """
		if not self._sents_cache:
			self._sents_cache = OrderedDict((a, self._word(b))
					for a, b in self._block_cache.items())
		return self._sents_cache
	def tagged_sents(self):
		""" Return an ordered dictionary of tagged sentences,
		each tagged sentence being a list of (word, tag) pairs. """
		if not self._tagged_sents_cache:
			# for each sentence, zip its words & tags together in a list.
			# this assumes that .sents() and .parsed_sents() correctly remove
			# punctuation if requested.
			self._tagged_sents_cache = OrderedDict((n,
				list(zip(sent, map(itemgetter(1), sorted(tree.pos())))))
				for (n, sent), tree in zip(self.sents().items(),
						self.parsed_sents().values()))
		return self._tagged_sents_cache
	def blocks(self, includetransformations=False):
		""" Return a list of strings containing the raw representation of
		trees in the treebank, verbatim or with transformations applied."""
	def _read_blocks(self):
		""" No-op. For line-oriented formats re-reading is cheaper than
		caching. """
	def _parse(self, block):
		""" Return a parse tree given a string. """
	def _word(self, block, orig=False):
		""" Return a list of words given a string.
		When orig is True, return original sentence verbatim;
		otherwise it will follow parameters for punctuation. """

class NegraCorpusReader(CorpusReader):
	""" Read a corpus in the Negra export format. """
	def blocks(self, includetransformations=False):
		if includetransformations:
			return OrderedDict((x[2], writetree(*x)) for x in zip(
				self.parsed_sents().values(), self.sents().values(),
				self.sents().keys(), repeat("export")))
		return OrderedDict((a, "#BOS %s\n%s\n#EOS %s\n" % (a,
				"\n".join("\t".join(c) for c in b), a))
				for a, b in self._block_cache.items())
	def _read_blocks(self):
		""" Read corpus and return list of blocks corresponding to each
		sentence."""
		def normalize(fields):
			""" take a line and add dummy fields (lemma, sec. edge) if those
			fields are absent. """
			if "%%" in fields: # we don't want comments.
				fields[fields.index("%%"):] = []
			lena = len(fields)
			if lena == 5:
				fields[1:1] = ['']
				fields.extend(['', ''])
			elif lena == 6:
				fields.extend(['', ''])
			elif lena == 8:
				pass
			else:
				raise ValueError("expected at least 5 columns: %r" % fields)
			return fields
		result = OrderedDict()
		started = False
		for filename in self._filenames:
			for line in io.open(filename, encoding=self._encoding):
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
					lines.append(normalize(line.split()))
		return result
	def _parse(self, block):
		def getchildren(parent):
			results = []
			for n, source in children[parent]:
				# n is the index in the block to record word indices
				m = EXPORTNONTERMINAL.match(source[WORD])
				if m:
					child = ParentedTree(source[TAG], getchildren(m.group(1)))
				else: # POS + terminal
					# escape Negra's paren tag to avoid hassles
					# w/bracket notation of trees
					label = source[MORPH if self.morphaspos else TAG]
					child = ParentedTree(
							label.replace('(', '[').replace(')', ']'), [n])
				child.source = tuple(source)
				results.append(child)
			return results
		children = {}
		for n, source in enumerate(block):
			children.setdefault(source[PARENT], []).append((n, source))
		result = ParentedTree("ROOT", getchildren("0"))
		sent = self._word(block, orig=True)
		# roughly order constituents by order in sentence
		for a in reversed(list(result.subtrees(lambda x: len(x) > 1))):
			a.sort(key=Tree.leaves)
		if self.punct == "remove":
			punctremove(result, sent)
		elif self.punct == "move":
			punctraise(result, sent)
			balancedpunctraise(result, sent)
			# restore order
			for a in reversed(list(result.subtrees(lambda x: len(x) > 1))):
				a.sort(key=Tree.leaves)
		elif self.punct == "root":
			punctroot(result, sent)
		if self.unfold:
			result = unfold(result)
		if self.headrules:
			for node in result.subtrees(lambda n: n and isinstance(n[0], Tree)):
				sethead(headfinder(node, self.headrules))
				headorder(node, self.headfinal, self.reverse)
				if self.markheads:
					headmark(node)
		if self.functiontags:
			addfunctions(result)
		return result
	def _word(self, block, orig=False):
		if orig or self.punct != "remove":
			return [a[WORD] for a in block
					if not EXPORTNONTERMINAL.match(a[WORD])]
		return [a[WORD] for a in block if not EXPORTNONTERMINAL.match(a[WORD])
			and not ispunct(a[WORD], a[TAG])]

class DiscBracketCorpusReader(CorpusReader):
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
	def sents(self):
		return OrderedDict((n, self._word(line))
				for n, line in self.blocks().items())
	def parsed_sents(self):
		if not self._parsed_sents_cache:
			self._parsed_sents_cache = OrderedDict(enumerate(
					map(self._parse, self.blocks().values()), 1))
		return self._parsed_sents_cache
	def blocks(self, includetransformations=False):
		if includetransformations:
			return ["%s\t%s\n" % (tree, sent)
					for tree, sent in zip(self.parsed_sents(), self.sents())]
		return OrderedDict(enumerate(filter(None,
			(line for filename in self._filenames
			for line in io.open(filename, encoding=self._encoding))), 1))
	def _parse(self, block):
		result = ParentedTree.parse(block.split("\t", 1)[0], parse_leaf=int)
		sent = self._word(block, orig=True)
		# roughly order constituents by order in sentence
		for a in reversed(list(result.subtrees(lambda x: len(x) > 1))):
			a.sort(key=Tree.leaves)
		if self.punct == "remove":
			punctremove(result, sent)
		elif self.punct == "move":
			punctraise(result, sent)
			balancedpunctraise(result, sent)
			# restore order
			for a in reversed(list(result.subtrees(lambda x: len(x) > 1))):
				a.sort(key=Tree.leaves)
		elif self.punct == "root":
			punctroot(result, sent)
		if self.unfold:
			result = unfold(result)
		if self.headrules:
			for node in result.subtrees(lambda n: n and isinstance(n[0], Tree)):
				sethead(headfinder(node, self.headrules))
				headorder(node, self.headfinal, self.reverse)
				if self.markheads:
					headmark(node)
		return result
	def _word(self, block, orig=False):
		sent = block.split("\t", 1)[1].rstrip("\n\r").split(" ")
		if orig or self.punct != "remove":
			return sent
		return [a for a in sent if not ispunct(a, None)]

class BracketCorpusReader(CorpusReader):
	""" A standard corpus reader where the phrase-structure is represented by a
	tree in bracket notation; e.g.:
	(S (NP John) (VP (VB is) (JJ rich)) (. .))
	# TODO: support traces & empty nodes
	"""
	def sents(self):
		if not self._sents_cache:
			self._sents_cache = OrderedDict((n, self._word(tree))
				for n, tree in self.blocks().items())
		return self._sents_cache
	def tagged_sents(self):
		if not self._tagged_sents_cache:
			self._tagged_sents_cache = OrderedDict(
				(n, list(zip(sent, POSRE.findall(block)))) for (n, sent), block
				in zip(self.sents().items(), self.blocks().values()))
		return self._tagged_sents_cache
	def parsed_sents(self):
		if not self._parsed_sents_cache:
			self._parsed_sents_cache = OrderedDict(enumerate(
					map(self._parse, self.blocks().values()), 1))
		return self._parsed_sents_cache
	def blocks(self, includetransformations=False):
		if includetransformations:
			return ["%s\n" % indexre.sub(lambda x: sent[int(x.group())],
					"%s\n" % tree)
					for tree, sent in zip(self.parsed_sents(), self.sents())]
		return OrderedDict(enumerate((line for filename in self._filenames
			for line in io.open(filename, encoding=self._encoding) if line), 1))
	def _parse(self, block):
		c = count()
		result = ParentedTree.parse(block, parse_leaf=lambda _: next(c))
		if result.label not in ('TOP', 'ROOT'):
			result = ParentedTree('TOP', [result])
		sent = self._word(block, orig=True)
		if self.punct == "remove":
			punctremove(result, sent)
		elif self.punct == "move":
			punctraise(result, sent)
			balancedpunctraise(result, sent)
			# restore order
			for a in reversed(list(result.subtrees(lambda x: len(x) > 1))):
				a.sort(key=Tree.leaves)
		elif self.punct == "root":
			punctroot(result, sent)
		if self.unfold:
			result = unfold(result)
		if self.headrules:
			for node in result.subtrees(lambda n: n and isinstance(n[0], Tree)):
				sethead(headfinder(node, self.headrules))
				headorder(node, self.headfinal, self.reverse)
				if self.markheads:
					headmark(node)
		if self.functiontags == False:
			stripfunctions(result)
		return result
	def _word(self, block, orig=False):
		sent = TERMINALSRE.findall(block)
		if orig or self.punct != "remove":
			return sent
		return [a for a in sent if not ispunct(a, None)]

class AlpinoCorpusReader(CorpusReader):
	""" Corpus reader for the dutch Alpino treebank in XML format. """
	def blocks(self, includetransformations=False):
		""" Return a list of strings containing the raw representation of
		trees in the treebank, verbatim or with transformations applied."""
		if includetransformations:
			raise NotImplementedError
		if self._block_cache is None:
			self._block_cache = self._read_blocks()
		return OrderedDict((n, unicode(ElementTree.tostring(a)))
				for n, a in self._block_cache.items())
	def _read_blocks(self):
		""" Read corpus and return list of blocks corresponding to each
		sentence."""
		results = OrderedDict()
		assert self._encoding in (None, 'utf8', 'utf-8'), (
				"Encoding specified in XML files.")
		for filename in self._filenames:
			block = ElementTree.parse(filename
					#io.open(filename, "rt", encoding=self._encoding)
					).getroot()
			#n = s.find('comments')[0].text.split('|', 1)[0], s
			# ../path/dir/file.xml => dir/file
			path, filename = os.path.split(filename)
			_, lastdir = os.path.split(path)
			n = os.path.join(lastdir, filename).rstrip(".xml")
			results[n] = block
		return results
	def _parse(self, block):
		""" Return a parse tree given a string. """
		def getsubtree(node):
			# FIXME: proper representation for arbitrary features
			source = [''] * len(FIELDS)
			source[WORD] = node.get('word') or ("#%s" % node.get('id'))
			source[LEMMA] = node.get('lemma') or node.get('root')
			source[MORPH] = node.get('postag') or node.get('frame')
			if 'cat' in node.keys():
				source[TAG] = node.get('cat')
			else:
				source[TAG] = node.get('pos')
			source[FUNC] = node.get('rel')
			if node.get('index'):
				source[SECEDGEPARENT] = node.get('index')
				source[SECEDGETAG] = node.get('rel') #NB: same relation as head
			if 'cat' in node.keys():
				label = node.get('cat')
				children = []
				for child in node:
					if 'word' in child.keys() or 'cat' in child.keys():
						subtree = getsubtree(child)
						subtree.source[PARENT] = node.get('id')
						subtree.source = tuple(subtree.source)
						children.append(subtree)
				result = ParentedTree(label.upper() , children)
			else: # leaf node
				assert 'word' in node.keys()
				if self.morphaspos:
					label = source[MORPH].replace('(', '[').replace(')', ']')
				else:
					label = source[TAG]
				idx = label.find('[')
				if idx == -1:
					idx = len(label)
				label = ''.join((label[:idx].upper(), label[idx:]))
				children = list(range(int(node.get('begin')),
						int(node.get('end'))))
				result = ParentedTree(label, children)
			assert children, node.tostring()
			result.source = source
			return result
		# NB: in contrast to Negra export format, don't need to add
		# root/top node
		result = getsubtree(block.find('node'))
		sent = self._word(block)
		if not sent:
			return result
		if self.punct == "remove":
			punctremove(result, sent)
		elif self.punct == "move":
			punctraise(result, sent)
			balancedpunctraise(result, sent)
			# restore order
			for a in reversed(list(result.subtrees(lambda x: len(x) > 1))):
				a.sort(key=Tree.leaves)
		elif self.punct == "root":
			punctroot(result, sent)
		if self.headrules:
			headlabels = frozenset({'hd', 'rhd', 'whd'})
			for node in result.subtrees(lambda n: n and isinstance(n[0], Tree)):
				sethead(headfinder(node, self.headrules, headlabels=headlabels))
				headorder(node, self.headfinal, self.reverse)
				if self.markheads:
					headmark(node)
		if self.functiontags:
			addfunctions(result)
		return result
	def _word(self, block, orig=False):
		""" Return a list of words given a string.
		When orig is True, return original sentence verbatim;
		otherwise it will follow parameters for punctuation. """
		return block.find('sentence').text.split()

def getreader(fmt):
	""" Return the appropriate corpus reader class given a format string. """
	if fmt == 'export':
		return NegraCorpusReader
	elif fmt == 'discbracket':
		return DiscBracketCorpusReader
	elif fmt == 'bracket':
		return BracketCorpusReader
	elif fmt == 'alpino':
		return AlpinoCorpusReader
	else:
		raise ValueError("unrecognized format: %r" % fmt)

indexre = re.compile(r" [0-9]+\)")
def writetree(tree, sent, n, fmt, headrules=None):
	""" Convert a tree with indices as leafs and a sentence with the
	corresponding non-terminals to a single string in the given format.
	Formats are bracket, discbracket, and Negra's export format,
	as well unlabelled dependency conversion into mst or conll format
	(requires head rules). Lemmas, functions, and morphology information will
	be empty unless nodes contain a 'source' attribute with such information.
	"""
	def getword(idx):
		word = sent[int(idx[:-1])]
		return word.replace('(', '-LRB-').replace(')', '-RRB-')
	if fmt == "alpino":
		fmt = "export" #FIXME implement Alpino XML output?

	if fmt == "bracket":
		return indexre.sub(lambda x: ' %s)' % getword(x.group()),
				"%s\n" % tree)
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
					str(500 + phrasalnodes.index(idx[:-2])
						if len(idx) > 2 else 0))))
		for idx in phrasalnodes:
			a = tree[idx]
			result.append("\t".join(("#%d" % (500 + phrasalnodes.index(idx)),
					a.label,
					(a.source[MORPH] or "--") if hasattr(a, "source") else "--",
					(a.source[FUNC] or "--") if hasattr(a, "source") else "--",
					str(500 + phrasalnodes.index(idx[:-1])
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

def addfunctions(tree, pos=False, top=False):
	""" Add function tags to phrasal labels e.g., 'VP' => 'VP-HD'.
	pos: whether to add function tags to POS tags.
	top: whether to add function tags to the top node."""
	for a in tree.subtrees():
		if not top and a is tree: # skip TOP label
			continue
		if pos or isinstance(a[0], Tree):
			# test for non-empty function tag (e.g., "---" is considered empty)
			if hasattr(a, "source") and any(a.source[FUNC].split("-")):
				a.label += "-%s" % a.source[FUNC].split("-")[0].upper()

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

def headfinder(tree, headrules, headlabels=frozenset({'HD'})):
	""" use head finding rules to select one child of tree as head. """
	candidates = [a for a in tree if hasattr(a, "source")
			and headlabels.intersection(a.source[FUNC].split("-"))]
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
	if "HD" not in child.source[FUNC].split("-"):
		x = list(child.source)
		if child.source[FUNC] in (None, '', "--"):
			x[FUNC] = "HD"
		else:
			x[FUNC] = x[FUNC] + "-HD"
		child.source = tuple(x)
		# better if 'source' remains unchanged; perhaps:
		#child.head = True

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

def headstats(trees):
	""" collects some information useful for writing headrules. """
	from collections import defaultdict
	heads = defaultdict(multiset)
	pos1 = defaultdict(multiset)
	pos2 = defaultdict(multiset)
	pos3 = defaultdict(multiset)
	unknown = defaultdict(multiset)
	for tree in trees:
		for a in tree.subtrees(lambda x: len(x) > 1):
			for n, b in enumerate(a):
				if 'hd' in b.source[FUNC]:
					heads[a.label][b.label] += 1
					pos1[a.label][n] += 1
					pos2[a.label][len(a) - (n + 2)] += 1
					pos3[a.label][len(a)] += 1
					break
			else:
				unknown[a.label].update(b.label for b in a)
	return heads, unknown, pos1, pos2, pos3

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
		return "HD" in tree.source[FUNC].upper().split("-")
	else:
		return False

def rindex(l, v):
	""" Like list.index(), but go from right to left. """
	return len(l) - 1 - l[::-1].index(v)

def labels(tree):
	return [a.label for a in tree if isinstance(a, Tree)]

def unfold(tree, transformations=('S-RC', 'VP-GF', 'NP')):
	for name in transformations:
		# negra
		if name == 'S-RC': # relative clause => S becomes SRC
			for s in tree.subtrees(lambda n: n.label == "S"
					and function(n) == "RC"):
				s.label = "S-RC"
		elif name == 'VP-GF': # VP category split based on head
			for vp in tree.subtrees(lambda n: n.label == "VP"):
				vp.label += "-" + function(vp)
		elif name == 'NP': # case
			for np in tree.subtrees(lambda n: n.label == "NP"):
				np.label += "-" + function(np)
		# wsj
		elif name == "S-WH":
			for sbar in tree.subtrees(lambda n: n.label == "SBAR"):
				for s in sbar:
					if (s.label == "S"
							and any(a.label.startswith("WH") for a in s)):
						s.label += "-WH"
		elif name == "VP-HD": # VP category split based on head
			for vp in tree.subtrees(lambda n: n.label == "VP"):
				hd = [x for x in vp if ishead(x)].pop()
				if hd.label == 'VB':
					vp.label += '-HINF'
				elif hd.label == 'TO':
					vp.label += '-HTO'
				elif hd.label in ('VBN', 'VBG'):
					vp.label += '-HPART'
		elif name == "S-INF":
			for s in tree.subtrees(lambda n: n.label == "S"):
				hd = [x for x in s if ishead(x)].pop()
				if hd.label in ('VP-HINF', 'VP-HTO'):
					s.label += '-INF'
		# alpino?
		# ...
		else:
			raise ValueError("unrecognized transformation %r" % name)
	return tree

def fold(tree):
	# maybe not necessary, if transforms only add -FUNC.
	for node in tree.subtrees(lambda n: "-" in n.label):
		node.label = node.label[:node.label.index("-")]
	#for a in transformations:
	#	if a == 'SRC':
	#		pass
	#	else:
	#		pass
	return tree

def unfold_orig(tree):
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
		functions = [function(x) for x in pp]
		if "AC" in functions and "NK" in functions:
			if functions.index("AC") < functions.index("NK"):
				ac[:0] = pp[:functions.index("AC")]
			if rindex(functions, "AC") > rindex(functions, "NK"):
				ac += pp[rindex(functions, "AC")+1:]
		#else:
		#	print("PP but no AC or NK", " ".join(functions))
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
			vp = [a for a in s if function(a) in addtovp]
			# introduce a VP unless it would lead to a unary VP -> VP production
			if len(vp) != 1 or vp[0].label != "VP":
				s[:] = [pop(a) for a in s if function(a) not in addtovp] + [
						pop(a) for a in vp]
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
	for a in toplevel_s:
		finitevp(a)

	# introduce POS tag for particle verbs
	for a in tree.subtrees(lambda n: any(function(x) == "SVP" for x in n)):
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

def fold_orig(tree):
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

#relative frequencies of punctuation in Negra: (is XY an annotation error?)
#1/1             $,      ,
#14793/17269     $.      .
#8598/13345      $[      "
#1557/13345      $[      )
#1557/13345      $[      (
#1843/17269      $.      :
#232/2669        $[      -
#343/13345       $[      /
#276/17269       $.      ?
#249/17269       $.      ;
#89/13345        $[      '
#101/17269       $.      !
#2/513           XY      :
#41/13345        $[      ...
#1/513           XY      -
#1/2467          $.      ·      #NB this is not a period but a \cdot ...

PUNCTUATION = ',."()&:-/!!!??;\'```....[]|\xc2\xab\xc2\xbb\\'
def ispunct(word, tag):
	# fixme: treebank specific parameters for detecting punctuation.
	return tag in ('$,', '$.', '$[', '$(',) or word in PUNCTUATION


def punctremove(tree, sent):
	for a in reversed(tree.treepositions("leaves")):
		if ispunct(sent[tree[a]], tree[a[:-1]].label):
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
	assert sorted(tree.leaves()) == list(range(len(tree.leaves()))), tree

def punctroot(tree, sent):
	""" Move punctuation directly under the ROOT node, as in the
	original Negra/Tiger treebanks. """
	punct = []
	for a in reversed(tree.treepositions("leaves")):
		if ispunct(sent[tree[a]], tree[a[:-1]].label):
			# store punctuation node
			punct.append(tree[a[:-1]])
			# remove this punctuation node and any empty ancestors
			for n in range(1, len(a)):
				del tree[a[:-n]]
				if tree[a[:-(n + 1)]]:
					break
	tree.extend(punct)

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
				print("moving", node, "under", candidate.label)
				candidate.insert(i + 1, node)
				break
			elif num < max(termdom):
				lower(node, child)
				break
	for a in tree.treepositions("leaves"):
		if ispunct(sent[tree[a]], tree[a[:-1]].label):
			b = tree[a[:-1]]
			del tree[a[:-1]]
			lower(b, tree)

def punctraise(tree, sent):
	""" Trees in the Negra corpus have punctuation attached to the root;
	i.e., it is not part of the phrase-structure.  This function attaches
	punctuation nodes (that is, a POS tag with punctuation terminal) to an
	appropriate constituent. """
	#punct = [node for node in tree.subtrees() if isinstance(node[0], int)
	punct = [node for node in tree if isinstance(node[0], int)
			and ispunct(sent[node[0]], node.label)]
	while punct:
		node = punct.pop()
		# dedicated punctation node (??)
		if all(isinstance(a[0], int) and ispunct(sent[a[0]], a)
				for a in node.parent):
			continue
		node.parent.pop(node.parent_index)
		phrasalnode = lambda x: len(x) and isinstance(x[0], Tree)
		for candidate in tree.subtrees(phrasalnode):
			# add punctuation mark next to biggest constituent which it borders
			#if any(node[0] - 1 in borders(sorted(a.leaves())) for a in candidate):
			#if any(node[0] - 1 == max(a.leaves()) for a in candidate):
			if any(node[0] + 1 == min(a.leaves()) for a in candidate):
				candidate.append(node)
				break
		else:
			tree.append(node)

BALANCEDPUNCTMATCH = {'"': '"', '[': ']', '(': ')', '-': '-', "'": "'",
		'\xc2\xab': '\xc2\xbb'} # the last ones are unicode for << and >>.
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
	>>> max(map(slowfanout, tree.subtrees()))
	1
	>>> nopunct = Tree.parse("(ROOT (S (NP (ART 0) (ADJA 1) (NN 2) (NP (CARD 3)"
	... " (NN 4) (PP (APPR 5) (CNP (NN 6) (ADV 7)))) (S (PRELS 8) (MPN (NE 9) "
	... "(NE 10)) (ADJD 11) (VVFIN 12))) (VVFIN 13) (ADV 14) (NP (ADJA 15) "
	... "(NN 16))))", parse_leaf=int)
	>>> max(map(slowfanout, nopunct.subtrees()))
	1
	"""
	assert isinstance(tree, ParentedTree)
	# right punct str as key, mapped to left index as value
	punctmap = {}
	# punctuation indices mapped to preterminal nodes
	termparent = {a[0]: a for a in tree.subtrees()
			if a and isinstance(a[0], int) and ispunct(sent[a[0]], a.label)}
	for terminal in sorted(termparent):
		preterminal = termparent[terminal]
		# do we know the matching punctuation mark for this one?
		if sent[terminal] in punctmap:
			right = terminal
			left = punctmap[sent[right]]
			rightparent = preterminal.parent
			leftparent = termparent[left].parent
			if max(leftparent.leaves()) == right - 1:
				node = termparent[right]
				leftparent.append(node.parent.pop(node.parent_index))
			elif min(rightparent.leaves()) == left + 1:
				node = termparent[left]
				rightparent.insert(0, node.parent.pop(node.parent_index))
			if sent[right] in punctmap:
				del punctmap[sent[right]]
		elif sent[terminal] in BALANCEDPUNCTMATCH:
			punctmap[BALANCEDPUNCTMATCH[sent[terminal]]] = terminal

def puncttest():
	""" Verify that punctuation movement does not increase fan-out. """
	from treetransforms import slowfanout as fanout
	filename = 'sample2.export' #'negraproc.export'
	mangledtrees = NegraCorpusReader(".", filename, headrules=None,
			encoding="iso-8859-1", punct="move")
	nopunct = list(NegraCorpusReader(".", filename, headrules=None,
			encoding="iso-8859-1", punct="remove").parsed_sents().values())
	originals = list(NegraCorpusReader(".", filename, headrules=None,
			encoding="iso-8859-1").parsed_sents().values())
	phrasal = lambda x: len(x) and isinstance(x[0], Tree)
	for n, mangled, sent, nopunct, original in zip(count(),
			mangledtrees.parsed_sents().values(), mangledtrees.sents().values(),
			nopunct, originals):
		print(n, end='')
		for a, b in zip(sorted(mangled.subtrees(phrasal),
			key=lambda n: min(n.leaves())),
			sorted(nopunct.subtrees(phrasal), key=lambda n: min(n.leaves()))):
			if fanout(a) != fanout(b):
				print(" ".join(sent))
				print(mangled)
				print(nopunct)
				print(original)
			assert fanout(a) == fanout(b), "%d %d\n%s\n%s" % (
				fanout(a), fanout(b), a, b)
	print()

def numbase(key):
	""" Turn a file name into a numeric sorting key if possible. """
	path, base = os.path.split(key)
	base = base.split(".", 1)
	try:
		base[0] = int(base[0])
	except ValueError:
		pass
	return [path] + base

def alpinotest():
	from treedraw import DrawTree
	t = AlpinoCorpusReader("../Alpino/Treebank/lot_test_suite1", "*.xml")
	for ((n, sent), tree) in zip(
			t.sents().items(),
			t.parsed_sents().values()):
		print(n, tree, sent)
		print(DrawTree(tree, sent).text(unicodelines=True))

def main():
	"""" Test whether the Tiger transformations (fold / unfold) are
	reversible. """
	from treetransforms import canonicalize
	headrules = None #"negra.headrules"
	n = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1",
			headrules=headrules)
	nn = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1",
			headrules=headrules, dounfold=True)
	print("\nunfolded")
	correct = exact = d = 0
	nk = set()
	mo = set()
	fnk = multiset()
	fmo = multiset()
	for a, b, c in islice(zip(n.parsed_sents().values(),
			nn.parsed_sents().values(), n.sents().values()), 100):
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
			precision = len(set(b1) & set(b2)) / len(set(b1))
			recall = len(set(b1) & set(b2)) / len(set(b2))
			if precision != 1.0 or recall != 1.0 or d == z:
				print(d, " ".join(":".join((str(n),
					a.encode('unicode-escape'))) for n, a in enumerate(c)))
				print("no match", precision, recall)
				print(len(b1), len(b2), "gold-fold", set(b2) - set(b1),
						"fold-gold", set(b1) - set(b2))
				print(a)
				print(foldb)
				addfunctions(a)
				print(a)
				print(b)
				print()
			else:
				correct += 1
		else:
			exact += 1
			correct += 1
		d += 1
	print("matches", correct, "/", d, 100 * correct / d, "%")
	print("exact", exact)
	print()
	print("nk & mo", " ".join(nk & mo))
	print("nk - mo", " ".join(nk - mo))
	print("mo - nk", " ".join(mo - nk))
	for x in nk & mo:
		print(x, "as nk", fnk[x], "as mo", fmo[x])

if __name__ == '__main__':
	main()
	puncttest()
	alpinotest()
