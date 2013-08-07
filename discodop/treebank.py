""" Read and write treebanks. """
from __future__ import division, print_function, unicode_literals
import io
import os
import re
import xml.etree.cElementTree as ElementTree
from glob import glob
from itertools import count, chain
from collections import OrderedDict, Counter as multiset
from operator import itemgetter
from .tree import Tree, ParentedTree
from .treebanktransforms import punctremove, punctraise, balancedpunctraise, \
		punctroot, ispunct
FIELDS = tuple(range(8))
WORD, LEMMA, TAG, MORPH, FUNC, PARENT, SECEDGETAG, SECEDGEPARENT = FIELDS
POSRE = re.compile(r"\(([^() ]+) [^ ()]+\)")
TERMINALSRE = re.compile(r" ([^ ()]+)\)")
EXPORTNONTERMINAL = re.compile(r"^#([0-9]+)$")
LEAVESRE = re.compile(r" ([^ ()]*)\)")
FRONTIERNTRE = re.compile(r" \)")


class CorpusReader(object):
	""" Abstract corpus reader. """
	def __init__(self, root, fileids, encoding='utf-8', headrules=None,
				headfinal=True, headreverse=False, markheads=False, punct=None,
				functions=None, morphology=None):
		""" headrules: if given, read rules for assigning heads and apply them
				by ordering constituents according to their heads
			headfinal: whether to put the head in final or in frontal position
			headreverse: the head is made final/frontal by reversing everything
				before or after the head. When true, the side on which the head
				is will be the reversed side.
			markheads: add '^' to phrasal label of heads.
			punct: one of ...
				None: leave punctuation as is.
				'move': move punctuation to appropriate constituents
						using heuristics.
				'remove': eliminate punctuation.
				'root': attach punctuation directly to root
						(as in original Negra/Tiger treebanks).
			functions: one of ...
				None: leave syntactic labels as is.
				'add': concatenate grammatical function to syntactic label,
					separated by a hypen: e.g., NP => NP-SBJ
				'remove': strip away hyphen-separated grammatical function,
					e.g., NP-SBJ => NP
				'replace': replace syntactic label with grammatical function,
					e.g., NP => SBJ
			morphology: one of ...
				None: use POS tags as preterminals
				'add': concatenate morphological information to POS tags,
					e.g., DET/sg.def
				'replace': use morphological information as preterminal label
				'between': add node with morphological information between
					POS tag and word, e.g., (DET (sg.def the)) """
		self.reverse = headreverse
		self.headfinal = headfinal
		self.markheads = markheads
		self.functions = functions
		self.punct = punct
		self.morphology = morphology
		self.headrules = readheadrules(headrules) if headrules else {}
		self._encoding = encoding
		if fileids == '':
			fileids = '*'
		self._filenames = sorted(glob(os.path.join(root, fileids)), key=numbase)
		assert functions in (None, 'leave', 'add', 'remove', 'replace'), (
				functions)
		assert punct in (None, 'move', 'remove', 'root')
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
			self._parsed_sents_cache = OrderedDict((a, self._parsetree(b))
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

	def blocks(self):
		""" Return a list of strings containing the raw representation of
		trees in the original treebank."""

	def _read_blocks(self):
		""" No-op. For line-oriented formats re-reading is cheaper than
		caching. """

	def _parse(self, block):
		""" Return a parse tree given a string from the treebank file. """

	def _parsetree(self, block):
		""" Return a transformed parse tree. """
		tree, sent = self._parse(block)
		if not sent:
			return tree
		# roughly order constituents by order in sentence
		for a in reversed(list(tree.subtrees(lambda x: len(x) > 1))):
			a.sort(key=Tree.leaves)
		if self.punct == "remove":
			punctremove(tree, sent)
		elif self.punct == "move":
			punctraise(tree, sent)
			balancedpunctraise(tree, sent)
			# restore order
			for a in reversed(list(tree.subtrees(lambda x: len(x) > 1))):
				a.sort(key=Tree.leaves)
		elif self.punct == "root":
			punctroot(tree, sent)
		if self.headrules:
			for node in tree.subtrees(lambda n: n and isinstance(n[0], Tree)):
				sethead(headfinder(node, self.headrules))
				headorder(node, self.headfinal, self.reverse)
				if self.markheads:
					headmark(node)
		handlefunctions(self.functions, tree)
		return tree

	def _word(self, block, orig=False):
		""" Return a list of words given a string.
		When orig is True, return original sentence verbatim;
		otherwise it will follow parameters for punctuation. """


class NegraCorpusReader(CorpusReader):
	""" Read a corpus in the Negra export format. """
	def blocks(self):
		return OrderedDict((a, "#BOS %s\n%s\n#EOS %s\n" % (a,
				"\n".join("\t".join(c) for c in b), a))
				for a, b in self._block_cache.items())

	def _read_blocks(self):
		""" Read corpus and return list of blocks corresponding to each
		sentence."""
		result = OrderedDict()
		started = False
		for filename in self._filenames:
			for line in io.open(filename, encoding=self._encoding):
				if line.startswith('#BOS '):
					assert not started, ("beginning of sentence marker while "
							"previous one still open: %s" % line)
					started = True
					sentid = line.strip().split()[1]
					lines = []
				elif line.startswith('#EOS '):
					assert started, "end of sentence marker while none started"
					thissentid = line.strip().split()[1]
					assert sentid == thissentid, ("unexpected sentence id: "
							"start=%s, end=%s" % (sentid, thissentid))
					started = False
					assert sentid not in result, (
							"duplicate sentence ID: %s" % sentid)
					result[sentid] = lines
				elif started:
					lines.append(exportsplit(line))
		return result

	def _parse(self, block):
		tree = exportparse(block, self.morphology)
		sent = self._word(block, orig=True)
		return tree, sent

	def _word(self, block, orig=False):
		if orig or self.punct != "remove":
			return [a[WORD] for a in block
					if not EXPORTNONTERMINAL.match(a[WORD])]
		return [a[WORD] for a in block if not EXPORTNONTERMINAL.match(a[WORD])
			and not ispunct(a[WORD], a[TAG])]


class TigerXMLCorpusReader(CorpusReader):
	""" Corpus reader for the Tiger XML format. """
	def blocks(self):
		raise NotImplementedError

	def _read_blocks(self):
		raise NotImplementedError
		#results = OrderedDict()
		#for filename in self._filenames:
		#	# use iterparse()
		#	block = ElementTree.parse(filename).getroot()
		#	for sent in block.get('body').findall('s'):
		#		results[sent.get('id')] = sent
		#return results

	def _parse(self, block):
		raise NotImplementedError

	def _word(self, block, orig=False):
		raise NotImplementedError


class DiscBracketCorpusReader(CorpusReader):
	""" A corpus reader where the phrase-structure is represented by a tree in
	bracket notation, where the leaves are indices pointing to words in a
	separately represented sentence; e.g.:
	(S (NP 1) (VP (VB 0) (JJ 2)) (? 3))	is John rich ?
	Note that the tree and the sentence are separated by a tab, while the words
	in the sentence are separated by spaces. There is one tree and sentence
	per line.
	Compared to Negra's export format, this format lacks morphology, lemmas and
	functional edges. On the other hand, it is very close to the internal
	representation employed here, so it can be read efficiently. """
	def sents(self):
		return OrderedDict((n, self._word(line))
				for n, line in self.blocks().items())

	def parsed_sents(self):
		if not self._parsed_sents_cache:
			self._parsed_sents_cache = OrderedDict(enumerate(
					map(self._parsetree, self.blocks().values()), 1))
		return self._parsed_sents_cache

	def blocks(self):
		return OrderedDict(enumerate(filter(None,
			(line for filename in self._filenames
			for line in io.open(filename, encoding=self._encoding))), 1))

	def _parse(self, block):
		result = ParentedTree.parse(block.split("\t", 1)[0], parse_leaf=int)
		sent = self._word(block, orig=True)
		return result, sent

	def _word(self, block, orig=False):
		sent = block.split("\t", 1)[1].rstrip("\n\r").split(' ')
		if orig or self.punct != "remove":
			return sent
		return [a for a in sent if not ispunct(a, None)]


class BracketCorpusReader(CorpusReader):
	""" A standard corpus reader where the phrase-structure is represented by a
	tree in bracket notation; e.g.:
	(S (NP John) (VP (VB is) (JJ rich)) (. .))
	TODO: support traces & empty nodes. """
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
					map(self._parsetree, self.blocks().values()), 1))
		return self._parsed_sents_cache

	def blocks(self):
		return OrderedDict(enumerate((line for filename in self._filenames
			for line in io.open(filename, encoding=self._encoding) if line), 1))

	def _parse(self, block):
		c = count()
		result = ParentedTree.parse(block, parse_leaf=lambda _: next(c))
		if result.label not in ('TOP', 'ROOT'):
			result = ParentedTree('TOP', [result])
		sent = self._word(block, orig=True)
		return result, sent

	def _word(self, block, orig=False):
		sent = TERMINALSRE.findall(block)
		if orig or self.punct != "remove":
			return sent
		return [a for a in sent if not ispunct(a, None)]


class AlpinoCorpusReader(CorpusReader):
	""" Corpus reader for the Dutch Alpino treebank in XML format. """

	def blocks(self):
		""" Return a list of strings containing the raw representation of
		trees in the treebank, verbatim or with transformations applied."""
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
			block = ElementTree.parse(filename).getroot()
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
			""" Traverse Alpino XML tree and create Tree object. """
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
				source[SECEDGETAG] = node.get('rel')  # NB: same relation as head
			if 'cat' in node.keys():
				label = node.get('cat')
				children = []
				for child in node:
					if 'word' in child.keys() or 'cat' in child.keys():
						subtree = getsubtree(child)
						subtree.source[PARENT] = node.get('id')
						subtree.source = tuple(subtree.source)
						children.append(subtree)
				result = ParentedTree(label.upper(), children)
			else:  # leaf node
				assert 'word' in node.keys()
				children = list(range(int(node.get('begin')),
						int(node.get('end'))))
				result = ParentedTree('', children)
				handlemorphology(self.morphology, result, source)
			assert children, node.tostring()
			result.source = source
			return result
		# NB: in contrast to Negra export format, don't need to add
		# root/top node
		result = getsubtree(block.find('node'))
		sent = self._word(block)
		return result, sent

	def _word(self, block, orig=False):
		""" Return a list of words given a string.
		When orig is True, return original sentence verbatim;
		otherwise it will follow parameters for punctuation. """
		return block.find('sentence').text.split()


def numbase(key):
	""" Turn a file name into a numeric sorting key if possible. """
	path, base = os.path.split(key)
	base = base.split(".", 1)
	try:
		base[0] = int(base[0])
	except ValueError:
		pass
	return [path] + base


def exportsplit(line):
	""" take a line in export format and split into fields,
	add dummy fields lemma, sec. edge if those fields are absent. """
	if "%%" in line:  # we don't want comments.
		line = line[:line.index("%%")]
	fields = line.split()
	fieldlen = len(fields)
	if fieldlen == 5:
		fields[1:1] = ['']
		fields.extend(['', ''])
	elif fieldlen == 6:
		fields.extend(['', ''])
	elif fieldlen < 8 or fieldlen & 1:
		# NB: zero or more sec. edges come in pairs of parent id and label
		raise ValueError(
				'expected 5 or 6+ even number of columns: %r' % fields)
	return fields


def exportparse(block, morphology=None):
	""" Given a tree in export format as a list of lists,
	construct a Tree object for it. """
	def getchildren(parent):
		""" Traverse tree in export format and create Tree object. """
		results = []
		for n, source in children[parent]:
			# n is the index in the block to record word indices
			m = EXPORTNONTERMINAL.match(source[WORD])
			if m:
				child = ParentedTree(source[TAG], getchildren(m.group(1)))
			else:  # POS + terminal
				child = ParentedTree('', [n])
				handlemorphology(morphology, child, source)
			child.source = tuple(source)
			results.append(child)
		return results

	children = {}
	for n, source in enumerate(block):
		children.setdefault(source[PARENT], []).append((n, source))
	result = ParentedTree("ROOT", getchildren("0"))
	return result


def getreader(fmt):
	""" Return the appropriate corpus reader class given a format string. """
	return {'export': NegraCorpusReader,
		'discbracket': DiscBracketCorpusReader,
		'bracket': BracketCorpusReader,
		'alpino': AlpinoCorpusReader}[fmt]

indexre = re.compile(r" [0-9]+\)")


def writetree(tree, sent, n, fmt, headrules=None, morph=None):
	""" Convert a tree with indices as leafs and a sentence with the
	corresponding non-terminals to a single string in the given format.
	Formats are bracket, discbracket, and Negra's export format,
	as well unlabelled dependency conversion into mst or conll format
	(requires head rules). Lemmas, functions, and morphology information will
	be empty unless nodes contain a 'source' attribute with such information.
	"""
	def getword(idx):
		""" Get word given an index and restore parentheses. """
		return sent[int(idx[:-1])].replace('(', '-LRB-').replace(')', '-RRB-')
	if fmt == "alpino":
		fmt = "export"  # FIXME implement Alpino XML output?

	if fmt == "bracket":
		return indexre.sub(lambda x: ' %s)' % getword(x.group()),
				"%s\n" % tree)
	elif fmt == "discbracket":
		return "%s\t%s\n" % (tree, ' '.join(sent))
	elif fmt == "export":
		result = []
		if n is not None:
			result.append("#BOS %s" % n)
		indices = tree.treepositions('leaves')
		wordsandpreterminals = indices + [a[:-1] for a in indices]
		phrasalnodes = [a for a in tree.treepositions('postorder')
				if a not in wordsandpreterminals and a != ()]
		wordids = {tree[a]: a for a in indices}
		assert len(sent) == len(indices) == len(wordids)
		for i, word in enumerate(sent):
			assert word, 'empty word in sentence: %r' % sent
			idx = wordids[i]
			node = tree[idx[:-1]]
			postag = node.label.replace('$[', '$(') or '--'
			func = morph = '--'
			if hasattr(node, 'source'):
				morph = node.source[MORPH] or '--'
				func = node.source[FUNC] or '--'
			if morph == '--':
				morph = node.label if morph == 'replace' else '--'
			nodeid = str(500 + phrasalnodes.index(idx[:-2])
				if len(idx) > 2 else 0)
			result.append("\t".join((word, postag, morph, func, nodeid)))
		for idx in phrasalnodes:
			node = tree[idx]
			parent = '#%d' % (500 + phrasalnodes.index(idx))
			label = node.label or '--'
			func = morph = '--'
			if hasattr(node, 'source'):
				morph = node.source[MORPH] or '--'
				func = node.source[FUNC] or '--'
			nodeid = str(500 + phrasalnodes.index(idx[:-1])
					if len(idx) > 1 else 0)
			result.append('\t'.join((parent, label, morph, func, nodeid)))
		if n is not None:
			result.append("#EOS %s" % n)
		return "%s\n" % "\n".join(result)
	elif fmt in ("conll", "mst"):
		assert headrules, "dependency conversion requires head rules."
		deps = dependencies(tree, headrules)
		if fmt == "mst":  # MST parser can read this format
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


def handlefunctions(action, tree, pos=True, top=False):
	""" Add function tags to phrasal labels e.g., 'VP' => 'VP-HD'.
	action: one of {None, 'add', 'replace', 'remove'}
	pos: whether to add function tags to POS tags.
	top: whether to add function tags to the top node."""
	if action in (None, 'leave'):
		return
	for a in tree.subtrees():
		if action == 'remove':
			# e.g., NP-SUBJ or NP=2 => NP, but don't touch -NONE-
			x = a.label.find('-')
			if x > 0:
				a.label = a.label[:x]
			x = a.label.find('=')
			if x > 0:
				a.label = a.label[:x]
		else:
			if not top and a is tree:  # skip TOP label
				continue
			if pos or isinstance(a[0], Tree):
				# test for non-empty function tag ("---" is considered empty)
				if hasattr(a, "source") and any(a.source[FUNC].split("-")):
					func = a.source[FUNC].split("-")[0].upper()
					if action == 'add':
						a.label += "-%s" % func
					elif action == 'replace':
						a.label = func


def handlemorphology(action, preterminal, source):
	""" Given a preterminal, augment or replace its label with morphological
	information. """
	# escape any parentheses to avoid hassles w/bracket notation of trees
	tag = source[TAG].replace('(', '[').replace(')', ']')
	morph = source[MORPH].replace('(', '[').replace(')', ']')
	if action in (None, 'no'):
		preterminal.label = tag
	elif action == 'add':
		preterminal.label = '%s/%s' % (tag, morph)
	elif action == 'replace':
		preterminal.label = morph
	elif action == 'between':
		preterminal[:] = preterminal.__class__(preterminal.label, preterminal)
		preterminal.label = morph
	else:
		raise ValueError
	return preterminal


def readheadrules(filename):
	""" Read a file containing heuristic rules for head assigment.
	The file containing head assignment rules for negra is part of rparse,
	under src/de/tuebingen/rparse/treebank/constituent/negra/
	Example line: "s right-to-left vmfin vafin vaimp", which means
	traverse siblings of an S constituent from right to left, the first child
	with a label of vmfin, vafin, or vaimp will be marked as head. """
	headrules = {}
	for line in open(filename):
		line = line.strip().upper()
		if line and not line.startswith("%") and len(line.split()) > 2:
			label, lr, heads = line.split(None, 2)
			headrules.setdefault(label, []).append((lr, heads.split()))
	return headrules


def headfinder(tree, headrules, headlabels=frozenset({'HD'})):
	""" use head finding rules to select one child of tree as head. """
	candidates = [a for a in tree if hasattr(a, 'source')
			and headlabels.intersection(a.source[FUNC].upper().split('-'))]
	if candidates:
		return candidates[0]
	for lr, heads in headrules.get(tree.label, []):
		if lr == 'LEFT-TO-RIGHT':
			children = tree
		elif lr == 'RIGHT-TO-LEFT':
			children = tree[::-1]
		else:
			raise ValueError
		for head in heads:
			for child in children:
				if (isinstance(child, Tree)
						and child.label.split('[')[0] == head):
					return child
	# default head is initial nonterminal
	for child in tree:
		if isinstance(child, Tree):
			return child


def sethead(child):
	""" mark node as head in an auxiliary field. """
	child.source = getattr(child, "source", 6 * [''])
	if 'HD' not in child.source[FUNC].upper().split("-"):
		x = list(child.source)
		if child.source[FUNC] in (None, '', '--'):
			x[FUNC] = '-HD'
		else:
			x[FUNC] = x[FUNC] + '-HD'
		child.source = tuple(x)


def headmark(tree):
	""" add marker to label of head node. """
	head = [a for a in tree if hasattr(a, 'source')
			and 'HD' in a.source[FUNC].upper().split('-')]
	if not head:
		return
	head[-1].label += '-HD'


def headorder(tree, headfinal, reverse):
	""" change order of constituents based on head (identified with
	function tag). """
	head = [n for n, a in enumerate(tree)
		if hasattr(a, "source") and 'HD' in a.source[FUNC].upper().split("-")]
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
			tree[:] = nodes[headidx + 1:][::-1] + nodes[:headidx + 1]
	else:
		if reverse:
			# head first, reverse lhs: A B C^ D E => C^ B A D E
			tree[:] = nodes[:headidx + 1][::-1] + nodes[headidx + 1:]
		else:
			# head first, reverse rhs: A B C^ D E => C^ D E B A
			tree[:] = nodes[headidx:] + nodes[:headidx][::-1]


def saveheads(tree, tailmarker):
	""" When a head-outward binarization is used, this function ensures the
	head is known when the tree is converted to export format. """
	if not tailmarker:
		return
	for node in tree.subtrees(lambda n: tailmarker in n.label):
		node.source = ['--'] * 6
		node.source[FUNC] = 'HD'


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


def incrementaltreereader(treeinput, morphology=None, functions=None):
	""" Incremental corpus reader support brackets, discbrackets, and export
	format. The format is autodetected. Expects an iterator giving one line at
	a time. Yields tuples with a Tree object and a separate lists of terminals.
	"""
	treeinput = chain(iter(treeinput), ('(', None))  # hack
	line = next(treeinput)
	# try the following readers; on the first match the others are dropped.
	readers = [segmentbrackets('()'), segmentbrackets('[]'),
			segmentexport(morphology, functions)]
	for reader in readers:
		reader.send(None)
	x = -1
	while True:
		# status 0: line not part of tree; status 1: waiting for end of tree.
		res, status = None, 1
		for n, reader in enumerate(readers if x == -1 else readers[x:x + 1]):
			while res is None:
				res, status = reader.send(line)
				if status == 0:
					# do not give this line to this reader anymore.
					break
				line = next(treeinput)
			if res is not None:
				if x == -1:
					x = n
				for tree, sent in res:
					yield tree, sent
				break
		if line is None:  # this was the last line
			return
		if res is None:  # none of the readers accepted this line
			line = next(treeinput)


def segmentbrackets(brackets):
	""" Co-routine that accepts one line at a time;
	yields tuples (result, status) where
	result is None or one or more S-expressions as a list of
		tuples (tree, rest), where rest is the string outside of brackets
		between this S-expression and the next.
	status is 1 if the line was consumed, else 0. """
	lb, rb = brackets
	strtermre = re.compile('[^0-9\\%s]\\%s' % (rb, rb))
	parens = 0  # number of open parens
	prev = ''  # pass on if tree is not yet complete in current line
	result = ''  # tree as string
	results = []  # trees found in current line
	line = (yield None, 1)
	while True:
		start = 0  # index where current tree starts
		a, b = line.find(lb, len(prev)), line.find(rb, len(prev))
		prev = line
		while a != -1 or b != -1:
			if a != -1 and (a < b or b == -1):  # left bracket
				if parens == 0:
					rest, prev = line[start:a], line[a:]
					if result:
						results.append(
								brackettree(result, rest, brackets, strtermre))
						result = ''
						start = a
				parens += 1
				a = line.find(lb, a + 1)
			elif b != -1 and (b < a or a == -1):  # right bracket
				parens -= 1
				if parens == 0:
					result, prev = line[start:b + 1], line[b + 1:]
					start = b + 1
				elif parens < 0:
					#raise ValueError('unbalanced parentheses')
					parens = 0
				b = line.find(rb, b + 1)
		status = 1 if results or result or parens else 0
		line = (yield results or None, status)
		if results:
			results = []
		if parens or result or results:
			line = prev + line
		else:
			prev = ''


def segmentexport(morphology, functions):
	""" Co-routine that accepts one line at a time.
	Yields: tuples (result, status) where
	result is None or a segment delimited by '#BOS ' and '#EOS '
		as a list of lines;
	status is 1 if the line was consumed, else 0. """
	cur = []
	inblock = False
	line = (yield None, 1)
	while line is not None:
		if line.startswith('#BOS '):
			cur = []
			inblock = True
			line = (yield None, 1)
		elif line.startswith('#EOS '):
			tree, sent = exporttree(cur, morphology)
			handlefunctions(functions, tree)
			line = (yield ((tree, sent), ), 1)
			inblock = False
			cur = []
		elif line.strip():
			if inblock:
				cur.append(line)
			line = (yield None, (1 if inblock else 0))
		else:
			line = (yield None, 0)


def brackettree(treestr, sent, brackets, strtermre):
	""" Parse a single tree presented in bracket format, whether with indices
	or not; sent may be None / empty. """
	if strtermre.search(treestr):  # terminals are not all indices
		treestr = FRONTIERNTRE.sub(' ...)', treestr)
		sent = TERMINALSRE.findall(treestr)
		cnt = count()
		tree = Tree.parse(treestr, brackets=brackets,
				parse_leaf=lambda x: next(cnt))
	else:  # disc. trees with integer indices as terminals
		tree = Tree.parse(treestr, parse_leaf=int,
			brackets=brackets)
		sent = (sent.split() if sent.strip()
				else map(str, range(max(tree.leaves()) + 1)))
	return tree, sent


def exporttree(data, morphology):
	""" Wrapper to get both tree and sentence for tree in export format given
	as list of lines. """
	data = [exportsplit(x) for x in data]
	tree = exportparse(data, morphology)
	sent = []
	for a in data:
		if EXPORTNONTERMINAL.match(a[WORD]):
			break
		sent.append(a[WORD])
	return tree, sent


if __name__ == '__main__':
	pass
