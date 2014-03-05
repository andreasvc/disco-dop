"""Read and write treebanks."""
from __future__ import division, print_function, unicode_literals
import io
import os
import re
import xml.etree.cElementTree as ElementTree
from glob import glob
from itertools import count, chain, islice
from collections import defaultdict, OrderedDict
from operator import itemgetter
from discodop.tree import Tree, ParentedTree
from discodop.treebanktransforms import punctremove, punctraise, \
		balancedpunctraise, punctroot, ispunct, readheadrules, headfinder, \
		sethead, headmark, headorder

FIELDS = tuple(range(6))
WORD, LEMMA, TAG, MORPH, FUNC, PARENT = FIELDS
POSRE = re.compile(r"\(([^() ]+) [^ ()]+\)")
TERMINALSRE = re.compile(r" ([^ ()]+)\)")
EXPORTNONTERMINAL = re.compile(r"^#([0-9]+)$")
LEAVESRE = re.compile(r" ([^ ()]*)\)")
FRONTIERNTRE = re.compile(r" \)")
INDEXRE = re.compile(r" [0-9]+\)")


class CorpusReader(object):
	"""Abstract corpus reader."""
	def __init__(self, path, encoding='utf-8', ensureroot=None,
			headrules=None, headfinal=True, headreverse=False, markheads=False,
			punct=None, functions=None, morphology=None, lemmas=None):
		"""
		:param path: filename or pattern of corpus files; e.g., ``wsj*.mrg``
		:param ensureroot: add root node with given label if necessary
		:param headrules: if given, read rules for assigning heads and apply
			them by ordering constituents according to their heads
		:param headfinal: whether to put the head in final or in frontal
			position
		:param headreverse: the head is made final/frontal by reversing
			everything before or after the head. When true, the side on which
			the head is will be the reversed side.
		:param markheads: add '^' to phrasal label of heads.
		:param punct: one of ...
			:None: leave punctuation as is.
			:'move': move punctuation to appropriate constituents
					using heuristics.
			:'remove': eliminate punctuation.
			:'root': attach punctuation directly to root
					(as in original Negra/Tiger treebanks).
		:param functions: one of ...
			:None: leave syntactic labels as is.
			:'add': concatenate grammatical function to syntactic label,
				separated by a hypen: e.g., NP => NP-SBJ
			:'remove': strip away hyphen-separated grammatical function,
				e.g., NP-SBJ => NP
			:'replace': replace syntactic label with grammatical function,
				e.g., NP => SBJ
		:param morphology: one of ...
			:None: use POS tags as preterminals
			:'add': concatenate morphological information to POS tags,
				e.g., DET/sg.def
			:'replace': use morphological information as preterminal label
			:'between': add node with morphological information between
				POS tag and word, e.g., (DET (sg.def the))
		:param lemmas: one of ...
			:None: ignore lemmas
			:'between': insert lemma as node between POS tag and word."""
		self.ensureroot = ensureroot
		self.reverse = headreverse
		self.headfinal = headfinal
		self.markheads = markheads
		self.functions = functions
		self.punct = punct
		self.morphology = morphology
		self.lemmas = lemmas
		self.headrules = readheadrules(headrules) if headrules else {}
		self._encoding = encoding
		self._filenames = sorted(glob(path), key=numbase)
		for opts, opt in {
				(None, 'leave', 'add', 'remove', 'replace'): functions,
				(None, 'leave', 'add', 'replace', 'between'): morphology,
				(None, 'move', 'remove', 'root'): punct,
				(None, 'leave', 'between'): lemmas}.items():
			assert opt in opts, 'Expected one of %r. Got: %r' % (opts, opt)
		assert self._filenames, (
				"no files matched pattern %s" % path)
		self._sents_cache = None
		self._tagged_sents_cache = None
		self._trees_cache = None
		self._block_cache = None

	def trees(self):
		""":returns: an ordered dictionary of parse trees \
			(``Tree`` objects with integer indices as leaves)."""
		if not self._trees_cache:
			if self._block_cache is None:
				self._block_cache = OrderedDict(self._read_blocks())
			self._trees_cache = OrderedDict((a, self._parsetree(b))
					for a, b in self._block_cache.items())
		return self._trees_cache

	def itertrees(self, start=None, end=None):
		""":returns: an iterator returning tuples (key, tree, sent) of \
			sentences in corpus. Useful when the dictionary of all trees in \
			corpus would not fit in memory."""
		for a, b in islice(self._read_blocks(), start, end):
			yield a, self._parsetree(b), self._word(b)

	def sents(self):
		""":returns: an ordered dictionary of sentences, \
			each sentence being a list of words."""
		if not self._sents_cache:
			if self._block_cache is None:
				self._block_cache = OrderedDict(self._read_blocks())
			self._sents_cache = OrderedDict((a, self._word(b))
					for a, b in self._block_cache.items())
		return self._sents_cache

	def tagged_sents(self):
		""":returns: an ordered dictionary of tagged sentences, \
			each tagged sentence being a list of (word, tag) pairs."""
		if not self._tagged_sents_cache:
			if self._block_cache is None:
				self._block_cache = OrderedDict(self._read_blocks())
			# for each sentence, zip its words & tags together in a list.
			# this assumes that .sents() and .trees() correctly remove
			# punctuation if requested.
			self._tagged_sents_cache = OrderedDict((n,
				list(zip(sent, map(itemgetter(1), sorted(tree.pos())))))
				for (n, sent), tree in zip(self.sents().items(),
						self.trees().values()))
		return self._tagged_sents_cache

	def blocks(self):
		""":returns: a list of strings containing the raw representation of \
			trees in the original treebank."""

	def _read_blocks(self):
		"""An iterator over blocks from corpus file that correspond to
		individual annotated sentences."""

	def _parse(self, block):
		""":returns: a parse tree given a string from the treebank file."""

	def _parsetree(self, block):
		""":returns: a transformed parse tree."""
		tree, sent = self._parse(block)
		if not sent:
			return tree
		if self.ensureroot and tree.label != self.ensureroot:
			tree = ParentedTree(self.ensureroot, [tree])
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
		for node in list(tree.subtrees(lambda n: isinstance(n[0], int))):
			handlemorphology(self.morphology, self.lemmas, node, node.source)
		handlefunctions(self.functions, tree)
		return tree

	def _word(self, block, orig=False):
		""":returns: a list of words given a string.

		When orig is True, return original sentence verbatim;
		otherwise it will follow parameters for punctuation."""


class NegraCorpusReader(CorpusReader):
	"""Read a corpus in the Negra export format."""
	def blocks(self):
		if self._block_cache is None:
			self._block_cache = OrderedDict(self._read_blocks())
		return OrderedDict((a, "#BOS %s\n%s\n#EOS %s\n" % (a,
				"\n".join("\t".join(c) for c in b), a))
				for a, b in self._block_cache.items())

	def _read_blocks(self):
		"""Read corpus and yield blocks corresponding to each sentence."""
		results = set()
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
					assert sentid not in results, (
							"duplicate sentence ID: %s" % sentid)
					results.add(sentid)
					yield sentid, lines
				elif started:
					lines.append(exportsplit(line))

	def _parse(self, block):
		tree = exportparse(block)
		sent = self._word(block, orig=True)
		return tree, sent

	def _word(self, block, orig=False):
		if orig or self.punct != "remove":
			return [a[WORD] for a in block
					if not EXPORTNONTERMINAL.match(a[WORD])]
		return [a[WORD] for a in block if not EXPORTNONTERMINAL.match(a[WORD])
			and not ispunct(a[WORD], a[TAG])]


class DiscBracketCorpusReader(CorpusReader):
	"""A corpus reader for discontinuous trees in bracket notation, where the
	leaves are indices pointing to words in a separately specified sentence.

	For example::

		(S (NP 1) (VP (VB 0) (JJ 2)) (? 3))	is John rich ?

	Note that the tree and the sentence are separated by a tab, while the words
	in the sentence are separated by spaces. There is one tree and sentence
	per line. Compared to Negra's export format, this format lacks morphology,
	lemmas and functional edges. On the other hand, it is very close to the
	internal representation employed here, so it can be read efficiently."""
	def sents(self):
		return OrderedDict((n, self._word(line))
				for n, line in self.blocks().items())

	def trees(self):
		if not self._trees_cache:
			self._trees_cache = OrderedDict(enumerate(
					map(self._parsetree, self.blocks().values()), 1))
		return self._trees_cache

	def blocks(self):
		return OrderedDict(self._read_blocks())

	def _read_blocks(self):
		for n, block in enumerate((line for filename in self._filenames
				for line in io.open(filename, encoding=self._encoding)
				if line), 1):
			yield n, block

	def _parse(self, block):
		treestr = block.split("\t", 1)[0]
		tree = ParentedTree.parse(treestr, parse_leaf=int)
		sent = self._word(block, orig=True)
		assert all(0 <= n < len(sent) for n in tree.leaves()), (
				'All leaves must be in the interval 0..n with '
				'n=len(sent)\ntokens: %d indices: %r\nsent: %s' % (
				len(sent), tree.leaves(), sent))
		return tree, sent

	def _word(self, block, orig=False):
		sent = block.split("\t", 1)[1].rstrip("\n\r").split(' ')
		if orig or self.punct != "remove":
			return sent
		return [a for a in sent if not ispunct(a, None)]


class BracketCorpusReader(CorpusReader):
	"""Corpus reader for phrase-structures in bracket notation.

	For example::

		(S (NP John) (VP (VB is) (JJ rich)) (. .))

	TODO: support traces & empty nodes."""
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

	def trees(self):
		if not self._trees_cache:
			self._trees_cache = OrderedDict(enumerate(
					map(self._parsetree, self.blocks().values()), 1))
		return self._trees_cache

	def blocks(self):
		return OrderedDict(self._read_blocks())

	def _read_blocks(self):
		for n, block in enumerate((line for filename in self._filenames
				for line in io.open(filename, encoding=self._encoding)
				if line), 1):
			yield n, block

	def _parse(self, block):
		c = count()
		tree = ParentedTree.parse(LEAVESRE.sub(lambda _: ' %d)' % next(c),
				block), parse_leaf=int)
		sent = self._word(block, orig=True)
		return tree, sent

	def _word(self, block, orig=False):
		sent = [a or None for a in LEAVESRE.findall(block)]
		if orig or self.punct != "remove":
			return sent
		return [a for a in sent if not ispunct(a, None)]


class TigerXMLCorpusReader(CorpusReader):
	"""Corpus reader for the Tiger XML format."""
	def blocks(self):
		""":returns: a list of strings containing the raw representation of \
		trees in the treebank."""
		if self._block_cache is None:
			self._block_cache = OrderedDict(self._read_blocks())
		return OrderedDict((n, ElementTree.tostring(a))
				for n, a in self._block_cache.items())

	def _read_blocks(self):
		for filename in self._filenames:
			# iterator over elements in XML  file
			context = ElementTree.iterparse(filename,
					events=(b'start', b'end'))
			_, root = next(context)  # event == 'start' of root element
			for event, elem in context:
				if event == 'end' and elem.tag == 's':
					yield elem.get('id'), elem
				root.clear()

	def _parse(self, block):
		"""Translate Tiger XML structure to the fields of export format,
		so that code for export format can be used."""
		nodes = OrderedDict()
		root = block.find('graph').get('root')
		for term in block.find('graph').find('terminals'):
			fields = nodes.setdefault(term.get('id'), 6 * [None])
			fields[WORD] = term.get('word')
			fields[LEMMA] = term.get('lemma')
			fields[TAG] = term.get('pos')
			fields[MORPH] = term.get('morph')
			fields[PARENT] = '0' if term.get('id') == root else None
			nodes[term.get('id')] = fields
		for nt in block.find('graph').find('nonterminals'):
			if nt.get('id') == root:
				ntid = '0'
			else:
				fields = nodes.setdefault(nt.get('id'), 6 * [None])
				ntid = nt.get('id').split('_')[-1]
				fields[WORD] = '#' + ntid
				fields[TAG] = nt.get('cat')
				fields[LEMMA] = fields[MORPH] = '--'
			for edge in nt:
				idref = edge.get('idref')
				nodes.setdefault(idref, 6 * [None])
				if edge.tag == 'edge':
					assert nodes[idref][FUNC] is None, ('%s already has '
							'a parent: %r' % (idref, nodes[idref]))
					nodes[idref][FUNC] = edge.get('label')
					nodes[idref][PARENT] = ntid
				elif edge.tag == 'secedge':
					nodes[idref].extend((edge.get('label'), ntid))
				else:
					raise ValueError
		for idref in nodes:
			assert nodes[idref][PARENT] is not None, (
					'%s does not have a parent: %r' % (idref, nodes[idref]))
		tree = exportparse(list(nodes.values()))
		tree.label = nodes[root][TAG]
		sent = self._word(block, orig=True)
		return tree, sent

	def _word(self, block, orig=False):
		if orig or self.punct != "remove":
			return [term.get('word')
				for term in block.find('graph').find('terminals')]
		return [term.get('word')
			for term in block.find('graph').find('terminals')
			if not ispunct(term.get('word'), term.get('pos'))]


class AlpinoCorpusReader(CorpusReader):
	"""Corpus reader for the Dutch Alpino treebank in XML format.

	Expects a corpus in directory format, where every sentence is in a single
	``.xml`` file."""
	def blocks(self):
		""":returns: a list of strings containing the raw representation of \
		trees in the treebank."""
		if self._block_cache is None:
			self._block_cache = OrderedDict(self._read_blocks())
		return self._block_cache

	def _read_blocks(self):
		"""Read corpus and yield blocks corresponding to each sentence."""
		assert self._encoding in (None, 'utf8', 'utf-8'), (
				'Encoding specified in XML files, cannot be overriden.')
		for filename in self._filenames:
			block = open(filename).read()
			# ../path/dir/file.xml => dir/file
			path, filename = os.path.split(filename)
			_, lastdir = os.path.split(path)
			n = os.path.join(lastdir, filename)[:-len('.xml')]
			yield n, block

	def _parse(self, block):
		""":returns: a parse tree given a string."""
		if ElementTree.iselement(block):
			xmlblock = block
		else:  # NB: parse because raw XML might contain entities etc.
			xmlblock = ElementTree.fromstring(block)
		tree = alpinoparse(xmlblock)
		sent = self._word(xmlblock)
		return tree, sent

	def _word(self, block, orig=False):
		if ElementTree.iselement(block):
			xmlblock = block
		else:
			xmlblock = ElementTree.fromstring(block)
		sent = xmlblock.find('sentence').text.split(' ')
		if orig or self.punct != "remove":
			return sent
		return [word for word in sent
				if not ispunct(word, None)]  # fixme: don't have tag


class DactCorpusReader(AlpinoCorpusReader):
	"""Corpus reader for the Dutch Alpino treebank in Dact format,
	which consists of an XML database."""
	def _read_blocks(self):
		import alpinocorpus
		assert self._encoding in (None, 'utf8', 'utf-8'), (
				'Encoding specified in XML files, cannot be overriden.')
		for filename in self._filenames:
			corpus = alpinocorpus.CorpusReader(filename)
			for entry in corpus.entries():
				yield entry.name(), entry.contents()


def exportsplit(line):
	"""Take a line in export format and split into fields,
	add dummy fields lemma, sec. edge if those fields are absent."""
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


def exportparse(block, morphology=None, lemmas=None):
	"""Given a tree in export format as a list of lists,
	construct a Tree object for it."""
	def getchildren(parent):
		"""Traverse tree in export format and create Tree object."""
		results = []
		for n, source in children[parent]:
			# n is the index in the block to record word indices
			m = EXPORTNONTERMINAL.match(source[WORD])
			if m:
				child = ParentedTree(source[TAG], getchildren(m.group(1)))
			else:  # POS + terminal
				child = ParentedTree(source[TAG], [n])
				handlemorphology(morphology, lemmas, child, source)
			child.source = tuple(source)
			results.append(child)
		return results

	children = {}
	for n, source in enumerate(block):
		children.setdefault(source[PARENT], []).append((n, source))
	result = ParentedTree('ROOT', getchildren('0'))
	return result


def alpinoparse(node, morphology=None, lemmas=None):
	"""Construct a Tree object for an ElementTree with an Alpino XML tree."""
	def getsubtree(node, parent, morphology, lemmas):
		"""Parse a subtree of an Alpino tree."""
		# FIXME: proper representation for arbitrary features
		source = [''] * len(FIELDS)
		source[WORD] = node.get('word') or ("#%s" % node.get('id'))
		source[LEMMA] = node.get('lemma') or node.get('root')
		source[MORPH] = node.get('postag') or node.get('frame')
		source[FUNC] = node.get('rel')
		if 'cat' in node.keys():
			source[TAG] = node.get('cat')
			if node.get('index'):
				coindexed[node.get('index')] = source
			label = node.get('cat')
			result = ParentedTree(label.upper(), [])
			for child in node:
				subtree = getsubtree(child, result, morphology, lemmas)
				if subtree and (
						'word' in child.keys() or 'cat' in child.keys()):
					subtree.source[PARENT] = node.get('id')
					result.append(subtree)
			if not len(result):
				return None
		elif 'word' in node.keys():
			source[TAG] = node.get('pt') or node.get('pos')
			if node.get('index'):
				coindexed[node.get('index')] = source
			result = ParentedTree(source[TAG], list(
					range(int(node.get('begin')), int(node.get('end')))))
			handlemorphology(morphology, lemmas, result, source)
		elif 'index' in node.keys():
			coindexation[node.get('index')].extend(
					(node.get('rel'), parent))
			return None
		result.source = source
		return result
	coindexed = {}
	coindexation = defaultdict(list)
	# NB: in contrast to Negra export format, don't need to add root/top node
	result = getsubtree(node.find('node'), None, morphology, lemmas)
	# FIXME: need MultipleParentedTree for secedges
	#for index, secedges in coindexation.items():
	#	coindexed[index].extend(secedges)
	return result


def writetree(tree, sent, n, fmt, headrules=None, morphology=None):
	"""Convert a tree to a string representation in the given treebank format.

	:param tree: should have indices as terminals
	:param sent: contains the words corresponding to the indices in ``tree``
	:param n: an identifier for this tree; part of the output with some formats
	:param fmt: Formats are ``bracket``, ``discbracket``, Negra's ``export``
		format, and ``alpino`` XML format, as well unlabeled dependency
		conversion into ``mst`` or ``conll`` format (requires head rules).
		The formats ``tokens`` and ``wordpos`` are to strip away tree structure
		and leave only lines with tokens or ``token/POS``.

	Lemmas, functions, and morphology information will be empty unless nodes
	contain a 'source' attribute with such information."""
	def getword(idx):
		"""Get word given an index and quote parentheses."""
		return sent[int(idx[:-1])].replace('(', '-LRB-').replace(')', '-RRB-')

	if fmt == "bracket":
		return INDEXRE.sub(lambda x: ' %s)' % getword(x.group()),
				"%s\n" % tree)
	elif fmt == "discbracket":
		return "%s\t%s\n" % (tree, ' '.join(sent))
	elif fmt == "tokens":
		return "%s\n" % ' '.join(sent)
	elif fmt == "wordpos":
		return "%s\n" % ' '.join('%s/%s' % (word, pos) for word, (_, pos)
				in zip(sent, sorted(tree.pos())))
	elif fmt == "export":
		return writeexporttree(tree, sent, n, morphology)
	elif fmt == "alpino":
		return writealpinotree(tree, sent, n)
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


def writeexporttree(tree, sent, n, morphology):
	"""Return string with given tree in Negra's export format."""
	result = []
	if n is not None:
		result.append("#BOS %s" % n)
	indices = tree.treepositions('leaves')
	wordsandpreterminals = indices + [a[:-1] for a in indices]
	phrasalnodes = [a for a in tree.treepositions('postorder')
			if a not in wordsandpreterminals and a != ()]
	wordids = {tree[a]: a for a in indices}
	assert len(sent) == len(indices) == len(wordids), (
			n, str(tree), sent, wordids.keys())
	for i, word in enumerate(sent):
		assert word, 'empty word in sentence: %r' % sent
		idx = wordids[i]
		node = tree[idx[:-1]]
		lemma = '--'
		postag = node.label.replace('$[', '$(') or '--'
		func = morphtag = '--'
		secedges = []
		if getattr(node, 'source', None):
			lemma = node.source[LEMMA] or '--'
			morphtag = node.source[MORPH] or '--'
			func = node.source[FUNC] or '--'
			secedges = node.source[6:]
		if morphtag == '--':
			morphtag = node.label if morphology == 'replace' else '--'
		nodeid = str(500 + phrasalnodes.index(idx[:-2])
				if len(idx) > 2 else 0)
		result.append("\t".join((word, lemma, postag, morphtag, func,
				nodeid) + tuple(secedges)))
	for idx in phrasalnodes:
		node = tree[idx]
		parent = '#%d' % (500 + phrasalnodes.index(idx))
		lemma = '--'
		label = node.label or '--'
		func = morphtag = '--'
		secedges = []
		if getattr(node, 'source', None):
			morphtag = node.source[MORPH] or '--'
			func = node.source[FUNC] or '--'
			secedges = node.source[6:]
		nodeid = str(500 + phrasalnodes.index(idx[:-1])
				if len(idx) > 1 else 0)
		result.append('\t'.join((parent, lemma, label, morphtag, func,
				nodeid) + tuple(secedges)))
	if n is not None:
		result.append("#EOS %s" % n)
	return "%s\n" % "\n".join(result)


def writealpinotree(tree, sent, n):
	"""Return XML string with tree in AlpinoXML format."""
	def addchildren(tree, sent, parent, cnt):
		"""Recursively add children of ``tree`` to XML object ``node``"""
		node = ElementTree.SubElement(parent, 'node')
		node.set('id', str(next(cnt)))
		node.set('begin', str(min(tree.leaves())))
		node.set('end', str(max(tree.leaves()) + 1))
		if getattr(tree, 'source', None):
			node.set('rel', tree.source[FUNC] or '--')
		if isinstance(tree[0], Tree):
			node.set('cat', tree.label.lower())
		else:
			assert isinstance(tree[0], int)
			node.set('pos', tree.label.lower())
			node.set('word', sent[tree[0]])
			node.set('lemma', '--')
			node.set('postag', '--')
			if getattr(tree, 'source', None):
				node.set('lemma', tree.source[LEMMA] or '--')
				node.set('postag', tree.source[MORPH] or '--')
				# TODO: decompose postag into individual morphological features
				# could give 'frame' and 'root' attributes
				# with data from 'postag' and 'lemma'.
		for child in tree:
			if isinstance(child, Tree):
				addchildren(child, sent, node, cnt)

	result = ElementTree.Element('alpino_ds')
	result.set('version', '1.3')
	addchildren(tree, sent, result, count())
	sentence = ElementTree.SubElement(result, 'sentence')
	sentence.text = ' '.join(sent)
	comment = ElementTree.SubElement(result, 'comment')
	comment.text = str(n)
	return ElementTree.tostring(result).decode('utf-8')  # hack


def handlefunctions(action, tree, pos=True, top=False):
	"""Add function tags to phrasal labels e.g., 'VP' => 'VP-HD'.

	:param action: one of {None, 'add', 'replace', 'remove'}
	:param pos: whether to add function tags to POS tags.
	:param top: whether to add function tags to the top node."""
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
				# test for non-empty function tag ("--" is considered empty)
				if (getattr(a, 'source', None)
						and a.source[FUNC] and a.source[FUNC] != '--'):
					func = a.source[FUNC].split("-")[0].upper()
					if action == 'add':
						a.label += "-%s" % func
					elif action == 'replace':
						a.label = func


def handlemorphology(action, lemmaaction, preterminal, source):
	"""Given a preterminal, augment or replace its label with morphological
	information."""
	if not source:
		return
	# escape any parentheses to avoid hassles w/bracket notation of trees
	tag = source[TAG].replace('(', '[').replace(')', ']')
	morph = source[MORPH].replace('(', '[').replace(')', ']').replace(' ', '_')
	lemma = (source[LEMMA].replace('(', '[').replace(')', ']').replace(' ', '_')
			or '--')
	if lemmaaction == 'add':
		raise NotImplementedError
		#sent[preterminal[0]] = '%s|%s' % (word, lemma)
	elif lemmaaction == 'replace':
		raise NotImplementedError
		#sent[preterminal[0]] = lemma
	elif lemmaaction == 'between':
		preterminal[:] = [preterminal.__class__(lemma, preterminal)]
	elif lemmaaction not in (None, 'no'):
		raise ValueError

	if action in (None, 'no'):
		preterminal.label = tag
	elif action == 'add':
		preterminal.label = '%s/%s' % (tag, morph)
	elif action == 'replace':
		preterminal.label = morph
	elif action == 'between':
		preterminal[:] = [preterminal.__class__(morph, [preterminal.pop()])]
		preterminal.label = tag
	else:
		raise ValueError
	return preterminal


def dependencies(root, headrules):
	"""Lin (1995): A Dependency-based Method
	for Evaluating Broad-Coverage Parser."""
	deps = []
	deps.append(('ROOT', makedep(root, deps, headrules), 0))
	return sorted(deps)


def makedep(root, deps, headrules):
	"""Traverse a tree marking heads and extracting dependencies."""
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
	"""Incremental corpus reader.

	Supports brackets, discbrackets, and export format. The format is
	autodetected. Expects an iterator giving one line at a time. Yields tuples
	with a Tree object, a separate lists of terminals, and a string with any
	other data following the tree."""
	treeinput = chain(iter(treeinput), ('(', None, None))  # hack
	line = next(treeinput)
	# try the following readers on each line in this order
	readers = [segmentbrackets('()'), segmentbrackets('[]'),
			segmentexport(morphology, functions)]
	for reader in readers:
		reader.send(None)
	while True:
		# status 0: line not part of tree; status 1: waiting for end of tree.
		res, status = None, 1
		for reader in readers:
			while res is None:
				res, status = reader.send(line)
				if status == 0:
					break  # there was no tree, or a complete tree was read
				line = next(treeinput)
			if res is not None:
				for tree in res:
					yield tree
				break
		if res is None:  # none of the readers accepted this line
			line = next(treeinput)


def segmentbrackets(brackets, strict=False):
	"""Co-routine that accepts one line at a time.

	Yields tuples (result, status) where ...

	- result is None or one or more S-expressions as a list of
		tuples (tree, rest), where rest is the string outside of brackets
		between this S-expression and the next.
	- status is 1 if the line was consumed, else 0."""
	lb, rb = brackets
	# regex to check if the tree contains any terminals (as opposed to indices)
	strtermre = re.compile(' (?:[^ %(br)s]*[^ 0-9%(br)s][^ %(br)s]*)?%(rb)s' %
			dict(br=re.escape(brackets), rb=re.escape(rb)))
	newlb = re.compile(r'(?:.*[\n\r])?\s*')
	parens = 0  # number of open parens
	prev = ''  # incomplete tree currently being read
	result = ''  # string of complete tree
	results = []  # trees found in current line
	line = (yield None, 1)
	while True:
		start = 0  # index where current tree starts
		a, b = line.find(lb, len(prev)), line.find(rb, len(prev))
		# ignore first left bracket when not preceded by whitespace
		if parens == 0 and a > 0 and newlb.match(prev) is None:
			a = -1
		prev = line
		while a != -1 or b != -1:
			if a != -1 and (a < b or b == -1):  # left bracket
				if parens == 0:
					rest, prev = line[start:a], line[a:]
					if result:
						try:
							results.append(brackettree(
									result, rest, brackets, strtermre))
						except:
							print('error in:')
							print(result)
							print(rest)
							raise
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
					if strict:
						raise ValueError('unbalanced parentheses')
					parens = 0
				b = line.find(rb, b + 1)
		status = 1 if results or result or parens else 0
		line = (yield results or None, status)
		if results:
			results = []
		if line is None:
			if result:
				try:
					results.append(brackettree(
							result, rest, brackets, strtermre))
				except:
					print('error in:')
					print(result)
					print(rest)
					raise
			status = 1 if results or result or parens else 0
			yield results or None, status
			line = ''
			if results:
				results = []
		if parens or result:
			line = prev + line
		else:
			prev = ''


def segmentexport(morphology, functions):
	"""Co-routine that accepts one line at a time.
	Yields tuples ``(result, status)`` where ...

	- result is ``None`` or a segment delimited by
		``#BOS`` and ``#EOS`` as a list of lines;
	- status is 1 if the line was consumed, else 0."""
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
			line = (yield ((tree, sent, ''), ), 1)
			inblock = False
			cur = []
		elif line.strip():
			if inblock:
				cur.append(line)
			line = (yield None, (1 if inblock else 0))
		else:
			line = (yield None, 0)


def brackettree(treestr, sent, brackets, strtermre):
	"""Parse a single tree presented in bracket format, whether with indices
	or not; in the latter case ``sent`` is ignored."""
	if strtermre.search(treestr):  # terminals are not all indices
		cnt = count()
		tree = ParentedTree.parse(
				LEAVESRE.sub(lambda _: ' %d)' % next(cnt), treestr),
				parse_leaf=int, brackets=brackets)
		rest = sent.strip()
		sent = [a or None for a in LEAVESRE.findall(treestr)]
	else:  # disc. trees with integer indices as terminals
		tree = ParentedTree.parse(treestr, parse_leaf=int,
			brackets=brackets)
		if sent.strip():
			maxleaf = max(tree.leaves())
			sent = sent.split(' ', maxleaf + 1)
			rest = sent[maxleaf + 1].strip() if len(sent) > maxleaf + 1 else ''
			sent = sent[:maxleaf + 1]
		else:
			sent = map(str, range(max(tree.leaves()) + 1))
			rest = ''
	return tree, sent, rest


def exporttree(data, morphology):
	"""
	Get tree and sentence from tree in export format given as list of lines.
	"""
	data = [exportsplit(x) for x in data]
	tree = exportparse(data, morphology)
	sent = []
	for a in data:
		if EXPORTNONTERMINAL.match(a[WORD]):
			break
		sent.append(a[WORD])
	return tree, sent


def alpinotree(block, morphology=None, lemmas=None, functions=None):
	"""Get tree and sentence from tree in Alpino format given as an etree XML
	object."""
	tree = alpinoparse(block, morphology, lemmas)
	sent = block.find('sentence').text.split(' ')
	handlefunctions(functions, tree)
	return tree, sent


def treebankfanout(trees):
	"""Get maximal fan-out of a list of trees."""
	from discodop.treetransforms import addbitsets, fanout
	# avoid max over empty sequence: 'treebank' may only have unary productions
	try:
		return max((fanout(a), n) for n, tree in enumerate(trees)
				for a in addbitsets(tree).subtrees(lambda x: len(x) > 1))
	except ValueError:
		return 1, 0


def numbase(key):
	"""Turn a file name into a numeric sorting key if possible."""
	path, base = os.path.split(key)
	base = base.split(".", 1)
	try:
		base[0] = int(base[0])
	except ValueError:
		pass
	return [path] + base


READERS = OrderedDict((('export', NegraCorpusReader),
		('bracket', BracketCorpusReader),
		('discbracket', DiscBracketCorpusReader),
		('tiger', TigerXMLCorpusReader),
		('alpino', AlpinoCorpusReader), ('dact', DactCorpusReader)))
WRITERS = ('export', 'bracket', 'discbracket', 'dact',
		'conll', 'mst', 'tokens', 'wordpos')
