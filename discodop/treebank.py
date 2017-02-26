"""Read and write treebanks."""
from __future__ import division, print_function, absolute_import, \
		unicode_literals
import os
import re
import sys
if sys.version_info[0] == 2:
	import xml.etree.cElementTree as ElementTree
else:
	import xml.etree.ElementTree as ElementTree
from glob import glob
from itertools import count, chain, islice
from collections import defaultdict
try:
	from cyordereddict import OrderedDict
except ImportError:
	from collections import OrderedDict
from .tree import Tree, ParentedTree, brackettree, escape, unescape, \
		writebrackettree, writediscbrackettree, SUPERFLUOUSSPACERE
from .treetransforms import removeemptynodes
from .punctuation import applypunct
from .heads import applyheadrules, readheadrules
from .util import openread, ishead

FIELDS = tuple(range(6))
WORD, LEMMA, TAG, MORPH, FUNC, PARENT = FIELDS
EXPORTHEADER = '%% word\tlemma\ttag\tmorph\tedge\tparent\tsecedge\n'
EXPORTNONTERMINAL = re.compile(r'^#([5-9][0-9][0-9])$')
POSRE = re.compile(r'\(([^() ]+)\s+[^ ()]+\s*\)')
TERMINALSRE = re.compile(r' ([^ ()]+)\s*\)')
LEAVESRE = re.compile(r' ([^ ()]*)\s*\)')


class Item(object):
	"""A treebank item."""
	__slots__ = ('tree', 'sent', 'comment', 'block')

	def __init__(self, tree, sent, comment, block):
		self.tree = tree  # A ParentedTree
		self.sent = sent  # list of str
		self.comment = comment  # a string or None
		self.block = block  # a string with tree in original treebank format


class CorpusReader(object):
	"""Abstract corpus reader."""

	def __init__(self, path, encoding='utf8', ensureroot=None, punct=None,
			headrules=None, removeempty=False,
			functions=None, morphology=None, lemmas=None):
		"""
		:param path: filename or pattern of corpus files; e.g., ``wsj*.mrg``.
		:param ensureroot: add root node with given label if necessary.
		:param removeempty: remove empty nodes and any empty ancestors; a
			terminal is empty if it is equal to None, '', or '-NONE-'.
		:param headrules: if given, read rules for assigning heads and apply
			them by ordering constituents according to their heads.
		:param punct: one of ...

			:None: leave punctuation as is [default].
			:'move': move punctuation to appropriate constituents
					using heuristics.
			:'moveall': same as 'move', but moves all preterminals under root,
					instead of only recognized punctuation.
			:'prune': prune away leading & ending quotes & periods, then move.
			:'remove': eliminate punctuation.
			:'removeall': eliminate all preterminals directly under root.
			:'root': attach punctuation directly to root
					(as in original Negra/Tiger treebanks).
		:param functions: one of ...

			:None, 'leave': leave syntactic labels as is [default].
			:'add': concatenate grammatical function to syntactic label,
				separated by a hypen: e.g., ``NP => NP-SBJ``.
			:'remove': strip away hyphen-separated grammatical function,
				e.g., ``NP-SBJ => NP``.
			:'replace': replace syntactic label with grammatical function,
				e.g., ``NP => SBJ``.
		:param morphology: one of ...

			:None, 'no': use POS tags as preterminals [default].
			:'add': concatenate morphological information to POS tags,
				e.g., ``DET/sg.def``.
			:'replace': use morphological information as preterminal label
			:'between': add node with morphological information between
				POS tag and word, e.g., ``(DET (sg.def the))``.
		:param lemmas: one of ...

			:None: ignore lemmas [default].
			:'add': concatenate lemma to terminals, e.g., men/man.
			:'replace': use lemmas as terminals.
			:'between': insert lemma as node between POS tag and word."""
		self.removeempty = removeempty
		self.ensureroot = ensureroot
		self.functions = functions
		self.punct = punct
		self.morphology = morphology
		self.lemmas = lemmas
		self.headrules = readheadrules(headrules) if headrules else {}
		self._encoding = encoding
		try:
			self._filenames = (sorted(glob(path), key=numbase)
					if path != '-' else ['-'])
		except TypeError:
			print('all sentence IDs must have the same type signature '
					'(number, string)')
			raise
		for opts, opt in (
				((None, 'leave', 'add', 'replace', 'remove', 'between'),
					functions),
				((None, 'no', 'add', 'replace', 'between'), morphology),
				((None, 'no', 'move', 'moveall', 'remove', 'removeall',
					'prune', 'root'), punct),
				((None, 'no', 'add', 'replace', 'between'), lemmas),
				):
			if opt not in opts:
				raise ValueError('Expected one of %r. Got: %r' % (opts, opt))
		if not self._filenames:
			raise ValueError("no files matched pattern '%s' in %s" % (
					path, os.getcwd()))
		self._block_cache = None
		self._trees_cache = None

	def itertrees(self, start=None, end=None):
		"""
		:returns: an iterator returning tuples ``(key, item)``
			of sentences in corpus, where ``item`` is an :py:class:Item
			instance with ``tree``, ``sent``, and ``comment`` attributes.
			Useful when the dictionary of all trees in corpus would not fit in
			memory."""
		for n, a in islice(self._read_blocks(), start, end):
			yield n, self._parsetree(a)

	def trees(self):
		"""
		:returns: an ordered dictionary of parse trees
			(``Tree`` objects with integer indices as leaves)."""
		if not self._trees_cache:
			self._trees_cache = OrderedDict((n, self._parsetree(a))
					for n, a in self._read_blocks())
		return OrderedDict((n, a.tree) for n, a in self._trees_cache.items())

	def sents(self):
		"""
		:returns: an ordered dictionary of sentences,
			each sentence being a list of words."""
		if not self._trees_cache:
			self._trees_cache = OrderedDict((n, self._parsetree(a))
					for n, a in self._read_blocks())
		return OrderedDict((n, a.sent) for n, a in self._trees_cache.items())

	def tagged_sents(self):
		"""
		:returns: an ordered dictionary of tagged sentences,
			each tagged sentence being a list of (word, tag) pairs."""
		if not self._trees_cache:
			self._trees_cache = OrderedDict((n, self._parsetree(a))
					for n, a in self._read_blocks())
		return OrderedDict(
				(n, [(w, t) for w, (_, t) in zip(a.sent, sorted(a.tree.pos()))])
				for n, a in self._trees_cache.items())

	def blocks(self):
		"""
		:returns: a list of strings containing the raw representation of
			trees in the original treebank."""

	def _read_blocks(self):
		"""Iterate over blocks in corpus file corresponding to parse trees."""

	def _parse(self, block):
		""":returns: a parse tree given a string from the treebank file."""

	def _parsetree(self, block):
		""":returns: a transformed parse tree and sentence."""
		item = self._parse(block)
		if not item.sent:  # ??3
			return item
		if self.removeempty:
			removeemptynodes(item.tree, item.sent)
		if self.ensureroot and item.tree.label != self.ensureroot:
			item.tree = ParentedTree(self.ensureroot, [item.tree])
		if not isinstance(self, BracketCorpusReader):
			# roughly order constituents by order in sentence
			for a in reversed(list(item.tree.subtrees(lambda x: len(x) > 1))):
				a.children.sort(key=Tree.leaves)
		if self.punct:
			applypunct(self.punct, item.tree, item.sent)
		if self.headrules:
			applyheadrules(item.tree, self.headrules)
		return item

	def _word(self, block):
		""":returns: a list of words given a string."""
		if self.punct in {'remove', 'prune'}:
			return self._parsetree(block).sent
		return self._parse(block).sent


class BracketCorpusReader(CorpusReader):
	"""Corpus reader for phrase-structures in bracket notation.

	For example::

		(S (NP John) (VP (VB is) (JJ rich)) (. .))"""

	def blocks(self):
		return OrderedDict(self._read_blocks())

	def _read_blocks(self):
		for n, block in enumerate((line for filename in self._filenames
				for line in openread(filename, encoding=self._encoding)
				if line), 1):
			yield n, block

	def _parse(self, block):
		c = count()
		block = SUPERFLUOUSSPACERE.sub(')', block)
		tree = ParentedTree(LEAVESRE.sub(lambda _: ' %d)' % next(c), block))
		for node in tree.subtrees():
			for char in '-=':  # map NP-SUBJ and NP=2 to NP; don't touch -NONE-
				x = node.label.find(char)
				if x > 0:
					if char == '-' and not node.label[x + 1:].isdigit():
						if node.source is None:
							node.source = [None] * len(FIELDS)
						node.source[FUNC] = node.label[x + 1:].rstrip(
								'=0123456789')
					if self.functions == 'remove':
						node.label = node.label[:x]
		sent = [escape(token) for token in LEAVESRE.findall(block)]
		return Item(tree, sent, None, block)


class DiscBracketCorpusReader(BracketCorpusReader):
	"""A corpus reader for discontinuous trees in bracket notation.

	Leaves are consist of an index and a word, with the indices indicating
	the word order of the sentence. For example::

		(S (NP 1=John) (VP (VB 0=is) (JJ 2=rich)) (? 3=?))

	There is one tree per line. Optionally, the tree may be followed by a
	comment, separated by a TAB. Compared to Negra's export format, this format
	lacks morphology, lemmas and functional edges. On the other hand, it is
	close to the internal representation employed here, so it can be read
	efficiently."""

	def _parse(self, block):
		treestr, comment = block, None
		if '\t' in block:
			treestr, comment = block.rstrip('\n\r').split('\t', 1)

		sent = {}

		def substleaf(x):
			"""Collect token and return index."""
			idx, token = x.split('=', 1)
			idx = int(idx)
			sent[idx] = unescape(token)
			return int(idx)

		tree = ParentedTree.parse(treestr, parse_leaf=substleaf)
		sent = [sent.get(n, None) for n in range(max(sent) + 1)]

		if not all(0 <= n < len(sent) for n in tree.leaves()):
			raise ValueError('All leaves must be in the interval 0..n with '
					'n=len(sent)\ntokens: %d indices: %r\nsent: %s' % (
					len(sent), tree.leaves(), sent))
		return Item(tree, sent, comment, block)


class NegraCorpusReader(CorpusReader):
	"""Read a corpus in the Negra export format."""

	def blocks(self):
		if self._block_cache is None:
			self._block_cache = OrderedDict(self._read_blocks())
		return OrderedDict((a, '\n'.join(b) + '\n')
				for a, b in self._block_cache.items())

	def _read_blocks(self):
		"""Read corpus and yield blocks corresponding to each sentence."""
		results = set()
		started = False
		for filename in self._filenames:
			for line in openread(filename, encoding=self._encoding):
				if line.startswith('#BOS '):
					if started:
						raise ValueError('beginning of sentence marker while '
								'previous one still open: %s' % line)
					started = True
					line = line.strip()
					sentid = line.split()[1]
					lines = [line]
				elif line.startswith('#EOS '):
					if not started:
						raise ValueError('end of sentence marker while '
								'none started')
					thissentid = line.strip().split()[1]
					if sentid != thissentid:
						raise ValueError('unexpected sentence id: '
							'start=%s, end=%s' % (sentid, thissentid))
					started = False
					if sentid in results:
						raise ValueError('duplicate sentence ID: %s' % sentid)
					results.add(sentid)
					lines.append(line.strip())
					yield sentid, lines
				elif started:
					lines.append(line.strip())
				# other lines are ignored, such as #FORMAT x, %% comments, ...

	def _parse(self, block):
		return exporttree(block, self.functions, self.morphology, self.lemmas)


class TigerXMLCorpusReader(CorpusReader):
	"""Corpus reader for the Tiger XML format."""

	def blocks(self):
		"""
		:returns: a list of strings containing the raw representation of
			trees in the treebank."""
		if self._block_cache is None:
			self._block_cache = OrderedDict(self._read_blocks())
		return OrderedDict((n, ElementTree.tostring(a))
				for n, a in self._block_cache.items())

	def _read_blocks(self):
		for filename in self._filenames:
			# iterator over elements in XML  file
			context = ElementTree.iterparse(filename,
					events=('start', 'end'))
			_, root = next(context)  # event == 'start' of root element
			for event, elem in context:
				if event == 'end' and elem.tag == 's':
					yield elem.get('id'), elem
				root.clear()

	def _parse(self, block):
		"""Translate Tiger XML structure to the fields of export format."""
		nodes = OrderedDict()
		root = block.find('graph').get('root')
		for term in block.find('graph').find('terminals'):
			fields = nodes.setdefault(term.get('id'), 6 * [None])
			fields[WORD] = term.get('word')
			fields[LEMMA] = term.get('lemma')
			fields[TAG] = term.get('pos')
			fields[MORPH] = term.get('morph')
			fields[PARENT] = '0' if term.get('id') == root else None
			fields[FUNC] = '--'
			nodes[term.get('id')] = fields
		for nt in block.find('graph').find('nonterminals'):
			if nt.get('id') == root:
				ntid = '0'
			else:
				fields = nodes.setdefault(nt.get('id'), 6 * [None])
				ntid = nt.get('id').split('_')[-1]
				fields[WORD] = '#' + ntid
				fields[TAG] = nt.get('cat')
				fields[LEMMA] = fields[MORPH] = fields[FUNC] = '--'
			for edge in nt:
				idref = edge.get('idref')
				nodes.setdefault(idref, 6 * [None])
				if edge.tag == 'edge':
					if nodes[idref][FUNC] not in (None, '--'):
						raise ValueError('%s already has a parent: %r'
								% (idref, nodes[idref]))
					nodes[idref][FUNC] = edge.get('label')
					nodes[idref][PARENT] = ntid
				elif edge.tag == 'secedge':
					nodes[idref].extend((edge.get('label'), ntid))
				else:
					raise ValueError("expected 'edge' or 'secedge' tag.")
		for idref in nodes:
			if nodes[idref][PARENT] is None:
				raise ValueError('%s does not have a parent: %r' % (
						idref, nodes[idref]))
		item = exporttree(
				['#BOS ' + block.get('id')]
				+ ['\t'.join(a) for a in nodes.values()]
				+ ['#EOS ' + block.get('id')],
				self.functions, self.morphology, self.lemmas)
		item.tree.label = root.split('_', 1)[1]
		item.block = ElementTree.tostring(block)
		return item


class AlpinoCorpusReader(CorpusReader):
	"""Corpus reader for the Dutch Alpino treebank in XML format.

	Expects a corpus in directory format, where every sentence is in a single
	``.xml`` file."""

	def blocks(self):
		"""
		:returns: a list of strings containing the raw representation of
			trees in the treebank."""
		if self._block_cache is None:
			self._block_cache = OrderedDict(self._read_blocks())
		return self._block_cache

	def _read_blocks(self):
		"""Read corpus and yield blocks corresponding to each sentence."""
		if self._encoding not in (None, 'utf8', 'utf-8'):
			raise ValueError('Encoding specified in XML files, '
					'cannot be overriden.')
		for filename in self._filenames:
			block = open(filename, 'rb').read()  # NB: store XML data as bytes
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
			try:
				xmlblock = ElementTree.fromstring(block)
			except ElementTree.ParseError:
				print('Problem with:\n%s' %
						block.decode('utf8', errors='replace'),
						file=sys.stderr)
				raise
		return alpinotree(
				xmlblock, self.functions, self.morphology, self.lemmas)


class DactCorpusReader(AlpinoCorpusReader):
	"""Corpus reader for Alpino trees in Dact format (DB XML)."""

	def _read_blocks(self):
		import alpinocorpus
		if self._encoding not in (None, 'utf8', 'utf-8'):
			raise ValueError('Encoding specified in XML files, '
					'cannot be overriden.')
		for filename in self._filenames:
			corpus = alpinocorpus.CorpusReader(filename)
			for entry in corpus.entries():
				yield entry.name(), entry.contents()


def exporttree(block, functions=None, morphology=None, lemmas=None):
	"""Get tree, sentence from tree in export format given as list of lines."""
	def getchildren(parent):
		"""Traverse tree in export format and create Tree object."""
		results = []
		for n, source in children.get(parent, []):
			# n is the index in the block to record word indices
			m = EXPORTNONTERMINAL.match(source[WORD])
			if m:
				child = ParentedTree(source[TAG], getchildren(m.group(1)))
			else:  # POS + terminal
				child = ParentedTree(source[TAG], [n])
				handlemorphology(morphology, lemmas, child, source, sent)
			child.source = tuple(source)
			results.append(child)
		return results

	comment = block[0].split('%%')[1].strip() if '%%' in block[0] else None
	table = [exportsplit(x) for x in block[1:-1]]
	sent = []
	children = {'0': []}
	for source in table:
		m = EXPORTNONTERMINAL.match(source[WORD])
		if m:
			children[m.group(1)] = []
		else:
			sent.append(source[WORD])
	for n, source in enumerate(table):
		children[source[PARENT]].append((n, source))
	tree = ParentedTree('ROOT', getchildren('0'))
	handlefunctions(functions, tree, morphology=morphology)
	return Item(tree, sent, comment, '\n'.join(block) + '\n')


def exportsplit(line):
	"""Take a line in export format and split into fields.

	Add dummy fields lemma, sec. edge if those fields are absent."""
	if "%%" in line:  # we don't want comments.
		line = line[:line.index("%%")]
	fields = line.split()
	fieldlen = len(fields)
	if fieldlen < 5:
		raise ValueError('expected at least 5 columns: %r' % fields)
	elif fieldlen & 1:  # odd number of fields?
		fields[1:1] = ['--']  # add empty lemma field
	return fields


def alpinotree(block, functions=None, morphology=None, lemmas=None):
	"""Get tree, sent from tree in Alpino format given as etree XML object."""
	def getsubtree(node, parentid, morphology, lemmas):
		"""Parse a subtree of an Alpino tree."""
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
				subtree = getsubtree(child, node.get('id'), morphology, lemmas)
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
			handlemorphology(morphology, lemmas, result, source, sent)
		elif 'index' in node.keys():
			coindexation[node.get('index')].extend([node.get('rel'), parentid])
			return None
		source[:] = [a.replace(' ', '_') if a else a for a in source]
		result.source = source
		return result

	coindexed = {}
	coindexation = defaultdict(list)
	sent = block.find('sentence').text.split(' ')
	tree = getsubtree(block.find('node'), 0, morphology, lemmas)
	for i in coindexation:
		coindexed[i].extend(coindexation[i])
	comment = block.find('comments/comment')  # NB: only use first comment
	if comment is not None:
		comment = comment.text
	handlefunctions(functions, tree, morphology=morphology)
	return Item(tree, sent, comment, ElementTree.tostring(block))


def writetree(tree, sent, key, fmt, comment=None, morphology=None,
		sentid=False):
	"""Convert a tree to a string representation in the given treebank format.

	:param tree: should have indices as terminals
	:param sent: contains the words corresponding to the indices in ``tree``
	:param key: an identifier for this tree; part of the output with some
		formats or when ``sentid`` is True.
	:param fmt: Formats are ``bracket``, ``discbracket``, Negra's ``export``
		format, and ``alpino`` XML format, as well unlabeled dependency
		conversion into ``mst`` or ``conll`` format (requires head rules).
		The formats ``tokens`` and ``wordpos`` are to strip away tree structure
		and leave only lines with space-separated tokens or ``token/POS``.
		When using ``bracket``, make sure tree is canonicalized.
	:param comment: optionally, a string that will go in the format's comment
		field (supported by ``export`` and ``alpino``), or at the end of the
		line preceded by a tab (``discbracket``); ignored by other formats.
		Should be a single line.
	:param sentid: for line-based formats, prefix output by ``key|``.

	Lemmas, functions, and morphology information will be empty unless nodes
	contain a 'source' attribute with such information."""
	if fmt == 'bracket':
		result = writebrackettree(tree, sent)
		# if comment:
		# 	result = '# %s\n%s\n' % (comment, result.rstrip('\n'))
	elif fmt == 'discbracket':
		result = writediscbrackettree(tree, sent)
		if comment:
			result = '%s\t%s\n' % (result.rstrip('\n'), comment)
	elif fmt == 'tokens':
		result = '%s\n' % ' '.join(sent)
	elif fmt == 'wordpos':
		result = '%s\n' % ' '.join('%s/%s' % (word, pos) for word, (_, pos)
				in zip(sent, sorted(tree.pos())))
	elif fmt == 'export':
		result = writeexporttree(tree, sent, key, comment, morphology)
	elif fmt == 'alpino':
		result = writealpinotree(tree, sent, key, comment)
	elif fmt in ('conll', 'mst'):
		result = writedependencies(tree, sent, fmt)
	else:
		raise ValueError('unrecognized format: %r' % fmt)
	if sentid and fmt in ('tokens', 'wordpos', 'bracket', 'discbracket'):
		return '%s|%s' % (key, result)
	return result


def writeexporttree(tree, sent, key, comment, morphology):
	"""Return string with given tree in Negra's export format."""
	result = []
	if key is not None:
		cmt = (' %% ' + comment) if comment else ''
		result.append('#BOS %s%s' % (key, cmt))
	# visit nodes in post-order traversal
	preterms, phrasalnodes = {}, []
	agenda = list(tree)
	while agenda:
		node = agenda.pop()
		if not node or isinstance(node[0], Tree):
			# NB: to get a proper post-order traversal, children need to be
			# reversed, but for the assignment of IDs this does not matter.
			agenda.extend(node)
			phrasalnodes.append(node)
		else:
			preterms[node[0]] = node
	phrasalnodes.reverse()
	if len(sent) != len(preterms):
		raise ValueError('sentence and terminals length mismatch:  '
				'sentno: %s\ntree: %s\nsent (len=%d): %r\nleaves (len=%d): %r'
				% (key, tree, len(sent), sent, len(preterms), preterms))
	idindex = [id(node) for node in phrasalnodes]
	nodeidindex = [int(node.source[WORD].lstrip('#')) if node.source else 0
			for node in phrasalnodes]
	for n, word in enumerate(sent):
		if not word:
			# raise ValueError('empty word in sentence: %r' % sent)
			word = '...'
		node = preterms[n]
		lemma = '--'
		postag = node.label.replace('$[', '$(') or '--'
		func = morphtag = '--'
		secedges = []
		if getattr(node, 'source', None):
			lemma = node.source[LEMMA] or '--'
			morphtag = node.source[MORPH] or '--'
			func = node.source[FUNC] or '--'
			for rel, pid in zip(node.source[6::2], node.source[7::2]):
				secedges.append(rel)
				secedges.append('%d' % (500 + nodeidindex.index(int(pid))))
		if morphtag == '--' and morphology == 'replace':
			morphtag = postag
		elif morphtag == '--' and morphology == 'add' and '/' in postag:
			postag, morphtag = postag.split('/', 1)
		parentid = '%d' % (0 if node.parent is tree
				else 500 + idindex.index(id(node.parent)))
		result.append("\t".join((word, lemma, postag, morphtag, func,
				parentid) + tuple(secedges)))
	for n, node in enumerate(phrasalnodes):
		nodeid = '#%d' % (500 + n)
		lemma = '--'
		label = node.label or '--'
		func = morphtag = '--'
		secedges = []
		if getattr(node, 'source', None):
			morphtag = node.source[MORPH] or '--'
			func = node.source[FUNC] or '--'
			for rel, pid in zip(node.source[6::2], node.source[7::2]):
				secedges.append(rel)
				secedges.append('%d' % (500 + nodeidindex.index(int(pid))))
		parentid = '%d' % (0 if node.parent is tree
				else 500 + idindex.index(id(node.parent)))
		result.append('\t'.join((nodeid, lemma, label, morphtag, func,
				parentid) + tuple(secedges)))
	if key is not None:
		result.append("#EOS %s" % key)
	return "%s\n" % "\n".join(result)


def writealpinotree(tree, sent, key, commentstr):
	"""Return XML string with tree in AlpinoXML format."""
	def addchildren(tree, sent, parent, cnt, depth=1, last=False):
		"""Recursively add children of ``tree`` to XML object ``node``"""
		node = ElementTree.SubElement(parent, 'node')
		node.set('id', str(next(cnt)))
		node.set('begin', str(min(tree.leaves())))
		node.set('end', str(max(tree.leaves()) + 1))
		if getattr(tree, 'source', None):
			node.set('rel', tree.source[FUNC] or '--')
		if isinstance(tree[0], Tree):
			node.set('cat', tree.label.lower())
			node.text = '\n  ' + '  ' * depth
		else:
			assert isinstance(tree[0], int)
			node.set('pos', tree.label.lower())
			node.set('word', sent[tree[0]])
			if getattr(tree, 'source', None):
				node.set('lemma', tree.source[LEMMA] or '--')
				node.set('postag', tree.source[MORPH] or '--')
				# FIXME: split features in multiple attributes
			else:
				node.set('lemma', '--')
				node.set('postag', '--')
		node.tail = '\n' + '  ' * (depth - last)
		for x, child in enumerate(tree, 1):
			if isinstance(child, Tree):
				addchildren(child, sent, node, cnt, depth + 1, x == len(tree))

	result = ElementTree.Element('alpino_ds')
	result.set('version', '1.3')
	# FIXME: add coindexed nodes
	addchildren(tree, sent, result, count())
	sentence = ElementTree.SubElement(result, 'sentence')
	sentence.text = ' '.join(sent)
	comment = ElementTree.SubElement(result, 'comment')
	comment.text = ('%s|%s' % (key, commentstr)) if commentstr else str(key)
	result.text = sentence.tail = '\n  '
	result.tail = comment.tail = '\n'
	return ElementTree.tostring(result).decode('utf8')  # hack


def writedependencies(tree, sent, fmt):
	"""Convert tree to unlabeled dependencies in `mst` or `conll` format."""
	deps = dependencies(tree)
	if fmt == 'mst':  # MST parser can read this format
		# fourth line with function tags is left empty.
		return "\n".join((
			'\t'.join(word for word in sent),
			'\t'.join(tag for _, tag in sorted(tree.pos())),
			'\t'.join(str(head) for _, _, head in deps))) + '\n\n'
	elif fmt == 'conll':
		return '\n'.join('%d\t%s\t_\t%s\t%s\t_\t%d\t%s\t_\t_' % (
			n, word, tag, tag, head, rel)
			for word, (_, tag), (n, rel, head)
			in zip(sent, sorted(tree.pos()), deps)) + '\n\n'


def dependencies(root):
	"""Lin (1995): A Dependency-based Method for Evaluating [...] Parsers.

	http://ijcai.org/Proceedings/95-2/Papers/052.pdf

	:returns: list of tuples of the form ``(headidx, label, depidx)``."""
	deps = []
	if root:
		deps.append((_makedep(root, deps), 'ROOT', 0))
	return sorted(deps)


def _makedep(node, deps):
	"""Traverse a head-marked tree and extract dependencies."""
	if isinstance(node[0], int):
		return node[0] + 1
	headchild = next(iter(a for a in node if ishead(a)))
	lexhead = _makedep(headchild, deps)
	for child in node:
		if child is headchild:
			continue
		lexheadofchild = _makedep(child, deps)
		func = '-'
		if (getattr(child, 'source', None)
				and child.source[FUNC] and child.source[FUNC] != '--'):
			func = child.source[FUNC]
		deps.append((lexheadofchild, func, lexhead))
	return lexhead


def deplen(deps):
	"""Compute dependency length from result of ``dependencies()``.

	:returns: tuple ``(totaldeplen, numdeps)``."""
	total = sum(abs(a - b) for a, label, b in deps
			if label != 'ROOT')
	return (total, float(len(deps) - 1))  # discount ROOT


def handlefunctions(action, tree, pos=True, top=False, morphology=None):
	"""Add function tags to phrasal labels e.g., 'VP' => 'VP-HD'.

	:param action: one of {None, 'add', 'replace', 'remove'}
	:param pos: whether to add function tags to POS tags.
	:param top: whether to add function tags to the top node.
	:param morphology: if morphology='between', skip those nodes."""
	if action in (None, 'leave'):
		return
	for a in tree.subtrees():
		if action == 'remove':
			for char in '-=':  # map NP-SUBJ and NP=2 to NP; don't touch -NONE-
				x = a.label.find(char)
				if x > 0:
					a.label = a.label[:x]
		elif morphology == 'between' and not isinstance(a[0], Tree):
			continue
		elif (not top or action == 'between') and a is tree:  # skip TOP label
			continue
		elif pos or isinstance(a[0], Tree):
			# test for non-empty function tag ('--' is considered empty)
			func = None
			if (getattr(a, 'source', None)
					and a.source[FUNC] and a.source[FUNC] != '--'):
				func = a.source[FUNC]
			if func and action == 'add':
				a.label += '-%s' % func
			elif action == 'replace':
				a.label = func or '--'
			elif action == 'between':
				parent, idx = a.parent, a.parent_index
				newnode = ParentedTree('-' + (func or '--'), [parent.pop(idx)])
				parent.insert(idx, newnode)


def handlemorphology(action, lemmaaction, preterminal, source, sent=None):
	"""Augment/replace preterminal label with morphological information."""
	if not source:
		return
	# escape any parentheses to avoid hassles w/bracket notation of trees
	tag = source[TAG].replace('(', '[').replace(')', ']')
	morph = source[MORPH].replace('(', '[').replace(')', ']').replace(' ', '_')
	lemma = (source[LEMMA].replace('(', '[').replace(')', ']').replace(
			' ', '_') or '--')
	if lemmaaction == 'add':
		if sent is None:
			raise ValueError('adding lemmas requires passing sent argument.')
		sent[preterminal[0]] += '/' + lemma
	elif lemmaaction == 'replace':
		if sent is None:
			raise ValueError('adding lemmas requires passing sent argument.')
		sent[preterminal[0]] = lemma
	elif lemmaaction == 'between':
		preterminal[:] = [preterminal.__class__(lemma, preterminal)]
	elif lemmaaction not in (None, 'no'):
		raise ValueError('unrecognized action: %r' % lemmaaction)

	if action in (None, 'no'):
		preterminal.label = tag
	elif action == 'add':
		preterminal.label = '%s/%s' % (tag, morph)
	elif action == 'replace':
		preterminal.label = morph
	elif action == 'between':
		preterminal[:] = [preterminal.__class__(morph, [preterminal.pop()])]
		preterminal.label = tag
	elif action not in (None, 'no'):
		raise ValueError('unrecognized action: %r' % action)
	return preterminal

CONSUMED = True
NEWLB = re.compile(r'(?:.*[\n\r])?\s*')


def incrementaltreereader(treeinput, morphology=None, functions=None,
		strict=False, robust=True, othertext=False):
	"""Incremental corpus reader.

	Supports brackets, discbrackets, export and alpino-xml format.
	The format is autodetected.

	:param treeinput: an iterator giving one line at a time.
	:param strict: if True, raise ValueError on malformed data.
	:param robust: if True, only return trees with more than 2 brackets;
		e.g., (DT the) is not recognized as a tree.
	:param othertext: if True, yield non-tree data as ``(None, None, line)``.
		By default, text in lines without trees is ignored.
	:yields: tuples ``(tree, sent, comment)`` with a Tree object, a separate
		lists of terminals, and a string with any other data following the
		tree."""
	treeinput = chain(iter(treeinput), ('(', None, None))  # hack
	line = next(treeinput)
	# try the following readers on each line in this order
	readers = [segmentexport(morphology, functions, strict),
			segmentalpino(morphology, functions),
			segmentbrackets(strict, robust)]
	for reader in readers:
		reader.send(None)
	while True:
		# status 0: line not consumed, not part of tree;
		# status 1: line consumed, waiting for end of tree.
		res, status = None, CONSUMED
		for reader in readers:
			while res is None:
				res, status = reader.send(line)
				if status != CONSUMED:
					break  # there was no tree, or a complete tree was read
				line = next(treeinput)
			if res is not None:
				for tree, sent, rest in res:
					x = -1 if rest is None else rest.find('\n')
					if othertext and x != -1:
						yield tree, sent, rest[:x]
						yield None, None, rest[x:]
					else:
						yield tree, sent, rest
				break
		if res is None:  # none of the readers accepted this line
			if othertext:
				yield None, None, line.rstrip()
			line = next(treeinput)


def segmentbrackets(strict=False, robust=True):
	"""Co-routine that accepts one line at a time.

	Yields tuples ``(result, status)`` where ...

	- result is None or one or more S-expressions as a list of
		tuples (tree, sent, rest), where rest is the string outside of brackets
		between this S-expression and the next.
	- status is 1 if the line was consumed, else 0.

	:param strict: if True, raise ValueError for improperly nested brackets.
	:param robust: if True, only return trees with at least 2 brackets;
		e.g., (DT the) is not recognized as a tree.
	"""
	def tryparse(result, rest):
		"""Add a tree to the results list."""
		try:
			tree, sent = brackettree(result)
		except Exception as err:
			raise ValueError('%r\nwhile parsing:\n%r' % (
					err, dict(result=result, rest=rest, parens=parens,
						depth=depth, prev=prev)))
		else:
			results.append((tree, sent, rest.rstrip()))

	lb, rb = '()'
	parens = 0  # number of open parens
	depth = 0  # max. number of open parens
	prev = ''  # incomplete tree currently being read
	result = ''  # string of complete tree
	results = []  # trees found in current line
	rest = ''  # any non-tree data after a tree
	line = (yield None, CONSUMED)
	while True:
		start = 0  # index where current tree starts
		a, b = line.find(lb, len(prev)), line.find(rb, len(prev))
		# ignore first left bracket when not preceded by whitespace
		if parens == 0 and a > 0 and NEWLB.match(prev) is None:
			a = -1
		prev = line
		while a != -1 or b != -1:
			if a != -1 and (a < b or b == -1):  # left bracket
				# look ahead to see whether this will be a tree with depth > 1
				if parens == 0 and (b == -1 or
							(not robust or 0 <= line.find(lb, a + 1) < b)):
					rest, prev = line[start:a], line[a:]
					if result:
						tryparse(result, rest)
						result, start = '', a
				parens += 1
				depth = max(depth, parens)
				a = line.find(lb, a + 1)
			elif b != -1 and (b < a or a == -1):  # right bracket
				parens -= 1
				if parens == 0 and (not robust or depth > 1):
					result, prev = line[start:b + 1], line[b + 1:]
					start = b + 1
					depth = 0
				elif parens < 0:
					if strict:
						raise ValueError('unbalanced parentheses')
					parens = 0
				b = line.find(rb, b + 1)
		status = CONSUMED if results or result or parens else not CONSUMED
		line = (yield results or None, status)
		if results:
			results = []
		if line is None:
			if result:
				tryparse(result, rest)
			status = CONSUMED if results or result or parens else not CONSUMED
			yield results or None, status
			line = ''
			if results:
				results = []
		if parens or result:
			line = prev + line
		else:
			prev = ''


def segmentalpino(morphology, functions):
	"""Co-routine that accepts one line at a time.
	Yields tuples ``(result, status)`` where ...

	- result is ``None`` or a segment delimited by
		``<alpino_ds>`` and ``</alpino_ds>`` as a list of lines;
	- status is 1 if the line was consumed, else 0."""
	cur = []
	inblock = 0
	line = (yield None, CONSUMED)
	while line is not None:
		if line.startswith('<alpino_ds'):
			cur = ['<?xml version="1.0" encoding="UTF-8"?>', line]
			inblock = 1
			line = (yield None, CONSUMED)
		elif line.startswith('</alpino_ds>'):
			cur.append(line)
			block = ElementTree.fromstring('\n'.join(cur).encode('utf8'))
			item = alpinotree(block, functions, morphology)
			line = (yield ((item.tree, item.sent, item.comment), ), CONSUMED)
			inblock = 0
			cur = []
		elif line.strip():
			if inblock == 1:
				cur.append(line)
			line = line.lstrip()
			line = (yield None, (CONSUMED if inblock
						or line.startswith('<?xml')
					else not CONSUMED))
		else:
			line = (yield None, not CONSUMED)


def segmentexport(morphology, functions, strict=False):
	"""Co-routine that accepts one line at a time.
	Yields tuples ``(result, status)`` where ...

	- result is ``None`` or a segment delimited by
		``#BOS`` and ``#EOS`` as a list of lines;
	- status is 1 if the line was consumed, else 0."""
	cur = []
	inblock = 0
	line = (yield None, CONSUMED)
	while line is not None:
		if line.startswith('#BOS ') or line.startswith('#BOT '):
			if strict and inblock != 0:
				raise ValueError('nested #BOS or #BOT')
			cur = [line]
			inblock = 1 if line.startswith('#BOS ') else 2
			line = (yield None, CONSUMED)
		elif line.startswith('#EOS ') or line.startswith('#EOT '):
			if strict and inblock == 0:
				raise ValueError('#EOS or #EOT without start tag')
			cur.append(line)
			item = exporttree(cur, functions, morphology)
			line = (yield ((item.tree, item.sent, item.comment), ), CONSUMED)
			inblock = 0
			cur = []
		elif line.strip():
			if inblock == 1:
				cur.append(line)
			line = line.lstrip()
			line = (yield None, (CONSUMED if inblock
					or line.startswith('%%')
					or line.startswith('#FORMAT ')
				else not CONSUMED))
		else:
			line = (yield None, not CONSUMED)


def numbase(key):
	"""Split file name in numeric and string components to use as sort key."""
	path, base = os.path.split(key)
	components = re.split(r'[-.,_ ]', os.path.splitext(base)[0])
	components = [int(a) if re.match(r'[0-9]+$', a) else a for a in components]
	return [path] + components


READERS = OrderedDict((('export', NegraCorpusReader),
		('bracket', BracketCorpusReader),
		('discbracket', DiscBracketCorpusReader),
		('tiger', TigerXMLCorpusReader),
		('alpino', AlpinoCorpusReader), ('dact', DactCorpusReader)))
WRITERS = ('export', 'bracket', 'discbracket', 'dact',
		'conll', 'mst', 'tokens', 'wordpos')

__all__ = ['Item', 'CorpusReader', 'BracketCorpusReader',
		'DiscBracketCorpusReader', 'NegraCorpusReader', 'AlpinoCorpusReader',
		'TigerXMLCorpusReader', 'DactCorpusReader', 'exporttree',
		'exportsplit', 'alpinotree', 'writetree', 'writeexporttree',
		'writealpinotree', 'writedependencies', 'dependencies', 'deplen',
		'handlefunctions', 'handlemorphology', 'incrementaltreereader',
		'segmentbrackets', 'segmentexport', 'segmentalpino', 'numbase']
