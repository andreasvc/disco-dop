"""Objects for searching through collections of trees."""

# Possible improvements:
# - cache raw results from _query() before conversion?
# - return/cache trees as strings?

import io
import os
import re
import concurrent.futures
import subprocess
from collections import Counter, OrderedDict
from itertools import islice, count, takewhile
try:
	from multiprocessing import cpu_count  # pylint: disable=E0611
except ImportError:
	from os import cpu_count  # pylint: disable=E0611
from lru import LRU
try:
	import alpinocorpus
	import xml.etree.cElementTree as ElementTree
	ALPINOCORPUSLIB = True
except ImportError:
	ALPINOCORPUSLIB = False
from discodop import treebank
from discodop.tree import Tree
from discodop.parser import workerfunc, which

CACHESIZE = 500
GETLEAVES = re.compile(r' ([^ ()]+)(?=[ )])')
ALPINOLEAVES = re.compile('<sentence>(.*)</sentence>')
MORPH_TAGS = re.compile(
		r'([_/*A-Z0-9]+)(?:\[[^ ]*\][0-9]?)?((?:-[_A-Z0-9]+)?(?:\*[0-9]+)? )')
FUNC_TAGS = re.compile(r'-[_A-Z0-9]+')

# abbreviations for Alpino POS tags
ABBRPOS = {
	'PUNCT': 'PUNCT',
	'COMPLEMENTIZER': 'COMP',
	'PROPER_NAME': 'NAME',
	'PREPOSITION': 'PREP',
	'PRONOUN': 'PRON',
	'DETERMINER': 'DET',
	'ADJECTIVE': 'ADJ',
	'ADVERB': 'ADV',
	'HET_NOUN': 'HET',
	'NUMBER': 'NUM',
	'PARTICLE': 'PRT',
	'ARTICLE': 'ART',
	'NOUN': 'NN',
	'VERB': 'VB'}


class CorpusSearcher(object):
	"""Abstract base class to wrap a set of corpus files that can be queried."""
	def __init__(self, files, macros=None, numthreads=None):
		"""
		:param files: a sequence of filenames of corpora
		:param macros: a filename with macros that can be used in queries.
		:param numthreads: the number of concurrent threads to use;
			None to disable threading."""
		self.files = OrderedDict.fromkeys(files)
		self.macros = macros
		self.numthreads = numthreads
		self.cache = LRU(CACHESIZE)
		self.pool = concurrent.futures.ThreadPoolExecutor(
				numthreads or cpu_count())
		assert self.files, 'no files found matching ' + files

	def counts(self, query, subset=None, limit=None, indices=False):
		"""Run query and return a dict of the form {corpus1: nummatches, ...}.

		:param query: the search query
		:param subset: an iterable of filenames to run the query on; by default
			all filenames are used.
		:param limit: the maximum number of sentences to query per corpus; by
			default, all sentences are queried.
		:param indices: if True, return a multiset of indices of matching
			occurrences, instead of an integer count."""

	def trees(self, query, subset=None, maxresults=10,
			nofunc=False, nomorph=False):
		"""Run query and return list of matching trees.

		:param maxresults: the maximum number of matches to return.
		:returns: list of tuples of the form
			``(corpus, sentno, tree, sent, highlight)``
			highlight is a set of nodes from tree."""

	def sents(self, query, subset=None, maxresults=100, brackets=False):
		"""Run query and yield matching sentences.

		:param maxresults: the maximum number of matches to return.
		:param brackets: if True, return trees as they appear in the treebank;
			by default sentences are returned as a sequence of tokens.
		:returns: list of tuples of the form
			``(corpus, sentno, sent, highlight)``\
			sent is a single string with space-separated tokens;
			highlight is a set of integer indices."""


class TgrepSearcher(CorpusSearcher):
	"""Search a corpus with tgrep2."""
	def __init__(self, files, macros=None, numthreads=None):
		def convert(filename):
			"""Convert files not ending in .t2c.gz to tgrep2 format."""
			if (not filename.endswith('.t2c.gz')
					and not os.path.exists(filename + '.t2c.gz')):
				subprocess.check_call(
						args=[which('tgrep2'), '-p', filename,
							filename + '.t2c.gz'], shell=False,
						stdout=subprocess.PIPE, stderr=subprocess.PIPE)
				return filename + '.t2c.gz'
			return filename

		super(TgrepSearcher, self).__init__(files, macros, numthreads)
		self.files = {convert(filename): None for filename in self.files}

	def counts(self, query, subset=None, limit=None, indices=False):
		subset = tuple(subset or self.files)
		try:
			result = self.cache['counts', query, subset, limit]
		except KeyError:
			result = OrderedDict()
		else:
			return result
		# %s the sentence number
		fmt = r'%s:::\n'
		if indices:
			jobs = {self.pool.submit(workerfunc(lambda x: Counter(n for n, _
					in self._query(query, x, fmt, None, limit))), filename):
					filename for filename in subset}
		else:
			jobs = {self.pool.submit(workerfunc(lambda x: sum(1 for _
					in self._query(query, x, fmt, None, limit))), filename):
					filename for filename in subset}
		for future in concurrent.futures.as_completed(jobs):
			filename = jobs[future]
			result[filename] = future.result()
		self.cache['counts', query, subset, limit] = result
		return result

	def trees(self, query, subset=None, maxresults=10,
			nofunc=False, nomorph=False):
		subset = tuple(subset or self.files)
		try:
			result, maxresults2 = self.cache['trees', query, subset,
					nofunc, nomorph]
			if maxresults <= maxresults2:
				return result[:maxresults]
		except KeyError:
			result = []
		# %s the sentence number
		# %w complete tree in bracket notation
		# %h the matched subtree in bracket notation
		fmt = r'%s:::%w:::%h\n'
		jobs = {self.pool.submit(workerfunc(lambda x: list(self._query(
				query, x, fmt, maxresults))), filename):
				filename for filename in subset}
		for future in concurrent.futures.as_completed(jobs):
			filename = jobs[future]
			for sentno, line in future.result():
				treestr, match = line.split(':::')
				treestr = filterlabels(treestr, nofunc, nomorph)
				treestr = treestr.replace(" )", " -NONE-)")
				cnt = count()
				if match.startswith('('):
					treestr = treestr.replace(match, '%s_HIGH %s' % tuple(
							match.split(None, 1)), 1)
				else:
					match = ' %s)' % match
					treestr = treestr.replace(match, '_HIGH%s' % match)
				tree = Tree.parse(treestr, parse_leaf=lambda _: next(cnt))
				sent = re.findall(r" +([^ ()]+)(?=[ )])", treestr)
				high = list(tree.subtrees(lambda n: n.label.endswith("_HIGH")))
				if high:
					high = high.pop()
					high.label = high.label.rsplit("_", 1)[0]
					high = list(high.subtrees()) + high.leaves()
				result.append((filename, sentno, tree, sent, high))
		self.cache['trees', query, subset, nofunc, nomorph] = result, maxresults
		return result

	def sents(self, query, subset=None, maxresults=100, brackets=False):
		subset = tuple(subset or self.files)
		try:
			result, maxresults2 = self.cache['sents', query, subset, brackets]
			if maxresults <= maxresults2:
				return result[:maxresults]
		except KeyError:
			result = []
		# %s the sentence number
		# %w complete tree in bracket notation
		# %h the matched subtree in bracket notation
		fmt = r'%s:::%w:::%h\n'
		jobs = {self.pool.submit(workerfunc(lambda x: list(self._query(
				query, x, fmt, maxresults))), filename):
				filename for filename in subset}
		for future in concurrent.futures.as_completed(jobs):
			filename = jobs[future]
			for sentno, line in future.result():
				sent, match = line.split(':::')
				if not brackets:
					idx = sent.index(match)
					prelen = len(GETLEAVES.findall(sent[:idx]))
					sent = ' '.join(GETLEAVES.findall(sent))
					match = GETLEAVES.findall(
							match) if '(' in match else [match]
					match = set(range(prelen, prelen + len(match)))
				result.append((filename, sentno, sent, match))
		self.cache['sents', query, subset, brackets] = result, maxresults
		return result

	def _query(self, query, filename, fmt, maxresults=None, limit=None):
		"""Run a query on a single file."""
		cmd = [which('tgrep2'), '-z', '-a',
				'-m', fmt,
				'-c', os.path.join(filename)]
		if self.macros:
			cmd.append(self.macros)
		cmd.append(query)
		proc = subprocess.Popen(args=cmd,
				bufsize=-1, shell=False,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE)
		out, err = proc.communicate()
		out = out.decode('utf8')  # pylint: disable=E1103
		err = err.decode('utf8')  # pylint: disable=E1103
		proc.stdout.close()
		proc.stderr.close()
		proc.wait()
		if proc.returncode != 0:
			raise ValueError(err)
		# TODO: can we do this without reading stdout completely?
		results = ((int(match.group(1)), match.group(2)) for match
				in re.finditer(r'([0-9]+):::([^\n]*)\n', out))
		if limit is not None:
			results = takewhile(lambda x: x[0] < limit, results)
		return islice(results, maxresults)


class DactSearcher(CorpusSearcher):
	"""Search a dact corpus with xpath."""
	def __init__(self, files, macros=None, numthreads=None):
		super(DactSearcher, self).__init__(files, macros, numthreads)
		self.ids = {}
		self.indices = {}
		for filename in self.files:
			try:
				self.files[filename] = alpinocorpus.CorpusReader(
					filename, macrosFilename=macros)
			except TypeError:
				assert macros is None, 'macros not supported'
				self.files[filename] = alpinocorpus.CorpusReader(filename)
			# NB: this is wrong but reading the IDs is expensive
			# so better to cache the information somewhere else,
			# and call updateindex again
			self.updateindex(filename, [None] + ['%d.xml' % (n + 1)
					for n in range(self.files[filename].size())])

	def updateindex(self, filename, ids):
		"""Store mapping of sentence numbers to IDs.

		:param ids: a list of sentence IDs occurring in file"""
		self.ids[filename] = ids
		# store reverse mapping
		self.indices[filename] = {a: n for n, a
				in enumerate(self.ids[filename])}

	def counts(self, query, subset=None, limit=None, indices=False):
		subset = tuple(subset or self.files)
		try:
			result = self.cache['counts', query, subset, limit]
		except KeyError:
			result = OrderedDict()
		else:
			return result
		if indices:
			jobs = {self.pool.submit(workerfunc(lambda x: Counter(n for n, _
					in self._query(query, x, None, limit))), filename):
					filename for filename in subset}
		else:
			jobs = {self.pool.submit(workerfunc(lambda x: sum(1 for _
					in self._query(query, x, None, limit))), filename):
					filename for filename in subset}
		for future in concurrent.futures.as_completed(jobs):
			filename = jobs[future]
			result[filename] = future.result()
		self.cache['counts', query, subset, limit] = result
		return result

	def trees(self, query, subset=None, maxresults=10,
			nofunc=False, nomorph=False):
		subset = tuple(subset or self.files)
		try:
			result, maxresults2 = self.cache['trees', query, subset,
					nofunc, nomorph]
			if maxresults <= maxresults2:
				return result[:maxresults]
		except KeyError:
			result = []
		jobs = {self.pool.submit(workerfunc(lambda x: list(
				self._query(query, x, maxresults))), filename): filename
				for filename in subset}
		for future in concurrent.futures.as_completed(jobs):
			filename = jobs[future]
			for sentno, match in future.result():
				treestr = self.files[filename].read(match.name())
				match = match.contents().decode('utf8')
				tree, sent = treebank.alpinotree(
						ElementTree.fromstring(treestr),
						functions=None if nofunc else 'add',
						morphology=None if nomorph else 'replace')
				highwords = re.findall('<node[^>]*begin="([0-9]+)"[^>]*/>',
						match)
				high = set(re.findall(r'\bid="(.+?)"', match))
				high = list(tree.subtrees(lambda n:
						n.source[treebank.PARENT] in high or
						n.source[treebank.WORD].lstrip('#') in high))
				high += [int(a) for a in highwords]
				result.append((filename, sentno, tree, sent, high))
		self.cache['trees', query, subset,
				nofunc, nomorph] = result, maxresults
		return result

	def sents(self, query, subset=None, maxresults=100, brackets=False):
		subset = tuple(subset or self.files)
		try:
			result, maxresults2 = self.cache[
					'sents', query, subset, brackets]
			if maxresults <= maxresults2:
				return result[:maxresults]
		except KeyError:
			result = []
		jobs = {self.pool.submit(workerfunc(lambda x: list(
				self._query(query, x, maxresults))), filename): filename
				for filename in subset}
		for future in concurrent.futures.as_completed(jobs):
			filename = jobs[future]
			for sentno, match in future.result():
				treestr = self.files[filename].read(match.name()).decode('utf8')
				match = match.contents().decode('utf8')
				if not brackets:
					treestr = ALPINOLEAVES.search(treestr).group(1)
					# extract starting index of highlighted words
					match = set(int(a) for a in re.findall(
							'<node[^>]*begin="([0-9]+)"[^>]*/>', match))
				result.append((filename, sentno, treestr, match))
		self.cache['sents', query, subset, brackets] = result, maxresults
		return result

	def _query(self, query, filename, maxresults=None, limit=None):
		"""Run a query on a single file."""
		# NB: results aren't sorted, so we need to iterate exhaustively
		indices = self.indices[filename]
		results = ((n, entry) for n, entry
				in ((indices[entry.name()], entry)
					for entry in self.files[filename].xpath(query))
				if limit is None or n < limit)
		return islice(results, maxresults)


class RegexSearcher(CorpusSearcher):
	"""Search a plain text file in UTF-8 with regular expressions.

	Assumes that lines correspond to sentences."""
	def __init__(self, files, macros=None, numthreads=None):
		super(RegexSearcher, self).__init__(files, macros, numthreads)
		if macros:
			raise NotImplementedError
		for filename in self.files:
			self.files[filename] = list(io.open(filename, encoding='utf-8'))

	def counts(self, query, subset=None, limit=None, indices=False):
		subset = tuple(subset or self.files)
		try:
			result = self.cache['counts', query, subset, limit]
		except KeyError:
			result = OrderedDict()
		else:
			return result
		if indices:
			jobs = {self.pool.submit(workerfunc(lambda x: Counter(n for n, _
					in self._query(query, x, None, limit))), filename):
					filename for filename in subset}
		else:
			jobs = {self.pool.submit(workerfunc(lambda x: sum(1 for _
					in self._query(query, x, None, limit))), filename):
					filename for filename in subset}
		for future in concurrent.futures.as_completed(jobs):
			filename = jobs[future]
			result[filename] = future.result()
		self.cache['counts', query, subset, limit] = result
		return result

	def trees(self, query, subset=None, maxresults=10,
			nofunc=False, nomorph=False):
		raise ValueError

	def sents(self, query, subset=None, maxresults=100, brackets=False):
		if brackets:
			raise ValueError
		subset = tuple(subset or self.files)
		try:
			result, maxresults2 = self.cache['sents', query, subset]
			if maxresults <= maxresults2:
				return result[:maxresults]
		except KeyError:
			result = []
		jobs = {self.pool.submit(workerfunc(lambda x: list(self._query(
				query, x, maxresults))), filename):
				filename for filename in subset}
		for future in concurrent.futures.as_completed(jobs):
			filename = jobs[future]
			for sentno, match in future.result():
				sent = match.string
				start, end = match.span()
				prelen = len(sent[:start].split())
				matchlen = len(sent[start:end].split())
				highlight = set(range(prelen, prelen + matchlen))
				result.append((filename, sentno, sent.rstrip(), highlight))
		self.cache['sents', query, subset] = result, maxresults
		return result

	def _query(self, query, filename, maxresults=None, limit=None):
		"""Run a query on a single file."""
		regex = re.compile(query)
		results = ((n, match) for n, match in
				enumerate((regex.search(a) for a in self.files[filename]), 1)
				if match is not None)
		if limit is not None:
			results = takewhile(lambda x: x[0] < limit, results)
		return islice(results, maxresults)


def filterlabels(line, nofunc, nomorph):
	"""Remove morphological and/or grammatical function labels from tree(s)."""
	if nofunc:
		line = FUNC_TAGS.sub('', line)
	if nomorph:
		line = MORPH_TAGS.sub(lambda g: '%s%s' % (
				ABBRPOS.get(g.group(1), g.group(1)), g.group(2)), line)
	return line
