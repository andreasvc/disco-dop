"""Objects for searching through collections of trees."""

# Possible improvements:
# - cache raw results from _query() before conversion?
# - return/cache trees as strings?

from __future__ import division, print_function, absolute_import, \
		unicode_literals
import io
import os
import re
import sys
import gzip
import concurrent.futures
import multiprocessing
import subprocess
from collections import Counter, OrderedDict, namedtuple
from itertools import islice
try:
	import cPickle as pickle
except ImportError:
	import pickle
try:
	import alpinocorpus
	import xml.etree.cElementTree as ElementTree
	ALPINOCORPUSLIB = True
except ImportError:
	ALPINOCORPUSLIB = False
from roaringbitmap import RoaringBitmap
from discodop import treebank, treetransforms, _fragments
from discodop.tree import Tree
from discodop.parser import workerfunc, which
from discodop.treedraw import ANSICOLOR, DrawTree
from discodop.containers import Vocabulary

SHORTUSAGE = '''Search through treebanks with queries.
Usage: %s [--engine=(tgrep2|xpath|regex)] [-t|-s|-c] <query> <treebank>...\
''' % sys.argv[0]
CACHESIZE = 1024
GETLEAVES = re.compile(r' ([^ ()]+)(?=[ )])')
ALPINOLEAVES = re.compile('<sentence>(.*)</sentence>')
MORPH_TAGS = re.compile(r'([/*\w]+)(?:\[[^ ]*\]\d?)?((?:-\w+)?(?:\*\d+)? )')
FUNC_TAGS = re.compile(r'-\w+')

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

CorpusInfo = namedtuple('CorpusInfo',
		['len', 'numwords', 'numnodes', 'maxnodes'])


class CorpusSearcher(object):
	"""Abstract base class to wrap corpus files that can be queried."""
	def __init__(self, files, macros=None, numthreads=None):
		"""
		:param files: a sequence of filenames of corpora
		:param macros: a filename with macros that can be used in queries.
		:param numthreads: the number of concurrent threads to use;
			pass 1 to disable threading."""
		if not isinstance(files, (list, tuple, set, dict)) or (
				not all(isinstance(a, str) for a in files)):
			raise ValueError('"files" argument must be a sequence of filenames.')
		self.files = OrderedDict.fromkeys(files)
		self.macros = macros
		self.numthreads = numthreads
		self.cache = FIFOOrederedDict(CACHESIZE)
		self.pool = concurrent.futures.ThreadPoolExecutor(
				numthreads or cpu_count())
		if not self.files:
			raise ValueError('no files found matching ' + files)

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
		:param nofunc, nomorph: whether to remove / add function tags and
			morphological features from trees.
		:returns: list of tuples of the form
			``(corpus, sentno, tree, sent, highlight)``
			highlight is a list of matched Tree nodes from tree."""

	def sents(self, query, subset=None, maxresults=100, brackets=False):
		"""Run query and return matching sentences.

		:param maxresults: the maximum number of matches to return.
		:param brackets: if True, return trees as they appear in the treebank,
			match is a string with the matching subtree.
			If False (default), sentences are returned as a sequence of tokens.
		:returns: list of tuples of the form
			``(corpus, sentno, sent, match)``
			sent is a single string with space-separated tokens;
			match is an iterable of integer indices of tokens matched
			by the query."""

	def batchcounts(self, queries, subset=None, limit=None):
		"""Like counts, but executes a sequence of queries.

		Useful in combination with ``pandas.DataFrame``.

		:returns: a dict of the form
			``{corpus1: {query1: nummatches, query2: nummatches, ...}, ...}``.
		"""
		result = OrderedDict((name, OrderedDict())
				for name in subset or self.files)
		for query in queries:
			for filename, value in self.counts(query, subset, limit).items():
				result[filename][query] = value
		return result

	def extract(self, filename, indices,
			nofunc=False, nomorph=False, sents=False):
		"""Extract a range of trees / sentences.

		:param filename: one of the filenames in ``self.files``
		:param indices: indices of sentences to extract
			(1-based, excluding empty lines)
		:param sents: if True, return sentences instead of trees.
			Sentences are strings with space-separated tokens.
		:param nofunc, nomorph: same as for ``trees()`` method.
		:returns: a list of Tree objects or sentences."""

	def _submit(self, func, filename):
		"""Submit a job to the thread pool."""
		if self.numthreads == 1:
			return NoFuture(func, filename)
		return self.pool.submit(workerfunc(func), filename)

	def _as_completed(self, jobs):
		"""Return jobs as they are completed."""
		if self.numthreads == 1:
			return jobs
		return concurrent.futures.as_completed(jobs)


class TgrepSearcher(CorpusSearcher):
	"""Search a corpus with tgrep2."""
	def __init__(self, files, macros=None, numthreads=None):
		def convert(filename):
			"""Convert files not ending in .t2c.gz to tgrep2 format."""
			if filename.endswith('.t2c.gz'):
				return filename
			elif not os.path.exists(filename + '.t2c.gz'):
				subprocess.check_call(
						args=[which('tgrep2'), '-p', filename,
							filename + '.t2c.gz'], shell=False,
						stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			return filename + '.t2c.gz'

		super(TgrepSearcher, self).__init__(files, macros, numthreads)
		self.files = {convert(filename): None for filename in self.files}

	def counts(self, query, subset=None, limit=None, indices=False):
		subset = subset or self.files
		result = OrderedDict()
		jobs = {}
		# %s the sentence number
		fmt = r'%s:::\n'
		for filename in subset:
			try:
				result[filename] = self.cache[
						'counts', query, filename, limit, indices]
			except KeyError:
				if indices:
					jobs[self._submit(lambda x: Counter(n for n, _
							in self._query(query, x, fmt, None, limit)),
							filename)] = filename
				else:
					jobs[self._submit(lambda x: sum(1 for _
						in self._query(query, x, fmt, None, limit)),
						filename)] = filename
		for future in self._as_completed(jobs):
			filename = jobs[future]
			self.cache['counts', query, filename, limit, indices
					] = result[filename] = future.result()
		return result

	def trees(self, query, subset=None, maxresults=10,
			nofunc=False, nomorph=False):
		subset = subset or self.files
		# %s the sentence number
		# %w complete tree in bracket notation
		# %m all marked nodes, or the head node if none are marked
		fmt = r'%s:::%w:::%m\n'
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['trees', query, filename,
						nofunc, nomorph]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(lambda x: list(self._query(
						query, x, fmt, maxresults)), filename)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for sentno, line in future.result():
				treestr, match = line.split(':::')
				treestr = filterlabels(treestr, nofunc, nomorph)
				treestr = treestr.replace(" )", " -NONE-)")
				if match.startswith('('):
					treestr = treestr.replace(match, '%s_HIGH %s' % tuple(
							match.split(None, 1)), 1)
				else:
					match = ' %s)' % match
					treestr = treestr.replace(match, '_HIGH%s' % match)
				tree, sent = treebank.termindices(treestr)
				tree = treetransforms.mergediscnodes(Tree(tree))
				sent = [word.replace('-LRB-', '(').replace('-RRB-', ')')
						for word in sent]
				high = list(tree.subtrees(lambda n: n.label.endswith("_HIGH")))
				if high:
					high = high.pop()
					high.label = high.label.rsplit("_", 1)[0]
					high = list(high.subtrees()) + high.leaves()
				x.append((filename, sentno, tree, sent, high))
			self.cache['trees', query, filename,
					nofunc, nomorph] = x, maxresults
			result.extend(x)
		return result

	def sents(self, query, subset=None, maxresults=100, brackets=False):
		subset = subset or self.files
		# %s the sentence number
		# %w complete tree in bracket notation
		# %m all marked nodes, or the head node if none are marked
		fmt = r'%s:::%w:::%m\n'
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['sents', query, filename, brackets]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(lambda x: list(self._query(
						query, x, fmt, maxresults)), filename)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for sentno, line in future.result():
				sent, match = line.split(':::')
				if not brackets:
					idx = sent.index(match if match.startswith('(')
							else ' %s)' % match)
					prelen = len(GETLEAVES.findall(sent[:idx]))
					sent = ' '.join(
							word.replace('-LRB-', '(').replace('-RRB-', ')')
							for word in GETLEAVES.findall(sent))
					match = GETLEAVES.findall(
							match) if '(' in match else [match]
					match = range(prelen, prelen + len(match))
				x.append((filename, sentno, sent, match))
			self.cache['sents', query, filename, brackets] = x, maxresults
			result.extend(x)
		return result

	def extract(self, filename, indices,
			nofunc=False, nomorph=False, sents=False):
		if not filename.endswith('.t2c.gz'):
			filename += '.t2c.gz'
		cmd = [which('tgrep2'),
				'-e', '-',  # extraction mode
				'-c', filename]
		if sents:
			cmd.append('-t')
		proc = subprocess.Popen(args=cmd,
				bufsize=0,
				shell=False,
				stdin=subprocess.PIPE,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE)
		out, err = proc.communicate(''.join(
				'%d:1\n' % n for n in indices if n > 0))
		proc.stdout.close()
		proc.stderr.close()
		if proc.returncode != 0:
			raise ValueError(err.decode('utf8'))
		result = out.decode('utf8').splitlines()
		if sents:
			return result
		return [(treetransforms.mergediscnodes(Tree(tree)),
				[word.replace('-LRB-', '(').replace('-RRB-', ')')
					for word in sent])
				for tree, sent
				in (treebank.termindices(filterlabels(
					treestr, nofunc, nomorph)) for treestr in result)]

	def _query(self, query, filename, fmt, maxresults=None, limit=None):
		"""Run a query on a single file."""
		cmd = [which('tgrep2'), '-a',  # print all matches for each sentence
				# '-z',  # pretty-print search pattern on stderr
				'-m', fmt,
				'-c', os.path.join(filename)]
		if self.macros:
			cmd.append(self.macros)
		cmd.append(query)
		proc = subprocess.Popen(
				args=cmd, shell=False, bufsize=0,
				stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		linere = re.compile(r'([0-9]+):::([^\n]*)')
		if limit or maxresults:
			results = []
			for n, line in enumerate(iter(proc.stdout.readline, b'')):
				match = linere.match(line.decode('utf8'))
				m, a = int(match.group(1)), match.group(2)
				if (limit and m >= limit) or (maxresults and n >= maxresults):
					proc.stdout.close()
					proc.stderr.close()
					proc.terminate()
					return results
				results.append((m, a))
			if proc.wait() != 0:
				proc.stdout.close()
				err = proc.stderr.read().decode('utf8')
				proc.stderr.close()
		else:
			out, err = proc.communicate()
			out = out.decode('utf8')
			err = err.decode('utf8')
			proc.stdout.close()
			proc.stderr.close()
			results = ((int(match.group(1)), match.group(2)) for match
					in re.finditer(r'([0-9]+):::([^\n]*)\n', out))
		if proc.returncode != 0:
			raise ValueError(err)
		return results


class DactSearcher(CorpusSearcher):
	"""Search a dact corpus with xpath."""
	def __init__(self, files, macros=None, numthreads=None):
		super(DactSearcher, self).__init__(files, macros, numthreads)
		if not ALPINOCORPUSLIB:
			raise ImportError('Could not import `alpinocorpus` module.')
		for filename in self.files:
			self.files[filename] = alpinocorpus.CorpusReader(filename)
		if macros is not None:
			try:
				self.macros = alpinocorpus.Macros(macros)
			except NameError:
				raise ValueError('macros not supported')

	def counts(self, query, subset=None, limit=None, indices=False):
		subset = subset or self.files
		result = OrderedDict()
		jobs = {}
		for filename in subset:
			try:
				result[filename] = self.cache[
						'counts', query, filename, limit, indices]
			except KeyError:
				if indices:
					jobs[self._submit(lambda x: Counter(n for n, _
							in self._query(query, x, None, limit)),
							filename)] = filename
				else:
					jobs[self._submit(lambda x: sum(1 for _
						in self._query(query, x, None, limit)),
						filename)] = filename
		for future in self._as_completed(jobs):
			filename = jobs[future]
			self.cache['counts', query, filename, limit, indices
					] = result[filename] = future.result()
		return result

	def trees(self, query, subset=None, maxresults=10,
			nofunc=False, nomorph=False):
		subset = subset or self.files
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['trees', query, filename,
						nofunc, nomorph]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(lambda x: list(self._query(
						query, x, maxresults)), filename)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for sentno, match in future.result():
				treestr = self.files[filename].read(match.name())
				match = match.contents().decode('utf8')
				item = treebank.alpinotree(
						ElementTree.fromstring(treestr),
						functions=None if nofunc else 'add',
						morphology=None if nomorph else 'replace')
				highwords = re.findall('<node[^>]*begin="([0-9]+)"[^>]*/>',
						match)
				high = set(re.findall(r'\bid="(.+?)"', match))
				high = [node for node in item.tree.subtrees()
						if node.source[treebank.PARENT] in high
						or node.source[treebank.WORD].lstrip('#') in high]
				high += [int(a) for a in highwords]
				x.append((filename, sentno, item.tree, item.sent, high))
			self.cache['trees', query, filename,
					nofunc, nomorph] = x, maxresults
			result.extend(x)
		return result

	def sents(self, query, subset=None, maxresults=100, brackets=False):
		subset = subset or self.files
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['sents', query, filename, brackets]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(lambda x: list(self._query(
						query, x, maxresults)), filename)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for sentno, match in future.result():
				treestr = self.files[filename].read(match.name()).decode('utf8')
				match = match.contents().decode('utf8')
				if not brackets:
					treestr = ALPINOLEAVES.search(treestr).group(1)
					# extract starting index of highlighted words
					match = {int(a) for a in re.findall(
							'<node[^>]*begin="([0-9]+)"[^>]*/>', match)}
				x.append((filename, sentno, treestr, match))
			self.cache['sents', query, filename, brackets] = x, maxresults
			result.extend(x)
		return result

	def extract(self, filename, indices,
			nofunc=False, nomorph=False, sents=False):
		results = [self.files[filename].read('%8d' % n)
					for n in indices if n > 0]
		if sents:
			return [ElementTree.fromstring(result).find('sentence').text
					for result in results]
		else:
			return [(item.tree, item.sent) for item
					in (treebank.alpinotree(
						ElementTree.fromstring(treestr),
						functions=None if nofunc else 'add',
						morphology=None if nomorph else 'replace')
					for treestr in results)]

	def _query(self, query, filename, maxresults=None, limit=None):
		"""Run a query on a single file."""
		if self.macros is not None:
			query = self.macros.expand(query)
		results = ((n, entry) for n, entry
				in ((entry.name(), entry)
					for entry in self.files[filename].xpath(query))
				if limit is None or n < limit)
		return islice(results, maxresults)


class FragmentSearcher(CorpusSearcher):
	"""Search for fragments in a bracket treebank.

	Format of treebanks and queries can be bracket, discbracket, or
	export (autodetected).
	Each query consists of one or more tree fragments, and the results
	will be merged together, except with batchcounts(), which returns
	the results for each fragment separately."""
	def __init__(self, files, macros=None, numthreads=None):
		super(FragmentSearcher, self).__init__(files, macros, numthreads)
		for filename in self.files:
			if os.path.exists('%s.pkl.gz' % filename):
				corpus, vocab = pickle.loads(
						gzip.open('%s.pkl.gz' % filename).read())
			else:
				# get format from extension
				ext = {'export': 'export', 'mrg': 'bracket',
						'dbr': 'discbracket'}
				fmt = ext[filename.split('.')[-1]]
				vocab = Vocabulary()
				corpus = _fragments.readtreebank(
						filename, vocab, fmt=fmt)
				corpus.indextrees(vocab)
				gzip.open('%s.pkl.gz' % filename, 'w').write(
						pickle.dumps((corpus, vocab), protocol=-1))
			self.files[filename] = CorpusInfo(
					len=corpus.len, numwords=corpus.numwords,
					numnodes=corpus.numnodes, maxnodes=corpus.maxnodes)
		self.macros = {}
		if macros:
			with io.open(macros, encoding='utf8') as tmp:
				self.macros = dict(line.split('=', 1) for line in tmp)

	def counts(self, query, subset=None, limit=None, indices=False):
		subset = subset or self.files
		result = OrderedDict()
		jobs = {}
		for filename in subset:
			try:
				result[filename] = self.cache[
						'counts', query, filename, limit, indices]
			except KeyError:
				if indices:
					jobs[self._submit(lambda x: self._query(
							query, x, None, limit, indices=True, trees=False),
							filename)] = filename
				else:
					jobs[self._submit(lambda x: self._query(
							query, x, None, limit, indices=indices,
							trees=False), filename)] = filename
		for future in self._as_completed(jobs):
			filename = jobs[future]
			if indices:
				result[filename] = Counter()
				for _, b in future.result():
					result[filename].update(b)
			else:
				result[filename] = sum(b for _, b in future.result())
			self.cache['counts', query, filename, limit, indices
					] = result[filename]
		return result

	def batchcounts(self, queries, subset=None, limit=None):
		subset = subset or self.files
		result = OrderedDict()
		jobs = {}
		query = '\n'.join(queries) + '\n'
		for filename in subset:
			# try:
			# 	result[filename] = self.cache[
			# 			'counts', query, filename, limit, indices]
			# except KeyError:
			jobs[self._submit(lambda x:
				self._query(query, x, None, limit, indices=False, trees=False),
				filename)] = filename
		for future in self._as_completed(jobs):
			filename = jobs[future]
			# self.cache['counts', query, filename, limit, False] =
			result[filename] = OrderedDict(future.result())
		return result

	def trees(self, query, subset=None, maxresults=10,
			nofunc=False, nomorph=False):
		if nofunc or nomorph:
			raise NotImplementedError
		subset = subset or self.files
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['trees', query, filename,
						nofunc, nomorph]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(lambda x: list(self._query(
						query, x, maxresults, trees=True)),
						filename)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for frag, matches in future.result():
				for sentno, (tree, sent) in matches.items():
					high = []  # FIXME matching subtree not returned
					x.append((filename, sentno, tree, sent, high))
			self.cache['trees', query, filename,
					nofunc, nomorph] = x, maxresults
			result.extend(x)
		return result

	def sents(self, query, subset=None, maxresults=100, brackets=False):
		subset = subset or self.files
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['sents', query, filename, brackets]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(lambda x: list(self._query(
						query, x, maxresults, trees=True)),
						filename)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for frag, matches in future.result():
				for sentno, (tree, sent) in matches.items():
					sent = ' '.join(sent)
					if brackets:
						sent = str(tree) + '\t' + sent
						match = '\t'.join((
							str(frag[0]),
							' '.join(a or '' for a in frag[1])))
					else:
						xsent = sent.split(' ')
						match = {xsent.index(a) for a in frag[1]
								if a is not None}
					x.append((filename, sentno, sent, match))
			self.cache['sents', query, filename, brackets] = x, maxresults
			result.extend(x)
		return result

	def extract(self, filename, indices,
			nofunc=False, nomorph=False, sents=False):
		if nofunc or nomorph:
			raise NotImplementedError
		corpus, vocab = pickle.loads(gzip.open('%s.pkl.gz' % filename).read())
		if sents:
			return [' '.join(corpus.extractsent(n - 1, vocab)) for n in indices]
		return [(treetransforms.mergediscnodes(Tree(treestr)), sent)
				for treestr, sent
				in (corpus.extract(n - 1, vocab) for n in indices)]

	def _query(self, query, filename, maxresults=None, limit=None,
			indices=True, trees=False):
		"""Run a query on a single file."""
		if self.macros is not None:
			query = query.format(**self.macros)
		corpus, vocab = pickle.loads(gzip.open('%s.pkl.gz' % filename).read())
		trees, sents = zip(*[(treetransforms.binarize(a, dot=True), b)
				for a, b, _ in treebank.incrementaltreereader(
					io.StringIO(query))])
		queries = _fragments.getctrees(
				list(trees), list(sents), vocab=vocab)
		maxnodes = max(queries['trees1'].maxnodes,
					max(self.files[filename].maxnodes for filename
						in self.files))
		fragments = _fragments.completebitsets(
				queries['trees1'], vocab, maxnodes, discontinuous=True)
		results = ((frag, multiset) for frag, multiset in zip(
					fragments.keys(),
					_fragments.exactcounts(
						queries['trees1'],
						corpus,
						list(fragments.values()),
						indices=1 if indices else 0,
						maxnodes=maxnodes, limit=limit)))
		if maxresults:
			results = ((a, list(islice(b, maxresults))) for a, b in results)
		if indices and trees:
			results = ((a, {n + 1: corpus.extract(n, vocab) for n in b})
					for a, b in results)
		elif indices:
			results = ((a, {n + 1: m for n, m in b.items()})
					for a, b in results)
		return results


class RegexSearcher(CorpusSearcher):
	"""Search a plain text file in UTF-8 with regular expressions.

	Assumes that non-empty lines correspond to sentences; empty lines
	do not count towards line numbers (e.g., when used as paragraph breaks).

	:param macros: a file containing lines of the form ``'name=regex'``;
		an occurrence of ``'{name}'`` will be suitably replaced when it
		appears in a query."""
	def __init__(self, files, macros=None, numthreads=None):
		super(RegexSearcher, self).__init__(files, macros, numthreads)
		self.macros = {}
		if macros:
			with io.open(macros, encoding='utf8') as tmp:
				self.macros = dict(line.split('=', 1) for line in tmp)
		self.lineindex = dict.fromkeys(files)

	def _indexfile(self, filename, data):
		"""Cache locations of non-empty lines."""
		self.lineindex[filename] = RoaringBitmap(
				match.end() for match in re.finditer(
					r'[^ \t\n\r][ \t]*[\r\n]+', data))

	def counts(self, query, subset=None, limit=None, indices=False):
		subset = subset or self.files
		result = OrderedDict()
		jobs = {}
		for filename in subset:
			try:
				result[filename] = self.cache[
						'counts', query, filename, limit, indices]
			except KeyError:
				if indices:
					jobs[self._submit(lambda x: Counter(y[0] for y
							in self._query(query, x, None, limit)),
							filename)] = filename
				else:
					jobs[self._submit(lambda x: sum(1 for _
						in self._query(query, x, None, limit)),
						filename)] = filename
		for future in self._as_completed(jobs):
			filename = jobs[future]
			self.cache['counts', query, filename, limit, indices
					] = result[filename] = future.result()
		return result

	def sents(self, query, subset=None, maxresults=100, brackets=False):
		if brackets:
			raise ValueError('not applicable with plain text corpus.')
		subset = subset or self.files
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['sents', query, filename]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(lambda x: list(self._query(
						query, x, maxresults)), filename)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for sentno, sent, start, end in future.result():
				prelen = len(sent[:start].split())
				matchlen = len(sent[start:end].split())
				highlight = range(prelen, prelen + matchlen)
				x.append((filename, sentno, sent.rstrip(), highlight))
			self.cache['sents', query, filename] = x, maxresults
			result.extend(x)
		return result

	def trees(self, query, subset=None, maxresults=10,
			nofunc=False, nomorph=False):
		raise ValueError('not applicable with plain text corpus.')

	def extract(self, filename, indices,
			nofunc=False, nomorph=False, sents=True):
		if not sents:
			raise ValueError('not applicable with plain text corpus.')
		with io.open(filename, encoding='utf8') as tmp:
			data = tmp.read()
		if self.lineindex[filename] is None:
			self._indexfile(filename, data)

		end = next(reversed(self.lineindex[filename]))
		result = []
		for n in indices:
			a, b = 0, end
			if 1 < n < len(self.lineindex[filename]):
				a = self.lineindex[filename].select(n - 1)
			if 1 <= n < len(self.lineindex[filename]):
				b = self.lineindex[filename].select(n)
			result.append(data[a:b])
		return result

	def _query(self, query, filename, maxresults=None, limit=None):
		"""Run a query on a single file."""
		with io.open(filename, encoding='utf8') as tmp:
			data = tmp.read()
			if self.lineindex[filename] is None:
				self._indexfile(filename, data)
			for match in islice(re.finditer(
					query.format(**self.macros), data), maxresults):
				start, end = match.span()
				lineno = self.lineindex[filename].rank(start)
				if limit and lineno >= limit:
					break
				offset, nextoffset = 0, len(match.string)
				if lineno > 0:
					offset = self.lineindex[filename].select(lineno - 1)
				if lineno < len(self.lineindex[filename]):
					nextoffset = self.lineindex[filename].select(lineno)
				sent = match.string[offset:nextoffset]
				yield lineno + 1, sent, start - offset, end - offset


class NoFuture(object):
	"""A non-asynchronous version of concurrent.futures.Future."""
	def __init__(self, func, arg):
		self._result = func(arg)

	def result(self, timeout=None):  # pylint: disable=unused-argument
		"""Return the precomputed result."""
		return self._result


class FIFOOrederedDict(OrderedDict):
	"""FIFO cache with maximum number of elements based on OrderedDict."""
	def __init__(self, limit):
		super(FIFOOrederedDict, self).__init__()
		self.limit = limit

	def __setitem__(self, key, value):  # pylint: disable=arguments-differ
		if key in self:
			self.pop(key)
		elif len(self) >= self.limit:
			self.pop(next(iter(self)))
		super(FIFOOrederedDict, self).__setitem__(key, value)


def filterlabels(line, nofunc, nomorph):
	"""Remove morphological and/or grammatical function labels from tree(s)."""
	if nofunc:
		line = FUNC_TAGS.sub('', line)
	if nomorph:
		line = MORPH_TAGS.sub(lambda g: '%s%s' % (
				ABBRPOS.get(g.group(1), g.group(1)), g.group(2)), line)
	return line


def cpu_count():
	"""Return number of CPUs or 1."""
	try:
		return multiprocessing.cpu_count()
	except NotImplementedError:
		return 1


def main():
	"""CLI."""
	from getopt import gnu_getopt, GetoptError
	shortoptions = 'e:m:M:stcbnofh'
	options = ('engine= macros= numthreads= max-count= '
			'trees sents brackets counts only-matching line-number file help')
	try:
		opts, args = gnu_getopt(sys.argv[1:], shortoptions, options.split())
		query, corpora = args[0], args[1:]
		if isinstance(query, bytes):
			query = query.decode('utf8')
		if not corpora:
			raise ValueError('enter one or more corpus files')
	except (GetoptError, IndexError, ValueError) as err:
		print('error: %r' % err, file=sys.stderr)
		print(SHORTUSAGE)
		sys.exit(2)
	opts = dict(opts)
	if '--file' in opts or '-f' in opts:
		query = io.open(query, encoding='utf8').read()
	macros = opts.get('--macros', opts.get('-M'))
	engine = opts.get('--engine', opts.get('-e', 'tgrep2'))
	maxresults = int(opts.get('--max-count', opts.get('-m', 100)))
	numthreads = int(opts.get('--numthreads', 0)) or None
	if engine == 'tgrep2':
		searcher = TgrepSearcher(corpora, macros=macros, numthreads=numthreads)
	elif engine == 'xpath':
		searcher = DactSearcher(corpora, macros=macros, numthreads=numthreads)
	elif engine == 'regex':
		searcher = RegexSearcher(corpora, macros=macros, numthreads=numthreads)
	elif engine == 'frag':
		searcher = FragmentSearcher(
				corpora, macros=macros, numthreads=numthreads)
	else:
		raise ValueError('incorrect --engine value: %r' % engine)
	if '--counts' in opts or '-c' in opts:
		for filename, cnt in searcher.counts(
				query, limit=maxresults).items():
			if len(corpora) > 1:
				print('\x1b[%dm%s\x1b[0m:' % (
					ANSICOLOR['magenta'], filename), end='')
			print(cnt)
	elif '--trees' in opts or '-t' in opts:
		for filename, sentno, tree, sent, high in searcher.trees(
				query, maxresults=maxresults):
			if '--only-matching' in opts or '-o' in opts:
				tree = max(high, key=lambda x:
						len(x.leaves()) if isinstance(x, Tree) else 1)
			out = DrawTree(tree, sent, highlight=high).text(
					unicodelines=True, ansi=True)
			if len(corpora) > 1:
				print('\x1b[%dm%s\x1b[0m:' % (
						ANSICOLOR['magenta'], filename), end='')
			if '--line-number' in opts or '-n' in opts:
				print('\x1b[0m:\x1b[%dm%s\x1b[0m:'
						% (ANSICOLOR['green'], sentno), end='')
			print(out)
	else:  # sentences or brackets
		brackets = '--brackets' in opts or '-b' in opts
		for filename, sentno, sent, high in searcher.sents(
				query, brackets=brackets, maxresults=maxresults):
			if brackets:
				if '--only-matching' in opts or '-o' in opts:
					out = high
				else:
					out = sent.replace(high, "\x1b[%d;1m%s\x1b[0m" % (
							ANSICOLOR['red'], high))
			else:
				if '--only-matching' in opts or '-o' in opts:
					out = ' '.join(word if n in high else ''
							for n, word in enumerate(sent.split())).strip()
				else:
					out = ' '.join(
							'\x1b[%d;1m%s\x1b[0m' % (ANSICOLOR['red'], word)
							if x in high else word
							for x, word in enumerate(sent.split()))
			if len(corpora) > 1:
				print('\x1b[%dm%s\x1b[0m:' % (
						ANSICOLOR['magenta'], filename), end='')
			if '--line-number' in opts or '-n' in opts:
				print('\x1b[0m:\x1b[%dm%s\x1b[0m:'
						% (ANSICOLOR['green'], sentno), end='')
			print(out)


__all__ = ['CorpusSearcher', 'TgrepSearcher', 'DactSearcher', 'RegexSearcher',
		'NoFuture', 'FIFOOrederedDict', 'filterlabels', 'cpu_count']

if __name__ == '__main__':
	main()
