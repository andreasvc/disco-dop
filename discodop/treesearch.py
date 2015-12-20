"""Objects for searching through collections of trees."""

# Possible improvements:
# - cache raw results from _query() before conversion?
# - return/cache trees as strings?

from __future__ import division, print_function, absolute_import, \
		unicode_literals
import io
import os
import re
import csv
import sys
import gzip
import mmap
import array
import concurrent.futures
import multiprocessing
import subprocess
from collections import Counter, namedtuple
import collections
from itertools import islice
PY2 = sys.version_info[0] == 2
try:
	import cPickle as pickle
except ImportError:
	import pickle
try:
	import re2
	RE2LIB = True
except ImportError:
	RE2LIB = False
try:
	import alpinocorpus
	import xml.etree.cElementTree as ElementTree
	ALPINOCORPUSLIB = True
except ImportError:
	ALPINOCORPUSLIB = False
from cyordereddict import OrderedDict
from roaringbitmap import RoaringBitmap
from . import treebank, _fragments
from .tree import Tree, DrawTree, DiscTree, brackettree, ptbunescape
from .treetransforms import binarize, mergediscnodes, handledisc
from .util import which, workerfunc, openread, ANSICOLOR
from .containers import Vocabulary

SHORTUSAGE = '''Search through treebanks with queries.
Usage: discodop treesearch [-e (tgrep2|xpath|frag|regex)] [-t|-s|-c] \
<query> <treebank1>...'''
CACHESIZE = 32767
GETLEAVES = re.compile(r' (?:[0-9]+=)?([^ ()]+)(?=[ )])')
LEAFINDICES = re.compile(r' ([0-9]+)=')
LEAFINDICESWORDS = re.compile(r' ([0-9]+)=([^ ()]+)\)')
ALPINOLEAVES = re.compile('<sentence>(.*)</sentence>')
MORPH_TAGS = re.compile(r'([/*\w]+)(?:\[[^ ]*\]\d?)?((?:-\w+)?(?:\*\d+)? )')
FUNC_TAGS = re.compile(r'-\w+')

FRAG_FILES = None
FRAG_MACROS = None
VOCAB = None
REGEX_LINEINDEX = None
REGEX_MACROS = None
CorpusInfo = namedtuple('CorpusInfo',
		['len', 'numwords', 'numnodes', 'maxnodes'])


class CorpusSearcher(object):
	"""Abstract base class to wrap corpus files that can be queried."""
	def __init__(self, files, macros=None, numproc=None):
		"""
		:param files: a sequence of filenames of corpora
		:param macros: a filename with macros that can be used in queries.
		:param numproc: the number of concurrent threads / processes to use;
			pass 1 to use a single core."""
		if not isinstance(files, (list, tuple, set, dict)):
			raise ValueError('"files" argument must be a sequence.')
		for a in files:
			if not os.path.isfile(a):
				raise ValueError('filenames in "files" argument must exist. '
						'%r not found.' % a)
		self.files = OrderedDict.fromkeys(files)
		self.macros = macros
		self.numproc = numproc or cpu_count()
		self.cache = FIFOOrederedDict(CACHESIZE)
		self.pool = concurrent.futures.ThreadPoolExecutor(
				self.numproc)
		if not self.files:
			raise ValueError('no files found: %s' % files)

	def counts(self, query, subset=None, start=None, end=None, indices=False,
			breakdown=False):
		"""Run query and return a dict of the form {corpus1: nummatches, ...}.

		:param query: the search query
		:param subset: an iterable of filenames to run the query on; by default
			all filenames are used.
		:param start, end: the interval of sentences to query in each corpus;
			by default, all sentences are queried. 1-based, inclusive.
		:param indices: if True, return a sequence of indices of matching
			occurrences, instead of an integer count.
		:param breakdown: if True, return a Counter mapping matches to counts.
		"""

	def trees(self, query, subset=None, start=None, end=None, maxresults=10,
			nofunc=False, nomorph=False):
		"""Run query and return list of matching trees.

		:param start, end: the interval of sentences to query in each corpus;
			by default, all sentences are queried. 1-based, inclusive.
		:param maxresults: the maximum number of matches to return.
		:param nofunc, nomorph: whether to remove / add function tags and
			morphological features from trees.
		:returns: list of tuples of the form
			``(corpus, sentno, tree, sent, highlight)``
			highlight is a list of matched Tree nodes from tree."""

	def sents(self, query, subset=None, start=None, end=None, maxresults=100,
			brackets=False):
		"""Run query and return matching sentences.

		:param start, end: the interval of sentences to query in each corpus;
			by default, all sentences are queried. 1-based, inclusive.
		:param maxresults: the maximum number of matches to return;
			pass ``None`` for no limit.
		:param brackets: if True, return trees as they appear in the treebank,
			match1 and match2 are strings with the matching subtree.
			If False (default), sentences are returned as a sequence of tokens.
		:returns: list of tuples of the form
			``(corpus, sentno, sent, match1, match2)``
			sent is a single string with space-separated tokens;
			match1 and match2 are iterables of integer indices of characters
			matched by the query. If the distinction is applicable, match2
			contains the complete subtree, of which match1 is a subset."""

	def batchcounts(self, queries, subset=None, start=None, end=None):
		"""Like ``counts()``, but executes multiple queries on multiple files.

		Useful in combination with ``pandas.DataFrame``; e.g.::

			queries = ['NP < PP', 'VP < PP']
			corpus = treesearch.TgrepSearcher(glob.glob('*.mrg'))
			pandas.DataFrame.from_items(list(corpus.batchcounts(queries)),
					orient='index', columns=queries)

		:param queries: an iterable of strings.
		:param start, end: the interval of sentences to query in each corpus;
			by default, all sentences are queried. 1-based, inclusive.
		:yields: tuples of the form
			``(corpus1, [count1, count2, ...])``.
			where ``count1, count2, ...`` corresponds to ``queries``.
			Order of queries and corpora is preserved.
		"""
		result = OrderedDict((name, [])
				for name in subset or self.files)
		for query in queries:
			for filename, value in self.counts(
					query, subset, start, end).items():
				result[filename].append(value)
		for filename, counts in result.items():
			yield filename, counts

	def batchsents(self, queries, subset=None, start=None, end=None,
			maxresults=100, brackets=False):
		"""Variant of sents() to run a batch of queries."""
		result = OrderedDict((name, [])
				for name in subset or self.files)
		for query in queries:
			for value in self.sents(
					query, subset, start, end, maxresults, brackets):
				result[value[0]].append(value[1:])
		for filename, values in result.items():
			yield filename, values

	def extract(self, filename, indices,
			nofunc=False, nomorph=False, sents=False):
		"""Extract a range of trees / sentences.

		:param filename: one of the filenames in ``self.files``
		:param indices: iterable of indices of sentences to extract
			(1-based, excluding empty lines)
		:param sents: if True, return sentences instead of trees.
			Sentences are strings with space-separated tokens.
		:param nofunc, nomorph: same as for ``trees()`` method.
		:returns: a list of Tree objects or sentences."""

	def _submit(self, func, *args, **kwargs):
		"""Submit a job to the thread/process pool."""
		if self.numproc == 1:
			return NoFuture(func, *args, **kwargs)
		return self.pool.submit(func, *args, **kwargs)

	def _map(self, func, *args, **kwargs):
		"""Map with thread/process pool.

		args is a sequence of iterables to map over; the same kwargs are passed
		for each iteration."""
		if self.numproc == 1:
			return (func(*xargs, **kwargs) for xargs in zip(*args))
		fs = [self.pool.submit(func, *xargs, **kwargs) for xargs in zip(*args)]

		def result_iterator():
			"""Yield results one by one."""
			try:
				for future in fs:
					yield future.result()
			finally:
				for future in fs:
					future.cancel()

		return result_iterator()

	def _as_completed(self, jobs):
		"""Return jobs as they are completed."""
		if self.numproc == 1:
			return jobs
		return concurrent.futures.as_completed(jobs)


class TgrepSearcher(CorpusSearcher):
	"""Search a corpus with tgrep2."""
	def __init__(self, files, macros=None, numproc=None):
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

		super(TgrepSearcher, self).__init__(files, macros, numproc)
		self.files = {convert(filename): None for filename in self.files}

	def counts(self, query, subset=None, start=None, end=None, indices=False,
			breakdown=False):
		if breakdown and indices:
			raise NotImplementedError
		subset = subset or self.files
		result = OrderedDict()
		jobs = {}
		# %s the sentence number
		fmt = r'%s\n:::\n'
		for filename in subset:
			try:
				result[filename] = self.cache[
						'counts', query, filename, start, end, indices,
						breakdown]
			except KeyError:
				if indices:
					jobs[self._submit(lambda x: [n for n, _
							in self._query(query, x, fmt, start, end, None)],
							filename)] = filename
				elif breakdown:
					jobs[self._submit(lambda x: Counter(match for _, match in
							self._query(query, x, r'%s\n%m:::\n', start, end,
							None)), filename)] = filename
				else:
					jobs[self._submit(lambda x: sum(1 for _
						in self._query(query, x, fmt, start, end, None)),
						filename)] = filename
		for future in self._as_completed(jobs):
			filename = jobs[future]
			self.cache['counts', query, filename, start, end, indices,
					breakdown] = result[filename] = future.result()
		return result

	def trees(self, query, subset=None, start=None, end=None, maxresults=10,
			nofunc=False, nomorph=False):
		subset = subset or self.files
		# %s the sentence number
		# %w complete tree in bracket notation
		# %m all marked nodes, or the head node if none are marked
		fmt = r'%s\n%w\n%m:::\n'
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['trees', query, filename,
						start, end, nofunc, nomorph]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(lambda x: list(self._query(
						query, x, fmt, start, end, maxresults)),
						filename)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for sentno, line in future.result():
				lines = line.splitlines()
				treestr, matches = lines[0], lines[1:]
				treestr = filterlabels(treestr, nofunc, nomorph)
				treestr = treestr.replace(" )", " -NONE-)")
				for match in matches:
					if match.startswith('('):
						treestr = treestr.replace(match, '%s_HIGH %s' % tuple(
								match.split(None, 1)), 1)
					else:
						match = ' %s)' % match
						treestr = treestr.replace(match, '_HIGH%s' % match)

				tree, sent = brackettree(treestr)
				tree = mergediscnodes(tree)
				high = list(tree.subtrees(lambda n: n.label.endswith("_HIGH")))
				tmp = {}
				for marked in high:
					marked.label = marked.label.rsplit("_", 1)[0]
					for node in marked.subtrees():
						tmp[id(node)] = node
					tmp.update((id(a), a) for a in marked.leaves())
				x.append((filename, sentno, tree, sent, list(tmp.values())))
			self.cache['trees', query, filename, start, end,
					nofunc, nomorph] = x, maxresults
			result.extend(x)
		return result

	def sents(self, query, subset=None, start=None, end=None, maxresults=100,
			brackets=False):
		subset = subset or self.files
		# %s the sentence number
		# %w complete tree in bracket notation
		# %m all marked nodes, or the head node if none are marked
		fmt = r'%s\n%w\n%m:::\n'
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['sents', query, filename,
						start, end, brackets]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(lambda x: list(self._query(
						query, x, fmt, start, end, maxresults)),
						filename)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for sentno, line in future.result():
				lines = line.splitlines()
				sent, matches = lines[0], lines[1:]
				if brackets:
					match1 = matches[0]
					match2 = ''
				else:
					tmp = set()
					for match in matches:
						idx = sent.index(match if match.startswith('(')
								else ' %s)' % match)
						prelen = len(' '.join(ptbunescape(token) for token
								in GETLEAVES.findall(sent[:idx])))
						match = (' '.join(ptbunescape(token) for token
								in GETLEAVES.findall(match))
								if '(' in match else ptbunescape(match))
						tmp.update(range(prelen, prelen + len(match) + 1))
					sent = ' '.join(ptbunescape(token)
							for token in GETLEAVES.findall(sent))
					match1 = tmp
					match2 = set()
				x.append((filename, sentno, sent, match1, match2))
			self.cache['sents', query, filename,
					start, end, brackets] = x, maxresults
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
		out, err = proc.communicate((''.join(
				'%d:1\n' % n for n in indices if n > 0)).encode('utf8'))
		proc.stdout.close()
		proc.stderr.close()
		if proc.returncode != 0:
			raise ValueError(err.decode('utf8'))
		result = out.decode('utf8').splitlines()
		if sents:
			return result
		return [(mergediscnodes(tree), sent)
				for tree, sent
				in (brackettree(filterlabels(
					treestr, nofunc, nomorph)) for treestr in result)]

	@workerfunc
	def _query(self, query, filename, fmt, start=None, end=None,
			maxresults=None):
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
		linere = re.compile(r'([0-9]+)\n(.*?):::\n', flags=re.DOTALL)
		if start or end or maxresults:
			start = start or 1
			results = []
			for n, match in enumerate(linere.finditer(
					proc.stdout.read().decode('utf8'))):
				m, a = int(match.group(1)), match.group(2)
				if m < start:
					continue
				elif (end and m > end) or (maxresults and n >= maxresults):
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
					in linere.finditer(out))
		if proc.returncode != 0:
			raise ValueError(err)
		return results


class DactSearcher(CorpusSearcher):
	"""Search a dact corpus with xpath."""
	def __init__(self, files, macros=None, numproc=None):
		super(DactSearcher, self).__init__(files, macros, numproc)
		if not ALPINOCORPUSLIB:
			raise ImportError('Could not import `alpinocorpus` module.')
		for filename in self.files:
			self.files[filename] = alpinocorpus.CorpusReader(filename)
		if macros is not None:
			try:
				self.macros = alpinocorpus.Macros(macros)
			except NameError:
				raise ValueError('macros not supported')

	def counts(self, query, subset=None, start=None, end=None, indices=False,
			breakdown=False):
		if breakdown and indices:
			raise NotImplementedError
		subset = subset or self.files
		result = OrderedDict()
		jobs = {}
		for filename in subset:
			try:
				result[filename] = self.cache[
						'counts', query, filename, start, end, indices]
			except KeyError:
				if indices:
					jobs[self._submit(lambda x: [n for n, _
							in self._query(query, x, start, end, None)],
							filename)] = filename
				elif breakdown:
					jobs[self._submit(lambda x: Counter(
						match.contents().decode('utf8')
						for _, match in self._query(
							query, x, start, end, None)), filename)] = filename
				else:
					jobs[self._submit(lambda x: sum(1 for _
						in self._query(query, x, start, end, None)),
						filename)] = filename
		for future in self._as_completed(jobs):
			filename = jobs[future]
			self.cache['counts', query, filename, start, end, indices
					] = result[filename] = future.result()
		return result

	def trees(self, query, subset=None, start=None, end=None, maxresults=10,
			nofunc=False, nomorph=False):
		subset = subset or self.files
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['trees', query, filename,
						start, end, nofunc, nomorph]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(lambda x: list(self._query(
						query, x, start, end, maxresults)),
						filename)] = filename
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
			self.cache['trees', query, filename, start, end,
					nofunc, nomorph] = x, maxresults
			result.extend(x)
		return result

	def sents(self, query, subset=None, start=None, end=None,
			maxresults=100, brackets=False):
		subset = subset or self.files
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['sents', query, filename,
						start, end, brackets]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(lambda x: list(self._query(
						query, x, start, end, maxresults)),
						filename)] = filename
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
					match = charindices(treestr.split(), match)
				x.append((filename, sentno, treestr, match, set()))
			self.cache['sents', query, filename,
					start, end, brackets] = x, maxresults
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

	@workerfunc
	def _query(self, query, filename, start=None, end=None,
			maxresults=None):
		"""Run a query on a single file."""
		if self.macros is not None:
			query = self.macros.expand(query)
		results = ((n, entry) for n, entry
				in ((entry.name(), entry)
					for entry in self.files[filename].xpath(query))
				if (start is None or start <= n)
				and (end is None or n <= end))
		return islice(results, maxresults)


class FragmentSearcher(CorpusSearcher):
	"""Search for fragments in a bracket treebank.

	Format of treebanks and queries can be bracket, discbracket, or
	export (autodetected).
	Each query consists of one or more tree fragments, and the results
	will be merged together, except with batchcounts(), which returns
	the results for each fragment separately.

	Example queries::
		(S (NP (DT The) (NN )) (VP ))
		(NP (DT 0=The) (NN 1=queen))

	:param inmemory: if True, keep all corpora in memory; otherwise, use
		pickle to load them from disk with each query.
	"""
	# NB: pickling arrays is efficient in Python 3
	def __init__(self, files, macros=None, numproc=None, inmemory=True):
		global FRAG_FILES, FRAG_MACROS, VOCAB
		super(FragmentSearcher, self).__init__(files, macros, numproc)
		path = os.path.dirname(next(iter(files)))
		newvocab = True
		self.disc = False
		vocabpath = os.path.join(path, 'treesearchvocab.pkl')
		if os.path.exists(vocabpath):
			mtime = os.stat(vocabpath).st_mtime
			if all(mtime > os.stat(a).st_mtime for a in files):
				try:
					self.vocab = pickle.load(open(vocabpath, 'rb'))
				except ValueError:  # e.g., unsupported pickle protocol
					pass
				else:
					newvocab = False
		if newvocab:
			self.vocab = Vocabulary()
		for filename in self.files:
			self.disc = self.disc or not filename.endswith('.mrg')
			if (newvocab or not os.path.exists('%s.pkl.gz' % filename)
					or os.stat('%s.pkl.gz' % filename).st_mtime
					< os.stat(filename).st_mtime):
				# get format from extension
				ext = {'export': 'export',
						'mrg': 'bracket',
						'dbr': 'discbracket'}
				fmt = ext[filename.split('.', 1)[-1]]
				corpus = _fragments.readtreebank(
						filename, self.vocab, fmt=fmt)
				corpus.indextrees(self.vocab)
				with gzip.open('%s.pkl.gz' % filename, 'w', compresslevel=1
						) as out:
					out.write(pickle.dumps(corpus, protocol=-1))
				if inmemory:
					self.files[filename] = corpus
				newvocab = True
			elif inmemory:
				corpus = pickle.loads(gzip.open(
						'%s.pkl.gz' % filename, 'rb').read())
				self.files[filename] = corpus
		if newvocab:
			with open(vocabpath, 'wb') as out:
				pickle.dump(self.vocab, out, protocol=-1)
		self.macros = None
		if macros:
			with openread(macros) as tmp:
				self.macros = dict(line.strip().split('=', 1) for line in tmp)
		if VOCAB is not None:
			raise ValueError('only one instance possible.')
		VOCAB = self.vocab
		FRAG_FILES = self.files
		FRAG_MACROS = self.macros
		self.pool = concurrent.futures.ProcessPoolExecutor(self.numproc)

	def __del__(self):
		global FRAG_FILES, FRAG_MACROS, VOCAB
		VOCAB = None
		FRAG_FILES = None
		FRAG_MACROS = None

	def counts(self, query, subset=None, start=None, end=None, indices=False,
			breakdown=False):
		if breakdown:
			if indices:
				raise NotImplementedError
			queries = query.splitlines()
			result = self.batchcounts(queries, subset, start, end)
			return OrderedDict(
					(filename, OrderedDict(
						(query, a) for query, a in zip(queries, values)))
					for filename, values in result)
		subset = subset or self.files
		result = OrderedDict()
		jobs = {}
		for filename in subset:
			try:
				tmp = self.cache[
						'counts', query, filename, start, end, indices]
				if indices:
					result[filename] = [b for a in tmp for b in a]
				else:
					result[filename] = sum(tmp)
			except KeyError:
				jobs[self._submit(_frag_query, query, filename,
						start, end, None, indices=indices, trees=False,
						disc=self.disc)] = filename
		for future in self._as_completed(jobs):
			filename = jobs[future]
			tmp = future.result()
			self.cache['counts', query, filename, start, end, indices] = tmp
			if indices:
				result[filename] = [b for a in tmp for b in a]
			else:
				result[filename] = sum(tmp)
		return result

	def batchcounts(self, queries, subset=None, start=None, end=None):
		subset = subset or self.files
		jobs = {}
		queries, bitsets, maxnodes = _frag_parse_query(queries, disc=self.disc)
		for filename in subset:
			# NB: not using cache.
			jobs[self._submit(_frag_run_query, queries, bitsets, maxnodes,
					filename, start, end, None, indices=False, trees=False,
					)] = filename
		for future in self._as_completed(jobs):
			filename = jobs[future]
			yield filename, future.result()

	def trees(self, query, subset=None, start=None, end=None, maxresults=10,
			nofunc=False, nomorph=False):
		subset = subset or self.files
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['trees', query, filename,
						start, end, nofunc, nomorph]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(_frag_query, query, filename,
						start, end, maxresults, indices=True, trees=True,
						disc=self.disc)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for matches in future.result():
				for sentno, treestr, match in matches:
					treestr = filterlabels(treestr, nofunc, nomorph)
					# NB: this highlights the whole subtree, of which
					# frag may be a subgraph.
					treestr = treestr.replace(
							match,
							'%s_HIGH %s' % tuple(match.split(None, 1)),
							1)
					tree, sent = brackettree(treestr)
					tree = mergediscnodes(tree)
					high = list(tree.subtrees(
							lambda n: n.label.endswith("_HIGH")))
					if high:
						high = high.pop()
						high.label = high.label.rsplit("_", 1)[0]
						high = list(high.subtrees()) + high.leaves()
					x.append((filename, sentno, tree, sent, high))
			self.cache['trees', query, filename, start, end,
					nofunc, nomorph] = x, maxresults
			result.extend(x)
		return result

	def sents(self, query, subset=None, start=None, end=None,
			maxresults=100, brackets=False):
		subset = subset or self.files
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['sents', query, filename,
						start, end, brackets]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(_frag_query, query, filename,
						start, end, maxresults, indices=True, trees=True,
						disc=self.disc)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for frag, matches in zip(query.splitlines(), future.result()):
				for sentno, treestr, match in matches:
					if brackets:
						sent = treestr
						if not self.disc:
							sent = LEAFINDICES.sub(' ', sent)
							match = LEAFINDICES.sub(' ', match)
						match1, match2 = match, ''
					else:
						_, xsent = brackettree(treestr)
						sent = ' '.join(xsent)
						fragwords = set(GETLEAVES.findall(frag))
						match1 = {int(a) for a, b
								in LEAFINDICESWORDS.findall(match)
								if b in fragwords}
						match2 = {int(a) for a, _
								in LEAFINDICESWORDS.findall(match)}
						match1, match2 = charindices(xsent, match1, match2)
					x.append((filename, sentno, sent, match1, match2))
			self.cache['sents', query, filename,
					start, end, brackets] = x, maxresults
			result.extend(x)
		return result

	def extract(self, filename, indices,
			nofunc=False, nomorph=False, sents=False):
		if self.files[filename] is not None:
			corpus = self.files[filename]
		else:
			corpus = pickle.loads(gzip.open(
					'%s.pkl.gz' % filename, 'rb').read())
		if sents:
			return [' '.join(ptbunescape(token)
					for token in corpus.extractsent(n - 1, self.vocab))
					for n in indices]
		result = []
		for n in indices:
			treestr = corpus.extract(n - 1, self.vocab)
			tree, sent = brackettree(
					filterlabels(treestr, nofunc, nomorph))
			result.append((mergediscnodes(tree), sent))
		return result

	def getinfo(self, filename):
		"""Return named tuple with members len, numnodes, and numwords."""
		if self.files[filename] is not None:
			corpus = self.files[filename]
		else:
			corpus = pickle.loads(gzip.open(
					'%s.pkl.gz' % filename, 'rb').read())
		return CorpusInfo(len=corpus.len, numwords=corpus.numwords,
				numnodes=corpus.numnodes, maxnodes=corpus.maxnodes)


@workerfunc
def _frag_query(query, filename, start=None, end=None, maxresults=None,
		indices=True, trees=False, disc=False):
	"""Run a fragment query on a single file."""
	queries, bitsets, maxnodes = _frag_parse_query(query, disc=disc)
	return _frag_run_query(queries, bitsets, maxnodes, filename,
			start=start, end=end, maxresults=maxresults, indices=indices,
			trees=trees)


def _frag_parse_query(query, disc=False):
	"""Prepare fragment query."""
	if FRAG_MACROS is not None:
		query = query.format(**FRAG_MACROS)
	if isinstance(query, list):
		qitems = (brackettree(a) for a in query)
	else:
		qitems = treebank.incrementaltreereader(
				io.StringIO(query), strict=True, robust=False)
	qtrees, qsents = [], []
	for item in qitems:
		# rightmostunary necessary to handle discontinuous substitution sites
		qtrees.append(binarize(
				handledisc(item[0]) if disc else item[0], dot=True))
		qsents.append(item[1])
	if not qtrees:
		raise ValueError('no valid fragments found.')
	queries = _fragments.getctrees(
			list(qtrees), list(qsents), vocab=VOCAB)
	maxnodes = queries['trees1'].maxnodes
	_fragmentkeys, bitsets = _fragments.completebitsets(
			queries['trees1'], VOCAB, maxnodes, disc=disc)
	return queries, bitsets, maxnodes


def _frag_run_query(queries, bitsets, maxnodes, filename, start=None, end=None,
		maxresults=None, indices=True, trees=False):
	"""Run a prepared fragment query on a single file."""
	if FRAG_FILES[filename] is not None:
		corpus = FRAG_FILES[filename]
	else:
		corpus = pickle.loads(gzip.open(
				'%s.pkl.gz' % filename, 'rb').read())
	if start is not None:
		start -= 1
	results = _fragments.exactcountsslice(queries['trees1'], corpus,
			bitsets, indices=indices + trees if indices else 0,
			maxnodes=maxnodes, start=start, end=end,
			maxresults=maxresults)
	if indices and trees:
		results = [[(n + 1,
					corpus.extract(n, VOCAB, disc=True),
					corpus.extract(n, VOCAB, disc=True, node=m))
					for n, m in zip(b, c)]
				for b, c in results]
	elif indices:
		results = [[n + 1 for n in b] for b in results]
	return results


class RegexSearcher(CorpusSearcher):
	"""Search a plain text file in UTF-8 with regular expressions.

	Assumes that non-empty lines correspond to sentences; empty lines
	do not count towards line numbers (e.g., when used as paragraph breaks).

	:param macros: a file containing lines of the form ``'name=regex'``;
		an occurrence of ``'{name}'`` will be suitably replaced when it
		appears in a query.
	:param ignorecase: ignore case in all queries."""
	def __init__(self, files, macros=None, numproc=None, ignorecase=False):
		global REGEX_LINEINDEX, REGEX_MACROS
		super(RegexSearcher, self).__init__(files, macros, numproc)
		self.macros = None
		self.lineindex = {}
		self.wordcount = {}
		self.flags = re.MULTILINE
		if ignorecase:
			self.flags |= re.IGNORECASE
		if macros:
			with openread(macros) as tmp:
				self.macros = dict(line.strip().split('=', 1) for line in tmp)
		path = os.path.dirname(next(iter(files)))
		lineidxpath = os.path.join(path, 'treesearchlineidx.pkl')
		if os.path.exists(lineidxpath):
			mtime = os.stat(lineidxpath).st_mtime
			if all(mtime > os.stat(a).st_mtime for a in files):
				try:
					(self.lineindex, self.wordcount) = pickle.load(
							open(lineidxpath, 'rb'))
				except ValueError:  # e.g., unsupported pickle protocol
					pass
		newindex = False
		for name in set(files) - set(self.lineindex):
			self.lineindex[name], self.wordcount[name] = _indexfile(name)
			newindex = True
		if newindex:
			with open(lineidxpath, 'wb') as out:
				pickle.dump((self.lineindex, self.wordcount), out, protocol=-1)
		if REGEX_LINEINDEX is not None:
			raise ValueError('only one instance possible.')  # FIXME
		REGEX_LINEINDEX = self.lineindex
		REGEX_MACROS = self.macros
		self.pool = concurrent.futures.ProcessPoolExecutor(self.numproc)

	def __del__(self):
		global REGEX_LINEINDEX, REGEX_MACROS
		REGEX_LINEINDEX = None
		REGEX_MACROS = None

	def counts(self, query, subset=None, start=None, end=None, indices=False,
			breakdown=False):
		if breakdown and indices:
			raise NotImplementedError
		subset = subset or self.files
		result = OrderedDict()
		jobs = {}
		pattern = _regex_parse_query(query, self.flags)
		for filename in subset:
			try:
				result[filename] = self.cache[
						'counts', query, filename, start, end, indices, False,
						breakdown]
			except KeyError:
				jobs[self._submit(_regex_run_query, pattern, filename,
						start, end, None, indices, False, breakdown)
						] = filename
		for future in self._as_completed(jobs):
			filename = jobs[future]
			self.cache['counts', query, filename, start, end, indices, False,
					breakdown] = result[filename] = future.result()
		return result

	def sents(self, query, subset=None, start=None, end=None, maxresults=100,
			brackets=False):
		if brackets:
			raise ValueError('not applicable with plain text corpus.')
		subset = subset or self.files
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['sents', query, filename,
						start, end, True, True]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(_regex_query,
						query, filename, self.flags, start, end, maxresults,
						True, True)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for sentno, sent, start, end in future.result():
				highlight = range(start, end)
				x.append((filename, sentno, sent.rstrip(), highlight, ()))
			self.cache['sents', query, filename, start, end, True, True
					] = x, maxresults
			result.extend(x)
		return result

	def trees(self, query, subset=None, start=None, end=None, maxresults=10,
			nofunc=False, nomorph=False):
		raise ValueError('not applicable with plain text corpus.')

	def batchcounts(self, queries, subset=None, start=None, end=None):
		patterns = [_regex_parse_query(query, self.flags) for query in queries]
		result = OrderedDict((name, [])
				for name in subset or self.files)
		chunksize = max(int(len(patterns) / (self.numproc * 4)), 1)
		chunkedqueries = [queries[n:n + chunksize]
				for n in range(0, len(patterns), chunksize)]
		patterns = [[_regex_parse_query(query, self.flags) for query in a]
				for a in chunkedqueries]
		for filename in subset or self.files:
			result = array.array(
					b'I' if PY2 else 'I')
			for tmp in self._map(_regex_run_batch, patterns,
					filename=filename, start=start, end=end):
				result.extend(tmp)
			yield filename, result

	def batchsents(self, queries, subset=None, start=None, end=None,
			maxresults=100, brackets=False):
		"""Variant of sents() to run a batch of queries."""
		if brackets:
			raise ValueError('not applicable with plain text corpus.')
		patterns = [_regex_parse_query(query, self.flags) for query in queries]
		result = OrderedDict((name, [])
				for name in subset or self.files)
		chunksize = max(int(len(patterns) / (self.numproc * 4)), 1)
		chunkedqueries = [queries[n:n + chunksize]
				for n in range(0, len(patterns), chunksize)]
		patterns = [[_regex_parse_query(query, self.flags) for query in a]
				for a in chunkedqueries]
		for filename in subset or self.files:
			result = []
			for tmp in self._map(_regex_run_batch, patterns,
					filename=filename, start=start, end=end,
					maxresults=maxresults, sents=True):
				result.extend(tmp)
			yield filename, result

	def extract(self, filename, indices,
			nofunc=False, nomorph=False, sents=True):
		if not sents:
			raise ValueError('not applicable with plain text corpus.')
		result = []
		end = next(reversed(self.lineindex[filename]))
		with open(filename, 'r+b') as tmp:
			data = mmap.mmap(tmp.fileno(), 0, access=mmap.ACCESS_READ)
			for n in indices:
				a, b = 0, end
				if 1 <= n < len(self.lineindex[filename]):
					a = self.lineindex[filename].select(n - 1)
				if 1 <= n < len(self.lineindex[filename]):
					b = self.lineindex[filename].select(n)
				result.append(data[a:b].decode('utf8'))
			data.close()
		return result


@workerfunc
def _regex_query(query, filename, flags, start=None, end=None, maxresults=None,
		indices=True, sents=False, breakdown=False):
	"""Run a query on a single file."""
	pattern = _regex_parse_query(query, flags)
	return _regex_run_query(pattern, filename, start=start, end=end,
			maxresults=maxresults, indices=indices, sents=sents,
			breakdown=breakdown)


def _regex_parse_query(query, flags):
	"""Prepare regex query."""
	if REGEX_MACROS is not None:
		query = query.format(**REGEX_MACROS)
	pattern = None
	# could use .find() / .count() with plain queries
	try:
		if RE2LIB:
			try:
				pattern = re2.compile(  # pylint: disable=no-member
						query.encode('utf8'), flags=flags | re.UNICODE,
						max_mem=8 << 26)  # 500 MB
			except ValueError:
				pass
		if pattern is None:
			pattern = re.compile(query.encode('utf8'), flags=flags)
	except re.error:
		print('problem compiling query:', query)
		raise
	return pattern


def _regex_run_query(pattern, filename, start=None, end=None, maxresults=None,
		indices=False, sents=False, breakdown=False):
	"""Run a prepared query on a single file."""
	lineindex = REGEX_LINEINDEX[filename]
	if indices and sents:
		result = []
	elif indices:
		result = array.array(b'I' if PY2 else 'I')
	elif breakdown:
		result = Counter()
	else:
		result = 0
	# TODO: is it advantageous to keep mmap'ed files open?
	with open(filename, 'r+b') as tmp:
		data = mmap.mmap(tmp.fileno(), 0, access=mmap.ACCESS_READ)
		startidx = lineindex.select(start - 1 if start else 0)
		endidx = (lineindex.select(end) if end is not None
				and end < len(lineindex) else len(data))
		if indices or sents:
			for match in islice(
					pattern.finditer(data, startidx, endidx),
					maxresults):
				mstart = match.start()
				mend = match.end()
				lineno = lineindex.rank(mstart)
				offset, nextoffset = 0, len(data)
				if lineno > 0:
					offset = lineindex.select(lineno - 1)
				if lineno <= len(lineindex):
					nextoffset = lineindex.select(lineno)
				if sents:
					sent = data[offset:nextoffset].decode('utf8')
					mstart = len(data[offset:mstart].decode('utf8'))
					mend = len(data[offset:mend].decode('utf8'))
					# (lineno, sent, startspan, endspan)
					result.append((lineno, sent, mstart, mend))
				else:
					result.append(lineno)
		else:
			if breakdown:
				matches = pattern.findall(data, startidx, endidx)[:maxresults]
				result.update(a.decode('utf8') for a in matches)
			else:
				try:
					result = pattern.count(data, startidx, endidx)
				except AttributeError:
					result = len(pattern.findall(data, startidx, endidx))
				result = max(result, maxresults or 0)
		data.close()
	return result


def _regex_run_batch(patterns, filename, start=None, end=None, maxresults=None,
		sents=False):
	"""Run a batch of queries on a single file."""
	lineindex = REGEX_LINEINDEX[filename]
	with open(filename, 'r+b') as tmp:
		data = mmap.mmap(tmp.fileno(), 0, access=mmap.ACCESS_READ)
		startidx = lineindex.select(start - 1 if start else 0)
		endidx = (lineindex.select(end) if end is not None
				and end < len(lineindex) else len(data))
		if sents:
			result = []
			for pattern in patterns:
				for match in islice(
						pattern.finditer(data, startidx, endidx),
						maxresults):
					mstart = match.start()
					mend = match.end()
					lineno = lineindex.rank(mstart)
					offset, nextoffset = 0, len(data)
					if lineno > 0:
						offset = lineindex.select(lineno - 1)
					if lineno <= len(lineindex):
						nextoffset = lineindex.select(lineno)
					if sents:
						sent = data[offset:nextoffset].decode('utf8')
						mstart = len(data[offset:mstart].decode('utf8'))
						mend = len(data[offset:mend].decode('utf8'))
						#  sentno, sent, high1, high2
						result.append((lineno, sent, range(mstart, mend), ()))
		else:
			result = array.array(b'I' if PY2 else 'I')
			for pattern in patterns:
				try:
					result.append(pattern.count(data, startidx, endidx))
				except AttributeError:
					result.append(len(pattern.findall(data, startidx, endidx)))
		data.close()
	return result


def _indexfile(filename):
	"""Create bitmap with locations of non-empty lines."""
	result = array.array(b'I' if PY2 else 'I')
	offset = 0
	wordcount = 0
	with open(filename, 'rb') as tmp:
		for line in tmp:
			if not line.isspace():
				result.append(offset)
			offset += len(line)
			wordcount += line.count(b' ') + 1
	result.append(offset)
	return RoaringBitmap(result), wordcount


class NoFuture(object):
	"""A non-asynchronous version of concurrent.futures.Future."""
	def __init__(self, func, *args, **kwargs):
		self._result = func(*args, **kwargs)

	def result(self, timeout=None):  # pylint: disable=unused-argument
		"""Return the precomputed result."""
		return self._result


class FIFOOrederedDict(collections.OrderedDict):
	"""FIFO cache with maximum number of elements based on OrderedDict."""
	def __init__(self, limit):
		super(FIFOOrederedDict, self).__init__()
		self.limit = limit

	def __setitem__(self, key, value):  # pylint: disable=arguments-differ
		if self.limit == 0:
			return
		elif key in self:
			self.pop(key)
		elif len(self) >= self.limit:
			self.pop(next(iter(self)))
		super(FIFOOrederedDict, self).__setitem__(key, value)


def filterlabels(line, nofunc, nomorph):
	"""Remove morphological and/or grammatical function labels from tree(s)."""
	if nofunc:
		line = FUNC_TAGS.sub('', line)
	if nomorph:
		line = MORPH_TAGS.sub(lambda g: g.group(1) + g.group(2), line)
	return line


def charindices(sent, indices, indices2=None):
	"""Project token indices to character indices.

	>>> sorted(charindices(['The', 'cat', 'is', 'on', 'the', 'mat'], {0, 2, 4}))
	[0, 1, 2, 3, 8, 9, 10, 14, 15, 16, 17]"""
	cur = 0
	ind = {}
	for n, a in enumerate(sent):
		ind[n] = range(cur, cur + len(a)
				+ (n != len(sent) - 1))
		cur += len(a) + 1
	result = {a for n in indices for a in ind[n]}
	if indices2 is not None:
		return result, {a for n in indices2 for a in ind[n]}
	return result


def cpu_count():
	"""Return number of CPUs or 1."""
	try:
		return multiprocessing.cpu_count()
	except NotImplementedError:
		return 1


def applyhighlight(sent, high1, high2):
	"""Highlight character indices high1 & high2 in sent with ANSI colors."""
	cur = None
	start = 0
	out = []
	for n, _ in enumerate(sent):
		if n in high1:
			if cur != 'red':
				cur = 'red'
				out.append(sent[start:n]
						+ '\x1b[%d;1m' % ANSICOLOR[cur])
				start = n
		elif n in high2:
			if cur != 'blue':
				cur = 'blue'
				out.append(sent[start:n]
						+ '\x1b[%d;1m' % ANSICOLOR[cur])
				start = n
		else:
			if cur is not None:
				out.append(sent[start:n])
				start = n
				cur = None
				out.append('\x1b[0m')
	out.append(sent[start:])
	if cur is not None:
		out.append('\x1b[0m')
	return ''.join(out)


def writecounts(results, flat=False, columns=None):
	"""Write a dictionary of dictionaries to stdout as CSV or in a flat format.

	:param data: a dictionary of dictionaries.
	:param flat: if True, do not write CSV but a simple a flat format,
		skipping zero counts.
	:param columns: if given, data is an iterable of (filename, list/array)
		tuples, with columns being a list specifying the column names."""
	if flat:
		if columns is None:
			for filename in results:
				print(filename)
				for match, value in results[filename].items():
					if value:
						print('%s\t%s' % (match, value))
				print()
		else:
			for filename, values in results:
				print(filename)
				for query, value in zip(columns, values):
					if value:
						print('%s\t%s' % (query, value))
				print()
		return
	writer = csv.writer(sys.stdout)
	if columns is None:
		writer.writerow(['filename']
				+ list(next(iter(results.values())).keys()))
		writer.writerows(
				[filename] + list(results[filename].values())
				for filename in results)
	else:
		writer.writerow(['filename'] + list(columns))
		writer.writerows(
				[filename] + list(values)
				for filename, values in results)


def main():
	"""CLI."""
	global CACHESIZE
	CACHESIZE = 0
	from getopt import gnu_getopt, GetoptError
	shortoptions = 'e:m:M:stcbnofih'
	options = ('engine= macros= numproc= max-count= slice= '
			'trees sents brackets counts indices breakdown only-matching '
			'line-number file ignore-case csv help')
	try:
		opts, args = gnu_getopt(sys.argv[2:], shortoptions, options.split())
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
		with openread(query) as tmp:
			query = tmp.read()
	macros = opts.get('--macros', opts.get('-M'))
	engine = opts.get('--engine', opts.get('-e', 'frag'))
	maxresults = int(opts.get('--max-count', opts.get('-m', 100))) or None
	numproc = int(opts.get('--numproc', 0)) or None
	if len(corpora) == 1:
		numproc = 1
	start, end = opts.get('--slice', ':').split(':')
	start, end = (int(start) if start else None), (int(end) if end else None)
	ignorecase = '--ignore-case' in opts or '-i' in opts
	if ignorecase and engine != 'regex':
		raise ValueError('--ignore-case is only supported with --engine=regex')
	if engine == 'tgrep2':
		searcher = TgrepSearcher(corpora, macros=macros, numproc=numproc)
	elif engine == 'xpath':
		searcher = DactSearcher(corpora, macros=macros, numproc=numproc)
	elif engine == 'regex':
		searcher = RegexSearcher(corpora, macros=macros, numproc=numproc,
				ignorecase=ignorecase)
	elif engine == 'frag':
		searcher = FragmentSearcher(
				corpora, macros=macros, numproc=numproc, inmemory=False)
	else:
		raise ValueError('incorrect --engine value: %r' % engine)
	if '--counts' in opts or '-c' in opts or '--indices' in opts:
		indices = '--indices' in opts
		queries = query.splitlines()
		if '--breakdown' in opts:
			if indices:
				raise ValueError('--indices only supported with single query.')
			results = searcher.counts(
					query, start=start, end=end, breakdown=True)
			writecounts(results, flat='--csv' not in opts)
		elif len(queries) > 1:
			if indices:
				raise ValueError('--indices only supported with single query.')
			results = searcher.batchcounts(
					queries, start=start, end=end)
			writecounts(results, flat='--csv' not in opts, columns=queries)
		else:
			for filename, cnt in searcher.counts(
					query, start=start, end=end, indices=indices).items():
				if len(corpora) > 1:
					print('\x1b[%dm%s\x1b[0m:' % (
						ANSICOLOR['magenta'], filename), end='')
				print(cnt)
	elif '--trees' in opts or '-t' in opts:
		results = searcher.trees(
				query, start=start, end=end, maxresults=maxresults)
		if '--breakdown' in opts:
			breakdown = Counter(DiscTree(
				max(high, key=lambda x: len(x.leaves())
					if isinstance(x, Tree) else 1).freeze(), sent)
				for _, _, _, sent, high in results if high)
			for match, cnt in breakdown.most_common():
				print('count: %5d\n%s\n\n' % (
						cnt, DrawTree(match, match.sent).text(
							unicodelines=True, ansi=True)))
			return
		for filename, sentno, tree, sent, high in results:
			if '--only-matching' in opts or '-o' in opts:
				tree = max(high, key=lambda x:
						len(x.leaves()) if isinstance(x, Tree) else 1)
			out = DrawTree(tree, sent, highlight=high).text(
					unicodelines=True, ansi=True)
			if len(corpora) > 1:
				print('\x1b[%dm%s\x1b[0m:' % (
						ANSICOLOR['magenta'], filename), end='')
			if '--line-number' in opts or '-n' in opts:
				print('\x1b[%dm%s\x1b[0m:'
						% (ANSICOLOR['green'], sentno), end='')
			print(out)
	else:  # sentences or brackets
		brackets = '--brackets' in opts or '-b' in opts
		queries = query.splitlines()
		breakdown = Counter()
		for filename, result in searcher.batchsents(
				queries, start=start, end=end, maxresults=maxresults,
				brackets=brackets):
			if '--breakdown' in opts:
				if brackets:
					breakdown.update(high for _, _, high, _ in result)
				else:
					breakdown.update(re.sub(
						' {2,}', ' ... ',
						''.join(char if n in high1 or n in high2 else ' '
							for n, char in enumerate(sent)))
						for _, sent, high1, high2 in result)
				continue
			for sentno, sent, high1, high2 in result:
				if brackets:
					if '--only-matching' in opts or '-o' in opts:
						out = high1
					else:
						out = sent.replace(high1, "\x1b[%d;1m%s\x1b[0m" % (
								ANSICOLOR['red'], high1))
				else:
					if '--only-matching' in opts or '-o' in opts:
						out = ''.join(char if n in (high2 or high1) else ''
								for n, char in enumerate(sent)).strip()
					else:
						out = applyhighlight(sent.strip(), high1, high2)
				if len(corpora) > 1:
					print('\x1b[%dm%s\x1b[0m:' % (
							ANSICOLOR['magenta'], filename), end='')
				if '--line-number' in opts or '-n' in opts:
					print('\x1b[%dm%s\x1b[0m:'
							% (ANSICOLOR['green'], sentno), end='')
				print(out)
		if '--breakdown' in opts:
			for match, cnt in breakdown.most_common():
				print('%5d %s' % (cnt, match))


__all__ = ['CorpusSearcher', 'TgrepSearcher', 'DactSearcher', 'RegexSearcher',
		'FragmentSearcher', 'NoFuture', 'FIFOOrederedDict', 'filterlabels',
		'cpu_count']
