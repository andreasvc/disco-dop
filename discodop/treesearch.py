"""Objects for searching through collections of trees."""

# Possible improvements:
# - cache raw results from _query() before conversion?
# - return/cache trees as strings?

import io
import os
import re
import csv
import sys
import mmap
import array
import tempfile
import subprocess
import multiprocessing
import concurrent.futures
from itertools import islice
from collections import Counter, OrderedDict, namedtuple
try:
	import re2
	RE2LIB = True
except ImportError:
	RE2LIB = False
from roaringbitmap import RoaringBitmap, MultiRoaringBitmap
from . import treebank, _fragments
from .tree import (Tree, DrawTree, DiscTree, brackettree, discbrackettree,
		ptbunescape)
from .treetransforms import binarize, mergediscnodes, handledisc
from .util import which, workerfunc, openread, readbytes, run, ANSICOLOR
from .containers import Vocabulary, FixedVocabulary, Ctrees

SHORTUSAGE = '''Search through treebanks with queries.
Usage: discodop treesearch [-e (tgrep2|frag|regex)] [-t|-s|-c] \
<query> <treebank1>...'''
CACHESIZE = 32767
GETLEAVES = re.compile(r' (?:[0-9]+=)?([^ ()]+)(?=[ )])')
LEAFINDICES = re.compile(r' ([0-9]+)=')
LEAFINDICESWORDS = re.compile(r' ([0-9]+)=([^ ()]+)\)')
ALPINOLEAVES = re.compile('<sentence>(.*)</sentence>')
MORPH_TAGS = re.compile(r'([/*\w]+)(?:\[[^ ]*\]\d?)?((?:-\w+)?(?:\*\d+)? )')
FUNC_TAGS = re.compile(r'-\w+')

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
		self.pool = concurrent.futures.ThreadPoolExecutor(self.numproc)
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
			nofunc=False, nomorph=False, detectdisc=True, mergedisc=True):
		"""Run query and return list of matching trees.

		:param start, end: the interval of sentences to query in each corpus;
			by default, all sentences are queried. 1-based, inclusive.
		:param maxresults: the maximum number of matches to return.
		:param nofunc, nomorph: whether to remove / add function tags and
			morphological features from trees.
		:param detectdisc: whether to detect discbracket format.
		:param mergedisc: whether to merge split discontinuities.
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
		yield from result.items()

	def batchsents(self, queries, subset=None, start=None, end=None,
			maxresults=100, brackets=False):
		"""Variant of sents() to run a batch of queries.

		:yields: tuples of the form ``(corpus1, matches)``
			where ``matches`` is in the same format returned by ``sents()``
			excluding the filename, with the results of different patterns
			merged together.
		"""
		result = OrderedDict((name, [])
				for name in subset or self.files)
		for query in queries:
			for value in self.sents(
					query, subset, start, end, maxresults, brackets):
				result[value[0]].append(value[1:])
		yield from result.items()

	def extract(self, filename, indices, nofunc=False, nomorph=False,
			detectdisc=True, mergedisc=True, sents=False):
		"""Extract a range of trees / sentences.

		:param filename: one of the filenames in ``self.files``
		:param indices: iterable of indices of sentences to extract
			(1-based, excluding empty lines)
		:param sents: if True, return sentences instead of trees.
			Sentences are strings with space-separated tokens.
		:returns: a list of Tree objects or sentences.

		For the other options, see the ``trees()`` method."""

	def getinfo(self, filename):
		"""Return named tuple with members len, numnodes, and numwords."""

	def __enter__(self):
		return self

	def __exit__(self, _type, _value, _traceback):
		self.close()

	def close(self):
		"""Close files and free memory."""
		pass

	def _submit(self, func, *args, **kwargs):
		"""Submit a job to the thread/process pool."""
		if self.numproc == 1:
			return NoFuture(func, *args, **kwargs)
		return self.pool.submit(func, *args, **kwargs)

	def _map(self, func, *args, **kwargs):
		"""Map with thread/process pool.

		``args`` is a sequence of iterables to map over;
		the same ``kwargs`` are passed for each iteration."""
		if self.numproc == 1:
			return (func(*xargs, **kwargs) for xargs in zip(*args))
		fs = [self.pool.submit(func, *xargs, **kwargs) for xargs in zip(*args)]

		def result_iterator():
			"""Yield results one by one."""
			try:
				for future in fs:
					yield future.result()
			finally:
				for future_ in fs:
					future_.cancel()

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
			"""Create tgrep2 indexed files (.t2c) if necessary."""
			if not os.path.exists(self._internalfilename(filename)):
				origfile = filename
				if filename.endswith('.gz') or filename.endswith('.zst'):
					with tempfile.NamedTemporaryFile(delete=False) as tmp:
						tmp.write(readbytes(filename))
						origfile = tmp.name
				try:
					args = [which('tgrep2'), '-p', origfile,
							self._internalfilename(filename)]
					returncode, _, stderr = run(args=args)
					if 'must use the -K flag' in stderr.decode('utf8'):
						returncode, _, stderr = run(
								args=[args[0], '-K'] + args[1:])
					if returncode != 0:
						raise ValueError('Error creating tgrep2 index of %r:\n'
								'%s' % (filename, stderr.decode('utf8')))
				finally:
					if filename != origfile:
						os.unlink(tmp.name)
			return filename

		super().__init__(files, macros, numproc)
		self._compressext = 'gz'  # the compression format to use for t2c files
		if which('zstd', exception=False):  # https://facebook.github.io/zstd/
			self._compressext = 'zst'
		elif which('lz4', exception=False):  # https://github.com/lz4/lz4/
			self._compressext = 'lz4'
		self.files = {convert(filename): None for filename in self.files}

	def counts(self, query, subset=None, start=None, end=None, indices=False,
			breakdown=False):
		if breakdown and indices:
			raise NotImplementedError
		subset = subset or self.files
		result = OrderedDict()
		jobs = {}
		# %s the sentence number
		# %h the tree matched by the head (first) node of the pattern
		fmt = r'%s\n:::\n'
		fmtbreakdown = r'%s\n%h:::\n',
		for filename in subset:
			result[filename] = self.cache.get((
					'counts', query, filename, start, end, indices,
					breakdown))
			if result[filename] is not None:
				pass
			elif indices:
				jobs[self._submit(
						lambda x: [n for n, _ in self._query(
							[query], x, fmt, start, end, None)],
						filename)] = filename
			elif breakdown:
				jobs[self._submit(
						lambda x: Counter(match for _, match in self._query(
							[query], x, fmtbreakdown, start, end, None)),
						filename)] = filename
			else:
				jobs[self._submit(
						lambda x: sum(1 for _ in self._query(
							[query], x, fmt, start, end, None)),
						filename)] = filename
		for future in self._as_completed(jobs):
			filename = jobs[future]
			self.cache['counts', query, filename, start, end, indices,
					breakdown] = result[filename] = future.result()
		return result

	def batchcounts(self, queries, subset=None, start=None, end=None):
		subset = subset or self.files
		jobs = {}
		# %s the sentence number
		# %p the number of the matching pattern
		fmt = r'%s\n%p:::\n'
		for filename in subset:
			# NB: not using cache
			jobs[self._submit(
					lambda x: [cnt for _, cnt in sorted(Counter(int(queryno)
						for _, queryno in self._query(
							queries, x, fmt, start, end, None)).items())],
					filename)] = filename
		for future in self._as_completed(jobs):
			filename = jobs[future]
			yield filename, future.result()

	def batchsents(self, queries, subset=None, start=None, end=None,
			maxresults=100, brackets=False):
		# FIXME: this is highly similar to sents(), but no caching
		subset = subset or self.files
		# %s the sentence number
		# %w complete tree in bracket notation
		# %h the tree matched by the head (first) node of the pattern
		# %yh %zh the terminal index of the first/last terminal in the match
		fmt = r'%s\n%w\n%h\n%yh\n%zh:::\n'
		jobs = {}
		for filename in subset:
			jobs[self._submit(
					lambda x: list(self._query(
						queries, x, fmt, start, end, maxresults)),
					filename)] = filename
		for future in self._as_completed(jobs):
			filename = jobs[future]
			result = []
			for sentno, line in future.result():
				sent, match, begin, end = line.splitlines()
				if brackets:
					match1 = match
					match2 = ''
				else:
					begin, end = int(begin) - 1, int(end)
					tokens = [ptbunescape(token)
							for token in GETLEAVES.findall(sent)]
					sent = ' '.join(tokens)
					prelen = len(' '.join(tokens[:begin]))
					match = ' '.join(tokens[begin:end])
					match1 = set(range(prelen, prelen + len(match) + 1))
					match2 = set()
				result.append((sentno, sent, match1, match2))
			yield filename, result

	def trees(self, query, subset=None, start=None, end=None, maxresults=10,
			nofunc=False, nomorph=False, detectdisc=True, mergedisc=False):
		subset = subset or self.files
		# %s the sentence number
		# %w complete tree in bracket notation
		# %h the tree matched by the head (first) node of the pattern
		# %nh the depth-first pre-order node number of the tree head
		fmt = r'%s\n%w\n%h\n%nh:::\n'
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['trees', query, filename,
						start, end, nofunc, nomorph, detectdisc, mergedisc]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(
						lambda x: list(self._query(
							[query], x, fmt, start, end, maxresults)),
						filename)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for sentno, line in future.result():
				treestr, _match, nodenum = line.splitlines()
				treestr = filterlabels(treestr, nofunc, nomorph)
				treestr = treestr.replace(" )", " -NONE-)")
				nodenum = int(nodenum)
				tree, sent = brackettree(treestr, detectdisc=detectdisc)
				if mergedisc:
					raise NotImplementedError
					# FIXME cannot do this, would change node numbers; how
					# could a match on a discontinuous component be recovered?
					# tree = mergediscnodes(tree)
				high = []
				n = 0
				for node in tree.subtrees():
					n += 1
					if n == nodenum:
						high = list(node.subtrees()) + list(node.leaves())
						break
					if isinstance(node[0], int):
						n += 1
						if n == nodenum:
							high = list(node.leaves())
							break
				else:
					raise ValueError('Matching node %d not found in tree:\n%s'
							% (nodenum, tree))
				x.append((filename, sentno, tree, sent, high))
			self.cache['trees', query, filename, start, end,
					nofunc, nomorph, detectdisc, mergedisc] = x, maxresults
			result.extend(x)
		return result

	def sents(self, query, subset=None, start=None, end=None, maxresults=100,
			brackets=False):
		subset = subset or self.files
		# %s the sentence number
		# %w complete tree in bracket notation
		# %h the tree matched by the head (first) node of the pattern
		# %yh %zh the terminal index of the first/last terminal in the match
		fmt = r'%s\n%w\n%h\n%yh\n%zh:::\n'
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['sents', query, filename,
						start, end, brackets]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(
						lambda x: list(self._query(
							[query], x, fmt, start, end, maxresults)),
						filename)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for sentno, line in future.result():
				sent, match, begin, end = line.splitlines()
				if brackets:
					match1 = match
					match2 = ''
				else:
					begin, end = int(begin) - 1, int(end)
					tokens = [ptbunescape(token)
							for token in GETLEAVES.findall(sent)]
					sent = ' '.join(tokens)
					prelen = len(' '.join(tokens[:begin]))
					match = ' '.join(tokens[begin:end])
					match1 = set(range(prelen, prelen + len(match) + 1))
					match2 = set()
				x.append((filename, sentno, sent, match1, match2))
			self.cache['sents', query, filename,
					start, end, brackets] = x, maxresults
			result.extend(x)
		return result

	def extract(self, filename, indices, nofunc=False, nomorph=False,
			detectdisc=True, mergedisc=True, sents=False):
		cmd = [which('tgrep2'),
				'-e', '-',  # extraction mode
				'-c', self._internalfilename(filename)]
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
        return [(mergediscnodes(tree) if mergedisc else tree, sent)
                for tree, sent
                in (brackettree(
						filterlabels(treestr, nofunc, nomorph),
						detectdisc=detectdisc)
					for treestr in result)]

	def getinfo(self, filename):
		with openread(filename) as inp:
			data = inp.read()
		return CorpusInfo(
				len=data.count('\n'),
				numwords=len(GETLEAVES.findall(data)),
				numnodes=data.count('('),
				maxnodes=None)

	def _internalfilename(self, filename):
		"""Strip possible .gz extension and add .t2c.gz or .t2c.lz4."""
		return '%s.t2c.%s' % (re.sub(r'\.gz$', '', filename),
				self._compressext)

	@workerfunc
	def _query(self, queries, filename, fmt, start=None, end=None,
			maxresults=None):
		"""Run queries on a single file."""
		cmd = [which('tgrep2'), '-a',  # print all matches for each sentence
				# '-z',  # pretty-print search pattern on stderr
				'-m', fmt,
				'-c', self._internalfilename(filename)]
		if self.macros:  # tgrep2 accepts a filename to read patterns from
			cmd.append(self.macros)
		cmd.extend(queries)
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
			raise ValueError('command: %s\n%s' % (' '.join(cmd), err))
		return results


class FragmentSearcher(CorpusSearcher):
	"""Search for fragments in a bracket treebank.

	Format of treebanks and queries can be bracket, discbracket, or
	export (detected based on file extension).
	Each query consists of one or more tree fragments, and the results
	will be merged together, except with batchcounts(), which returns
	the results for each fragment separately.

	Example queries::
		(S (NP (DT The) (NN )) (VP ))
		(NP (DT 0=The) (NN 1=queen))

	:param macros: a file containing lines of the form ``'name=fragment'``;
		an occurrence of ``'{name}'`` will be replaced with ``fragment`` when
		it appears in a query.
	:param inmemory: if True, keep all corpora in memory; otherwise,
		load them from disk with each query.
	"""

	# TODO: allow single terminals as queries: word
	# 		alternatively, allow wildcard: (* word)
	# TODO: allow regex labels: /label/
	# 		expand to multiple queries; feasible?
	# TODO: interpret multiple fragments in a single query as AND query,
	# 		optionally with order constraint: (NN cat) (NN dog)
	# TODO: compiled query set, re-usable on new documents.
	def __init__(self, files, macros=None, numproc=None, inmemory=True):
		super().__init__(files, macros, numproc)
		self.disc = False
		newvocab = True
		path = os.path.dirname(next(iter(sorted(files))))
		self.vocabpath = os.path.join(path, 'treesearchvocab.idx')
		if os.path.exists(self.vocabpath):
			self.vocab = FixedVocabulary.fromfile(self.vocabpath)
			mtime = os.stat(self.vocabpath).st_mtime
			if all(os.path.exists(a + '.ct')
						and mtime > os.stat(a + '.ct').st_mtime
						> os.stat(a).st_mtime for a in files):
				self.vocab.makeindex()
				newvocab = False
		if newvocab:
			self.vocab = Vocabulary()
		for filename in self.files:
			self.disc = self.disc or not filename.endswith('.mrg')
			if newvocab:
				# get format from extension
				ext = {'export': 'export',
						'mrg': 'bracket',
						'dbr': 'discbracket'}
				fmt = ext[filename.rsplit('.', 1)[1]]
				corpus = _fragments.readtreebank(filename, self.vocab, fmt=fmt)
				corpus.indextrees(self.vocab)
				corpus.tofile('%s.ct' % filename)
				newvocab = True
			if inmemory:
				self.files[filename] = Ctrees.fromfile('%s.ct' % filename)
		if newvocab:
			self.vocab.tofile(self.vocabpath)
		self.macros = None
		if macros:
			with openread(macros) as tmp:
				self.macros = dict(line.strip().split('=', 1) for line in tmp)
		self.pool = concurrent.futures.ProcessPoolExecutor(self.numproc)

	def close(self):
		if hasattr(self.vocab, 'close'):
			self.vocab.close()
		for a in self.files:
			a.close()
		self.vocab = self.files = None

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
		if self.macros is not None:
			query = query.format(**self.macros)
		result = OrderedDict()
		jobs = {}
		cquery = None
		for filename in subset:
			try:
				tmp = self.cache[
						'counts', query, filename, start, end, indices]
				if indices:
					result[filename] = [b for a in tmp for b in a]
				else:
					result[filename] = sum(tmp)
			except KeyError:
				if cquery is None:
					cquery, bitsets, maxnodes = self._parse_query(
							query, disc=self.disc)
				jobs[self._submit(
						_frag_query if self.numproc == 1 else _frag_query_mp,
						cquery, bitsets, maxnodes, filename, self.vocabpath,
						start, end, None, indices=indices, trees=False,
						)] = filename
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
		if self.macros is not None:
			queries = [query.format(**self.macros) for query in queries]
		cqueries, bitsets, maxnodes = self._parse_query(queries, disc=self.disc)
		for filename in subset:
			# NB: not using cache.
			jobs[self._submit(
					_frag_query,
					cqueries, bitsets, maxnodes, filename, self.vocabpath,
					start, end, None, indices=False, trees=False,
					)] = filename
		for future in self._as_completed(jobs):
			filename = jobs[future]
			yield filename, future.result()

	def trees(self, query, subset=None, start=None, end=None, maxresults=10,
			nofunc=False, nomorph=False, detectdisc=True, mergedisc=False):
		subset = subset or self.files
		if self.macros is not None:
			query = query.format(**self.macros)
		result = []
		jobs = {}
		cquery = None
		for filename in subset:
			try:
				x, maxresults2 = self.cache['trees', query, filename,
						start, end, nofunc, nomorph, detectdisc, mergedisc]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				if cquery is None:
					cquery, bitsets, maxnodes = self._parse_query(
							query, disc=self.disc)
				jobs[self._submit(
						_frag_query if self.numproc == 1 else _frag_query_mp,
						cquery, bitsets, maxnodes, filename, self.vocabpath,
						start, end, maxresults, indices=True, trees=True,
						)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for matches in future.result():
				for sentno, treestr, match in matches:
					treestr = filterlabels(treestr, nofunc, nomorph)
					# FIXME: this highlights the whole subtree, of which
					# frag may be a subgraph.
					treestr = treestr.replace(
							match,
							'%s_HIGH %s' % tuple(match.split(None, 1)),
							1)
					tree, sent = brackettree(treestr, detectdisc=detectdisc)
					if mergedisc:
						tree = mergediscnodes(tree)
					high = list(tree.subtrees(
							lambda n: n.label.endswith("_HIGH")))
					if high:
						high = high.pop()
						high.label = high.label.rsplit("_", 1)[0]
						high = list(high.subtrees()) + high.leaves()
					x.append((filename, sentno, tree, sent, high))
			self.cache['trees', query, filename, start, end,
					nofunc, nomorph, detectdisc, mergedisc] = x, maxresults
			result.extend(x)
		return result

	def sents(self, query, subset=None, start=None, end=None,
			maxresults=100, brackets=False):
		subset = subset or self.files
		if self.macros is not None:
			query = query.format(**self.macros)
		result = []
		jobs = {}
		cquery = None
		for filename in subset:
			try:
				x, maxresults2 = self.cache['sents', query, filename,
						start, end, brackets]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				if cquery is None:
					cquery, bitsets, maxnodes = self._parse_query(
							query, disc=self.disc)
				jobs[self._submit(
						_frag_query if self.numproc == 1 else _frag_query_mp,
						cquery, bitsets, maxnodes, filename, self.vocabpath,
						start, end, maxresults, indices=True, trees=True,
						)] = filename
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
						_, xsent = discbrackettree(treestr)
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

	def extract(self, filename, indices, nofunc=False, nomorph=False,
			detectdisc=True, mergedisc=False, sents=False):
		if self.files[filename] is not None:
			corpus = self.files[filename]
		else:
			corpus = Ctrees.fromfile('%s.ct' % filename)
		if sents:
			return [' '.join(ptbunescape(token)
					for token in corpus.extractsent(n - 1, self.vocab))
					for n in indices]
		return [(mergediscnodes(tree) if mergedisc else tree, sent)
				for tree, sent
				in (brackettree(filterlabels(
						corpus.extract(n - 1, self.vocab), nofunc, nomorph),
						detectdisc=detectdisc)
					for n in indices)]

	def getinfo(self, filename):
		if self.files[filename] is not None:
			corpus = self.files[filename]
		else:
			corpus = Ctrees.fromfile('%s.ct' % filename)
		return CorpusInfo(len=corpus.len, numwords=corpus.numwords,
				numnodes=corpus.numnodes, maxnodes=corpus.maxnodes)

	def _parse_query(self, query, disc=False):
		"""Prepare fragment query."""
		if isinstance(query, list):
			qitems = (brackettree(a. detectdisc=True) for a in query)
		else:
			qitems = treebank.incrementaltreereader(
					io.StringIO(query), strict=True, robust=False)
		qitems = (
				(binarize(handledisc(item[0]) if disc else item[0], dot=True),
				item[1]) for item in qitems)
		# FIXME: this function could be parallelized.
		queries = _fragments.getctrees(qitems, vocab=self.vocab, index=False)
		if not queries['trees1']:
			raise ValueError('no valid fragments in query.')
		maxnodes = queries['trees1'].maxnodes
		_fragmentkeys, bitsets = _fragments.completebitsets(
				queries['trees1'], self.vocab, maxnodes, disc=disc,
				tostring=False)
		return queries, bitsets, maxnodes


@workerfunc
def _frag_query_mp(queries, bitsets, maxnodes, filename, vocabpath,
		start=None, end=None, maxresults=None, indices=True, trees=False):
	"""Multiprocessing wrapper."""
	return _frag_query(
			queries, bitsets, maxnodes, filename, vocabpath, start, end,
			maxresults, indices, trees)


def _frag_query(queries, bitsets, maxnodes, filename, vocabpath,
		start=None, end=None, maxresults=None, indices=True, trees=False):
	"""Run a prepared fragment query on a single file."""
	corpus = Ctrees.fromfile('%s.ct' % filename)
	if start:
		start -= 1
	results = _fragments.exactcountsslice(
			bitsets, queries['trees1'], corpus,
			indices=indices + trees if indices else 0,
			maxnodes=maxnodes, start=start, end=end,
			maxresults=maxresults)
	if indices and trees:
		vocab = FixedVocabulary.fromfile(vocabpath)
		results = [[(n + 1,
					corpus.extract(n, vocab, disc=True),
					corpus.extract(n, vocab, disc=True, node=m))
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
		an occurrence of ``'{name}'`` will be replaced with ``regex`` when it
		appears in a query.
	:param ignorecase: ignore case in all queries."""

	def __init__(self, files, macros=None, numproc=None, ignorecase=False,
			inmemory=False):
		super().__init__(files, macros, numproc)
		self.macros = None
		self.flags = re.MULTILINE
		if ignorecase:
			self.flags |= re.IGNORECASE
		if macros:
			with openread(macros) as tmp:
				self.macros = dict(line.strip().split('=', 1) for line in tmp)
		self.fileno = {filename: n for n, filename in enumerate(sorted(files))}
		maxmtime = max(os.stat(a).st_mtime for a in files)
		path = os.path.dirname(next(iter(sorted(files))))
		self.lineidxpath = os.path.join(path, 'treesearchline.idx')
		if os.path.exists(self.lineidxpath):
			mtime = os.stat(self.lineidxpath).st_mtime
			tmp = MultiRoaringBitmap.fromfile(self.lineidxpath)
		else:
			mtime, tmp = 0, []
		if len(tmp) == len(files) and mtime > maxmtime:
			self.lineindex = tmp
		else:
			tmp = [_indexfile(name) for name in sorted(files)]
			self.lineindex = MultiRoaringBitmap(tmp, filename=self.lineidxpath)
		if inmemory:
			for filename in self.files:
				fileno = os.open(filename, os.O_RDONLY)
				buf = mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)
				self.files[filename] = (fileno, buf)
		self.pool = concurrent.futures.ProcessPoolExecutor(self.numproc)

	def close(self):
		if self.files is None:
			return
		for val in self.files.values():
			if val is not None:
				fileno, buf = val
				buf.close()
				os.close(fileno)
		if hasattr(self.lineindex, 'close'):
			self.lineindex.close()
		self.files = None

	def counts(self, query, subset=None, start=None, end=None, indices=False,
			breakdown=False):
		if breakdown and indices:
			raise NotImplementedError
		subset = subset or self.files
		if self.macros is not None:
			query = query.format(**self.macros)
		result = OrderedDict()
		jobs = {}
		pattern = _regex_parse_query(query, self.flags)
		for filename in subset:
			try:
				result[filename] = self.cache[
						'counts', query, filename, start, end, indices, False,
						breakdown]
			except KeyError:
				jobs[self._submit(
						_regex_run_query,
						pattern, filename, self.fileno[filename],
						self.lineidxpath, start, end, None, indices, False,
						breakdown,
						)] = filename
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
		if self.macros is not None:
			query = query.format(**self.macros)
		result = []
		jobs = {}
		for filename in subset:
			try:
				x, maxresults2 = self.cache['sents', query, filename,
						start, end, True, True]
			except KeyError:
				maxresults2 = 0
			if not maxresults or maxresults > maxresults2:
				jobs[self._submit(
						_regex_query if self.numproc == 1 else _regex_query_mp,
						query, filename, self.fileno[filename],
						self.lineidxpath, self.flags, start, end, maxresults,
						True, True,
						)] = filename
			else:
				result.extend(x[:maxresults])
		for future in self._as_completed(jobs):
			filename = jobs[future]
			x = []
			for sentno, sent, start, end in future.result():
				highlight = range(start, end)
				x.append((filename, sentno, sent, highlight, ()))
			self.cache['sents', query, filename, start, end, True, True
					] = x, maxresults
			result.extend(x)
		return result

	def trees(self, query, subset=None, start=None, end=None, maxresults=10,
			nofunc=False, nomorph=False, detectdisc=True, mergedisc=True):
		raise ValueError('not applicable with plain text corpus.')

	def batchcounts(self, queries, subset=None, start=None, end=None):
		if self.macros is None:
			patterns = [_regex_parse_query(query, self.flags)
					for query in queries]
		else:
			patterns = [_regex_parse_query(query.format(
					**self.macros), self.flags) for query in queries]
		chunksize = max(int(len(patterns) / (self.numproc * 4)), 1)
		chunkedpatterns = [patterns[n:n + chunksize]
				for n in range(0, len(patterns), chunksize)]
		result = OrderedDict((name, [])
				for name in subset or self.files)
		for filename in subset or self.files:
			result = array.array('I')
			for tmp in self._map(_regex_run_batch, chunkedpatterns,
					filename=filename, fileno=self.fileno[filename],
					lineidxpath=self.lineidxpath, start=start, end=end):
				result.extend(tmp)
			yield filename, result

	def batchsents(self, queries, subset=None, start=None, end=None,
			maxresults=100, brackets=False):
		"""Variant of sents() to run a batch of queries."""
		if brackets:
			raise ValueError('not applicable with plain text corpus.')
		if self.macros is None:
			patterns = [_regex_parse_query(query, self.flags)
					for query in queries]
		else:
			patterns = [_regex_parse_query(query.format(
				**self.macros), self.flags) for query in queries]
		chunksize = max(int(len(patterns) / (self.numproc * 4)), 1)
		chunkedpatterns = [patterns[n:n + chunksize]
				for n in range(0, len(patterns), chunksize)]
		result = OrderedDict((name, [])
				for name in subset or self.files)
		for filename in subset or self.files:
			result = []
			for tmp in self._map(_regex_run_batch, chunkedpatterns,
					filename=filename, fileno=self.fileno[filename],
					lineidxpath=self.lineidxpath, start=start, end=end,
					maxresults=maxresults, sents=True):
				result.extend(tmp)
			yield filename, result

	def extract(self, filename, indices, nofunc=False, nomorph=False,
			detectdisc=True, mergedisc=True, sents=True):
		if not sents:
			raise ValueError('not applicable with plain text corpus.')
		lineindex = self.lineindex[self.fileno[filename]]
		result = []
		if self.files[filename] is not None:
			_, data = self.files[filename]
			for lineno in indices:
				offset, nextoffset = _getoffsets(lineno, lineindex, data)
				result.append(data[offset:nextoffset].decode('utf8'))
		else:
			with open(filename, 'rb') as tmp:
				for lineno in indices:
					offset, nextoffset = _getoffsets(lineno, lineindex, None)
					tmp.seek(offset)
					result.append(tmp.read(nextoffset - offset
							).rstrip(b'\n').decode('utf8'))
		return result

	def getinfo(self, filename):
		numlines = len(self.lineindex[self.fileno[filename]])
		if self.files[filename] is None:
			with openread(filename) as inp:
				data = inp.read()
		else:
			_, data = self.files[filename]
		numwords = data.count(' ') + numlines
		return CorpusInfo(
				len=numlines, numwords=numwords,
				numnodes=0, maxnodes=None)


@workerfunc
def _regex_query_mp(query, filename, fileno, lineidxpath, flags,
		start=None, end=None, maxresults=None, indices=True, sents=False,
		breakdown=False):
	"""Multiprocessing wrapper."""
	return _regex_query(query, filename, fileno, lineidxpath, flags,
			start, end, maxresults, indices, sents, breakdown)


def _regex_query(query, filename, fileno, lineidxpath, flags,
		start=None, end=None, maxresults=None, indices=True, sents=False,
		breakdown=False):
	"""Run a query on a single file."""
	pattern = _regex_parse_query(query, flags)
	return _regex_run_query(pattern, filename, fileno, lineidxpath,
			start=start, end=end, maxresults=maxresults, indices=indices,
			sents=sents, breakdown=breakdown)


def _regex_parse_query(query, flags):
	"""Prepare regex query."""
	pattern = None
	if RE2LIB:
		try:
			pattern = re2.compile(  # pylint: disable=no-member
					query.encode('utf8'), flags=flags | re.UNICODE,
					max_mem=8 << 26)  # 500 MB
		except ValueError:
			pass
	if pattern is None:
		pattern = re.compile(query.encode('utf8'), flags=flags)
	return pattern


def _regex_run_query(pattern, filename, fileno, lineidxpath,
		start=None, end=None, maxresults=None, indices=False, sents=False,
		breakdown=False):
	"""Run a prepared query on a single file."""
	mrb = MultiRoaringBitmap.fromfile(lineidxpath)
	lineindex = mrb.get(fileno)
	if indices and sents:
		result = []
	elif indices:
		result = array.array('I')
	elif breakdown:
		result = Counter()
	else:
		result = 0
	lastline = end is None or end > len(lineindex) - 1
	if lastline:
		end = len(lineindex) - 1
	if start and start > len(lineindex):
		return result
	startidx = lineindex.select(start - 1 if start else 0)
	endidx = lineindex.select(end)
	with open(filename, 'rb') as tmp:
		if startidx == 0 and lastline:
			chunkoffset = 0
			data = mmap.mmap(tmp.fileno(), 0, access=mmap.ACCESS_READ)
		else:
			chunkoffset = startidx
			tmp.seek(chunkoffset)
			startidx, endidx = 0, endidx - chunkoffset
			data = tmp.read(endidx)
		try:
			if (start or 0) >= len(lineindex):
				return result
			if indices or sents:
				for match in islice(
						pattern.finditer(data, startidx, endidx), maxresults):
					mstart = match.start()
					mend = match.end()
					lineno = lineindex.rank(mstart + chunkoffset)
					if not sents:
						result.append(lineno)
						continue
					offset, nextoffset = _getoffsets(lineno, lineindex, None)
					offset -= chunkoffset
					nextoffset -= chunkoffset
					sent = data[offset:nextoffset].rstrip(b'\n').decode('utf8')
					mstart = len(data[offset:mstart].decode('utf8'))
					mend = len(data[offset:mend].decode('utf8'))
					# (lineno, sent, startspan, endspan)
					result.append((lineno, sent, mstart, mend))
			else:
				if breakdown:
					matches = pattern.findall(
							data, startidx, endidx)[:maxresults]
					result.update(a.decode('utf8') for a in matches)
				else:
					try:
						result = pattern.count(data, startidx, endidx)
					except AttributeError:
						result = len(pattern.findall(data, startidx, endidx))
					result = max(result, maxresults or 0)
		finally:
			if isinstance(data, mmap.mmap):
				data.close()
			if hasattr(mrb, 'close'):
				mrb.close()
			del mrb
	return result


def _regex_run_batch(patterns, filename, fileno, lineidxpath,
		start=None, end=None, maxresults=None, sents=False):
	"""Run a batch of queries on a single file."""
	mrb = MultiRoaringBitmap.fromfile(lineidxpath)
	lineindex = mrb.get(fileno)
	if sents:
		result = []
	else:
		result = array.array('I')
	if start and start >= len(lineindex):
		return result
	with open(filename, 'rb') as tmp:
		data = mmap.mmap(tmp.fileno(), 0, access=mmap.ACCESS_READ)
		try:
			startidx = lineindex.select(start - 1 if start else 0)
			endidx = (lineindex.select(end) if end is not None
					and end < len(lineindex) else len(data))
			if sents:
				for pattern in patterns:
					for match in islice(
							pattern.finditer(data, startidx, endidx),
							maxresults):
						mstart = match.start()
						mend = match.end()
						lineno = lineindex.rank(mstart)
						offset, nextoffset = _getoffsets(
								lineno, lineindex, data)
						sent = data[offset:nextoffset].decode('utf8')
						mstart = len(data[offset:mstart].decode('utf8'))
						mend = len(data[offset:mend].decode('utf8'))
						# sentno, sent, high1, high2
						result.append((lineno, sent, range(mstart, mend), ()))
			else:
				for pattern in patterns:
					try:
						result.append(pattern.count(data, startidx, endidx))
					except AttributeError:
						result.append(len(pattern.findall(
								data, startidx, endidx)))
		finally:
			data.close()
			if hasattr(mrb, 'close'):
				mrb.close()
			del mrb
	return result


def _getoffsets(lineno, lineindex, data):
	"""Return the (start, end) byte offsets for a given 1-based line number."""
	offset = 0
	if 1 <= lineno < len(lineindex):
		offset = lineindex.select(lineno - 1)
	else:
		raise IndexError
	nextoffset = lineindex.select(lineno)
	# there is at least one newline, but there may be more
	# because empty lines are not indexed.
	while data is not None and data[nextoffset - 1] == 10:  # b'\n':
		nextoffset -= 1
	return offset, nextoffset


def _indexfile(filename):
	"""Create bitmap with locations of non-empty lines."""
	result = RoaringBitmap()
	offset = 0
	with open(filename, 'rb') as tmp:
		for line in tmp:
			if not line.isspace():
				result.add(offset)
			offset += len(line)
	result.add(offset)
	return result.freeze()


class NoFuture(object):
	"""A non-asynchronous version of concurrent.futures.Future."""

	def __init__(self, func, *args, **kwargs):
		self._result = func(*args, **kwargs)

	def result(self, timeout=None):  # pylint: disable=unused-argument
		"""Return the precomputed result."""
		return self._result


class FIFOOrederedDict(OrderedDict):
	"""FIFO cache with maximum number of elements based on OrderedDict."""

	def __init__(self, limit):
		super().__init__()
		self.limit = limit

	def __setitem__(self, key, value):  # pylint: disable=arguments-differ
		if self.limit == 0:
			return
		elif key in self:
			self.pop(key)
		elif len(self) >= self.limit:
			self.pop(next(iter(self)))
		super().__setitem__(key, value)


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


def applyhighlight(sent, high1, high2, reset=False,
		high1color='red', high2color='blue'):
	"""Highlight character indices high1 & high2 in sent with ANSI colors.

	:param reset: if True, reset to normal color before every change
		(useful in IPython notebook)."""
	cur = None
	start = 0
	out = []
	reset = '\x1b[0m' if reset else ''
	high1color = ANSICOLOR[high1color]
	high2color = ANSICOLOR[high2color]
	for n, _ in enumerate(sent):
		if n in high1:
			if cur != high1color:
				cur = high1color
				out.append('%s%s\x1b[%d;1m' % (sent[start:n], reset, cur))
				start = n
		elif n in high2:
			if cur != high2color:
				cur = high2color
				out.append('%s%s\x1b[%d;1m' % (sent[start:n], reset, cur))
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
			'line-number file ignore-case no-filename csv help')
	try:
		opts, args = gnu_getopt(sys.argv[2:], shortoptions, options.split())
		query, corpora = args[0], args[1:]
		if isinstance(query, bytes):
			query = query.decode('utf8')
		if not corpora:
			raise ValueError('enter one or more corpus files')
	except (GetoptError, IndexError, ValueError) as err:
		print(err, file=sys.stderr)
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
	# FIXME: support negative indices? why 1-based indices?
	ignorecase = '--ignore-case' in opts or '-i' in opts
	printfname = (len(corpora) > 1 and '--no-filename' not in opts
			and '-h' not in opts)
	printlineno = '--line-number' in opts or '-n' in opts
	if ignorecase and engine != 'regex':
		raise ValueError('--ignore-case is only supported with --engine=regex')
	if engine == 'tgrep2':
		searcher = TgrepSearcher(corpora, macros=macros, numproc=numproc)
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
		elif len(queries) > 1 or '--csv' in opts:
			if indices:
				raise ValueError('--indices only supported with single query.')
			results = searcher.batchcounts(
					queries, start=start, end=end)
			writecounts(results, flat='--csv' not in opts, columns=queries)
		else:
			for filename, cnt in searcher.counts(
					query, start=start, end=end, indices=indices).items():
				if printfname:
					print('\x1b[%dm%s\x1b[0m:' % (
						ANSICOLOR['magenta'], filename), end='')
				print(list(cnt) if indices else cnt)
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
			if printfname:
				print('\x1b[%dm%s\x1b[0m:' % (
						ANSICOLOR['magenta'], filename), end='')
			if printlineno:
				print('\x1b[%dm%s\x1b[0m:'
						% (ANSICOLOR['green'], sentno), end='')
			if printfname or printlineno:
				print('\n', end='')
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
								for n, char in enumerate(sent))
					else:
						out = applyhighlight(sent, high1, high2)
				if printfname:
					print('\x1b[%dm%s\x1b[0m:' % (
							ANSICOLOR['magenta'], filename), end='')
				if printlineno:
					print('\x1b[%dm%s\x1b[0m:'
							% (ANSICOLOR['green'], sentno), end='')
				print(out)
		if '--breakdown' in opts:
			for match, cnt in breakdown.most_common():
				print('%5d %s' % (cnt, match))


__all__ = ['CorpusSearcher', 'TgrepSearcher', 'RegexSearcher',
		'FragmentSearcher', 'NoFuture', 'FIFOOrederedDict', 'filterlabels',
		'cpu_count', 'charindices', 'applyhighlight']
