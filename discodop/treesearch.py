"""Objects for searching through collections of trees."""
import io
import os
import re
import glob
import concurrent.futures
import subprocess
from itertools import islice, count, takewhile
try:
	from multiprocessing import cpu_count
except ImportError:
	from os import cpu_count
from lru import LRU
try:
	import alpinocorpus
	import xml.etree.cElementTree as ElementTree
	ALPINOCORPUSLIB = True
except ImportError:
	ALPINOCORPUSLIB = False
from discodop.tree import Tree
from discodop import treebank

CACHESIZE = 100
GETLEAVES = re.compile(r' ([^ ()]+)(?=[ )])')
ALPINOLEAVES = re.compile('<sentence>(.*)</sentence>')
MORPH_TAGS = re.compile(
		r'([_/*A-Z0-9]+)(?:\[[^ ]*\][0-9]?)?((?:-[_A-Z0-9]+)?(?:\*[0-9]+)? )')
FUNC_TAGS = re.compile(r'-[_A-Z0-9]+')

# Goals:
# [x] incrementality; per text, per sentence
# [x] multithreading
# [x] caching
# [x] usable from Python console
# [x] tgrep2
# [x] XPath
# [x] regex
# [ ] brackets output
# [ ] export results to file
# [ ] fragment queries?
# [ ] CLI
# Issues:
# - caching? what about partially iterated results etc.
#   cache each result of _query before conversion? or after?
# - highlighting? return tree + list of highlighted nodes / sent + slice.
# - how to select from corpus: integers, or names? filenames.
# - what to return? Trees / lists of tokens.
class CorpusSearcher(object):
	"""Abstract base class of a set of corpus files that can be queried.

	:param path: a pattern describing the corpus files; e.g., '*.txt'
	:param macros: a filename with macros that can be used in queries.
	:param numthreads: the number of concurrent threads to use;
		None to disable threading."""

	def counts(self, query, subset=None, limit=None):
		"""Run query and return a dict of the form {corpus1: nummatches, ...}.

		:param query: the search query
		:param subset: an iterable of filenames to run the query on; by default
			all filenames are used.
		:param limit: the maximum number of sentences to query per corpus; by
			default, all sentences are queried."""

	def trees(self, query, subset=None, maxresults=10,
			nofunc=False, nomorph=False):
		"""Run query and return list of matching trees.

		:param maxresults: the maximum number of matches to return.
		:returns: list of tuples of the form
			``(corpus, sentno, tree, sent, highlight)``
			highlight is a set of nodes from tree."""

	def sents(self, query, subset=None, maxresults=100):
		"""Run query and yield matching sentences.

		:param maxresults: the maximum number of matches to return.
		:returns: list of tuples of the form
			``(corpus, sentno, sent, highlight)``
			highlight is a set of indices."""

	def export(self, query, output, linenos, subset=None):
		"""Run tgrep2 on each text, yield results as single string."""


class TgrepSearcher(CorpusSearcher):
	"""Search a corpus with tgrep2."""
	def __init__(self, path, macros, numthreads=None):
		self.files = tuple(glob.glob(path))
		# TODO: autoconvert .mrg to .t2c.gz
		self.macros = macros
		self.numthreads = numthreads
		self.cache = LRU(CACHESIZE)
		self.pool = concurrent.futures.ThreadPoolExecutor(
				numthreads or cpu_count())
		assert self.files, 'no files found matching ' + path

	def counts(self, query, subset=None, limit=None):
		subset = self.files if subset is None else tuple(subset)
		try:
			result = self.cache['counts', query, subset, limit]
		except KeyError:
			result = {}
		else:
			return result
		# %s the sentence number
		fmt = r'%s:::\n'
		jobs = {self.pool.submit(lambda x: sum(1 for _ in self._query(
				query, x, fmt, None, limit)), filename):
				filename for filename in subset}
		for future in concurrent.futures.as_completed(jobs):
			filename = jobs[future]
			result[filename] = future.result()
		self.cache['counts', query, subset, limit] = result
		return result

	def trees(self, query, subset=None, maxresults=10,
			nofunc=False, nomorph=False):
		subset = self.files if subset is None else tuple(subset)
		try:
			result, maxresults2 = self.cache['trees', query, subset]
			if maxresults <= maxresults2:
				return result[:maxresults]
		except KeyError:
			result = []
		# %s the sentence number
		# %w complete tree in bracket notation
		# %h the matched subtree in bracket notation
		fmt = r'%s:::%w:::%h\n'
		jobs = {self.pool.submit(lambda x: list(self._query(
				query, x, fmt, maxresults)), filename):
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
		self.cache['trees', query, subset] = result, maxresults
		return result

	def sents(self, query, subset=None, maxresults=100):
		subset = self.files if subset is None else tuple(subset)
		try:
			result, maxresults2 = self.cache['sents', query, subset]
			if maxresults <= maxresults2:
				return result[:maxresults]
		except KeyError:
			result = []
		# %s the sentence number
		# %w complete tree in bracket notation
		# %h the matched subtree in bracket notation
		fmt = r'%s:::%w:::%h\n'
		jobs = {self.pool.submit(lambda x: list(self._query(
				query, x, fmt, maxresults)), filename):
				filename for filename in subset}
		for future in concurrent.futures.as_completed(jobs):
			filename = jobs[future]
			for sentno, line in future.result():
				treestr, match = line.split(':::')
				pre, highlight, post = treestr.partition(match)
				pre = GETLEAVES.findall(pre)
				match = (GETLEAVES.findall(highlight)
						if '(' in highlight else [highlight])
				post = GETLEAVES.findall(post)
				sent = pre + match + post
				highlight = range(len(pre), len(pre) + len(match))
				result.append((filename, sentno, sent, highlight))
		self.cache['sents', query, subset] = result, maxresults
		return result

	def export(self, query, output, linenos, subset=None):
		raise NotImplementedError
		subset = self.files if subset is None else tuple(subset)
		if output == 'sents':
			fmt = r'%f:%s|%tw\n' if linenos else r'%tw\n'
		elif output == 'trees' or output == 'brackets':
			fmt = r"%f:%s|%w\n" if linenos else r"%w\n"
		else:
			raise ValueError
		for filename in subset:
			out, err = self._query(query, filename, fmt)
			yield filename, out, err

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
		# TODO: check for err, raise if fatal.
		out = out.decode('utf8')  # pylint: disable=E1103
		err = err.decode('utf8')  # pylint: disable=E1103
		proc.stdout.close()
		proc.stderr.close()
		proc.wait()
		# TODO: can we do this without reading stdout completely?
		results = ((int(match.group(1)), match.group(2)) for match
				in re.finditer(r'([0-9]+):::([^\n]*)\n', out))
		if limit is not None:
			results = takewhile(lambda x: x[0] < limit, results)
		return islice(results, maxresults)


class DactSearcher(CorpusSearcher):
	def __init__(self, path, macros, numthreads=None):
		try:
			self.files = {filename: alpinocorpus.CorpusReader(
				filename, macrosFilename=macros)
				for filename in glob.glob(path)}
		except TypeError:
			assert macros is None, 'macros not supported'
			self.files = {filename: alpinocorpus.CorpusReader(filename)
				for filename in glob.glob(path)}
		self.macros = macros
		self.numthreads = numthreads
		self.cache = LRU(CACHESIZE)
		self.pool = concurrent.futures.ThreadPoolExecutor(
				numthreads or cpu_count())
		assert self.files, 'no files found matching ' + path

	def counts(self, query, subset=None, limit=None):
		subset = tuple(subset or self.files)
		try:
			result = self.cache['counts', query, subset, limit]
		except KeyError:
			result = {}
		else:
			return result
		jobs = {self.pool.submit(lambda x: sum(1 for _ in self._query(
				query, x, None, limit)), filename):
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
			result, maxresults2 = self.cache['trees', query, subset]
			if maxresults <= maxresults2:
				return result[:maxresults]
		except KeyError:
			result = []
		jobs = {self.pool.submit(lambda x: list(self._query(query, x,
				maxresults)), filename): filename for filename in subset}
		for future in concurrent.futures.as_completed(jobs):
			filename = jobs[future]
			for match in future.result():
				sentno = int(match.name().rsplit('.', 1)[0])
				treestr = self.files[filename].read(match.name())
				match = match.contents()  # .decode('utf8')
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
		self.cache['trees', query, subset] = result, maxresults
		return result

	def sents(self, query, subset=None, maxresults=100):
		subset = tuple(subset or self.files)
		try:
			result, maxresults2 = self.cache[
					'sents', query, subset]
			if maxresults <= maxresults2:
				return result[:maxresults]
		except KeyError:
			result = []
		jobs = {self.pool.submit(lambda x: list(self._query(query, x,
				maxresults)), filename): filename for filename in subset}
		for future in concurrent.futures.as_completed(jobs):
			filename = jobs[future]
			for match in future.result():
				sentno = int(match.name().rsplit('.', 1)[0])
				treestr = self.files[filename].read(match.name())
				sent = ALPINOLEAVES.search(treestr).group(1).split()
				# extract starting index of highlighted words
				match = match.contents()  # .decode('utf8')
				highlight = set(re.findall(
						'<node[^>]*begin="([0-9]+)"[^>]*/>', match))
				result.append((filename, sentno, sent, highlight))
		self.cache['sents', query, subset] = result, maxresults
		return result

	def _query(self, query, filename, maxresults=None, limit=None):
		# NB: results aren't sorted, so we need to iterate exhaustively
		results = (entry for entry in self.files[filename].xpath(query)
				if limit is None or int(entry.name().split('.')[0]) < limit)
		return islice(results, maxresults)



class RegexSearcher(CorpusSearcher):
	def __init__(self, path, macros, numthreads=None):
		if macros:
			raise NotImplementedError
		# TODO: autoconvert treebank files to tokenized sents files
		self.files = {filename: list(io.open(filename, encoding='utf-8'))
				for filename in glob.glob(path)}
		self.numthreads = numthreads
		self.cache = LRU(CACHESIZE)
		self.pool = concurrent.futures.ThreadPoolExecutor(
				numthreads or cpu_count())
		assert self.files, 'no files found matching ' + path

	def counts(self, query, subset=None, limit=None):
		subset = tuple(subset or self.files)
		try:
			result = self.cache['counts', query, subset, limit]
		except KeyError:
			result = {}
		else:
			return result
		jobs = {self.pool.submit(lambda x: sum(1 for _ in self._query(
				query, x, None, limit)), filename):
				filename for filename in subset}
		for future in concurrent.futures.as_completed(jobs):
			filename = jobs[future]
			result[filename] = future.result()
		self.cache['counts', query, subset, limit] = result
		return result

	def trees(self, query, subset=None, maxresults=10,
			nofunc=False, nomorph=False):
		raise NotImplementedError

	def sents(self, query, subset=None, maxresults=100):
		subset = tuple(subset or self.files)
		try:
			result, maxresults2 = self.cache['sents', query, subset]
			if maxresults <= maxresults2:
				return result[:maxresults]
		except KeyError:
			result = []
		jobs = {self.pool.submit(lambda x: list(self._query(
				query, x, maxresults)), filename):
				filename for filename in subset}
		for future in concurrent.futures.as_completed(jobs):
			filename = jobs[future]
			for sentno, sent, match in future.result():
				pre, highlight, post = sent.partition(match)
				pre = GETLEAVES.findall(pre)
				match = (GETLEAVES.findall(highlight)
						if '(' in highlight else [highlight])
				post = GETLEAVES.findall(post)
				sent = pre + match + post
				highlight = range(len(pre), len(pre) + len(match))
				result.append((filename, sentno, sent, highlight))
		self.cache['sents', query, subset] = result, maxresults
		return result

	def _query(self, query, filename, maxresults=None, limit=None):
		regex = re.compile(query)
		results = ((n, match.string, match.group()) for n, match in
				enumerate((regex.search(a) for a in self.files[filename]), 1)
				if match is not None)
		if limit is not None:
			results = takewhile(lambda x: x < limit, results)
		return islice(results, maxresults)


def filterlabels(line, nofunc, nomorph):
	"""Remove morphological and/or grammatical function labels from tree(s)."""
	if nofunc:
		line = FUNC_TAGS.sub('', line)
	if nomorph:
		line = MORPH_TAGS.sub(
				lambda g: '%s%s' % (g.group(1), g.group(2)),
				line)
	return line


def which(program):
	"""Return first match for program in search path."""
	for path in os.environ.get('PATH', os.defpath).split(":"):
		if path and os.path.exists(os.path.join(path, program)):
			return os.path.join(path, program)
	raise ValueError('%r not found in path; please install it.' % program)
