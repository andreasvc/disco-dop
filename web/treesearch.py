"""Web interface to search a treebank. Requires Flask, tgrep2
or alpinocorpus-python (for xpath queries), style. Expects one or more
treebanks with .mrg or .dact extension in the directory corpus/"""
# stdlib
import io
import os
import re
import cgi
import csv
import glob
import logging
import tempfile
import subprocess
from heapq import nlargest
from urllib import quote
from datetime import datetime, timedelta
from itertools import islice, groupby
from operator import itemgetter
from collections import Counter, OrderedDict, defaultdict
try:
	import cPickle as pickle
except ImportError:
	import pickle
import pandas
# Flask & co
from flask import Flask, Response
from flask import request, render_template, send_from_directory
from werkzeug.urls import url_encode
# alpinocorpus
try:
	import alpinocorpus
	import xml.etree.cElementTree as ElementTree
	ALPINOCORPUSLIB = True
except ImportError:
	ALPINOCORPUSLIB = False
# disco-dop
from discodop.treedraw import DrawTree
from discodop import treebank, fragments
from discodop.treesearch import TgrepSearcher, DactSearcher, RegexSearcher, \
		filterlabels

MINFREQ = 2  # filter out fragments which occur just once or twice
MINNODES = 3  # filter out fragments with only three nodes (CFG productions)
TREELIMIT = 10  # max number of trees to draw in search resuluts
FRAGLIMIT = 250  # max amount of search results for fragment extraction
SENTLIMIT = 1000  # max number of sents/brackets in search results
STYLELANG = 'nl'  # language to use when running style(1)
CORPUS_DIR = "corpus/"

APP = Flask(__name__)

MORPH_TAGS = re.compile(
		r'([_/*A-Z0-9]+)(?:\[[^ ]*\][0-9]?)?((?:-[_A-Z0-9]+)?(?:\*[0-9]+)? )')
FUNC_TAGS = re.compile(r'-[_A-Z0-9]+')
GETLEAVES = re.compile(r' ([^ ()]+)(?=[ )])')
ALPINOLEAVES = re.compile('<sentence>(.*)</sentence>')
GETFRONTIERNTS = re.compile(r"\(([^ ()]+) \)")
READGRADERE = re.compile(r'([- A-Za-z]+): ([0-9]+(?:\.[0-9]+)?)[\n /]')
AVERAGERE = re.compile(
		r'([a-z]+), average length ([0-9]+(?:\.[0-9]+)?) ([A-Za-z]+)')
PERCENTAGE1RE = re.compile(
		r'([A-Za-z][A-Za-z ()]+) ([0-9]+(?:\.[0-9]+)?)% \([0-9]+\)')
PERCENTAGE2RE = re.compile(
		r'([0-9]+(?:\.[0-9]+)?)% \([0-9]+\) ([A-Za-z ()]+)\n')
# the extensions for corpus files for each query engine:
EXTRE = re.compile(r'\.(?:mrg(?:\.t2c\.gz)?|dact$)$')
EXT = {
		'tgrep2': '.mrg.t2c.gz',
		'xpath': '.dact',
		'regex': '.tok'
	}
COLORS = dict(enumerate(
		'black red orange blue wheat khaki'.split()))


@APP.route('/')
@APP.route('/counts')
@APP.route('/trees')
@APP.route('/sents')
@APP.route('/brackets')
@APP.route('/fragments')
def main():
	"""Main search form & results page."""
	output = None
	if request.path != '/':
		output = request.path.lstrip('/')
	elif 'output' in request.args:
		output = request.args['output']
	selected = selectedtexts(request.args)
	if output:
		if output not in DISPATCH:
			return 'Invalid argument', 404
		elif request.args.get('export'):
			return export(request.args, output)
		# For debugging purposes:
		return render_template('searchresults.html',
				form=request.args, texts=TEXTS, selectedtexts=selected,
				output=output, results=DISPATCH[output](request.args),
				havexpath=ALPINOCORPUSLIB)
		# To send results incrementally:
		#return Response(stream_template('searchresults.html',
		#		form=request.args, texts=TEXTS, selectedtexts=selected,
		#		output=output, results=DISPATCH[output](request.args),
		#		havexpath=ALPINOCORPUSLIB))
	return render_template('search.html', form=request.args, output='counts',
			texts=TEXTS, selectedtexts=selected, havexpath=ALPINOCORPUSLIB)


def export(form, output):
	"""Export search results to a file for download."""
	# NB: no distinction between trees from different texts
	selected = {CORPUS_DIR + TEXTS[n] + EXT[form['engine']]: n for n in
			selectedtexts(request.args)}
	if output == 'counts':
		results = counts(form, doexport=True)
		filename = 'counts.csv'
	elif output == 'fragments':
		results = fragmentsinresults(form, doexport=True)
		filename = 'fragments.txt'
	elif output in ('sents', 'brackets', 'trees'):
		if form.get('engine') == 'xpath' and output != 'sents':
			fmt = '<!-- %s:%s -->\n%s\n\n'
		else:
			fmt = '%s:%s|%s\n'
		results = ((fmt % (a[0], a[1], (a[2]))
				if form.get('linenos') else (a[2] + '\n')).encode('utf-8')
				for a in CORPORA[form.get('engine', 'tgrep2')].sents(
					form['query'], selected, maxresults=SENTLIMIT,
					brackets=output in ('brackets', 'trees')))
		filename = output + '.txt'
	else:
		raise ValueError('cannot export %s' % output)
	resp = Response(results, mimetype='text/plain')
	resp.headers['Content-Disposition'] = 'attachment; filename=' + filename
	return resp


def counts(form, doexport=False):
	"""Produce graphs and tables for a set of queries.

	Queries should be given one per line, optionally prefixed by a name and
	a normalization query::

		[name: ][normquery<tab>]query

	returns one graph for each query, and an overview with totals (optionally
	per category, if the first letters of each corpus name form a small set);
	"""
	# TODO:
	# - offer graph for individual texts with all queries together.
	# - show multiple colors in combined results concordance plot
	# - side-by-side comparison of two passages, with matching queries
	#   highlighted in different colors
	norm = form.get('norm', 'sents')
	selected = {CORPUS_DIR + TEXTS[n] + EXT[form['engine']]: n for n in
			selectedtexts(form)}
	if not doexport:
		url = 'counts?' + url_encode(dict(export=1, **form))
		yield ('Counts from queries '
				'(<a href="%s">export to CSV</a>):\n' % url)
	if norm == 'query':
		normresults = CORPORA[form.get('engine', 'tgrep2')].counts(
				form['normquery'], selected)
	combined = defaultdict(Counter)
	combined1 = defaultdict(list)
	index = [TEXTS[n] for n in selected.values()]
	df = pandas.DataFrame(index=index)
	queries = [line.split(':')[0] for line in form['query'].splitlines()]
	if not doexport:
		yield '<ol>%s</ol>\n' % '\n'.join(
				'<li><a href="#q%d">%s</a>' % (n, query)
				for n, query in enumerate(queries, 1))
	for n, line in enumerate(form['query'].splitlines() + [None], 1):
		cnts = Counter()
		sumtotal = 0
		relfreq = {}
		if line is None:
			if len(df.columns) == 1:
				break
			name = 'Combined results'
			results = combined1
			query = '%sLegend:\t%s' % (64 * ' ', '\t'.join(
					'<font color=%s>%s</font>' % (COLORS.get(n, 'black'), query)
					for n, query in enumerate(queries)))
		else:
			if ':' in line:
				name, query = line.split(':', 1)
			else:
				name, query = 'Query %d' % n, line
			if '\t' in query:
				normquery, query = query.split('\t', 1)
				norm = 'query'
				normresults = CORPORA[form.get('engine', 'tgrep2')].counts(
						normquery, selected)
			else:
				norm = form.get('norm', 'sents')
			results = CORPORA[form.get('engine', 'tgrep2')].counts(
					query, selected, indices=True)
		if not doexport:
			yield '<a name=q%d><h3>%s</h3></a>\n<pre>\n%s\n' % (n, name, query)
		for filename, indices in sorted(results.items()):
			if line is None:
				cnt = sum(combined[filename].values())
			else:
				combined[filename].update(indices)
				combined1[filename].append(indices)
				cnt = sum(indices.values())
			textno = selected[filename]
			limit = (int(form.get('limit')) if form.get('limit')
					else NUMSENTS[textno])
			text = TEXTS[textno]
			cnts[text] = cnt
			if norm == 'consts':
				total = NUMCONST[textno]
			elif norm == 'words':
				total = NUMWORDS[textno]
			elif norm == 'sents':
				total = NUMSENTS[textno]
			elif norm == 'query':
				total = normresults[filename] or 1
			else:
				raise ValueError
			relfreq[text] = 100.0 * cnt / total
			sumtotal += total
			if not doexport:
				out = "%s%6d    %5.2f %%" % (
						text.ljust(40)[:40], cnt, relfreq[text])
				plot = concplot(indices, limit or NUMSENTS[textno])
				if cnt:
					yield out + plot + '\n'
				else:
					yield '<span style="color: gray; ">%s%s</span>\n' % (
							out, plot)
		if not doexport or line is not None:
			df[name] = pandas.Series(relfreq)
		if not doexport:
			yield ("%s%6d    %5.2f %%\n</span>\n" % (
					"TOTAL".ljust(40),
					sum(cnts.values()),
					100.0 * sum(cnts.values()) / sumtotal))
			yield '</pre>'
			if max(cnts.values()) == 0:
				continue
			elif form.get('limit'):
				# show absolute counts when all texts have been limited to same
				# number of sentences
				yield barplot(cnts, max(cnts.values()),
						'Absolute counts of %s:' % name, unit='matches')
			else:
				yield barplot(relfreq, max(relfreq.values()),
						'Relative frequency of %s: '
						'(count / num_%s * 100)' % (name, norm), unit='%')
	if doexport:
		tmp = io.BytesIO()
		df.to_csv(tmp)
		yield tmp.getvalue()
	else:
		fmt = lambda x: '%g' % round(x, 1)
		yield '<h3>Overview of patterns</h3>\n'
		# collate stats
		firstletters = {key[0] for key in df.index}
		if len(firstletters) <= 5:
			overview = OrderedDict(('%s_%s' % (letter, query),
					df[query].ix[[key for key in df.index
						if key[0] == letter]].mean())
					for query in df.columns
						for letter in firstletters)
			df['category'] = [key[0] for key in df.index]
			yield '<pre>\n%s\n</pre>' % (
				df.groupby('category').describe().to_string(
				float_format=fmt))
		else:
			overview = OrderedDict((query, df[query].mean())
				for query in df.columns)
			yield '<pre>\n%s\n</pre>' % df.describe().to_string(float_format=fmt)
		yield barplot(overview, max(overview.values()),
				'Relative frequencies of patterns: '
				'(count / num_%s * 100)' % norm, unit='%', dosort=False)


def trees(form):
	"""Return visualization of parse trees in search results."""
	gotresults = False
	selected = {CORPUS_DIR + TEXTS[n] + EXT[form['engine']]: n for n in
			selectedtexts(form)}
	# NB: we do not hide function or morphology tags when exporting
	url = 'trees?' + url_encode(dict(export=1, **form))
	yield ('<pre>Query: %s\n'
			'Trees (showing up to %d per text; '
			'export: <a href="%s">plain</a>, '
			'<a href="%s">with line numbers</a>):\n' % (
				form['query'] if len(form['query']) < 128
				else form['query'][:128] + '...',
				TREELIMIT, url, url + '&linenos=1'))
	for n, (filename, results) in enumerate(groupby(CORPORA[form.get(
			'engine', 'tgrep2')].trees(form['query'], selected,
			maxresults=TREELIMIT, nomorph='nomorph' in form,
			nofunc='nofunc' in form), itemgetter(0))):
		textno = selected[filename]
		text = TEXTS[textno]
		for m, (filename, sentno, tree, sent, high) in enumerate(results):
			if m == 0:
				gotresults = True
				yield ("==&gt; %s: [<a href=\"javascript: toggle('n%d'); \">"
						"toggle</a>]\n<span id=n%d>" % (text, n + 1, n + 1))
			link = ('<a href="/browse?text=%d&sent=%s%s%s">browse</a>'
					'|<a href="/browsesents?text=%d&sent=%s&highlight=%s">'
					'context</a>' % (textno, sentno,
					'&nofunc' if 'nofunc' in form else '',
					'&nomorph' if 'nomorph' in form else '',
					textno, sentno, sentno))
			try:
				treerepr = DrawTree(tree, sent, highlight=high).text(
						unicodelines=True, html=True)
			except ValueError as err:
				line = "#%s \nERROR: %s\n%s\n%s\n" % (
						sentno, err, tree, sent)
			else:
				line = "#%s [%s]\n%s\n" % (sentno, link, treerepr)
			yield line
		yield "</span>"
	yield '</pre>' if gotresults else "No matches."


def sents(form, dobrackets=False):
	"""Return search results as terminals or in bracket notation."""
	gotresults = False
	selected = {CORPUS_DIR + TEXTS[n] + EXT[form['engine']]: n for n in
			selectedtexts(form)}
	url = '%s?%s' % ('trees' if dobrackets else 'sents',
			url_encode(dict(export=1, **form)))
	yield ('<pre>Query: %s\n'
			'Sentences (showing up to %d per text; '
			'export: <a href="%s">plain</a>, '
			'<a href="%s">with line numbers</a>):\n' % (
				form['query'] if len(form['query']) < 128
				else form['query'][:128] + '...',
				SENTLIMIT, url, url + '&linenos=1'))
	for n, (filename, results) in enumerate(groupby(CORPORA[form.get(
			'engine', 'tgrep2')].sents(form['query'], selected,
			maxresults=SENTLIMIT, brackets=dobrackets), itemgetter(0))):
		textno = selected[filename]
		text = TEXTS[textno]
		for m, (filename, sentno, sent, high) in enumerate(results):
			if m == 0:
				gotresults = True
				yield ("\n%s: [<a href=\"javascript: toggle('n%d'); \">"
						"toggle</a>] <ol id=n%d>" % (text, n, n))
			link = ('<a href="/browse?text=%d&sent=%s%s%s">draw</a>'
					'|<a href="/browsesents?text=%d&sent=%s&highlight=%s">'
					'context</a>' % (textno, sentno,
					'&nofunc' if 'nofunc' in form else '',
					'&nomorph' if 'nomorph' in form else '',
					textno, sentno, sentno))
			if dobrackets:
				sent = cgi.escape(sent.replace(" )", " -NONE-)"))
				out = sent.replace(high, "<span class=r>%s</span>" % high)
			else:
				out = ' '.join('<span class=r>%s</span>' % word
						if x in high else word
						for x, word in enumerate(sent.split()))
			yield "<li>#%s [%s] %s" % (str(sentno).rjust(6), link, out)
		yield "</ol>"
	yield '</pre>' if gotresults else 'No matches.'


def brackets(form):
	"""Wrapper."""
	return sents(form, dobrackets=True)


def fragmentsinresults(form, doexport=False):
	"""Extract recurring fragments from search results."""
	if form.get('engine', 'tgrep2') != 'tgrep2':
		yield "Only implemented for TGrep2 queries."
		return
	gotresults = False
	selected = {CORPUS_DIR + TEXTS[n] + EXT[form['engine']]: n for n in
			selectedtexts(form)}
	uniquetrees = set()
	if not doexport:
		url = 'fragments?' + url_encode(dict(export=1, **form))
		yield ('<pre>Query: %s\n'
				'Fragments (showing up to %d fragments '
				'in the first %d search results from selected texts;\n'
				'ordered by (freq ** 0.5 * numwords ** 2) '
				'<a href="%s">Export</a>):\n'
				% (form['query'] if len(form['query']) < 128
					else form['query'][:128] + '...',
					FRAGLIMIT, SENTLIMIT, url))
	for n, (_, _, treestr, _) in enumerate(CORPORA[form.get(
			'engine', 'tgrep2')].sents(form['query'], selected,
			maxresults=SENTLIMIT, brackets=True)):
		if n == 0:
			gotresults = True
		if form.get('engine', 'tgrep2') == 'xpath':
			tree, sent = treebank.alpinotree(treestr)
			treestr = '%s\t%s' % (str(tree), ' '.join(sent))
			# FIXME: enable disc. bracketings in fragment extractor
		else:
			treestr = treestr.replace(" )", " -NONE-)") + '\n'
		uniquetrees.add(treestr.encode('utf8'))
	if not gotresults and not doexport:
		yield "No matches."
		return
	# TODO: get counts from whole text (preload)
	with tempfile.NamedTemporaryFile(delete=True) as tmp:
		tmp.writelines(uniquetrees)
		tmp.flush()
		results, approxcounts = fragments.regular((tmp.name, ), 1, 0, 'utf8')
	results = nlargest(FRAGLIMIT, zip(results, approxcounts),
			key=lambda ff: sum(1 for _
			in re.finditer(r'[^ ()]\)', ff[0])) ** 2 * ff[1] ** 0.5)
	#results = [(frag, freq) for frag, freq in results
	#		if (2 * frag.count(')')
	#			- frag.count(' (')
	#			- frag.count(' )')) > MINNODES and freq > MINFREQ]
	gotresults = False
	if not doexport:
		yield "<ol>"
	for treestr, freq in results:
		gotresults = True
		if doexport:
			yield '%s\t%s\n' % (treestr, freq)
		else:
			link = "<a href='/draw?tree=%s'>draw</a>" % (
					quote(treestr.encode('utf8')))
			treestr = GETLEAVES.sub(' <font color=red>\\1</font>',
					cgi.escape(treestr))
			treestr = GETFRONTIERNTS.sub('(<font color=blue>\\1</font> )',
					treestr)
			yield "<li>freq=%3d [%s] %s" % (freq, link, treestr)
	if not doexport:
		yield "</ol>"
		if gotresults:
			yield '</pre>'
		else:
			yield "No fragments with freq > %d & nodes > %d." % (
					MINNODES, MINFREQ)


@APP.route('/style')
def style():
	"""Use style(1) program to get staticstics for each text."""
	def generate():
		"""Generate plots from results."""
		if not glob.glob(os.path.join(CORPUS_DIR, '*.txt')):
			yield ("No .txt files found in corpus/\n"
					"Using sentences extracted from parse trees.\n"
					"Supply text files with original formatting\n"
					"to get meaningful paragraph information.\n\n")
		yield '<a href="style?export">Export to CSV</a>'
		# produce a plot for each field
		fields = ()
		for a in STYLETABLE:
			fields = sorted(STYLETABLE[a].keys())
			break
		for field in fields:
			data = {a: STYLETABLE[a].get(field, 0) for a in STYLETABLE}
			total = max(data.values())
			if total > 0:
				yield barplot(data, total, field + ':',
						unit='%' if '%' in field else '')

	def generatecsv():
		"""Generate CSV file."""
		tmp = io.BytesIO()
		keys = sorted(next(iter(STYLETABLE.values()))) if STYLETABLE else []
		writer = csv.writer(tmp)
		writer.writerow(['text'] + keys)
		writer.writerows([name] + [row[key] for key in keys]
				for name, row in sorted(STYLETABLE.items()))
		return tmp.getvalue()

	if 'export' in request.args:
		resp = Response(generatecsv(), mimetype='text/plain')
		resp.headers['Content-Disposition'] = 'attachment; filename=style.csv'
	else:
		resp = Response(stream_template('searchresults.html',
				form=request.args, texts=TEXTS,
				selectedtexts=selectedtexts(request.args), output='style',
				results=generate(), havexpath=ALPINOCORPUSLIB))
	resp.headers['Cache-Control'] = 'max-age=604800, public'
	#set Expires one day ahead (according to server time)
	resp.headers['Expires'] = (
		datetime.utcnow() + timedelta(7, 0)).strftime(
		'%a, %d %b %Y %H:%M:%S UTC')
	return resp


@APP.route('/draw')
def draw():
	"""Produce a visualization of a tree on a separate page."""
	if 'tree' in request.args:
		return "<pre>%s</pre>" % DrawTree(request.args['tree']).text(
				unicodelines=True, html=True)
	textno, sentno = int(request.args['text']), int(request.args['sent'])
	nofunc = 'nofunc' in request.args
	nomorph = 'nomorph' in request.args
	filename = os.path.join(CORPUS_DIR, TEXTS[textno] + '.mrg')
	if os.path.exists(filename):
		treestr = next(islice(open(filename),
				sentno - 1, sentno)).decode('utf8')
		result = DrawTree(filterlabels(treestr, nofunc, nomorph)).text(
					unicodelines=True, html=True)
	elif ALPINOCORPUSLIB:
		filename = CORPUS_DIR + TEXTS[textno] + '.dact'
		sentid = CORPORA['xpath'].ids[filename][sentno]
		treestr = CORPORA['xpath'].files[filename].read(sentid)
		tree, sent = treebank.alpinotree(
				ElementTree.fromstring(treestr),
				functions=None if nofunc else 'add',
				morphology=None if nomorph else 'replace')
		result = DrawTree(tree, sent).text(unicodelines=True, html=True)
	else:
		raise ValueError
	return '<pre id="t%s">%s</pre>' % (sentno, result)


@APP.route('/browsesents')
def browsesents():
	"""Browse through sentences in a file."""
	chunk = 20  # number of sentences per page
	if 'text' in request.args and 'sent' in request.args:
		textno = int(request.args['text'])
		sentno = int(request.args['sent']) - 1
		highlight = int(request.args.get('highlight', 0)) - 1
		start = max(0, sentno - chunk // 2)
		maxtree = min(start + chunk, NUMSENTS[textno])
		filename = os.path.join(CORPUS_DIR, TEXTS[textno] + '.mrg')
		if os.path.exists(filename):
			results = [' '.join(GETLEAVES.findall(a)) for a
					in islice(io.open(filename, encoding='utf8'),
					start, maxtree)]
		elif ALPINOCORPUSLIB:
			filename = CORPUS_DIR + TEXTS[textno] + '.dact'
			results = [ALPINOLEAVES.search(
					CORPORA['xpath'].files[filename].read(
						CORPORA['xpath'].ids[filename][n + 1])).group(1)
					for n in range(start, maxtree)]
		else:
			raise ValueError
		results = [('<font color=red>%s</font>' % cgi.escape(a))
				if n == highlight else cgi.escape(a)
				for n, a in enumerate(results, start)]
		prevlink = '<a id=prev>prev</a>'
		if sentno > chunk:
			prevlink = '<a href="browsesents?text=%d&sent=%d" id=prev>prev</a>' % (
					textno, sentno - chunk + 1)
		nextlink = '<a id=next>next</a>'
		if sentno < NUMSENTS[textno] - chunk:
			nextlink = '<a href="browsesents?text=%d&sent=%d" id=next>next</a>' % (
					textno, sentno + chunk + 1)
		return render_template('browsesents.html', textno=textno,
				sentno=sentno + 1, text=TEXTS[textno],
				totalsents=NUMSENTS[textno], sents=results, prevlink=prevlink,
				nextlink=nextlink, chunk=chunk, mintree=start + 1,
				maxtree=maxtree)
	return '<h1>Browse through sentences</h1>\n<ol>\n%s</ol>\n' % '\n'.join(
			'<li><a href="browsesents?text=%d&sent=1&nomorph">%s</a> '
			'(%d sentences)' % (n, text, NUMSENTS[n])
			for n, text in enumerate(TEXTS))


@APP.route('/browse')
def browse():
	"""Browse through trees in a file."""
	chunk = 20  # number of trees to fetch for one request
	if 'text' in request.args and 'sent' in request.args:
		textno = int(request.args['text'])
		sentno = int(request.args['sent']) - 1
		start = max(0, sentno - sentno % chunk)
		maxtree = min(start + chunk, NUMSENTS[textno])
		nofunc = 'nofunc' in request.args
		nomorph = 'nomorph' in request.args
		filename = os.path.join(CORPUS_DIR, TEXTS[textno] + '.mrg')
		if ALPINOCORPUSLIB:
			filename = CORPUS_DIR + TEXTS[textno] + '.dact'
			drawntrees = [DrawTree(*treebank.alpinotree(
					ElementTree.fromstring(CORPORA['xpath'].files[
						filename].read(CORPORA['xpath'].ids[filename][n + 1])),
					functions=None if nofunc else 'add',
					morphology=None if nomorph else 'replace')).text(
					unicodelines=True, html=True)
					for n in range(start, maxtree)]
		elif os.path.exists(filename):
			drawntrees = [DrawTree(filterlabels(
					line.decode('utf8'), nofunc, nomorph)).text(
					unicodelines=True, html=True)
					for line in islice(open(filename), start, maxtree)]
		else:
			raise ValueError
		results = ['<pre id="t%s"%s>%s</pre>' % (n + 1,
				' style="display: none; "' if 'ajax' in request.args else '',
				tree) for n, tree in enumerate(drawntrees, start)]
		if 'ajax' in request.args:
			return '\n'.join(results)

		prevlink = '<a id=prev>prev</a>'
		if sentno > chunk:
			prevlink = '<a href="browse?text=%d&sent=%d" id=prev>prev</a>' % (
					textno, sentno - chunk + 1)
		nextlink = '<a id=next>next</a>'
		if sentno < NUMSENTS[textno] - chunk:
			nextlink = '<a href="browse?text=%d&sent=%d" id=next>next</a>' % (
					textno, sentno + chunk + 1)
		return render_template('browse.html', textno=textno, sentno=sentno + 1,
				text=TEXTS[textno], totalsents=NUMSENTS[textno], trees=results,
				prevlink=prevlink, nextlink=nextlink, chunk=chunk,
				nofunc=nofunc, nomorph=nomorph,
				mintree=start + 1, maxtree=maxtree)
	return '<h1>Browse through trees</h1>\n<ol>\n%s</ol>\n' % '\n'.join(
			'<li><a href="browse?text=%d&sent=1&nomorph">%s</a> '
			'(%d sentences)' % (n, text, NUMSENTS[n])
			for n, text in enumerate(TEXTS))


@APP.route('/favicon.ico')
def favicon():
	"""Serve the favicon."""
	return send_from_directory(os.path.join(APP.root_path, 'static'),
			'treesearch.ico', mimetype='image/vnd.microsoft.icon')


def barplot(data, total, title, width=800.0, unit='', dosort=True):
	"""A HTML bar plot given a dictionary and max value."""
	result = ['<div class=barplot>',
			('<text style="font-family: sans-serif; font-size: 16px; ">'
			'%s</text>' % title)]
	firstletters = {key[0] for key in data}
	color = {}
	if len(firstletters) <= 5:
		color.update(zip(firstletters, range(1, 6)))
	keys = sorted(data, key=data.get, reverse=True) if dosort else data
	for key in keys:
		result.append('<br><div style="width:%dpx;" class=b%d></div>'
				'<span>%s: %g %s</span>' % (round(width * data[key] / total),
				color.get(key[0], 1) if data[key] else 0,
				cgi.escape(key), data[key], unit,))
	result.append('</div>\n')
	return '\n'.join(result)


def concplot(indices, total, width=800.0):
	"""Draw a concordance plot from a list of indices.

	:param indices: a list of sets or Counter objects, where each element is
		a sentence number. Each element of indices will be drawn in a
		different color.
	:param total: the total number of sentences."""
	result = ('\t<svg version="1.1" xmlns="http://www.w3.org/2000/svg"'
			' width="%dpx" height="10px" >\n'
			'<rect x=0 y=0 width="%dpx" height=10 '
			'fill=white stroke=black />\n' % (width, width))
	for n, a in enumerate(indices if isinstance(indices, list) else [indices]):
		if not a:
			continue
		strokes = []
		start = 0
		seq = [-1] + sorted(a) + [-1]
		for prev, idx, nextidx in zip(seq, seq[1:], seq[2:]):
			if idx != prev + 1 and idx != nextidx - 1:
				strokes.append('M %d 0v 10' % idx)
			elif idx != prev + 1:
				start = idx
			elif idx != nextidx - 1:
				strokes.append(  # draw a rectangle covering start:idx
						'M %d 0l 0 10 %d 0 0 -10' % (start, idx - start))
		result += ('<g transform="scale(%g, 1)">\n'
				'<path stroke=%s d="%s" /></g>' % (
				width / total, COLORS.get(n, 'black'), ''.join(strokes)))
	return result + '</svg>'


def selectedtexts(form):
	"""Find available texts and parse selected texts argument."""
	selected = set()
	if 'texts' in form:
		for a in filter(None, form['texts'].replace('.', ',').split(',')):
			if '-' in a:
				b, c = a.split('-')
				selected.update(n for n in range(int(b), int(c)))
			else:
				selected.add(int(a))
	else:
		selected.update(range(len(TEXTS)))
	return selected


def tokenized(text):
	"""Return iterable with tokenized sentences of a text, one sentence at a
	time, in the form of a byte string (with newline)."""
	base = EXTRE.sub('', text)
	if os.path.exists(base + '.tok'):
		return open(base + '.tok')
	elif os.path.exists(base + '.mrg.t2c.gz'):
		tgrep = subprocess.Popen(
				args=[which('tgrep2'), '-t', '-c', base + '.mrg.t2c.gz', '*'],
				shell=False, bufsize=-1, stdout=subprocess.PIPE)
		return tgrep.stdout
	elif os.path.exists(base + '.mrg'):
		return (' '.join(GETLEAVES.findall(line)) + '\n'
				for line in open(base + '.mrg'))
	elif os.path.exists(base + '.dact'):
		# may be in arbitrary order, so sort
		result = {entry.name(): ALPINOLEAVES.search(
				entry.contents()).group(1) + '\n' for entry
				in alpinocorpus.CorpusReader(base + '.dact').entries()}
		return [result[a] for a in sorted(result, key=treebank.numbase)]
	raise ValueError('no file found for %s' % text)


def getstyletable(texts):
	"""Run style(1) on all files and store results in a dictionary."""
	files = glob.glob(os.path.join(CORPUS_DIR, '*.txt'))
	if not files:
		files = [os.path.join(CORPUS_DIR, a) for a in texts]
	styletable = {}
	for filename in sorted(files):
		cmd = [which('style'), '--language', STYLELANG]
		stdin = subprocess.PIPE
		proc = subprocess.Popen(args=cmd, shell=False, bufsize=-1,
				stdin=stdin, stdout=subprocess.PIPE,
				stderr=subprocess.PIPE)
		try:
			if filename.endswith('.txt'):
				# .txt files may have one paragraph per line;
				# style expects paragraphs separated by two linebreaks.
				proc.stdin.write(open(filename).read().replace('\n', '\n\n'))
			else:
				proc.stdin.writelines(tokenized(filename))
		except IOError as err:
			APP.logger.error('%s\n%s', err, proc.stderr.read())
			return {}
		proc.stdin.close()
		out = proc.stdout.read()
		proc.stdout.close()
		proc.wait()
		name = os.path.basename(filename)
		styletable[name] = parsestyleoutput(out)
	return styletable


def parsestyleoutput(out):
	"""Extract readability grades, averages, and percentages from style output
	(i.e., all figures except for absolute counts)."""
	result = {}
	for key, val in READGRADERE.findall(out):
		result[key.strip()] = float(val)
	for key1, val, key2 in AVERAGERE.findall(out):
		result['average %s per %s' % (key2, key1[:-1])] = float(val)
	m = re.search(r'([0-9]+(?:\.[0-9]+)?) syllables', out)
	if m:
		result['average syllables per word'] = float(m.group(1))
	for key, val in PERCENTAGE1RE.findall(out):
		result['%% %s' % key.strip()] = float(val)
	for val, key in PERCENTAGE2RE.findall(out):
		result['%% %s' % key.strip()] = float(val)
	return result


def getcorpus():
	"""Get list of files and number of lines in them."""
	texts = []
	corpora = {}
	if os.path.exists('/tmp/treesearchcorpus.pickle'):
		try:
			texts, numsents, numconst, numwords, styletable, ids = pickle.load(
					open('/tmp/treesearchcorpus.pickle'))
		except ValueError:
			pass
	tfiles = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.mrg')))
	afiles = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.dact')))
	tokfiles = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.tok')))
	if not tokfiles:
		# extract tokenized sentences from trees
		for filename in tfiles or afiles:
			newfile = EXTRE.sub('.tok', filename)
			converted = tokenized(filename)
			with open(newfile, 'w') as out:
				out.writelines(converted)
		tokfiles = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.tok')))

	# FIXME: only reload corpora if necessary here?
	# TODO: make trees optional, accept .txt and tokenize into .tok
	if tfiles:
		corpora['tgrep2'] = TgrepSearcher(tfiles, 'static/tgrepmacros.txt')
	if afiles and ALPINOCORPUSLIB:
		corpora['xpath'] = DactSearcher(afiles, 'static/xpathmacros.txt')
	if tokfiles:
		corpora['regex'] = RegexSearcher(tokfiles)

	assert tfiles or afiles, 'no .mrg or .dact files found in %s' % CORPUS_DIR
	if tfiles and afiles:
		assert len(tfiles) == len(afiles) and all(
				t.rsplit('.', 1)[0] == a.rsplit('.', 1)[0]
				for a, t in zip(tfiles, afiles)), (
				'expected either .mrg or .dact files, '
				'or corresponding .mrg and .dact files')
	picklemtime = 0
	if os.path.exists('/tmp/treesearchcorpus.pickle'):
		picklemtime = os.stat('/tmp/treesearchcorpus.pickle').st_mtime
	currentfiles = {os.path.splitext(os.path.basename(filename))[0]
		for filename in tfiles + afiles}
	if (set(texts) != currentfiles or
			any(os.stat(a).st_mtime > picklemtime for a in tfiles + afiles)):
		if tfiles:
			numsents = [len(open(filename).readlines())
					for filename in tfiles if filename.endswith('.mrg')]
			numconst = [open(filename).read().count('(')
					for filename in tfiles if filename.endswith('.mrg')]
			numwords = [len(GETLEAVES.findall(open(filename).read()))
					for filename in tfiles if filename.endswith('.mrg')]
		elif ALPINOCORPUSLIB and afiles:
			numsents = [corpus.size() for corpus
					in corpora['xpath'].files.values()]
			numconst, numwords = [], []
			for filename in afiles:
				tmp = alpinocorpus.CorpusReader(filename)
				const = words = 0
				for entry in tmp.entries():
					const += entry.contents().count('<node ')
					words += entry.contents().count('word=')
				numconst.append(const)
				numwords.append(words)
				print(filename)
		else:
			raise ValueError
		if ALPINOCORPUSLIB:
			ids = {}
			for filename in afiles:
				# FIXME: memory leak here?
				tmp = alpinocorpus.CorpusReader(filename)
				ids[filename] = [None] + sorted((entry.name() for entry
						in tmp.entries()), key=treebank.numbase)
		texts = [os.path.splitext(os.path.basename(a))[0]
				for a in tfiles or afiles]
		styletable = getstyletable(texts)
	for filename in afiles:
		corpora['xpath'].updateindex(filename, ids[filename])
	pickle.dump((texts, numsents, numconst, numwords, styletable, ids),
			open('/tmp/treesearchcorpus.pickle', 'wb'), protocol=-1)
	return texts, numsents, numconst, numwords, styletable, corpora


def stream_template(template_name, **context):
	"""Pass an iterator to a template; from Flask documentation."""
	APP.update_template_context(context)
	templ = APP.jinja_env.get_template(template_name)
	result = templ.stream(context)
	result.enable_buffering(5)
	return result


def which(program):
	"""Return first match for program in search path."""
	for path in os.environ.get('PATH', os.defpath).split(":"):
		if path and os.path.exists(os.path.join(path, program)):
			return os.path.join(path, program)
	raise ValueError('%r not found in path; please install it.' % program)


fragments.PARAMS.update(disc=False, debug=False, cover=False, complete=False,
		quadratic=False, complement=False, quiet=True, nofreq=False,
		approx=True, indices=False, fmt='bracket')

# this is redundant but used to support both javascript-enabled /foo
# as well as non-javascript fallback /?output=foo
DISPATCH = {
	'counts': counts,
	'trees': trees,
	'sents': sents,
	'brackets': brackets,
	'fragments': fragmentsinresults,
}
logging.basicConfig()
for log in (logging.getLogger(), APP.logger):
	log.setLevel(logging.DEBUG)
	log.handlers[0].setFormatter(logging.Formatter(
			fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
TEXTS, NUMSENTS, NUMCONST, NUMWORDS, STYLETABLE, CORPORA = getcorpus()
ALPINOCORPUSLIB = ALPINOCORPUSLIB and CORPORA.get('xpath')


if __name__ == '__main__':
	APP.run(debug=True, host='0.0.0.0')
