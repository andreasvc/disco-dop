"""Web interface to search a treebank. Requires Flask, tgrep2
or alpinocorpus-python (for xpath queries), style. Expects one or more
treebanks with .mrg or .dact extension in the directory corpus/"""
# stdlib
import io
import os
import re
import cgi
import csv
import json
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
from discodop.parser import which
from discodop.treesearch import TgrepSearcher, DactSearcher, RegexSearcher, \
		filterlabels

MINFREQ = 2  # filter out fragments which occur just once or twice
MINNODES = 3  # filter out fragments with only three nodes (CFG productions)
TREELIMIT = 10  # max number of trees to draw in search resuluts
FRAGLIMIT = 250  # max amount of search results for fragment extraction
SENTLIMIT = 1000  # max number of sents/brackets in search results
LANG = 'nl'  # language to use when running style(1) or ucto(1)
CORPUS_DIR = "corpus/"

APP = Flask(__name__)

MORPH_TAGS = re.compile(
		r'([_/*A-Z0-9]+)(?:\[[^ ]*\][0-9]?)?((?:-[_A-Z0-9]+)?(?:\*[0-9]+)? )')
FUNC_TAGS = re.compile(r'-[_A-Z0-9]+')
GETLEAVES = re.compile(r' ([^ ()]+)(?=[ )])')
GETFRONTIERNTS = re.compile(r"\(([^ ()]+) \)")
# the extensions for corpus files for each query engine:
EXTRE = re.compile(r'\.(?:mrg(?:\.t2c\.gz)?|dact|txt)$')
EXT = {
		'tgrep2': '.mrg.t2c.gz',
		'xpath': '.dact',
		'regex': '.tok'
	}
COLORS = dict(enumerate(
		'black red orange blue green turquoise slategray peru teal'.split()))


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
				havexpath='xpath' in CORPORA, havetgrep='tgrep2' in CORPORA)
		# To send results incrementally:
		#return Response(stream_template('searchresults.html',
		#		form=request.args, texts=TEXTS, selectedtexts=selected,
		#		output=output, results=DISPATCH[output](request.args),
		#		havexpath='xpath' in CORPORA))
	return render_template('search.html', form=request.args, output='counts',
			texts=TEXTS, selectedtexts=selected,
			havexpath='xpath' in CORPORA, havetgrep='tgrep2' in CORPORA)


def export(form, output):
	"""Export search results to a file for download."""
	# NB: no distinction between trees from different texts
	selected = {CORPUS_DIR + TEXTS[n] + EXT[form['engine']]: n for n in
			selectedtexts(request.args)}
	if output == 'counts':
		results = counts(form, doexport=True)
		if form.get('export') == 'json':
			resp = Response(results, mimetype='application/json')
			return resp
		filename = 'counts.csv'
	elif output == 'fragments':
		results = fragmentsinresults(form, doexport=True)
		filename = 'fragments.txt'
	elif output in ('sents', 'brackets', 'trees'):
		if form.get('engine') == 'xpath' and output != 'sents':
			fmt = '<!-- %s:%s -->\n%s\n\n'  # hack
		else:
			fmt = '%s:%s|%s\n'
		results = CORPORA[form.get('engine', 'tgrep2')].sents(
					form['query'], selected, maxresults=SENTLIMIT,
					brackets=output in ('brackets', 'trees'))
		if form.get('export') == 'json':
			return Response(json.dumps(results, cls=JsonSetEncoder, indent=2),
					mimetype='application/json')
		results = ((fmt % (a[0], a[1], (a[2]))
				if form.get('linenos') else (a[2] + '\n')).encode('utf-8')
				for a in results)
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
	# TODO: option to arrange graphs by text instead of by query
	norm = form.get('norm', 'sents')
	selected = {CORPUS_DIR + TEXTS[n] + EXT[form['engine']]: n for n in
			selectedtexts(form)}
	if not doexport:
		url = 'counts?' + url_encode(dict(export='csv', **form))
		yield ('Counts from queries '
				'(<a href="%s">export to CSV</a>):\n' % url)
	if norm == 'query':
		normresults = CORPORA[form.get('engine', 'tgrep2')].counts(
				form['normquery'], selected)
	combined = defaultdict(Counter)
	combined1 = defaultdict(list)
	index = [TEXTS[n] for n in selected.values()]
	df = pandas.DataFrame(index=index)
	queries = querydict(form['query'])
	if not doexport:
		yield '<ol>%s</ol>\n' % '\n'.join(
				'<li><a href="#q%d">%s</a>' % (n, query)
				for n, query in enumerate(list(queries) + [
					'Combined results', 'Overview'], 1))
	for n, (name, (normquery, query)) in enumerate(
			list(queries.iteritems()) + [('Combined results', ('', None))], 1):
		cnts = Counter()
		sumtotal = 0
		relfreq = {}
		if query is None:
			if len(df.columns) == 1:
				break
			results = combined1
			legend = '%sLegend:\t%s' % (64 * ' ', '\t'.join(
					'<font color=%s>%s</font>' % (COLORS.get(n, 'black'), query)
					for n, query in enumerate(queries)))
		else:
			legend = ''
			if normquery:
				normquery, query = query.split('\t', 1)
				norm = 'query'
				normresults = CORPORA[form.get('engine', 'tgrep2')].counts(
						normquery, selected)
			else:
				norm = form.get('norm', 'sents')
			results = CORPORA[form.get('engine', 'tgrep2')].counts(
					query, selected, indices=True)
		if not doexport:
			yield '<a name=q%d><h3>%s</h3></a>\n<pre>\n%s\n' % (
					n, name, query or legend)
		for filename, indices in sorted(results.items()):
			if query is None:
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
				out = '%s (<a href="browsesents?%s">browse</a>)    %5d %5.2f %%' % (
						text.ljust(40)[:40],
						url_encode(dict(text=textno, sent=1,
							query=query or form['query'],
							engine=form.get('engine', 'tgrep2'))),
						cnt, relfreq[text])
				plot = concplot(indices, limit or NUMSENTS[textno])
				if cnt:
					yield out + plot + '\n'
				else:
					yield '<span style="color: gray; ">%s%s</span>\n' % (
							out, plot)
		if not doexport or query is not None:
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
		if form.get('export') == 'json':
			yield json.dumps(df.to_dict(), indent=2)
		else:
			with io.BytesIO() as tmp:
				df.to_csv(tmp)
				yield tmp.getvalue()
	else:
		fmt = lambda x: '%g' % round(x, 1)
		yield '<h3><a name=q%d>Overview of patterns</a></h3>\n' % (
				len(queries) + 2)
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
	url = 'trees?' + url_encode(dict(export='csv', **form))
	yield ('<pre>Query: %s\n'
			'Trees (showing up to %d per text; '
			'export: <a href="%s">plain</a>, '
			'<a href="%s">with line numbers</a>):\n' % (
				form['query'] if len(form['query']) < 128
				else form['query'][:128] + '...',
				TREELIMIT, url, url + '&linenos=1'))
	for n, (filename, results) in enumerate(groupby(sorted(
			CORPORA[form.get('engine', 'tgrep2')].trees(form['query'],
			selected, maxresults=TREELIMIT, nomorph='nomorph' in form,
			nofunc='nofunc' in form)), itemgetter(0))):
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
			url_encode(dict(export='csv', **form)))
	yield ('<pre>Query: %s\n'
			'Sentences (showing up to %d per text; '
			'export: <a href="%s">plain</a>, '
			'<a href="%s">with line numbers</a>):\n' % (
				form['query'] if len(form['query']) < 128
				else form['query'][:128] + '...',
				SENTLIMIT, url, url + '&linenos=1'))
	for n, (filename, results) in enumerate(groupby(sorted(
			CORPORA[form.get('engine', 'tgrep2')].sents(form['query'],
				selected, maxresults=SENTLIMIT, brackets=dobrackets)),
			itemgetter(0))):
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
	if form.get('engine', 'tgrep2') == 'regex':
		yield "Only implemented for tgrep2 and xpath queries."
		return
	gotresults = False
	selected = {CORPUS_DIR + TEXTS[n] + EXT[form['engine']]: n for n in
			selectedtexts(form)}
	uniquetrees = set()
	if not doexport:
		url = 'fragments?' + url_encode(dict(export='csv', **form))
		yield ('<pre>Query: %s\n'
				'Fragments (showing up to %d fragments '
				'in the first %d search results from selected texts;\n'
				'ordered by (freq ** 0.5 * numwords ** 2) '
				'<a href="%s">Export</a>):\n'
				% (form['query'] if len(form['query']) < 128
					else form['query'][:128] + '...',
					FRAGLIMIT, SENTLIMIT, url))
	disc = form.get('engine', 'tgrep2') == 'xpath'
	if disc:
		fragments.PARAMS.update(disc=True, fmt='discbracket')
	else:
		fragments.PARAMS.update(disc=False, fmt='bracket')
	for n, (_, _, treestr, _) in enumerate(CORPORA[form.get(
			'engine', 'tgrep2')].sents(form['query'], selected,
			maxresults=SENTLIMIT, brackets=True)):
		if n == 0:
			gotresults = True
		if disc:
			tree, sent = treebank.alpinotree(
					ElementTree.fromstring(treestr.encode('utf-8')))
			line = '%s\t%s\n' % (str(tree), ' '.join(sent))
		else:
			line = treestr.replace(" )", " -NONE-)") + '\n'
		uniquetrees.add(line.encode('utf8'))
	if not gotresults and not doexport:
		yield "No matches."
		return
	# TODO: get counts from whole text (preload)
	with tempfile.NamedTemporaryFile(delete=True) as tmp:
		tmp.writelines(uniquetrees)
		tmp.flush()
		results, approxcounts = fragments.regular([tmp.name], 1, None, 'utf8')
	if disc:
		results = nlargest(FRAGLIMIT, zip(results, approxcounts), key=lambda ff:
				sum(1 for a in ff[0][1] if a) ** 2 * ff[1] ** 0.5)
	else:
		results = nlargest(FRAGLIMIT, zip(results, approxcounts),
				key=lambda ff: sum(1 for _
				in re.finditer(r'[^ ()]\)', ff[0])) ** 2 * ff[1] ** 0.5)
	gotresults = False
	if not doexport:
		yield "<ol>"
	for tree, freq in results:
		gotresults = True
		if disc:
			tree, sent = tree
			sent = ' '.join(a or '' for a in sent)
		if doexport:
			if disc:
				yield '%s\t%s\t%s\n' % (tree, sent, freq)
			else:
				yield '%s\t%s\n' % (tree, freq)
		else:
			if disc:
				link = "<a href='/draw?tree=%s&sent=%s'>draw</a>" % (
						quote(tree.encode('utf8')), quote(sent.encode('utf8')))
				sent = GETLEAVES.sub(' <font color=red>\\1</font>',
						cgi.escape(' ' + sent + ' '))
				tree = cgi.escape(tree) + ' ' + sent
			else:
				link = "<a href='/draw?tree=%s'>draw</a>" % (
						quote(tree.encode('utf8')))
				tree = GETLEAVES.sub(' <font color=red>\\1</font>',
						cgi.escape(tree))
			tree = GETFRONTIERNTS.sub('(<font color=blue>\\1</font> )', tree)
			yield "<li>freq=%3d [%s] %s" % (freq, link, tree)
	if not doexport:
		yield "</ol>"
		if gotresults:
			yield '</pre>'
		else:
			yield "No fragments with freq > %d & nodes > %d." % (
					MINNODES, MINFREQ)


@APP.route('/style')
def style():
	"""Show simple surface characteristics of texts."""
	def generate():
		"""Generate plots from results."""
		if not glob.glob(os.path.join(CORPUS_DIR, '*.txt')):
			yield ("No .txt files found in corpus/\n"
					"Using sentences extracted from parse trees.\n"
					"Supply text files with original formatting\n"
					"to get meaningful paragraph information.\n\n")
		yield '<a href="style?export=csv">Export to CSV</a><br>'
		yield 'Results based on first %d sentences.' % min(NUMSENTS)

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

	if request.args.get('export') == 'csv':
		resp = Response(generatecsv(), mimetype='text/plain')
		resp.headers['Content-Disposition'] = 'attachment; filename=style.csv'
	elif request.args.get('export') == 'json':
		resp = Response(json.dumps(STYLETABLE, indent=2),
				mimetype='application/json')
	else:
		resp = Response(stream_template('searchresults.html',
				form=request.args, texts=TEXTS,
				selectedtexts=selectedtexts(request.args), output='style',
				results=generate(), havexpath='xpath' in CORPORA))
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
		return "<pre>%s</pre>" % DrawTree(request.args['tree'],
				[a or None for a in request.args['sent'].split(' ')]
				if 'sent' in request.args else None).text(
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
	elif 'xpath' in CORPORA:
		filename = CORPUS_DIR + TEXTS[textno] + '.dact'
		sentid = '%d' % sentno
		treestr = CORPORA['xpath'].files[filename].read(sentid)
		tree, sent = treebank.alpinotree(
				ElementTree.fromstring(treestr),
				functions=None if nofunc else 'add',
				morphology=None if nomorph else 'replace')
		result = DrawTree(tree, sent).text(unicodelines=True, html=True)
	else:
		raise ValueError('no treebank available for "%s".' % TEXTS[textno])
	return '<pre id="t%s">%s</pre>' % (sentno, result)


@APP.route('/browsesents')
def browsesents():
	"""Browse through sentences in a file; optionally highlight matches."""
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
		elif 'xpath' in CORPORA:
			filename = CORPUS_DIR + TEXTS[textno] + '.dact'
			results = [ElementTree.fromstring(
					CORPORA['xpath'].files[filename].read('%8d' % (n + 1))
					).find('sentence').text.split()
					for n in range(start, maxtree)]
		else:
			raise ValueError('no treebank available for "%s".' % TEXTS[textno])
		results = [('<font color=red>%s</font>' % cgi.escape(a))
				if n == highlight else cgi.escape(a)
				for n, a in enumerate(results, start)]
		legend = queryparams = ''
		if request.args.get('query', ''):
			queryparams = '&' + url_encode(dict(
					query=request.args['query'],
					engine=request.args.get('engine', 'tgrep2')))
			filename = CORPUS_DIR + TEXTS[textno] + EXT[
					request.args.get('engine', 'tgrep2')]
			queries = querydict(request.args['query'])
			legend = 'Legend:\t%s' % ('\t'.join(
					'<font color=%s>%s</font>' % (COLORS.get(n, 'gray'), query)
					for n, query in enumerate(queries, 1)))
			for n, (_, query) in enumerate(queries.values()):
				matches = CORPORA[request.args['engine']].sents(
						query, subset=(filename,), maxresults=None)
				for _, m, sent, high in matches:
					if start <= m < maxtree:
						sent = sent.split()
						match = ' '.join(sent[a] for a in high)
						results[m - start - 1] = results[m - start - 1].replace(
								match, '<font color=%s>%s</font>' % (
								COLORS.get(n + 1, 'gray'), cgi.escape(match)))
					elif m > maxtree:
						break
		prevlink = '<a id=prev>prev</a>'
		if sentno > chunk:
			prevlink = ('<a href="browsesents?text=%d&sent=%d%s" id=prev>'
					'prev</a>' % (textno, sentno - chunk + 1, queryparams))
		nextlink = '<a id=next>next</a>'
		if sentno < NUMSENTS[textno] - chunk:
			nextlink = ('<a href="browsesents?text=%d&sent=%d%s" id=next>'
					'next</a>' % (textno, sentno + chunk + 1, queryparams))
		return render_template('browsesents.html', textno=textno,
				sentno=sentno + 1, text=TEXTS[textno],
				totalsents=NUMSENTS[textno], sents=results, prevlink=prevlink,
				nextlink=nextlink, chunk=chunk, mintree=start + 1,
				maxtree=maxtree, legend=legend,
				query=request.args.get('query', ''),
				engine=request.args.get('engine', ''))
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
		if 'xpath' in CORPORA:
			filename = CORPUS_DIR + TEXTS[textno] + '.dact'
			drawntrees = [DrawTree(*treebank.alpinotree(
					ElementTree.fromstring(CORPORA['xpath'].files[
						filename].read('%8d' % (n + 1))),
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
			raise ValueError('no treebank available for "%s".' % TEXTS[textno])
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
				'<span>%s: %g %s</span>' % (
				round(width * data[key] / total) if data[key] else 0,
				color.get(key[0], 1) if data[key] else 0,
				cgi.escape(key), data[key], unit,))
	result.append('</div>\n')
	return '\n'.join(result)


def concplot(indices, total, width=800.0, runle=False):
	"""Draw a concordance plot from a list of indices.

	:param indices: a list of sets or Counter objects, where each element is
		a sentence number. Each element of indices will be drawn in a
		different color.
	:param total: the total number of sentences.
	:param runle: use a more compact, run-length encoded representation."""
	result = ('\t<svg version="1.1" xmlns="http://www.w3.org/2000/svg"'
			' width="%dpx" height="10px" >\n'
			'<rect x=0 y=0 width="%dpx" height=10 '
			'fill=white stroke=black />\n' % (width, width))
	for n, a in enumerate(indices if isinstance(indices, list) else [indices]):
		if not a:
			continue
		if runle:
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
							'M %d 0l 0 10 %d 0 0 -10' % (
								start - 1, idx - start - 1))
		else:
			strokes = ['M %d 0v 10' % (idx - 1) for idx in sorted(a)]
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


def querydict(queries):
	"""Return an OrderedDict of names and queries.

	name is abbreviated query if not given."""
	result = OrderedDict()
	for line in (x for x in queries.splitlines() if x.strip()):
		if ':' in line:
			name, query = line.split(':', 1)
		else:
			name = line[:100] + ('' if len(line) < 100 else '...')
			query = line
		if '\t' in query:
			normquery, query = query.split('\t')
		else:
			normquery, query = None, query
			result[name] = normquery, query
	return result


def tokenize(filename):
	"""Create a tokenized copy of a text, one sentence per line."""
	base = EXTRE.sub('', filename)
	try:
		ucto = which('ucto')
	except ValueError:
		ucto = None
	if os.path.exists(base + '.tok'):
		return
	elif ucto and filename.endswith('.txt'):
		newfile = base + '.tok'
		proc = subprocess.Popen(args=[which('ucto'),
				'-L', LANG, '-s', '', '-n',
				filename, newfile], shell=False)
		proc.wait()
		return
	elif os.path.exists(base + '.mrg.t2c.gz'):
		tgrep = subprocess.Popen(
				args=[which('tgrep2'), '-t', '-c', base + '.mrg.t2c.gz', '*'],
				shell=False, bufsize=-1, stdout=subprocess.PIPE)
		converted = tgrep.stdout
	elif os.path.exists(base + '.mrg'):
		converted = (' '.join(GETLEAVES.findall(line)) + '\n'
				for line in open(base + '.mrg'))
	elif os.path.exists(base + '.dact'):
		result = {entry.name(): ElementTree.fromstring(
				entry.contents()).find('sentence').text + '\n' for entry
				in alpinocorpus.CorpusReader(base + '.dact').entries()}
		converted = [result[a] for a in sorted(result, key=treebank.numbase)]
	else:
		raise ValueError('no file found for "%s" and ucto not installed.'
				% filename)
	newfile = EXTRE.sub('.tok', filename)
	with open(newfile, 'w') as out:
		out.writelines(converted)


def getreadabilitymeasures(numsents):
	"""Get readability of all files and store results in a dictionary."""
	try:
		import readability
	except ImportError:
		APP.logger.warning(
			'readability module not found; install with:\npip install'
			' https://github.com/andreasvc/readability/tarball/master')
		return {}
	files = glob.glob(os.path.join(CORPUS_DIR, '*.tok'))
	results = {}
	# consider a fixed number of sentences to get comparable results
	cutoff = min(numsents)
	for filename in sorted(files):
		name = os.path.basename(filename)
		# flatten results into a single dictionary of (key, value) pairs.
		results[name] = {key: value
				for data in readability.getmeasures(
						islice(io.open(filename, encoding='utf-8'), cutoff),
						lang=LANG).values()
					for key, value in data.items()}
	return results


def getcorpus():
	"""Get list of files and number of lines in them."""
	texts = []
	corpora = {}
	picklefile = os.path.join(CORPUS_DIR, 'treesearchcorpus.pickle')
	if os.path.exists(picklefile):
		try:
			texts, numsents, numconst, numwords, styletable = pickle.load(
					open(picklefile))
		except ValueError:
			pass
	tfiles = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.mrg')))
	afiles = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.dact')))
	txtfiles = glob.glob(os.path.join(CORPUS_DIR, '*.txt'))
	# get tokenized sents from trees or ucto
	for filename in tfiles or afiles or txtfiles:
		tokenize(filename)
	tokfiles = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.tok')))
	if tfiles and set(tfiles) != set(corpora.get('tgrep2', ())):
		corpora['tgrep2'] = TgrepSearcher(tfiles, 'static/tgrepmacros.txt')
	if afiles and ALPINOCORPUSLIB and set(afiles) != set(
			corpora.get('xpath', ())):
		corpora['xpath'] = DactSearcher(afiles, 'static/xpathmacros.txt')
	if tokfiles and set(tokfiles) != set(corpora.get('regex', ())):
		corpora['regex'] = RegexSearcher(tokfiles, 'static/regexmacros.txt')

	assert tfiles or afiles or tokfiles, ('no files with extension '
			'.mrg, .dact, or .txt found in %s' % CORPUS_DIR)
	if tfiles and afiles:
		assert len(tfiles) == len(afiles) and all(
				t.rsplit('.', 1)[0] == a.rsplit('.', 1)[0]
				for a, t in zip(tfiles, afiles)), (
				'expected either .mrg or .dact files, '
				'or corresponding .mrg and .dact files')
	picklemtime = 0
	if os.path.exists(picklefile):
		picklemtime = os.stat(picklefile).st_mtime
	currentfiles = {os.path.splitext(os.path.basename(filename))[0]
		for filename in tfiles + afiles + tokfiles}
	if (set(texts) != currentfiles or any(os.stat(a).st_mtime > picklemtime
				for a in tfiles + afiles + tokfiles)):
		if corpora.get('tgrep2'):
			numsents = [len(open(filename).readlines())
					for filename in tfiles if filename.endswith('.mrg')]
			numconst = [open(filename).read().count('(')
					for filename in tfiles if filename.endswith('.mrg')]
			numwords = [len(GETLEAVES.findall(open(filename).read()))
					for filename in tfiles if filename.endswith('.mrg')]
		elif corpora.get('xpath'):
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
		elif corpora.get('regex'):  # only tokenized sentences, no trees
			numsents = [len(open(filename).readlines())
					for filename in tokfiles]
			numwords = [1 + open(filename).read().count(' ')
					for filename in tokfiles]
			numconst = [0 for filename in tokfiles]
		else:
			raise ValueError('no texts found.')
		texts = [os.path.splitext(os.path.basename(a))[0]
				for a in tfiles or afiles or tokfiles]
		styletable = getreadabilitymeasures(numsents)
	pickle.dump((texts, numsents, numconst, numwords, styletable),
			open(picklefile, 'wb'), protocol=-1)
	return texts, numsents, numconst, numwords, styletable, corpora


def stream_template(template_name, **context):
	"""Pass an iterator to a template; from Flask documentation."""
	APP.update_template_context(context)
	templ = APP.jinja_env.get_template(template_name)
	result = templ.stream(context)
	result.enable_buffering(5)
	return result


class JsonSetEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, set):
			return list(obj)
		return json.JSONEncoder.default(self, obj)


fragments.PARAMS.update(quiet=True, debug=False, disc=False, complete=False,
		cover=False, quadratic=False, complement=False, adjacent=False,
		twoterms=False, nofreq=False, approx=True, indices=False,
		fmt='bracket')

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


if __name__ == '__main__':
	APP.run(debug=True, host='0.0.0.0')
