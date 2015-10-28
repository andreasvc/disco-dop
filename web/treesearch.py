"""Web interface to search a treebank. Requires Flask, tgrep2
or alpinocorpus-python (for xpath queries), style. Expects one or more
treebanks with .mrg or .dact extension in the directory corpus/"""
# stdlib
from __future__ import print_function, absolute_import
import io
import os
import re
import cgi
import csv
import sys
import json
import glob
import logging
import tempfile
import subprocess
from heapq import nlargest
from datetime import datetime, timedelta
from itertools import islice, groupby
from operator import itemgetter
from collections import Counter, OrderedDict, defaultdict
if sys.version[0] == '2':
	import cPickle as pickle  # pylint: disable=import-error
	from itertools import ifilter as filter  # pylint: disable=E0611,W0622
	from urllib import quote  # pylint: disable=no-name-in-module
else:
	import pickle
	from urllib.parse import quote  # pylint: disable=F0401,E0611
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
from discodop import treebank, fragments, treesearch
from discodop.tree import Tree, DiscTree, DrawTree
from discodop.util import which

DEBUG = True
MINFREQ = 2  # filter out fragments which occur just once or twice
MINNODES = 3  # filter out fragments with only three nodes (CFG productions)
TREELIMIT = 10  # max number of trees to draw in search resuluts
FRAGLIMIT = 250  # max amount of search results for fragment extraction
SENTLIMIT = 1000  # max number of sents/brackets in search results
INDICESMAXRESULTS = 1024  # max number of results for which to obtain indices.
	# Indices are used to display a dispersion plot.
LANG = 'nl'  # language to use when running style(1) or ucto(1)
CORPUS_DIR = "corpus/"

APP = Flask(__name__)

MORPH_TAGS = re.compile(
		r'([_/*A-Z0-9]+)(?:\[[^ ]*\][0-9]?)?((?:-[_A-Z0-9]+)?(?:\*[0-9]+)? )')
FUNC_TAGS = re.compile(r'-[_A-Z0-9]+')
GETLEAVES = re.compile(r' ([^ ()]+)(?=[ )])')
GETFRONTIERNTS = re.compile(r"\(([^ ()]+) \)")
# the extensions for corpus files for each query engine:
EXTRE = re.compile(r'\.(?:mrg(?:\.t2c\.gz)?|dact|export|dbr|txt|tok)$')

COLORS = dict(enumerate('''\
		Black Red Orange Blue Green Turquoise SlateGray Peru Teal Aqua
		Aquamarine BlanchedAlmond Brown Burlywood CadetBlue Chartreuse
		Chocolate Coral Crimson Cyan Firebrick ForestGreen Fuchsia Gainsboro
		Gold Goldenrod Gray GreenYellow HotPink IndianRed Indigo Khaki Lime
		YellowGreen Magenta Maroon Yellow MidnightBlue Moccasin NavyBlue Olive
		OliveDrab Orchid PapayaWhip Pink Plum PowderBlue Purple RebeccaPurple
		RoyalBlue SaddleBrown Salmon SandyBrown SeaGreen Sienna Silver SkyBlue
		SlateBlue Tan Thistle Tomato Violet Wheat'''.split()))


@APP.route('/')
@APP.route('/counts')
@APP.route('/trees')
@APP.route('/sents')
@APP.route('/brackets')
@APP.route('/fragments')
def main(debug=DEBUG):
	"""Main search form & results page."""
	output = None
	if request.path != '/':
		output = request.path.lstrip('/')
	elif 'output' in request.args:
		output = request.args['output']
	selected = selectedtexts(request.args)
	args = dict(
			form=request.args,
			texts=TEXTS,
			selectedtexts=selected,
			output='counts',
			havetgrep='tgrep2' in CORPORA,
			havexpath='xpath' in CORPORA,
			havefrag='frag' in CORPORA,
			)
	if output:
		if output not in DISPATCH:
			return 'Invalid argument', 404
		elif request.args.get('export'):
			return export(request.args, output)
		args['output'] = output
		args['results'] = DISPATCH[output](request.args)
		if debug:  # For debugging purposes:
			return render_template('searchresults.html', **args)
		else:  # send results incrementally:
			return Response(stream_template('searchresults.html', **args))
	return render_template('search.html', **args)


def export(form, output):
	"""Export search results to a file for download."""
	# NB: no distinction between trees from different texts
	filenames = {EXTRE.sub('', os.path.basename(a)): a
			for a in CORPORA[form['engine']].files}
	selected = {filenames[TEXTS[n]]: n for n in selectedtexts(form)}
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
		results = ((fmt % (filename, sentno, sent)
				if form.get('linenos') else (sent + '\n')).encode('utf8')
				for filename, sentno, sent, _ in results)
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
	filenames = {EXTRE.sub('', os.path.basename(a)): a
			for a in CORPORA[form['engine']].files}
	selected = {filenames[TEXTS[n]]: n for n in selectedtexts(form)}
	start, end = getslice(form.get('slice'))
	if not doexport:
		url = 'counts?' + url_encode(dict(export='csv', **form),
				separator=b';')
		yield ('Counts from queries '
				'(<a href="%s">export to CSV</a>):\n' % url)
	if norm == 'query':
		normresults = CORPORA[form.get('engine', 'tgrep2')].counts(
				form['normquery'], selected)
	# Combined results of all queries on each file
	combined = defaultdict(int)
	index = [TEXTS[n] for n in selected.values()]
	df = pandas.DataFrame(index=index)
	queries = querydict(form['query'])
	if not doexport:
		yield '<ol>%s</ol>\n' % '\n'.join(
				'<li><a href="#q%d">%s</a>' % (n, query)
				for n, query in enumerate(list(queries)
				+ ['Combined results', 'Overview'], 1))
	for n, (name, (normquery, query)) in enumerate(
			list(queries.items()) + [('Combined results', ('', None))], 1):
		cnts = Counter()
		sumtotal = 0
		relfreq = {}
		resultsindices = None
		if query is None:
			if len(df.columns) == 1:
				break
			results = combined
			legend = '%sLegend:\t%s\n' % (64 * ' ', '\t'.join(
					'\n<font color=%s>%s</font>' % (
						COLORS.get(n, 'black'), query)
					for n, query in enumerate(queries)))
		else:
			legend = ''
			if normquery:
				norm = 'query'
				normresults = CORPORA[form.get('engine', 'tgrep2')].counts(
						normquery, selected, start, end)
			else:
				norm = form.get('norm', 'sents')
			results = CORPORA[form.get('engine', 'tgrep2')].counts(
					query, selected, start, end, indices=False)
			if len(results) <= 32 and all(
					results[filename] < INDICESMAXRESULTS
					for filename in results):
				resultsindices = CORPORA[form.get('engine', 'tgrep2')].counts(
						query, selected, start, end, indices=True)
		if not doexport:
			yield '<a name=q%d><h3>%s</h3></a>\n<pre>\n%s\n' % (
					n, name, query or legend)
		COLWIDTH = min(40, max(map(len, TEXTS)) + 2)
		for filename, cnt in sorted(results.items()):
			if query is None:
				cnt = combined[filename]
			else:
				combined[filename] += cnt
			textno = selected[filename]
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
				out = ('%s (<a href="browsesents?%s">browse</a>)    '
						'%5d %5.2f %%' % (
						text.ljust(COLWIDTH)[:COLWIDTH],
						url_encode(
							dict(text=textno, sent=1,
								query=query or form['query'],
								engine=form.get('engine', 'tgrep2')),
							separator=b';'),
						cnt, relfreq[text]))
				plot = ''
				if resultsindices is not None:
					plot = dispplot(resultsindices[filename],
							start or 1, end or NUMSENTS[textno])
				if cnt:
					yield out + plot + '\n'
				else:
					yield '<span style="color: gray; ">%s%s</span>\n' % (
							out, plot)
		if not doexport or query is not None:
			df[name] = pandas.Series(relfreq)
		if not doexport:
			yield ('%s             %5d %5.2f %%\n</span>\n' % (
					'TOTAL'.ljust(COLWIDTH),
					sum(cnts.values()),
					100.0 * sum(cnts.values()) / sumtotal))
			yield '</pre>'
			if max(cnts.values()) == 0:
				continue
			elif form.get('slice'):
				# show absolute counts when all texts have been limited to same
				# number of sentences
				yield barplot(cnts, max(cnts.values()),
						'Absolute counts of \'%s\':' % name, unit='matches')
			else:
				yield barplot(relfreq, max(relfreq.values()),
						'Relative frequency of \'%s\': '
						'(count / num_%s * 100)' % (name, norm), unit='%')
	if doexport:
		if form.get('export') == 'json':
			yield json.dumps(df.to_dict(), indent=2)
		else:
			yield df.to_csv(None)
	else:
		fmt = lambda x: '%g' % round(x, 1)
		yield '<h3><a name=q%d>Overview of patterns</a></h3>\n' % (
				len(queries) + 2)
		# collate stats
		keys = [key.split('_')[0] if '_' in key else key[0]
				for key in df.index]
		keyset = set(keys)
		if len(keyset) <= 5:
			overview = OrderedDict(('%s_%s' % (letter, query),
					df[query].ix[[key for key in df.index
						if letter == (key.split('_')[0] if '_' in key
							else key[0])]].mean())
					for query in df.columns
						for letter in keyset)
			df['category'] = keys
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
	filenames = {EXTRE.sub('', os.path.basename(a)): a
			for a in CORPORA[form['engine']].files}
	selected = {filenames[TEXTS[n]]: n for n in selectedtexts(form)}
	start, end = getslice(form.get('slice'))
	# NB: we do not hide function or morphology tags when exporting
	url = 'trees?' + url_encode(dict(export='csv', **form), separator=b';')
	yield ('<pre>Query: %s\n'
			'Trees (showing up to %d per text; '
			'export: <a href="%s">plain</a>, '
			'<a href="%s">with line numbers</a>):\n' % (
				form['query'] if len(form['query']) < 128
				else form['query'][:128] + '...',
				TREELIMIT, url, url + ';linenos=1'))
	for n, (filename, results) in enumerate(groupby(sorted(
			CORPORA[form.get('engine', 'tgrep2')].trees(form['query'],
			selected, start, end, maxresults=TREELIMIT,
			nomorph='nomorph' in form, nofunc='nofunc' in form),
			key=itemgetter(0)), itemgetter(0))):
		textno = selected[filename]
		text = TEXTS[textno]
		if 'breakdown' in form:
			breakdown = Counter(DiscTree(
						max(high, key=lambda x: len(x.leaves())
							if isinstance(x, Tree) else 1).freeze(), sent)
					for _, _, _, sent, high in results if high)
			yield '\n%s\n' % text
			for match, cnt in breakdown.most_common():
				gotresults = True
				yield 'count: %5d\n%s\n\n' % (
						cnt, DrawTree(match, match.sent).text(
							unicodelines=True, html=True))
			continue
		for m, (filename, sentno, tree, sent, high) in enumerate(results):
			if m == 0:
				gotresults = True
				yield ("==&gt; %s: [<a href=\"javascript: toggle('n%d'); \">"
						"toggle</a>]\n<span id=n%d>" % (text, n + 1, n + 1))
			link = ('<a href="/browse?text=%d;sent=%s%s%s">browse</a>'
					'|<a href="/browsesents?text=%d;sent=%s;highlight=%s">'
					'context</a>' % (textno, sentno,
					';nofunc' if 'nofunc' in form else '',
					';nomorph' if 'nomorph' in form else '',
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
	filenames = {EXTRE.sub('', os.path.basename(a)): a
			for a in CORPORA[form['engine']].files}
	selected = {filenames[TEXTS[n]]: n for n in selectedtexts(form)}
	start, end = getslice(form.get('slice'))
	url = '%s?%s' % ('trees' if dobrackets else 'sents',
			url_encode(dict(export='csv', **form), separator=b';'))
	yield ('<pre>Query: %s\n'
			'Sentences (showing up to %d per text; '
			'export: <a href="%s">plain</a>, '
			'<a href="%s">with line numbers</a>):\n' % (
				form['query'] if len(form['query']) < 128
				else form['query'][:128] + '...',
				SENTLIMIT, url, url + ';linenos=1'))
	for n, (filename, results) in enumerate(groupby(sorted(
			CORPORA[form.get('engine', 'tgrep2')].sents(form['query'],
				selected, start, end, maxresults=SENTLIMIT,
				brackets=dobrackets), key=itemgetter(0)),
			itemgetter(0))):
		textno = selected[filename]
		text = TEXTS[textno]
		if 'breakdown' in form:
			if dobrackets:
				breakdown = Counter(high for _, _, _, high, _ in results)
			else:
				breakdown = Counter(re.sub(
					' {2,}', ' ... ',
					' '.join(char if n in high1 or n in high2 else ' '
						for n, char in enumerate(sent.split())))
					for _, _, sent, high1, high2 in results)
			yield '\n%s\n' % text
			for match, cnt in breakdown.most_common():
				gotresults = True
				yield '%5d  %s\n' % (cnt, match)
			continue
		for m, (filename, sentno, sent, high1, high2) in enumerate(results):
			if m == 0:
				gotresults = True
				yield ("\n%s: [<a href=\"javascript: toggle('n%d'); \">"
						"toggle</a>] <ol id=n%d>" % (text, n, n))
			link = ('<a href="/browse?text=%d;sent=%s%s%s">tree</a>'
					'|<a href="/browsesents?text=%d;sent=%s;highlight=%s">'
					'context</a>' % (textno, sentno,
					';nofunc' if 'nofunc' in form else '',
					';nomorph' if 'nomorph' in form else '',
					textno, sentno, sentno))
			if dobrackets:
				sent = cgi.escape(sent.replace(" )", " -NONE-)"))
				out = sent.replace(high1, "<span class=r>%s</span>" % high1)
			else:
				out = applyhighlight(sent, high1, high2)
			yield "<li>#%s [%s] %s\n" % (str(sentno).rjust(6), link, out)
		yield "</ol>"
	yield '</pre>' if gotresults else 'No matches.'


def applyhighlight(sent, high1, high2, colorvec=None):
	"""Return a version of sent where given char. indices are highlighted."""
	cur = None
	start = 0
	out = []
	for n, _ in enumerate(sent):
		if colorvec is not None:
			if cur != COLORS.get(colorvec[n], 'gray'):
				out.append(sent[start:n])
				if cur is not None:
					out.append('</font>')
				out.append('<font color=%s>' % COLORS.get(colorvec[n], 'gray'))
				start = n
				cur = COLORS.get(colorvec[n], 'gray')
		elif n in high1:
			if cur != 'red':
				out.append(sent[start:n])
				if cur is not None:
					out.append('</span>')
				out.append('<span class=r>')
				start = n
				cur = 'red'
		elif n in high2:
			if cur != 'blue':
				out.append(sent[start:n])
				if cur is not None:
					out.append('</span>')
				out.append('<span class=b>')
				start = n
				cur = 'blue'
		else:
			if cur is not None:
				out.append(sent[start:n])
				out.append('</span>')
				start = n
				cur = None
	out.append(sent[start:])
	if cur is not None:
		out.append('</font>')
	return ''.join(out)


def brackets(form):
	"""Wrapper."""
	return sents(form, dobrackets=True)


def fragmentsinresults(form, doexport=False):
	"""Extract recurring fragments from search results."""
	if form.get('engine', 'tgrep2') not in ('tgrep2', 'xpath', 'frag'):
		yield "Only applicable to treebanks."
		return
	gotresults = False
	filenames = {EXTRE.sub('', os.path.basename(a)): a
			for a in CORPORA[form['engine']].files}
	selected = {filenames[TEXTS[n]]: n for n in selectedtexts(form)}
	start, end = getslice(form.get('slice'))
	uniquetrees = set()
	if not doexport:
		url = 'fragments?' + url_encode(dict(export='csv', **form),
				separator=b';')
		yield ('<pre>Query: %s\n'
				'Fragments (showing up to %d fragments '
				'in the first %d search results from selected texts;\n'
				'ordered by (freq ** 0.5 * numwords ** 2) '
				'<a href="%s">Export</a>):\n'
				% (form['query'] if len(form['query']) < 128
					else form['query'][:128] + '...',
					FRAGLIMIT, SENTLIMIT, url))
	disc = form.get('engine', 'tgrep2') != 'tgrep2'
	if disc:
		fragments.PARAMS.update(disc=True, fmt='discbracket')
	else:
		fragments.PARAMS.update(disc=False, fmt='bracket')
	for n, (_, _, treestr, _) in enumerate(CORPORA[form.get(
			'engine', 'tgrep2')].sents(form['query'], selected, start, end,
			maxresults=SENTLIMIT, brackets=True)):
		if n == 0:
			gotresults = True
		if form.get('engine', 'tgrep2') == 'tgrep2':
			line = treestr.replace(" )", " -NONE-)") + '\n'
		elif form.get('engine', 'tgrep2') == 'xpath':
			item = treebank.alpinotree(
					ElementTree.fromstring(treestr.encode('utf8')))
			line = '%s\t%s\n' % (str(item.tree), ' '.join(item.sent))
		elif form.get('engine', 'tgrep2') == 'frag':
			line = treestr + '\n'
		else:
			raise ValueError
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
				link = "<a href='/draw?tree=%s;sent=%s'>draw</a>" % (
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
		shortest = min(NUMSENTS)
		yield ('Results based on first %d sentences (=shortest text %s).'
				% (shortest, TEXTS[NUMSENTS.index(shortest)]))

		# produce a plot for each field
		fields = ()
		for a in STYLETABLE:
			fields = sorted(STYLETABLE[a].keys())
			break
		yield '<ol>\n'
		for field in fields:
			yield '<li><a href="#%s">%s</a>\n' % (field, field)
		yield '</ol>\n'
		for field in fields:
			data = {a: STYLETABLE[a].get(field, 0) for a in STYLETABLE}
			total = max(data.values())
			if total > 0:
				yield barplot(data, total,
						'<a name="%s">%s:</a>' % (field, field),
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
				results=generate(),
				havetgrep='tgrep2' in CORPORA,
				havexpath='xpath' in CORPORA,
				havefrag='frag' in CORPORA,
				))
	resp.headers['Cache-Control'] = 'max-age=604800, public'
	# set Expires one day ahead (according to server time)
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
	if 'xpath' in CORPORA:
		filename = os.path.join(CORPUS_DIR, TEXTS[textno] + '.dact')
		tree, sent = CORPORA['xpath'].extract(
				filename, [sentno], nofunc, nomorph).pop()
	elif 'tgrep2' in CORPORA:
		filename = os.path.join(CORPUS_DIR, TEXTS[textno] + '.mrg')
		tree, sent = CORPORA['tgrep2'].extract(
				filename, [sentno], nofunc, nomorph).pop()
	elif 'frag' in CORPORA:
		filename = os.path.join(CORPUS_DIR, TEXTS[textno] + '.dbr')
		tree, sent = CORPORA['frag'].extract(
				filename, [sentno], nofunc, nomorph).pop()
	else:
		raise ValueError('no treebank available for "%s".' % TEXTS[textno])
	result = DrawTree(tree, sent).text(unicodelines=True, html=True)
	return '<pre id="t%s">%s</pre>' % (sentno, result)


@APP.route('/browse')
def browsetrees():
	"""Browse through trees in a file."""
	chunk = 20  # number of trees to fetch for one request
	if 'text' in request.args and 'sent' in request.args:
		textno = int(request.args['text'])
		sentno = int(request.args['sent'])
		start = max(1, sentno - sentno % chunk)
		maxtree = min(start + chunk, NUMSENTS[textno] + 1)
		nofunc = 'nofunc' in request.args
		nomorph = 'nomorph' in request.args
		if 'xpath' in CORPORA:
			filename = os.path.join(CORPUS_DIR, TEXTS[textno] + '.dact')
			drawntrees = [DrawTree(tree, sent).text(
					unicodelines=True, html=True)
					for tree, sent in CORPORA['xpath'].extract(
						filename, range(start, maxtree), nofunc, nomorph)]
		elif 'tgrep2' in CORPORA:
			filename = os.path.join(CORPUS_DIR, TEXTS[textno] + '.mrg')
			drawntrees = [DrawTree(tree, sent).text(
					unicodelines=True, html=True)
					for tree, sent in CORPORA['tgrep2'].extract(
						filename, range(start, maxtree), nofunc, nomorph)]
		elif 'frag' in CORPORA:
			filename = os.path.join(CORPUS_DIR, TEXTS[textno] + '.dbr')
			drawntrees = [DrawTree(tree, sent).text(
					unicodelines=True, html=True)
					for tree, sent in CORPORA['frag'].extract(
						filename, range(start, maxtree), nofunc, nomorph)]
		else:
			raise ValueError('no treebank available for "%s".' % TEXTS[textno])
		results = ['<pre id="t%s"%s>%s</pre>' % (n,
				' style="display: none; "' if 'ajax' in request.args else '',
				tree) for n, tree in enumerate(drawntrees, start)]
		if 'ajax' in request.args:
			return '\n'.join(results)

		prevlink = '<a id=prev>prev</a>'
		if sentno > chunk:
			prevlink = '<a href="browse?text=%d;sent=%d" id=prev>prev</a>' % (
					textno, sentno - chunk + 1)
		nextlink = '<a id=next>next</a>'
		if sentno < NUMSENTS[textno] - chunk:
			nextlink = '<a href="browse?text=%d;sent=%d" id=next>next</a>' % (
					textno, sentno + chunk + 1)
		return render_template('browse.html', textno=textno, sentno=sentno,
				text=TEXTS[textno], totalsents=NUMSENTS[textno], trees=results,
				prevlink=prevlink, nextlink=nextlink, chunk=chunk,
				nofunc=nofunc, nomorph=nomorph,
				mintree=start, maxtree=maxtree)
	return '<h1>Browse through trees</h1>\n<ol>\n%s</ol>\n' % '\n'.join(
			'<li><a href="browse?text=%d;sent=1;nomorph">%s</a> '
			'(%d sentences; %d words)' % (n, text, NUMSENTS[n], NUMWORDS[n])
			for n, text in enumerate(TEXTS))


@APP.route('/browsesents')
def browsesents():
	"""Browse through sentences in a file; optionally highlight matches."""
	chunk = 20  # number of sentences per page
	if 'text' in request.args and 'sent' in request.args:
		textno = int(request.args['text'])
		sentno = int(request.args['sent'])
		highlight = int(request.args.get('highlight', 0))
		sentno = min(max(chunk // 2 + 1, sentno),
				NUMSENTS[textno] - chunk // 2)
		start = max(1, sentno - chunk // 2)
		maxsent = min(start + chunk, NUMSENTS[textno] + 1)
		if 'tgrep2' in CORPORA:
			filename = os.path.join(CORPUS_DIR, TEXTS[textno] + '.mrg')
			results = CORPORA['tgrep2'].extract(
					filename, range(start, maxsent), sents=True)
		elif 'xpath' in CORPORA:
			filename = os.path.join(CORPUS_DIR, TEXTS[textno] + '.dact')
			results = CORPORA['xpath'].extract(
					filename, range(start, maxsent), sents=True)
		elif 'regex' in CORPORA:
			filename = os.path.join(CORPUS_DIR, TEXTS[textno] + '.tok')
			results = CORPORA['regex'].extract(
					filename, range(start, maxsent), sents=True)
		elif 'frag' in CORPORA:
			filename = os.path.join(CORPUS_DIR, TEXTS[textno] + '.dbr')
			results = CORPORA['frag'].extract(
					filename, range(start, maxsent), sents=True)
		else:
			raise ValueError('no treebank available for "%s".' % TEXTS[textno])
		results = [('<font color=red>%s</font>' % cgi.escape(a))
				if n == highlight else cgi.escape(a)
				for n, a in enumerate(results, start)]
		legend = queryparams = ''
		if request.args.get('query', ''):
			queryparams = ';' + url_encode(dict(
					query=request.args['query'],
					engine=request.args.get('engine', 'tgrep2')),
					separator=b';')
			filenames = {EXTRE.sub('', os.path.basename(a)): a
					for a in CORPORA[request.args['engine']].files}
			filename = filenames[TEXTS[textno]]
			queries = querydict(request.args['query'])
			legend = 'Legend:\t%s\n' % ('\t'.join(
					'\n<font color=%s>%s</font>' % (COLORS.get(n, 'gray'),
						query)
					for n, query in enumerate(queries, 1)))
			resultshighlight = {}
			for n, (_, query) in enumerate(queries.values()):
				matches = CORPORA[request.args['engine']].sents(
						query, subset=(filename, ),
						start=start, end=maxsent - 1,
						maxresults=2 * chunk)
				for _, m, sent, high, _ in matches:
					if m - start not in resultshighlight:
						resultshighlight[m - start] = [0] * len(sent)
					for x in high:
						resultshighlight[m - start][x] = n + 1
			for m, high in resultshighlight.items():
				results[m] = applyhighlight(
						results[m], set(), set(), colorvec=high)
		prevlink = '<a id=prev>prev</a>'
		if sentno > chunk:
			prevlink = ('<a href="browsesents?text=%d;sent=%d%s" id=prev>'
					'prev</a>' % (textno, sentno - chunk, queryparams))
		nextlink = '<a id=next>next</a>'
		if sentno < NUMSENTS[textno] - chunk:
			nextlink = ('<a href="browsesents?text=%d;sent=%d%s" id=next>'
					'next</a>' % (textno, sentno + chunk, queryparams))
		return render_template('browsesents.html', textno=textno,
				sentno=sentno, text=TEXTS[textno], totalsents=NUMSENTS[textno],
				sents=results, prevlink=prevlink, nextlink=nextlink,
				chunk=chunk, mintree=start, legend=legend,
				query=request.args.get('query', ''),
				engine=request.args.get('engine', ''))
	return '<h1>Browse through sentences</h1>\n<ol>\n%s</ol>\n' % '\n'.join(
			'<li><a href="browsesents?text=%d;sent=1;nomorph">%s</a> '
			'(%d sentences; %d words)' % (n, text, NUMSENTS[n], NUMWORDS[n])
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
	keys = {key.split('_')[0] if '_' in key else key[0] for key in data}
	color = {}
	if len(keys) <= 5:
		color.update(zip(keys, range(1, 6)))
	keys = sorted(data, key=data.get, reverse=True) if dosort else data
	for key in keys:
		result.append('<br><div style="width:%dpx;" class=b%d></div>'
				'<span>%s: %g %s</span>' % (
				int(round(width * data[key] / total)) if data[key] else 0,
				color.get(key.split('_')[0] if '_' in key else key[0], 1)
					if data[key] else 0,
				cgi.escape(key), data[key], unit,))
	result.append('</div>\n')
	return '\n'.join(result)


def dispplot(indices, start, end, width=800.0, runle=False):
	"""Draw a dispersion plot from a list of indices.

	:param indices: a tuple of lists where each element is a sentence number.
		Each element of indices will be drawn in a different color to represent
		a different query.
	:param start, end: the range of sentences numbers.
	:param runle: use a more compact, run-length encoded representation."""
	result = ('\t<svg version="1.1" xmlns="http://www.w3.org/2000/svg"'
			' width="%dpx" height="10px" >\n'
			'<rect x=0 y=0 width="%dpx" height=10 '
			'fill=white stroke=black />\n' % (width, width))
	for n, a in enumerate(indices if isinstance(indices, tuple) else [indices]):
		if not a:
			continue
		if runle:  # FIXME: use start
			strokes = []
			idx0 = 0
			seq = [-1] + sorted(a) + [-1]
			for prev, idx, nextidx in zip(seq, seq[1:], seq[2:]):
				if idx != prev + 1 and idx != nextidx - 1:
					strokes.append('M %d 0v 10' % idx)
				elif idx != prev + 1:
					idx0 = idx
				elif idx != nextidx - 1:
					strokes.append(  # draw a rectangle covering idx0:idx
							'M %d 0l 0 10 %d 0 0 -10' % (
								idx0 - 1, idx - idx0 - 1))
		else:
			strokes = ['M %d 0v 10' % (idx - start) for idx in sorted(a)]
		result += ('<g transform="scale(%g, 1)">\n'
				'<path stroke=%s d="%s" /></g>' % (
				width / (end - start), COLORS.get(n, 'black'),
				''.join(strokes)))
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


def getslice(a):
	"""Parse slice argument of the form n-m, where n and m are optional."""
	if not a:
		return None, None
	elif '-' in a:
		start, end = a.split('-')
	else:
		start, end = None, a
	return (int(start) if start else None), (int(end) if end else None)


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
	elif os.path.exists(base + '.mrg.t2c.gz'):
		tgrep = subprocess.Popen(
				args=[which('tgrep2'), '-t', '-c', base + '.mrg.t2c.gz', '*'],
				shell=False, bufsize=-1, stdout=subprocess.PIPE)
		converted = (a.replace('-LRB-', '(').replace('-RRB-', ')')
				for a in tgrep.stdout)
	elif os.path.exists(base + '.mrg'):
		converted = (' '.join(GETLEAVES.findall(line)
				).replace('-LRB-', '(').replace('-RRB-', ')') + '\n'
				for line in open(base + '.mrg'))
	elif os.path.exists(base + '.dact'):
		result = {entry.name(): ElementTree.fromstring(entry.contents()).find(
				'sentence').text.encode('utf8') + '\n' for entry
				in alpinocorpus.CorpusReader(base + '.dact').entries()}
		converted = [result[a] for a in sorted(result, key=treebank.numbase)]
	elif ucto and filename.endswith('.txt'):
		newfile = base + '.tok'
		proc = subprocess.Popen(args=[which('ucto'),
				'-L', LANG, '-s', '', '-n',
				filename, newfile], shell=False)
		proc.wait()
		return
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
						islice(io.open(filename, encoding='utf8'), cutoff),
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
					open(picklefile, 'rb'))
		except ValueError:
			pass
	tfiles = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.mrg')))
	ffiles = sorted(
			glob.glob(os.path.join(CORPUS_DIR, '*.export'))
			or glob.glob(os.path.join(CORPUS_DIR, '*.dbr'))
			or glob.glob(os.path.join(CORPUS_DIR, '*.mrg')))
	afiles = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.dact')))
	txtfiles = glob.glob(os.path.join(CORPUS_DIR, '*.txt'))
	# get tokenized sents from trees or ucto
	for filename in tfiles or afiles or txtfiles:
		tokenize(filename)
	tokfiles = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.tok')))
	if tfiles and set(tfiles) != set(corpora.get('tgrep2', ())):
		corpora['tgrep2'] = treesearch.TgrepSearcher(
				tfiles, macros='static/tgrepmacros.txt')
		log.info('tgrep2 corpus loaded.')
	if ffiles and set(ffiles) != set(corpora.get('frag', ())):
		corpora['frag'] = treesearch.FragmentSearcher(
				ffiles, macros='static/fragmacros.txt', inmemory=False)
		log.info('frag corpus loaded.')
	if afiles and ALPINOCORPUSLIB and set(afiles) != set(
			corpora.get('xpath', ())):
		corpora['xpath'] = treesearch.DactSearcher(
				afiles, macros='static/xpathmacros.txt')
		log.info('xpath corpus loaded.')
	if tokfiles and set(tokfiles) != set(corpora.get('regex', ())):
		corpora['regex'] = treesearch.RegexSearcher(
				tokfiles, macros='static/regexmacros.txt')
		log.info('regex corpus loaded.')

	assert tfiles or afiles or ffiles or tokfiles, (
			'no corpus files with extension .mrg, .dbr, .export, .dact, '
			'or .txt found in %s' % CORPUS_DIR)
	assert len(set(
			frozenset(b.rsplit('.', 1)[0] for b in files)
			for files in (tfiles, afiles, ffiles, tokfiles) if files)) == 1, (
			'files in different formats do not match.')
	picklemtime = 0
	if os.path.exists(picklefile):
		picklemtime = os.stat(picklefile).st_mtime
	currentfiles = {os.path.splitext(os.path.basename(filename))[0]
		for filename in tfiles + afiles + ffiles + tokfiles}
	if (set(texts) != currentfiles or any(os.stat(a).st_mtime > picklemtime
				for a in tfiles + afiles + ffiles + tokfiles)):
		if corpora.get('tgrep2'):
			numsents = [len(open(filename).readlines())
					for filename in tfiles]
			numconst = [open(filename).read().count('(')
					for filename in tfiles]
			numwords = [len(GETLEAVES.findall(open(filename).read()))
					for filename in tfiles]
		elif corpora.get('frag'):
			numsents, numconst, numwords = [], [], []
			for filename in ffiles:
				info = corpora['frag'].getinfo(filename)
				numsents.append(info.len)
				numconst.append(info.numnodes)
				numwords.append(info.numwords)
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
			numsents = [len(corpora['regex'].lineindex[a]) for a in tokfiles]
			numwords = [open(filename).read().count(' ') + n
					for filename, n in zip(tokfiles, numsents)]
			numconst = [0 for _ in tokfiles]
		else:
			raise ValueError('no texts found.')
		texts = [os.path.splitext(os.path.basename(a))[0]
				for a in tfiles or afiles or ffiles or tokfiles]
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
	"""Convert sets to lists for JSON encoding."""
	def default(self, obj):  # pylint: disable=method-hidden
		"""Do conversion."""
		if isinstance(obj, set):
			return list(obj)
		return json.JSONEncoder.default(self, obj)


class QueryStringRedirectMiddleware(object):
	"""Support ; as query delimiter.

	http://flask.pocoo.org/snippets/43/"""
	def __init__(self, application):
		self.application = application

	def __call__(self, environ, start_response):
		qs = environ.get('QUERY_STRING', '')
		environ['QUERY_STRING'] = qs.replace(';', '&')
		return self.application(environ, start_response)


APP.wsgi_app = QueryStringRedirectMiddleware(APP.wsgi_app)

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
log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.handlers[0].setFormatter(logging.Formatter(
		fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
log.info('loading corpus.')
TEXTS, NUMSENTS, NUMCONST, NUMWORDS, STYLETABLE, CORPORA = getcorpus()
log.info('corpus loaded.')

if __name__ == '__main__':
	from getopt import gnu_getopt, GetoptError
	try:
		opts, _args = gnu_getopt(sys.argv[1:], '', ['port=', 'ip='])
		opts = dict(opts)
	except GetoptError as err:
		print('error: %r' % err, file=sys.stderr)
		sys.exit(2)
	APP.run(debug=DEBUG, use_reloader=False,
			host=opts.get('--ip', '0.0.0.0'),
			port=int(opts.get('--port', 5001)))
