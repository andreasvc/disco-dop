""" Web interface to search a treebank. Requires Flask, tgrep2
or alpinocorpus-python (for xpath queries), style. Expects one or more
treebanks with .mrg or .dact extension in the directory corpus/ """
# stdlib
import io
import os
import re
import cgi
import csv
import sys
import glob
import logging
import tempfile
import subprocess
from heapq import nlargest
from urllib import quote
from datetime import datetime, timedelta
from itertools import islice, count
from collections import Counter
try:
	import cPickle as pickle
except ImportError:
	import pickle
# Flask & co
from flask import Flask, Response
from flask import request, render_template, send_from_directory
# alpinocorpus
try:
	import alpinocorpus
	import xml.etree.cElementTree as ElementTree
	ALPINOCORPUSLIB = True
except ImportError:
	ALPINOCORPUSLIB = False
# disco-dop
from discodop.tree import Tree
from discodop.treedraw import DrawTree
from discodop import fragments, treebank

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


@APP.route('/')
@APP.route('/counts')
@APP.route('/trees')
@APP.route('/sents')
@APP.route('/brackets')
@APP.route('/fragments')
def main():
	""" Main search form & results page. """
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
		#return render_template('searchresults.html',
		#		form=request.args, texts=TEXTS, selectedtexts=selected,
		#		output=output, results=DISPATCH[output](request.args),
		#		havexpath=ALPINOCORPUSLIB)
		# To send results incrementally:
		return Response(stream_template('searchresults.html',
				form=request.args, texts=TEXTS, selectedtexts=selected,
				output=output, results=DISPATCH[output](request.args),
				havexpath=ALPINOCORPUSLIB))
	return render_template('search.html', form=request.args, output='counts',
			texts=TEXTS, selectedtexts=selected, havexpath=ALPINOCORPUSLIB)


@APP.route('/style')
def style():
	""" Use style(1) program to get staticstics for each text. """
	def generate():
		""" Generate plots from results. """
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
		""" Generate CSV file. """
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
	""" Produce a visualization of a tree on a separate page. """
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
		treestr = XMLCORPORA[textno].read('%d.xml' % sentno)
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
	""" Browse through sentences in a file. """
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
			results = [ALPINOLEAVES.search(XMLCORPORA[textno].read(
					'%d.xml' % (n + 1, ))).group(1)
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
	return '<ol>\n%s</ol>\n' % '\n'.join(
			'<li><a href="browsesents?text=%d&sent=1&nomorph">%s</a> '
			'(%d sentences)' % (n, text, NUMSENTS[n])
			for n, text in enumerate(TEXTS))


@APP.route('/browse')
def browse():
	""" Browse through trees in a file. """
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
			drawntrees = [DrawTree(*treebank.alpinotree(
					ElementTree.fromstring(XMLCORPORA[textno].read(
						'%d.xml' % (n + 1, ))),
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
	return '<ol>\n%s</ol>\n' % '\n'.join(
			'<li><a href="browse?text=%d&sent=1&nomorph">%s</a> '
			'(%d sentences)' % (n, text, NUMSENTS[n])
			for n, text in enumerate(TEXTS))


@APP.route('/favicon.ico')
def favicon():
	""" Serve the favicon. """
	return send_from_directory(os.path.join(APP.root_path, 'static'),
			'treesearch.ico', mimetype='image/vnd.microsoft.icon')


def export(form, output):
	""" Export search results to a file for download. """
	# NB: no distinction between trees from different texts
	# (Does consttreeviewer support # comments?)
	if output == 'counts':
		results = counts(form, doexport=True)
		filename = 'counts.csv'
	elif output == 'fragments':
		results = fragmentsinresults(form, doexport=True)
		filename = 'fragments.txt'
	else:
		results = (a + '\n' for a in doqueries(
				form, lines=False, doexport=output))
		filename = '%s.txt' % output
	resp = Response(results, mimetype='text/plain')
	resp.headers['Content-Disposition'] = 'attachment; filename=' + filename
	return resp


def counts(form, doexport=False):
	""" Produce counts of matches for each text. """
	cnts = Counter()
	relfreq = {}
	sumtotal = 0
	norm = form.get('norm', 'sents')
	gotresult = False
	for n, (textno, results, stderr) in enumerate(doqueries(form, lines=False)):
		if n == 0:
			if not doexport:
				url = 'counts?query=%s&norm=%s&texts=%s&engine=%s&export=1' % (
						form['query'], form['norm'], form['texts'],
						form.get('engine', 'tgrep2'))
				yield ('<pre>Query: %s\n'
						'NB: the counts reflect the total number of '
						'times a pattern matches for each tree.\n\n'
						'Counts (<a href="%s">export to CSV</a>):\n' % (
						stderr, url))
		cnt = results.count('\n')
		if cnt:
			gotresult = True
		text = TEXTS[textno]
		cnts[text] = cnt
		totalsent = NUMSENTS[textno]
		if norm == 'consts':
			total = NUMCONST[textno]
		elif norm == 'words':
			total = NUMWORDS[textno]
		elif norm == 'sents':
			total = NUMSENTS[textno]
		elif norm == 'query':
			total = next(doqueries_(form.get('engine', 'tgrep2'),
					form['normquery'], {textno}))[1].count('\n') or 1
		else:
			raise ValueError
		relfreq[text] = 100.0 * cnt / total
		sumtotal += total
		if not doexport:
			line = "%s%6d    %5.2f %%" % (
					text.ljust(40)[:40], cnt, relfreq[text])
			indices = {int(line[:line.index(':::')])
					for line in results.splitlines()}
			plot = concplot(indices, totalsent)
			if cnt:
				yield line + plot + '\n'
			else:
				yield '<span style="color: gray; ">%s%s</span>\n' % (line, plot)
	if not doexport:
		yield '</pre>'
	if doexport:
		tmp = io.BytesIO()
		csvexport = csv.writer(tmp)
		csvexport.writerow(['text', 'count', 'relfreq'])
		csvexport.writerows([text, cnts[text], relfreq[text]] for text in cnts)
		yield tmp.getvalue()
	elif gotresult:
		yield ("%s%6d    %5.2f %%\n</span>\n" % (
				"TOTAL".ljust(40),
				sum(cnts.values()),
				100.0 * sum(cnts.values()) / sumtotal))
		yield barplot(relfreq, max(relfreq.values()),
				'Relative frequency of pattern: (count / num_%s * 100)' % norm,
				unit='%')
		#yield barplot(cnts, max(cnts.values()), 'Absolute counts of pattern')


def trees(form):
	""" Return visualization of parse trees in search results. """
	gotresults = False
	for n, (textno, results, stderr) in enumerate(
			doqueries(form, lines=True)):
		if n == 0:
			# NB: we do not hide function or morphology tags when exporting
			url = 'trees?query=%s&texts=%s&engine=%s&export=1' % (
					quote(form['query']), form['texts'],
					form.get('engine', 'tgrep2'))
			yield ('<pre>Query: %s\n'
					'Trees (showing up to %d per text; '
					'export: <a href="%s">plain</a>, '
					'<a href="%s">with line numbers</a>):\n' % (
						stderr, TREELIMIT, url, url + '&linenos=1'))
		for m, line in enumerate(islice(results, TREELIMIT)):
			lineno, text, treestr, match = line.split(":::")
			if m == 0:
				gotresults = True
				yield ("==&gt; %s: [<a href=\"javascript: toggle('n%d'); \">"
						"toggle</a>]\n<span id=n%d>" % (text, n + 1, n + 1))
			if form.get('engine', 'tgrep2') == 'tgrep2':
				cnt = count()
				treestr = treestr.replace(" )", " -NONE-)")
				match = match.strip()
				if match.startswith('('):
					treestr = treestr.replace(match, '%s_HIGH %s' % tuple(
							match.split(None, 1)))
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
			elif form.get('engine') == 'xpath':
				tree, sent = treebank.alpinotree(
						ElementTree.fromstring(treestr.encode('utf-8')),
						functions=None if 'nofunc' in form else 'add',
						morphology=None if 'nomorph' in form else 'replace')
				highwords = re.findall('<node[^>]*begin="([0-9]+)"[^>]*/>',
						match)
				high = set(re.findall(r'\bid="(.+?)"', match))
				high = list(tree.subtrees(lambda n:
						n.source[treebank.PARENT] in high or
						n.source[treebank.WORD].lstrip('#') in high))
				high += [int(a) for a in highwords]
			link = ('<a href="/browse?text=%d&sent=%s%s%s">browse</a>'
					'|<a href="/browsesents?text=%d&sent=%s&highlight=%s">'
					'context</a>' % (textno, lineno,
					'&nofunc' if 'nofunc' in form else '',
					'&nomorph' if 'nomorph' in form else '',
					textno, lineno, lineno))
			try:
				treerepr = DrawTree(tree, sent, highlight=high).text(
						unicodelines=True, html=True)
			except ValueError as err:
				line = "#%s \nERROR: %s\n%s\n%s\n" % (
						lineno, err, treestr, tree)
			else:
				line = "#%s [%s]\n%s\n" % (lineno, link, treerepr)
			yield line
		yield "</span>"
	yield '</pre>' if gotresults else "No matches."


def sents(form, dobrackets=False):
	""" Return search results as terminals or in bracket notation. """
	gotresults = False
	for n, (textno, results, stderr) in enumerate(
			doqueries(form, lines=True)):
		if n == 0:
			url = '%s?query=%s&texts=%s&engine=%s&export=1' % (
					'trees' if dobrackets else 'sents',
					quote(form['query']), form['texts'],
					form.get('engine', 'tgrep2'))
			yield ('<pre>Query: %s\n'
					'Sentences (showing up to %d per text; '
					'export: <a href="%s">plain</a>, '
					'<a href="%s">with line numbers</a>):\n' % (
						stderr, SENTLIMIT, url, url + '&linenos=1'))
		for m, line in enumerate(islice(results, SENTLIMIT)):
			lineno, text, treestr, match = line.rstrip().split(":::")
			if m == 0:
				gotresults = True
				yield ("\n%s: [<a href=\"javascript: toggle('n%d'); \">"
						"toggle</a>] <ol id=n%d>" % (text, n, n))
			link = ('<a href="/browse?text=%d&sent=%s%s%s">draw</a>'
					'|<a href="/browsesents?text=%d&sent=%s&highlight=%s">'
					'context</a>' % (textno, lineno,
					'&nofunc' if 'nofunc' in form else '',
					'&nomorph' if 'nomorph' in form else '',
					textno, lineno, lineno))
			if dobrackets:
				treestr = cgi.escape(treestr.replace(" )", " -NONE-)"))
				out = treestr.replace(match,
						"<span class=r>%s</span>" % match)
			else:
				if form.get('engine') == 'xpath':
					out = ALPINOLEAVES.search(treestr).group(1)
					# extract starting index of highlighted words
					high = set(re.findall(
							'<node[^>]*begin="([0-9]+)"[^>]*/>', match))
					out = ' '.join('<span class=r>%s</span>' % word
							if str(n) in high
							else word for n, word in enumerate(out.split()))
				else:
					treestr = cgi.escape(treestr.replace(" )", " -NONE-)"))
					pre, highlight, post = treestr.partition(match)
					out = "%s <span class=r>%s</span> %s " % (
							' '.join(GETLEAVES.findall(pre)),
							' '.join(GETLEAVES.findall(highlight))
									if '(' in highlight else highlight,
							' '.join(GETLEAVES.findall(post)))
			yield "<li>#%s [%s] %s" % (lineno.rjust(6), link, out)
		yield "</ol>"
	yield '</pre>' if gotresults else 'No matches.'


def brackets(form):
	""" Wrapper. """
	return sents(form, dobrackets=True)


def fragmentsinresults(form, doexport=False):
	""" Extract recurring fragments from search results. """
	if form.get('engine', 'tgrep2') != 'tgrep2':
		yield "Only implemented for TGrep2 queries."
		return
	gotresults = False
	uniquetrees = set()
	for n, (_, results, stderr) in enumerate(
			doqueries(form, lines=True)):
		if n == 0 and not doexport:
			url = 'fragments?query=%s&texts=%s&engine=%s&export=1' % (
					quote(form['query']), form['texts'],
					form.get('engine', 'tgrep2'))
			yield ('<pre>Query: %s\n'
					'Fragments (showing up to %d fragments '
					'in the first %d search results from selected texts; '
					'<a href="%s">Export</a>):\n'
					% (stderr, FRAGLIMIT, SENTLIMIT, url))
		for m, line in enumerate(islice(results, SENTLIMIT)):
			if m == 0:
				gotresults = True
			_, _, treestr, _ = line.rstrip().split(":::")
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
			key=lambda ff: (2 * ff[0].count(')') - ff[0].count(' (')
				- ff[0].count(' )')) * ff[1])
	results = [(frag, freq) for frag, freq in results
			if (2 * frag.count(')')
				- frag.count(' (')
				- frag.count(' )')) > MINNODES and freq > MINFREQ]
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


def doqueries(form, lines=False, doexport=None):
	""" Wrapper. """
	return doqueries_(form.get('engine', 'tgrep2'), form['query'],
			selectedtexts(form), lines, doexport, form.get('linenos'),
			form.get('nofunc'), form.get('nomorph'))


def doqueries_(engine, query, selected, lines=False, doexport=None,
		linenos=False, nofunc=False, nomorph=False):
	""" Run query engine on each text. """
	if engine == 'tgrep2':
		return dotgrep2queries(query, selected, lines, doexport,
				linenos, nofunc, nomorph)
	elif engine == 'xpath':
		return doxpathqueries(query, selected, lines, doexport,
				linenos)
	elif engine == 'regex':
		return doregexqueries(query, selected, lines, doexport,
				linenos)
	raise ValueError('unexpected query engine: %s' % engine)


def dotgrep2queries(query, selected, lines=False, doexport=None,
		linenos=False, nofunc=False, nomorph=False):
	""" Run tgrep2 on each text. """
	if doexport == 'sents':
		fmt = r'%f:%s|%tw\n' if linenos else r'%tw\n'
	elif doexport == 'trees' or doexport == 'brackets':
		fmt = r"%f:%s|%w\n" if linenos else r"%w\n"
	elif doexport is None:
		# %s the sentence number
		# %f the corpus name
		# %w complete tree in bracket notation
		# %h the matched subtree in bracket notation
		fmt = r'%s:::%f:::%w:::%h\n'
	else:
		raise ValueError
	for n, text in enumerate(TEXTS):
		if n not in selected:
			continue
		cmd = [which('tgrep2'), '-z', '-a',
				'-m', fmt,
				'-c', os.path.join(CORPUS_DIR, text + '.mrg.t2c.gz'),
				'static/tgrepmacros.txt',
				query]
		proc = subprocess.Popen(args=cmd,
				bufsize=-1, shell=False,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE)
		out, err = proc.communicate()
		out = out.decode('utf8')  # pylint: disable=E1103
		err = err.decode('utf8')  # pylint: disable=E1103
		proc.stdout.close()  # pylint: disable=E1101
		proc.stderr.close()  # pylint: disable=E1101
		proc.wait()  # pylint: disable=E1101
		if lines:
			yield n, filterlabels(out, nofunc, nomorph).splitlines(), err
		elif doexport is None:
			yield n, out, err
		else:
			if linenos:
				yield re.sub(r'(^|\n)corpus/', '\\1', out)
			else:
				yield out


def doxpathqueries(query, selected, lines=False, doexport=None,
		linenos=False):
	""" Run xpath query on each text. """
	for n, text in enumerate(TEXTS):
		if n not in selected:
			continue
		out = err = ''
		try:
			out = XMLCORPORA[n].xpath(query)
			if lines:
				yield n, (('%s:::%s:::%s:::%s' % (
							match.name().rsplit('.', 1)[0],
							text,
							XMLCORPORA[n].read(match.name()),
							match.contents())).decode('utf8')
						for match in out), err
			elif doexport is None:
				yield n, ''.join(match.name().replace('.xml', ':::\n')
						for match in out).decode('utf8'), err
			elif doexport == 'sents':
				yield ''.join(('%s%s\n' % (
							(('%s:%s|' % (text,
								match.name().rsplit('.', 1)[0]))
							if linenos else ''),
							ALPINOLEAVES.search(
								XMLCORPORA[n].read(match.name())).group(1)))
							for match in out).decode('utf8')
			elif doexport == 'trees' or doexport == 'brackets':
				yield ''.join(('%s%s' % (
						(('<!-- %s:%s -->\n' % (text, match.name()))
						if linenos else ''),
						XMLCORPORA[n].read(match.name())))
						for match in out).decode('utf8')
		except RuntimeError as err:
			if lines or doexport is None:
				yield n, (), str(err)
			else:
				yield str(err)


def doregexqueries(query, selected, lines=False, doexport=None,
		linenos=False):
	""" Run regex query on tokenized sentences. """
	for n, text in enumerate(TEXTS):
		if n not in selected:
			continue
		out = err = ''
		try:
			regex = re.compile(query)
			out = enumerate((regex.search(a) for a in tokenized(text)), 1)
			if lines:
				yield n, (('%d:::%s:::%s:::%s' % (n, text,
						match.string, match.group())).decode('utf8')
						for n, match in out if match is not None), err
			elif doexport is None:
				yield n, ''.join('%d:::\n' % n
						for n, match in out if match is not None), err
			elif doexport == 'sents':
				yield ''.join('%s%s\n' % (
						('%s:%d|' % (text, n)) if linenos else '',
						match.string)
						for n, match in out if match is not None)
			else:
				raise NotImplementedError
		except re.error as err:
			if lines:
				yield n, (), str(err)
			elif doexport is None:
				yield n, '', str(err)
			else:
				yield str(err)


def filterlabels(line, nofunc, nomorph):
	""" Optionally remove morphological and grammatical function labels
	from parse tree. """
	if nofunc:
		line = FUNC_TAGS.sub('', line)
	if nomorph:
		line = MORPH_TAGS.sub(lambda g: '%s%s' % (
				ABBRPOS.get(g.group(1), g.group(1)), g.group(2)), line)
	return line


def barplot(data, total, title, width=800.0, unit=''):
	""" A HTML bar plot given a dictionary and max value. """
	result = ['<div class=barplot>',
			('<text style="font-family: sans-serif; font-size: 16px; ">'
			'%s</text>' % title)]
	firstletters = {key[0] for key in data}
	color = {}
	if len(firstletters) <= 5:
		color.update(zip(firstletters, range(1, 6)))
	for key in sorted(data, key=data.get, reverse=True):
		result.append('<br><div style="width:%dpx;" class=b%d></div>'
				'<span>%s: %g %s</span>' % (round(width * data[key] / total),
				color.get(key[0], 1) if data[key] else 0,
				cgi.escape(key), data[key], unit,))
	result.append('</div>\n')
	return '\n'.join(result)


def concplot(indices, total, width=800.0):
	""" Draw a concordance plot from a sequence of indices and the total number
	of items. """
	result = ('\t<svg version="1.1" xmlns="http://www.w3.org/2000/svg"'
			' width="%dpx" height="10px" >\n'
			'<rect x=0 y=0 width="%dpx" height=10 '
			'fill=white stroke=black />\n' % (width, width))
	if indices:
		strokes = []
		start = 0
		seq = [-1] + sorted(indices) + [-1]
		for prev, idx, nextidx in zip(seq, seq[1:], seq[2:]):
			if idx != prev + 1 and idx != nextidx - 1:
				strokes.append('M %d 0v 10' % idx)
			elif idx != prev + 1:
				start = idx
			elif idx != nextidx - 1:
				strokes.append(  # draw a rectangle covering start:idx
						'M %d 0l 0 10 %d 0 0 -10' % (start, idx - start))
		result += ('<g transform="scale(%g, 1)">\n'
				'<path stroke=black d="%s" /></g>' % (
				width / total, ''.join(strokes)))
	return result + '</svg>'


def selectedtexts(form):
	""" Find available texts and parse selected texts argument. """
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


def preparecorpus():
	""" Produce indexed versions of parse trees in .mrg files """
	for a in sorted(glob.glob(os.path.join(CORPUS_DIR, '*.mrg'))):
		if not os.path.exists(a + '.t2c.gz'):
			subprocess.check_call(
					args=[which('tgrep2'), '-p', a, a + '.t2c.gz'],
					shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def tokenized(text):
	""" Return iterable with tokenized sentences of a text, one sentence at a
	time, in the form of a byte string (with newline). """
	base = os.path.join(CORPUS_DIR, text)
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


def getstyletable():
	""" Run style(1) on all files and store results in a dictionary. """
	files = glob.glob(os.path.join(CORPUS_DIR, '*.txt'))
	if not files:
		files = TEXTS
	styletable = {}
	for filename in sorted(files):
		cmd = [which('style'), '--language', STYLELANG]
		stdin = subprocess.PIPE
		proc = subprocess.Popen(args=cmd, shell=False, bufsize=-1,
				stdin=stdin, stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT)
		if filename.endswith('.txt'):
			# .txt files may have one paragraph per line;
			# style expects paragraphs separated by two linebreaks.
			proc.stdin.write(open(filename).read().replace('\n', '\n\n'))
		else:
			proc.stdin.writelines(tokenized(filename))  # pylint: disable=E1101
		proc.stdin.close()  # pylint: disable=E1101
		out = proc.stdout.read()  # pylint: disable=E1101
		proc.stdout.close()  # pylint: disable=E1101
		proc.wait()  # pylint: disable=E1101
		name = os.path.basename(filename)
		styletable[name] = parsestyleoutput(out)
	return styletable


def parsestyleoutput(out):
	""" Extract readability grades, averages, and percentages from style output
	(i.e., all figures except for absolute counts). """
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
	""" Get list of files and number of lines in them. """
	texts = []
	xmlcorpora = []
	if os.path.exists('/tmp/treesearchcorpus.pickle'):
		texts, numsents, numconst, numwords, styletable = pickle.load(
				open('/tmp/treesearchcorpus.pickle'))
	tfiles = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.mrg')))
	afiles = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.dact')))
	assert tfiles or afiles, 'no .mrg or .dact files found in %s' % CORPUS_DIR
	if tfiles and afiles:
		assert len(tfiles) == len(afiles) and all(
				t.rsplit('.', 1)[0] == a.rsplit('.', 1)[0]
				for a, t in zip(tfiles, afiles)), (
				'expected either .mrg or .dact files, '
				'or corresponding .mrg and .dact files')
	if afiles:
		assert ALPINOCORPUSLIB, ('.dact files found in corpus '
				'but could not import alpinocorpus library.')
		try:
			xmlcorpora = [alpinocorpus.CorpusReader(filename,
					macrosFilename='static/xpathmacros.txt')
					for filename in afiles]
		except TypeError:
			xmlcorpora = [alpinocorpus.CorpusReader(filename)
					for filename in afiles]
	picklemtime = 0
	if os.path.exists('/tmp/treesearchcorpus.pickle'):
		picklemtime = os.stat('/tmp/treesearchcorpus.pickle').st_mtime
	scriptmtime = (os.stat(sys.argv[0]).st_mtime
			if os.path.exists(sys.argv[0]) else 0)
	currentfiles = {os.path.splitext(os.path.basename(filename))[0]
		for filename in tfiles + afiles}
	if (scriptmtime > picklemtime or set(texts) != currentfiles or
			any(os.stat(a).st_mtime > picklemtime for a in tfiles + afiles)):
		if tfiles:
			numsents = [len(open(filename).readlines())
					for filename in tfiles if filename.endswith('.mrg')]
			numconst = [open(filename).read().count('(')
					for filename in tfiles if filename.endswith('.mrg')]
			numwords = [len(GETLEAVES.findall(open(filename).read()))
					for filename in tfiles if filename.endswith('.mrg')]
		elif ALPINOCORPUSLIB:
			numsents = [corpus.size() for corpus in xmlcorpora]
			numconst, numwords = [], []
			for n, corpus in enumerate(xmlcorpora):
				const = words = 0
				for entry in corpus.entries():
					const += entry.contents().count('<node ')
					words += entry.contents().count('word=')
				numconst.append(const)
				numwords.append(words)
				try:  # overwrite previous instance as garbage collection kludge
					xmlcorpora[n] = alpinocorpus.CorpusReader(filename,
								macrosFilename='static/xpathmacros.txt')
				except TypeError:
					xmlcorpora[n] = alpinocorpus.CorpusReader(filename)
				print(afiles[n])
		texts = [os.path.splitext(os.path.basename(a))[0]
				for a in tfiles or afiles]
		styletable = getstyletable()
	pickle.dump((texts, numsents, numconst, numwords, styletable),
			open('/tmp/treesearchcorpus.pickle', 'wb'), protocol=-1)
	return texts, numsents, numconst, numwords, styletable, xmlcorpora


def stream_template(template_name, **context):
	""" From Flask documentation: pass an iterator to a template. """
	APP.update_template_context(context)
	templ = APP.jinja_env.get_template(template_name)
	result = templ.stream(context)
	result.enable_buffering(5)
	return result


def which(program):
	""" Return first match for program in search path. """
	for path in os.environ.get('PATH', os.defpath).split(":"):
		if path and os.path.exists(os.path.join(path, program)):
			return os.path.join(path, program)
	raise ValueError('%r not found in path; please install it.' % program)

preparecorpus()
TEXTS, NUMSENTS, NUMCONST, NUMWORDS, STYLETABLE, XMLCORPORA = getcorpus()
ALPINOCORPUSLIB = ALPINOCORPUSLIB and XMLCORPORA
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


if __name__ == '__main__':
	logging.basicConfig()
	for log in (logging.getLogger(), APP.logger):
		log.setLevel(logging.DEBUG)
		log.handlers[0].setFormatter(logging.Formatter(
				fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
	#APP.logger.info('ready')
	APP.run(debug=True, host='0.0.0.0')
