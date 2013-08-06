""" Web interface to search a treebank. Requires Flask, tgrep2.
Expects one or more treebanks in the directory corpus/ """
# stdlib
import os
import re
import cgi
import glob
import logging
import tempfile
import subprocess
from heapq import nlargest
from urllib import quote
from datetime import datetime, timedelta
from itertools import islice, count
from collections import Counter
#from functools import wraps
# Flask & co
from flask import Flask, Response
from flask import request, render_template, send_from_directory
#from werkzeug.contrib.cache import SimpleCache
# disco-dop
from discodop.tree import Tree
from discodop.treedraw import DrawTree
from discodop import fragments

CORPUS_DIR = "corpus/"
MINFREQ = 2  # filter out fragments which occur just once or twice
MINNODES = 3  # filter out fragments with only three nodes (CFG productions)
TREELIMIT = 10  # max number of trees to draw in search resuluts
FRAGLIMIT = 250  # max amount of search results for fragment extraction
SENTLIMIT = 1000  # max number of sents/brackets in search results

APP = Flask(__name__)
MORPH_TAGS = re.compile(
		r'([_*A-Z0-9]+)(?:\[[^ ]*\][0-9]?)?((?:-[_A-Z0-9]+)?(?:\*[0-9]+)? )')
FUNC_TAGS = re.compile(r'-[_A-Z0-9]+')
GETLEAVES = re.compile(r" ([^ ()]+)(?=[ )])")
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


def preparecorpus():
	""" Produce indexed versions of parse trees in .mrg files """
	files = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.mrg')))
	assert files, 'Expected one or more .mrg files with parse trees in corpus/'
	for a in files:
		if not os.path.exists(a + '.t2c.gz'):
			subprocess.check_call(
					args=[which('tgrep2'), '-p', a, a + '.t2c.gz'],
					shell=False,
					stdout=subprocess.PIPE,
					stderr=subprocess.PIPE)


def getcorpus():
	""" Get list of files and number of lines in them. """
	files = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.mrg')))
	texts = [os.path.basename(a) for a in files]
	numsents = [len(open(filename).readlines()) for filename in files]
	numconst = [open(filename).read().count('(') for filename in files]
	numwords = [len(GETLEAVES.findall(open(filename).read()))
			for filename in files]
	return texts, numsents, numconst, numwords


fragments.PARAMS.update(disc=False, debug=False, cover=False, complete=False,
		quadratic=False, complement=False, quiet=True, nofreq=False,
		approx=True, indices=False)
preparecorpus()
TEXTS, NUMSENTS, NUMCONST, NUMWORDS = getcorpus()


def stream_template(template_name, **context):
	""" From Flask documentation. """
	APP.update_template_context(context)
	templ = APP.jinja_env.get_template(template_name)
	result = templ.stream(context)
	result.enable_buffering(5)
	return result


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
		return Response(stream_template('searchresults.html',
				form=request.args, texts=TEXTS, selectedtexts=selected,
				output=output, results=DISPATCH[output](request.args)))
	return render_template('search.html', form=request.args, output='counts',
			texts=TEXTS, selectedtexts=selected)


def parsestyleoutput(out):
	""" Extract readability grades, averages, and percentages from style output
	(i.e., all figures except for absolute counts). """
	result = {}
	for key, val in READGRADERE.findall(out):
		result[key] = float(val)
	for key1, val, key2 in AVERAGERE.findall(out):
		result['average %s per %s' % (key2, key1[:-1])] = float(val)
	for key, val in PERCENTAGE1RE.findall(out):
		result['%% %s' % key] = float(val)
	for val, key in PERCENTAGE2RE.findall(out):
		result['%% %s' % key] = float(val)
	return result


@APP.route('/style')
def style():
	""" Use style(1) program to get staticstics for each text. """
	def generate(lang, doexport):
		""" Generator for results. """
		files = glob.glob('corpus/*.txt')
		if files and not doexport:
			yield "NB: formatting errors may distort paragraph counts etc.\n\n"
		else:
			files = glob.glob('corpus/*.t2c.gz')
			if not doexport:
				yield ("No .txt files found in corpus/\n"
						"Using sentences extracted from parse trees.\n"
						"Supply text files with original formatting\n"
						"to get meaningful paragraph information.\n\n")
		table = {}
		for n, a in enumerate(sorted(files)):
			if a.endswith('.t2c.gz'):
				tgrep = subprocess.Popen(
						args=[which('tgrep2'), '-t', '-c', a, '*'],
						shell=False, bufsize=-1, stdout=subprocess.PIPE)
				cmd = [which('style'), '--language', lang]
				stdin = tgrep.stdout  # pylint: disable=E1101
			else:
				cmd = [which('style'), '--language', lang, a]
				stdin = None
			if n == 0 and not doexport:
				yield '<pre>' + ' '.join(cmd) + '\n\n'
			proc = subprocess.Popen(args=cmd, shell=False, bufsize=-1,
					stdin=stdin, stdout=subprocess.PIPE,
					stderr=subprocess.STDOUT)
			if a.endswith('.t2c.gz'):
				tgrep.wait()  # pylint: disable=E1101
			out = proc.stdout.read()  # pylint: disable=E1101
			if proc.stdin:  # pylint: disable=E1101
				proc.stdin.close()  # pylint: disable=E1101
			proc.stdout.close()  # pylint: disable=E1101
			proc.wait()  # pylint: disable=E1101
			if a.endswith('.t2c.gz'):
				tgrep.stdout.close()  # pylint: disable=E1101
				tgrep.wait()  # pylint: disable=E1101
			name = os.path.basename(a)
			table[name] = parsestyleoutput(out)
			if doexport:
				if n == 0:
					yield ','.join('"%s"' % key for key in sorted(table[name]))
				yield ','.join('%s' % val
						for _, val in sorted(table[name].items()))
			else:
				yield "%s\n%s\n%s\n\n" % (name, '=' * len(name), out)
		if not doexport:
			yield '</pre>'
		# produce a plot for each field
		for a in table:
			fields = sorted(table[a].keys())
			break
		for field in () if doexport else fields:
			data = {a: table[a][field] for a in table}
			total = 100 if '%' in field else max(data.values())
			if total > 0:
				yield barplot(data, total, field + ':', barstyle='chart1',
						unit='%' if '%' in field else '')
	if 'export' in request.args:
		resp = Response(generate(request.args.get('lang', 'en'), True),
				mimetype='text/plain')
		resp.headers['Content-Disposition'] = 'attachment; filename=style.csv'
	else:
		resp = Response(stream_template('searchresults.html',
				form=request.args, texts=TEXTS, selectedtexts=(),
				output='style', results=generate(
					request.args.get('lang', 'en'), False)))
	resp.headers['Cache-Control'] = 'max-age=604800, public'
	#set Expires one day ahead (according to server time)
	resp.headers['Expires'] = (
		datetime.utcnow() + timedelta(7, 0)).strftime(
		'%a, %d %b %Y %H:%M:%S UTC')
	return resp


def export(form, output):
	""" Export search results to a file for download. """
	# NB: no distinction between trees from different texts
	# (Does consttreeviewer support # comments?)
	if output == 'counts':
		results = counts(form, doexport=True)
		filename = 'counts.csv'
	else:
		results = (a + '\n' for a in doqueries(
				form, lines=False, doexport=output))
		filename = '%s.txt' % output
	resp = Response(results, mimetype='text/plain')
	resp.headers['Content-Disposition'] = 'attachment; filename=' + filename
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
	filename = os.path.join(CORPUS_DIR, TEXTS[textno].replace('.t2c.gz', ''))
	treestr = next(islice(open(filename), sentno - 1, sentno)).decode('utf8')
	return '<pre id="t%s">%s</pre>' % (sentno, DrawTree(
			filterlabels(treestr, nofunc, nomorph)).text(
				unicodelines=True, html=True))


@APP.route('/browse')
def browse():
	""" Browse through trees in a file. """
	chunk = 10  # number of trees to fetch for one request
	if 'text' in request.args and 'sent' in request.args:
		textno = int(request.args['text'])
		sentno = int(request.args['sent']) - 1
		filename = os.path.join(CORPUS_DIR,
				TEXTS[textno].replace('.t2c.gz', ''))
		start = sentno - sentno % chunk
		maxtree = min(start + chunk, NUMSENTS[textno])
		nofunc = 'nofunc' in request.args
		nomorph = 'nomorph' in request.args
		lines = islice(open(filename), start, maxtree)
		results = ['<pre id="t%s"%s>%s</pre>' % (n + 1,
				' style="display: none; "' if 'ajax' in request.args else '',
				DrawTree(filterlabels(
					line.decode('utf8'), nofunc, nomorph)).text(
					unicodelines=True, html=True))
				for n, line in enumerate(lines, start)]
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


def counts(form, doexport=False):
	""" Produce counts of matches for each text. """
	cnts = Counter()
	relfreq = {}
	sumtotal = 0
	norm = form.get('norm', 'sents')
	gotresult = False
	for n, (textno, results, stderr) in enumerate(doqueries(form, lines=False)):
		if n == 0:
			if doexport:
				yield '"text","count","relfreq"\r\n'
			else:
				url = 'counts?query=%s&norm=%s&texts=%s&export=1' % (
						form['query'], form['norm'], form['texts'])
				yield ('Query: %s\n'
						'NB: the counts reflect the total number of '
						'times a pattern matches for each tree.\n\n'
						'Counts (<a href="%s">export to CSV</a>):\n' % (
						stderr, url))
		cnt = results.count('\n')
		if cnt:
			gotresult = True
		text = TEXTS[textno].replace('.t2c.gz', '')
		cnts[text] = cnt
		total = totalsent = NUMSENTS[textno]
		if norm == 'consts':
			total = NUMCONST[textno]
		elif norm == 'words':
			total = NUMWORDS[textno]
		elif norm != 'sents':
			raise ValueError
		relfreq[text] = 100.0 * cnt / total
		sumtotal += total
		if doexport:
			yield '"%s",%d,%g\r\n' % (text, cnt, relfreq[text])
		else:
			line = "%s%6d    %5.2f %%" % (
					text.ljust(40)[:40], cnt, relfreq[text])
			indices = {int(line[:line.index(':::')])
					for line in results.splitlines()}
			plot = concplot(indices, totalsent)
			if cnt:
				yield line + plot + '\n'
			else:
				yield '<span style="color: gray; ">%s%s</span>\n' % (line, plot)
	if gotresult and not doexport:
		yield ("%s%6d    %5.2f %%\n</span>\n" % (
				"TOTAL".ljust(40),
				sum(cnts.values()),
				100.0 * sum(cnts.values()) / sumtotal))
		yield barplot(relfreq, max(relfreq.values()),
				'Relative frequency of pattern: (count / num_%s * 100)' % norm,
				barstyle='chart1', unit='%')
		#yield barplot(cnts, max(cnts.values()), 'Absolute counts of pattern',
		#		barstyle='chart2')


def trees(form):
	""" Return visualization of parse trees in search results. """
	# TODO: show context of x sentences around result, offer pagination.
	gotresults = False
	for n, (_textno, results, stderr) in enumerate(
			doqueries(form, lines=True)):
		if n == 0:
			# NB: we do not hide function or morphology tags when exporting
			url = 'trees?query=%s&texts=%s&export=1' % (
					form['query'], form['texts'])
			yield ('Query: %s\n'
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
			try:
				treerepr = DrawTree(tree, sent, highlight=high).text(
						unicodelines=True, html=True)
			except ValueError as err:
				line = "#%s \nERROR: %s\n%s\n%s\n" % (lineno, err, treestr, tree)
			else:
				line = "#%s\n%s\n" % (lineno, treerepr)
			yield line
		yield "</span>"
	if not gotresults:
		yield "No matches."


def sents(form, dobrackets=False):
	""" Return search results as terminals or in bracket notation. """
	gotresults = False
	for n, (textno, results, stderr) in enumerate(
			doqueries(form, lines=True)):
		if n == 0:
			url = '%s?query=%s&texts=%s&export=1' % (
					'trees' if dobrackets else 'sents',
					form['query'], form['texts'])
			yield ('Query: %s\n'
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
			treestr = cgi.escape(treestr.replace(" )", " -NONE-)"))
			link = "<a href='/browse?text=%d&sent=%s%s%s'>draw</a>" % (
					textno, lineno,
					'&nofunc' if 'nofunc' in form else '',
					'&nomorph' if 'nomorph' in form else '')
			if dobrackets:
				out = treestr.replace(match,
						"<span class=r>%s</span>" % match)
			else:
				pre, highlight, post = treestr.partition(match)
				out = "%s <span class=r>%s</span> %s " % (
						' '.join(GETLEAVES.findall(pre)),
						' '.join(GETLEAVES.findall(highlight))
								if '(' in highlight else highlight,
						' '.join(GETLEAVES.findall(post)))
			yield "<li>#%s [%s] %s" % (lineno.rjust(6), link, out)
		yield "</ol>"
	if not gotresults:
		yield "No matches."


def brackets(form):
	""" Wrapper. """
	return sents(form, dobrackets=True)


def fragmentsinresults(form):
	""" Extract recurring fragments from search results. """
	gotresults = False
	uniquetrees = set()
	for n, (_, results, stderr) in enumerate(
			doqueries(form, lines=True)):
		if n == 0:
			#url = 'fragments?query=%s&texts=%s&export=1' % (
			#		form['query'], form['texts'])
			yield ('Query: %s\n'
					'Fragments (showing up to %d fragments '
					'in the first %d search results from selected texts) '
					#'export: <a href="%s">plain</a>, '
					#'<a href="%s">with line numbers</a>):\n'
					% (stderr, FRAGLIMIT, SENTLIMIT))  # url, url + '&linenos=1'
		for m, line in enumerate(islice(results, SENTLIMIT)):
			if m == 0:
				gotresults = True
			_, _, treestr, _ = line.rstrip().split(":::")
			treestr = treestr.replace(" )", " -NONE-)") + '\n'
			uniquetrees.add(treestr.encode('utf8'))
	if not gotresults:
		yield "No matches."
		return
	# TODO: get counts from whole text (preload); export fragments to file
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
	yield "<ol>"
	for treestr, freq in results:
		gotresults = True
		link = "<a href='/draw?tree=%s'>draw</a>" % (
				quote(treestr.encode('utf8')))
		yield "<li>freq=%3d [%s] %s" % (freq, link, cgi.escape(treestr))
	yield "</ol>"
	if not gotresults:
		yield "No fragments with freq > %d & nodes > %d." % (MINNODES, MINFREQ)
		return


def doqueries(form, lines=False, doexport=None):
	""" Run tgrep2 on each text """
	selected = selectedtexts(form)
	for n, text in enumerate(TEXTS):
		if n not in selected:
			continue
		if doexport == 'sents':
			fmt = r'%f:%s|%tw\n' if form.get('linenos') else r'%tw\n'
		elif doexport == 'trees' or doexport == 'brackets':
			# NB: no distinction between trees from different texts
			# (Does consttreeviewer support # comments?)
			fmt = r"%f:%s|%w\n" if form.get('linenos') else r"%w\n"
		elif not doexport:
			fmt = r'%s:::%f:::%w:::%h\n'
		else:
			raise ValueError
		cmd = [which('tgrep2'), '-z', '-a',
				'-m', fmt,
				'-c', os.path.join(CORPUS_DIR, text + '.t2c.gz'),
				'static/macros.txt',
				form['query']]
		proc = subprocess.Popen(args=cmd,
				bufsize=-1, shell=False,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE)
		out, err = proc.communicate()
		out, err = out.decode('utf8'), err.decode('utf8')
		proc.stdout.close()  # pylint: disable=E1101
		proc.stderr.close()  # pylint: disable=E1101
		proc.wait()  # pylint: disable=E1101
		if lines:
			yield n, filterlabels(out, 'nofunc' in form,
					'nomorph' in form).splitlines(), err
		elif doexport and form.get('linenos'):
			yield out.lstrip('corpus/').replace('\ncorpus/', '\n')
		else:
			yield n, out, err


def filterlabels(line, nofunc, nomorph):
	""" Optionally remove morphological and grammatical function labels
	from parse tree. """
	if nofunc:
		line = FUNC_TAGS.sub('', line)
	if nomorph:
		line = MORPH_TAGS.sub(lambda g: '%s%s' % (
				ABBRPOS.get(g.group(1), g.group(1)), g.group(2)), line)
	return line


def barplot(data, total, title, barstyle='chart1', width=800.0, unit=''):
	""" A HTML bar plot given a dictionary and max value. """
	result = ['</pre><div class=chart>',
			('<text style="font-family: sans-serif; font-size: 16px; ">'
			'%s</text>' % title)]
	for key in sorted(data, key=data.get, reverse=True):
		result.append('<div class=%s style="width: %gpx" >%s: %g %s</div>' % (
				barstyle if data[key] else 'chart',
				width * data[key] / total, key, data[key], unit))
	result.append('</div>\n<pre>')
	return '\n'.join(result)


def concplot(indices, total, width=800.0):
	""" Draw a concordance plot from a sequence of indices and the total number
	of items. """
	result = ('\t<svg version="1.1" xmlns="http://www.w3.org/2000/svg"'
			' width="%dpx" height="10px" >\n'
			'<rect x=0 y=0 width="%dpx" height=10 '
			'fill=white stroke=black />\n' % (width, width))
	if indices:
		result += ('<g transform="scale(%g, 1)">\n'
				'<path stroke=black d="%s" /></g>') % (width / total,
				''.join('M %d 0v 10' % idx for idx in sorted(indices)))
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
	elif 't' in form:
		selected.update(int(n) for n in form.getlist('t'))
	else:
		selected.update(range(len(TEXTS)))
	return selected


def which(program):
	""" Return first match for program in search path. """
	for path in os.environ.get('PATH', os.defpath).split(":"):
		if path and os.path.exists(os.path.join(path, program)):
			return os.path.join(path, program)


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
	preparecorpus()
	APP.run(debug=True, host='0.0.0.0')
