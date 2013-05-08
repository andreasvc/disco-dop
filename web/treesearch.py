""" Web interface to search a treebank. Requires Flask, tgrep2.
Expects one or more treebanks in the directory corpus/ """
# stdlib
import os
import re
import glob
import logging
import subprocess
from urllib import quote
from datetime import datetime, timedelta
from itertools import islice, count
from collections import Counter
from functools import wraps
# Flask & co
from flask import Flask, Markup, Response
from flask import request, render_template
#from flask import send_from_directory
from werkzeug.contrib.cache import SimpleCache
# disco-dop
from discodop.tree import Tree
from discodop.treedraw import DrawTree

CORPUS_DIR = "corpus/"
TGREP_BIN = '/usr/local/bin/tgrep2'

APP = Flask(__name__)
MORPH_TAGS = re.compile(
		r'\(([_*A-Z0-9]+)(?:\[[^ ]*\][0-9]?)?((?:-[_A-Z0-9]+)?(?:\*[0-9]+)? )')
FUNC_TAGS = re.compile(r'-[_A-Z0-9]+')
GETLEAVES = re.compile(r" ([^ ()]+)(?=[ )])")

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

def stream_template(template_name, **context):
	""" From Flask documentation. """
	APP.update_template_context(context)
	templ = APP.jinja_env.get_template(template_name)
	result = templ.stream(context)
	result.enable_buffering(5)
	return result

def cached(timeout=3600):
	def decorator(func):
		func.cache = SimpleCache()
		@wraps(func)
		def decorated_function():
			key = (request.path, ) + tuple(sorted(request.args.items()))
			result = func.cache.get(key)
			if result is None:
				result = func()
				func.cache.set(key, result, timeout=timeout)
			return result
		return decorated_function
	return decorator

@APP.route('/')
@APP.route('/counts')
@APP.route('/trees')
@APP.route('/sents')
@APP.route('/brackets')
#@cached(timeout=24 * 3600) #FIXME does not work with generators
def main():
	""" Main search form & results page. """
	output = None
	if 'output' in request.args:
		output = request.args['output']
	elif request.path != '/':
		output = request.path.lstrip('/')
	texts, selected = selectedtexts(request.args)
	if output:
		if output not in DISPATCH:
			return 'Invalid argument', 404
		return Response(stream_template('searchresults.html', form=request.args,
				texts=texts, selectedtexts=selected,
				output=output, results=DISPATCH[output](request.args)))
	return render_template('search.html', form=request.args,
			texts=texts, selectedtexts=selected)

@APP.route('/style')
def style():
	""" Use style(1) program to get staticstics for each text. """
	# TODO: tabulate & plot
	def generate():
		""" Generator for results. """
		files = glob.glob('corpus/*.txt')
		if files:
			yield "NB: formatting errors may distort paragraph counts etc.\n\n"
		else:
			files = glob.glob('corpus/*.t2c.gz')
			yield ("No .txt files found in corpus/\n"
					"Using sentences extracted from parse trees.\n"
					"Text files with original formatting are preferrable.\n\n")
		for a in files:
			if a.endswith('.t2c.gz'):
				tgrep = subprocess.Popen(args=['tgrep2', '-t', '-c', a, '*'],
						shell=False, stdout=subprocess.PIPE)
				cmd = ['/usr/bin/style']
				stdin = tgrep.stdout
			else:
				#cmd = ['/usr/local/bin/style', '--language', 'nl', a]
				cmd = ['/usr/bin/style', a]
				stdin = None
			proc = subprocess.Popen(args=cmd, shell=False, stdin=stdin,
					stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
			yield "%s\n%s\n%s\n\n" % (os.path.basename(a), '=' * len(a),
					proc.communicate()[0])
	resp = Response(generate(), mimetype='text/plain')
	resp.headers['Cache-Control'] = 'max-age=604800, public'
	#set Expires one day ahead (according to server time)
	resp.headers['Expires'] = (
		datetime.utcnow() + timedelta(7, 0 )
		).strftime('%a, %d %b %Y %H:%M:%S UTC')
	return resp

@APP.route('/export')
def export():
	""" Export search results to a file for download. """
	# NB: no distinction between trees from different texts
	# (Does consttreeviewer support # comments?)
	results = (a[1] + '\n' for a in
			doqueries(request.args, lines=False, doexport=True))
	resp = Response(results, mimetype='text/plain')
	resp.headers['Content-Disposition'] = 'attachment; filename=trees.txt'
	return resp

@APP.route('/draw')
def draw():
	""" Produce a visualization of a tree on a separate page. """
	cnt = count()
	treestr = request.args['tree']
	tree = Tree.parse(treestr, parse_leaf=lambda _: next(cnt))
	sent = re.findall(r" +([^ ()]+)(?=[ )])", treestr)
	return "<pre>%s</pre>" % DrawTree(tree, sent).text(
				unicodelines=True, html=True)

#@APP.route('/favicon.ico')
#def favicon():
#	""" Serve the favicon. """
#	return send_from_directory(os.path.join(APP.root_path, 'static'),
#			'favicon.ico', mimetype='image/vnd.microsoft.icon')

def counts(form):
	""" Produce counts of matches for each text. """
	# todo:
	# - name should not be cut off by bar;
	#   separate div for bar in background with abs. pos.:
	#   z-index:1;position:absolute;top:0;background:PaleTurquoise;border:1px
	#   solid #a6a6a6
	# - show dispersion of matches in each text
	cnts = Counter()
	relfreq = {}
	sumtotal = 0
	norm = form.get('norm', 'sents')
	gotresult = False
	for n, (text, results, stderr) in enumerate(
			doqueries(form, lines=False)):
		if n == 0:
			gotresult = True
			yield ("Query: %s\n"
					"NB: the counts reflect the total number of "
					"times a pattern matches for each tree.\n\n"
					"Counts:\n" % stderr)
		cnt = results.count('\n')
		text = text.replace('.t2c.gz', '')
		filename = os.path.join(CORPUS_DIR, text)
		cnts[text] = cnt
		if norm == 'sents':
			total = len(open(filename).readlines())
		elif norm == 'consts':
			total = open(filename).read().count('(')
		elif norm == 'words':
			total = len(GETLEAVES.findall(open(filename).read()))
		else:
			raise ValueError
		relfreq[text] = float(cnt) / total
		sumtotal += total
		line = "%s%6d    %5.2f %%" % (
				text.ljust(40)[:40], cnt, 100 * relfreq[text])
		if cnt:
			yield line + '\n'
		else:
			yield '<span style="color: black; ">%s</span>\n' % line
	if gotresult:
		yield ("%s%6d    %5.2f %%\n</span>" % (
				"TOTAL".ljust(40),
				sum(cnts.values()),
				100.0 * sum(cnts.values()) / sumtotal))
		sortedcounts = repr(sorted(cnts, key=cnts.get))
		maxcounts = max(cnts.values())
		sortedrelfreq = repr(sorted(relfreq, key=relfreq.get))
		yield APP.jinja_env.get_template('graph').render(
				data=repr(dict(cnts)), sortedcounts=sortedcounts,
				maxcounts=maxcounts, relfreq=repr(relfreq),
				sortedrelfreq=sortedrelfreq)
	else:
		yield 'no results.'

def trees(form):
	""" Return visualization of parse trees in search results. """
	# TODO:
	# - allow to view context of x number of sentences.
	# - paginate to more than 10 matching trees per text
	for n, (text, results, stderr) in enumerate(
			doqueries(form, lines=True)):
		if n == 0:
			yield ("Query: %s\n"
					"Trees (showing up to 10 per text):\n" % stderr)
		for m, line in enumerate(islice(results, 10)):
			if m == 0:
				yield ("==&gt; %s: [<a href=\"javascript: toggle('n%d'); \">"
						"toggle</a>]\n<span id=n%d>" % (text, n + 1, n + 1))
			lineno, text, treestr, match = line.split(":::")
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
			except Exception as err:
				line = "%s \nERROR: %s\n%s\n%s\n" % (lineno, err, treestr, tree)
			else:
				line = "%s\n%s\n" % (lineno, treerepr)
			yield Markup(line)
		yield "</span>"

def sents(form, dobrackets=False):
	""" Return search results as terminals or in bracket notation. """
	for n, (text, results, stderr) in enumerate(
			doqueries(form, lines=True)):
		if n == 0:
			yield ("Query: %s\n"
					"Trees (showing up to 1000 per text):\n" % stderr)
		for m, line in enumerate(islice(results, 1000)):
			if m == 0:
				yield ("%s: [<a href=\"javascript: toggle('n%d'); \">toggle</a>]"
						"<ol id=n%d>" % (text, n, n))
			lineno, text, treestr, match = line.rstrip().split(":::")
			treestr = treestr.replace(" )", " -NONE-)")
			link = "<a href='/draw?tree=%s'>draw</a>" % (
					quote(treestr.encode('utf8')))
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
			yield "<li>%s [%s] %s" % (lineno.rjust(6), link, Markup(out))
		yield "</ol>"

def brackets(form):
	""" Wrapper. """
	return sents(form, dobrackets=True)

def doqueries(form, lines=False, doexport=False):
	""" Run tgrep2 on each text """
	texts, selected = selectedtexts(form)
	for n, text in enumerate(texts):
		if n not in selected:
			continue
		cmd = [TGREP_BIN, '-z', '-a',
				'-m', (r"%w\n" if doexport else r"#%s:::%f:::%w:::%h\n"),
				'-c', os.path.join(CORPUS_DIR, text + '.t2c.gz'),
				form['query']]
		proc = subprocess.Popen(args=cmd,
				shell=False,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE)
		if n == 0:
			logging.debug(' '.join(cmd + [text + '.t2c.gz']))
		if lines:
			yield text, (filterlabels(form, a.decode('utf8'))
					for a in proc.stdout), proc.stderr.read()
		else:
			yield text, proc.stdout.read().decode('utf8'), proc.stderr.read()

def filterlabels(form, line):
	""" Optionally remove morphological and grammatical function labels
	from parse tree. """
	if 'nofunc' in form:
		line = FUNC_TAGS.sub('', line)
	if 'nomorph' in form:
		#line = MORPH_TAGS.sub(r'(\1\2', line)
		line = MORPH_TAGS.sub(lambda g: '(%s%s' % (
				ABBRPOS.get(g.group(1), g.group(1)), g.group(2)), line)
	return line

def selectedtexts(form):
	""" Find available texts and parse selected texts argument. """
	files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.mrg")))
	texts = [os.path.basename(a) for a in files]
	selected = set()
	if 'texts' in form:
		for a in filter(None, form['texts'].split(',')):
			if '-' in a:
				b, c = a.split('-')
				selected.update(n for n in range(int(b), int(c)))
			else:
				selected.add(int(a))
	elif 't' in form:
		selected.update(int(n) for n in form.getlist('t'))
	else:
		selected.update(range(len(texts)))
	return texts, selected

def preparecorpus():
	""" Produce indexed versions of parse trees in .mrg files """
	files = glob.glob('corpus/*.mrg')
	assert files, "Expected one or more .mrg files with parse trees in corpus/"
	for a in files:
		if not os.path.exists(a + '.t2c.gz'):
			subprocess.check_call(args=[TGREP_BIN, '-p', a, a + '.t2c.gz'],
					shell=False,
					stdout=subprocess.PIPE,
					stderr=subprocess.PIPE)

# this is redundant but needed to support both javascript-enabled /foo
# as well as non-javascript /?output=foo
DISPATCH = {
	'counts': counts,
	'trees': trees,
	'sents': sents,
	'brackets': brackets,
	#'style': style,
	#'export': export
}

if __name__ == '__main__':
	logging.basicConfig()
	for log in (logging.getLogger(), APP.logger):
		log.setLevel(logging.DEBUG)
		log.handlers[0].setFormatter(logging.Formatter(
				fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
	preparecorpus()
	APP.run(debug=True, host='0.0.0.0')
