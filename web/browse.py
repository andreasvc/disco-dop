"""Web interface to browse a corpus with various visualizations."""
# stdlib
from __future__ import print_function, absolute_import
import os
import re
import sys
import glob
import math
import logging
from collections import OrderedDict
from functools import wraps
import matplotlib
matplotlib.use('AGG')
import matplotlib.cm as cm
import pandas
# Flask & co
from flask import Flask, Response
from flask import request, render_template
# disco-dop
from discodop import treebank, treebanktransforms
from discodop.tree import DrawTree

DEBUG = False  # when True: enable debugging interface, disable multiprocessing
PASSWD = None  # optionally, dict with user=>pass strings
HEADRULES = '../alpino.headrules'

APP = Flask(__name__)
STANDALONE = __name__ == '__main__'
CORPUS_DIR = "corpus/"

COLORS = dict(enumerate('''\
		Black Red Green Orange Blue Turquoise SlateGray Peru Teal Aqua
		Aquamarine BlanchedAlmond Brown Burlywood CadetBlue Chartreuse
		Chocolate Coral Crimson Cyan Firebrick ForestGreen Fuchsia Gainsboro
		Gold Goldenrod Gray GreenYellow HotPink IndianRed Indigo Khaki Lime
		YellowGreen Magenta Maroon Yellow MidnightBlue Moccasin NavyBlue Olive
		OliveDrab Orchid PapayaWhip Pink Plum PowderBlue Purple RebeccaPurple
		RoyalBlue SaddleBrown Salmon SandyBrown SeaGreen Sienna Silver SkyBlue
		SlateBlue Tan Thistle Tomato Violet Wheat'''.split()))

WORDLIST = pandas.read_table('sonar-word.freqsort.lower.gz',
		encoding='utf8', index_col=0, header=None, names=['word', 'count'],
		nrows=20000).index


def getdeplen(item):
	tree = item.tree.copy(True)
	deps = treebank.dependencies(tree)
	a, b = treebank.deplen(deps)
	return ([abs(x - y) > 7 for x, _, y in deps], a / b if b else 0)
	# cannot highlight due to removing punct
	# return (None, a / b if b else 0)


def getmodifiers(item):
	nodes = list(item.tree.subtrees(lambda n: n.label in ('REL', 'PP')
			and treebanktransforms.function(n) == 'mod'))
	return toboolvec(len(item.sent), {a for x in nodes
		for a in x.leaves()}), len(nodes)


def toboolvec(length, indices):
	return [n in indices for n in range(length)]


# Functions that accept item object with item.tree and item.sent members;
# return tuple (wordhighlights, sentweight).
FILTERS = {
		'average dependency length': getdeplen,
		'd-level': lambda i: (None, treebanktransforms.dlevel(i.tree)),
		'rare words': lambda i: (list(~pandas.Index(
			t.lower() for t in i.sent
			).isin(WORDLIST)
			& pandas.Series([  # filter names
			'eigen' not in n.source[treebank.MORPH]
			for n in
			sorted(i.tree.subtrees(lambda n: isinstance(n[0], int)),
				key=lambda n: n[0])])
			), None),
		'PP/REL modifiers': getmodifiers,
		'punctuation': lambda i: (
			max('.,\'"?!(:;'.find(t) + 1 for t in i.sent)),
		'direct speech': lambda i:
			(None, re.match(r"^- .*$|(?:^|.* )['\"](?: .*|$)",
			' '.join(i.sent)) is not None),
}


def torgb(val, mappable):
	return '#%02x%02x%02x' % mappable.to_rgba(val, bytes=True)[:3]


def charvalues(sent, values):
	"""Project token values to character values.

	>>> sorted(charvalues(['The', 'cat', 'is', 'on', 'the', 'mat'],
	...		[0, 0, 1, 1, 0, 1]))
	[0, 1, 2, 3, 8, 9, 10, 14, 15, 16, 17]
	"""
	assert len(sent) == len(values)
	result = []
	for a, b in zip(sent, values):
		result.extend([b] * (len(a) + 1))
	return result


# http://flask.pocoo.org/snippets/8/
def check_auth(username, password):
	"""This function is called to check if a username / password
	combination is valid."""
	return PASSWD is None or (username in PASSWD
			and password == PASSWD[username])


def authenticate():
	"""Sends a 401 response that enables basic auth."""
	return Response(
			'Could not verify your access level for that URL.\n'
			'You have to login with proper credentials', 401,
			{'WWW-Authenticate': 'Basic realm="Login Required"'})


def requires_auth(f):
	"""Decorator to require basic authentication for route."""
	@wraps(f)
	def decorated(*args, **kwargs):
		"""This docstring intentionally left blank."""
		auth = request.authorization
		if not auth or not check_auth(auth.username, auth.password):
			return authenticate()
		return f(*args, **kwargs)
	return decorated
# end snipppet


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


def addsentweight(x):
	wordhighlights, sentweight = x
	if sentweight is None:
		return wordhighlights, sum(wordhighlights)
	return x


@APP.route('/browse')
@requires_auth
def browsetrees():
	"""Browse through trees in a file."""
	chunk = 20  # number of trees to fetch for one request
	if 'text' in request.args and 'sent' in request.args:
		textno = int(request.args['text'])
		sentno = int(request.args['sent'])
		start = max(1, sentno - sentno % chunk)
		stop = start + chunk
		nofunc = 'nofunc' in request.args
		nomorph = 'nomorph' in request.args
		filename = os.path.join(CORPUS_DIR, TEXTS[textno] + '.export')
		trees = CORPORA[filename].itertrees(start, stop)
		results = ['<pre id="t%s"%s>%s\n%s</pre>' % (n,
				' style="display: none; "' if 'ajax' in request.args else '',
				', '.join('%s: %.3g' % (f, addsentweight(FILTERS[f](item))[1])
					for f in sorted(FILTERS)),
				DrawTree(item.tree, item.sent).text(
					unicodelines=True, html=True))
				for n, (_key, item) in enumerate(trees, start)]
		if 'ajax' in request.args:
			return '\n'.join(results)

		prevlink = '<a id=prev>prev</a>'
		if sentno > chunk:
			prevlink = '<a href="browse?text=%d;sent=%d" id=prev>prev</a>' % (
					textno, sentno - chunk + 1)
		nextlink = '<a id=next>next</a>'
		nextlink = '<a href="browse?text=%d;sent=%d" id=next>next</a>' % (
				textno, sentno + chunk + 1)
		return render_template('browse.html', textno=textno, sentno=sentno,
				text=TEXTS[textno], totalsents=1000,
				trees=results, prevlink=prevlink, nextlink=nextlink,
				chunk=chunk, nofunc=nofunc, nomorph=nomorph,
				mintree=start, maxtree=stop)
	return '<h1>Browse through trees</h1>\n<ol>\n%s</ol>\n' % '\n'.join(
			'<li><a href="browse?text=%d;sent=1;nomorph">%s</a> ' % (n, text)
			for n, text in enumerate(TEXTS))


@APP.route('/')
@APP.route('/browsesents')
@requires_auth
def browsesents():
	"""Browse through sentences in a file; highlight selectable features."""
	chunk = 20  # number of sentences per page
	if 'text' in request.args and 'sent' in request.args:
		textno = int(request.args['text'])
		sentno = int(request.args['sent'])
		sentno = max(chunk // 2 + 1, sentno)
		start = max(1, sentno - chunk // 2)
		stop = start + chunk
		filename = os.path.join(CORPUS_DIR, TEXTS[textno] + '.export')
		feat = request.args.get('feat', next(iter(FILTERS)))
		trees = list(CORPORA[filename].itertrees(start, stop))
		results = []
		values = [addsentweight(FILTERS[feat](item))
				for n, (_key, item) in enumerate(trees, start)]
		norm = matplotlib.colors.Normalize(
				vmin=0, vmax=max(a for _, a in values) * 2)
		mappable = cm.ScalarMappable(norm, 'YlOrBr')
		for n, ((_key, item), (wordhighlights, sentweight)) in enumerate(
				zip(trees, values), start):
			if sentweight is None:
				sentweight = sum(wordhighlights)
			if wordhighlights is not None:
				xsent = applyhighlight(
						' '.join(item.sent), None, None,
							colorvec=charvalues(item.sent, wordhighlights))
			else:
				xsent = ' '.join(item.sent)
			results.append(
					'<a href="browse?text=%d;sent=%d" '
					'style="text-decoration: none; color: black;">'
					'<span style="background: %s; " title="%s: %.3g">'
					' %s </span></a>' % (textno, n,
					torgb(sentweight, mappable), feat, sentweight, xsent))
		legend = 'Feature: [ %s ]<br>' % ', '.join(f if f == feat
				else ('<a href="browsesents?text=%d;sent=%d;feat=%s">'
						'%s</a>' % (textno, sentno, f, f))
				for f in sorted(FILTERS))
		legend += 'Legend: ' + ''.join(
				'<span style="background-color: %s; width: 30px; '
				'display: inline-block; text-align: center; ">'
				'%d</span>' % (torgb(n, mappable), n)
				for n in range(0,
					int(math.ceil(max(a for _, a in values))) + 1))
		prevlink = '<a id=prev>prev</a>'
		if sentno > chunk:
			prevlink = (
					'<a href="browsesents?text=%d;sent=%d;feat=%s" id=prev>'
					'prev</a>' % (textno, sentno - chunk, feat))
		nextlink = '<a id=next>next</a>'
		nextlink = ('<a href="browsesents?text=%d;sent=%d;feat=%s" id=next>'
				'next</a>' % (textno, sentno + chunk, feat))
		return render_template('browsesents.html', textno=textno,
				sentno=sentno, text=TEXTS[textno],
				totalsents='??',  # FIXME
				sents=results, prevlink=prevlink, nextlink=nextlink,
				chunk=chunk, mintree=start, legend=legend,
				query=request.args.get('query', ''),
				engine='')
	return render_template('browsemain.html',
			texts=TEXTS)


def querydict(queries):
	"""Return an OrderedDict of names and queries.

	name is abbreviated query if not given."""
	result = OrderedDict()
	for line in (x for x in queries.splitlines() if x.strip()):
		if ':' in line and line[:line.index(':')].isalnum():
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


def getcorpus():
	"""Get list of files and number of lines in them."""
	files = sorted(glob.glob(os.path.join(CORPUS_DIR, '*.export')))
	assert files, ('no corpus files with extension .export '
			'found.')
	texts = [os.path.splitext(os.path.basename(a))[0] for a in files]
	corpora = {filename: treebank.NegraCorpusReader(filename,
			headrules=HEADRULES, punct='move')
			for filename in files}
	if os.path.exists('metadata.csv'):
		metadata = pandas.read_csv('metadata.csv', index_col=0)
		assert set(metadata.index) == set(texts), (
				'metadata.csv does not match list of files.\n'
				'only in metadata: %s\nonly in files: %s' % (
				set(metadata.index) - set(texts),
				set(texts) - set(metadata.index)))
		metadata = metadata.loc[texts]
	else:
		metadata = None
	return texts, corpora, metadata


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

logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.handlers[0].setFormatter(logging.Formatter(
		fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
log.info('loading corpus.')
if STANDALONE:
	from getopt import gnu_getopt, GetoptError
	try:
		opts, _args = gnu_getopt(sys.argv[1:], '',
				['port=', 'ip=', 'numproc=', 'debug'])
		opts = dict(opts)
	except GetoptError as err:
		print('error: %r' % err, file=sys.stderr)
		sys.exit(2)
	DEBUG = '--debug' in opts
# NB: load corpus regardless of whether running standalone:
(TEXTS, CORPORA, METADATA) = getcorpus()
log.info('corpus loaded.')
try:
	with open('treesearchpasswd.txt', 'rt') as fileobj:
		PASSWD = {a.strip(): b.strip() for a, b
				in (line.split(':', 1) for line in fileobj)}
	log.info('password protection enabled.')
except IOError:
	log.info('no password protection.')
if STANDALONE:
	APP.run(use_reloader=False,
			host=opts.get('--ip', '0.0.0.0'),
			port=int(opts.get('--port', 5003)),
			debug=DEBUG)
