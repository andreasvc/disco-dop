""" Web interface to draw trees. Requires Flask.
Optional: pdflatex, imagemagick. """
# stdlib
import os
import re
from subprocess import Popen, PIPE
try:
	from itertools import zip_longest
except ImportError:
	from itertools import izip_longest as zip_longest
#from datetime import datetime, timedelta
#from functools import wraps
# Flask & co
from flask import Flask, Response
from flask import request, render_template, redirect, url_for
from flask import send_from_directory
#from werkzeug.contrib.cache import SimpleCache
# disco-dop
from discodop.tree import Tree
from discodop.treedraw import DrawTree
from discodop import treebank


LIMIT = 1024 * 10  # ~10KB
MORPH_TAGS = re.compile(
		r'\(([_*A-Z0-9]+)(?:\[[^ ]*\][0-9]?)?((?:-[_A-Z0-9]+)?(?:\*[0-9]+)? )')
FUNC_TAGS = re.compile(r'-[_A-Z0-9]+')
GETLEAVES = re.compile(r' ([^ ()]+)(?=[ )])')
PREAMBLE = r"""\documentclass{article}
\usepackage[landscape]{geometry}
\usepackage[utf8]{inputenc}
%s
%% NB: preview is optional, to make a cropped pdf
\usepackage[active, tightpage]{preview} \setlength{\PreviewBorder}{0.2cm}
\begin{document}
\pagestyle{empty}
\fontfamily{phv}\selectfont
\begin{preview}
"""
POSTAMBLE = r"""
\end{preview}
\end{document}"""


APP = Flask(__name__)


@APP.route('/')
def main():
	""" Redirect to avoid trailing slash hassles. """
	return redirect(url_for('index'))


@APP.route('/index')
def index():
	""" Form for trees & parameters. """
	return render_template('draw.html')


@APP.route('/favicon.ico')
def favicon():
	""" Serve the favicon. """
	return send_from_directory(os.path.join(APP.root_path, 'static'),
			'treesearch.ico', mimetype='image/vnd.microsoft.icon')


@APP.route('/draw')
def draw():
	""" Wrapper to parse & draw tree(s). """
	if len(request.args['tree']) > LIMIT:
		return 'Too much data. Limit: %d bytes' % LIMIT
	dts = gettrees(request.args)
	return drawtrees(request.args, dts)


def gettrees(form):
	""" Detect format of trees and parse into DrawTree objects. """
	brackets = ''
	if form['tree'].lstrip().startswith('('):
		brackets = '()'
	elif form['tree'].lstrip().startswith('['):
		brackets = '[]'
	trees, sents = [], []
	if brackets:  # bracket notation
		# Parse trees presented in bracket format, whether with indices or not,
		# and regardless of whitespace (multiple trees per line, trees spread
		# over multiple lines, trees alternated with sentence, etc.).
		brackettrees, rest = segmentbrackets(form['tree'], brackets)
		strtermre = re.compile('[^0-9\\%s]\\%s' % (brackets[1], brackets[1]))
		for treestr, sent in zip_longest(brackettrees,
				form.get('sent', '').splitlines() or rest):
			if strtermre.search(treestr):  # terminals are not (all) indices
				tree = noempty(Tree.parse(treestr, brackets=brackets))
				sents.append(tree.leaves())
				trees.append(renumber(tree))
			else: # disc. trees with integer indices as terminals
				trees.append(Tree.parse(treestr, parse_leaf=int,
					brackets=brackets))
				sents.append(sent and sent.split())
		if trees and (not sents or not sents[0]):
			# use indices as leaves
			sents = [map(str, range(max(tree.leaves()) + 1)) for tree in trees]
	if not trees:  # discontinuous, export format, one or more trees
		cur = []
		for line in form['tree'].splitlines():
			if line.startswith('#BOS'):
				cur = []
			elif line.startswith('#EOS'):
				tree, sent = exporttree(cur)
				trees.append(tree)
				sents.append(sent)
				cur = []
			elif line.strip():
				cur.append(line)
		if cur:
			tree, sent = exporttree(cur)
			trees.append(tree)
			sents.append(sent)
	if form.get('abbr', None):
		for tree in trees:
			for n in tree.subtrees():
				n.label = n.label[:5]
	return [DrawTree(tree, sent) for tree, sent in zip(trees, sents)]


def segmentbrackets(formtree, brackets):
	""" Segment a series of S-expressions of brackets into a list of strings,
	along with a list of strings found outside of S-expressions. """
	lb, rb = brackets
	formtree = formtree.lstrip(' \t\n\r')
	if formtree[0] != lb:
		return (), ()
	start = 0
	parens = 0
	results = []
	rest = []
	for n, char in enumerate(formtree):
		if char == lb:
			parens += 1
			if n and parens == 0 and formtree[start:n + 1]:
				rest.append(formtree[start:n + 1])
		elif char == rb:
			parens -= 1
			if parens == 0:
				results.append(formtree[start:n + 1])
				start = n + 1
			elif parens < 0:
				return (), ()  # unbalanced parentheses
	if parens == 0 and formtree[start:n + 1]:
		rest.append(formtree[start:n + 1])
	return results, rest


def drawtrees(form, dts):
	""" Draw trees in the requested format. """
	os.chdir('/tmp')
	if form.get('type', 'matrix') == 'matrix':
		latexcode = ''.join((PREAMBLE % (
			'\\usepackage{helvet,tikz}\n'
			'\\usetikzlibrary{matrix,positioning}'),
			'\n\n'.join(
				dt.tikzmatrix(
						leafcolor=('blue' if form.get('color', '') else ''),
						nodecolor=('red' if form.get('color', '') else ''))
				for dt in dts),
				POSTAMBLE))
	elif form.get('type', None) == 'qtree':
		result = [PREAMBLE % r'\usepackage{helvet,tikz,tikz-qtree}']
		for dt in dts:
			for pos, word in zip(
					dt.tree.subtrees(lambda n: n and isinstance(n[0], int)),
					dt.sent):
				pos[0] = word
			result.append('\n\n'.join(dt.tree.pprint_latex_qtree()
					for dt in dts))
		result.append(POSTAMBLE)
		latexcode = ''.join(result)
	else:
		latexcode = ''.join((PREAMBLE % (
			'\\usepackage{helvet,tikz}\n'
			'\\usetikzlibrary{positioning}'),
			'\n\n'.join(
				dt.tikznode(
						leafcolor='blue' if form.get('color', '') else '',
						nodecolor='red' if form.get('color', '') else '')
				for dt in dts),
				POSTAMBLE))
	if form.get('output', 'text') == 'latex':
		return Response(latexcode, mimetype='text/plain')
	elif form.get('output', 'text') == 'svg':
		if len(dts) == 1:
			return Response(dts[0].svg().encode('utf-8'),
					mimetype='image/svg+xml')
		else:
			result = [('<!doctype html>\n<html>\n<head>\n'
				'\t<meta http-equiv="Content-Type" '
				'content="text/html; charset=UTF-8">\n</head>\n<body>')]
			for dt in dts:
				result.append(
						'<div>\n%s\n</div>\n\n' % dt.svg().encode('utf-8'))
			result.append('</body></html>')
			return Response('\n'.join(result), mimetype='text/html')
	elif form.get('output', 'text') == 'text':
		html = form.get('color', False)
		ascii = not form.get('unicode', 0)
		result = []
		if html:
			mimetype = 'text/html'
			result.append('<!doctype html>\n<html>\n<head>\n'
				'\t<meta http-equiv="Content-Type" '
				'content="text/html; charset=UTF-8">\n</head>\n<body>\n<pre>')
		else:
			mimetype = 'text/plain'
		for dt in dts:
			result.append(
					dt.text(unicodelines=not ascii, html=html).encode('utf-8'))
		if html:
			result.append('</pre></body></html>')
		return Response('\n'.join(result), mimetype=mimetype)
	else:
		with open('/tmp/dtree.tex', 'w') as tex:
			tex.write(latexcode)
		proc = Popen('/usr/bin/pdflatex -halt-on-error  /tmp/dtree.tex'.split(),
				stdin=None, stdout=PIPE, stderr=PIPE, shell=False)
		proc.wait()  # pylint: disable=E1101
		if form.get('output', 'text') == 'pdf':
			try:
				return send_from_directory('/tmp', 'dtree.pdf',
						mimetype='application/pdf')
			except IOError:
				pass
			for ext in ('aux', 'log', 'pdf'):
				os.remove('/tmp/dtree.' + ext)
		if form.get('output', 'text') == 'png':
			proc = Popen('/usr/bin/convert -density 125 /tmp/dtree.pdf \
					/tmp/dtree.png'.split(),  # -trim
					stdin=None, stdout=PIPE, stderr=PIPE, shell=False)
			proc.wait()  # pylint: disable=E1101
			#set Expires one day ahead (according to server time)
			#req.headers_out['Expires'] = (
			#	datetime.utcnow() + timedelta( 1, 0 )
			#	).strftime('%a, %d %b %Y %H:%M:%S UTC')
			#req.headers_out['Cache-Control'] = 'max-age=604800, public'
			try:
				return send_from_directory('/tmp', 'dtree.png',
						mimetype='image/png')
			except IOError:
				pass
			for filename in ('/tmp/dtree.aux', '/tmp/dtree.log',
					'/tmp/dtree.tex', '/tmp/dtree.pdf', '/tmp/dtree.png'):
				try:
					os.remove(filename)
				except OSError:
					break


def exporttree(data):
	""" Wrapper to get both tree and sentence for tree in export format given
	as list of lines. """
	data = [treebank.exportsplit(x) for x in data]
	tree = treebank.exportparse(data)
	sent = [a[treebank.WORD] for a in data
			if not a[treebank.WORD].startswith('#')]
	return tree, sent


def noempty(tree):
	""" Add sentinel child (None) to empty nodes. """
	for a in tree.subtrees(lambda n: len(n) == 0):
		a.append(None)
	return tree


def renumber(tree):
	""" Replace leaves with indices. """
	for n, a in enumerate(tree.subtrees(
			lambda n: len(n) == 1 and not isinstance(n[0], Tree))):
		a[0] = n
	return tree


if __name__ == '__main__':
	APP.run(debug=True, host='0.0.0.0')
