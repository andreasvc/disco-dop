"""Web interface to draw trees. Requires Flask.
Optional: pdflatex, tikz, imagemagick."""
# stdlib
from __future__ import absolute_import
import io
import os
from subprocess import Popen, PIPE
# Flask & co
from flask import Flask, Response
from flask import request, render_template, redirect, url_for
from flask import send_from_directory
# disco-dop
from discodop.treebank import incrementaltreereader
from discodop.tree import DrawTree

LIMIT = 1024 * 10  # ~10KB
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
	dts = []
	try:
		trees = list(incrementaltreereader(
				request.args['tree'].splitlines(),
				morphology='add' if 'morph' in request.args else None,
				functions='add' if 'func' in request.args else None))
	except Exception as err:  # pylint: disable=broad-except
		return Response(str(err), mimetype='text/plain')
	for tree, sent, _rest in trees:
		try:
			dts.append(DrawTree(tree, sent, abbr='abbr' in request.args))
		except Exception as err:  # pylint: disable=broad-except
			return Response(str(err), mimetype='text/plain')
	if not dts:
		return Response('No trees!', mimetype='text/plain')
	return drawtrees(request.args, dts)


def drawtrees(form, dts):
	""" Draw trees in the requested format. """
	if form.get('output', 'text') == 'svg':
		if len(dts) == 1:
			return Response(dts[0].svg(
						funcsep='-' if 'func' in request.args else None,
						).encode('utf8'), mimetype='image/svg+xml')
		else:
			preamble, postamble = DrawTree.templates['svg']
			result = [preamble.encode('utf8')]
			for dt in dts:
				result.append(
						('<div>\n%s\n</div>\n\n' % dt.svg(
						funcsep='-' if 'func' in request.args else None,
							)).encode('utf8'))
			result.append(postamble.encode('utf8'))
			return Response(b'\n'.join(result), mimetype='text/html')
	elif form.get('output', 'text') == 'text':
		html = form.get('color', False)
		useascii = not form.get('unicode', 0)
		preamble, postamble = DrawTree.templates['html']
		result = []
		if html:
			mimetype = 'text/html'
			result.append(preamble.encode('utf8'))
		else:
			mimetype = 'text/plain'
		for dt in dts:
			result.append(
					dt.text(unicodelines=not useascii, html=html,
						funcsep='-' if 'func' in request.args else None,
						).encode('utf8'))
		if html:
			result.append(postamble.encode('utf8'))
		return Response(b'\n'.join(result), mimetype=mimetype)

	# LaTeX based output
	if form.get('type', None) == 'qtree':
		preamble, postamble = DrawTree.templates['latex']
		for dt in dts:
			for pos in dt.tree.subtrees(lambda n: n and isinstance(n[0], int)):
				pos[0] = dt.sent[pos[0]]
		latexcode = (preamble + '\n\n'.join(dt.tikzqtree() for dt in dts)
				+ postamble)
	elif form.get('type', 'matrix') == 'matrix':
		preamble, postamble = DrawTree.templates['latex']
		latexcode = (preamble + '\n\n'.join(
				dt.tikzmatrix(
						leafcolor=('blue' if form.get('color', '') else ''),
						nodecolor=('red' if form.get('color', '') else ''),
						funcsep='-' if 'func' in request.args else None)
				for dt in dts) + postamble)
	else:
		preamble, postamble = DrawTree.templates['latex']
		latexcode = (preamble + '\n\n'.join(
				dt.tikznode(
						leafcolor='blue' if form.get('color', '') else '',
						nodecolor='red' if form.get('color', '') else '',
						funcsep='-' if 'func' in request.args else None)
				for dt in dts) + postamble)
	if form.get('output', 'text') == 'latex':
		return Response(latexcode, mimetype='text/plain')
	with io.open('/tmp/dtree.tex', 'w', encoding='utf8') as tex:
		tex.write(latexcode)
	os.chdir('/tmp')
	proc = Popen('/usr/bin/pdflatex -halt-on-error  /tmp/dtree.tex'.split(),
			stdin=None, stdout=PIPE, stderr=PIPE, shell=False)
	proc.wait()
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
		proc.wait()
		# set Expires one day ahead (according to server time)
		# req.headers_out['Expires'] = (
		# 	datetime.utcnow() + timedelta( 1, 0 )
		# 	).strftime('%a, %d %b %Y %H:%M:%S UTC')
		# req.headers_out['Cache-Control'] = 'max-age=604800, public'
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


if __name__ == '__main__':
	APP.run(debug=True, host='0.0.0.0')
