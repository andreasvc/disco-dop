""" Web interface to the disco-dop parser. Requires Flask.
Expects a series of grammars produced by runexp in subdirectories of grammar/

Also usable from the command line:
$ curl http://localhost:5000/parse -G --data-urlencode "sent=What's up?"
"""
import os
import re
import cgi
import glob
import heapq
import string  # pylint: disable=W0402
import random
import logging
from functools import wraps
from operator import itemgetter
from flask import Flask, Markup, Response
from flask import request, render_template, send_from_directory
from werkzeug.contrib.cache import SimpleCache
from discodop import treetransforms, treebank
from discodop.tree import Tree
from discodop.treedraw import DrawTree
from discodop.runexp import readparam
from discodop.parser import Parser, readgrammars, probstr

APP = Flask(__name__)
morphtags = re.compile(
		r'\(([_*A-Z0-9]+)(?:\[[^ ]*\][0-9]?)?((?:-[_A-Z0-9]+)?(?:\*[0-9]+)? )')
limit = 40  # maximum sentence length
prunek = 5000  # number of PLCFRS derivations to use for DOP parsing
grammars = {}
parsers = {}
backtransforms = {}


@APP.route('/')
def main():
	""" Serve the main form. """
	return render_template('parse.html', result=Markup(parse()))


@APP.route('/parse')
def parse():
	""" Parse sentence and return a textual representation of a parse tree,
	in a HTML fragment or plain text. To be invoked by an AJAX call."""
	sent = request.args.get('sent', None)
	objfun = request.args.get('objfun', 'mpp')
	marg = request.args.get('marg', 'nbest')
	html = request.args.get('html', False)
	if not sent:
		return ''
	frags = nbest = None
	senttok = tokenize(sent)
	if not senttok or not 1 <= len(senttok) <= limit:
		return 'Sentence too long: %d words, maximum %d' % (len(senttok), limit)
	lang = guesslang(senttok)
	parsers[lang].stages[-1].objective = objfun
	parsers[lang].stages[-1].kbest = marg in ('nbest', 'both')
	parsers[lang].stages[-1].sample = marg in ('sample', 'both')
	results = list(parsers[lang].parse(senttok))
	if results[-1].noparse:
		parsetrees = {}
		result = 'no parse!'
		frags = nbest = ''
		msg = ''
	else:
		if parsers[lang].relationalrealizational:
			treebank.handlefunctions('add', results[-1].parsetree, pos=True)
		tree = str(results[-1].parsetree)
		prob = results[-1].prob
		parsetrees = results[-1].parsetrees or {}
		parsetrees = heapq.nlargest(10, parsetrees.items(), key=itemgetter(1))
		fragments = results[-1].fragments or ()
		msg = '\n'.join(stage.msg for stage in results)
		APP.logger.info('[%s] %s' % (probstr(prob), tree))
		tree = morphtags.sub(r'(\1\2', tree)
		tree = Tree.parse(tree, parse_leaf=int)
		result = Markup(DrawTree(tree, senttok).text(
				unicodelines=True, html=html))
		frags = Markup('Phrasal fragments used in the most probable derivation '
				'of the highest ranked parse tree:\n'
				+ '\n\n'.join(
				DrawTree(Tree.parse(frag, parse_leaf=int), terminals).text(
						unicodelines=True, html=html)
				for frag, terminals in fragments))
		nbest = Markup('\n\n'.join('%d. [%s]\n%s' % (
				n + 1, probstr(prob),
				DrawTree(treetransforms.removefanoutmarkers(
					treetransforms.unbinarize(Tree.parse(morphtags.sub(
						r'(\1\2', tree), parse_leaf=int))),
					senttok).text(unicodelines=True, html=html))
				for n, (tree, prob) in enumerate(parsetrees)))
	elapsed = [stage.elapsedtime for stage in results]
	elapsed = 'CPU time elapsed: %s => %gs' % (
			' '.join('%gs' % a for a in elapsed), sum(elapsed))
	info = Markup('\n'.join(('sentence length: %d; objfun=%s; marg=%s' % (
			len(senttok), objfun, marg), msg, elapsed,
			'10 most probable parse trees:',
			'\n'.join('%d. [%s] %s' % (n + 1, probstr(prob), cgi.escape(tree))
					for n, (tree, prob) in enumerate(parsetrees)) + '\n')))
	if not html:
		return Response('\n'.join((nbest, frags, info, result)),
				mimetype='text/plain')
	return render_template('parsetree.html', sent=sent, result=result,
			frags=frags, nbest=nbest, info=info, randid=randid())


@APP.route('/favicon.ico')
def favicon():
	""" Serve the favicon. """
	return send_from_directory(os.path.join(APP.root_path, 'static'),
			'parse.ico', mimetype='image/vnd.microsoft.icon')


def loadparsers():
	""" Load grammars if necessary. """
	if not parsers:
		for directory in glob.glob('grammars/*/'):
			_, lang = os.path.split(os.path.dirname(directory))
			APP.logger.info('Loading grammar %r', lang)
			params = readparam(os.path.join(directory, 'params.prm'))
			params['resultdir'] = directory
			stages = params['stages']
			postagging = params['postagging']
			readgrammars(directory, stages, postagging)
			parsers[lang] = Parser(stages,
					transformations=params['transformations'],
					tailmarker=params['tailmarker'], postagging=postagging
					if postagging and postagging['method'] == 'unknownword'
					else None,
					relationalrealizational=params.get(
						'relationalrealizational'))
			APP.logger.info('Grammar for %s loaded.' % lang)


def cached(timeout=3600):
	""" Caching decorator from Flask documentation """
	def decorator(func):
		""" Wrapper """
		func.cache = SimpleCache()

		@wraps(func)
		def decorated_function(*args, **kwargs):
			""" memoize on function arguments. """
			cache_key = args
			result = func.cache.get(cache_key)
			if result is not None:
				return result
			result = func(*args, **kwargs)
			func.cache.set(cache_key, result, timeout=timeout)
			return result

		return decorated_function
	return decorator


def randid():
	""" return a string with 6 random letters. """
	return ''.join(random.choice(string.ascii_letters)
		for _ in range(6))


# List of contractions adapted from Robert MacIntyre's tokenizer.
CONTRACTIONS2 = re.compile(
		"(?i)(?:%s)\b" % "|".join([
		r"(.)('ll|'re|'ve|n't|'s|'m|'d)",
		r"\b(can)(not)",
		r"\b(D)('ye)",
		r"\b(Gim)(me)",
		r"\b(Gon)(na)",
		r"\b(Got)(ta)",
		r"\b(Lem)(me)",
		r"\b(Mor)('n)",
		r"\b(T)(is)",
		r"\b(T)(was)",
		r"\b(Wan)(na)"]))
CONTRACTIONS3 = re.compile(r"(?i)\b(?:(Whad)(dd)(ya)|(Wha)(t)(cha))\b")


def tokenize(text):
	""" Adapted from nltk.tokenize.TreebankTokenizer. """
	text = CONTRACTIONS2.sub(r'\1 \2', text)
	text = CONTRACTIONS3.sub(r'\1 \2 \3', text)
	# Separate most punctuation
	text = re.sub(r"([^\w\.\'\-\/,&])", r' \1 ', text, flags=re.UNICODE)
	# Separate commas if they're followed by space; e.g., don't separate 2,500
	# Separate single quotes if they're followed by a space.
	text = re.sub(r"([,']\s)", r' \1', text)
	# Separate periods that come before newline or end of string.
	text = re.sub(r'\. *(\n|$)', ' . ', text)
	return tuple(text.split())


def guesslang(sent):
	""" simple heuristic: language that contains most words from input. """
	lang = max(parsers, key=lambda x: len(set(sent).intersection(
			parsers[x].stages[0].grammar.lexical)))
	APP.logger.info('Lang: %s; Sent: %s' % (lang, ' '.join(sent)))
	return lang


if __name__ == '__main__':
	logging.basicConfig()
	for log in (logging.getLogger(), APP.logger):
		log.setLevel(logging.DEBUG)
		log.handlers[0].setFormatter(logging.Formatter(
				fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
	loadparsers()
	APP.run(debug=False, host='0.0.0.0')
