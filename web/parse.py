""" Web interface to the disco-dop parser. Requires Flask.
Expects a series of grammars in subdirectories of grammar/ """
# Wishlist:
# - shortest derivation, SL-DOP, MPSD, &c.
# - arbitrary configuration of CTF stages;
#   should become class also used by runexp.py & parser.py
import os
import re
import cgi
import glob
import gzip
import heapq
import time
import string
import random
import codecs
import logging
from functools import wraps
from operator import itemgetter
from flask import Flask, Markup, request, render_template, send_from_directory
from werkzeug.contrib.cache import SimpleCache

from discodop import treetransforms, disambiguation, coarsetofine
from discodop import lexicon, pcfg, plcfrs
from discodop.tree import Tree
from discodop.treedraw import DrawTree
from discodop.containers import Grammar

APP = Flask(__name__)
morphtags = re.compile(
		r'\(([_*A-Z0-9]+)(?:\[[^ ]*\][0-9]?)?((?:-[_A-Z0-9]+)?(?:\*[0-9]+)? )')
limit = 40  # maximum sentence length
prunek = 5000  # number of PLCFRS derivations to use for DOP parsing
grammars = {}
backtransforms = {}
knownwords = {}


@APP.route('/')
def main():
	""" Serve the main form. """
	return render_template('parse.html', result=Markup(parse()))


@APP.route('/parse')
def parse():
	""" Parse sentence and return a textual representation of a parse tree,
	in a HTML fragment. To be invoked by an AJAX call."""
	sent = request.args.get('sent', None)
	objfun = request.args.get('objfun', 'mpp')
	marg = request.args.get('marg', 'nbest')
	if not sent:
		return ''
	frags = nbest = None
	senttok = tokenize(sent)
	if not senttok or not 1 <= len(senttok) <= limit:
		return 'Sentence too long: %d words, maximum %d' % (len(senttok), limit)
	result = getparse(senttok, objfun, marg)
	if result == 'no parse':
		return "no parse!\n"
	(parsetrees, fragments, elapsed, msg1, msg2, msg3) = result
	tree, prob = parsetrees[0]
	APP.logger.info('[%g] %s' % (prob, tree))
	tree = morphtags.sub(r'(\1\2', tree)
	tree = Tree.parse(tree, parse_leaf=int)
	treetransforms.unbinarize(tree)
	treetransforms.removefanoutmarkers(tree)
	result = Markup(DrawTree(tree, senttok).text(
			unicodelines=True, html=True))
	frags = Markup('\n\n'.join(
			DrawTree(Tree.parse(frag, parse_leaf=int), terminals).text(
					unicodelines=True, html=True)
			for frag, terminals in fragments))
	elapsed = 'CPU time elapsed: %s => %gs' % (
			' '.join('%gs' % a for a in elapsed), sum(elapsed))
	nbest = Markup('\n\n'.join('%d. [p=%g]\n%s' % (
				n + 1, prob,
				DrawTree(treetransforms.removefanoutmarkers(
					treetransforms.unbinarize(Tree.parse(morphtags.sub(
						r'(\1\2', tree), parse_leaf=int))),
					senttok).text(unicodelines=True, html=True))
				for n, (tree, prob) in enumerate(parsetrees)))
	info = Markup('\n'.join(('sentence length: %d; objfun=%s; marg=%s' % (
			len(senttok), objfun, marg), msg1, msg2, msg3, elapsed,
			'10 most probable parse trees:',
			'\n'.join('%d. [p=%g] %s' % (n + 1, prob, cgi.escape(tree))
					for n, (tree, prob) in enumerate(parsetrees)) + '\n')))
	return render_template('parsetree.html', sent=sent, result=result,
			frags=frags, nbest=nbest, info=info, randid=randid())


@APP.route('/favicon.ico')
def favicon():
	""" Serve the favicon. """
	return send_from_directory(os.path.join(APP.root_path, 'static'),
			'parse.ico', mimetype='image/vnd.microsoft.icon')


def loadgrammars():
	""" Load grammars if necessary. """
	if grammars != {}:
		return
	for folder in glob.glob('grammars/*/'):
		_, lang = os.path.split(os.path.dirname(folder))
		APP.logger.info('Loading grammar %r', lang)
		grammarlist = []
		for stagename in ('pcfg', 'plcfrs', 'dop'):
			rules = gzip.open("%s/%s.rules.gz" % (folder, stagename)).read()
			lexical = codecs.getreader('utf-8')(gzip.open("%s/%s.lex.gz" % (
					folder, stagename))).read()
			if stagename == 'pcfg':
				grammarlist.append(Grammar(rules, lexical,
						logprob=False, bitpar=True))
			else:
				grammarlist.append(Grammar(rules, lexical))
			assert grammarlist[-1].testgrammar(), stagename
			if stagename == 'plcfrs':
				_ = grammarlist[-1].getmapping(grammarlist[-2],
						striplabelre=re.compile(b'@.+$'),
						neverblockre=re.compile(b'.+}<'),
						splitprune=True, markorigin=True)
			elif stagename == 'dop':
				_ = grammarlist[-1].getmapping(grammarlist[-2],
						striplabelre=re.compile(b'@.+$'),
						neverblockre=re.compile(b'.+}<'),
						splitprune=False, markorigin=False)
				backtransforms[lang] = dict(enumerate(gzip.open(
						"%s/dop.backtransform.gz" % folder).read().splitlines()))
		grammars[lang] = grammarlist
		knownwords[lang] = {w for w in grammars[lang][0].lexical
				if not w.startswith("UNK")}
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


@cached(timeout=24 * 3600)
def getparse(senttok, objfun, marg):
	""" Do the actual parsing. """
	elapsed = []
	begin = time.clock()
	lang = guesslang(senttok)
	grammar = grammars[lang]
	knownword = knownwords[lang]
	backtransform = backtransforms[lang]
	unksent = [w if w in knownword
			else lexicon.unknownword4(w, n, knownword)
			for n, w in enumerate(senttok)]
	inside, outside, start, _ = pcfg.doinsideoutside(
			unksent, grammar[0], tags=None)
	elapsed.append(time.clock() - begin)
	begin = time.clock()
	if start:
		(whitelist, _, _, _, _) = coarsetofine.whitelistfromposteriors(
				inside, outside, start,
				grammar[0], grammar[1], 1e-5, True, True)
		elapsed.append(time.clock() - begin)
		begin = time.clock()
		chart, start, _ = plcfrs.parse(unksent, grammar[1],
				exhaustive=True, whitelist=whitelist,
				splitprune=True, markorigin=True)
		elapsed.append(time.clock() - begin)
		begin = time.clock()
	else:
		APP.logger.warning('stage 1 fail')
	if start:
		whitelist, items = coarsetofine.prunechart(
				chart, start, grammar[1], grammar[2],
				prunek, False, False, False)
		elapsed.append(time.clock() - begin)
		msg1 = "PLCFRS items: %d; In %d-best derivations: %d" % (
				prunek, len(chart), items)
		begin = time.clock()
		chart, start, msg2 = plcfrs.parse(unksent, grammar[2],
				exhaustive=objfun != 'mpd', whitelist=whitelist,
				splitprune=False, markorigin=False)
		elapsed.append(time.clock() - begin)
		begin = time.clock()
	else:
		APP.logger.warning('stage 2 fail')
	if start:
		parsetrees, msg3 = disambiguation.marginalize(objfun, chart,
				start, grammar[2], 10000,
				kbest=marg in ('nbest', 'both'),
				sample=marg in ('sample', 'both'),
				sent=unksent, tags=None, backtransform=backtransform)
		fragments = disambiguation.extractfragments(
				chart, start, grammar[2], backtransform)
		elapsed.append(time.clock() - begin)
		return (heapq.nlargest(10, parsetrees.items(), key=itemgetter(1)),
				fragments, elapsed, msg1, msg2, msg3)
	else:
		APP.logger.warning('stage 3 fail')
		return 'no parse'


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
	lang = max(knownwords, key=lambda x: len(knownwords[x] & set(sent)))
	APP.logger.info('Lang: %s; Sent: %s' % (lang, ' '.join(sent)))
	return lang


if __name__ == '__main__':
	logging.basicConfig()
	for log in (logging.getLogger(), APP.logger):
		log.setLevel(logging.DEBUG)
		log.handlers[0].setFormatter(logging.Formatter(
				fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
	loadgrammars()
	APP.run(debug=False, host='0.0.0.0')
