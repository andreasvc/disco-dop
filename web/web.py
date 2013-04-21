""" Web interface to the disco-dop parser. Requires Flask.
Expects a series of grammars in subdirectories of grammar/ """
# Wishlist:
# - shortest derivation, SL-DOP, MPSD, &c.
import os, re, cgi, sys, gzip, heapq, time, string, random, codecs
from operator import itemgetter
from flask import Flask, Markup, request, render_template
from flask import send_from_directory

sys.path.append('..')
import treetransforms, disambiguation, coarsetofine
import lexicon, pcfg, plcfrs
from tree import Tree
from treedraw import DrawTree
from containers import Grammar

app = Flask(__name__)
morphtags = re.compile(
		r'\(([_*A-Z0-9]+)(?:\[[^ ]*\][0-9]?)?((?:-[_A-Z0-9]+)?(?:\*[0-9]+)? )')
limit = 40 # maximum sentence length
prunek = 5000 # number of PLCFRS derivations to use for DOP parsing
grammars = {}
backtransforms = {}
knownwords = {}

@app.route('/')
def main():
	""" Serve the main form. """
	return render_template('form.html', result=Markup(parse()))

@app.route('/parse')
def parse():
	""" Parse sentence and return a textual representation of a parse tree,
	in a HTML fragment. To be invoked by an AJAX call."""
	sent = request.args.get('sent', None)
	objfun = request.args.get('objfun', 'mpp')
	marg = request.args.get('marg', 'nbest')
	if not sent:
		return ''
	result = frags = nbest = None
	senttok = tokenize(sent)
	key = (senttok, objfun, marg)
	if key in parse.cache:
		return parse.cache[key]
	if senttok and 1 <= len(senttok) <= limit:
		result, frags, nbest = getparse(senttok, objfun, marg)
	else:
		result = 'Sentence too long: %d words, maximum %d' % (
				len(senttok), limit)
	parse.cache[key] = render_template('parse.html', sent=sent, result=result,
			frags=frags, nbest=nbest, randid=randid())
	return parse.cache[key]
parse.cache = {} #FIXME: cache should expire items

@app.route('/favicon.ico')
def favicon():
	""" Serve the favicon. """
	return send_from_directory(os.path.join(app.root_path, 'static'),
			'favicon.ico', mimetype='image/vnd.microsoft.icon')

def loadgrammars():
	""" Load grammars if necessary. """
	if grammars != {}:
		return
	for lang in ('alpino', 'negra', 'wsj'):
		folder = 'grammars/' + lang
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
		app.logger.debug('Grammar for %s loaded.' % lang)

def getparse(senttok, objfun, marg):
	""" Do the actual parsing. """
	result = frags = nbest = None
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
	result = "no parse!\n"
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
		app.logger.debug('stage 1 fail')
	if start:
		whitelist, items = coarsetofine.prunechart(
				chart, start, grammar[1], grammar[2],
				prunek, False, False, False)
		elapsed.append(time.clock() - begin)
		msg1 = "PLCFRS items: %d; In %d-best derivations: %d" % (
				prunek, len(chart), items)
		begin = time.clock()
		chart, start, msg2 = plcfrs.parse(unksent, grammar[2],
				exhaustive=True, whitelist=whitelist,
				splitprune=False, markorigin=False)
		elapsed.append(time.clock() - begin)
		begin = time.clock()
	else:
		app.logger.debug('stage 2 fail')
	if start:
		parsetrees, msg3 = disambiguation.marginalize(objfun, chart,
				start, grammar[2], 10000,
				kbest=marg in ('nbest', 'both'),
				sample=marg in ('sample', 'both'),
				sent=unksent, tags=None, backtransform=backtransform)
		elapsed.append(time.clock() - begin)
		begin = time.clock()
		tree, prob = max(parsetrees.items(), key=itemgetter(1))
		app.logger.debug('[%g] %s' % (prob, tree))
		tree = morphtags.sub(r'(\1\2', tree)
		tree = Tree.parse(tree, parse_leaf=int)
		treetransforms.unbinarize(tree)
		treetransforms.removefanoutmarkers(tree)
		result = Markup(DrawTree(tree, senttok).text(
				unicodelines=True, html=True))
		frags = Markup('\n\n'.join(
				DrawTree(Tree.parse(frag, parse_leaf=int), terminals
					).text(unicodelines=True, html=True)
				for frag, terminals in disambiguation.extractfragments(
						chart, start, grammar[2], backtransform)))
		elapsed = 'CPU time elapsed: %s => %gs' % (
				' '.join('%gs' % a for a in elapsed), sum(elapsed))
		nbest = Markup('\n'.join((
				'\n\n'.join('%d. [p=%g]\n%s' % (n + 1, prob,
					DrawTree(
						treetransforms.removefanoutmarkers(
							treetransforms.unbinarize(
								Tree.parse(morphtags.sub(r'(\1\2', tree),
									parse_leaf=int))), senttok
						).text(unicodelines=True, html=True))
					for n, (tree, prob) in enumerate(heapq.nlargest(10,
						parsetrees.items(), key=itemgetter(1)))),
				msg1, msg2, msg3, elapsed,
				'10 most probable parse trees:',
				'\n'.join('%d. [p=%g] %s' % (n + 1, prob, cgi.escape(tree))
					for n, (tree, prob) in enumerate(heapq.nlargest(10,
						parsetrees.items(), key=itemgetter(1)))) + '\n')))
	else:
		app.logger.debug('stage 3 fail')
	return result, frags, nbest

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
	text = re.sub('\. *(\n|$)', ' . ', text)
	return tuple(text.split())

def guesslang(sent):
	""" simple heuristic: language that contains most words from input. """
	lang = max(knownwords, key=lambda x: len(knownwords[x] & set(sent)))
	app.logger.debug('Sent: %r; lang: %r.' % (sent, lang))
	return lang

if __name__ == '__main__':
	app.logger.debug('Loading grammars.')
	loadgrammars()
	app.run(debug=False, host='0.0.0.0')
