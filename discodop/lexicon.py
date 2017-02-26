# -*- coding: UTF-8 -*-
"""Add rules to handle unknown words and smooth lexical probabilities.

Rare words in the training set are replaced with word signatures, such that
unknown words can receive similar tags. Given a function to produce such
signatures from words, the flow is as follows:

- Simple lexical smoothing:

  #. getunknownwordmodel (get statistics)
  #. replaceraretrainwords (adjust trees)
  #. [ read off grammar ]
  #. simplesmoothlexicon (add extra lexical productions)

- Sophisticated smoothing (untested):

  #. getunknownwordmodel
  #. getlexmodel
  #. replaceraretrainwords
  #. [ read off grammar ]
  #. smoothlexicon

- During parsing:

  #. replaceraretestwords (only give known words and signatures to parser)
  #. restore original words in derivations

"""
# pylint: disable=abstract-class-instantiated
from __future__ import division, print_function, absolute_import, \
		unicode_literals
import os
import re
import logging
import tempfile
from operator import itemgetter
from subprocess import Popen, PIPE
from collections import defaultdict, Counter
try:
	from cyordereddict import OrderedDict
except ImportError:
	from collections import OrderedDict
from fractions import Fraction
from .treebanktransforms import YEARRE
from .tree import escape
from .util import which

UNK = '_UNK'
NUMBERRE = re.compile('^(?:[0-9]*[,.\'])?[0-9]+$')
TREETAGGERHELP = '''tree tagger not found. commands to install:
mkdir tree-tagger && cd tree-tagger
wget ftp://ftp.ims.uni-stuttgart.de/pub/corpora/tree-tagger-linux-3.2.tar.gz
tar -xzf tree-tagger-linux-3.2.tar.gz
wget ftp://ftp.ims.uni-stuttgart.de/pub/corpora/tagger-scripts.tar.gz
tar -xzf ftp://ftp.ims.uni-stuttgart.de/pub/corpora/tagger-scripts.tar.gz
mkdir lib && cd lib && wget \
ftp://ftp.ims.uni-stuttgart.de/pub/corpora/german-par-linux-3.2-utf8.bin.gz
gunzip german-par-linux-3.2-utf8.bin.gz'''
STANFORDTAGGERHELP = '''Stanford tagger not found. Commands to install:
wget http://nlp.stanford.edu/software/stanford-postagger-full-2012-07-09.tgz
tar -xzf stanford-postagger-full-2012-07-09.tgz'''


def getunknownwordmodel(tagged_sents, unknownword,
		unknownthreshold, openclassthreshold):
	"""Collect statistics for an unknown word model.

	:param tagged_sents: the sentences from the training set with the gold POS
			tags from the treebank.
	:param unknownword: a function that returns a signature for a given word;
			e.g., "eschewed" => "_UNK-L-d".
	:param unknownthreshold: words with frequency lower than or equal to this
			are replaced by their signature.
	:param openclassthreshold: tags that rewrite to at least this much word
			types are considered to be open class categories."""
	wordsfortag = defaultdict(set)
	tags = Counter()
	wordtags = Counter()
	sigs = Counter()
	sigtag = Counter()
	words = Counter(word for sent in tagged_sents for word, tag in sent)
	lexicon = {word for word, freq in words.items()
			if freq > unknownthreshold}
	wordsig = {}
	for sent in tagged_sents:
		for n, (word, tag) in enumerate(sent):
			wordsfortag[tag].add(word)
			tags[tag] += 1
			wordtags[word, tag] += 1
			sig = unknownword(word, n, lexicon)
			wordsig[word] = sig  # NB: sig may also depend on n and lexicon
			sigtag[sig, tag] += 1
	if openclassthreshold:
		openclasstags = {tag: len({w.lower() for w in ws})
				for tag, ws in wordsfortag.items()
				if len({w.lower() for w in ws}) >= openclassthreshold}
		closedclasstags = {tag: len({w.lower() for w in wordsfortag[tag]})
				for tag in tags if tag not in openclasstags}
		closedclasswords = {word for tag in closedclasstags
				for word in wordsfortag[tag]}
		openclasswords = lexicon - closedclasswords
		# add rare closed-class words back to lexicon
		lexicon.update(closedclasswords)
	else:
		openclasstags = {}
		openclasswords = {}
	for sent in tagged_sents:
		for n, (word, _) in enumerate(sent):
			if word not in lexicon:
				sig = unknownword(word, n, lexicon)
				sigs[sig] += 1
	msg = 'known words: %d, signature types seen: %d\n' % (
			len(lexicon), len(sigs))
	msg += 'open class tags: %s\n\n' % ' '.join(sorted(
			'%s:%d' % a for a in openclasstags.items()))
	msg += 'closed class tags: %s' % ' '.join(sorted(
			'%s:%d' % a for a in closedclasstags.items()))
	return (sigs, words, lexicon, wordsfortag, openclasstags,
			openclasswords, tags, wordtags,
			wordsig, sigtag), msg


def replaceraretrainwords(tagged_sents, unknownword, lexicon):
	"""Replace train set words not in lexicon w/signature from unknownword()."""
	def repl(n, word):
		"""Replace word w/signature if needed."""
		if YEARRE.match(word):
			return '1970'
		elif NUMBERRE.match(word):
			return '000'
		elif word not in lexicon:
			return unknownword(word, n, lexicon)
		return word

	return [[repl(n, word) for n, (word, _) in enumerate(sent)]
				for sent in tagged_sents]


def replaceraretestwords(sent, unknownword, lexicon, sigs):
	"""Replace test set words not in lexicon w/signature from unknownword().

	If only a lowercase version of a word is in the grammar, that will be used
	instead. If the returned signature is not part of the grammar, a default
	one is returned."""
	for n, word in enumerate(sent):
		if YEARRE.match(word):
			yield '1970'
		elif NUMBERRE.match(word):
			yield '000'
		elif word in lexicon:
			yield word
		elif word.lower() in lexicon:
			yield word.lower()
		else:
			sig = unknownword(word, n, lexicon)
			if sig in sigs:
				yield sig
			else:
				yield UNK


def simplesmoothlexicon(lexmodel, epsilon=1. / 100):
	"""Collect new lexical productions.

	- unobserved combinations of tags with known open class words.
	- unobserved signatures which are mapped to ``'_UNK'``.

	:param epsilon: 'frequency' of productions for unseen tag, word pair.
	:returns: a dictionary of lexical rules, with pseudofrequencies as values.
	"""
	(lexicon, wordsfortag, openclasstags,
			openclasswords, tags, wordtags) = lexmodel
	newrules = {}
	# rare words as signature AND as word:
	for word, tag in wordtags:
		if word not in lexicon:
			# needs to be normalized later
			newrules[(tag, 'Epsilon'), (escape(word), )] = wordtags[word, tag]
			# print(tag, '=>', word, wordstags[word, tag], file=sys.stderr)
	for tag in openclasstags:  # open class tag-word pairs
		for word in openclasswords - wordsfortag[tag] - {UNK}:
			newrules[(tag, 'Epsilon'), (escape(word), )] = epsilon
	for tag in tags:  # catch all unknown signature
		newrules[(tag, 'Epsilon'), (UNK, )] = epsilon
	return newrules


def getlexmodel(sigs, words, _lexicon, wordsfortag, openclasstags,
			openclasswords, tags, wordtags, wordsig, sigtag,
			openclassoffset=1, kappa=1):
	"""Compute a smoothed lexical model.

	:returns: a dictionary giving P(word_or_sig | tag).
	:param openclassoffset: for words that only appear with open class tags,
		add unseen combinations of open class (tag, word) with this count.
	:param kappa: FIXME; cf. Klein & Manning (2003), footnote 5.
		http://aclweb.org/anthology/P03-1054"""
	for tag in openclasstags:
		for word in openclasswords - wordsfortag[tag]:
			wordtags[word, tag] += openclassoffset
			words[word] += openclassoffset
			tags[tag] += openclassoffset
		# unseen signatures
		sigs[UNK] += 1
		sigtag[UNK, tag] += 1
	# Compute P(tag|sig)
	tagtotal = sum(tags.values())
	wordstotal = sum(words.values())
	sigstotal = sum(sigs.values())
	P_tag = {}
	for tag in tags:
		P_tag[tag] = Fraction(tags[tag], tagtotal)
	P_word = defaultdict(int)
	for word in words:
		P_word[word] = Fraction(words[word], wordstotal)
	P_tagsig = defaultdict(Fraction)  # ??
	for sig in sigs:
		P_tagsig[tag, sig] = Fraction(P_tag[tag],
				Fraction(sigs[sig], sigstotal))
		# print("P(%s | %s) = %s " % ()
		# 		tag, sig, P_tagsig[tag, sig], file=sys.stderr)
	# Klein & Manning (2003) Accurate unlexicalized parsing
	# http://aclweb.org/anthology/P03-1054
	# P(tag|word) = [count(tag, word) + kappa * P(tag|sig)]
	# 		/ [count(word) + kappa]
	P_tagword = defaultdict(int)
	for word, tag in wordtags:
		P_tagword[tag, word] = Fraction(wordtags[word, tag]
				+ kappa * P_tagsig[tag, wordsig[word]],
				words[word] + kappa)
		# print("P(%s | %s) = %s " % ()
		# 		tag, word, P_tagword[tag, word], file=sys.stderr)
	# invert with Bayes theorem to get P(word|tag)
	P_wordtag = defaultdict(int)
	for tag, word in P_tagword:
		# wordorsig = word if word in lexicon else wordsig[word]
		wordorsig = word
		P_wordtag[wordorsig, tag] += Fraction((P_tagword[tag, word]
				* P_word[word]), P_tag[tag])
		# print("P(%s | %s) = %s " % ()
		# 		word, tag, P_wordtag[wordorsig, tag], file=sys.stderr)
	msg = "(word, tag) pairs in model: %d" % len(P_tagword)
	return P_wordtag, msg


def smoothlexicon(grammar, P_wordtag):
	"""Replace lexical probabilities using given unknown word model.
	Ignores lexical productions of known subtrees (tag contains '@')
	introduced by DOP, i.e., we only modify lexical depth 1 subtrees."""
	newrules = []
	for (rule, yf), w in grammar:
		if rule[1] == 'Epsilon' and '@' not in rule[0]:
			wordorsig = yf[0]
			tag = rule[0]
			newrule = (((tag, 'Epsilon'), (wordorsig, )),
					P_wordtag[wordorsig, tag])
			newrules.append(newrule)
		else:
			newrules.append(((rule, yf), w))
	return newrules


# === functions for unknown word signatures ============

HASDIGIT = re.compile(r"\d", re.UNICODE)
HASNONDIGIT = re.compile(r"\D", re.UNICODE)
# NB: includes '-', hyphen, non-breaking hyphen
# does NOT include: figure-dash, em-dash, en-dash (these are punctuation,
# not word-combining) u2012-u2015; nb: these are hex values.
HASDASH = re.compile("[-\u2010\u2011]")
# FIXME: exclude accented characters for model 6?
HASLOWER = re.compile('[a-z\xe7\xe9\xe0\xec\xf9\xe2\xea\xee\xf4\xfb\xeb'
		'\xef\xfc\xff\u0153\xe6]')
HASUPPER = re.compile('[A-Z\xc7\xc9\xc0\xcc\xd9\xc2\xca\xce\xd4\xdb\xcb'
		'\xcf\xdc\u0178\u0152\xc6]')
HASLETTER = re.compile('[A-Za-z\xe7\xe9\xe0\xec\xf9\xe2\xea\xee\xf4\xfb'
		'\xeb\xef\xfc\xff\u0153\xe6\xc7\xc9\xc0\xcc\xd9\xc2\xca\xce\xd4'
		'\xdb\xcb\xcf\xdc\u0178\u0152\xc6]')
# Cf. http://en.wikipedia.org/wiki/French_alphabet
LOWER = ('abcdefghijklmnopqrstuvwxyz\xe7\xe9\xe0\xec\xf9\xe2\xea\xee\xf4\xfb'
		'\xeb\xef\xfc\xff\u0153\xe6')
UPPER = ('ABCDEFGHIJKLMNOPQRSTUVWXYZ\xc7\xc9\xc0\xcc\xd9\xc2\xca\xce\xd4\xdb'
		'\xcb\xcf\xdc\u0178\u0152\xc6')
LOWERUPPER = LOWER + UPPER


def unknownword6(word, loc, lexicon):
	"""Model 6 of the Stanford parser (for WSJ treebank)."""
	wlen = len(word)
	numcaps = 0
	sig = UNK
	numcaps = len(HASUPPER.findall(word))
	lowered = word.lower()
	if numcaps > 1:
		sig += "-CAPS"
	elif numcaps > 0:
		if loc == 0:
			sig += "-INITC"
			if lowered in lexicon:
				sig += "-KNOWNLC"
		else:
			sig += "-CAP"
	elif HASLOWER.search(word):
		sig += "-LC"
	if HASDIGIT.search(word):
		sig += "-NUM"
	if HASDASH.search(word):
		sig += "-DASH"
	if lowered.endswith('s') and wlen >= 3:
		if lowered[-2] not in 'siu':
			sig += '-s'
	elif wlen >= 5 and not HASDASH.search(word) and not (
			HASDIGIT.search(word) and numcaps > 0):
		suffixes = ('ed', 'ing', 'ion', 'er', 'est', 'ly', 'ity', 'y', 'al')
		for a in suffixes:
			if lowered.endswith(a):
				sig += "-%s" % a
				break
	return sig


def unknownword4(word, loc, _lexicon):
	"""Model 4 of the Stanford parser. Relatively language agnostic."""
	sig = UNK

	# letters
	if word and word[0] in UPPER:
		if not HASLOWER.search(word):
			sig += "-AC"
		elif loc == 0:
			sig += "-SC"
		else:
			sig += "-C"
	elif HASLOWER.search(word):
		sig += "-L"
	elif HASLETTER.search(word):
		sig += "-U"
	else:
		sig += "-S"  # no letter

	# digits
	if HASDIGIT.search(word):
		if HASNONDIGIT.search(word):
			sig += "-n"
		else:
			sig += "-N"

	# punctuation
	if "-" in word:
		sig += "-H"
	if "." in word:
		sig += "-P"
	if "," in word:
		sig += "-C"
	if len(word) > 3:
		if word[-1] in LOWERUPPER:
			sig += "-%s" % word[-2:].lower()
	return sig


def unknownwordbase(word, _loc, _lexicon):
	"""BaseUnknownWordModel of the Stanford parser.
	Relatively language agnostic."""
	sig = UNK

	# letters
	if word[0] in UPPER:
		sig += "-C"
	else:
		sig += "-c"

	# digits
	if HASDIGIT.search(word):
		if HASNONDIGIT.search(word):
			sig += "-n"
		else:
			sig += "-N"

	# punctuation
	if "-" in word:
		sig += "-H"
	if word == ".":
		sig += "-P"
	if word == ",":
		sig += "-C"
	if len(word) > 3:
		if word[-1] in LOWERUPPER:
			sig += "-%s" % word[-2:].lower()
	return sig

NOUNSUFFIX = re.compile("(ier|ière|ité|ion|ison|isme|ysme|iste|esse|eur|euse"
		"|ence|eau|erie|ng|ette|age|ade|ance|ude|ogue|aphe|ate|duc|anthe"
		"|archie|coque|érèse|ergie|ogie|lithe|mètre|métrie|odie|pathie|phie"
		"|phone|phore|onyme|thèque|scope|some|pole|ôme|chromie|pie)s?$")
ADJSUFFIX = re.compile("(iste|ième|uple|issime|aire|esque|atoire|ale|al|able"
		"|ible|atif|ique|if|ive|eux|aise|ent|ois|oise|ante|el|elle|ente|oire"
		"|ain|aine)s?$")
POSSIBLEPLURAL = re.compile("(s|ux)$")
VERBSUFFIX = re.compile("(ir|er|re|ez|ont|ent|ant|ais|ait|ra|era|eras"
		"|é|és|ées|isse|it)$")
ADVSUFFIX = re.compile("(iment|ement|emment|amment)$")
HASPUNC = re.compile("([\u0021-\u002F\u003A-\u0040\u005B\u005C\u005D"
		"\u005E-\u0060\u007B-\u007E\u00A1-\u00BF\u2010-\u2027\u2030-\u205E"
		"\u20A0-\u20B5])+")
ISPUNC = re.compile("([\u0021-\u002F\u003A-\u0040\u005B\u005C\u005D"
		"\u005E-\u0060\u007B-\u007E\u00A1-\u00BF\u2010-\u2027\u2030-\u205E"
		"\u20A0-\u20B5])+$")


def unknownwordftb(word, loc, _lexicon):
	"""Model 2 for French of the Stanford parser."""
	sig = UNK

	if ADVSUFFIX.search(word):
		sig += "-ADV"
	elif VERBSUFFIX.search(word):
		sig += "-VB"
	elif NOUNSUFFIX.search(word):
		sig += "-NN"

	if ADJSUFFIX.search(word):
		sig += "-ADV"
	if HASDIGIT.search(word):
		sig += "-NUM"
	if POSSIBLEPLURAL.search(word):
		sig += "-PL"

	if ISPUNC.search(word):
		sig += "-ISPUNC"
	elif HASPUNC.search(word):
		sig += "-HASPUNC"

	if loc > 0 and len(word) > 0 and word[0] in UPPER:
		sig += "-UP"

	return sig


UNKNOWNWORDFUNC = {
		"4": unknownword4,
		"6": unknownword6,
		"base": unknownwordbase,
		"ftb": unknownwordftb,
		}


# === Performing POS tagging with external tools ============
def externaltagging(usetagger, model, sents, overridetag, tagmap):
	"""Use an external tool to tag a list of sentences."""
	logging.info('Start tagging.')
	goldtags = [t for sent in sents.values() for _, t in sent]
	if usetagger == 'treetagger':  # Tree-tagger
		if not os.path.exists('tree-tagger/bin/tree-tagger'):
			raise ValueError(TREETAGGERHELP)
		infile, inname = tempfile.mkstemp(text=True)
		with os.fdopen(infile, 'w') as infile:
			for tagsent in sents.values():
				sent = map(itemgetter(0), tagsent)
				infile.write('\n'.join(w.encode('utf-8')
					for w in sent) + '\n<S>\n')
		filtertags = ''
		if not model:
			model = 'tree-tagger/lib/german-par-linux-3.2-utf8.bin'
			filtertags = '| tree-tagger/cmd/filter-german-tags'
		tagger = Popen('tree-tagger/bin/tree-tagger -token -sgml'
				' %s %s %s' % (model, inname, filtertags),
				stdout=PIPE, shell=True)
		tagout = tagger.stdout.read(
				).decode('utf-8').split('<S>')[:-1]
		os.unlink(inname)
		taggedsents = OrderedDict((n, [tagmangle(a, None, overridetag, tagmap)
					for a in tags.splitlines() if a.strip()])
					for n, tags in zip(sents, tagout))
	elif usetagger == 'stanford':  # Stanford Tagger
		if not os.path.exists('stanford-postagger-full-2012-07-09'):
			raise ValueError(STANFORDTAGGERHELP)
		infile, inname = tempfile.mkstemp(text=True)
		with os.fdopen(infile, 'w') as infile:
			for tagsent in sents.values():
				sent = map(itemgetter(0), tagsent)
				infile.write(' '.join(w.encode('utf-8')
					for w in sent) + '\n')
		if not model:
			model = 'models/german-hgc.tagger'
		tagger = Popen(args=(
				'/usr/bin/java -mx2G -classpath stanford-postagger.jar'
				' edu.stanford.nlp.tagger.maxent.MaxentTagger'
				' -tokenize false -encoding utf-8'
				' -model %s -textFile %s' % (model, inname)).split(),
				cwd='stanford-postagger-full-2012-07-09',
				shell=False, stdout=PIPE)
		tagout = tagger.stdout.read(
				).decode('utf-8').splitlines()
		os.unlink(inname)
		taggedsents = OrderedDict((n, [tagmangle(a, '_', overridetag, tagmap)
			for a in tags.split()]) for n, tags in zip(sents, tagout))
	elif usetagger == 'frog':  # Dutch 'frog' tagger
		tagger = Popen(args=[which('frog')] +
					'-n --skip=tacmnp -t /dev/stdin'.split(),
				shell=False, stdin=PIPE, stdout=PIPE)
		tagout, stderr = tagger.communicate(''.join(
				' '.join(w for w in map(itemgetter(0), tagsent)) + '\n'
				for tagsent in sents.values()).encode('utf8'))
		logging.info(stderr)
		# lines consist of: 'idx token lemma POS score'
		taggedsents = OrderedDict((n,
				[(line.split()[1],
					line.split()[3].replace('(', '[').replace(')', ']'))
					for line in lines.splitlines()]) for n, lines
				in zip(sents, tagout.decode('utf-8').split('\n\n')))
	if len(taggedsents) != len(sents):
		raise ValueError('mismatch in number of sentences after tagging.')
	for n, tags in taggedsents.items():
		if len(sents[n]) != len(tags):
			raise ValueError('mismatch in number of tokens after tagging.\n'
				'before: %r\nafter: %r' % (sents[n], tags))
	newtags = [t for sent in taggedsents.values() for _, t in sent]
	logging.info('Tag accuracy: %5.2f\ngold - cand: %r\ncand - gold %r',
		(100 * accuracy(goldtags, newtags)),
		set(goldtags) - set(newtags), set(newtags) - set(goldtags))
	return taggedsents


def accuracy(gold, cand):
	"""Compute fraction of equivalent pairs in two sequences."""
	return sum(a == b for a, b in zip(gold, cand)) / len(gold)


def tagmangle(a, splitchar, overridetag, tagmap):
	"""Function to filter tags after they are produced by the tagger."""
	word, tag = a.rsplit(splitchar, 1)
	for newtag in overridetag:
		if word in overridetag[newtag]:
			tag = newtag
	return word, tagmap.get(tag, tag)


__all__ = ['getunknownwordmodel', 'replaceraretrainwords',
		'replaceraretestwords', 'simplesmoothlexicon', 'getlexmodel',
		'smoothlexicon', 'unknownword6', 'unknownword4', 'unknownwordbase',
		'unknownwordftb', 'externaltagging', 'tagmangle']
