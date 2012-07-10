import re, logging
from random import random
from heapq import nlargest
from math import fsum, exp, log
from bisect import bisect_right
from collections import defaultdict
from operator import itemgetter
from nltk import Tree
from grammar import induce_srcg, recoverfromfragments_str, canonicalize
from plcfrs import parse
from treetransforms import unbinarize
import cython
assert cython.compiled

infinity = float('infinity')
removeids = re.compile("@[0-9]+")

def marginalize(chart, start, tolabel, n=10, sample=False, both=False,
	shortest=False, secondarymodel=None, mpd=False, backtransform=None):
	""" approximate MPP or MPD by summing over n random/best derivations from
	chart, return a dictionary mapping parsetrees to probabilities """
	parsetrees = {}
	derivations = set()
	if sample or both:
		derivations = getsamples(chart, start, n, tolabel)
	if not sample or both:
		derivations.update(lazykbest(chart, start, n, tolabel))
	if shortest:
		maxprob = min(derivations, key=itemgetter(1))[1]
		derivations = [(a,b) for a, b in derivations if b == maxprob]
	if backtransform is not None:
		derivations = list(derivations)
		for n, (deriv, prob) in enumerate(derivations):
			tree = Tree.parse(deriv, parse_leaf=int)
			unbinarize(tree, childChar="}")
			derivations[n] = (recoverfromfragments_str(canonicalize(tree),
				backtransform), prob)
	for deriv, prob in derivations:
		# simple way of adding probabilities (too easy):
		#parsetrees[removeids.sub("", deriv)] += exp(-prob)
		# restore linear precedence (disabled, seems to make no difference):
		#parsetree = Tree(removeids.sub("", deriv))
		#for a in list(parsetree.subtrees())[::-1]:
		#	a.sort(key=lambda x: x.leaves())
		#parsetrees[parsetree.pprint(margin=999)].append(-prob)
		if shortest:
			# calculate the derivation probability in a different model.
			# because we don't keep track of which rules have been used,
			# read off the rules from the derivation ...
			tree = Tree.parse(removeids.sub("", deriv), parse_leaf=int)
			sent = sorted(tree.leaves())
			rules = induce_srcg([tree], [sent], arity_marks=False)
			prob = -fsum([secondarymodel[r] for r, w in rules
											if r[0][1] != 'Epsilon'])

		tree = removeids.sub("@" if mpd else "", deriv)
		if tree in parsetrees: parsetrees[tree].append(-prob)
		else: parsetrees[tree] = [-prob]
	# Adding probabilities in log space
	# http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point-numbers
	# https://facwiki.cs.byu.edu/nlp/index.php/Log_Domain_Computations
	for parsetree in parsetrees:
		maxprob = max(parsetrees[parsetree])
		parsetrees[parsetree] = exp(fsum([maxprob, log(fsum([exp(prob - maxprob)
									for prob in parsetrees[parsetree]]))]))
	logging.debug("(%d derivations, %d parsetrees)" % (
		len(derivations), len(parsetrees)))
	return parsetrees

def sldop(chart, start, sent, tags, dopgrammar, secondarymodel, m, sldop_n,
	sample=False, both=False):
	""" `proper' method for sl-dop. parses sentence once more to find shortest
	derivations, pruning away any chart item not occurring in the n most
	probable parse trees. Returns the first result of the intersection of the
	most probable parse trees and shortest derivations.
	NB: doesn't seem to work so well, so may contain a subtle bug.
	NB2: assumes ROOT is the top node."""
	# get n most likely derivations
	derivations = set()
	if sample:
		derivations = getsamples(chart, start, 10*m, dopgrammar.tolabel)
	if not sample or both:
		derivations.update(lazykbest(chart, start, m, dopgrammar.tolabel))
	derivations = dict(derivations)
	# sum over Goodman derivations to get parse trees
	idsremoved = defaultdict(set)
	for t, p in derivations.items():
		idsremoved[removeids.sub("", t)].add(t)
	mpp1 = dict((tt, sumderivs(ts, derivations))
					for tt, ts in idsremoved.iteritems())
	prunelist = [{} for a in secondarymodel.toid]
	for a in chart:
		prunelist[getlabel(a)][getvec(a)] = infinity
	for tt in nlargest(sldop_n, mpp1, key=lambda t: mpp1[t]):
		for n in Tree(tt).subtrees():
			vec = sum([1L << int(x) for x in n.leaves()])
			prunelist[secondarymodel.toid[n.node]][vec] = 0.0
	for label, n in secondarymodel.toid.items():
		prunelist[n] = prunelist[secondarymodel.toid[label.split("@")[0]]]
	mpp2 = {}
	for tt in nlargest(sldop_n, mpp1, key=lambda t: mpp1[t]):
		mpp2[tt] = mpp1[tt]

	words = [a[0] for a in sent]
	tagsornil = [a[1] for a in sent] if tags else []
	chart2, start2 = parse(words, secondarymodel, tagsornil,
					1, #secondarymodel.toid[top],
					True, None, prunelist=prunelist)
	if start2:
		shortestderivations = lazykbest(chart2, start2, m,
			secondarymodel.tolabel)
	else:
		shortestderivations = []
		logging.warning("shortest derivation parsing failed") # error?
	mpp = [ max(mpp2.items(), key=itemgetter(1)) ]
	for t, s in shortestderivations:
		tt = removeids.sub("", t)
		if tt in mpp2:
			mpp = [ ( tt, (s / log(0.5)	, mpp2[tt])) ]
			break
	else:
		logging.warning("no matching derivation found") # error?
	logging.debug("(%d derivations, %d of %d parsetrees)" % (
		len(derivations), min(sldop_n, len(mpp1)), len(mpp1)))
	return mpp

def sldop_simple(chart, start, dopgrammar, m, sldop_n):
	""" simple sl-dop method; estimates shortest derivation directly from
	number of addressed nodes in the k-best derivations. After selecting the n
	best parse trees, the one with the shortest derivation is returned."""
	# get n most likely derivations
	derivations = lazykbest(chart, start, m, dopgrammar.tolabel)
	x = len(derivations); derivations = set(derivations)
	xx = len(derivations); derivations = dict(derivations)
	if xx != len(derivations):
		logging.error("duplicates w/different probabilities %d => %d => %d" % (
			x, xx, len(derivations)))
	elif x != xx:
		logging.error("DUPLICATES DUPLICATES %d => %d" % (x, len(derivations)))
	# sum over Goodman derivations to get parse trees
	idsremoved = defaultdict(set)
	for t, p in derivations.items():
		idsremoved[removeids.sub("", t)].add(t)
	mpp1 = dict((tt, sumderivs(ts, derivations))
					for tt, ts in idsremoved.items())
	# the number of fragments used is the number of
	# nodes (open parens), minus the number of interior
	# (addressed) nodes.
	mpp = [(tt, (-minunaddressed(tt, idsremoved), mpp1[tt]))
				for tt in nlargest(sldop_n, mpp1, key=lambda t: mpp1[t])]
	logging.debug("(%d derivations, %d of %d parsetrees)" % (len(derivations),
														len(mpp), len(mpp1)))
	return mpp

def sumderivs(ts, derivations):
	#return fsum([exp(-derivations[t]) for t in ts])
	return sum([exp(-derivations[t]) for t in ts])

def minunaddressed(tt, idsremoved):
	return min([(t.count("(") - t.count("@")) for t in idsremoved[tt]])

def samplechart(chart, start, tolabel, tables):
	""" Samples a derivation from a chart. """
	#NB: this does not sample properly, as it ignores the distribution of
	#probabilities and samples uniformly instead. 
	#edge = choice(chart[start])
	rnd = random() * tables[start][-1]
	idx = bisect_right(tables[start], rnd)
	edge = chart[start][idx]
	if edge.left.label == 0: # == "Epsilon":
		return "(%s %d)" % (tolabel[start.label], edge.left.vec), edge.prob
	children = [samplechart(chart, child, tolabel, tables)
				for child in (edge.left, edge.right) if child.label]
	tree = "(%s %s)" % (tolabel[start.label],
							" ".join([a for a,b in children]))
	return tree, edge.prob + sum([b for a,b in children])

	#probmass = sum([exp(-edge.prob) for edge in edges])
	#minprob = min([edge.prob for edge in edges])
	#probmass = exp(fsum([minprob,
	#					log(fsum([exp(edge.prob - minprob)
	#								for edge in edges]))]))

def getsamples(chart, start, n, tolabel):
	tables = {}
	for item in chart:
		chart[item].sort(key=lambda edge: edge.prob)
		tables[item] = []; prev = 0.0
		minprob = chart[item][0].prob
		for edge in chart[item]:
			#prev += exp(-edge.prob); tables[item].append(prev)
			prev += exp(-minprob - edge.prob)
			tables[item].append(exp(minprob + log(prev)))
	derivations = set([samplechart(chart, start, tolabel, tables)
						for x in range(n)])
	derivations.discard(None)
	return derivations

def viterbiderivation(chart, start, tolabel):
	edge = edgecast(min(chart[start]))
	return getviterbi(chart, start, tolabel), edge.inside

def getviterbi(chart, start, tolabel):
	edge = min(chart[start])
	if edge.right.label: #binary
		return "(%s %s %s)" % (tolabel[start.label],
					getviterbi(chart, edge.left, tolabel),
					getviterbi(chart, edge.right, tolabel))
	else: #unary or terminal
		return "(%s %s)" % (tolabel[start.label],
					getviterbi(chart, edge.left, tolabel) if edge.left.label
									else str(edge.left.vec))

def main():
	from nltk import Tree
	from grammar import dop_srcg_rules
	from containers import Grammar
	from plcfrs import parse
	def e(x):
		if isinstance(x[1], tuple):
			return x[0].replace("@", ""), (int(abs(x[1][0])), x[1][1])
		return x[0].replace("@", ""), x[1]
	def f(x):
		return x[0], exp(-x[1])
	trees = map(lambda t: Tree.parse(t, parse_leaf=int),
		"""(ROOT (A (A 0) (B 1)) (C 2))
		(ROOT (C 0) (A (A 1) (B 2)))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (B (A 0) (B 1)) (C 2))
		(ROOT (A 0) (C (B 1) (C 2)))
		(ROOT (A 0) (C (B 1) (C 2)))""".splitlines())
	sents = [a.split() for a in
		"""d b c
		c a b
		a e f
		a e f
		a e f
		a e f
		d b f
		d b f
		d b f
		d b g
		e f c
		e f c
		e f c
		e f c
		e f c
		e f c
		f b c
		a d e""".splitlines()]
	grammar = Grammar(dop_srcg_rules(trees, sents))
	shortest, secondarymodel = dop_srcg_rules(trees, sents, shortestderiv=True)
	shortest = Grammar(shortest)
	sent = "a b c".split()
	chart, start = parse(sent, grammar, None, grammar.toid['ROOT'], True)
	vit = viterbiderivation(chart, start, grammar.tolabel)
	mpd = marginalize(chart, start, grammar.tolabel, n=1000, mpd=True)
	mpp = marginalize(chart, start, grammar.tolabel, n=1000)
	mppsampled = marginalize(chart, start, grammar.tolabel, n=1000, sample=True)
	sldopsimple = sldop_simple(chart, start, grammar, 1000, 7)
	sldop1 = sldop(chart, start, sent, None, grammar, shortest, 1000, 7,
		sample=False, both=False)
	short = marginalize(chart, start, shortest.tolabel, 1000, shortest=True,
		secondarymodel=secondarymodel)
	print
	print "vit:\t\t%s %r" % e((removeids.sub("", vit[0]), exp(-vit[1])))
	print "MPD:\t\t%s %r" % e(max(mpd.items(), key=itemgetter(1)))
	print "MPP:\t\t%s %r" % e(max(mpp.items(), key=itemgetter(1)))
	print "MPP sampled:\t%s %r" % e(max(mppsampled.items(), key=itemgetter(1)))
	print "SL-DOP n=7:\t%s %r" % e(max(sldop1, key=itemgetter(1)))
	print "simple SL-DOP:\t%s %r" % e(max(sldopsimple, key=itemgetter(1)))
	print "shortest:\t%s %r" % f(min(short.items(), key=itemgetter(1)))

if __name__ == '__main__': main()
