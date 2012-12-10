import re, logging
from heapq import nlargest
from math import fsum, exp, log
from random import random
from bisect import bisect_right
from operator import itemgetter, attrgetter
from itertools import count
from collections import defaultdict, OrderedDict
from nltk import Tree
from kbest import lazykbest, lazykthbest, getderiv
from parser import parse
from agenda cimport Entry
from grammar import induce_plcfrs, rangeheads
from treetransforms import unbinarize #, canonicalize
from containers cimport ChartItem, SmallChartItem, FatChartItem, CFGChartItem,\
		Edge, LCFRSEdge, CFGEdge, RankedEdge, RankedCFGEdge, Grammar,\
		UChar, UInt, ULong, ULLong
cimport cython

from libc.string cimport memset
cdef extern from "macros.h":
	void SETBIT(ULong a[], int b)

infinity = float('infinity')
removeids = re.compile("@[-0-9]+")
removewordtags = re.compile("@[^ )]+")
termsre = re.compile(r" ([0-9]+)\b")

cpdef marginalize(method, chart, ChartItem start, Grammar grammar, int n,
		bint sample=False, bint kbest=True, list sent=None, bint tags=False,
		secondarymodel=None, int sldop_n=7,
		dict backtransform=None, bint newdd=False):
	""" approximate MPP or MPD by summing over n random/best derivations from
	chart, return a dictionary mapping parsetrees to probabilities """
	cdef bint mpd = method == "mpd"
	cdef bint shortest = method == "shortest"
	cdef Entry entry
	cdef dict parsetrees = <dict>defaultdict(float)
	cdef list derivations = [], entries
	cdef str treestr, deriv, debin = None if newdd or shortest else "}<"
	cdef double prob, maxprob
	cdef int m

	assert kbest or sample
	if kbest:
		derivations, D = lazykbest(chart, start, n, grammar.tolabel,
				debin, derivs=not newdd)
	if sample:
		assert not newdd, "sampling not implemented for new double dop."
		assert not isinstance(start, CFGChartItem), (
				"sampling not implemented for PCFG charts.")
		derivations.extend(getsamples(chart, start, n, grammar.tolabel, debin))

	if method == "sl-dop":
		assert backtransform is None, "sl-dop not implemented for double-dop"
		assert not isinstance(start, CFGChartItem), (
				"sl-dop not implemented for PCFG charts.")
		return sldop(dict(derivations), chart, start, sent, tags, grammar,
				secondarymodel, n, sldop_n, backtransform)
	elif method == "sl-dop-simple":
		assert not newdd, "%s not implemented for new double dop" % method
		return sldop_simple(dict(derivations), n, sldop_n, backtransform)
	elif method == "shortest":
		# filter out all derivations which are not shortest
		if derivations:
			_, maxprob = min(derivations, key=itemgetter(1))
			derivations = [(a, b) for a, b in derivations if b == maxprob]

	if newdd:
		if isinstance(start, CFGChartItem):
			entries = D[(<CFGChartItem>start).start][
					(<CFGChartItem>start).end][start.label]
		else:
			entries = D[start]
		for entry in entries:
			prob = entry.value
			treestr = recoverfragments_new(entry.key, D,
					grammar, backtransform)
			if mpd:
				if exp(-prob) > parsetrees[treestr]:
					parsetrees[treestr] = exp(-prob)
			else:
				# simple way of adding probabilities (too easy):
				parsetrees[treestr] += exp(-prob)
				#if treestr in parsetrees:
				#	parsetrees[treestr].append(-prob)
				#else:
				#	parsetrees[treestr] = [-prob]
	else: #DOP reduction / old double dop method
		for deriv, prob in derivations:
			if shortest:
				# for purposes of tie breaking, calculate the derivation
				# probability in a different model. because we don't keep track
				# of which rules have been used, read off the rules from the
				# derivation ...
				tree = Tree.parse(deriv, parse_leaf=int)
				prob = -sum([secondarymodel.get(r, 0.0) for r, _
					in induce_plcfrs([tree], [[w for w, _ in sent]])])
				if backtransform is not None:
					# tie breaking relies on binarized productions,
					# to recover derivation we need to unbinarize
					deriv = unbinarize(tree, childChar="}").pprint(margin=9999)
			if backtransform is None:
				treestr = removeids.sub("@" if mpd else "", deriv)
			else:
				treestr = recoverfragments(deriv, backtransform)
			if backtransform is None or not mpd:
				# simple way of adding probabilities (too easy):
				parsetrees[treestr] += exp(-prob)
				#if treestr in parsetrees:
				#	parsetrees[treestr].append(-prob)
				#else:
				#	parsetrees[treestr] = [-prob]
			else:
				if exp(-prob) > parsetrees[treestr]:
					parsetrees[treestr] = exp(-prob)

	# Adding probabilities in log space
	# http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point-numbers
	# https://facwiki.cs.byu.edu/nlp/index.php/Log_Domain_Computations
	#for parsetree in parsetrees:
	#	maxprob = max(parsetrees[parsetree])
	#	parsetrees[parsetree] = exp(fsum([maxprob, log(fsum([exp(prob - maxprob)
	#								for prob in parsetrees[parsetree]]))]))
	msg = "%d derivations, %d parsetrees" % (
			len(entries if newdd else derivations), len(parsetrees))
	return parsetrees, msg

cdef sldop(dict derivations, chart, ChartItem start, list sent, bint tags,
		Grammar dopgrammar, Grammar secondarymodel, int m, int sldop_n,
		dict backtransform):
	""" `proper' method for sl-dop. parses sentence once more to find shortest
	derivations, pruning away any chart item not occurring in the n most
	probable parse trees. Returns the first result of the intersection of the
	most probable parse trees and shortest derivations.
	NB: doesn't seem to work so well, so may contain a subtle bug.
		should be rewritten to support PCFG charts, double-dop, etc.
	NB2: assumes ROOT is the top node."""
	cdef ChartItem item
	cdef FatChartItem fitem
	# collect derivations for each parse tree
	derivsfortree = defaultdict(set)
	for deriv in derivations:
		if backtransform is None:
			derivsfortree[removeids.sub("", deriv)].add(deriv)
		else:
			tree = recoverfragments(deriv, backtransform)
	# sum over derivations to get parse trees
	parsetreeprob = {}
	for tree, derivs in derivsfortree.iteritems():
		parsetreeprob[tree] = sum([exp(-derivations[d]) for d in derivs])

	# use getmapping and prunechart here instead of manually built whitelist
	# prunechart(chart, n, ...)
	# parse ...
	# OR: use two coarse-to-fine stages, the second collects SL-DOP parse
	whitelist = [{} for a in secondarymodel.toid]
	for a in chart:
		whitelist[(<ChartItem>a).label][a] = infinity
	for tt in nlargest(sldop_n, parsetreeprob, key=parsetreeprob.get):
		for n in Tree(tt).subtrees():
			if len(sent) < sizeof(ULLong) * 8:
				item = SmallChartItem(0, sum([1L << int(x)
						for x in n.leaves()]))
			else:
				fitem = item = FatChartItem(0)
				memset(<char *>fitem.vec, 0, sizeof(fitem.vec))
				for x in n.leaves():
					SETBIT(fitem.vec, x)
			whitelist[secondarymodel.toid[n.node]][item] = 0.0
	for label, n in secondarymodel.toid.items():
		whitelist[n] = whitelist[secondarymodel.toid[label.split("@")[0]]]
	mpp2 = {}
	for tt in nlargest(sldop_n, parsetreeprob, key=parsetreeprob.get):
		mpp2[tt] = parsetreeprob[tt]

	words = [a[0] for a in sent]
	tagsornil = [a[1] for a in sent] if tags else []
	chart2, start2, _ = parse(words, secondarymodel, tagsornil,
					exhaustive=True, estimates=None, whitelist=whitelist)
	if start2:
		shortestderivations = lazykbest(chart2, start2, m,
			secondarymodel.tolabel)[0]
	else:
		shortestderivations = []
		logging.warning("shortest derivation parsing failed") # error?
	result = dict([max(mpp2.items(), key=itemgetter(1))])
	for deriv, s in shortestderivations:
		tt = removeids.sub("", deriv)
		if tt in mpp2:
			result = dict([(tt, (s / log(0.5), mpp2[tt]))])
			break
	else:
		logging.warning("no matching derivation found") # error?
	msg = "(%d derivations, %d of %d parsetrees)" % (
		len(derivations), min(sldop_n, len(parsetreeprob)), len(parsetreeprob))
	return result, msg

cdef sldop_simple(dict derivations, int m, int sldop_n, dict backtransform):
	""" simple sl-dop method; estimates shortest derivation directly from
	number of addressed nodes in the k-best derivations. After selecting the n
	best parse trees, the one with the shortest derivation is returned."""
	derivsfortree = defaultdict(set)
	# collect derivations for each parse tree
	for deriv in derivations:
		if backtransform is None:
			tree = removeids.sub("", deriv)
		else:
			tree = recoverfragments(deriv, backtransform)
		derivsfortree[tree].add(deriv)

	# sum over derivations to get parse trees
	parsetreeprob = {}
	for tree, derivs in derivsfortree.iteritems():
		parsetreeprob[tree] = sum([exp(-derivations[d]) for d in derivs])
	selectedtrees = nlargest(sldop_n, parsetreeprob, key=parsetreeprob.get)

	# the number of fragments used is the number of
	# nodes (open parens), minus the number of interior
	# (addressed) nodes.
	result = dict([(tree, (-min([deriv.count("(") - (
			deriv.count("@") + deriv.count("(#"))
		for deriv in derivsfortree[tree]]), parsetreeprob[tree]))
				for tree in selectedtrees])
	msg = "(%d derivations, %d of %d parsetrees)" % (
			len(derivations), len(result), len(parsetreeprob))
	return result, msg

cdef samplechart(dict chart, ChartItem start, dict tolabel, dict tables,
		str debin):
	""" Samples a derivation from a chart. """
	cdef LCFRSEdge edge
	cdef ChartItem child
	#NB: this does not sample properly, as it ignores the distribution of
	#probabilities and samples uniformly instead. 
	#edge = choice(chart[start])
	if start.label == 0:
		return str(start.lexidx()), 0.0
	rnd = random() * tables[start][-1]
	idx = bisect_right(tables[start], rnd)
	edge = chart[start][idx]
	if edge.left.label == 0: # == "Epsilon":
		idx = edge.left.lexidx()
		return "(%s %d)" % (tolabel[start.label], idx), edge.inside
	children = [samplechart(chart, child, tolabel, tables, debin)
		for child in (edge.left, edge.right) if child.label]
	if debin is not None and debin in tolabel[start.label]:
		tree = " ".join([a for a, _ in children])
	else:
		tree = "(%s %s)" % (tolabel[start.label],
								" ".join([a for a,b in children]))
	return tree, edge.rule.prob + sum([b for _, b in children])

def getsamples(chart, start, n, tolabel, debin=None):
	""" Samples n derivations from a chart. """
	cdef LCFRSEdge edge
	cdef dict tables = {}, chartcopy = {}
	for item in chart:
		#FIXME: work w/inside prob right?
		chartcopy[item] = sorted(chart[item])
		#chart[item].sort(key=attrgetter('prob'))
		tables[item] = []
		prev = 0.0
		minprob = (<LCFRSEdge>min(chart[item])).inside
		for edge in chartcopy[item]:
			prev += exp(-minprob - edge.inside)
			tables[item].append(exp(minprob + log(prev)))
	derivations = set([samplechart(chartcopy, start, tolabel, tables, debin)
						for x in range(n)])
	derivations.discard(None)
	return derivations

cpdef viterbiderivation(chart, ChartItem start, dict tolabel):
	cdef Edge edge
	cdef CFGChartItem tmp
	# Ask for at least 10 derivations because unary cycles.
	derivations, _ = lazykbest(chart, start, 10, tolabel)
	return derivations[0]

def repl(d):
	def replfun(x):
		return " %d" % d[int(x.group(1))]
	return replfun
cpdef recoverfragments(derivation, dict backtransform):
	""" Reconstruct a DOP derivation from a DOP derivation with
	flattened fragments. `derivation' should be a string, and
	backtransform should contain strings as keys and values.
	Returns expanded derivation as a string. """
	cdef list leaves, childlabels, childleaves, childheads, childintheads
	cdef list children = []
	cdef dict leafmap = {}
	cdef str prod, rprod, result, frontier, replacement
	cdef int parens = 0, start = 0, prev = -2, prevparent, n
	#assert derivation.count("(") == derivation.count(")")

	# handle ambiguous fragments with nodes of the form "#n"
	n = derivation.index(" ") + 1
	if derivation[n:].startswith("(#"):
		derivation = derivation[n:len(derivation)-1]

	# tokenize tree structure into direct children
	for n, a in enumerate(derivation):
		if a == '(':
			if parens == 1:
				start = n
			parens += 1
		elif a == ')':
			parens -= 1
			if parens == 1:
				children.append(derivation[start:n+1])
				start = n + 2

	# extract the top production with its ranges
	if children[0].startswith("("):
		childlabels = [a[1:a.index(" ")] for a in children]
		childleaves = [map(int, termsre.findall(a)) for a in children]
		childintheads = [rangeheads(sorted(a)) for a in childleaves]
		childheads = [" ".join(map(str, a)) for a in childintheads]
		prod = "%s %s)" % (derivation[:derivation.index(" ")],
				" ".join(["(%s %s)" % a for a in zip(childlabels, childheads)]))
	else:
		prod = derivation

	#renumber the ranges to canonical form
	leafmap = dict([(b, n) for n, a in enumerate(childleaves) for b in a])
	prev = prevparent = n = -2
	for a in sorted(leafmap):
		if a != prev + 1: # a discontinuity
			n += 2
			prevparent = leafmap[a]
			leafmap[a] = n
		elif leafmap[a] != prevparent: # same component, different non-terminal
			n += 1
			prevparent = leafmap[a]
			leafmap[a] = n
		else:
			del leafmap[a]
		prev = a

	rprod = termsre.sub(repl(leafmap), prod)
	leafmap = dict(zip(leafmap.values(), leafmap.keys()))
	# fetch the actual fragment corresponding to this production
	try:
		result = backtransform[rprod] #, rprod)
	except KeyError:
		#print 'backtransform'
		#for a in backtransform:
		#	print a
		print '\nleafmap', leafmap
		print prod
		print derivation
		print zip(childlabels, childheads)
		raise

	# revert renumbering
	result = termsre.sub(repl(leafmap), result)
	# recursively expand all substitution sites
	for t, theads in zip(children, childheads):
		#if t.startswith("(") and "(" in t[1:]: # redundant?
		if "(" in t[1:]:
			frontier = "%s %s)" % (t[:t.index(" ")], theads)
			#assert frontier in result, (frontier, result)
			replacement = recoverfragments(t, backtransform)
			result = result.replace(frontier, replacement, 1)
	#assert result.count("(") == result.count(")")
	return result

cpdef str recoverfragments_new(derivation, D, Grammar grammar,
		dict backtransform):
	""" Reconstruct a DOP derivation from a DOP derivation with
	flattened fragments which are left-binarized. `derivation' should be
	a RankedEdge representing a derivation, and backtransform should contain
	rule numbers as keys and strings as values. Uses the first binarized
	production as key, which map to string templates as values. Returns
	expanded derivation as a string.

	Does on-the-fly debinarization following labels that are not mapped to a
	label in the coarse grammar, i.e., it assumes that neverblockre is only
	used to avoid blocking nonterminals from the double-dop binarization
	(containing the string '}<'). Note that this means getmapping() has to have
	been called on `grammar', even when not doing coarse-to-fine parsing. """
	if isinstance(derivation, RankedEdge):
		return removewordtags.sub("", recoverfragments_new_lcfrs(
				derivation, D, grammar, backtransform))
	elif isinstance(derivation, RankedCFGEdge):
		return removewordtags.sub("", recoverfragments_new_cfg(
				derivation, D, grammar, backtransform))
	raise ValueError("derivation should be RankedEdge or RankedCFGEdge.")

cdef str recoverfragments_new_lcfrs(RankedEdge derivation, dict D,
		Grammar grammar, dict backtransform):
	cdef RankedEdge child
	cdef LCFRSEdge childedge, derivedge = derivation.edge
	cdef list children = []

	# get fragment
	result = backtransform[(<LCFRSEdge>derivation.edge).rule.no]

	# recursively expand all substitution sites,
	# w/on the fly left-factored debinarization
	if derivedge.right.label: # is there a right child?
		# keep going while left child is part of same binarized constituent
		while grammar.mapping[derivedge.left.label] == 0:
			# one of the right children
			child = (<Entry>D[derivedge.right][derivation.right]).key
			childedge = child.edge
			children.append(('(%s %d)' % (
				grammar.tolabel[child.head.label], childedge.left.lexidx()))
				if childedge.rule is NULL else recoverfragments_new_lcfrs(
						child, D, grammar, backtransform))
			# move on to next node in this binarized constituent
			derivation = (<Entry>D[derivedge.left][derivation.left]).key
			derivedge = derivation.edge
		# last right child
		if derivedge.right.label: # is there a right child?
			child = (<Entry>D[derivedge.right][derivation.right]).key
			childedge = child.edge
			children.append('(%s %d)' % (
				grammar.tolabel[child.head.label], childedge.left.lexidx())
				if childedge.rule is NULL else recoverfragments_new_lcfrs(
						child, D, grammar, backtransform))
	elif grammar.mapping[derivedge.left.label] == 0:
		derivation = (<Entry>D[derivedge.left][derivation.left]).key
		derivedge = derivation.edge
	# left-most child
	child = (<Entry>D[derivedge.left][derivation.left]).key
	childedge = child.edge
	children.append(('(%s %d)' % (
		grammar.tolabel[child.head.label], childedge.left.lexidx()))
		if childedge.rule is NULL else recoverfragments_new_lcfrs(
				child, D, grammar, backtransform))
	children.reverse()
	return result.format(*children)

cdef str recoverfragments_new_cfg(RankedCFGEdge derivation, list D,
		Grammar grammar, dict backtransform):
	cdef RankedCFGEdge child
	cdef CFGEdge childedge, derivedge = derivation.edge
	cdef list children = []

	# get fragment
	result = backtransform[(<CFGEdge>derivation.edge).rule.no]

	# recursively expand all substitution sites,
	# w/on the fly left-factored debinarization
	if derivedge.rule.rhs2: # is there a right child?
		# keep going while left child is part of same binarized constituent
		# this shortcut assumes that neverblockre is only used to avoid
		# blocking nonterminals from the double-dop binarization.
		while grammar.mapping[derivedge.rule.rhs1] == 0:
			# one of the right children
			child = (<Entry>D[derivedge.mid][derivation.end][
					derivedge.rule.rhs2][derivation.right]).key
			childedge = child.edge
			children.append(('(%s %d)' % (grammar.tolabel[child.label],
				child.start)) if childedge.rule is NULL else
				recoverfragments_new_cfg(child, D, grammar, backtransform))
			# move on to next node in this binarized constituent
			derivation = (<Entry>D[derivation.start][derivedge.mid][
					derivedge.rule.rhs1][derivation.left]).key
			derivedge = derivation.edge
		# last right child
		if derivedge.rule.rhs2: # is there a right child?
			child = (<Entry>D[derivedge.mid][derivation.end][
					derivedge.rule.rhs2][derivation.right]).key
			childedge = child.edge
			children.append(('(%s %d)' % (grammar.tolabel[child.label],
				child.start)) if childedge.rule is NULL else
				recoverfragments_new_cfg(child, D, grammar, backtransform))
	elif grammar.mapping[derivedge.rule.rhs1] == 0:
		derivation = (<Entry>D[derivation.start][derivedge.mid][
				derivedge.rule.rhs1][derivation.left]).key
		derivedge = derivation.edge
	# left-most child
	child = (<Entry>D[derivation.start][derivedge.mid][
			derivedge.rule.rhs1][derivation.left]).key
	childedge = child.edge
	children.append(('(%s %d)' % (grammar.tolabel[child.label], child.start))
		if childedge.rule is NULL else recoverfragments_new_cfg(
				child, D, grammar, backtransform))
	children.reverse()
	return result.format(*children)

def main():
	from nltk import Tree
	from grammar import dopreduction
	from containers import Grammar
	from parser import parse
	def e(x):
		if isinstance(x[1], tuple):
			return x[0].replace("@", ""), (int(abs(x[1][0])), x[1][1])
		return x[0].replace("@", ""), x[1]
	def f(x):
		return x[0], exp(-x[1])
	def maxitem(d):
		return max(d.iteritems(), key=itemgetter(1))
	trees = [Tree.parse(t, parse_leaf=int) for t in
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
		(ROOT (A 0) (C (B 1) (C 2)))""".splitlines()]
	sents = [a.split() for a in
		"""d b c\n c a b\n a e f\n a e f\n a e f\n a e f\n d b f\n d b f
		d b f\n d b g\n e f c\n e f c\n e f c\n e f c\n e f c\n e f c\n f b c
		a d e""".splitlines()]
	grammar = Grammar(dopreduction(trees, sents))
	shortest, secondarymodel = dopreduction(trees, sents, shortestderiv=True)
	shortest = Grammar(shortest)
	sent = "a b c".split()
	chart, start, _ = parse(sent, grammar, None, grammar.toid['ROOT'], True)
	vit = viterbiderivation(chart, start, grammar.tolabel)
	mpd, _ = marginalize("mpd", chart, start, grammar, 1000)
	mpp, _ = marginalize("mpp", chart, start, grammar, 1000)
	mppsampled, _ = marginalize("mpp", chart, start, grammar, 1000,
			sample=True, kbest=False)
	sldop1, _ = marginalize("sl-dop", chart, start, grammar, 1000,
			sldop_n=7, sent=sent, secondarymodel=shortest)
	sldopsimple, _ = marginalize("sl-dop-simple", chart, start, grammar, 1000,
			sldop_n=7, sent=sent)
	short, _ = marginalize("shortest", chart, start, shortest,
		1000, sent=sent, secondarymodel=secondarymodel)
	print
	print "vit:\t\t%s %r" % e((removeids.sub("", vit[0]), exp(-vit[1])))
	print "MPD:\t\t%s %r" % e(maxitem(mpd))
	print "MPP:\t\t%s %r" % e(maxitem(mpp))
	print "MPP sampled:\t%s %r" % e(maxitem(mppsampled))
	print "SL-DOP n=7:\t%s %r" % e(maxitem(sldop1))
	print "simple SL-DOP:\t%s %r" % e(maxitem(sldopsimple))
	print "shortest:\t%s %r" % f(min(short.iteritems()))

if __name__ == '__main__':
	main()
