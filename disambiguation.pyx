import re, logging
from heapq import nlargest
from math import fsum, exp, log
from random import random
from bisect import bisect_right
from operator import itemgetter
from itertools import count
from collections import defaultdict
from nltk import Tree
from kbest import lazykbest
from parser import parse
from grammar import induce_plcfrs, canonicalize, rangeheads
from treetransforms import unbinarize
from containers cimport ChartItem, Edge, Grammar
cimport cython

infinity = float('infinity')
removeids = re.compile("@[-0-9]+")
termsre = re.compile(r" ([0-9]+)\b")

cpdef marginalize(dict chart, ChartItem start, dict tolabel, int n=10,
	bint sample=False, bint both=False, bint shortest=False,
	secondarymodel=None, bint mpd=False, dict backtransform=None):
	""" approximate MPP or MPD by summing over n random/best derivations from
	chart, return a dictionary mapping parsetrees to probabilities """
	cdef Edge edge
	cdef dict parsetrees = {}
	cdef str treestr, deriv, debin = None
	cdef double prob, maxprob
	cdef int m
	derivations = set()
	if sample or both:
		assert backtransform is None, "not implemented for double dop."
		derivations = getsamples(chart, start, n, tolabel)
	if not sample or both:
		if backtransform is not None: debin = "}<"
		derivations.update(lazykbest(chart, start, n, tolabel, debin))
	if shortest:
		# filter out all derivations which are not shortest
		maxprob = min(derivations, key=itemgetter(1))[1]
		derivations = [(a,b) for a, b in derivations if b == maxprob]
	for deriv, prob in derivations:
		if shortest:
			# calculate the derivation probability in a different model.
			# because we don't keep track of which rules have been used,
			# read off the rules from the derivation ...
			tree = Tree.parse(removeids.sub("", deriv), parse_leaf=int)
			sent = sorted(tree.leaves())
			rules = induce_plcfrs([tree], [sent], arity_marks=False)
			prob = -fsum([secondarymodel[r] for r, w in rules
											if r[0][1] != 'Epsilon'])
			del tree

		if backtransform is None: #DOP reduction
			treestr = removeids.sub("@" if mpd else "", deriv)
		else: # double dop
			try: treestr = recoverfragments_strstr(deriv, backtransform)
			except KeyError:
				print 'deriv', deriv
				raise
			#tree = Tree.parse(deriv, parse_leaf=int)
			##treestr = recoverfragments_str(canonicalize(tree), backtransform)
			#treestr = recoverfragments_str(tree, backtransform)
			assert "#" not in treestr, ("unrecovered ambiguous fragment in tree."
					" deriv: %s\ntree:%s" % (deriv, treestr))
			#FIXME: avoidable?
			#treestr = canonicalize(unbinarize(
			#	Tree.parse(treestr, parse_leaf=int))).pprint(margin=9999)
			#del tree

		# simple way of adding probabilities (too easy):
		#parsetrees[treestr] += exp(-prob)
		if treestr in parsetrees: parsetrees[treestr].append(-prob)
		else: parsetrees[treestr] = [-prob]
	# Adding probabilities in log space
	# http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point-numbers
	# https://facwiki.cs.byu.edu/nlp/index.php/Log_Domain_Computations
	for parsetree in parsetrees:
		maxprob = max(parsetrees[parsetree])
		parsetrees[parsetree] = exp(fsum([maxprob, log(fsum([exp(prob - maxprob)
									for prob in parsetrees[parsetree]]))]))
	msg = "%d derivations, %d parsetrees" % (
		len(derivations), len(parsetrees))
	return parsetrees, msg

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
	whitelist = [{} for a in secondarymodel.toid]
	for a in chart:
		whitelist[(<ChartItem>a).label][(<ChartItem>a).vec] = infinity
	for tt in nlargest(sldop_n, mpp1, key=lambda t: mpp1[t]):
		for n in Tree(tt).subtrees():
			vec = sum([1L << int(x) for x in n.leaves()])
			whitelist[secondarymodel.toid[n.node]][vec] = 0.0
	for label, n in secondarymodel.toid.items():
		whitelist[n] = whitelist[secondarymodel.toid[label.split("@")[0]]]
	mpp2 = {}
	for tt in nlargest(sldop_n, mpp1, key=lambda t: mpp1[t]):
		mpp2[tt] = mpp1[tt]

	words = [a[0] for a in sent]
	tagsornil = [a[1] for a in sent] if tags else []
	chart2, start2, _ = parse(words, secondarymodel, tagsornil,
					1, #secondarymodel.toid[top],
					True, None, whitelist=whitelist)
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
	msg = "(%d derivations, %d of %d parsetrees)" % (
		len(derivations), min(sldop_n, len(mpp1)), len(mpp1))
	return dict(mpp), msg

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
	msg = "(%d derivations, %d of %d parsetrees)" % (
			len(derivations), len(mpp), len(mpp1))
	return dict(mpp), msg

cdef inline double sumderivs(ts, derivations):
	#return fsum([exp(-derivations[t]) for t in ts])
	cdef double result = 0.0
	for t in ts: result += exp(-derivations[t])
	return result
	#return sum([exp(-derivations[t]) for t in ts])

cdef inline int minunaddressed(tt, idsremoved):
	cdef int result = 0
	for t in idsremoved[tt]: result += t.count("(") - t.count("@")
	return result
	#return min([(t.count("(") - t.count("@")) for t in idsremoved[tt]])

cpdef samplechart(dict chart, ChartItem start, dict tolabel, dict tables):
	""" Samples a derivation from a chart. """
	#NB: this does not sample properly, as it ignores the distribution of
	#probabilities and samples uniformly instead. 
	#edge = choice(chart[start])
	cdef Edge edge
	cdef ChartItem child
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

cpdef viterbiderivation(dict chart, ChartItem start, dict tolabel):
	cdef Edge edge = min(chart[start])
	return getviterbi(chart, start, tolabel), edge.inside

cdef getviterbi(dict chart, ChartItem start, dict tolabel):
	cdef Edge edge = min(chart[start])
	if edge.right.label: #binary
		return "(%s %s %s)" % (tolabel[start.label],
					getviterbi(chart, edge.left, tolabel),
					getviterbi(chart, edge.right, tolabel))
	else: #unary or terminal
		return "(%s %s)" % (tolabel[start.label],
					getviterbi(chart, edge.left, tolabel) if edge.left.label
									else str(edge.left.vec))

def renumber(tree):
	leaves = tree.leaves()
	leafmap = dict(zip(sorted(leaves), count()))
	for a in tree.treepositions('leaves'):
		tree[a] = leafmap[tree[a]]
	return tree.freeze(), dict(zip(count(), leaves))
def topproduction(tree):
	return Tree(tree.node,
		[Tree(a.node, rangeheads(sorted(a.leaves())))
				if isinstance(a, Tree) else a for a in tree])
def frontierorterm(tree):
	return not isinstance(tree[0], Tree)
cpdef recoverfragments(derivation, backtransform):
	""" Reconstruct a DOP derivation from a double DOP derivation with
	flattened fragments. Returns expanded derivation as Tree object. """
	# handle ambiguous fragments with nodes of the form "#n"
	if (len(derivation) == 1 and isinstance(derivation[0], Tree)
		and derivation[0].node[0] == "#"):
		derivation = derivation[0]
	prod = topproduction(derivation)
	rprod, leafmap = renumber(prod)
	# fetch the actual fragment corresponding to this production
	#if rprod not in backtransform: print "not found", rprod
	result = Tree.convert(backtransform.get(rprod, rprod))
	# revert renumbering
	for a in result.treepositions('leaves'):
		result[a] = leafmap.get(result[a], result[a])
	# recurse on non-terminals of derivation
	for r, t in zip(result.subtrees(frontierorterm), derivation):
		if isinstance(r, Tree) and isinstance(t, Tree):
			new = recoverfragments(t, backtransform)
			r[:] = new[:]
	return result

def repl(d):
	def f(x): return " %d" % d[int(x.group(1))]
	return f
cpdef recoverfragments_str(derivation, dict backtransform):
	""" Reconstruct a DOP derivation from a DOP derivation with
	flattened fragments. "derivation" should be a Tree object, while
	backtransform should contain strings as keys and values.
	Returns expanded derivation as a string. """
	cdef list leaves
	cdef dict leafmap
	cdef str prod, rprod, result, frontier, replacement
	# handle ambiguous fragments with nodes of the form "#n"
	if (len(derivation) == 1 and isinstance(derivation[0], Tree)
		and derivation[0].node[0] == "#"):
		derivation = derivation[0]
	#topproduction_str(derivation):
	if isinstance(derivation[0], Tree):
		children = ["(%s %s)" % (a.node, " ".join(map(str,
			rangeheads(sorted(a.leaves()))))) for a in derivation]
		prod = "(%s %s)" % (derivation.node, " ".join(children))
	else:
		prod = str(derivation)
	#renumber(prod):
	leaves = map(int, termsre.findall(prod))
	leafmap = dict(zip(sorted(leaves), count()))
	rprod = termsre.sub(repl(leafmap), prod)
	leafmap = dict(zip(leafmap.values(), leafmap.keys()))
	# fetch the actual fragment corresponding to this production
	result = backtransform.get(rprod, rprod)
	# revert renumbering
	result = termsre.sub(repl(leafmap), result)
	# recursively expand all substitution sites
	for t in derivation:
		if isinstance(t, Tree):
			if not isinstance(t[0], Tree): continue
			frontier = "(%s %s)" % (t.node,
				" ".join(map(str, rangeheads(sorted(t.leaves())))))
			assert frontier in result, (frontier, result)
			replacement = recoverfragments_str(t, backtransform)
			result = result.replace(frontier, replacement, 1)
	return result

cpdef recoverfragments_strstr(derivation, dict backtransform):
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
	n = derivation.index(" ")+1
	if derivation[n:].startswith("(#"):
		derivation = derivation[n:len(derivation)-1]

	# tokenize tree structure into direct children
	for n, a in enumerate(derivation):
		if a == '(':
			if parens == 1: start = n
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
	else: prod = derivation

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
		else: del leafmap[a]
		prev = a

	rprod = termsre.sub(repl(leafmap), prod)
	leafmap = dict(zip(leafmap.values(), leafmap.keys()))
	# fetch the actual fragment corresponding to this production
	try: result = backtransform[rprod] #, rprod)
	except KeyError:
		#print 'backtransform'
		#for a in backtransform: print a
		print '\nleafmap', leafmap
		print prod
		print derivation
		print zip(childlabels, childheads)
		raise

	# revert renumbering
	result = termsre.sub(repl(leafmap), result)
	# recursively expand all substitution sites
	for t, theads in zip(children, childheads):
		if t.startswith("(") and "(" in t[1:]: # redundant?
			frontier = "%s %s)" % (t[:t.index(" ")], theads)
			assert frontier in result, (frontier, result)
			replacement = recoverfragments_strstr(t, backtransform)
			result = result.replace(frontier, replacement, 1)
	assert result.count("(") == result.count(")")
	return result

def main():
	from nltk import Tree
	from grammar import dop_lcfrs_rules
	from containers import Grammar
	from parser import parse
	def e(x):
		if isinstance(x[1], tuple):
			return x[0].replace("@", ""), (int(abs(x[1][0])), x[1][1])
		return x[0].replace("@", ""), x[1]
	def f(x):
		return x[0], exp(-x[1])
	def maxitem(d): return max(d.iteritems(), key=itemgetter(1))
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
		"""d b c\n c a b\n a e f\n a e f\n a e f\n a e f\n d b f\n d b f
		d b f\n d b g\n e f c\n e f c\n e f c\n e f c\n e f c\n e f c\n f b c
		a d e""".splitlines()]
	grammar = Grammar(dop_lcfrs_rules(trees, sents))
	shortest, secondarymodel = dop_lcfrs_rules(trees, sents, shortestderiv=True)
	shortest = Grammar(shortest)
	sent = "a b c".split()
	chart, start, _ = parse(sent, grammar, None, grammar.toid['ROOT'], True)
	vit = viterbiderivation(chart, start, grammar.tolabel)
	mpd, _ = marginalize(chart, start, grammar.tolabel, n=1000, mpd=True)
	mpp, _ = marginalize(chart, start, grammar.tolabel, n=1000)
	mppsampled, _ = marginalize(chart, start, grammar.tolabel, n=1000, sample=True)
	sldopsimple, _ = sldop_simple(chart, start, grammar, 1000, 7)
	sldop1, _ = sldop(chart, start, sent, None, grammar, shortest, 1000, 7,
		sample=False, both=False)
	short, _ = marginalize(chart, start, shortest.tolabel, 1000, shortest=True,
		secondarymodel=secondarymodel)
	print
	print "vit:\t\t%s %r" % e((removeids.sub("", vit[0]), exp(-vit[1])))
	print "MPD:\t\t%s %r" % e(maxitem(mpd))
	print "MPP:\t\t%s %r" % e(maxitem(mpp))
	print "MPP sampled:\t%s %r" % e(maxitem(mppsampled))
	print "SL-DOP n=7:\t%s %r" % e(maxitem(sldop1))
	print "simple SL-DOP:\t%s %r" % e(maxitem(sldopsimple))
	print "shortest:\t%s %r" % f(min(short.iteritems()))

if __name__ == '__main__': main()
