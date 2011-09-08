"""Assorted functions to project items from a coarse chart to corresponding
items for a fine grammar.
"""
import re, logging
from collections import defaultdict
from nltk import Tree
from treetransforms import mergediscnodes, un_collinize, fanout
from agenda import Entry
try: dictcast({})
except NameError:
	from containers import * #ChartItem, Edge, dictcast, itemcast, edgecast
	from kbest import * #lazykthbest, lazykbest

infinity = float('infinity')
removeids = re.compile("@[0-9]+")
removeparentannot = re.compile("\^<.*>")
reducemarkov = [re.compile("\|<[^>]*>"),
				re.compile("\|<([^->]*)-?[^>]*>"),
				re.compile("\|<([^->]*-[^->]*)-?[^>]*>"),
				re.compile("\|<([^->]*-[^->]*-[^->]*)-?[^>]*>")]

def prunelist_fromchart(chart, goal, coarsegrammar, finegrammar, k,
		removeparentannotation=False, mergesplitnodes=False, reduceh=999):
	""" Produce a white list of chart items occurring in the k-best derivations
	of chart, where labels X in prunetoid are projected to the labels X and X@n
	in 'toid', for possible values of n.  When k==0, the chart is merely
	filtered to contain only items that contribute to a complete derivation."""

	d = dictcast(defaultdict(dict))
	prunetoid = coarsegrammar.toid
	prunetolabel = coarsegrammar.tolabel
	toid = finegrammar.toid
	tolabel = finegrammar.tolabel
	prunelist = [None] * len(toid)
	l = [{} for a in prunetoid]
	if mergesplitnodes:
		d = merged_kbest(chart, goal, k, coarsegrammar)
	else:
		# construct a table mapping each nonterminal A or A@x
		# to the outside score for A in the chart to prune with
		kbest = kbest_outside(chart, goal, k)
	if removeparentannotation:
		# uses string labels of coarse chart
		for Ih in chart:
			label = removeparentannot.sub("", prunetolabel[itemcast(Ih).label])
			if reduceh != 999:
				label = reducemarkov[reduceh].sub("|<\\1>" if reduceh else "", label)
			d[label][itemcast(Ih).vec] = min(d[label].get(itemcast(Ih).vec, infinity),
										kbest.get(Ih, infinity))
	if removeparentannotation or mergesplitnodes:
		for a, label in toid.iteritems():
			prunelist[label] = d[removeids.sub("", a)]
	else:
		# uses ids of labels in coarse chart
		for Ih in chart:
			l[itemcast(Ih).label][itemcast(Ih).vec] = kbest.get(Ih, infinity)
		for a, label in toid.iteritems():
			prunelist[label] = l[prunetoid.get(removeids.sub("", a), 1)]
	logging.debug('pruning with %d nonterminals, %d items' % (
			len(filter(None, prunelist)), len(chart)))
	return prunelist

def merged_kbest(chart, start, k, grammar):
	""" Like kbest_outside, but apply the reverse of the Boyd (2007)
	transformation to the k-best derivations."""
	#if k == 0: 
	derivs = [Tree.parse(a, parse_leaf=int) for a, b
				in lazykbest(chart, start, k, grammar.tolabel)]
	map(un_collinize, derivs)
	map(mergediscnodes, derivs)
	newchart = dictcast(defaultdict(dict))
	for tree in derivs:
		# no binarization! hence only unbinarized nodes can be pruned.
		for node in tree.subtrees():
			arity = fanout(node)
			if arity > 1: label = "%s_%d" % (node.node, arity)
			else: label = node.node
			newchart[label][sum([1L << n for n in node.leaves()])] = 0.0
	return newchart

def kbest_outside(chart, start, k):
	""" produce a dictionary of ChartItems with the best outside score
	according to the k-best derivations in a chart. """
	D = {}
	outside = { start : 0.0 }
	if k == 0:
		outside = filterchart(chart, start)
		for a in outside:
			e = outside[a]
			outside[a] = e.inside
	else:
		lazykthbest(start, k, k, D, {}, chart, set())
		for entry in D[start]:
			getitems(entry.key, entry.value, D, chart, outside)
	return outside

def getitems(ej, rootprob, D, chart, outside):
	""" Traverse a derivation e,j, noting outside costs relative to its root
	edge """
	e = ej.edge
	if e.left in chart:
		if e.left in D:
			entry = D[e.left][ej.left]
			eejj = entry.key; prob = entry.value
		elif ej.left == 0:
			eejj = RankedEdge(e.left, min(chart[e.left]), 0, 0)
			prob = eejj.edge.inside
		else: raise ValueError
		if e.left not in outside:
			outside[e.left] = rootprob - prob
		getitems(eejj, rootprob, D, chart, outside)
	if e.right.label:
		if e.right in D:
			entry = D[e.right][ej.right]
			eejj = entry.key; prob = entry.value
		elif ej.right == 0:
			eejj = RankedEdge(e.right, min(chart[e.right]), 0, 0)
			prob = eejj.edge.inside
		else: raise ValueError
		if e.right not in outside:
			outside[e.right] = rootprob - prob
		getitems(eejj, rootprob, D, chart, outside)

def filterchart(chart, start):
	""" remove all entries that do not contribute to a complete derivation
	headed by "start" """
	chart2 = {}
	filter_subtree(start, dictcast(chart), chart2)
	return chart2

def filter_subtree(start, chart, chart2):
	chart2[start] = chart[start]
	for edge in chart[start]:
		item = edge.left
		if item.label and item not in chart2:
			filter_subtree(edge.left, chart, chart2)
		item = edge.right
		if item.label and item not in chart2:
			filter_subtree(edge.right, chart, chart2)

def main():
	from treetransforms import splitdiscnodes
	from negra import NegraCorpusReader, readheadrules
	from grammar import splitgrammar, induce_srcg, dop_srcg_rules
	headrules = readheadrules()
	k = 50
	corpus = NegraCorpusReader(".", "sample2\.export",
		encoding="iso-8859-1", headorder=True,
		headfinal=True, headreverse=False)
	trees = list(corpus.parsed_sents())
	sents = corpus.sents()
	
	dtrees = [t.copy(True) for t in trees]
	cftrees = [splitdiscnodes(t.copy(True)) for t in trees]
	parenttrees = [t.copy(True) for t in trees]
	for t in trees: t.chomsky_normal_form(vertMarkov=0)
	for t in cftrees: t.chomsky_normal_form(vertMarkov=0, horzMarkov=1)
	for t in parenttrees: t.chomsky_normal_form(vertMarkov=2)
	for t in dtrees: t.chomsky_normal_form(vertMarkov=0, horzMarkov=1)

	normal = splitgrammar(induce_srcg(trees, sents))
	parent = splitgrammar(induce_srcg(parenttrees, sents))
	split = splitgrammar(induce_srcg(cftrees, sents))
	fine999 = splitgrammar(dop_srcg_rules(trees, sents))
	fine1 = splitgrammar(dop_srcg_rules(dtrees, sents))
	for msg, coarse, fine, settings in zip(
		"normal parentannot cf-split".split(),
		(normal, parent, split),
		(fine999, fine1, fine1),
		((False, False), (True, False), (False, True))):
		print "coarse grammar:", msg
		for sent, tree in zip(sents, trees):
			doctf(coarse, fine, sent, tree, k,
					1 if msg == "parentannot" else 999, headrules, *settings)

def doctf(coarse, fine, sent, tree, k, doph, headrules, pa, split,
			verbose=False):
	try: from plcfrs import parse, pprint_chart
	except ImportError: from oldplcfrs import parse, pprint_chart
	#from coarsetofine import kbest_outside, merged_kbest, prunelist_fromchart
	from disambiguation import mostprobableparse
	from treetransforms import mergediscnodes
	from containers import getlabel, getvec
	from grammar import canonicalize, rem_marks
	from math import exp
	print " C O A R S E "
	p, start = parse(sent, coarse, start=coarse.toid['ROOT'])
	if start:
		mpp = mostprobableparse(p, start, coarse.tolabel)
		for t in mpp:
			print exp(-mpp[t]),
			t = Tree.parse(t, parse_leaf=int); t.un_chomsky_normal_form()
			if split: mergediscnodes(t)
			t = canonicalize(rem_marks(t))
			print t.pprint(margin=999)
			print "exact match" if t == canonicalize(tree) else "no match"
	else:
		print "no parse"
		pprint_chart(p, sent, coarse.tolabel)
	l = prunelist_fromchart(p, start, coarse, fine, k,
				removeparentannotation=pa, mergesplitnodes=split,
				reduceh=doph)
	if verbose:
		print "\nitems in 50-best of coarse chart"
		if split:
			d = merged_kbest(p, start, k, coarse)
			print
			for label in d:
				print label, map(bin, d[label].keys())
		else:
			kbest = kbest_outside(p, start, k)
			for a,b in kbest.items():
				print coarse.tolabel[getlabel(a)], bin(getvec(a)), b
		print "\nprunelist:"
		for n,x in enumerate(l):
			print fine.tolabel[n], [(bin(v), s) for v,s in x.items()]
		print
	print " F I N E "
	p, start = parse(sent, fine, start=fine.toid['ROOT'], prunelist=l, neverblockmarkovized=pa or split)
	print
	if start:
		mpp = mostprobableparse(p, start, fine.tolabel)
		for t in mpp:
			print exp(-mpp[t]),
			t = Tree.parse(t, parse_leaf=int); t.un_chomsky_normal_form()
			t = canonicalize(rem_marks(t))
			print t.pprint(margin=999)
			print "exact match" if t == canonicalize(tree) else "no match"
	else:
		print "no parse"
		if verbose: pprint_chart(p, sent, fine.tolabel)
	print

if __name__ == '__main__': main()
