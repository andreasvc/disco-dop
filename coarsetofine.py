"""Assorted functions to project items from a coarse chart to corresponding
items for a fine grammar.
"""
import re
from collections import defaultdict
from nltk import Tree
from treetransforms import mergediscnodes, un_collinize, fanout
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
	print 'pruning with %d nonterminals, %d items' % (
			len(filter(None, prunelist)), len(chart))
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
			newchart[label][sum(1L << n for n in node.leaves())] = 0.0
	return newchart

def kbest_outside(chart, start, k):
	""" produce a dictionary of ChartItems with the best outside score
	according to the k-best derivations in a chart. """
	D = {}
	outside = { start : 0.0 }
	if k == 0:
		return filterchart(chart, start)
	lazykthbest(start, k, k, D, {}, chart, set())
	for (e, j), rootedge in D[start]:
		getitems(e, j, rootedge, D, chart, outside)
	return outside

def getitems(e, j, rootedge, D, chart, outside):
	""" Traverse a derivation e,j, noting outside costs relative to its root
	edge """
	if e.left in chart:
		if e.left in D: (ee, jj), ee2 = D[e.left][j[0]]
		elif j[0] == 0: jj = (0, 0); ee = ee2 = min(chart[e.left])
		else: raise ValueError
		if e.left not in outside:
			outside[e.left] = rootedge.inside - ee2.inside
		getitems(ee, jj, rootedge, D, chart, outside)
	if e.right.label:
		if e.right in D: (ee, jj), ee2 = D[e.right][j[1]]
		elif j[1] == 0: jj = (0, 0); ee = ee2 = min(chart[e.right])
		else: raise ValueError
		if e.right not in outside:
			outside[e.right] = rootedge.inside - ee2.inside
		getitems(ee, jj, rootedge, D, chart, outside)

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

