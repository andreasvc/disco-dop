"""Assorted functions to project items from a coarse chart to corresponding
items for a fine grammar.
"""
import re, logging
from collections import defaultdict
from nltk import Tree
from treetransforms import mergediscnodes, unbinarize, slowfanout
import cython
assert cython.compiled

infinity = float('infinity')
removeids = re.compile("@[0-9]+")
removeparentannot = re.compile("\^<.*>")
reducemarkov = [re.compile("\|<[^>]*>"),
				re.compile("\|<([^->]*)-?[^>]*>"),
				re.compile("\|<([^->]*-[^->]*)-?[^>]*>"),
				re.compile("\|<([^->]*-[^->]*-[^->]*)-?[^>]*>")]

def prunechart(chart, goal, coarse, fine, k, reduceh=999,
		removeparentannotation=False, mergesplitnodes=False):
	""" Produce a white list of chart items occurring in the k-best derivations
	of chart, where labels X in coarse.toid are projected to the labels
	X and X@n in fine.toid, for possible values of n.  When k==0, the chart is
	merely filtered to contain only items that contribute to a complete
	derivation."""

	d = dictcast(defaultdict(dict))
	whitelist = [None] * len(fine.toid)
	l = [{} for a in coarse.toid]
	if mergesplitnodes:
		d = merged_kbest(chart, goal, k, coarse)
		assert not removeparentannotation, "not implemented"
	else:
		# construct a list of the k-best nonterminals to prune with
		kbest = kbest_items(chart, goal, k)
	if removeparentannotation:
		# uses string labels of coarse chart
		for Ih in chart:
			label = removeparentannot.sub("", coarse.tolabel[itemcast(Ih).label])
			if reduceh != 999:
				label = reducemarkov[reduceh].sub("|<\\1>" if reduceh else "",
					label)
			d[label][itemcast(Ih).vec] = min(d[label].get(itemcast(Ih).vec,
				infinity), kbest.get(Ih, infinity))
	if removeparentannotation or mergesplitnodes:
		for a, label in fine.toid.iteritems():
			whitelist[label] = d[removeids.sub("", a)]
	else:
		# uses ids of labels in coarse chart
		for Ih in chart:
			l[itemcast(Ih).label][itemcast(Ih).vec] = kbest.get(Ih, infinity)
		for a, label in fine.toid.iteritems():
			whitelist[label] = l[coarse.toid.get(removeids.sub("", a), 1)]
	logging.debug('pruning with %d nonterminals, %d items' % (
			len(filter(None, whitelist)), len(chart)))
	return whitelist

def merged_kbest(chart, start, k, grammar):
	""" Like kbest_items, but apply the reverse of the Boyd (2007)
	transformation to the k-best derivations."""
	derivs = [Tree.parse(a, parse_leaf=int) for a, _
				in lazykbest(chart, start, k, grammar.tolabel)]
	for a in derivs: unbinarize(a, childChar=":", parentChar="!")
	map(mergediscnodes, derivs)
	newchart = dictcast(defaultdict(dict))
	for tree in derivs:
		for node in tree.subtrees():
			arity = slowfanout(node)
			if arity > 1: label = "%s_%d" % (node.node, arity)
			else: label = node.node
			newchart[label][sum([1L << n for n in node.leaves()])] = 0.0
	return newchart

def kbest_items(chart, start, k):
	""" produce a dictionary with ChartItems as keys (value is currently unused)
	according to the k-best derivations in a chart. """
	D = {}
	items = { start : 0.0 }
	if k == 0:
		items = filterchart(chart, start)
		for a in items:
			# use if probabilities matter
			#e = min(items[a])
			#items[a] = e.inside
			items[a] = 0.0
	else:
		lazykthbest(start, k, k, D, {}, chart, set())
		for entry in D[start]:
			getitems(entry.key, entry.value, D, chart, items)
	return items

def getitems(ej, rootprob, D, chart, items):
	""" Traverse a derivation e,j, collecting all items belonging to it, and
	noting (Viterbi) outside costs relative to its root edge (these costs are
	currently unused; only presence or absence in this list is exploited)"""
	e = ej.edge
	if e.left in chart:
		if e.left in D:
			entry = D[e.left][ej.left]
			eejj = entry.key; prob = entry.value
		elif ej.left == 0:
			eejj = RankedEdge(e.left, min(chart[e.left]), 0, 0)
			prob = eejj.edge.inside
		else: raise ValueError
		if e.left not in items:
			items[e.left] = rootprob - prob
		getitems(eejj, rootprob, D, chart, items)
	if e.right.label:
		if e.right in D:
			entry = D[e.right][ej.right]
			eejj = entry.key; prob = entry.value
		elif ej.right == 0:
			eejj = RankedEdge(e.right, min(chart[e.right]), 0, 0)
			prob = eejj.edge.inside
		else: raise ValueError
		if e.right not in items:
			items[e.right] = rootprob - prob
		getitems(eejj, rootprob, D, chart, items)

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

def doctf(coarse, fine, sent, tree, k, doph, pa, split, verbose=False):
	from plcfrs import parse #, pprint_chart
	#from coarsetofine import kbest_items, merged_kbest, prunechart
	from disambiguation import marginalize
	from treetransforms import mergediscnodes, unbinarize
	from containers import getlabel, getvec
	from grammar import canonicalize, rem_marks
	from math import exp
	sent, tags = zip(*sent)
	print " C O A R S E ",
	p, start = parse(sent, coarse, start=coarse.toid['ROOT'], tags=tags)
	if start:
		mpp = marginalize(p, start, coarse.tolabel)
		for t in mpp:
			print exp(-mpp[t]),
			t = Tree.parse(t, parse_leaf=int)
			if split:
				unbinarize(t, childChar=":", parentChar="!")
				mergediscnodes(t)
			unbinarize(t)
			t = canonicalize(rem_marks(t))
			print "exact match" if t == canonicalize(tree) else "no match" #, t
	else:
		print "no parse"
		return
		#pprint_chart(p, sent, coarse.tolabel)
	l = prunechart(p, start, coarse, fine, k, #0 if split else k,
				removeparentannotation=pa, mergesplitnodes=False,
				reduceh=doph)
	if verbose:
		print "\nitems in 50-best of coarse chart"
		if split:
			d = merged_kbest(p, start, k, coarse)
			for label in d:
				print label, map(bin, d[label].keys())
		else:
			kbest = kbest_items(p, start, k)
			for a,b in kbest.items():
				print coarse.tolabel[getlabel(a)], bin(getvec(a)), b
		print "\nwhitelist:"
		for n,x in enumerate(l):
			print fine.tolabel[n], [(bin(v), s) for v,s in x.items()]
	print " F I N E ",
	p = filterchart(p, start) if split else None
	pp, start = parse(sent, fine, start=fine.toid['ROOT'], tags=tags,
		whitelist=l, coarsechart=p, coarsegrammar=coarse,
		neverblocksubstr="|" if pa else None, neverblockdiscontinuous=False,
		splitprune=split, markorigin=True)
	if start:
		mpp = marginalize(pp, start, fine.tolabel)
		for t in mpp:
			print exp(-mpp[t]),
			t = Tree.parse(t, parse_leaf=int)
			unbinarize(t)
			t = canonicalize(rem_marks(t))
			#print t.pprint(margin=999)
			print "exact match" if t == canonicalize(tree) else "no match", t
	else:
		print "no parse. problem."; return
		#if verbose:
		#xp = set((coarse.tolabel[a.label], a.vec) for a in p.keys() if p[a])
		#xpp = set((fine.tolabel[a.label], a.vec) for a in pp.keys() if pp[a])
		#print "difference:"
		#for a in xp - xpp:
		#	if "*" not in a[0]: print a[0], bin(a[1])
		#print "\nfine chart:"
		#for a in xpp:
		#	if "@" not in a[0]: print a[0], bin(a[1])

		#pprint_chart(pp, sent, fine.tolabel)

def main():
	from treetransforms import splitdiscnodes, binarize
	from treebank import NegraCorpusReader
	from grammar import induce_srcg, dop_srcg_rules,\
			subsetgrammar
	from containers import Grammar
	from time import clock
	k = 50
	#corpus = NegraCorpusReader(".", "toytb.export", encoding="iso-8859-1")
	#corpus = NegraCorpusReader("../rparse", "negraproc.export",
	#	encoding="utf-8", headorder=True, headfinal=True, headreverse=False)
	#train = 400; test = 40; testmaxlen = 999;
	corpus = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1",
		headorder=False, headfinal=True, headreverse=False)
	train = 0; test = 3; testmaxlen = 999;
	#trees = corpus.parsed_sents()[:train]
	#sents = corpus.sents()[:train]
	trees = corpus.parsed_sents()
	sents = corpus.sents()

	dtrees = [t.copy(True) for t in trees]
	parenttrees = [t.copy(True) for t in trees]
	for t in trees: binarize(t, vertMarkov=0, horzMarkov=1)
	cftrees = [splitdiscnodes(t.copy(True), markorigin=True) for t in trees]
	for t in cftrees:
		#t.chomsky_normal_form(childChar=":")
		binarize(t, horzMarkov=999, tailMarker='', leftMostUnary=True,
			childChar=":") #NB leftMostUnary is important
	for t in parenttrees: binarize(t, vertMarkov=2)
	for t in dtrees: binarize(t, vertMarkov=0, horzMarkov=1)
	# mark heads, canonicalize, binarize head outward
	normalsrcg = induce_srcg(trees, sents)
	normal = Grammar(normalsrcg)
	parent = Grammar(induce_srcg(parenttrees, sents))
	split = Grammar(induce_srcg(cftrees, sents))
	for t,s in zip(cftrees, sents):
		for (r,yf),w in induce_srcg([t], [s]): assert len(yf) == 1
	fine999srcg = dop_srcg_rules(trees, sents)
	fine999 = Grammar(fine999srcg)
	fine1 = Grammar(dop_srcg_rules(dtrees, sents))
	trees = list(corpus.parsed_sents()[train:train+test])
	sents = corpus.tagged_sents()[train:train+test]
	if subsetgrammar(normalsrcg, fine999srcg):
		print "DOP grammar is a superset"
	else: print "DOP grammar is NOT a superset!"
	for msg, coarse, fine, settings, enabled in zip(
		"normal parentannot cf-split".split(),
		(normal, parent, split),
		(fine999, fine1, fine1),
		((False, False), (True, False), (False, True)),
		(True, False, True)):
		if not enabled: continue
		print "coarse grammar:", msg
		begin = clock()
		for n, (sent, tree) in enumerate(zip(sents, trees)):
			if len(sent) > testmaxlen: continue
			print n,
			doctf(coarse, fine, sent, tree, k, 1 if msg=="parentannot" else 999,
				*settings, verbose=False)
		print "time elapsed", clock() - begin, "s"

if __name__ == '__main__': main()
