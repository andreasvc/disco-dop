"""Disambiguate parse forests with various methods for parse selection.

Use as follows:

>>> getderivations(chart, 1000)  # doctest: +SKIP
>>> parses, msg = marginalize('mpp', chart)  # doctest: +SKIP
"""

from __future__ import print_function
import re
from heapq import nlargest
from math import exp, log, isinf, fsum
from random import random
from bisect import bisect_right
from operator import itemgetter, attrgetter
from itertools import count
from functools import partial
from collections import defaultdict
from . import plcfrs, _fragments
from .tree import Tree, ParentedTree, ImmutableTree, writediscbrackettree, \
		brackettree
from .kbest import lazykbest
from .kbest cimport getderiv
from .grammar import lcfrsproductions, spinal, REMOVEDEC
from .treetransforms import addbitsets, unbinarize, canonicalize, \
		collapseunary, mergediscnodes, binarize
from .bit import pyintnextset, pyintbitcount

cimport cython
from cython.operator cimport dereference
from libc.string cimport memset
from libc.stdint cimport uint64_t
from libc.math cimport HUGE_VAL as INFINITY
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from .bit cimport abitcount
from .containers cimport Prob, Grammar, ProbRule, LexicalRule, Chart, \
		SmallChartItem, FatChartItem, Edge, RankedEdge, Whitelist, Label, \
		ItemNo, sparse_hash_map, logprobadd, logprobsum, yieldranges


cdef extern from "macros.h":
	uint64_t TESTBIT(uint64_t a[], int b)
	void SETBIT(uint64_t a[], int b)
	void CLEARBIT(uint64_t a[], int b)
	int BITNSLOTS(int nb)

include "constants.pxi"

REMOVEIDS = re.compile('@[-0-9]+')
REMOVEWORDTAGS = re.compile('@[^ )]+')
# NB: similar to the one in _grammar.pxi, but used to match labels within parse
# trees instead of individual labels
REMOVESTATESPLITS = re.compile(
		r'(\([^-/^|:\s()]+?'
		r'(?:-[^/^|:\s()]+)?'
		r'(?:/[^^|:\s()]+)?)'
		r'(?:\^[^|:\s()]+)? '
		)

cdef str NONCONSTLABEL = ''
cdef str NEGATIVECONSTLABEL = '-#-'

cpdef getderivations(Chart chart, int k, derivstrings=True):
	"""Get *k*-best derivations from chart.

	:param k: number of derivations to extract from chart
	:param derivstrings: whether to create derivations as strings
	:returns: ``None``. Modifies ``chart.derivations`` and
		``chart.rankededges`` in-place.

		:chart.derivations: list of tuples ``(deriv, logprob)`` where ``deriv``
			is a string; or list of ``None`` if ``derivstrings==False``
		:chart.rankededges[chart.root()]: corresponding list of RankedEdge
			objects for the derivations in ``chart.derivations``.
	"""
	chart.rankededges.clear()
	chart.derivations = lazykbest(chart, k, derivs=derivstrings)


cpdef marginalize(method, Chart chart, list sent=None, list tags=None,
		int k=1000, int sldop_n=7, double mcplambda=1.0, set mcplabels=None,
		bint ostag=False, set require=None, set block=None):
	"""Take a list of derivations and optimizes a given objective function.

	1. Rewrites derivations into the intended parse trees.
	2. Marginalizes (sum / combine) weights of derivations for same parse tree.
	3. Applies objective function (scoring criterion) for ranking parse trees.

	Dependending on the value of chart.grammar.backtransform, one of two ways
	is employed to turn derivations into parse trees:

		:``None``: assume derivations are from DOP reduction and strip
			annotations of the form ``@123`` from labels. This option is also
			applicable when the grammar is not a TSG and derivations already
			coincide with parse trees.
		:list of strings: assume Double-DOP and recover derivations by
			substituting fragments in this list for flattened rules in
			derivations.

	:param method:
		Available objective functions:

		:'mpp': Most Probable Parse
		:'mpd': Most Probable Derivation
		:'shortest': Most Probable Shortest Derivation
		:'sl-dop': Simplicity-Likelihood DOP; select most likely parse from the
			``sldop_n`` parse trees with the shortest derivations.
		:'sl-dop-simple': Approximation of Simplicity-Likelihood DOP
	:param k: when ``method='sl-dop``, number of derivations to consider.
	:param require: optionally, a list of tuples ``(label, indices)``; only
		parse trees containing these labeled spans will be kept.
		For example. ``('NP', [0, 1, 2])``.
	:param block: optionally, a list of tuples ``(label, indices)``;
		parse trees with these labeled spans will be pruned.
	:returns:
		``(parses, msg)``.

		:parses: a list of tuples ``(parsetree, probability, fragments)``
			where ``parsetree`` is a string, ``probability`` is a float
			(0 < p <= 1), or a tuple ``(numfrags, p)`` for shortest derivation
			parsing, and ``fragments`` is a list of fragments in the most
			probable derivation for this parse.
			NB: the list is in an arbitrary order.
		:msg: a message reporting the number of derivations / parses.
	"""
	cdef list backtransform = chart.grammar.backtransform
	cdef bint mpd = method == 'mpd'
	cdef bint shortest = method == 'shortest'
	cdef bint dopreduction = backtransform is None
	cdef pair[RankedEdge, Prob] entry
	cdef vector[pair[RankedEdge, Prob]] entries
	cdef sparse_hash_map[string, vector[Prob]] mpptrees
	cdef dict mpdtrees = {}
	cdef dict derivlen = {}  # parsetree => (derivlen, derivprob)
	cdef dict derivs = {}
	cdef str treestr, deriv
	cdef Prob prob, maxprob
	cdef size_t n
	cdef ItemNo root = chart.root()

	if method == 'sl-dop':
		return sldop(chart, sent, tags, k, sldop_n)
	elif method == 'sl-dop-simple':
		return sldop_simple(sldop_n, chart)
	elif method == 'mcp':
		return maxconstituentsparse(chart, mcplambda, mcplabels)
	elif method == 'shortest':
		# filter out all derivations which are not shortest
		if not dopreduction:
			maxprob = INFINITY
			for entry in chart.rankededges[root]:
				if entry.second < maxprob:
					maxprob = entry.second
			for entry in chart.rankededges[root]:
				if entry.second == maxprob:
					entries.push_back(entry)
			chart.rankededges[root] = entries
		elif chart.derivations:
			_, maxprob = min(chart.derivations, key=itemgetter(1))
			chart.derivations = [(deriv, prob)
					for deriv, prob in chart.derivations
					if prob == maxprob]

	if not dopreduction:  # Double-DOP
		for n in range(chart.rankededges[root].size()):
			entry = chart.rankededges[root][n]
			prob = entry.second
			try:
				treestr = recoverfragments(
						root, entry.first, chart, backtransform)
			except:
				continue
				# print(getderiv(
				# 		root, entry.first, chart).decode('utf8'))
				# raise
			if shortest:
				# for purposes of tie breaking, calculate the derivation
				# probability in a different model.
				newprob = exp(-getderivprob(
						root, entry.first, chart, sent))
				score = (int(prob / log(0.5)), newprob)
				if treestr not in derivlen or score > derivlen[treestr]:
					derivlen[treestr] = score
					derivs[treestr] = n
			elif mpd:
				if (treestr not in mpdtrees
							or -prob > mpdtrees[treestr]):
					mpdtrees[treestr] = -prob
					derivs[treestr] = n
			else:
				mpptrees[<string>treestr.encode('utf8')].push_back(-prob)
				if treestr not in derivs:
					derivs[treestr] = n
	else:  # DOP reduction
		for n, (deriv, prob) in enumerate(chart.derivations):
			if ostag:
				deriv = removeadjunaries(deriv)
				treestr = REMOVEDEC.sub('@1' if mpd or shortest else '', deriv)
			elif dopreduction:
				treestr = REMOVEIDS.sub('@1' if mpd or shortest else '', deriv)
			else:
				raise ValueError
			if shortest:
				entry = chart.rankededges[root][n]
				newprob = getderivprob(root, entry.first, chart, sent)
				score = (int(prob / log(0.5)), exp(-newprob))
				if treestr not in derivlen or (not dopreduction
						and score > derivlen[treestr]):
					derivlen[treestr] = score
					derivs[treestr] = deriv
				elif dopreduction:
					oldscore = derivlen[treestr]
					derivlen[treestr] = oldscore[0], oldscore[1] + score[1]
			elif mpd:
				if (treestr not in mpdtrees or
						-prob > mpdtrees[treestr]):
					mpdtrees[treestr] = -prob
					derivs[treestr] = deriv
			else:
				mpptrees[<string>treestr.encode('utf8')].push_back(-prob)
				if treestr not in derivs:
					derivs[treestr] = deriv

	if ostag:
		results = []
		for it in mpptrees:
			treestr = REMOVEDEC.sub('', it.first.decode('utf8'))
			probs = it.second
			results.append((treestr, logprobsum(probs),
					ostagderivation(derivs[treestr], chart.sent)))
	elif shortest:
		if dopreduction:
			results = [(REMOVEIDS.sub('', treestr), (-a, b),
					fragmentsinderiv_str(derivs[treestr], chart, backtransform))
					for treestr, (a, b) in derivlen.items()]
		else:
			results = [(treestr, (-a, b),
					fragmentsinderiv_re(
						root, chart.rankededges[root][derivs[treestr]].first,
						chart, backtransform))
					for treestr, (a, b) in derivlen.items()]
	elif mpd:
		if dopreduction:
			results = [(REMOVEIDS.sub('', treestr), exp(prob),
					fragmentsinderiv_str(derivs[treestr], chart, backtransform))
					for treestr, prob in mpdtrees.items()]
		else:
			results = [(treestr, exp(prob),
					fragmentsinderiv_re(
						root, chart.rankededges[root][derivs[treestr]].first,
						chart, backtransform))
					for treestr, prob in mpdtrees.items()]
	elif dopreduction:
		results = []
		for it in mpptrees:
			treestr = it.first.decode('utf8')
			probs = it.second
			results.append((treestr, logprobsum(probs),
					fragmentsinderiv_str(derivs[treestr], chart, backtransform)))
	else:
		results = []
		for it in mpptrees:
			treestr = it.first.decode('utf8')
			probs = it.second
			results.append((treestr, logprobsum(probs),
					fragmentsinderiv_re(
						root, chart.rankededges[root][derivs[treestr]].first,
						chart, backtransform)))

	msg = '%d derivations, %d parsetrees' % (
			len(chart.derivations) if dopreduction
				else chart.rankededges[root].size(),
			len(mpdtrees) or len(derivlen) or mpptrees.size())
	if require or block:
		results = [(treestr, score, frags) for treestr, score, frags in results
				if testconstraints(treestr, require, block)]
		msg += '; %d parsetrees match constraints' % len(results)
	return results, msg


def testconstraints(treestr, require, block):
	"""Test whether tree satisfies constraints of required/blocked sets of
	labeled spans."""
	spans = {(node.label, tuple(node.leaves()))
			for node in ImmutableTree(
				REMOVESTATESPLITS.sub(r'\1 ', treestr)).subtrees()}
	return spans.issuperset(require) and spans.isdisjoint(block)


cdef maxconstituentsparse(Chart chart, double labda, set labels=None):
	"""Approximate the Max Constituents Parse (MCP) parse from k-best list.

	Also known as Most Constituents Correct.
	:param chart: the chart, with k-best derivations as strings
	:param labda:
		weight to assign to recall rate vs. the mistake rate;
		1.0 assigns equal weight to both.
	:param labels:
		if given, the set of labels to optimize for;
		by default, all labels are optimized.
	"""
	# Cannot maximize this objective directly from chart because
	# derivations need to be expanded first.
	# NB: this requires derivation strings not entries.
	cdef double sentprob = 0.0, maxscore = 0.0
	cdef double prob, score, maxcombscore, contribution
	cdef short start, spanlen
	cdef object span, leftspan, rightspan, maxleft  # bitsets as Python ints
	cdef list backtransform = chart.grammar.backtransform
	cdef list derivations = []
	cdef ItemNo root = chart.root()
	# FIXME: optimize datastructures
	# table[start][spanlen][span][label] = prob
	table = [[defaultdict(dict)
				for _ in range(len(chart.sent) - n + 1)]
			for n in range(len(chart.sent) + 1)]
	tree = None
	if backtransform is None:
		for deriv, prob in chart.derivations:
			derivations.append((REMOVEIDS.sub('', deriv), prob))
	else:
		for n in range(chart.rankededges[root].size()):
			entry = chart.rankededges[root][n]
			derivations.append((
					recoverfragments(root, entry.first, chart, backtransform),
					entry.second))
	# get marginal probabilities
	for treestr, prob in derivations:
		# Rebinarize because we optimize only for constituents in the tree as
		# it will be evaluated. Collapse unaries because we only select the
		# single best label in each cell.
		tree = addbitsets(binarize(
				collapseunary(
					unbinarize(
						mergediscnodes(unbinarize(
							Tree(treestr),
							childchar=':', expandunary=False)),
						expandunary=False),
					joinchar='+',
					collapsepos=True,
					collapseroot=True)))
		sentprob += exp(-prob)
		for t in tree.subtrees():
			span = t.bitset
			start = pyintnextset(span, 0)
			spanlen = pyintbitcount(span)
			tablecell = table[start][spanlen][span]
			tablecell.setdefault(t.label, 0.0)
			if '|<' not in t.label and (labels is None or t.label in labels):
				tablecell[t.label] += exp(-prob)

	cells = defaultdict(dict)  # cells[span] = (label, score, leftspan)
	# select best derivation
	for spanlen in range(1, len(chart.sent) + 1):
		for start in range(len(chart.sent) - spanlen + 1):
			for span, tablecell in table[start][spanlen].items():
				maxlabel, maxscore = NONCONSTLABEL, 0.0
				if tablecell:
					maxlabel, maxscore = max(tablecell.items(),
							key=itemgetter(1))
				maxcombscore = 0.0
				maxleft = None
				for spans in table[start][1:spanlen]:
					for leftspan in spans:
						if (span & leftspan != leftspan
								or cells[leftspan][0] == NONCONSTLABEL):
							continue
						rightspan = span & ~leftspan
						if (rightspan in cells
								and cells[rightspan][0] != NONCONSTLABEL):
							score = cells[leftspan][1] + cells[rightspan][1]
							if score > maxcombscore:
								maxcombscore = score
								maxleft = leftspan
				score = maxscore / sentprob
				contribution = score - labda * (1 - score)
				if contribution < 0:
					cells[span] = (NEGATIVECONSTLABEL
							if 1 < pyintbitcount(span) < len(chart.sent)
							else maxlabel,
							maxcombscore, maxleft)
				else:
					maxcombscore += contribution
					cells[span] = (maxlabel, maxcombscore, maxleft)

	# reconstruct tree
	tmp = ''
	try:
		tmp = gettree(cells, tree.bitset)
		result = unbinarize(Tree(tmp),
				childchar='NONE', unarychar='+', expandunary=True)
	except (ValueError, AttributeError):
		return [], 'MCP failed. %s' % tmp
	else:
		return [(str(result), maxscore,
				None)], '%d derivations; sentprob: %g' % (
				len(chart.derivations), sentprob)


def gettree(cells, span):
	"""Extract parse tree from most constituents correct table."""
	if span not in cells:
		raise ValueError('MCP: span not in cell: %r' % bin(span))
	label, unused_score, leftspan = cells[span]
	if leftspan not in cells:
		return '(%s %d)' % (label, pyintnextset(span, 0))
	rightspan = span & ~leftspan
	if label == NONCONSTLABEL or label == NEGATIVECONSTLABEL:
		return '%s %s' % (gettree(cells, leftspan), gettree(cells, rightspan))
	return '(%s %s %s)' % (label,
			gettree(cells, leftspan), gettree(cells, rightspan))


cdef sldop(Chart chart, list sent, list tags, int m, int sldop_n):
	"""'Proper' method for sl-dop.

	Parses sentence once more to find shortest derivations, pruning away any
	chart item not occurring in the *n* most probable parse trees; we need to
	parse again because we have to consider all derivations for the *n* most
	likely trees.

	:returns: the intersection of the most probable parse trees and their
		shortest derivations, with probabilities of the form (subtrees, prob).

	NB: doesn't seem to work so well, so may contain a subtle bug.
		Does not support PCFG charts."""
	cdef dict derivations = {}
	cdef dict derivs = {}
	cdef list backtransform = chart.grammar.backtransform
	cdef pair[RankedEdge, Prob] entry
	cdef Chart chart2
	cdef int n
	cdef ItemNo root = chart.root()
	# collect derivations for each parse tree
	derivsfortree = defaultdict(set)
	if backtransform is None:
		derivations = dict(chart.derivations)
		for deriv in derivations:
			derivsfortree[REMOVEIDS.sub('', deriv)].add(deriv)
	else:
		for entry in chart.rankededges[root]:
			deriv = <bytes>getderiv(root, entry.first, chart).decode('utf8')
			derivations[deriv] = entry.second
			derivsfortree[recoverfragments(
					root, entry.first, chart, backtransform)].add(deriv)
	# sum over probs of derivations to get probs of parse trees
	parsetreeprob = {tree: logprobsum([-derivations[d] for d in ds])
			for tree, ds in derivsfortree.items()}
	nmostlikelytrees = set(nlargest(sldop_n, parsetreeprob,
			key=parsetreeprob.get))

	model = chart.grammar.currentmodel
	chart.grammar.switch(u'shortest', logprob=True)
	shortestderivations, msg, chart2 = treeparsing(
			nmostlikelytrees, sent, chart.grammar, m, backtransform, tags)
	if not chart2:
		return [], 'SL-DOP couldn\'t find parse for tree'
	result = {}
	root = chart2.root()
	for n, (deriv, s) in enumerate(shortestderivations):
		entry = chart2.rankededges[root][n]
		if backtransform is None:
			treestr = REMOVEIDS.sub('', deriv)
		else:
			treestr = recoverfragments(
					root, entry.first, chart2, backtransform)
		if treestr in nmostlikelytrees and treestr not in result:
			result[treestr] = (-abs(int(s / log(0.5))), parsetreeprob[treestr])
			if backtransform is None:
				derivs[treestr] = fragmentsinderiv_str(
						deriv, chart2, backtransform)
			else:
				derivs[treestr] = fragmentsinderiv_re(
						root, entry.first, chart2, backtransform)
			if len(result) > sldop_n:
				break
	chart.grammar.switch(model, logprob=True)
	if not len(result):
		return [], 'no matching derivation found'
	msg = '(%d derivations, %d of %d parsetrees)' % (
		len(derivations), min(sldop_n, len(parsetreeprob)), len(parsetreeprob))
	return [(tree, result[tree], derivs[tree]) for tree in result], msg


cdef sldop_simple(int sldop_n, Chart chart):
	"""Simple sl-dop method.

	Estimates shortest derivation directly from number of addressed nodes in
	the *k*-best derivations. After selecting the *n* best parse trees, the one
	with the shortest derivation is returned. In other words, selects shortest
	derivation among the list of available derivations, instead of finding the
	shortest among all possible derivations using Viterbi."""
	cdef pair[RankedEdge, Prob] entry
	cdef dict derivations = {}
	cdef dict derivs = {}, keys = {}
	cdef list backtransform = chart.grammar.backtransform
	cdef int n
	cdef ItemNo root = chart.root()
	derivsfortree = defaultdict(set)
	# collect derivations for each parse tree
	if backtransform is None:
		derivations = dict(chart.derivations)
		for deriv in derivations:
			tree = REMOVEIDS.sub('', deriv)
			derivsfortree[tree].add(deriv)
	else:
		for n in range(<signed>chart.rankededges[root].size()):
			entry = chart.rankededges[root][n]
			deriv = <bytes>getderiv(root, entry.first, chart).decode('utf8')
			deriv = str(unbinarize(Tree(deriv), childchar='}'))
			tree = recoverfragments(root, entry.first, chart, backtransform)
			keys[deriv] = n
			derivations[deriv] = entry.second
			derivsfortree[tree].add(deriv)

	# sum over derivations to get parse trees
	parsetreeprob = {tree: logprobsum([-derivations[d] for d in ds])
			for tree, ds in derivsfortree.items()}
	selectedtrees = nlargest(sldop_n, parsetreeprob, key=parsetreeprob.get)

	# the number of fragments used is the number of
	# nodes (open parens), minus the number of interior
	# (addressed) nodes.
	result = {}
	for tree in selectedtrees:
		score, deriv = min([(deriv.count('(') -
				len([a for a in deriv.split() if '@' in a or '}<' in a]),
				deriv)
				for deriv in derivsfortree[tree]])
		result[tree] = (-score, parsetreeprob[tree])
		if backtransform is None:
			derivs[tree] = fragmentsinderiv_str(
					deriv,
					chart, backtransform)
		else:
			derivs[tree] = fragmentsinderiv_re(
					root, chart.rankededges[root][keys[deriv]].first,
					chart, backtransform)
	msg = '(%d derivations, %d of %d parsetrees)' % (
			len(derivations), len(result), len(parsetreeprob))
	return [(tree, result[tree], derivs[tree]) for tree in result], msg


cdef str recoverfragments(ItemNo root, RankedEdge deriv, Chart chart,
		list backtransform):
	"""Reconstruct a DOP derivation from a derivation with flattened fragments.

	:param deriv: a RankedEdge representing a derivation.
	:param backtransform: a list with fragments (as string templates)
		corresponding to grammar rules.
	:returns: expanded derivation as a string.

	The flattened fragments in the derivation should be left-binarized.

	Does on-the-fly debinarization following labels that are not mapped to a
	label in the coarse grammar, i.e., it assumes that neverblockre is only
	used to avoid blocking nonterminals from the double-dop binarization
	(containing the string '}<'). Note that this means getmapping() has to have
	been called on `chart.grammar`, even when not doing coarse-to-fine
	parsing."""
	if deriv.edge.rule is NULL:
		return '(%s %d)' % (
				chart.grammar.tolabel[chart.label(root)],
				chart.lexidx(deriv.edge))
	else:
		result = recoverfragments_(root, deriv, chart, backtransform)
	return REMOVEWORDTAGS.sub('', result)


cdef str recoverfragments_(ItemNo v, RankedEdge deriv, Chart chart,
		list backtransform):
	cdef RankedEdge child
	cdef list children = []
	cdef vector[ItemNo] childitems
	cdef vector[int] childranks
	cdef int n
	cdef str frag = backtransform[deriv.edge.rule.no]  # template
	ruleno = deriv.edge.rule.no  # FIXME only for debugging

	# collect all children w/on the fly left-factored debinarization
	if deriv.edge.rule.rhs2:  # is there a right child?
		# keep going while left child is part of same binarized constituent
		# instead of looking for a binarization marker in the label string, we
		# use the fact that such labels do not have a mapping as proxy.
		while chart.grammar.selfmapping[deriv.edge.rule.rhs1] == 0:
			# one of the right children
			childitems.push_back(chart.right(v, deriv))
			childranks.push_back(deriv.right)
			# move on to next node in this binarized constituent
			v = chart.left(v, deriv)
			deriv = chart.rankededges[v][deriv.left].first
		# last right child
		if deriv.edge.rule.rhs2:  # is there a right child?
			childitems.push_back(chart.right(v, deriv))
			childranks.push_back(deriv.right)
	elif chart.grammar.selfmapping[deriv.edge.rule.rhs1] == 0:
		v = chart.left(v, deriv)
		deriv = chart.rankededges[v][deriv.left].first
	# left-most child
	childitems.push_back(chart.left(v, deriv))
	childranks.push_back(deriv.left)

	# recursively expand all substitution sites
	children = []
	for n in range(childitems.size() - 1, -1, -1):
		v = childitems[n]
		child = chart.rankededges[v][childranks[n]].first
		if child.edge.rule is NULL:
			children.append('(%s %d)' % (
					chart.grammar.tolabel[chart.label(v)],
					chart.lexidx(child.edge)))
		else:
			children.append(
					recoverfragments_(v, child, chart, backtransform))

	# substitute results in template
	try:  # FIXME for debugging only
		return frag.format(*children)
	except IndexError:
		for n, a in enumerate(chart.grammar.backtransform):
			print(n, chart.grammar.rulestr(chart.grammar.revrulemap[n]),
					'\t', a)
		print(ruleno)
		print(frag)
		print(children)
		raise

	# even better: build result incrementally; use a list of strings
	# extended in recursive calls w/strings from backtransform.
	# step 1: collect RankedEdges in a list (children);
	# 		i.e., exctract nodes from binarized constituent.
	# step 2: iterate over parts of template, alternately adding string from it
	# 		and making a recursive call to insert the relevant child RankedEdge
	# new backtransform format:
	# backtransform[prod] = (list_of_strs, list_of_idx)
	# backtransform[34] = (['(NP (DT ', ') (NN ', '))'], [0, 1])
	# alternatively: (better locality?)
	# frag = backtransform[34] = ['(NP (DT ', 0, ') (NN ', 1, '))']
	# result += frag[0]
	# for n in range(1, len(result), 2):
	# 	foo(result, children[frag[n]])
	# 	result += frag[n + 1]


cdef fragmentsinderiv_re(ItemNo root, RankedEdge deriv, chart,
		list backtransform):
	"""Extract the list of fragments that were used in a given derivation.

	:returns: a list of (fragment, weight) in discbracket format."""
	result = []
	if deriv.edge.rule is not NULL:
		fragmentsinderiv_re_(root, deriv, chart, backtransform, result)
	return [(_fragments.pygetsent(frag), w) for frag, w in result]


def fragmentsinderiv_str(str deriv, chart, list backtransform):
	"""Extract the list of fragments that were used in a given derivation.

	:returns: a list of (fragment, weight) in discbracket format."""
	result = []
	if backtransform is None:
		deriv1 = Tree(deriv)
		result = [(writediscbrackettree(
				REMOVEIDS.sub('', str(splitfrag(node))),
				chart.sent).rstrip(), '')
				for node in deriv1.subtrees(frontiernt)]
	else:
		raise NotImplementedError
	return [(_fragments.pygetsent(frag), w) for frag, w in result]


cdef fragmentsinderiv_re_(ItemNo v, RankedEdge deriv, Chart chart,
		list backtransform, list result):
	cdef RankedEdge child
	cdef vector[ItemNo] childitems
	cdef vector[int] childranks
	cdef Label lhs = deriv.edge.rule.lhs
	cdef double ruleprob
	cdef int ruleno = deriv.edge.rule.no
	cdef str frag = backtransform[ruleno]  # template
	cdef list tmp
	cdef int n
	ruleprob = (exp(-deriv.edge.rule.prob)
			if chart.grammar.logprob else deriv.edge.rule.prob)

	# collect all children w/on the fly left-factored debinarization
	if deriv.edge.rule.rhs2:  # is there a right child?
		# keep going while left child is part of same binarized constituent
		# instead of looking for a binarization marker in the label string, we
		# use the fact that such labels do not have a mapping as proxy.
		while chart.grammar.selfmapping[deriv.edge.rule.rhs1] == 0:
			# one of the right children
			childitems.push_back(chart.right(v, deriv))
			childranks.push_back(deriv.right)
			# move on to next node in this binarized constituent
			v = chart.left(v, deriv)
			deriv = chart.rankededges[v][deriv.left].first
		# last right child
		if deriv.edge.rule.rhs2:  # is there a right child?
			childitems.push_back(chart.right(v, deriv))
			childranks.push_back(deriv.right)
	elif chart.grammar.selfmapping[deriv.edge.rule.rhs1] == 0:
		v = chart.left(v, deriv)
		deriv = chart.rankededges[v][deriv.left].first
	# left-most child
	childitems.push_back(chart.left(v, deriv))
	childranks.push_back(deriv.left)

	tmp = []
	for n in range(childitems.size() - 1, -1, -1):
		v = childitems[n]
		child = chart.rankededges[v][childranks[n]].first
		tmp.append(
				'(%s %s)' % (
				chart.grammar.tolabel[chart.label(v)].split('@')[0],
				('%d=%s' % (chart.lexidx(child.edge),
					chart.sent[chart.lexidx(child.edge)])
					if '@' in chart.grammar.tolabel[chart.label(v)]
					else yieldranges(chart.indices(v)))))
	result.append((
			frag.format(*tmp),
			'rel. freq: %g/%g; weight: %g' % (
				chart.grammar.rulecounts[ruleno],
				chart.grammar.freqmass[lhs],
				ruleprob)))
	# recursively visit all substitution sites
	for n in range(childitems.size() - 1, -1, -1):
		v = childitems[n]
		child = chart.rankededges[v][childranks[n]].first
		if child.edge.rule is not NULL:
			fragmentsinderiv_re_(v, child, chart, backtransform, result)
		elif '@' not in chart.grammar.tolabel[chart.label(v)]:
			try:
				lexruleno = chart.lexruleno(v, child.edge)
				lexcount = chart.grammar.lexcounts[lexruleno]
				lexprob = chart.lexprob(v, child.edge)
			except ValueError:
				lexcount = lexprob = 0
			result.append((
				'(%s %d=%s)' % (
					chart.grammar.tolabel[chart.label(v)],
					chart.lexidx(child.edge),
					chart.sent[chart.lexidx(child.edge)]),
				'rel. freq: %g/%g; weight: %g' % (
					lexcount,
					chart.grammar.freqmass[chart.label(v)],
					lexprob)))


def frontiernt(node):
	"""Test whether node from a DOP derivation is a frontier nonterminal."""
	return '@' not in node.label


def splitfrag(node):
	"""Return a copy of a tree with subtrees labeled without '@' removed."""
	children = []
	for child in node:
		if not isinstance(child, Tree):
			children.append(child)
		elif '@' in child.label:
			children.append(child if isinstance(child[0], int)
					else splitfrag(child))
		else:
			children.append(Tree(child.label,
					['%d:%d' % (min(child.leaves()), max(child.leaves()))]))
	return Tree(node.label, children)


def treeparsing(trees, sent, Grammar grammar, int m, backtransform, tags=None,
		maskrules=True):
	"""Assign probabilities to a sequence of trees with a DOP grammar.

	Given a sequence of trees (as strings), parse them with a DOP grammar
	to get parse tree probabilities; will consider multiple derivations.

	:param maskrules: If True, prune any grammar rule not in the trees.
		If DOP reduction is used, requires selfrulemapping of grammar which
		should map to itself; e.g., 'NP@2 => DT@3 NN' should be mapped
		to 'NP => DT NN' in the same grammar.
	:returns: a tuple ``(derivations, msg, chart)``."""
	# Parsing & pruning inside the disambiguation module is rather kludgy,
	# but the problem is that we need to get probabilities of trees,
	# not just of derivations. Therefore the regular coarse-to-fine methods
	# do not apply directly.
	cdef SmallChartItem item
	cdef FatChartItem fitem
	cdef Whitelist whitelist = Whitelist()
	cdef int n, lensent = len(sent)
	# cdef int selected = 0
	if <unsigned>lensent < sizeof(uint64_t) * 8:
		whitelist.small.resize(grammar.nonterminals)
	else:
		whitelist.fat.resize(grammar.nonterminals)
	assert grammar.selfmapping.size(), ('treeparsing() requires '
			'self mapping; call grammar.getmapping(None, ...)')
	whitelist.mapping = &(grammar.selfmapping[0])
	whitelist.splitmapping = NULL
	if maskrules:
		grammar.setmask([])  # block all rules
	for treestr in trees:
		tree = Tree(treestr)
		for node, (r, yf) in zip(tree.subtrees(),
				lcfrsproductions(tree, sent)):
			leaves = node.leaves()
			try:
				label = grammar.toid[node.label]
			except KeyError:
				return [], "'%s' not in grammar" % node.label, None
			# whitelist based on (label, span)
			if <unsigned>lensent < sizeof(uint64_t) * 8:
				item = SmallChartItem(label, sum([1L << n for n in leaves]))
				whitelist.small[label].insert(item)
			else:
				fitem = FatChartItem(label)
				for n in leaves:
					SETBIT(fitem.vec, n)
				whitelist.fat[label].insert(fitem)
			# whitelist based on grammar rule
			if maskrules and isinstance(node[0], Tree):
				try:
					ruleno = grammar.getruleno(r, yf)
				except KeyError:
					return [], "not in grammar: %r %r" % (r, yf), None
				if grammar.selfrulemapping is None:
					CLEARBIT(&(grammar.mask[0]), ruleno)
				else:
					for n in grammar.selfrulemapping[ruleno]:
						CLEARBIT(&(grammar.mask[0]), n)
						# if TESTBIT(&(grammar.mask[0]), n):
						# 	CLEARBIT(&(grammar.mask[0]), n)
						# 	selected += 1

	# Finally, parse with the small set of allowed labeled spans & rules.
	# Do not parse with the PCFG parser even if possible, because that
	# requires a different way of pruning.
	# FIXME: two pruning mechanisms; want both?
	chart, _ = plcfrs.parse(sent, grammar, tags=tags, whitelist=whitelist)
	if maskrules:
		grammar.setmask(None)

	if not chart:
		return [], 'tree parsing failed', chart
	return lazykbest(chart, m), '', chart


cdef Prob getderivprob(ItemNo v, RankedEdge deriv, Chart chart, list sent):
	"""Recursively calculate probability of a derivation.

	Useful to obtain probability of derivation under different probability
	model of the same grammar."""
	cdef Prob result
	cdef ItemNo v1
	if deriv.edge.rule is NULL:  # is terminal
		return chart.lexprob(v, deriv.edge)
	result = deriv.edge.rule.prob
	v1 = chart.left(v, deriv)
	result += getderivprob(
			v1, chart.rankededges[v1][deriv.left].first,
			chart, sent)
	if deriv.edge.rule.rhs2:
		v1 = chart.right(v, deriv)
		result += getderivprob(
				v1, chart.rankededges[v1][deriv.right].first,
				chart, sent)
	return result

cpdef viterbiderivation(Chart chart):
	"""Wrapper to get Viterbi derivation from chart."""
	# Ask for at least 10 derivations because unary cycles.
	derivations = lazykbest(chart, 10)
	return derivations[0]


def doprerank(parsetrees, sent, k, Grammar coarse, Grammar fine):
	"""Rerank *k*-best coarse trees w/parse probabilities of DOP reduction.

	cf. ``dopparseprob()``."""
	cdef list results = []
	for derivstr, _, _ in nlargest(k, parsetrees, key=itemgetter(1)):
		deriv = addbitsets(derivstr)
		results.append((derivstr, exp(dopparseprob(deriv, sent, coarse, fine)),
				None))
	msg = 're-ranked %d parse trees; best tree at %d. ' % (
			len(results),
			max(range(len(results)), key=lambda x: results[x][1]) + 1)
	return results, msg


def dopparseprob(tree, sent, Grammar coarse, Grammar fine):
	"""Compute the exact DOP parse probability of a Tree in a DOP reduction.

	This follows up on a suggestion made by Goodman (2003, p. 143) of
	calculating DOP probabilities of given parse trees, although I'm not sure
	it has complexity *O(nP)* as he suggests (with *n* as number of nodes in
	input, and *P* as max number of rules consistent with a node in the input).
	Furthermore, the idea of sampling trees "long enough" until we have the MPP
	is no faster than sampling without applying this procedure, because to
	determine that some probability *p* is the maximal probability, we need to
	collect the probability mass *p_seen* of enough parse trees such that we
	have some parsetree with probability *p* > (1 - *p_seen*), which requires
	first seeing almost all parse trees, unless *p* is exceptionally high.
	Hence, this method is mostly useful in a reranking framework where it is
	known in advance that a small set of trees is of interest.

	Expects a mapping which gives a list of consistent rules from the reduction
	as produced by ``fine.getrulemapping(coarse, re.compile('@[-0-9]+$'))``.

	NB: this algorithm could also be used to determine the probability of
	derivations, but then the input would have to distinguish whether nodes are
	internal nodes of fragments, or whether they join two fragments."""
	cdef dict chart = {}  # chart[bitset][label] = prob
	cdef dict cell  # chart[bitset] = cell; cell[label] = prob
	cdef ProbRule *rule
	cdef LexicalRule lexrule
	cdef object n  # pyint
	cdef str pos
	if not fine.logprob:
		raise ValueError('Grammar should have log probabilities.')
	# Log probabilities are not ideal here because we do lots of additions,
	# but the probabilities are very small.
	# A possible alternative is to scale them somehow.

	# add all matching POS tags
	for n, pos in tree.pos():
		word = sent[n]
		chart[1 << n] = cell = {}
		it = fine.lexicalbyword.find(word.encode('utf8'))
		if it == fine.lexicalbyword.end():
			cell[fine.toid[pos]] = -0.0
			continue
		for lexruleno in dereference(it).second:
			lexrule = fine.lexical[lexruleno]
			if (fine.tolabel[lexrule.lhs] == pos
					or fine.tolabel[lexrule.lhs].startswith(pos + '@')):
				cell[lexrule.lhs] = -lexrule.prob

	# do post-order traversal (bottom-up)
	for node, (r, yf) in list(zip(tree.subtrees(),
			lcfrsproductions(tree, sent)))[::-1]:
		if not isinstance(node[0], Tree):
			continue
		if node.bitset not in chart:
			chart[node.bitset] = {}
		cell = chart[node.bitset]
		prod = coarse.getruleno(r, yf)
		if len(node) == 1:  # unary node
			for ruleno in fine.rulemapping[prod]:
				rule = &(fine.bylhs[0][fine.revrulemap[ruleno]])
				if rule.rhs1 in cell:
					if rule.lhs in cell:
						cell[rule.lhs] = logprobadd(cell[rule.lhs],
								-rule.prob + cell[rule.rhs1])
					else:
						cell[rule.lhs] = (-rule.prob + cell[rule.rhs1])
		elif len(node) == 2:  # binary node
			leftcell = chart[node[0].bitset]
			rightcell = chart[node[1].bitset]
			for ruleno in fine.rulemapping[prod]:
				rule = &(fine.bylhs[0][fine.revrulemap[ruleno]])
				if (rule.rhs1 in leftcell and rule.rhs2 in rightcell):
					newprob = (-rule.prob
							+ leftcell[rule.rhs1] + rightcell[rule.rhs2])
					if rule.lhs in cell:
						cell[rule.lhs] = logprobadd(cell[rule.lhs], newprob)
					else:
						cell[rule.lhs] = newprob
		else:
			raise ValueError('expected binary tree without empty nodes.')
	return chart[tree.bitset].get(fine.toid[tree.label], float('-inf'))


def mcrerank(parsetrees, sent, k, trees, vocab):
	"""Rerank *k*-best trees using tree fragments from training treebank.

	Searches for trees that share multiple fragments (multi component)."""
	cdef list results = []
	for derivstr, prob, _ in nlargest(k, parsetrees, key=itemgetter(1)):
		tmp = _fragments.getctrees(
				[(addbitsets(derivstr), sent)], vocab=vocab)
		frags = _fragments.extractfragments(
				tmp['trees1'], 0, 0, vocab, trees,
				disc=True, approx=False)
		frags = {frag: bitset for frag, bitset in frags.items()
				if frag[0].count('(') > 3}
		indices = _fragments.exactcounts(
				list(frags.values()), tmp['trees1'], trees, indices=True)
		score = 0
		rev = defaultdict(set)
		for frag, idx in zip(frags, indices):
			# from: frag => (tree idx, freq)...
			# to: tree idx => frags...
			for i in idx:
				rev[i].add(frag)
		for i in rev:
			if len(rev[i]) > 1:
				# score is the total number of nodes
				# of the common fragments consisting of at least 2 parts
				score += sum(frag[0].count('(') for frag in rev[i])
		# divide by number of nodes in derivation to avoid preferring
		# larger derivations
		score = float(score) / (derivstr.count('(') + len(sent))
		results.append((derivstr, (score, prob), None))
	msg = 're-ranked %d parse trees; best tree at %d. ' % (
			len(results),
			max(range(len(results)), key=lambda x: results[x][1]) + 1)
	return results, msg


def ostagderivation(derivtreestr, sent):
	"""Extract the list of fragments that were used in a given derivation.

	:returns: a list of fragments of the form ``(tree, sent)``"""
	derivtree = ParentedTree(derivtreestr)
	# tmp = [REMOVEDEC.sub('', str(splitostagfrag(node, sent)))
	try:
		tmp = [str(splitostagfrag(node, sent))
				for node in derivtree.subtrees(ostagfrontiernt)]
		return [(_fragments.pygetsent(frag), None) for frag in tmp]
	except:
		return []


def ostagfrontiernt(node):
	"""Test if osTAG derivation node is a substitution/adjunction site."""
	return ('@' not in node.label
			or '[' in node.label and '[' not in node.parent.label)


def splitostagfrag(node, sent):
	"""Return copy of tree after pruning subtrees that are subst/adj sites."""
	children = []
	# FIXME: hopefully this can be simplified...
	for child in node:
		if not isinstance(child, Tree):
			children.append('%d=%s' % (child, sent[child]))
		elif '@' not in child.label or (
				'[' not in child.label and '[' in node.label
				and node.label.endswith(child.label + ']')):
			# this is a substitution site, or a foot node:
			children.append(Tree(child.label,
					['%d:%d' % (min(child.leaves()), max(child.leaves()))]))
		elif '[' in child.label and '[' not in node.label:
			# this is an adjunction site, skip aux tree until foot node:
			while '[' in child.label:
				if (spinal(child[0])
						or child.label.endswith(child[0].label + ']')):
					child = child[0]
				else:
					child = child[1]
			if '@' in child.label:  # add rest of initial tree
				children.extend(
						['%d=%s' % (child[0], sent[child[0]])]
						if isinstance(child[0], int)
						else splitostagfrag(child, sent))
			else:  # this is a substitution site
				children.append(
						'%d:%d' % (min(child.leaves()), max(child.leaves())))
		else:
			children.append(
					ParentedTree(child.label,
						['%d=%s' % (child[0], sent[child[0]])])
					if isinstance(child[0], int)
					else splitostagfrag(child, sent))
	return Tree(node.label, children)


def removeadjunaries(tree):
	"""Remove artificial unary adjunction nodes from osTAG derivation."""
	tree = Tree(tree)
	for node in tree.subtrees(lambda n: '[' not in n.label
			and isinstance(n[0], Tree) and '[' in n[0].label):
		node.label = node[0].label
		node[:] = node[0]
	return str(tree)


def test():
	from .grammar import dopreduction
	from .containers import Grammar
	from . import plcfrs

	def e(x):
		if not x:
			raise ValueError('NO PARSE')
		a, b, _ = max(x, key=itemgetter(1))
		return (a, (int(abs(b[0])), b[1])) if isinstance(b, tuple) else (
				a, b)

	trees = [Tree(t) for t in
		'''(ROOT (A (A 0) (B 1)) (C 2))
		(ROOT (C 0) (A (A 1) (B 2)))
		(ROOT (A 0) (C (B 1) (C 2)))
		(ROOT (A 0) (C (B 1) (C 2)))'''.splitlines()
		+ 14 * ['(ROOT (B (A 0) (B 1)) (C 2))']]
	sents = [a.split() for a in
		'''d b c\n c a b\n a e f\n a e f\n a e f\n a e f\n d b f\n d b f
		d b f\n d b g\n e f c\n e f c\n e f c\n e f c\n e f c\n e f c\n f b c
		a d e'''.splitlines()]
	xgrammar, altweights = dopreduction(trees, sents)
	grammar = Grammar(xgrammar, altweights=altweights)
	grammar.getmapping(None, striplabelre=REMOVEIDS)
	print(grammar)
	sent = 'a b c'.split()
	chart, _ = plcfrs.parse(sent, grammar, None, True)
	assert chart
	vitderiv, vitprob = viterbiderivation(chart)
	getderivations(chart, 1000, derivstrings=True)
	mpd, _ = marginalize('mpd', chart)
	mpp, _ = marginalize('mpp', chart)
	mcp, _ = marginalize('mcp', chart)
	sldop_, _ = marginalize('sl-dop', chart, k=1000,
			sldop_n=7, sent=sent)
	sldopsimple, _ = marginalize('sl-dop-simple',
			chart, k=1000, sldop_n=7, sent=sent)
	short, _ = marginalize('shortest', chart, sent=sent)
	print('\nvit:\t\t%s %r' % (REMOVEIDS.sub('', vitderiv),
			exp(-vitprob)),
		'MPD:\t\t%s %r' % e(mpd), 'MCP:\t\t%s %r' % e(mcp),
		'MPP:\t\t%s %r' % e(mpp),
		'SL-DOP n=7:\t%s %r' % e(sldop_),
		'simple SL-DOP:\t%s %r' % e(sldopsimple),
		'shortest:\t%s %r' % e(short), sep='\n')

__all__ = ['getderivations', 'marginalize', 'gettree', 'treeparsing',
		'viterbiderivation', 'doprerank', 'dopparseprob', 'frontiernt',
		'splitfrag', 'testconstraints']
