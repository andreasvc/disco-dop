"""Disambiguate parse forests with various methods for parse selection.

Use as follows:

>>> derivs, entries = getderivations(chart, 1000)  # doctest: +SKIP
>>> parses, msg = marginalize('mpp', derivs, entries, chart)  # doctest: +SKIP
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
from .tree import Tree, writediscbrackettree
from .kbest import lazykbest, getderiv
from .grammar import lcfrsproductions
from .treetransforms import addbitsets, unbinarize, canonicalize, \
		collapseunary, mergediscnodes, binarize
from .bit import pyintnextset, pyintbitcount
from libc.stdint cimport uint8_t, uint32_t, uint64_t, intptr_t
from .bit cimport abitcount
from .plcfrs cimport DoubleEntry, new_DoubleEntry
from .containers cimport Grammar, ProbRule, LexicalRule, Chart, Edges, \
		MoreEdges, SmallChartItem, FatChartItem, Edge, RankedEdge, \
		new_RankedEdge, logprobadd, logprobsum, yieldranges
cimport cython

from libc.string cimport memset
cdef extern from "macros.h":
	uint64_t TESTBIT(uint64_t a[], int b)
	void SETBIT(uint64_t a[], int b)
	void CLEARBIT(uint64_t a[], int b)
	int BITNSLOTS(int nb)

include "constants.pxi"

REMOVEIDS = re.compile('@[-0-9]+')
REMOVEWORDTAGS = re.compile('@[^ )]+')
cdef str NONCONSTLABEL = ''
cdef str NEGATIVECONSTLABEL = '-#-'

cpdef getderivations(Chart chart, int k, bint kbest=True, bint sample=False,
		derivstrings=True):
	"""Get *k*-best and/or sampled derivations from chart.

	:param k: number of derivations to extract from chart
	:param sample: whether to *k* derivations sample from chart
	:param kbest: whether to extract *k*-best derivations from chart
	:param derivstrings: whether to create derivations as strings
	:returns: tuple ``(derivations, entries)``; two lists of equal length:

		:derivations: list of tuples ``(deriv, logprob)`` where deriv is a
			string; or list of None if ``derivstrings==False``
		:entries: list of RankedEdge objects.
	"""
	cdef list derivations = [], entries = []
	if not (kbest or sample):
		raise ValueError('at least one of kbest or sample needs to be True.')
	chart.rankededges = {}
	if kbest:
		derivations, unused_explored = lazykbest(chart, k, derivs=derivstrings)
		entries = chart.rankededges[chart.root()]
	if sample:
		derivations.extend(
				getsamples(chart, k, None))
		# filter out duplicate derivations
		entries = chart.rankededges[chart.root()]
		filteredderivations = dict(zip(derivations, entries))
		entries[:] = filteredderivations.values()
		derivations[:] = filteredderivations.keys()
	return derivations, entries


cpdef marginalize(method, list derivations, list entries, Chart chart,
		list backtransform=None, list sent=None, list tags=None,
		int k=1000, int sldop_n=7, double mcc_labda=1.0, set mcc_labels=None,
		bint bitpar=False):
	"""Take a list of derivations and optimize a given objective function.

	1. Rewrites derivations into the intended parse trees.
	2. Marginalizes (sum / combine) weights of derivations for same parse tree.
	3. Applies objective function (scoring criterion) for ranking parse trees.

	:param method:
		Available objective functions:

		:'mpp': Most Probable Parse
		:'mpd': Most Probable Derivation
		:'shortest': Most Probable Shortest Derivation
		:'sl-dop': Simplicity-Likelihood DOP; select most likely parse from the
			``sldop_n`` parse trees with the shortest derivations.
		:'sl-dop-simple': Approximation of Simplicity-Likelihood DOP
	:param backtransform:
		Dependending on the value of this parameter, one of two ways is
		employed to turn derivations into parse trees:

		:``None``: assume derivations are from DOP reduction and strip
			annotations of the form ``@123`` from labels. This option is also
			applicable when the grammar is not a TSG and derivations already
			coincide with parse trees.
		:list of strings: assume Double-DOP and recover derivations by
			substituting fragments in this list for flattened rules in
			derivations.
	:param k: when ``method='sl-dop``, number of derivations to consider.
	:param bitpar: whether bitpar was used in nbest mode.
	:returns:
		``(parses, msg)``.

		:parses: a list of tuples ``(parsetree, probability, fragments)``
			where ``parsetree`` is a string, ``probability`` is a float
			(0 < p <= 1), or a tuple ``(numfrags, p)`` for shortest derivation
			parsing, and ``fragments`` is a list of fragments in the most
			probable derivation for this parse.
		:msg: a message reporting the number of derivations / parses.
	"""
	cdef bint mpd = method == 'mpd'
	cdef bint shortest = method == 'shortest'
	cdef bint dopreduction = backtransform is None
	cdef DoubleEntry entry
	cdef LexicalRule lexrule
	cdef dict parsetrees = {}, derivs = {}
	cdef str treestr, deriv
	cdef double prob, maxprob
	cdef int m

	if method == 'sl-dop':
		return sldop(dict(derivations), chart, sent, tags, k, sldop_n,
				backtransform, entries, bitpar)
	elif method == 'sl-dop-simple':
		return sldop_simple(dict(derivations), entries, sldop_n, chart,
				backtransform, bitpar)
	elif method == 'shortest':
		# filter out all derivations which are not shortest
		if not dopreduction and not bitpar:
			maxprob = min([entry.value for entry in entries])
			entries = [entry for entry in entries if entry.value == maxprob]
		elif derivations:
			_, maxprob = min(derivations, key=itemgetter(1))
			derivations = [(deriv, prob) for deriv, prob in derivations
					if prob == maxprob]
	elif method == 'mcc':
		return maxconstituentscorrect(derivations, chart,
				backtransform, mcc_labda, mcc_labels)

	if not dopreduction and not bitpar:  # Double-DOP
		for entry in entries:
			prob = entry.value
			try:
				treestr = recoverfragments(entry.key, chart, backtransform)
			except:
				continue
			if shortest:
				# for purposes of tie breaking, calculate the derivation
				# probability in a different model.
				newprob = exp(-getderivprob(entry.key, chart, sent))
				score = (int(prob / log(0.5)), newprob)
				if treestr not in parsetrees or score > parsetrees[treestr]:
					parsetrees[treestr] = score
					derivs[treestr] = entry.key
			elif not mpd and treestr in parsetrees:
				parsetrees[treestr].append(-prob)
			elif not mpd or (treestr not in parsetrees
						or -prob > parsetrees[treestr][0]):
				parsetrees[treestr] = [-prob]
				derivs[treestr] = entry.key
	else:  # DOP reduction / bitpar
		for (deriv, prob), entry in zip(derivations, entries):
			if dopreduction:
				treestr = REMOVEIDS.sub('@1' if mpd or shortest else '', deriv)
			else:
				try:
					treestr = recoverfragments(deriv, chart, backtransform)
				except:
					continue
			if shortest:
				if bitpar:
					# because with bitpar we don't know which rules have been
					# used, read off the rules from the derivation ...
					tree = canonicalize(Tree(deriv))
					newprob = 0.0
					for t in tree.subtrees():
						if isinstance(t[0], Tree):
							if not 1 <= len(t) <= 2:
								raise ValueError('expected binarized tree.')
							m = chart.grammar.rulenos[nodeprod(t)]
							newprob += chart.grammar.bylhs[0][
									chart.grammar.revmap[m]].prob
						else:
							m = chart.grammar.toid[t.label]
							try:
								lexrule = chart.grammar.lexicalbylhs[
										m][sent[t[0]]]
							except KeyError:
								newprob += 30.0
							else:
								newprob += lexrule.prob
				else:
					newprob = getderivprob(entry.key, chart, sent)
				score = (int(prob / log(0.5)), exp(-newprob))
				if treestr not in parsetrees or (not dopreduction
						and score > parsetrees[treestr]):
					parsetrees[treestr] = score
					derivs[treestr] = deriv
				elif dopreduction:
					oldscore = parsetrees[treestr]
					parsetrees[treestr] = oldscore[0], oldscore[1] + score[1]
			elif treestr not in parsetrees or (not dopreduction and mpd
					and -prob > parsetrees[treestr][0]):
				parsetrees[treestr] = [-prob]
				derivs[treestr] = deriv
			elif treestr in parsetrees and (dopreduction or not mpd):
				parsetrees[treestr].append(-prob)

	if mpd and dopreduction:
		results = [(REMOVEIDS.sub('', treestr), logprobsum(probs),
				fragmentsinderiv(derivs[treestr], chart, backtransform))
				for treestr, probs in parsetrees.items()]
	elif shortest and dopreduction:
		results = [(REMOVEIDS.sub('', treestr), (-a, b),
				fragmentsinderiv(derivs[treestr], chart, backtransform))
				for treestr, (a, b) in parsetrees.items()]
	elif shortest:
		results = [(treestr, (-a, b),
				fragmentsinderiv(derivs[treestr], chart, backtransform))
				for treestr, (a, b) in parsetrees.items()]
	else:
		results = [(treestr, logprobsum(probs),
				fragmentsinderiv(derivs[treestr], chart, backtransform))
				for treestr, probs in parsetrees.items()]

	msg = '%d derivations, %d parsetrees' % (
			len(derivations if dopreduction else entries),
			len(parsetrees))
	return results, msg


cdef maxconstituentscorrect(list derivations, Chart chart,
		list backtransform, double labda, set labels=None):
	"""Approximate the Most Constituents Correct (MCC) parse from n-best list.

	:param derivations: list of derivations as strings
	:param chart: the chart
	:param backtransform: table of rules mapped to fragments
	:param labda:
		weight to assign to recall rate vs. the mistake rate;
		the default 1.0 assigns equal weight to both.
	:param labels:
		if given, the set of labels to optimize for;
		by default, all labels are optimized.
	"""
	# NB: this requires derivations not entries.
	cdef double sentprob = 0.0, maxscore = 0.0
	cdef double prob, score, maxcombscore, contribution
	cdef short start, spanlen
	cdef object span, leftspan, rightspan, maxleft  # bitsets as Python ints
	# table[start][spanlen][span][label] = prob
	table = [[defaultdict(dict)
				for _ in range(len(chart.sent) - n + 1)]
			for n in range(len(chart.sent))]
	tree = None
	# get marginal probabilities
	for deriv, prob in derivations:
		if backtransform is None:
			treestr = REMOVEIDS.sub('', deriv)
		else:
			treestr = recoverfragments(deriv, chart, backtransform)
		# rebinarize, collapse unaries, because we optimize only
		# for constituents in the tree as it will be evaluated.
		tree = addbitsets(
				binarize(
				collapseunary(unbinarize(
					mergediscnodes(unbinarize(
						Tree(treestr),
						childchar=':', expandunary=False)),
					expandunary=False),
					joinchar='++'),
					leftmostunary=True))
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
		for start, _ in enumerate(table):
			if len(table[start]) <= spanlen:
				continue
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
				childchar='NONE', unarychar='++')
	except (ValueError, AttributeError):
		return [], 'MCC failed. %s' % tmp
	else:
		result.label = tree.label  # fix root label
		return [(str(result), maxscore,
				None)], 'sentprob: %g' % sentprob


def gettree(cells, span):
	"""Extract parse tree from most constituents correct table."""
	if span not in cells:
		raise ValueError('MCC: span not in cell: %r' % bin(span))
	label, unused_score, leftspan = cells[span]
	if leftspan not in cells:
		return '(%s %d)' % (label, pyintnextset(span, 0))
	rightspan = span & ~leftspan
	if label == NONCONSTLABEL or label == NEGATIVECONSTLABEL:
		return '%s %s' % (gettree(cells, leftspan), gettree(cells, rightspan))
	return '(%s %s %s)' % (label,
			gettree(cells, leftspan), gettree(cells, rightspan))


cdef sldop(dict derivations, Chart chart, list sent, list tags,
		int m, int sldop_n, list backtransform, entries, bint bitpar):
	"""'Proper' method for sl-dop.

	Parses sentence once more to find shortest derivations, pruning away any
	chart item not occurring in the *n* most probable parse trees; we need to
	parse again because we have to consider all derivations for the *n* most
	likely trees.

	:returns: the intersection of the most probable parse trees and their
		shortest derivations, with probabilities of the form (subtrees, prob).

	NB: doesn't seem to work so well, so may contain a subtle bug.
		Does not support PCFG charts."""
	cdef dict derivs = {}
	# collect derivations for each parse tree
	derivsfortree = defaultdict(set)
	if backtransform is None:
		for deriv in derivations:
			derivsfortree[REMOVEIDS.sub('', deriv)].add(deriv)
	elif bitpar:
		for deriv in derivations:
			derivsfortree[recoverfragments(deriv, chart,
					backtransform)].add(deriv)
	else:
		for entry in entries:
			deriv = getderiv(chart.root(), (<DoubleEntry>entry).key, chart,
					None)
			derivations[deriv] = (<DoubleEntry>entry).value
			derivsfortree[recoverfragments((<DoubleEntry>entry).key, chart,
					backtransform)].add(deriv)
	# sum over probs of derivations to get probs of parse trees
	parsetreeprob = {tree: logprobsum([-derivations[d] for d in ds])
			for tree, ds in derivsfortree.items()}
	nmostlikelytrees = set(nlargest(sldop_n, parsetreeprob,
			key=parsetreeprob.get))

	model = chart.grammar.modelnames[chart.grammar.currentmodel]
	chart.grammar.switch(u'shortest', logprob=True)
	shortestderivations, unused_explored, chart2 = treeparsing(
			nmostlikelytrees, sent, chart.grammar, m, backtransform, tags)
	if not chart2.rankededges.get(chart2.root()):
		return [], 'SL-DOP couldn\'t find parse for tree'
	result = {}
	for (deriv, s), entry in zip(shortestderivations,
			chart2.rankededges[chart2.root()]):
		if backtransform is None:
			treestr = REMOVEIDS.sub('', deriv)
		else:
			treestr = recoverfragments(
					deriv if bitpar else (<DoubleEntry>entry).key,
					chart2, backtransform)
		if treestr in nmostlikelytrees and treestr not in result:
			result[treestr] = (-abs(int(s / log(0.5))), parsetreeprob[treestr])
			derivs[treestr] = fragmentsinderiv(
					deriv if bitpar or backtransform is None
					else (<DoubleEntry>entry).key,
					chart2, backtransform)
			if len(result) > sldop_n:
				break
	chart.grammar.switch(model, logprob=True)
	if not len(result):
		return [], 'no matching derivation found'
	msg = '(%d derivations, %d of %d parsetrees)' % (
		len(derivations), min(sldop_n, len(parsetreeprob)), len(parsetreeprob))
	return [(tree, result[tree], derivs[tree]) for tree in result], msg


cdef sldop_simple(dict derivations, list entries, int sldop_n,
		Chart chart, list backtransform, bint bitpar):
	"""Simple sl-dop method.

	Estimates shortest derivation directly from number of addressed nodes in
	the *k*-best derivations. After selecting the *n* best parse trees, the one
	with the shortest derivation is returned. In other words, selects shortest
	derivation among the list of available derivations, instead of finding the
	shortest among all possible derivations using Viterbi."""
	cdef DoubleEntry entry
	cdef dict derivs = {}, keys = {}
	derivsfortree = defaultdict(set)
	# collect derivations for each parse tree
	if backtransform is None:
		for deriv in derivations:
			tree = REMOVEIDS.sub('', deriv)
			derivsfortree[tree].add(deriv)
	elif bitpar:
		for deriv in derivations:
			tree = recoverfragments(deriv, chart, backtransform)
			derivsfortree[tree].add(deriv)
	else:
		for entry in entries:
			deriv = getderiv(chart.root(), entry.key, chart, '}<')
			tree = recoverfragments(entry.key, chart, backtransform)
			keys[deriv] = entry.key
			derivations[deriv] = entry.value
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
		derivs[tree] = fragmentsinderiv(
				deriv if bitpar or backtransform is None else keys[deriv],
				chart, backtransform)
	msg = '(%d derivations, %d of %d parsetrees)' % (
			len(derivations), len(result), len(parsetreeprob))
	return [(tree, result[tree], derivs[tree]) for tree in result], msg


cpdef str recoverfragments(deriv, Chart chart, list backtransform):
	"""Reconstruct a DOP derivation from a derivation with flattened fragments.

	:param deriv: a RankedEdge or a string representing a derivation.
	:param backtransform: a list with fragments (as string templates)
		corresponding to grammar rules.
	:returns: expanded derivation as a string.

	The flattened fragments in the derivation should be left-binarized, expect
	when ``deriv`` is a string in which case the derivation does not have to be
	binarized.

	Does on-the-fly debinarization following labels that are not mapped to a
	label in the coarse grammar, i.e., it assumes that neverblockre is only
	used to avoid blocking nonterminals from the double-dop binarization
	(containing the string '}<'). Note that this means getmapping() has to have
	been called on `chart.grammar`, even when not doing coarse-to-fine
	parsing."""
	if isinstance(deriv, RankedEdge):
		result = recoverfragments_(deriv, chart, backtransform)
	elif isinstance(deriv, str):
		deriv = Tree(deriv)
		result = recoverfragments_str(deriv, chart, backtransform)
	else:
		raise ValueError('derivation has unexpected type %r.' % type(deriv))
	return REMOVEWORDTAGS.sub('', result)


cdef str recoverfragments_(RankedEdge deriv, Chart chart,
		list backtransform):
	cdef RankedEdge child
	cdef list children = []
	cdef str frag = backtransform[deriv.edge.rule.no]  # template
	# NB: this is the only code that uses the .head field of RankedEdge

	# collect all children w/on the fly left-factored debinarization
	if deriv.edge.rule.rhs2:  # is there a right child?
		# keep going while left child is part of same binarized constituent
		# instead of looking for a binarization marker in the label string, we
		# use the fact that such labels do not have a mapping as proxy.
		while chart.grammar.mapping[deriv.edge.rule.rhs1] == 0:
			# one of the right children
			children.append((<DoubleEntry>chart.rankededges[
					chart.right(deriv)][deriv.right]).key)
			# move on to next node in this binarized constituent
			deriv = (<DoubleEntry>chart.rankededges[
					chart.left(deriv)][deriv.left]).key
		# last right child
		if deriv.edge.rule.rhs2:  # is there a right child?
			children.append((<DoubleEntry>chart.rankededges[
					chart.right(deriv)][deriv.right]).key)
	elif chart.grammar.mapping[deriv.edge.rule.rhs1] == 0:
		deriv = (<DoubleEntry>chart.rankededges[
				chart.left(deriv)][deriv.left]).key
	# left-most child
	children.append((<DoubleEntry>chart.rankededges[
			chart.left(deriv)][deriv.left]).key)

	# recursively expand all substitution sites
	children = ['(%s %d)' % (chart.grammar.tolabel[chart.label(child.head)],
			chart.lexidx(child.edge)) if child.edge.rule is NULL
			else recoverfragments_(child, chart, backtransform)
			for child in reversed(children)]

	# substitute results in template
	return frag.format(*children)

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


cdef str recoverfragments_str(deriv, Chart chart, list backtransform):
	cdef list children = []
	cdef str frag
	frag = backtransform[chart.grammar.rulenos[nodeprod(deriv)]]
	# collect children w/on the fly left-factored debinarization
	if len(deriv) >= 2:  # is there a right child?
		# keep going while left child is part of same binarized constituent
		# this shortcut assumes that neverblockre is only used to avoid
		# blocking nonterminals from the double-dop binarization.
		while '}<' in deriv[0].label:
			# one of the right children
			children.extend(reversed(deriv[1:]))
			# move on to next node in this binarized constituent
			deriv = deriv[0]
		# last right child
		children.extend(reversed(deriv[1:]))
	elif '}<' in deriv[0].label:
		deriv = deriv[0]
	# left-most child
	children.append(deriv[0])

	# recursively expand all substitution sites
	children = [recoverfragments_str(child, chart, backtransform)
			if isinstance(child[0], Tree)
			else ('(%s %d)' % (child.label, child[0]))
			for child in reversed(children)]

	# substitute results in template
	return frag.format(*children)


def fragmentsinderiv(deriv, chart, list backtransform):
	"""Extract the list of fragments that were used in a given derivation.

	:returns: a list of fragments of the form ``(frag, sent)`` where frag is a
		string and sent is a list of tokens; in ``sent``, ``None`` indicates a
		frontier non-terminal, and a string indicates a token. """
	result = []
	if isinstance(deriv, RankedEdge):
		fragmentsinderiv_(deriv, chart, backtransform, result)
	elif isinstance(deriv, str) and backtransform is None:
		deriv = Tree(deriv)
		result = [writediscbrackettree(
				REMOVEIDS.sub('', str(splitfrag(node))),
				chart.sent).rstrip()
				for node in deriv.subtrees(frontiernt)]
	elif isinstance(deriv, str):
		deriv = Tree(deriv)
		fragmentsinderiv_str(deriv, chart, backtransform, result)
	else:
		raise ValueError('deriv should be a RankedEdge or a string.')
	return [_fragments.pygetsent(frag) for frag in result]


cdef fragmentsinderiv_(RankedEdge deriv, Chart chart,
		list backtransform, list result):
	cdef RankedEdge child
	cdef list children = []
	cdef str frag = backtransform[deriv.edge.rule.no]  # template

	# collect all children w/on the fly left-factored debinarization
	if deriv.edge.rule.rhs2:  # is there a right child?
		# keep going while left child is part of same binarized constituent
		# instead of looking for a binarization marker in the label string, we
		# use the fact that such labels do not have a mapping as proxy.
		while chart.grammar.mapping[deriv.edge.rule.rhs1] == 0:
			# one of the right children
			children.append((<DoubleEntry>chart.rankededges[
					chart.right(deriv)][deriv.right]).key)
			# move on to next node in this binarized constituent
			deriv = (<DoubleEntry>chart.rankededges[
					chart.left(deriv)][deriv.left]).key
		# last right child
		if deriv.edge.rule.rhs2:  # is there a right child?
			children.append((<DoubleEntry>chart.rankededges[
					chart.right(deriv)][deriv.right]).key)
	elif chart.grammar.mapping[deriv.edge.rule.rhs1] == 0:
		deriv = (<DoubleEntry>chart.rankededges[
				chart.left(deriv)][deriv.left]).key
	# left-most child
	children.append((<DoubleEntry>chart.rankededges[
			chart.left(deriv)][deriv.left]).key)

	result.append(frag.format(*['(%s %s)' % (
				chart.grammar.tolabel[chart.label(deriv.head)].split('@')[0],
				('%d=%s' % (chart.lexidx(deriv.edge),
					chart.sent[chart.lexidx(deriv.edge)])
					if '@' in chart.grammar.tolabel[chart.label(deriv.head)]
					else yieldranges(chart.indices(deriv.head))))
				for deriv in reversed(children)]))
	# recursively visit all substitution sites
	for child in reversed(children):
		if child.edge.rule is not NULL:
			fragmentsinderiv_(child, chart, backtransform, result)
		elif '@' not in chart.grammar.tolabel[chart.label(child.head)]:
			result.append('(%s %d=%s)' % (
					chart.grammar.tolabel[chart.label(child.head)],
					chart.lexidx(child.edge),
					chart.sent[chart.lexidx(child.edge)]))


cdef fragmentsinderiv_str(deriv, Chart chart, list backtransform, list result):
	cdef list children = []
	cdef str frag
	frag = backtransform[chart.grammar.rulenos[nodeprod(deriv)]]
	# collect children w/on the fly left-factored debinarization
	if len(deriv) >= 2:  # is there a right child?
		# keep going while left child is part of same binarized constituent
		# this shortcut assumes that neverblockre is only used to avoid
		# blocking nonterminals from the double-dop binarization.
		while '}<' in deriv[0].label:
			# one of the right children
			children.extend(reversed(deriv[1:]))
			# move on to next node in this binarized constituent
			deriv = deriv[0]
		# last right child
		children.extend(reversed(deriv[1:]))
	elif '}<' in deriv[0].label:
		deriv = deriv[0]
	# left-most child
	children.append(deriv[0])

	result.append(frag.format(*['(%s %s)' % (
			child.label.split('@')[0],
			('%d=%s' % (child[0], chart.sent[child[0]]) if '@' in child.label
				else yieldranges(sorted(child.leaves()))))
			for child in reversed(children)]))
	# recursively visit all substitution sites
	for child in reversed(children):
		if isinstance(child[0], Tree):
			fragmentsinderiv_str(child, chart, backtransform, result)
		elif '@' not in child.label:
			result.append('(%s %d=%s)' % (child.label, child[0],
					chart.sent[child[0]]))


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
		maskrules=False):
	"""Assign probabilities to a sequence of trees with a DOP grammar.

	Given a sequence of trees (as strings), parse them with a DOP grammar
	to get parse tree probabilities; will consider multiple derivations.

	:param maskrules: If True, prune any grammar rule not in the trees.
		Exploits rulemapping of grammar which should be mapped to itself;
		e.g., 'NP@2 => DT@3 NN' should be mapped to 'NP => DT NN' in the
		same grammar. To remove the mask, issue ``grammar.setmask(None)``
	"""
	# Parsing & pruning inside the disambiguation module is rather kludgy,
	# but the problem is that we need to get probabilities of trees,
	# not just of derivations. Therefore the coarse-to-fine methods
	# do not apply directly.
	cdef FatChartItem fitem
	cdef int n, lensent = len(sent)
	# cdef int selected = 0
	whitelist = [set() for _ in grammar.toid]
	if maskrules:
		grammar.setmask([])  # block all rules
	for treestr in trees:
		tree = Tree(treestr)
		for node, (r, yf) in zip(tree.subtrees(),
				lcfrsproductions(tree, sent)):
			leaves = node.leaves()
			if lensent < sizeof(uint64_t) * 8:
				item = SmallChartItem(0, sum([1L << n for n in leaves]))
			else:
				fitem = item = FatChartItem(0)
				for n in leaves:
					SETBIT(fitem.vec, n)
			try:
				whitelist[grammar.toid[node.label]].add(item)
			except KeyError:
				return [], "'%s' not in grammar" % node.label, None

			if maskrules and isinstance(node[0], Tree):
				try:
					ruleno = grammar.rulenos[prodrepr(r, yf)]
				except KeyError:
					return [], "'%s' not in grammar" % prodrepr(r, yf), None
				for n in grammar.rulemapping[ruleno]:
					CLEARBIT(grammar.mask, n)
					# if TESTBIT(grammar.mask, n):
					# 	CLEARBIT(grammar.mask, n)
					# 	selected += 1
	# if maskrules:
	# 	grammar = SubsetGrammar(grammar, selected)

	# Project labels to all possible labels that generate that label. For DOP
	# reduction, all possible ids; for Double DOP, ignore artificial labels.
	for label, n in grammar.toid.items():
		if backtransform is None:
			whitelist[n] = whitelist[grammar.toid[REMOVEIDS.sub('', label)]]
		elif '@' in label or '}<' in label:
			whitelist[n] = None  # do not prune item

	# finally, we parse with the small set of allowed labeled spans.
	# we do not parse with the PCFG parser even if possible, because that
	# requires a different way of pruning.
	chart, _ = plcfrs.parse(sent, grammar, tags=tags, whitelist=whitelist)

	if not chart:
		return [], 'tree parsing failed', None
	return lazykbest(chart, m) + (chart, )


cdef double getderivprob(RankedEdge deriv, Chart chart, list sent):
	"""Recursively calculate probability of a derivation.

	Useful to obtain probability of derivation under different probability
	model of the same grammar."""
	cdef double result
	if deriv.edge.rule is NULL:  # is terminal
		label = chart.label(deriv.head)
		word = sent[chart.lexidx(deriv.edge)]
		return (<LexicalRule>chart.grammar.lexicalbylhs[label][word]).prob
	result = deriv.edge.rule.prob
	result += getderivprob((<DoubleEntry>chart.rankededges[
			chart.left(deriv)][deriv.left]).key,
			chart, sent)
	if deriv.edge.rule.rhs2:
		result += getderivprob((<DoubleEntry>chart.rankededges[
				chart.right(deriv)][deriv.right]).key,
				chart, sent)
	return result

cpdef viterbiderivation(Chart chart):
	"""Wrapper to get Viterbi derivation from chart."""
	# Ask for at least 10 derivations because unary cycles.
	derivations = lazykbest(chart, 10)[0]
	return derivations[0]


def getsamples(Chart chart, k, debin=None):
	"""Samples *k* derivations from a chart."""
	cdef dict tables = {}
	cdef Edge *edge
	cdef Edges edges
	cdef MoreEdges *edgelist
	cdef double prob, prev
	cdef size_t n
	chartidx = {}
	for item in chart.getitems():
		chartidx[item] = []
		edges = chart.getedges(item)
		edgelist = edges.head if edges is not None else NULL
		while edgelist is not NULL:
			for n in range(edges.len if edgelist is edges.head
					else EDGES_SIZE):
				edge = &(edgelist.data[n])
				# HACK: store pointer to edge as Python int
				if edge.rule is NULL:
					chartidx[item].append(
							(chart.subtreeprob(item), <intptr_t>edge))
				else:
					prob = edge.rule.prob
					# FIXME: work w/inside prob?
					# prob += chart.subtreeprob(chart._left(item, edge))
					# if edge.rule.rhs2:
					# 	prob += chart.subtreeprob(chart._right(item, edge))
					chartidx[item].append((prob, <intptr_t>edge))
			edgelist = edgelist.prev
		# sort edges so that highest prob (=lowest neglogprob) comes first
		chartidx[item].sort()
	for item in chartidx:
		tables[item] = []
		prev = 0.0
		for prob, _ in chartidx[item]:
			prev += exp(-prob)
			tables[item].append(prev)
	result = []
	for _ in range(k):
		result.append(samplechart(
				chart.root(), chart, chartidx, tables, debin))
	return result


cdef samplechart(item, Chart chart,
		dict chartidx, dict tables, str debin):
	"""Samples a derivation from a chart."""
	cdef Edge *edge
	cdef RankedEdge rankededge
	cdef double prob
	cdef list lst = tables[item]
	rnd = random() * lst[len(lst) - 1]
	idx = bisect_right(lst, rnd)
	_, ptr = chartidx[item][idx]
	edge = <Edge *><intptr_t>ptr  # hack: convert Python int into pointer
	label = chart.label(item)
	if edge.rule is NULL:  # terminal
		idx = chart.lexidx(edge)
		rankededge = new_RankedEdge(item, edge, -1, -1)
		chart.rankededges.setdefault(item, []).append(
				new_DoubleEntry(rankededge, chart.subtreeprob(item), 0))
		deriv = '(%s %d)' % (chart.grammar.tolabel[label], idx)
		return deriv, chart.subtreeprob(item)
	tree, p1 = samplechart(chart.copy(chart._left(item, edge)),
			chart, chartidx, tables, debin)
	prob = edge.rule.prob + p1
	if edge.rule.rhs2:
		tree2, p2 = samplechart(chart.copy(chart._right(item, edge)),
					chart, chartidx, tables, debin)
		tree += ' ' + tree2
		prob += p2
	if debin is None or debin not in chart.grammar.tolabel[label]:
		tree = '(%s %s)' % (chart.grammar.tolabel[label], tree)
	# create an edge that has as children the edges that were just created
	# by our recursive calls
	rankededge = new_RankedEdge(item, edge,
			len(chart.rankededges[chart._left(item, edge)]) - 1,
			(len(chart.rankededges[chart._right(item, edge)]) - 1)
				if edge.rule.rhs2 else -1)
	# NB: this is actually 'samplededges', not 'rankededges'
	chart.rankededges.setdefault(item, []).append(
			new_DoubleEntry(rankededge, prob, 0))
	return tree, prob


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
		if word not in fine.lexicalbyword:
			cell[fine.toid[pos]] = -0.0
			continue
		for lexrule in fine.lexicalbyword[word]:
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
		prod = coarse.rulenos[prodrepr(r, yf)]
		if len(node) == 1:  # unary node
			for ruleno in fine.rulemapping[prod]:
				rule = &(fine.bylhs[0][fine.revmap[ruleno]])
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
				rule = &(fine.bylhs[0][fine.revmap[ruleno]])
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
				tmp['trees1'], trees, list(frags.values()), indices=True)
		score = 0
		rev = defaultdict(set)
		for n, (frag, idx) in enumerate(zip(frags, indices)):
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


cdef str prodrepr(r, yf):
	"""Produce string repr as in ``Grammar.rulenos[]`` for a rule.

	>>> prodrepr((('X', 'A', '', 'C', 'D'), ((0, 1, 2, 3), )))
	'0123 X A B C D'"""
	yf = ','.join([''.join(map(str, component)) for component in yf])
	return ' '.join([yf, ' '.join(r)])


cdef str nodeprod(deriv):
	"""Produce string repr as in ``Grammar.rulenos[]`` of a non-disc. Tree.

	>>> nodeprod(Tree('(X (A 0) (B 1) (C 2) (D 3))'))
	'0123 X A B C D'"""
	return ('%s %s %s' % (''.join(map(str, range(len(deriv)))),
			deriv.label, ' '.join([a.label for a in deriv])))


def test():
	from .grammar import dopreduction
	from .containers import Grammar
	from . import plcfrs

	def e(x):
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
	grammar = Grammar(xgrammar)
	grammar.register(u'shortest', altweights['shortest'])
	print(grammar)
	sent = 'a b c'.split()
	chart, _ = plcfrs.parse(sent, grammar, None, True)
	assert chart
	vitderiv, vitprob = viterbiderivation(chart)
	derivations, entries = getderivations(chart, 1000, True, False, True)
	mpd, _ = marginalize('mpd', derivations, entries, chart)
	mpp, _ = marginalize('mpp', derivations, entries, chart)
	mcc, _ = marginalize('mcc', derivations, entries, chart)
	sldop_, _ = marginalize('sl-dop', derivations, entries, chart, k=1000,
			sldop_n=7, sent=sent)
	sldopsimple, _ = marginalize('sl-dop-simple', derivations, entries,
			chart, k=1000, sldop_n=7, sent=sent)
	short, _ = marginalize('shortest', derivations, entries, chart, sent=sent)
	derivations, entries = getderivations(chart, 1000, False, True, True)
	mppsampled, _ = marginalize('mpp', derivations, entries, chart)
	print('\nvit:\t\t%s %r' % (REMOVEIDS.sub('', vitderiv),
			exp(-vitprob)),
		'MPD:\t\t%s %r' % e(mpd), 'MCC:\t\t%s %r' % e(mcc),
		'MPP:\t\t%s %r' % e(mpp), 'MPP sampled:\t%s %r' % e(mppsampled),
		'SL-DOP n=7:\t%s %r' % e(sldop_),
		'simple SL-DOP:\t%s %r' % e(sldopsimple),
		'shortest:\t%s %r' % e(short), sep='\n')

__all__ = ['getderivations', 'marginalize', 'gettree', 'recoverfragments',
		'fragmentsinderiv', 'treeparsing', 'viterbiderivation', 'getsamples',
		'doprerank', 'dopparseprob', 'frontiernt', 'splitfrag']
