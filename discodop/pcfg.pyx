""" Probabilistic Context-Free Grammar (PCFG) parser using CKY. """

# python imports
from __future__ import print_function
from math import log, exp
from collections import defaultdict
from subprocess import Popen, PIPE
from itertools import count
from os import unlink
import re
import sys
import logging
import tempfile
import numpy as np
from tree import Tree
from agenda import EdgeAgenda


# cython imports
from libc.stdlib cimport malloc, calloc, free
from cpython cimport PyDict_Contains, PyDict_GetItem
cimport numpy as np
from agenda cimport EdgeAgenda
from containers cimport Grammar, Rule, LexicalRule, CFGEdge, CFGChartItem, \
		new_CFGChartItem, new_CFGEdge, UChar, UInt, ULong, ULLong, logprobadd

cdef extern from "math.h":
	bint isinf(double x)
	bint isfinite(double x)

np.import_array()
DEF SX = 1
DEF SXlrgaps = 2
cdef CFGChartItem NONE = new_CFGChartItem(0, 0, 0)


def parse(list sent, Grammar grammar, tags=None, start=1, chart=None):
	#assert all(grammar.fanout[a] == 1 for a in range(1, grammar.nonterminals))
	if grammar.nonterminals < 20000 and chart is None:
		return parse_dense(sent, grammar, start=start, tags=tags)
	return parse_sparse(sent, grammar, start=start, tags=tags, chart=chart)


def parse_dense(list sent, Grammar grammar, start=1, tags=None):
	""" A CKY parser modeled after Bodenstab's `fast grammar loop'
		and the Stanford parser. Tries to apply each grammar rule for all
		spans. Tracks the viterbi scores in a separate array. For grammars with
		up to 10,000 nonterminals.
		Edges are kept in a dictionary for each labelled span:
		chart[left][right][label] = {edge1: edge1, edge2: edge2} """
	cdef:
		short left, right, mid, span, lensent = len(sent)
		short narrowl, narrowr, widel, wider, minmid, maxmid
		UInt n, lhs, rhs1, spans = 0, items = 0, edges = 0
		double oldscore, prob
		Rule *rule
		LexicalRule lexrule
		EdgeAgenda unaryagenda = EdgeAgenda()
		list chart = [[{} for _ in range(lensent + 1)] for _ in range(lensent)]
		dict cell
		# the viterbi chart is initially filled with infinite log probabilities,
		# cells containing NaN are blocked.
		np.ndarray[np.double_t, ndim=3] viterbi = np.empty(
				(grammar.nonterminals, lensent, lensent + 1), dtype='d')
		# matrices for the filter which gives minima and maxima for splits
		np.ndarray[np.int16_t, ndim=2] minleft, maxleft, minright, maxright
	assert grammar.maxfanout == 1, "Not a PCFG! fanout = %d" % grammar.maxfanout
	assert grammar.logprob
	minleft = np.empty((grammar.nonterminals, lensent + 1), dtype='int16')
	maxleft = np.empty_like(minleft)
	minright = np.empty_like(minleft)
	maxright = np.empty_like(minleft)
	viterbi.fill(np.inf)
	minleft.fill(-1)
	maxleft.fill(lensent + 1)
	minright.fill(lensent + 1)
	maxright.fill(-1)

	# assign POS tags
	for left, word in enumerate(sent):
		tag = tags[left].encode('ascii') if tags else None
		right = left + 1
		cell = chart[left][right]
		recognized = False
		for lexrule in grammar.lexicalbyword.get(word, ()):
			lhs = lexrule.lhs
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if (not tags or grammar.tolabel[lhs] == tag
				or grammar.tolabel[lhs].startswith(tag + b'@')):
				x = viterbi[lhs, left, right] = lexrule.prob
				edge = new_CFGEdge(x, NULL, right)
				cell[lhs] = {edge: edge}
				recognized = True
				# update filter
				if left > minleft[lhs, right]:
					minleft[lhs, right] = left
				if left < maxleft[lhs, right]:
					maxleft[lhs, right] = left
				if right < minright[lhs, left]:
					minright[lhs, left] = right
				if right > maxright[lhs, left]:
					maxright[lhs, left] = right
		if not recognized:
			if tags and tag in grammar.toid:
				lhs = grammar.toid[tag]
				viterbi[lhs, left, right] = 0.0
				edge = new_CFGEdge(0.0, NULL, right)
				cell[lhs] = {edge: edge}
				# update filter
				if left > minleft[lhs, right]:
					minleft[lhs, right] = left
				if left < maxleft[lhs, right]:
					maxleft[lhs, right] = left
				if right < minright[lhs, left]:
					minright[lhs, left] = right
				if right > maxright[lhs, left]:
					maxright[lhs, left] = right
			else:
				return chart, NONE, "not covered: %r" % (tag or word, )

		# unary rules on the span of this POS tag
		# NB: for this agenda, only the probabilities of the edges matter
		unaryagenda.update([(
			rhs1, new_CFGEdge(viterbi[rhs1, left, right], NULL, 0))
			for rhs1 in range(grammar.nonterminals)
			if isfinite(viterbi[rhs1, left, right])
			and grammar.unary[rhs1].rhs1 == rhs1])
		while unaryagenda.length:
			rhs1 = unaryagenda.popentry().key
			for n in range(grammar.numrules):
				rule = &(grammar.unary[rhs1][n])
				if rule.rhs1 != rhs1:
					break
				lhs = rule.lhs
				prob = rule.prob + viterbi[rhs1, left, right]
				edge = new_CFGEdge(prob, rule, right)
				if isfinite(viterbi[lhs, left, right]):
					if prob < viterbi[lhs, left, right]:
						unaryagenda.setifbetter(lhs, <CFGEdge>edge)
						viterbi[lhs, left, right] = prob
						cell[lhs][edge] = edge
					elif (edge not in cell[lhs] or
							prob < (<CFGEdge>cell[lhs][edge]).inside):
						cell[lhs][edge] = edge
					continue
				unaryagenda.setifbetter(lhs, <CFGEdge>edge)
				viterbi[lhs, left, right] = prob
				cell[lhs] = {edge: edge}
				# update filter
				if left > minleft[lhs, right]:
					minleft[lhs, right] = left
				if left < maxleft[lhs, right]:
					maxleft[lhs, right] = left
				if right < minright[lhs, left]:
					minright[lhs, left] = right
				if right > maxright[lhs, left]:
					maxright[lhs, left] = right

	for span in range(2, lensent + 1):
		# constituents from left to right
		for left in range(lensent - span + 1):
			right = left + span
			cell = chart[left][right]
			# binary rules
			for n in range(grammar.numrules):
				rule = &(grammar.bylhs[0][n])
				if rule.lhs == grammar.nonterminals:
					break
				elif not rule.rhs2:
					continue
				lhs = rule.lhs
				narrowr = minright[rule.rhs1, left]
				if narrowr >= right:
					continue
				narrowl = minleft[rule.rhs2, right]
				if narrowl < narrowr:
					continue
				widel = maxleft[rule.rhs2, right]
				minmid = narrowr if narrowr > widel else widel
				wider = maxright[rule.rhs1, left]
				maxmid = wider if wider < narrowl else narrowl
				oldscore = viterbi[lhs, left, right]
				if isinf(oldscore):
					cell[lhs] = {}
				for mid in range(minmid, maxmid + 1):
					if (isfinite(viterbi[rule.rhs1, left, mid])
						and isfinite(viterbi[rule.rhs2, mid, right])):
						prob = (rule.prob + viterbi[rule.rhs1, left, mid]
								+ viterbi[rule.rhs2, mid, right])
						edge = new_CFGEdge(prob, rule, mid)
						if prob < viterbi[lhs, left, right]:
							if isinf(viterbi[lhs, left, right]):
								cell[lhs] = {}
							viterbi[lhs, left, right] = prob
							cell[lhs][edge] = edge
						elif (edge not in cell[lhs] or
								prob < (<CFGEdge>cell[lhs][edge]).inside):
							cell[lhs][edge] = edge
				# update filter
				if isinf(oldscore):
					if not cell[lhs]:
						del cell[lhs]
						continue
					if left > minleft[lhs, right]:
						minleft[lhs, right] = left
					if left < maxleft[lhs, right]:
						maxleft[lhs, right] = left
					if right < minright[lhs, left]:
						minright[lhs, left] = right
					if right > maxright[lhs, left]:
						maxright[lhs, left] = right

			# unary rules on this span
			unaryagenda.update([(rhs1,
				new_CFGEdge(viterbi[rhs1, left, right], NULL, 0))
				for rhs1 in range(grammar.nonterminals)
				if isfinite(viterbi[rhs1, left, right])
				and grammar.unary[rhs1].rhs1 == rhs1])
			while unaryagenda.length:
				rhs1 = unaryagenda.popentry().key
				for n in range(grammar.numrules):
					rule = &(grammar.unary[rhs1][n])
					if rule.rhs1 != rhs1:
						break
					prob = rule.prob + viterbi[rhs1, left, right]
					lhs = rule.lhs
					edge = new_CFGEdge(prob, rule, right)
					if isfinite(viterbi[lhs, left, right]):
						if prob < viterbi[lhs, left, right]:
							unaryagenda.setifbetter(lhs, <CFGEdge>edge)
							viterbi[lhs, left, right] = prob
							cell[lhs][edge] = edge
						elif (edge not in cell[lhs] or
								prob < (<CFGEdge>cell[lhs][edge]).inside):
							cell[lhs][edge] = edge
						continue
					unaryagenda.setifbetter(lhs, <CFGEdge>edge)
					viterbi[lhs, left, right] = prob
					cell[lhs] = {edge: edge}
					# update filter
					if left > minleft[lhs, right]:
						minleft[lhs, right] = left
					if left < maxleft[lhs, right]:
						maxleft[lhs, right] = left
					if right < minright[lhs, left]:
						minright[lhs, left] = right
					if right > maxright[lhs, left]:
						maxright[lhs, left] = right
			if cell:
				spans += 1
				items += len(cell)
				edges += sum(map(len, cell.values()))
	msg = "chart spans %d, items %d, edges %d" % (spans, items, edges)
	if chart[0][lensent].get(start):
		return chart, new_CFGChartItem(start, 0, lensent), msg
	else:
		return chart, NONE, "no parse " + msg


def parse_sparse(list sent, Grammar grammar, start=1, tags=None,
		list chart=None, int beamwidth=0):
	""" A CKY parser modeled after Bodenstab's `fast grammar loop,' filtered by
	the list of allowed items (if a pre-populated chart is given).
	This version keeps the Viterbi probabilities and the rest of chart in
	hash tables, useful for large grammars. The edge with the viterbi score
	for a labeled span is kept in viterbi[left][right][label]. """
	cdef:
		short left, right, mid, span, lensent = len(sent)
		short narrowl, narrowr, widel, wider, minmid, maxmid
		UInt n, lhs, rhs1, spans = 0, items = 0, edges = 0
		double oldscore, prob, infinity = float('infinity')
		Rule *rule
		LexicalRule lexrule
		EdgeAgenda unaryagenda = EdgeAgenda()
		list viterbi = [[{} for _ in range(lensent + 1)]
				for _ in range(lensent)]
		dict cell, viterbicell
		# matrices for the filter which gives minima and maxima for splits
		np.ndarray[np.int16_t, ndim=2] minleft, maxleft, minright, maxright
	assert grammar.maxfanout == 1, "Not a PCFG! fanout = %d" % grammar.maxfanout
	assert grammar.logprob, "Expecting grammar with log probabilities."
	minleft = np.empty((grammar.nonterminals, lensent + 1), dtype='int16')
	maxleft = np.empty_like(minleft)
	minright = np.empty_like(minleft)
	maxright = np.empty_like(minleft)
	minleft.fill(-1)
	maxleft.fill(lensent + 1)
	minright.fill(lensent + 1)
	maxright.fill(-1)
	if chart is None:
		chart = [[None] * (lensent + 1) for _ in range(lensent)]
		cell = dict.fromkeys(range(1, grammar.nonterminals))
		for left in range(lensent):
			for right in range(left, lensent):
				chart[left][right + 1] = cell.copy()
	# assign POS tags
	for left, word in enumerate(sent):
		tag = tags[left].encode('ascii') if tags else None
		right = left + 1
		viterbicell = viterbi[left][right]
		cell = chart[left][right]
		recognized = False
		for lexrule in <list>grammar.lexicalbyword.get(word, ()):
			if lexrule.lhs not in cell:
				continue
			lhs = lexrule.lhs
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if (not tags or grammar.tolabel[lhs] == tag
				or grammar.tolabel[lhs].startswith(tag + b'@')):
				edge = new_CFGEdge(lexrule.prob, NULL, right)
				viterbicell[lhs] = edge
				cell[lhs] = {edge: edge}
				recognized = True
				# update filter
				if left > minleft[lhs, right]:
					minleft[lhs, right] = left
				if left < maxleft[lhs, right]:
					maxleft[lhs, right] = left
				if right < minright[lhs, left]:
					minright[lhs, left] = right
				if right > maxright[lhs, left]:
					maxright[lhs, left] = right
		# NB: don't allow blocking of gold tags if given
		if not recognized:
			if tags and tag in grammar.toid:
				lhs = grammar.toid[tag]
				edge = new_CFGEdge(0.0, NULL, right)
				viterbicell[lhs] = edge
				cell[lhs] = {edge: edge}
				# update filter
				if left > minleft[lhs, right]:
					minleft[lhs, right] = left
				if left < maxleft[lhs, right]:
					maxleft[lhs, right] = left
				if right < minright[lhs, left]:
					minright[lhs, left] = right
				if right > maxright[lhs, left]:
					maxright[lhs, left] = right
			else:
				return chart, NONE, "not covered: %r" % (tag or word, )
		# unary rules on the span of this POS tag
		# NB: for this agenda, only the probabilities of the edges matter
		unaryagenda.update(viterbicell.items())
		while unaryagenda.length:
			rhs1 = unaryagenda.popentry().key
			for n in range(grammar.numrules):
				rule = &(grammar.unary[rhs1][n])
				if rule.rhs1 != rhs1:
					break
				elif rule.lhs not in cell:
					continue
				lhs = rule.lhs
				prob = rule.prob + (<CFGEdge>viterbicell[rhs1]).inside
				edge = new_CFGEdge(prob, rule, right)
				if cell[lhs]:
					if prob < (<CFGEdge>viterbicell[lhs]).inside:
						unaryagenda.setifbetter(lhs, <CFGEdge>edge)
						viterbi[left][right][lhs] = edge
						cell[lhs][edge] = edge
					elif (edge not in cell[lhs] or
							prob < (<CFGEdge>cell[lhs][edge]).inside):
						cell[lhs][edge] = edge
					continue
				unaryagenda.setifbetter(lhs, <CFGEdge>edge)
				viterbicell[lhs] = edge
				cell[lhs] = {edge: edge}
				# update filter
				if left > minleft[lhs, right]:
					minleft[lhs, right] = left
				if left < maxleft[lhs, right]:
					maxleft[lhs, right] = left
				if right < minright[lhs, left]:
					minright[lhs, left] = right
				if right > maxright[lhs, left]:
					maxright[lhs, left] = right

	for span in range(2, lensent + 1):
		# constituents from left to right
		for left in range(lensent - span + 1):
			right = left + span
			viterbicell = viterbi[left][right]
			cell = chart[left][right]
			# binary rules
			for lhs in cell:
				n = 0
				rule = &(grammar.bylhs[lhs][n])
				oldscore = ((<CFGEdge>viterbicell[lhs]).inside
						if lhs in viterbicell else infinity)
				while rule.lhs == lhs:
					narrowr = minright[rule.rhs1, left]
					narrowl = minleft[rule.rhs2, right]
					if rule.rhs2 == 0 or narrowr >= right or narrowl < narrowr:
						n +=1
						rule = &(grammar.bylhs[lhs][n])
						continue
					widel = maxleft[rule.rhs2, right]
					minmid = narrowr if narrowr > widel else widel
					wider = maxright[rule.rhs1, left]
					maxmid = wider if wider < narrowl else narrowl
					if lhs not in viterbicell:
						cell[lhs] = {}
					for mid in range(minmid, maxmid + 1):
						if (rule.rhs1 in viterbi[left][mid]
								and rule.rhs2 in viterbi[mid][right]):
							prob = (rule.prob + (<CFGEdge>viterbi[
								left][mid][rule.rhs1]).inside + (<CFGEdge>
								viterbi[mid][right][rule.rhs2]).inside)
							edge = new_CFGEdge(prob, rule, mid)
							if (lhs not in viterbicell or
									prob < (<CFGEdge>viterbicell[lhs]).inside):
								viterbicell[lhs] = edge
								cell[lhs][edge] = edge
							elif edge not in cell[lhs]:
								cell[lhs][edge] = edge
					n +=1
					rule = &(grammar.bylhs[lhs][n])

				# update filter
				if isinf(oldscore):
					if not cell[lhs]:
						del cell[lhs]
						continue
					if left > minleft[lhs, right]:
						minleft[lhs, right] = left
					if left < maxleft[lhs, right]:
						maxleft[lhs, right] = left
					if right < minright[lhs, left]:
						minright[lhs, left] = right
					if right > maxright[lhs, left]:
						maxright[lhs, left] = right

			# unary rules
			unaryagenda.update(viterbicell.items())
			while unaryagenda.length:
				rhs1 = unaryagenda.popentry().key
				for n in range(grammar.numrules):
					rule = &(grammar.unary[rhs1][n])
					if rule.rhs1 != rhs1:
						break
					elif rule.lhs not in cell:
						continue
					lhs = rule.lhs
					prob = rule.prob + (<CFGEdge>
							viterbicell[rhs1]).inside
					edge = new_CFGEdge(prob, rule, right)
					if cell[lhs]:
						if prob < (<CFGEdge>viterbicell[lhs]).inside:
							unaryagenda.setifbetter(lhs, <CFGEdge>edge)
							viterbicell[lhs] = edge
							cell[lhs][edge] = edge
						elif (edge not in cell[lhs] or
								prob < (<CFGEdge>cell[lhs][edge]).inside):
							cell[lhs][edge] = edge
						continue
					unaryagenda.setifbetter(lhs, <CFGEdge>edge)
					viterbicell[lhs] = edge
					cell[lhs] = {edge: edge}
					# update filter
					if left > minleft[lhs, right]:
						minleft[lhs, right] = left
					if left < maxleft[lhs, right]:
						maxleft[lhs, right] = left
					if right < minright[lhs, left]:
						minright[lhs, left] = right
					if right > maxright[lhs, left]:
						maxright[lhs, left] = right
			nonempty = filter(None, cell.values())
			if nonempty:
				spans += 1
				items += len(nonempty)
				# clean up labelled spans that were whitelisted but not used:
				#for label in cell:
				#	if cell[label] is None:
				#		del cell[label]
				edges += sum(map(len, nonempty))
	msg = "chart spans %d, items %d, edges %d" % (spans, items, edges)
	if chart[0][lensent].get(start):
		return chart, new_CFGChartItem(start, 0, lensent), msg
	else:
		return chart, NONE, "no parse " + msg


def symbolicparse(sent, Grammar grammar, start=1, tags=None):
	""" Parse sentence, a list of tokens, and produce a chart, either
	exhaustive or up until the first complete parse. Non-probabilistic version;
	returns a CFG chart with each 'probability' 1. Other parameters:
		- start: integer corresponding to the start symbol that analyses should
			have, e.g., grammar.toid['ROOT']
		- tags: optionally, a list with the corresponding POS tags
			for the words in sent. """
	cdef:
		short left, right, mid, span, lensent = len(sent)
		short narrowl, narrowr, widel, wider, minmid, maxmid
		UInt n, lhs, rhs1, spans = 0, items = 0, edges = 0
		bint newspan
		Rule *rule
		LexicalRule lexrule
		set unaryagenda = set()
		list chart = [[{} for _ in range(lensent + 1)] for _ in range(lensent)]
		dict cell
		# matrices for the filter which gives minima and maxima for splits
		np.ndarray[np.int16_t, ndim=2] minleft, maxleft, minright, maxright
	assert grammar.maxfanout == 1, "Not a PCFG! fanout = %d" % grammar.maxfanout
	minleft = np.empty((grammar.nonterminals, lensent + 1), dtype='int16')
	maxleft = np.empty_like(minleft)
	minright = np.empty_like(minleft)
	maxright = np.empty_like(minleft)
	minleft.fill(-1)
	maxleft.fill(lensent + 1)
	minright.fill(lensent + 1)
	maxright.fill(-1)
	# assign POS tags
	for left, word in enumerate(sent):
		tag = tags[left].encode('ascii') if tags else None
		right = left + 1
		cell = chart[left][right]
		recognized = False
		for lexrule in grammar.lexicalbyword.get(word, ()):
			lhs = lexrule.lhs
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if (not tags or grammar.tolabel[lhs] == tag
				or grammar.tolabel[lhs].startswith(tag + b'@')):
				edge = new_CFGEdge(0.0, NULL, right)
				cell[lhs] = {edge: edge}
				recognized = True
				# update filter
				if left > minleft[lhs, right]:
					minleft[lhs, right] = left
				if left < maxleft[lhs, right]:
					maxleft[lhs, right] = left
				if right < minright[lhs, left]:
					minright[lhs, left] = right
				if right > maxright[lhs, left]:
					maxright[lhs, left] = right
		if not recognized:
			if tags and tag in grammar.toid:
				lhs = grammar.toid[tag]
				edge = new_CFGEdge(0.0, NULL, right)
				cell[lhs] = {edge: edge}
				# update filter
				if left > minleft[lhs, right]:
					minleft[lhs, right] = left
				if left < maxleft[lhs, right]:
					maxleft[lhs, right] = left
				if right < minright[lhs, left]:
					minright[lhs, left] = right
				if right > maxright[lhs, left]:
					maxright[lhs, left] = right
			else:
				return chart, NONE, "not covered: %r" % (tag or word, )
		# unary rules on the span of this POS tag
		unaryagenda.update(cell)
		while unaryagenda:
			rhs1 = unaryagenda.pop()
			for n in range(grammar.numrules):
				rule = &(grammar.unary[rhs1][n])
				if rule.rhs1 != rhs1:
					break
				lhs = rule.lhs
				edge = new_CFGEdge(0.0, rule, right)
				if lhs in cell:
					cell[lhs][edge] = edge
					continue
				unaryagenda.add(lhs)
				cell[lhs] = {edge: edge}
				# update filter
				if left > minleft[lhs, right]:
					minleft[lhs, right] = left
				if left < maxleft[lhs, right]:
					maxleft[lhs, right] = left
				if right < minright[lhs, left]:
					minright[lhs, left] = right
				if right > maxright[lhs, left]:
					maxright[lhs, left] = right

	for span in range(2, lensent + 1):
		# constituents from left to right
		for left in range(lensent - span + 1):
			right = left + span
			cell = chart[left][right]
			# binary rules
			for n in range(grammar.numrules):
				rule = &(grammar.bylhs[0][n])
				if rule.lhs == grammar.nonterminals:
					break
				elif not rule.rhs2:
					continue
				lhs = rule.lhs
				narrowr = minright[rule.rhs1, left]
				if narrowr >= right:
					continue
				narrowl = minleft[rule.rhs2, right]
				if narrowl < narrowr:
					continue
				widel = maxleft[rule.rhs2, right]
				minmid = narrowr if narrowr > widel else widel
				wider = maxright[rule.rhs1, left]
				maxmid = wider if wider < narrowl else narrowl
				newspan = lhs not in cell
				if newspan:
					cell[lhs] = {}
				for mid in range(minmid, maxmid + 1):
					if (rule.rhs1 in chart[left][mid]
							and rule.rhs2 in chart[mid][right]):
						edge = new_CFGEdge(0.0, rule, mid)
						cell[lhs][edge] = edge

				# update filter
				if newspan:
					if not cell[lhs]:
						del cell[lhs]
						continue
					if left > minleft[lhs, right]:
						minleft[lhs, right] = left
					if left < maxleft[lhs, right]:
						maxleft[lhs, right] = left
					if right < minright[lhs, left]:
						minright[lhs, left] = right
					if right > maxright[lhs, left]:
						maxright[lhs, left] = right

			# unary rules
			unaryagenda.update(cell)
			while unaryagenda:
				rhs1 = unaryagenda.pop()
				for n in range(grammar.numrules):
					rule = &(grammar.unary[rhs1][n])
					if rule.rhs1 != rhs1:
						break
					lhs = rule.lhs
					edge = new_CFGEdge(0.0, rule, right)
					if lhs in cell:
						cell[lhs][edge] = edge
						continue
					unaryagenda.add(lhs)
					cell[lhs] = {edge: edge}
					# update filter
					if left > minleft[lhs, right]:
						minleft[lhs, right] = left
					if left < maxleft[lhs, right]:
						maxleft[lhs, right] = left
					if right < minright[lhs, left]:
						minright[lhs, left] = right
					if right > maxright[lhs, left]:
						maxright[lhs, left] = right
			nonempty = filter(None, cell.values())
			if nonempty:
				spans += 1
				items += len(nonempty)
				edges += sum(map(len, nonempty))
	msg = "chart spans %d, items %d, edges %d" % (spans, items, edges)
	if chart[0][lensent].get(start):
		return chart, new_CFGChartItem(start, 0, lensent), msg
	else:
		return chart, NONE, "no parse " + msg


def doinsideoutside(list sent, Grammar grammar, inside=None, outside=None,
		tags=None):
	assert grammar.maxfanout == 1, "Not a PCFG! fanout = %d" % grammar.maxfanout
	assert not grammar.logprob, "Grammar must not have log probabilities."
	lensent = len(sent)
	if inside is None:
		inside = np.zeros((lensent, lensent + 1,
				grammar.nonterminals), dtype='d')
	else:
		inside[:len(sent), :len(sent) + 1, :] = 0.0
	if outside is None:
		outside = np.zeros((lensent, lensent + 1,
				grammar.nonterminals), dtype='d')
	else:
		outside[:len(sent), :len(sent) + 1, :] = 0.0
	minmaxlr = insidescores(sent, grammar, inside, tags)
	if inside[0, len(sent), 1]:
		outside = outsidescores(grammar, sent, inside, outside, *minmaxlr)
		start = new_CFGChartItem(1, 0, lensent)
		msg = ""
	else:
		start = NONE
		msg = "no parse"
	return inside, outside, start, msg


def insidescores(list sent, Grammar grammar,
		np.ndarray[np.double_t, ndim=3] inside, tags=None):
	""" Compute inside scores. These are not viterbi scores, but sums of
	all derivations headed by a certain nonterminal + span. """
	cdef:
		short left, right, mid, span, lensent = len(sent)
		short narrowl, narrowr, minmid, maxmid
		double oldscore, prob, ls, rs, ins
		UInt n, lhs, rhs1
		bint foundbetter = False
		Rule *rule
		LexicalRule lexrule
		unicode word
		list cell = [{} for _ in grammar.toid]
		EdgeAgenda unaryagenda = EdgeAgenda()
		# matrices for the filter which give minima and maxima for splits
		np.ndarray[np.int16_t, ndim=2] minleft, maxleft, minright, maxright
		np.ndarray[np.double_t, ndim=1] unaryscores = np.empty((
				grammar.nonterminals), dtype='double')
	minleft = np.empty((grammar.nonterminals, lensent + 1), dtype='int16')
	maxleft = np.empty_like(minleft)
	minright = np.empty_like(minleft)
	maxright = np.empty_like(minleft)
	maxleft.fill(lensent + 1)
	minright.fill(lensent + 1)
	minleft.fill(-1)
	maxright.fill(-1)
	inside[:lensent, :lensent + 1, :] = 0.0
	# assign POS tags
	for left in range(lensent):
		tag = tags[left].encode('ascii') if tags else None
		right = left + 1
		for lexrule in grammar.lexicalbyword.get(sent[left], []):
			lhs = lexrule.lhs
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if not tags or (grammar.tolabel[lhs] == tag
					or grammar.tolabel[lhs].startswith(tag + b'@')):
				inside[left, right, lhs] = lexrule.prob
		if not inside[left, right].any():
			if tags is not None:
				lhs = grammar.toid[tag]
				if not tags or (grammar.tolabel[lhs] == tag
						or grammar.tolabel[lhs].startswith(tag + b'@')):
					inside[left, right, lhs] = 1.
			else:
				raise ValueError("not covered: %r" % (tag or sent[left]), )
		# unary rules on POS tags
		unaryagenda.update([(rhs1,
			new_CFGEdge(-inside[left, right, rhs1], NULL, 0))
			for rhs1 in range(grammar.nonterminals)
			if inside[left, right, rhs1]
			and grammar.unary[rhs1].rhs1 == rhs1])
		unaryscores.fill(0.0)
		while unaryagenda.length:
			rhs1 = unaryagenda.popentry().key
			for n in range(grammar.numrules):
				rule = &(grammar.unary[rhs1][n])
				if rule.rhs1 != rhs1:
					break
				prob = rule.prob * inside[left, right, rhs1]
				lhs = rule.lhs
				edge = new_CFGEdge(-prob, rule, right)
				if edge not in cell[lhs]:
					unaryagenda.setifbetter(lhs, <CFGEdge>edge)
					inside[left, right, lhs] += prob
					cell[lhs][edge] = edge
		for a in cell:
			a.clear()
		for lhs in range(grammar.nonterminals):
			if inside[left, right, lhs]:
				# update filter
				if left > minleft[lhs, right]:
					minleft[lhs, right] = left
				if left < maxleft[lhs, right]:
					maxleft[lhs, right] = left
				if right < minright[lhs, left]:
					minright[lhs, left] = right
				if right > maxright[lhs, left]:
					maxright[lhs, left] = right
	for span in range(2, lensent + 1):
		# constituents from left to right
		for left in range(lensent - span + 1):
			right = left + span
			# binary rules
			for n in range(grammar.numrules):
				rule = &(grammar.bylhs[0][n])
				lhs = rule.lhs
				if lhs == grammar.nonterminals:
					break
				elif not rule.rhs2:
					continue
				narrowr = minright[rule.rhs1, left]
				narrowl = minleft[rule.rhs2, right]
				if narrowr >= right or narrowl < narrowr:
					continue
				widel = maxleft[rule.rhs2, right]
				minmid = narrowr if narrowr > widel else widel
				wider = maxright[rule.rhs1, left]
				maxmid = wider if wider < narrowl else narrowl
				#oldscore = inside[left, right, lhs]
				foundbetter = False
				for split in range(minmid, maxmid + 1):
					ls = inside[left, split, rule.rhs1]
					if ls == 0.0:
						continue
					rs = inside[split, right, rule.rhs2]
					if rs == 0.0:
						continue
					foundbetter = True
					inside[left, right, lhs] += rule.prob * ls * rs
					assert 0.0 < inside[left, right, lhs] <= 1.0, (
						inside[left, right, lhs],
						left, right, grammar.tolabel[lhs])
				if foundbetter:  # and oldscore == 0.0:
					if left > minleft[lhs, right]:
						minleft[lhs, right] = left
					if left < maxleft[lhs, right]:
						maxleft[lhs, right] = left
					if right < minright[lhs, left]:
						minright[lhs, left] = right
					if right > maxright[lhs, left]:
						maxright[lhs, left] = right
			# unary rules on this span
			unaryagenda.update([(rhs1,
				new_CFGEdge(-inside[left, right, rhs1], NULL, 0))
				for rhs1 in range(grammar.nonterminals)
				if inside[left, right, rhs1]
				and grammar.unary[rhs1].rhs1 == rhs1])
			unaryscores.fill(0.0)
			while unaryagenda.length:
				rhs1 = unaryagenda.popentry().key
				for n in range(grammar.numrules):
					rule = &(grammar.unary[rhs1][n])
					if rule.rhs1 != rhs1:
						break
					prob = rule.prob * inside[left, right, rhs1]
					lhs = rule.lhs
					edge = new_CFGEdge(-prob, rule, right)
					if edge not in cell[lhs]:
						unaryagenda.setifbetter(lhs, <CFGEdge>edge)
						inside[left, right, lhs] += prob
						cell[lhs][edge] = edge
			for a in cell:
				a.clear()
			for lhs in range(grammar.nonterminals):
				# update filter
				if inside[left, right, lhs]:
					if left > minleft[lhs, right]:
						minleft[lhs, right] = left
					if left < maxleft[lhs, right]:
						maxleft[lhs, right] = left
					if right < minright[lhs, left]:
						minright[lhs, left] = right
					if right > maxright[lhs, left]:
						maxright[lhs, left] = right
	return minleft, maxleft, minright, maxright


def outsidescores(Grammar grammar, list sent,
		np.ndarray[np.double_t, ndim=3] inside,
		np.ndarray[np.double_t, ndim=3] outside,
		np.ndarray[np.int16_t, ndim=2] minleft,
		np.ndarray[np.int16_t, ndim=2] maxleft,
		np.ndarray[np.int16_t, ndim=2] minright,
		np.ndarray[np.int16_t, ndim=2] maxright):
	cdef:
		short left, right, mid, span, lensent = len(sent)
		short narrowl, narrowr, minmid, maxmid
		double ls, rs, os
		UInt n, lhs, rhs1, rhs2
		bint foundbetter = False
		Rule *rule
		LexicalRule lexrule
		EdgeAgenda unaryagenda = EdgeAgenda()
		list cell = [{} for _ in grammar.toid]
		np.ndarray[np.double_t, ndim=1] unaryscores = np.empty((
			grammar.nonterminals), dtype='double')
	outside[0, lensent, 1] = 1.0
	for span in range(lensent, 0, -1):
		for left in range(1 + lensent - span):
			right = left + span
			# unary rules
			unaryagenda.update([(lhs,
				new_CFGEdge(-outside[left, right, lhs], NULL, 0))
				for lhs in range(grammar.nonterminals)
				if outside[left, right, lhs]])
			unaryscores.fill(0.0)
			while unaryagenda.length:
				lhs = unaryagenda.popentry().key
				for n in range(grammar.numrules):
					rule = &(grammar.bylhs[lhs][n])
					if rule.lhs != lhs:
						break
					elif rule.rhs2:
						continue
					prob = rule.prob * outside[left, right, lhs]
					edge = new_CFGEdge(-prob, rule, right)
					if edge not in cell[lhs]:
						unaryagenda.setifbetter(rule.rhs1, <CFGEdge>edge)
						cell[lhs][edge] = edge
						outside[left, right, rule.rhs1] += prob
			for rhs1 in range(grammar.nonterminals):
				cell[rhs1].clear()
			# binary rules
			for n in range(grammar.numrules):
				rule = &(grammar.bylhs[0][n])
				lhs = rule.lhs
				if lhs == grammar.nonterminals:
					break
				elif not rule.rhs2 or outside[left, right, lhs] == 0.0:
					continue
				os = outside[left, right, lhs]
				narrowr = minright[rule.rhs1, left]
				narrowl = minleft[rule.rhs2, right]
				if narrowr >= right or narrowl < narrowr:
					continue
				widel = maxleft[rule.rhs2, right]
				minmid = narrowr if narrowr > widel else widel
				wider = maxright[rule.rhs1, left]
				maxmid = wider if wider < narrowl else narrowl
				for split in range(minmid, maxmid + 1):
					ls = inside[left, split, rule.rhs1]
					if ls == 0.0:
						continue
					rs = inside[split, right, rule.rhs2]
					if rs == 0.0:
						continue
					outside[left, split, rule.rhs1] += rule.prob * rs * os
					outside[split, right, rule.rhs2] += rule.prob * ls * os
					assert 0.0 < outside[left, split, rule.rhs1] <= 1.0
					assert 0.0 < outside[split, right, rule.rhs2] <= 1.0
	return outside


def dopparseprob(tree, Grammar grammar, dict rulemapping, lexchart):
	""" Given an NLTK tree, compute the exact DOP parse probability given
	a DOP reduction.

	This follows up on a suggestion made by Goodman (2003, p. 20)
	of calculating DOP probabilities of given parse trees, although I'm not
	sure it has complexity O(nP) as he suggests (with n as number of nodes in
	input, and P as max number of rules consistent with a node in the input).
	Furthermore, the idea of sampling trees "long enough" until we have the MPP
	is no faster than sampling without applying this procedure, because to
	determine that some probability p is the maximal probability, we need to
	collect the probability mass p_seen of enough parse trees such that we have
	some parsetree with probability p > (1 - p_seen), which requires first
	seeing almost all parse trees, unless p is exceptionally high. Hence, this
	method is mostly useful in a reranking framework where it is known in
	advance that a small set of trees is of interest.

	expects a mapping which gives a list of consistent rules from the reduction
	(as Rule objects) given a rule as key (as a tuple of strings); e.g. ('NP',
	'DT', 'NN') -> [Rule(...), Rule(...), ...]

	NB: this algorithm could also be used to determine the probability of
	derivations, but then the input would have to distinguish whether nodes are
	internal nodes of fragments, or whether they join two fragments. """
	neginf = float('-inf')
	cdef dict chart = {}  # chart[left, right][label]
	cdef tuple a, b, c
	cdef Rule *rule
	assert grammar.maxfanout == 1
	assert grammar.logprob, "Grammar should have log probabilities."
	# log probabilities are not ideal here because we do lots of additions,
	# but the probabilities are very small. a possible alternative is to scale
	# them somehow.

	# add all possible POS tags
	chart.update(lexchart)
	for n, word in enumerate(tree.leaves()):
		# replace leaves with indices so as to easily find spans
		tree[tree.leaf_treeposition(n)] = n

	# do post-order traversal (bottom-up)
	for node in list(tree.subtrees())[::-1]:
		if not isinstance(node[0], Tree):
			continue
		prod = (node.node,) + tuple(a.node for a in node)
		left = min(node.leaves())
		right = max(node.leaves()) + 1
		if len(node) == 1:  # unary node
			for ruleno in rulemapping[prod]:
				rule = grammar.bylhs[ruleno]
				b = (rule.rhs1, left, right)
				if b in chart:
					a = (rule.lhs, left, right)
					if a in chart:
						chart[a] = logprobadd(chart[a], -rule.prob + chart[b])
					else:
						chart[a] = (-rule.prob + chart[b])
		elif len(node) == 2:  # binary node
			split = min(node[1].leaves())
			for ruleno in rulemapping[prod]:
				rule = grammar.bylhs[ruleno]
				b = (rule.rhs1, left, split)
				c = (rule.rhs2, split, right)
				if b in chart and c in chart:
					a = (rule.lhs, left, right)
					if a in chart:
						chart[a] = logprobadd(chart[a],
							(-rule.prob + chart[b] + chart[c]))
					else:
						chart[a] = -rule.prob + chart[b] + chart[c]
		else:
			raise ValueError("expected binary tree.")
	return chart.get((grammar.toid[tree.node], 0, len(tree.leaves())), neginf)


def doplexprobs(tree, Grammar grammar):
	neginf = float('-inf')
	cdef dict chart = <dict>defaultdict(lambda: neginf)
	cdef LexicalRule lexrule

	for n, word in enumerate(tree.leaves()):
		for lexrule in grammar.lexicalbyword[word]:
			chart[lexrule.lhs, n, n + 1] = logprobadd(
				chart[lexrule.lhs, n, n + 1], -lexrule.prob)
	return chart


def getgrammarmapping(Grammar coarse, Grammar fine):
	""" producing a mapping of coarse rules to sets of fine rules;
	e.g. mapping["S", "NP", "VP"] == set(Rule(...), Rule(...), ...) """
	mapping = {}
	cdef Rule *rule
	for ruleno in range(coarse.numrules):
		rule = &(coarse.bylhs[0][ruleno])
		if rule.rhs2:
			mapping[coarse.tolabel[rule.lhs],
				coarse.tolabel[rule.rhs1], coarse.tolabel[rule.rhs2]] = []
		else:
			mapping[coarse.tolabel[rule.lhs], coarse.tolabel[rule.rhs1]] = []
	for ruleno in range(fine.numrules):
		rule = &(fine.bylhs[0][ruleno])
		if rule.rhs2:
			mapping[fine.tolabel[rule.lhs].rsplit("@", 1)[0],
				fine.tolabel[rule.rhs1].rsplit("@", 1)[0],
				fine.tolabel[rule.rhs2].rsplit("@", 1)[0]].append(ruleno)
		else:
			mapping[fine.tolabel[rule.lhs].rsplit("@", 1)[0],
				fine.tolabel[rule.rhs1].rsplit("@", 1)[0]].append(ruleno)
	return mapping


UNESCAPE = re.compile(r"\\([#{}\[\]<>\^$'])")


def parse_bitpar(rulesfile, lexiconfile, sent, n, startlabel, tags=None):
	""" Parse a single sentence with bitpar, given filenames of rules and
	lexicon. n is the number of derivations to ask for (max 1000).
	Result is a dictionary of derivations with their probabilities. """
	# TODO: get full viterbi parse forest, turn into chart w/ChartItems
	assert 1 <= n <= 1000
	if tags:
		_, lexiconfile = tempfile.mkstemp()
		with open(lexiconfile, 'w') as f:
			f.writelines(['%s\t%s 1\n' % (t, t) for t in set(tags)])
	proc = Popen(['bitpar', '-q', '-vp', '-b', str(n), '-s', startlabel,
			rulesfile, lexiconfile],
			shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE)
	results, msg = proc.communicate('\n'.join([
			word.replace('(', '-LRB-').replace(')', '-RRB-').encode('utf8')
			for word in (tags or sent)]) + '\n')
	results = results.strip('\t\n ')  # decode or not?
	if tags:
		unlink(lexiconfile)
	if not results or results.startswith("No parse"):
		return {}, None, '%s\n%s' % (results, msg)
	start = new_CFGChartItem(1, 0, len(sent))
	lines = UNESCAPE.sub(r'\1', results).replace(')(', ') (').splitlines()
	return {renumber(deriv): -log(float(prob[prob.index('=') + 1:]))
			for prob, deriv in zip(lines[::2], lines[1::2])}, start, msg


def renumber(deriv):
	""" Replace terminals of CF-derivation (string) with indices. """
	it = count()

	def closure(match):
		return ' %s)' % next(it)

	return re.sub(r' [^ )]+\)', closure, deriv)


def sortfunc(CFGEdge e):
	return e.inside


def pprint_chart(chart, sent, tolabel):
	cdef CFGEdge edge
	print("chart:")
	for left, _ in enumerate(chart):
		for right, _ in enumerate(chart[left]):
			for label in chart[left][right] or ():
				#if not chart[left][right][label]:
				#	continue
				print("%s[%d:%d] =>" % (
						tolabel[label].decode('ascii'), left, right))
				for edge in sorted(chart[left][right][label] or (),
						key=sortfunc):
					if edge.rule is NULL:
						print("%9.7f  %9.7f  '%s'" % (exp(-edge.inside), 1,
								sent[edge.mid - 1]), end='')
					else:
						print("%9.7f  %9.7f " % (exp(-edge.inside),
								exp(-edge.rule.prob)), "%s[%d:%d]" % (
									tolabel[edge.rule.rhs1].decode('ascii'),
								left, edge.mid), end='')
						if edge.rule.rhs2:
							print("\t%s[%d:%d]" % (
									tolabel[edge.rule.rhs2].decode('ascii'),
									edge.mid, right), end='')
					print()
				print()


def pprint_matrix(matrix, sent, tolabel, matrix2=None):
	""" Print a chart in a numpy matrix; optionally in parallel with another
	matrix. """
	for span in range(1, len(sent) + 1):
		for left in range(len(sent) - span + 1):
			right = left + span
			if matrix[left, right].any() or (
					matrix2 is not None and matrix2[left, right].any()):
				print("[%d:%d]" % (left, right))
				for lhs in range(len(matrix[left, right])):
					if matrix[left, right, lhs] or (
							matrix2 is not None and matrix2[left, right, lhs]):
						print("%20s\t%8.6g" % (tolabel[lhs].decode('ascii'),
								matrix[left, right, lhs]), end='')
						if matrix2 is not None:
							print("\t%8.6g" % matrix2[left, right, lhs], end='')
						print()


def main():
	from containers import Grammar
	cdef Rule rule
	cfg = Grammar([
		((('S', 'D'), ((0, ), )), 0.5),
		((('S', 'A'), ((0, ), )), 0.8),
		((('A', 'A'), ((0, ), )), 0.7),
		((('A', 'B'), ((0, ), )), 0.6),
		((('A', 'C'), ((0, ), )), 0.5),
		((('A', 'D'), ((0, ), )), 0.4),
		((('B', 'A'), ((0, ), )), 0.3),
		((('B', 'B'), ((0, ), )), 0.2),
		((('B', 'C'), ((0, ), )), 0.1),
		((('B', 'D'), ((0, ), )), 0.2),
		((('B', 'C'), ((0, ), )), 0.3),
		((('C', 'A'), ((0, ), )), 0.4),
		((('C', 'B'), ((0, ), )), 0.5),
		((('C', 'C'), ((0, ), )), 0.6),
		((('C', 'D'), ((0, ), )), 0.7),
		((('D', 'A'), ((0, ), )), 0.8),
		((('D', 'B'), ((0, ), )), 0.9),
		((('D', 'C'), ((0, ), )), 0.8),
		((('D', 'NP', 'VP'), ((0, 1), )), 1),
		((('NP', 'Epsilon'), ('mary', )), 1),
		((('VP', 'Epsilon'), ('walks', )), 1)],
		start='S')
	print(cfg)
	print("cfg parsing; sentence: mary walks")
	print("pcfg", end='')
	chart, start, _ = parse("mary walks".split(), cfg)
	assert start
	pprint_chart(chart, "mary walks".split(), cfg.tolabel)
	cfg1 = Grammar([
		((('S', 'NP', 'VP'), ((0, 1), )), 1),
		((('NP', 'Epsilon'), ('mary', )), 1),
		((('VP', 'Epsilon'), ('walks', )), 1)], start='S')
	cfg1.switch('default', False)
	i, o, start, _ = doinsideoutside("mary walks".split(), cfg1)
	assert start
	print(i[0, 2, cfg1.toid[b'S']], o[0, 2, cfg1.toid[b'S']])
	i, o, start, _ = doinsideoutside("walks mary".split(), cfg1)
	assert not start
	print(i[0, 2, cfg1.toid[b'S']], o[0, 2, cfg1.toid[b'S']])
	rules = [
		((('S', 'NP', 'VP'), ((0, 1), )), 1),
		((('PP', 'P', 'NP'), ((0, 1), )), 1),
		((('VP', 'V', 'NP'), ((0, 1), )), 0.7),
		((('VP', 'VP', 'PP'), ((0, 1), )), 0.3),
		((('NP', 'NP', 'PP'), ((0, 1), )), 0.4),
		((('P', 'Epsilon'), ('with', )), 1),
		((('V', 'Epsilon'), ('saw', )), 1),
		((('NP', 'Epsilon'), ('astronomers', )), 0.1),
		((('NP', 'Epsilon'), ('ears', )), 0.18),
		((('NP', 'Epsilon'), ('saw', )), 0.04),
		((('NP', 'Epsilon'), ('stars', )), 0.18),
		((('NP', 'Epsilon'), ('telescopes', )), 0.1)]
	cfg2 = Grammar(rules, start='S')
	cfg2.switch('default', False)
	sent = "astronomers saw stars with ears".split()
	inside, outside, _, msg = doinsideoutside(sent, cfg2)
	print(msg)
	pprint_matrix(inside, sent, cfg2.tolabel, outside)

	cfg2.switch('default', True)
	chart, start, msg = parse(sent, cfg2)
	from disambiguation import marginalize
	from operator import itemgetter
	mpp, _, _ = marginalize('mpp', chart, start, cfg2, 10)
	for a, p in sorted(mpp.items(), key=itemgetter(1), reverse=True):
		print(p, a)
	chart1, start1, msg1 = symbolicparse(sent, cfg2)
	print(msg)
	print(msg1)
