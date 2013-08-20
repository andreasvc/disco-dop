""" Probabilistic Context-Free Grammar (PCFG) parser using CKY. """
# python imports
from __future__ import print_function
from math import exp, log as pylog
from collections import defaultdict
import numpy as np
from discodop.tree import Tree
from discodop.agenda import EdgeAgenda
# cython imports
from libc.stdlib cimport malloc, calloc, free
from cpython cimport PyDict_Contains, PyDict_GetItem
from discodop.agenda cimport EdgeAgenda
from discodop._grammar cimport Grammar
from discodop.containers cimport CFGEdge, CFGChartItem, new_CFGChartItem, \
		new_CFGEdge, Rule, LexicalRule, UChar, UInt, ULong, ULLong

cdef extern from "math.h":
	bint isinf(double x)
	bint isfinite(double x)

DEF SX = 1
DEF SXlrgaps = 2
cdef CFGChartItem NONE = new_CFGChartItem(0, 0, 0)


def parse(sent, Grammar grammar, tags=None, start=None, chart=None):
	""" Parse a sentence with a PCFG. Automatically decides whether to use
	parse_dense or parse_sparse. """
	assert grammar.maxfanout == 1, 'Not a PCFG! fanout: %d' % grammar.maxfanout
	assert grammar.logprob, "Expecting grammar with log probabilities."
	if grammar.nonterminals < 20000 and chart is None:
		return parse_dense(sent, grammar, start=start, tags=tags)
	return parse_sparse(sent, grammar, start=start, tags=tags, chart=chart)


def parse_dense(sent, Grammar grammar, start=None, tags=None):
	""" A CKY parser modeled after Bodenstab's 'fast grammar loop'
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
		short [:, :] minleft, maxleft, minright, maxright
		double [:, :, :] viterbi
	viterbi = chartmatrix(grammar.nonterminals, lensent)
	assert len(viterbi) >= grammar.nonterminals
	assert len(viterbi[0]) >= lensent
	assert len(viterbi[0][0]) >= lensent + 1
	minleft, maxleft, minright, maxright = minmaxmatrices(
			grammar.nonterminals, lensent)
	if start is None:
		start = grammar.toid[grammar.start]
	for left, _ in enumerate(sent):
		chart[left][left + 1] = dict.fromkeys(range(1, grammar.nonterminals))
	viterbiedges, msg = populatepos(grammar, chart, sent, tags,
			minleft, maxleft, minright, maxright)
	if not viterbiedges:
		return chart, NONE, msg
	for left, viterbicell in enumerate(viterbiedges):
		for lhs, edge in viterbicell.items():
			viterbi[lhs, left, left + 1] = (<CFGEdge>edge).inside

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
	if not chart[0][lensent].get(start):
		return chart, NONE, "no parse " + msg
	return chart, new_CFGChartItem(start, 0, lensent), msg


def parse_sparse(sent, Grammar grammar, start=None, tags=None,
		list chart=None, int beamwidth=0):
	""" A CKY parser modeled after Bodenstab's 'fast grammar loop,' filtered by
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
		short [:, :] minleft, maxleft, minright, maxright
	minleft, maxleft, minright, maxright = minmaxmatrices(
			grammar.nonterminals, lensent)
	if start is None:
		start = grammar.toid[grammar.start]
	if chart is None:
		chart = [[None] * (lensent + 1) for _ in range(lensent)]
		for left in range(lensent):
			for right in range(left, lensent):
				chart[left][right + 1] = {
						n: None for n in range(1, grammar.nonterminals)}
	# assign POS tags
	viterbiedges, msg = populatepos(grammar, chart, sent, tags,
			minleft, maxleft, minright, maxright)
	if not viterbiedges:
		return chart, NONE, msg
	for left, viterbicell in enumerate(viterbiedges):
		viterbi[left][left + 1] = viterbicell

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
				if not cell[lhs]:
					cell[lhs] = {}
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
			nonempty = list(filter(None, cell.values()))
			if nonempty:
				spans += 1
				items += len(nonempty)
				# clean up labelled spans that were whitelisted but not used:
				for label in list(cell):
					if not cell[label]:
						del cell[label]
				edges += sum(map(len, nonempty))
	msg = "chart spans %d, items %d, edges %d" % (spans, items, edges)
	if not chart[0][lensent].get(start):
		return chart, NONE, "no parse " + msg
	return chart, new_CFGChartItem(start, 0, lensent), msg


def parse_symbolic(sent, Grammar grammar, start=None, tags=None):
	""" Parse sentence, a list of tokens, and produce a chart, either
	exhaustive or up until the first complete parse. Non-probabilistic version.

	:returns: a CFG chart with each 'probability' 1.
	:param start: integer corresponding to the start symbol that analyses
		should have, e.g., grammar.toid['ROOT']
	:param tags: optionally, a list with the corresponding POS tags
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
		short [:, :] minleft, maxleft, minright, maxright
	assert grammar.maxfanout == 1, "Not a PCFG! fanout: %d" % grammar.maxfanout
	minleft, maxleft, minright, maxright = minmaxmatrices(
			grammar.nonterminals, lensent)
	# assign POS tags
	viterbiedges, msg = populatepos(grammar, chart, sent, tags,
			minleft, maxleft, minright, maxright)
	if start is None:
		start = grammar.toid[grammar.start]
	if not viterbiedges:
		return chart, NONE, msg

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
			nonempty = list(filter(None, cell.values()))
			if nonempty:
				spans += 1
				items += len(nonempty)
				edges += sum(map(len, nonempty))
	msg = "chart spans %d, items %d, edges %d" % (spans, items, edges)
	if not chart[0][lensent].get(start):
		return chart, NONE, "no parse " + msg
	return chart, new_CFGChartItem(start, 0, lensent), msg


cdef populatepos(Grammar grammar, list chart, sent, tags,
		short [:, :] minleft, short [:, :] maxleft,
		short [:, :] minright, short [:, :] maxright):
	""" Assign all possible POS tags for a word, and apply all possible
	unary rules to them.

	:returns: a dictionary with the best scoring edges for each lhs,
		or None if no POS tag was found for this word. """
	cdef:
		EdgeAgenda unaryagenda = EdgeAgenda()
		Rule *rule
		LexicalRule lexrule
		UInt n, lhs, rhs1
		dict viterbicell
		short left, right
	viterbi = [{} for _ in sent]
	for left, word in enumerate(sent):  # assign POS tags
		tag = tags[left].encode('ascii') if tags else None
		right = left + 1
		cell = chart[left][right]
		viterbicell = viterbi[left]
		recognized = False
		for lexrule in grammar.lexicalbyword.get(word, ()):
			if lexrule.lhs not in cell:
				continue
			lhs = lexrule.lhs
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if (tag is None or grammar.tolabel[lhs] == tag
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
			if tag is None or tag not in grammar.toid:
				return None, "not covered: %r" % (tag or word, )
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
	return viterbi, ''


def doinsideoutside(sent, Grammar grammar, inside=None, outside=None,
		tags=None, start=None):
	assert grammar.maxfanout == 1, "Not a PCFG! fanout = %d" % grammar.maxfanout
	assert not grammar.logprob, "Grammar must not have log probabilities."
	lensent = len(sent)
	if start is None:
		start = grammar.toid[grammar.start]
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
	if inside[0, len(sent), start]:
		outsidescores(grammar, sent, start, inside, outside, *minmaxlr)
		msg = 'inside prob=%g' % inside[0, len(sent), start]
		start = new_CFGChartItem(start, 0, lensent)
	else:
		start = NONE
		msg = "no parse"
	return inside, outside, start, msg


def insidescores(sent, Grammar grammar, double [:, :, :] inside, tags=None):
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
		short [:, :] minleft, maxleft, minright, maxright
		double [:] unaryscores = np.empty(grammar.nonterminals, dtype='d')
	minleft, maxleft, minright, maxright = minmaxmatrices(
			grammar.nonterminals, lensent)
	for left in range(lensent):  # assign POS tags
		tag = tags[left].encode('ascii') if tags else None
		right = left + 1
		for lexrule in grammar.lexicalbyword.get(sent[left], []):
			lhs = lexrule.lhs
			# if we are given gold tags, make sure we only allow matching
			# tags - after removing addresses introduced by the DOP reduction
			if not tags or (grammar.tolabel[lhs] == tag
					or grammar.tolabel[lhs].startswith(tag + b'@')):
				inside[left, right, lhs] = lexrule.prob
		if not inside.base[left, right].any():
			if tags is not None:
				lhs = grammar.toid[tag]
				if not tags or (grammar.tolabel[lhs] == tag
						or grammar.tolabel[lhs].startswith(tag + b'@')):
					inside[left, right, lhs] = 1.
			else:
				raise ValueError("not covered: %r" % (tag or sent[left]), )
		# unary rules on POS tags (NB: agenda is a min-heap, negate probs)
		unaryagenda.update([(rhs1,
			new_CFGEdge(-inside[left, right, rhs1], NULL, 0))
			for rhs1 in range(grammar.nonterminals)
			if inside[left, right, rhs1]
			and grammar.unary[rhs1].rhs1 == rhs1])
		unaryscores[:] = 0.0
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
					#assert 0.0 < inside[left, right, lhs] <= 1.0, (
					#	inside[left, right, lhs],
					#	left, right, grammar.tolabel[lhs])
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
			unaryscores[:] = 0.0
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


def outsidescores(Grammar grammar, sent, UInt start,
		double [:, :, :] inside, double [:, :, :] outside,
		short [:, :] minleft, short [:, :] maxleft,
		short [:, :] minright, short [:, :] maxright):
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
	outside[0, lensent, start] = 1.0
	for span in range(lensent, 0, -1):
		for left in range(1 + lensent - span):
			right = left + span
			# unary rules
			unaryagenda.update([(lhs,
				new_CFGEdge(-outside[left, right, lhs], NULL, 0))
				for lhs in range(grammar.nonterminals)
				if outside[left, right, lhs]])
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
						#assert 0.0 < outside[left, right, rule.rhs1] <= 1.0, (
						#		'illegal value: outside[%d, %d, %s] = %g' % (
						#			left, right, grammar.tolabel[rule.rhs1],
						#			outside[left, right, rule.rhs1]),
						#		rule.prob, outside[left, right, lhs],
						#		grammar.tolabel[rule.lhs])
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
					#assert 0.0 < outside[left, split, rule.rhs1] <= 1.0, (
					#		'illegal value: outside[%d, %d, %s] = %g' % (
					#			left, split, grammar.tolabel[rule.rhs1],
					#			outside[left, split, rule.rhs1]),
					#		rule.prob, rs, os, grammar.tolabel[rule.lhs])
					#assert 0.0 < outside[split, right, rule.rhs2] <= 1.0, (
					#		'illegal value: outside[%d, %d, %s] = %g' % (
					#			split, right, grammar.tolabel[rule.rhs2],
					#			outside[split, right, rule.rhs2]))


def minmaxmatrices(nonterminals, lensent):
	""" Create matrices for the filter which tracks minima and maxima for
	splits of binary rules. """
	minleft = np.empty((nonterminals, lensent + 1), dtype='int16')
	maxleft = np.empty_like(minleft)
	minleft[...], maxleft[...] = -1, lensent + 1
	minright, maxright = maxleft.copy(), minleft.copy()
	return minleft, maxleft, minright, maxright


def chartmatrix(nonterminals, lensent):
	viterbi = np.empty((nonterminals, lensent, lensent + 1), dtype='d')
	viterbi[...] = np.inf
	return viterbi


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


def sortfunc(CFGEdge e):
	return e.inside


def pprint_matrix(matrix, sent, tolabel, matrix2=None):
	""" Print a numpy matrix chart; optionally in parallel with another. """
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
	from discodop._grammar import Grammar
	from discodop.disambiguation import marginalize
	from operator import itemgetter
	cdef Rule rule
	cfg = Grammar([
		((('A', 'A'), ((0, ), )), 0.7), ((('A', 'B'), ((0, ), )), 0.6),
		((('A', 'C'), ((0, ), )), 0.5), ((('A', 'D'), ((0, ), )), 0.4),
		((('B', 'A'), ((0, ), )), 0.3), ((('B', 'B'), ((0, ), )), 0.2),
		((('B', 'C'), ((0, ), )), 0.1), ((('B', 'D'), ((0, ), )), 0.2),
		((('B', 'C'), ((0, ), )), 0.3), ((('C', 'A'), ((0, ), )), 0.4),
		((('C', 'B'), ((0, ), )), 0.5), ((('C', 'C'), ((0, ), )), 0.6),
		((('C', 'D'), ((0, ), )), 0.7), ((('D', 'A'), ((0, ), )), 0.8),
		((('D', 'B'), ((0, ), )), 0.9), ((('D', 'NP', 'VP'), ((0, 1), )), 1),
		((('D', 'C'), ((0, ), )), 0.8), ((('S', 'D'), ((0, ), )), 0.5),
		((('S', 'A'), ((0, ), )), 0.8), ((('NP', 'Epsilon'), ('mary', )), 1),
		((('VP', 'Epsilon'), ('walks', )), 1)],
		start='S')
	print(cfg)
	print("cfg parsing; sentence: mary walks")
	print("pcfg", end='')
	chart, start, msg = parse_sparse("mary walks".split(), cfg)
	assert start, msg
	chart, start, msg = parse_dense("mary walks".split(), cfg)
	assert start, msg
	pprint_chart(chart, "mary walks".split(), cfg.tolabel)
	cfg1 = Grammar([
		((('NP', 'Epsilon'), ('mary', )), 1),
		((('S', 'NP', 'VP'), ((0, 1), )), 1),
		((('VP', 'Epsilon'), ('walks', )), 1)], start='S')
	cfg1.switch('default', False)
	i, o, start, _ = doinsideoutside("mary walks".split(), cfg1)
	assert start
	print(i[0, 2, cfg1.toid[b'S']], o[0, 2, cfg1.toid[b'S']])
	i, o, start, _ = doinsideoutside("walks mary".split(), cfg1)
	assert not start
	print(i[0, 2, cfg1.toid[b'S']], o[0, 2, cfg1.toid[b'S']])
	rules = [
		((('NP', 'NP', 'PP'), ((0, 1), )), 0.4),
		((('PP', 'P', 'NP'), ((0, 1), )), 1),
		((('S', 'NP', 'VP'), ((0, 1), )), 1),
		((('VP', 'V', 'NP'), ((0, 1), )), 0.7),
		((('VP', 'VP', 'PP'), ((0, 1), )), 0.3),
		((('NP', 'Epsilon'), ('astronomers', )), 0.1),
		((('NP', 'Epsilon'), ('ears', )), 0.18),
		((('V', 'Epsilon'), ('saw', )), 1),
		((('NP', 'Epsilon'), ('saw', )), 0.04),
		((('NP', 'Epsilon'), ('stars', )), 0.18),
		((('NP', 'Epsilon'), ('telescopes', )), 0.1),
		((('P', 'Epsilon'), ('with', )), 1)]
	cfg2 = Grammar(rules, start='S')
	cfg2.switch('default', False)
	sent = "astronomers saw stars with ears".split()
	inside, outside, _, msg = doinsideoutside(sent, cfg2)
	print(msg)
	pprint_matrix(inside, sent, cfg2.tolabel, outside)
	cfg2.switch('default', True)
	chart, start, msg = parse(sent, cfg2)
	mpp, _, _ = marginalize('mpp', chart, start, cfg2, 10)
	for a, p in sorted(mpp.items(), key=itemgetter(1), reverse=True):
		print(p, a)
	chart1, start1, msg1 = parse_symbolic(sent, cfg2)
	print(msg, '\n', msg1)
