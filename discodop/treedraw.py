"""Discontinuous tree drawing.

Interesting reference (not used for this code):
T. Eschbach et al., Orth. Hypergraph Drawing, Journal of
Graph Algorithms and Applications, 10(2) 141--157 (2006)149.
http://jgaa.info/accepted/2006/EschbachGuentherBecker2006.10.2.pdf
"""

from __future__ import division, print_function, unicode_literals
import re
import sys
import codecs
from cgi import escape
from collections import defaultdict, OrderedDict
from operator import itemgetter
from itertools import chain, islice
from discodop.tree import Tree
from discodop.treebank import READERS, incrementaltreereader
from discodop.treetransforms import disc
if sys.version[0] >= '3':
	basestring = str  # pylint: disable=W0622,C0103


USAGE = '''\
Usage: %s [<treebank>...] [options]
Options:
  --fmt=[%s]        Specify corpus format [default: export].
  --encoding=enc    Specify a different encoding than the default UTF-8.
  --functions=x     'leave'=default: leave syntactic labels as is,
                    'remove': strip functions off labels,
                    'add': show both syntactic categories and functions,
                    'replace': only show grammatical functions.
  --morphology=x    'no'=default: only show POS tags,
                    'add': concatenate morphology tags to POS tags,
                    'replace': replace POS tags with morphology tags,
                    'between': add morphological node between POS tag and word.
  --abbr            abbreviate labels longer than 5 characters.
  --plain           disable ANSI colors.
  -n, --numtrees=x  only display the first x trees from the input.
If no treebank is given, input is read from standard input; format is detected.
If more than one treebank is specified, trees will be displayed in parallel.
Pipe the output through 'less -R' to preserve the colors.\
''' % (sys.argv[0], '|'.join(READERS))
ANSICOLOR = {
		'black': 30,
		'red': 31,
		'green': 32,
		'yellow': 33,
		'blue': 34,
		'magenta': 35,
		'cyan': 36,
		'white': 37,
}
PTBPUNC = {'-LRB-', '-RRB-', '-LCB-', '-RCB-', '-LSB-', '-RSB-', '-NONE-'}


class DrawTree(object):
	"""Visualize a discontinuous tree in various formats.

	``DrawTree(tree, sent=None, highlight=(), abbr=False)``
	creates an object from which different visualizations can be created.

	:param tree: a Tree object or a string.
	:param sent: a list of words (strings). If sent is not given, and the tree
		contains non-integers as leaves, a continuous phrase-structure tree
		is assumed. If sent is not given and the tree contains only indices
		as leaves, the indices are displayed as placeholder terminals.
	:param abbr: when True, abbreviate labels longer than 5 characters.
	:param highlight: Optionally, a sequence of Tree objects in `tree` which
		should be highlighted. Has the effect of only applying colors to nodes
		in this sequence (nodes should be given as Tree objects, terminals as
		indices).
	:param highlightfunc: Similar to ``highlight``, but affects function tags.

	>>> print(DrawTree('(S (NP Mary) (VP walks))').text())
	... # doctest: +NORMALIZE_WHITESPACE
	      S
	  ____|____
	 NP        VP
	 |         |
	Mary     walks
	"""
	def __init__(self, tree, sent=None, abbr=False, highlight=(),
			highlightfunc=()):
		self.tree = tree
		self.sent = sent
		if isinstance(tree, basestring):
			self.tree = Tree.parse(tree,
					parse_leaf=None if sent is None else int)
		if sent is None:
			leaves = self.tree.leaves()
			if (leaves and not any(len(a) == 0 for a in self.tree.subtrees())
					and all(isinstance(a, int) for a in leaves)):
				self.sent = [str(a) for a in leaves]
			else:
				# this deals with empty nodes (frontier non-terminals)
				# and multiple/mixed terminals under non-terminals.
				self.tree = self.tree.copy(True)
				self.sent = []
				for a in self.tree.subtrees():
					if len(a) == 0:
						a.append(len(self.sent))
						self.sent.append(None)
					elif any(not isinstance(b, Tree) for b in a):
						for n, b in enumerate(a):
							if not isinstance(b, Tree):
								a[n] = len(self.sent)
								self.sent.append('%s' % b)
		if abbr:
			if self.tree is tree:
				self.tree = self.tree.copy(True)
			for n in self.tree.subtrees(lambda x: len(x.label) > 5):
				n.label = n.label[:4] + u'\u2026'  # unicode '...' ellipsis
		self.highlight = self.highlightfunc = None
		self.nodes, self.coords, self.edges = self.nodecoords(
				self.tree, self.sent, highlight, highlightfunc)

	def __str__(self):
		if sys.version[0] >= '3':
			return self.text(unicodelines=True)
		return self.text(unicodelines=True).encode('utf8')

	def __repr__(self):
		return '\n'.join('%d: coord=%r, parent=%r, node=%s' % (
						n, self.coords[n], self.edges.get(n), self.nodes[n])
					for n in sorted(self.nodes))

	def _repr_svg_(self):
		"""Return a rich representation for IPython notebook."""
		return self.svg()

	def nodecoords(self, tree, sent, highlight, highlightfunc):
		"""Produce coordinates of nodes on a grid.

		Objective:

		- Produce coordinates for a non-overlapping placement of nodes and
			horizontal lines.
		- Order edges so that crossing edges cross a minimal number of previous
			horizontal lines (never vertical lines).

		Approach:

		- bottom up level order traversal (start at terminals)
		- at each level, identify nodes which cannot be on the same row
		- identify nodes which cannot be in the same column
		- place nodes into a grid at (row, column)
		- order child-parent edges with crossing edges last

		Coordinates are (row, column); the origin (0, 0) is at the top left;
		the root node is on row 0. Coordinates do not consider the size of a
		node (which depends on font, &c), so the width of a column of the grid
		should be automatically determined by the element with the greatest
		width in that column. Alternatively, the integer coordinates could be
		converted to coordinates in which the distances between adjacent nodes
		are non-uniform.

		Produces tuple (nodes, coords, edges) where:

		- nodes[id]: Tree object for the node with this integer id
		- coords[id]: (n, m) coordinate where to draw node with id in the grid
		- edges[id]: parent id of node with this id (ordered dictionary)
		"""
		def findcell(m, matrix, startoflevel, children):
			"""Find vacant row, column index for node ``m``.

			Iterate over current rows for this level (try lowest first)
			and look for cell between first and last child of this node,
			add new row to level if no free row available."""
			candidates = [a for _, a in children[m]]
			minidx, maxidx = min(candidates), max(candidates)
			leaves = tree[m].leaves()
			center = scale * sum(leaves) // len(leaves)  # center of gravity
			if minidx < maxidx and not minidx < center < maxidx:
				center = sum(candidates) // len(candidates)
			if max(candidates) - min(candidates) > 2 * scale:
				center -= center % scale  # round to unscaled coordinate
				if minidx < maxidx and not minidx < center < maxidx:
					center += scale
			if ids[m] == 0:
				startoflevel = len(matrix)
			for rowidx in range(startoflevel, len(matrix) + 1):
				if rowidx == len(matrix):  # need to add a new row
					matrix.append([vertline if a not in (corner, None)
							else None for a in matrix[-1]])
				row = matrix[rowidx]
				i = j = center
				if len(children[m]) == 1:  # place unaries directly above child
					return rowidx, next(iter(children[m]))[1]
				elif all(a is None or a == vertline for a
						in row[min(candidates):max(candidates) + 1]):
					# find free column
					for n in range(scale):
						i = j = center + n
						while j > minidx or i < maxidx:
							if i < maxidx and (matrix[rowidx][i] is None
									or i in candidates):
								return rowidx, i
							elif j > minidx and (matrix[rowidx][j] is None
									or j in candidates):
								return rowidx, j
							i += scale
							j -= scale
			raise ValueError('could not find a free cell for:\n%s\n%s'
					'min=%d; max=%d' % (tree[m], minidx, maxidx, dumpmatrix()))

		def dumpmatrix():
			"""Dump matrix contents for debugging purposes."""
			return '\n'.join(
				'%2d: %s' % (n, ' '.join(('%2r' % i)[:2] for i in row))
				for n, row in enumerate(matrix))

		leaves = tree.leaves()
		if not all(isinstance(n, int) for n in leaves):
			raise ValueError('All leaves must be integer indices.')
		if len(leaves) != len(set(leaves)):
			raise ValueError('Indices must occur at most once.')
		if not all(0 <= n < len(sent) for n in leaves):
			raise ValueError('All leaves must be in the interval 0..n '
					'with n=len(sent)\ntokens: %d indices: '
					'%r\nsent: %s' % (len(sent), tree.leaves(), sent))
		vertline, corner = -1, -2  # constants
		tree = tree.copy(True)
		for a in tree.subtrees():
			a.sort(key=lambda n: min(n.leaves()) if isinstance(n, Tree) else n)
		scale = 2
		crossed = set()
		# internal nodes and lexical nodes (no frontiers)
		positions = tree.treepositions()
		maxdepth = max(map(len, positions)) + 1
		childcols = defaultdict(set)
		matrix = [[None] * (len(sent) * scale)]
		nodes = {}
		ids = {a: n for n, a in enumerate(positions)}
		self.highlight = {n for a, n in ids.items()
				if not highlight or tree[a] in highlight}
		self.highlightfunc = {n for a, n in ids.items()
				if (tree[a] in highlightfunc
				if highlightfunc else n in self.highlight)}
		levels = {n: [] for n in range(maxdepth - 1)}
		terminals = []
		for a in positions:
			node = tree[a]
			if isinstance(node, Tree):
				levels[maxdepth - node.height()].append(a)
			else:
				terminals.append(a)

		for n in levels:
			levels[n].sort(key=lambda n: max(tree[n].leaves())
					- min(tree[n].leaves()))
		terminals.sort()
		positions = set(positions)

		for m in terminals:
			i = int(tree[m]) * scale
			assert matrix[0][i] is None, (matrix[0][i], m, i)
			matrix[0][i] = ids[m]
			nodes[ids[m]] = sent[tree[m]]
			if nodes[ids[m]] is None:
				nodes[ids[m]] = '...'
				self.highlight.discard(ids[m])
				self.highlightfunc.discard(ids[m])
			positions.remove(m)
			childcols[m[:-1]].add((0, i))

		# add other nodes centered on their children,
		# if the center is already taken, back off
		# to the left and right alternately, until an empty cell is found.
		for n in sorted(levels, reverse=True):
			nodesatdepth = levels[n]
			startoflevel = len(matrix)
			matrix.append([vertline if a not in (corner, None) else None
					for a in matrix[-1]])
			for m in nodesatdepth:  # [::-1]:
				if n < maxdepth - 1 and childcols[m]:
					_, pivot = min(childcols[m], key=itemgetter(1))
					if ({a[:-1] for row in matrix[:-1] for a in row[:pivot]
							if isinstance(a, tuple)} &
						{a[:-1] for row in matrix[:-1] for a in row[pivot:]
							if isinstance(a, tuple)}):
						crossed.add(m)

				rowidx, i = findcell(m, matrix, startoflevel, childcols)
				positions.remove(m)

				# block positions where children of this node branch out
				for _, x in childcols[m]:
					matrix[rowidx][x] = corner
				# assert m == () or matrix[rowidx][i] in (None, corner), (
				# 		matrix[rowidx][i], m, str(tree), ' '.join(sent))
				# node itself
				matrix[rowidx][i] = ids[m]
				nodes[ids[m]] = tree[m]
				# add column to the set of children for its parent
				if m != ():
					childcols[m[:-1]].add((rowidx, i))
		assert len(positions) == 0

		# remove unused columns, right to left
		for m in range(scale * len(sent) - 1, -1, -1):
			if not any(isinstance(row[m], (Tree, int))
					for row in matrix):
				for row in matrix:
					del row[m]

		# remove unused rows, reverse
		matrix = [row for row in reversed(matrix)
				if not all(a is None or a == vertline for a in row)]

		# collect coordinates of nodes
		coords = {}
		for n, _ in enumerate(matrix):
			for m, i in enumerate(matrix[n]):
				if isinstance(i, int) and i >= 0:
					coords[i] = n, m

		# move crossed edges last
		positions = sorted([a for level in levels.values()
				for a in level], key=lambda a: a[:-1] in crossed)

		# collect edges from node to node
		edges = OrderedDict()
		for i in reversed(positions):
			for j, _ in enumerate(tree[i]):
				edges[ids[i + (j, )]] = ids[i]

		return nodes, coords, edges

	def svg(self, hscale=40, hmult=3, nodecolor='blue', leafcolor='red',
			funccolor='green', funcsep=None):
		""":returns: SVG representation of a discontinuous tree."""
		fontsize = 12
		vscale = 25
		hstart = vstart = 20
		width = max(col for _, col in self.coords.values())
		height = max(row for row, _ in self.coords.values())
		result = ['<svg version="1.1" xmlns="http://www.w3.org/2000/svg" '
				'width="%dem" height="%dem" viewBox="%d %d %d %d">' % (
						width * hmult,
						height * 2.5,
						-hstart, -vstart,
						width * hscale + hmult * hstart,
						height * vscale + 3 * vstart)]

		children = defaultdict(set)
		for n in self.nodes:
			if n:
				children[self.edges[n]].add(n)

		# horizontal branches from nodes to children
		for node in self.nodes:
			if not children[node]:
				continue
			y, x = self.coords[node]
			x *= hscale
			y *= vscale
			x += hstart
			y += vstart + fontsize // 2
			childx = [self.coords[c][1] for c in children[node]]
			xmin = hstart + hscale * min(childx)
			xmax = hstart + hscale * max(childx)
			result.append(
				'\t<polyline style="stroke:black; stroke-width:1; fill:none;" '
					'points="%g,%g %g,%g" />' % (
					xmin, y, xmax, y))
			result.append(
				'\t<polyline style="stroke:black; stroke-width:1; fill:none;" '
					'points="%g,%g %g,%g" />' % (
					x, y, x, y - fontsize // 3))

		# vertical branches from children to parents
		for child, parent in self.edges.items():
			y, _ = self.coords[parent]
			y *= vscale
			y += vstart + fontsize // 2
			childy, childx = self.coords[child]
			childx *= hscale
			childy *= vscale
			childx += hstart
			childy += vstart - fontsize
			result.append(
				'\t<polyline style="stroke:white; stroke-width:10; fill:none;"'
				' points="%g,%g %g,%g" />' % (childx, childy, childx, y + 5))
			result.append(
				'\t<polyline style="stroke:black; stroke-width:1; fill:none;" '
				'points="%g,%g %g,%g" />' % (childx, childy, childx, y))

		# write nodes with coordinates
		for n, (row, column) in self.coords.items():
			node = self.nodes[n]
			x = column * hscale + hstart
			y = row * vscale + vstart
			color = 'black'
			if n in self.highlight:
				color = nodecolor if isinstance(node, Tree) else leafcolor
			if (funcsep and isinstance(node, Tree) and funcsep in node.label
					and node.label not in PTBPUNC):
				cat, func = node.label.split(funcsep, 1)
			else:
				cat = node.label if isinstance(node, Tree) else node
				func = None
			result.append('\t<text x="%g" y="%g" '
					'style="text-anchor: middle; font-size: %dpx;" >'
					'<tspan style="fill: %s; ">%s</tspan>' % (
					x, y, fontsize,
					color, escape(cat)))
			if func:
				result[-1] += '%s<tspan style="fill: %s; ">%s</tspan>' % (
							funcsep, funccolor if n in self.highlightfunc
							else 'black', func)
			result[-1] += '</text>'
		result += ['</svg>']
		return '\n'.join(result)

	def text(self, nodedist=1, unicodelines=False, html=False, ansi=False,
				nodecolor='blue', leafcolor='red', funccolor='green',
				funcsep=None, maxwidth=16):
		""":returns: ASCII art for a discontinuous tree.

		:param nodedist: minimum number of horiziontal spaces between nodes
		:param unicodelines: whether to use Unicode line drawing characters
			instead of plain (7-bit) ASCII.
		:param html: whether to wrap output in html code (default plain text).
		:param ansi: whether to produce colors with ANSI escape sequences
			(only effective when html==False).
		:param leafcolor, nodecolor: specify colors of leaves and phrasal
			nodes; effective when either html or ansi is True.
		:param funccolor, funcsep: if ``funcsep`` is a string, it is taken as a
			separator for function tags, which will be drawn with
			``funccolor``.
		:param maxwidth: maximum number of characters before a label starts to
			wrap across multiple lines; pass None to disable."""
		if unicodelines:
			horzline = u'\u2500'
			leftcorner = u'\u250c'
			rightcorner = u'\u2510'
			vertline = u' \u2502 '
			tee = horzline + u'\u252C' + horzline
			bottom = horzline + u'\u2534' + horzline
			cross = horzline + u'\u253c' + horzline
		else:
			horzline = '_'
			leftcorner = rightcorner = ' '
			vertline = ' | '
			tee = 3 * horzline
			cross = bottom = '_|_'

		def crosscell(cur, x=vertline):
			"""Overwrite center of this cell with a vertical branch."""
			splitl = len(cur) - len(cur) // 2 - len(x) // 2 - 1
			lst = list(cur)
			lst[splitl:splitl + len(x)] = list(x)
			return ''.join(lst)

		result = []
		matrix = defaultdict(dict)
		maxnodewith = defaultdict(lambda: 3)
		maxnodeheight = defaultdict(lambda: 1)
		maxcol = 0
		minchildcol = {}
		maxchildcol = {}
		childcols = defaultdict(set)
		labels = {}
		wrapre = re.compile('(.{%d,%d}\\b\\W*|.{%d})' % (
				maxwidth - 4, maxwidth, maxwidth))
		# collect labels and coordinates
		for a in self.nodes:
			row, column = self.coords[a]
			matrix[row][column] = a
			maxcol = max(maxcol, column)
			label = (self.nodes[a].label if isinstance(self.nodes[a], Tree)
						else self.nodes[a])
			if maxwidth and len(label) > maxwidth:
				label = wrapre.sub(r'\1\n', label).strip()
			label = label.split('\n')
			maxnodeheight[row] = max(maxnodeheight[row], len(label))
			maxnodewith[column] = max(maxnodewith[column], max(map(len, label)))
			labels[a] = label
			if a not in self.edges:
				continue  # e.g., root
			parent = self.edges[a]
			childcols[parent].add((row, column))
			minchildcol[parent] = min(minchildcol.get(parent, column), column)
			maxchildcol[parent] = max(maxchildcol.get(parent, column), column)
		# bottom up level order traversal
		for row in sorted(matrix, reverse=True):
			noderows = [[''.center(maxnodewith[col]) for col in range(maxcol + 1)]
					for _ in range(maxnodeheight[row])]
			branchrow = [''.center(maxnodewith[col]) for col in range(maxcol + 1)]
			for col in matrix[row]:
				n = matrix[row][col]
				node = self.nodes[n]
				text = labels[n]
				if isinstance(node, Tree):
					# draw horizontal branch towards children for this node
					if n in minchildcol and minchildcol[n] < maxchildcol[n]:
						i, j = minchildcol[n], maxchildcol[n]
						a, b = (maxnodewith[i] + 1) // 2 - 1, maxnodewith[j] // 2
						branchrow[i] = ((' ' * a) + leftcorner).ljust(
								maxnodewith[i], horzline)
						branchrow[j] = (rightcorner + (' ' * b)).rjust(
								maxnodewith[j], horzline)
						for i in range(minchildcol[n] + 1, maxchildcol[n]):
							if i == col and any(
									a == i for _, a in childcols[n]):
								line = cross
							elif i == col:
								line = bottom
							elif any(a == i for _, a in childcols[n]):
								line = tee
							else:
								line = horzline
							branchrow[i] = line.center(maxnodewith[i], horzline)
					else:  # if n and n in minchildcol:
						branchrow[col] = crosscell(branchrow[col])
				text = [a.center(maxnodewith[col]) for a in text]
				color = nodecolor if isinstance(node, Tree) else leafcolor
				if html:
					text = [escape(a) for a in text]
				if (n in self.highlight or n in self.highlightfunc) and (
						html or ansi):
					newtext = []
					seensep = False
					# everything before dash in nodecolor,
					# after dash use funccolor
					for line in text:
						if (funcsep and not seensep and isinstance(node, Tree)
								and node.label not in PTBPUNC
								and funcsep in line):
							cat, func = line.split(funcsep, 1)
						elif seensep:
							cat, func = None, line
						else:
							cat, func = line, None
						if cat and (html or ansi) and n in self.highlight:
							if html:
								newtext.append('<font color=%s>%s</font>' % (
										color, cat))
							elif ansi:
								newtext.append('\x1b[%d;1m%s\x1b[0m' % (
										ANSICOLOR[color], cat))
						elif cat:
							newtext.append(cat)
						if func:
							seensep = True
							if html and n in self.highlightfunc:
								newtext[-1] += '%s<font color=%s>%s</font>' % (
										funcsep, funccolor, func)
							elif ansi and n in self.highlightfunc:
								newtext[-1] += '%s\x1b[%d;1m%s\x1b[0m' % (
										funcsep, ANSICOLOR[funccolor], func)
							else:
								newtext[-1] += funcsep + func
					text = newtext
				for x in range(maxnodeheight[row]):
					# draw vertical lines in partially filled multiline node
					# labels, but only if it's not a frontier node.
					noderows[x][col] = (text[x] if x < len(text)
							else (vertline if childcols[n] else ' ').center(
								maxnodewith[col], ' '))
			# for each column, if there is a node below us which has a parent
			# above us, draw a vertical branch in that column.
			if row != max(matrix):
				for n, (childrow, col) in self.coords.items():
					if (n > 0 and
							self.coords[self.edges[n]][0] < row < childrow):
						branchrow[col] = crosscell(branchrow[col])
						if col not in matrix[row]:
							for noderow in noderows:
								noderow[col] = crosscell(noderow[col])
				branchrow = [a + ((a[-1] if a[-1] != ' ' else b[0]) * nodedist)
						for a, b in zip(branchrow, branchrow[1:] + [' '])]
				result.append(''.join(branchrow))
			result.extend((' ' * nodedist).join(noderow)
					for noderow in reversed(noderows))
		return '\n'.join(reversed(result)) + '\n'

	def tikzmatrix(self, nodecolor='blue', leafcolor='red',
			funccolor='green', funcsep=None):
		"""Produce TiKZ code for use with LaTeX.

		PDF can be produced with pdflatex. Uses TiKZ matrices meaning that
		nodes are put into a fixed grid. Where the cells of each column all
		have the same width."""
		result = ['%% %s\n%% %s' % (
				self.tree, ' '.join(a or '' for a in self.sent)),
				r'''\begin{tikzpicture}[scale=0.75, align=center,
				inner sep=0mm, node distance=1mm]''',
				r'\footnotesize\sffamily',
				r'\matrix[row sep=0.5cm,column sep=0.1cm] {']

		# write matrix with nodes
		matrix = defaultdict(dict)
		maxcol = 0
		for a in self.nodes:
			row, column = self.coords[a]
			matrix[row][column] = a
			maxcol = max(maxcol, column)

		for row in sorted(matrix):
			line = []
			for col in range(maxcol + 1):
				if col in matrix[row]:
					n = matrix[row][col]
					node = self.nodes[n]
					cat, func = node.label, ''
					if isinstance(node, Tree):
						color = nodecolor
						if (funcsep and funcsep in node.label
								and node.label not in PTBPUNC):
							cat, func = node.label.split(funcsep)
							func = r'%s\textcolor{%s}{%s}' % (
									funcsep, funccolor, func)
						label = latexlabel(cat)
					else:
						color = leafcolor
						label = node
					if n not in self.highlight:
						color = 'black'
					line.append(r'\node (n%d) { \textcolor{%s}{%s}%s };' % (
							n, color, label, func))
				line.append('&')
			# new row: skip last column char '&', add newline
			result.append(' '.join(line[:-1]) + r' \\')
		result += ['};']
		result.extend(self._tikzedges())
		return '\n'.join(result)

	def tikznode(self, nodecolor='blue', leafcolor='red',
			funccolor='green', funcsep=None):
		"""Produce TiKZ code to draw a tree.

		Nodes are drawn with the \\node command so they can have arbitrary
		coordinates."""
		result = ['%% %s\n%% %s' % (
				self.tree, ' '.join(a or '' for a in self.sent)),
				r'''\begin{tikzpicture}[scale=0.75, align=center,
				inner sep=0mm, node distance=1mm]''',
				r'\footnotesize\sffamily',
				r'\path']

		bottom = max(row for row, _ in self.coords.values())
		# write nodes with coordinates
		for n, (row, column) in self.coords.items():
			node = self.nodes[n]
			cat, func = node.label, ''
			if isinstance(node, Tree):
				color = nodecolor
				if (funcsep and funcsep in node.label
						and node.label not in PTBPUNC):
					cat, func = node.label.split(funcsep)
					func = r'%s\textcolor{%s}{%s}' % (
							funcsep, funccolor, func)
				label = latexlabel(cat)
			else:
				color = leafcolor
				label = node
			if n not in self.highlight:
				color = 'black'
			result.append(r'	(%d, %d) node (n%d) { \textcolor{%s}{%s}%s }'
					% (column, bottom - row, n, color, label, func))
		result += [';']
		result.extend(self._tikzedges())
		return '\n'.join(result)

	def _tikzedges(self):
		"""Generate TiKZ code for drawing edges between nodes."""
		result = []
		shift = -0.5
		# write branches from node to node
		for child, parent in self.edges.items():
			if disc(self.nodes[parent]):
				result.append(
						'\\draw [white, -, line width=6pt] '
						'(n%d)  +(0, %g) -| (n%d);' % (parent, shift, child))
			result.append(
					'\\draw (n%d) -- +(0, %g) -| (n%d);' % (
					parent, shift, child))

		result += [r'\end{tikzpicture}']
		return result


def latexlabel(label):
	"""Quote/format label for latex."""
	newlabel = label.replace('$', r'\$').replace('_', r'\_')
	# turn binarization marker into subscripts in math mode
	if '|' in newlabel:
		cat, siblings = newlabel.split('|', 1)
		siblings = siblings.strip('<>')
		if '^' in siblings:
			siblings, parents = siblings.rsplit('^', 1)
			newlabel = '$ \\textsf{%s}_\\textsf{%s}^\\textsf{%s} $' % (
					cat, siblings[1:-1], parents)
		else:
			newlabel = '$ \\textsf{%s}_\\textsf{%s} $' % (
					cat, siblings)
	else:
		newlabel = newlabel.replace('<', '$<$').replace('>', '$>$')
	return newlabel


def test():
	"""Do some tests."""
	trees = '''(ROOT (S (ADV 0) (VVFIN 1) (NP (PDAT 2) (NN 3)) (PTKNEG 4) \
				(PP (APPRART 5) (NN 6) (NP (ART 7) (ADJA 8) (NN 9)))) ($. 10))
			(S (NP (NN 1) (EX 3)) (VP (VB 0) (JJ 2)))
			(S (VP (PDS 0) (ADV 3) (VVINF 4)) (PIS 2) (VMFIN 1))
			(top (du (comp 0) (smain (noun 1) (verb 2) (inf (verb 8) (inf \
				(adj 3) (pp (prep 4) (np (det 5) (noun 6))) (part 7) (verb 9) \
				(pp (prep 10) (np (det 11) (noun 12) (pp (prep 13) (mwu \
				(noun 14) (noun 15))))))))) (punct 16))
			(top (smain (noun 0) (verb 1) (inf (verb 5) (inf (np (det 2) \
				(adj 3) (noun 4)) (verb 6) (pp (prep 7) (noun 8))))) (punct 9))
			(top (smain (noun 0) (verb 1) (noun 2) (inf (adv 3) (verb 4))) \
				(punct 5))
			(top (punct 5) (du (smain (noun 0) (verb 1) (ppart (np (det 2) \
				(noun 3)) (verb 4))) (conj (sv1 (conj (noun 6) (vg 7) (np \
				(det 8) (noun 9))) (verb 10) (noun 11) (part 12)) (vg 13) \
				(sv1 (verb 14) (ti (comp 19) (inf (np (conj (det 15) (vg 16) \
				(det 17)) (noun 18)) (verb 20)))))) (punct 21))
			(top (punct 10) (punct 16) (punct 18) (smain (np (det 0) (noun 1) \
				(pp (prep 2) (np (det 3) (noun 4)))) (verb 5) (adv 6) (np \
				(noun 7) (noun 8)) (part 9) (np (det 11) (noun 12) (pp \
				(prep 13) (np (det 14) (noun 15)))) (conj (vg 20) (ppres \
				(adj 17) (pp (prep 22) (np (det 23) (adj 24) (noun 25)))) \
				(ppres (adj 19)) (ppres (adj 21)))) (punct 26))
			(top (punct 10) (punct 11) (punct 16) (smain (np (det 0) \
				(noun 1)) (verb 2) (np (det 3) (noun 4)) (adv 5) (du (cp \
				(comp 6) (ssub (noun 7) (verb 8) (inf (verb 9)))) (du \
				(smain (noun 12) (verb 13) (adv 14) (part 15)) (noun 17)))) \
				(punct 18) (punct 19))
			(top (smain (noun 0) (verb 1) (inf (verb 8) (inf (verb 9) (inf \
				(adv 2) (pp (prep 3) (noun 4)) (pp (prep 5) (np (det 6) \
				(noun 7))) (verb 10))))) (punct 11))
			(top (smain (noun 0) (verb 1) (pp (prep 2) (np (det 3) (adj 4) \
				(noun 5) (rel (noun 6) (ssub (noun 7) (verb 10) (ppart \
				(adj 8) (part 9) (verb 11))))))) (punct 12))
			(top (smain (np (det 0) (noun 1)) (verb 2) (ap (adv 3) (num 4) \
				(cp (comp 5) (np (det 6) (adj 7) (noun 8) (rel (noun 9) (ssub \
				(noun 10) (verb 11) (pp (prep 12) (np (det 13) (adj 14) \
				(adj 15) (noun 16))))))))) (punct 17))
			(top (smain (np (det 0) (noun 1)) (verb 2) (adv 3) (pp (prep 4) \
				(np (det 5) (noun 6)) (part 7))) (punct 8))
			(top (punct 7) (conj (smain (noun 0) (verb 1) (np (det 2) \
				(noun 3)) (pp (prep 4) (np (det 5) (noun 6)))) (smain \
				(verb 8) (np (det 9) (num 10) (noun 11)) (part 12)) (vg 13) \
				(smain (verb 14) (noun 15) (pp (prep 16) (np (det 17) \
				(noun 18) (pp (prep 19) (np (det 20) (noun 21))))))) \
				(punct 22))
			(top (smain (np (det 0) (noun 1) (rel (noun 2) (ssub (np (num 3) \
				(noun 4)) (adj 5) (verb 6)))) (verb 7) (ppart (verb 8) (pp \
				(prep 9) (noun 10)))) (punct 11))
			(top (conj (sv1 (np (det 0) (noun 1)) (verb 2) (ppart (verb 3))) \
				(vg 4) (sv1 (verb 5) (pp (prep 6) (np (det 7) (adj 8) \
				(noun 9))))) (punct 10))
			(top (smain (noun 0) (verb 1) (np (det 2) (noun 3)) (inf (adj 4) \
				(verb 5) (cp (comp 6) (ssub (noun 7) (adv 8) (verb 10) (ap \
				(num 9) (cp (comp 11) (np (det 12) (adj 13) (noun 14) (pp \
				(prep 15) (conj (np (det 16) (noun 17)) (vg 18) (np \
				(noun 19))))))))))) (punct 20))
			(top (punct 8) (smain (noun 0) (verb 1) (inf (verb 5) \
				(inf (verb 6) (conj (inf (pp (prep 2) (np (det 3) (noun 4))) \
				(verb 7)) (inf (verb 9)) (vg 10) (inf (verb 11)))))) \
				(punct 12))
			(top (smain (verb 2) (noun 3) (adv 4) (ppart (np (det 0) \
				(noun 1)) (verb 5))) (punct 6))
			(top (conj (smain (np (det 0) (noun 1)) (verb 2) (adj 3) (pp \
				(prep 4) (np (det 5) (noun 6)))) (vg 7) (smain (np (det 8) \
				(noun 9) (pp (prep 10) (np (det 11) (noun 12)))) (verb 13) \
				(pp (prep 14) (np (det 15) (noun 16))))) (punct 17))
			(top (conj (smain (noun 0) (verb 1) (inf (ppart (np (noun 2) \
				(noun 3)) (verb 4)) (verb 5))) (vg 6) (smain (noun 7) \
				(inf (ppart (np (det 8) (noun 9)))))) (punct 10))
			(A (B1 (t 6) (t 13)) (B2 (t 3) (t 7) (t 10))  (B3 (t 1) \
				(t 9) (t 11) (t 14) (t 16)) (B4 (t 0) (t 5) (t 8)))
			(A (B1 6 13) (B2 3 7 10)  (B3 1 \
				9 11 14 16) (B4 0 5 8))
			(VP (VB 0) (PRT 2))
			(VP (VP 0 3) (NP (PRP 1) (NN 2)))
			(ROOT (S (VP_2 (PP (APPR 0) (ART 1) (NN 2) (PP (APPR 3) (ART 4) \
				(ADJA 5) (NN 6))) (ADJD 10) (PP (APPR 11) (NN 12)) (VVPP 13)) \
				(VAFIN 7) (NP (ART 8) (NN 9))) ($. 14))'''
	sents = '''Leider stehen diese Fragen nicht im Vordergrund der \
				augenblicklichen Diskussion .
			is Mary happy there
			das muss man jetzt machen
			Of ze had gewoon met haar vriendinnen rond kunnen slenteren in de \
				buurt van Trafalgar Square .
			Het had een prachtige dag kunnen zijn in Londen .
			Cathy zag hen wild zwaaien .
			Het was een spel geworden , zij en haar vriendinnen kozen iemand \
				uit en probeerden zijn of haar nationaliteit te raden .
			Elk jaar in het hoogseizoen trokken daar massa's toeristen \
				voorbij , hun fototoestel in de aanslag , pratend , gillend \
				en lachend in de vreemdste talen .
			Haar vader stak zijn duim omhoog alsof hij wilde zeggen : " het \
				komt wel goed , joch " .
			Ze hadden languit naast elkaar op de strandstoelen kunnen gaan \
				liggen .
			Het hoorde bij de warme zomerdag die ze ginds achter had gelaten .
			De oprijlaan was niet meer dan een hobbelige zandstrook die zich \
				voortslingerde tussen de hoge grijze boomstammen .
			Haar moeder kleefde bijna tegen het autoraampje aan .
			Ze veegde de tranen uit haar ooghoeken , tilde haar twee koffers \
				op en begaf zich in de richting van het landhuis .
			Het meisje dat vijf keer juist raadde werd getrakteerd op ijs .
			Haar neus werd platgedrukt en leek op een jonge champignon .
			Cathy zag de BMW langzaam verdwijnen tot hij niet meer was dan \
				een zilveren schijnsel tussen de bomen en struiken .
			Ze had met haar moeder kunnen gaan winkelen , zwemmen of \
				terrassen .
			Dat werkwoord had ze zelf uitgevonden .
			De middagzon hing klein tussen de takken en de schaduwen van de \
				wolken drentelden over het gras .
			Zij zou mams rug ingewreven hebben en mam de hare .
			0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
			0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
			0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
			0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
			Mit einer Messe in der Sixtinischen Kapelle ist das Konklave \
				offiziell zu Ende gegangen .'''
	trees = [Tree(a) for a in trees.splitlines()]
	sents = [a.split() for a in sents.splitlines()]
	sents.extend([['Wake', None, 'up'],
		[None, 'your', 'friend', None]])
	for n, (tree, sent) in enumerate(zip(trees, sents)):
		drawtree = DrawTree(tree, sent)
		print('\ntree, sent', n, tree,
				' '.join('...' if a is None else a for a in sent),
				repr(drawtree),
				sep='\n')
		try:
			print(drawtree.text(unicodelines=True, ansi=True), sep='\n')
		except (UnicodeDecodeError, UnicodeEncodeError):
			print(drawtree.text(unicodelines=False, ansi=False), sep='\n')


def main():
	"""Text-based tree viewer."""
	from getopt import gnu_getopt, GetoptError
	flags = ('test', 'help', 'abbr', 'plain')
	options = ('fmt=', 'encoding=', 'functions=', 'morphology=', 'numtrees=')
	try:
		opts, args = gnu_getopt(sys.argv[1:], 'n:', flags + options)
	except GetoptError as err:
		print('error: %s\n%s' % (err, USAGE))
		sys.exit(2)
	opts = dict(opts)
	if '--test' in opts:
		test()
		return
	elif '--help' in opts:
		print(USAGE)
		return
	limit = opts.get('--numtrees', opts.get('-n'))
	limit = int(limit) if limit else None
	if args and opts.get('--fmt', 'export') != 'auto':
		reader = READERS[opts.get('--fmt', 'export')]
		corpora = []
		for path in args:
			corpus = reader(
					path,
					encoding=opts.get('--encoding', 'utf8'),
					functions=opts.get('--functions'),
					morphology=opts.get('--morphology'))
			corpora.append((corpus.trees(), corpus.sents()))
		numsents = len(corpus.sents())
		print('Viewing:', ' '.join(args))
		for n, sentid in enumerate(islice(corpora[0][0], 0, limit), 1):
			print('%d of %s (sentid=%s; len=%d):' % (
					n, numsents, sentid, len(corpora[0][1][sentid])))
			for trees, sents in corpora:
				tree, sent = trees[sentid], sents[sentid]
				print(DrawTree(tree, sent, abbr='--abbr' in opts
						).text(unicodelines=True, ansi='--plain' not in opts,
						funcsep='-' if opts.get('--functions')
							in ('add', 'between') else None))
	else:  # read from stdin + detect format
		reader = codecs.getreader(opts.get('--encoding', 'utf8'))
		stdin = (chain.from_iterable(reader(open(a)) for a in args)
				if args else reader(sys.stdin))
		trees = islice(incrementaltreereader(stdin,
				morphology=opts.get('--morphology'),
				functions=opts.get('--functions')),
				0, limit)
		try:
			for n, (tree, sent, rest) in enumerate(trees, 1):
				print('%d. (len=%d): %s' % (n, len(sent), rest))
				print(DrawTree(tree, sent, abbr='--abbr' in opts).text(
						unicodelines=True, ansi='--plain' not in opts,
						funcsep='-' if opts.get('--functions')
							in ('add', 'between') else None))
		except (IOError, KeyboardInterrupt):
			pass


__all__ = ['DrawTree', 'latexlabel']

if __name__ == '__main__':
	main()
