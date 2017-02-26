# -*- coding: utf-8 -*-
"""Tree edit distance implementations.

- Zhang & Shasha (1989) http://epubs.siam.org/doi/abs/10.1137/0218082
  Implementation licensed under a BSD style license.
- Billie (2005)
  http://www.imm.dtu.dk/~phbi/files/publications/2005asotedarpJ.pdf
  Implementation records edit script.
"""

# Notice for the Zhang-Shasha implementation:
#
# Copyright (c) 2012, Tim Henderson and Stephen Johnson
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of this software nor the names of its contributors may
#       be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import division, print_function, absolute_import
from collections import deque
from .tree import Tree


class Terminal(object):
	"""Auxiliary class to add indices to terminal nodes of Tree objects."""

	def __init__(self, node):
		self.prod = self.label = node

	def __repr__(self):
		return repr(self.label)

	def __hash__(self):
		return hash(self.label)

	def __iter__(self):
		return iter(())

	def __len__(self):
		return 0

	def __index__(self):
		return self.label

	def __getitem__(self, val):
		if isinstance(val, slice):
			return ()
		else:
			raise IndexError("A terminal has zero children.")


def prepare(tree, includeterms=False):
	"""Return a copy of tree prepared for tree edit distance calculation.

	- sort children to have canonical order
	- merge preterminals and terminals in single nodes
		(unless ``includeterms=True``)."""
	tree = Tree.convert(tree)
	# canonical order of children
	for a in tree.subtrees(lambda n: isinstance(n[0], Tree)):
		a.children.sort(key=lambda n: min(n.leaves()))
	if includeterms:
		for a in tree.treepositions('leaves'):
			tree[a] = Terminal(tree[a])
	else:
		for a in tree.treepositions('leaves'):
			tree[a[:-1]].label += "-%d" % tree[a]
			del tree[a]
	return tree


# begin Zhang-Shasha Tree Edit Distance Implementation.
class AnnotatedTree(object):
	"""Wrap a tree to add some extra information."""

	def __init__(self, root):
		self.root = root
		self.nodes = []  # a pre-order enumeration of the nodes in the tree
		self.leftmostdescendents = []  # left most descendents
		self.keyroots = None
		# k and k' are nodes specified in the pre-order enumeration.
		# keyroots = {k | there exists no k'>k such that lmd(k) == lmd(k')}
		# see paper for more on keyroots

		stack = [(root, deque())]
		pstack = []
		j = 0
		while stack:
			n, anc = stack.pop()
			# hack; this is actually an ID but can't add new attributes
			n.head = j
			for c in n:
				a = deque(anc)
				a.appendleft(n.head)
				stack.append((c, a))
			pstack.append((n, anc))
			j += 1
		leftmostdescendents = dict()
		keyroots = dict()
		i = 0
		while pstack:
			n, anc = pstack.pop()
			self.nodes.append(n)
			if len(n) == 0:
				lmd = i
				for a in anc:
					if a in leftmostdescendents:
						break
					else:
						leftmostdescendents[a] = i
			else:
				lmd = leftmostdescendents[n.head]
			self.leftmostdescendents.append(lmd)
			keyroots[lmd] = i
			i += 1
		self.keyroots = sorted(keyroots.values())


def strdist(a, b):
	"""Default categorical distance function."""
	return 0 if a == b else 1


def treedist(tree1, tree2, debug=False):
	"""Zhang-Shasha tree edit distance."""
	tree1 = AnnotatedTree(prepare(tree1))
	tree2 = AnnotatedTree(prepare(tree2))
	tree1lmd = tree1.leftmostdescendents
	tree2lmd = tree2.leftmostdescendents
	tree1nodes = tree1.nodes
	tree2nodes = tree2.nodes
	import numpy
	treedists = numpy.zeros((len(tree1.nodes), len(tree2.nodes)), int)
	for i in tree1.keyroots:
		for j in tree2.keyroots:
			m = i - tree1lmd[i] + 2
			n = j - tree2lmd[j] + 2
			table = numpy.zeros((m, n), int)
			ioff = tree1lmd[i] - 1
			joff = tree2lmd[j] - 1

			for x in range(1, m):  # δ(l(i1)..i, θ) = δ(l(1i)..1-1, θ) + γ(v → λ)
				table[x, 0] = table[x - 1, 0] + 1
			for y in range(1, n):  # δ(θ, l(j1)..j) = δ(θ, l(j1)..j-1) + γ(λ → w)
				table[0, y] = table[0, y - 1] + 1

			for x in range(1, m):
				for y in range(1, n):
					# only need to check if x is an ancestor of i
					# and y is an ancestor of j
					if (tree1lmd[i] == tree1lmd[x + ioff]
							and tree2lmd[j] == tree2lmd[y + joff]):
						#                 +-
						#                 | δ(l(i1)..i-1, l(j1)..j) + γ(v → λ)
						# δ(F1, F2) = min-+ δ(l(i1)..i , l(j1)..j-1) + γ(λ → w)
						#                 | δ(l(i1)..i-1, l(j1)..j-1) + γ(v → w)
						#                 +-
						labeldist = strdist(tree1nodes[x + ioff].label,
								tree2nodes[y + joff].label)
						table[x, y] = min(table[x - 1, y] + 1,
								table[x, y - 1] + 1,
								table[x - 1, y - 1] + labeldist)
						treedists[x + ioff, y + joff] = table[x, y]
					else:
						#                 +-
						#                 | δ(l(i1)..i-1, l(j1)..j) + γ(v → λ)
						# δ(F1, F2) = min-+ δ(l(i1)..i , l(j1)..j-1) + γ(λ → w)
						#                 | δ(l(i1)..l(i)-1, l(j1)..l(j)-1)
						#                 |                   + treedist(i1,j1)
						#                 +-
						a = tree1lmd[x + ioff] - 1 - ioff
						b = tree2lmd[y + joff] - 1 - joff
						table[x, y] = min(table[x - 1, y] + 1,
								table[x, y - 1] + 1,
								table[a, b] + treedists[x + ioff, y + joff])
		if debug:
			if isinstance(tree1nodes[i], Tree):
				astr = tree1nodes[i].label  # pprint()
			else:
				astr = str(tree1nodes[i])
			j = treedists[i].argmin()
			if isinstance(tree2nodes[j], Tree):
				bstr = tree2nodes[j].label  # pprint()
			else:
				bstr = str(tree2nodes[j])
			if treedists[i, j]:
				print("%s[%d] %s[%d] %d" % (astr, i, bstr, j, treedists[i, j]))
	return treedists[len(tree1.nodes) - 1, len(tree2.nodes) - 1]
# end Zhang-Shasha Tree Edit Distance Implementation.


def newtreedist(tree1, tree2, debug=False):
	"""Tree edit distance implementation as in Billie (2005).

	Based on rparse code. Slower than ``treedist()`` but records edit script.
	Should be rewritten to use a set of matrices as dynamic programming
	tables."""
	tree1 = prepare(tree1).freeze()
	tree2 = prepare(tree2).freeze()
	for n, a in enumerate(tree1.subtrees()):
		# hack; this is actually an ID but can't add new attributes
		a.head = n
	for n, a in enumerate(tree2.subtrees()):
		a.head = n
	result = geteditstats((tree1,), (tree2,))
	geteditstats.mem.clear()
	if debug:
		print(result)
	return result.distance


class EditStats(object):
	"""Collect edit operations on a tree."""
	__slots__ = ('distance', 'matched', 'editscript')

	def __init__(self, distance=0, matched=0, editscript=None):
		if editscript is None:
			editscript = ()
		self.distance = distance
		self.matched = matched
		self.editscript = editscript

	def __add__(self, other):
		return EditStats(self.distance + other.distance,
			self.matched + other.matched,
			self.editscript + other.editscript)

	def __lt__(self, other):
		return self.distance < other.distance

	def __repr__(self):
		return "%s(distance=%d, matched=%d, [\n\t%s])" % (
				self.__class__.__name__, self.distance, self.matched,
				",\n\t".join("%s(%s, %s)" % (a[0],
					"%s[%d]" % (a[1].label, a[1].head)
						if isinstance(a[1], Tree) else a[1],
					"%s[%d]" % (a[2].label, a[2].head)
						if isinstance(a[2], Tree) else a[2])
					for a in self.editscript))


def geteditstats(forest1, forest2):
	"""Recursively get edit distance."""
	try:
		return geteditstats.mem[forest1, forest2]
	except KeyError:
		pass
	flatforest1 = forest1 if forest1 == () else (
			forest1[:-1] + tuple(forest1[-1][:]))
	flatforest2 = forest2 if forest2 == () else (
			forest2[:-1] + tuple(forest2[-1][:]))
	if forest2 == ():
		if forest1 == ():
			result = EditStats(0, 0, ())
		else:
			tmp = geteditstats(flatforest1, ())
			result = EditStats(tmp.distance + 1, tmp.matched,
				(('D', forest1[-1], None), ) + tmp.editscript)
	elif forest1 == ():
		tmp = geteditstats((), flatforest2)
		result = EditStats(tmp.distance + 1, tmp.matched,
			(('I', None, forest2[-1]), ) + tmp.editscript)
	else:
		node1 = forest1[-1]
		node2 = forest2[-1]
		tmp = geteditstats(flatforest1, forest2)
		deletestats = EditStats(tmp.distance + 1, tmp.matched,
				(('D', node1, None),) + tmp.editscript)
		tmp = geteditstats(forest1, flatforest2)
		insertstats = EditStats(tmp.distance + 1, tmp.matched,
				(('I', None, node2), ) + tmp.editscript)
		matchorswapstats = (geteditstats(tuple(node1[:]), tuple(node2[:]))
				+ geteditstats(forest1[:-1], forest2[:-1]))
		if node1.label == node2.label:
			matchorswapstats = EditStats(matchorswapstats.distance,
				matchorswapstats.matched + 1, matchorswapstats.editscript)
		else:
			matchorswapstats = EditStats(matchorswapstats.distance + 1,
				matchorswapstats.matched,
				(('S', node1, node2), ) + matchorswapstats.editscript)
		result = min(deletestats, insertstats, matchorswapstats)
	geteditstats.mem[forest1, forest2] = result
	return result
geteditstats.mem = {}


def test():
	"""Tree edit distance demonstration."""
	a = Tree('(f (d (a 0) (c (b 1))) (e 2))')
	b = Tree('(f (c (d (a 0) (b 1)) (e 2)))')
	result1 = treedist(a, b, debug=True)
	assert result1 == 2
	print('%s\n%s\ndistance: %d' % (a, b, result1))
	result2 = newtreedist(a, b, debug=True)
	assert result2 == 2
	print('%s\n%s\ndistance: %d' % (a, b, result2))
	a = Tree('(f (d (x (a 0)) (b 1) (c 2)) (z 3))')
	b = Tree('(f (c (d (a 0) (x (b 1)) (c 2)) (z 3)))')
	result1 = treedist(a, b, debug=True)
	assert result1 == 3
	print('%s\n%s\ndistance: %d' % (a, b, result1))
	result2 = newtreedist(a, b, debug=True)
	assert result2 == 3
	print('%s\n%s\ndistance: %d' % (a, b, result2))


__all__ = ['Terminal', 'prepare', 'AnnotatedTree', 'strdist', 'treedist',
		'newtreedist', 'EditStats', 'geteditstats']
