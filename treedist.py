#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Zhang-Shasha Tree Edit Distance Implementation is licensed under a BSD
style license

Copyright (c) 2012, Tim Henderson and Stephen Johnson
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of this software nor the names of its contributors may
      be used to endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. """
# andreasvc:
# - modified to work with NLTK Tree objects.
# - added implementation as in Billie (2005), which records edit script.

import numpy
from collections import deque
from nltk import Tree

class Terminal:
	""" Auxiliary class to be able to add indices to terminal nodes of NLTK
	trees. """
	def __init__(self, node): self.prod = self.node = node
	def __repr__(self): return repr(self.node)
	def __hash__(self): return hash(self.node)
	def __iter__(self): return iter(())
	def __len__(self): return 0
	def __index__(self): return self.node
	def __getitem__(self, val):
		if isinstance(val, slice): return ()
		else: raise IndexError("A terminal has zero children.")

def prepare(tree, includeterms=False):
	tree = tree.copy(True)
	# canonical order of children
	for a in tree.subtrees(lambda n: isinstance(n[0], Tree)):
		a.sort(key=lambda n: min(n.leaves()))
	if includeterms:
		for a in tree.treepositions('leaves'): tree[a] = Terminal(tree[a])
	else:
		for a in tree.treepositions('leaves'):
			tree[a[:-1]].node += "-%d" % tree[a]
			del tree[a]
	return tree

# begin Zhang-Shasha Tree Edit Distance Implementation.
class AnnotatedTree:
	def __init__(self, root):
		self.root = root
		self.nodes = [] # a pre-order enumeration of the nodes in the tree
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
			n.id = j
			for c in n:
				a = deque(anc)
				a.appendleft(n.id)
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
					if a in leftmostdescendents: break
					else: leftmostdescendents[a] = i
			else:
				lmd = leftmostdescendents[n.id]
			self.leftmostdescendents.append(lmd)
			keyroots[lmd] = i
			i += 1
		self.keyroots = sorted(keyroots.itervalues())

def strdist(a, b): return 0 if a == b else 1
def treedist(A, B, debug=False):
	A = AnnotatedTree(prepare(A))
	B = AnnotatedTree(prepare(B))
	Al = A.leftmostdescendents
	Bl = B.leftmostdescendents
	An = A.nodes
	Bn = B.nodes
	treedists = numpy.zeros((len(A.nodes), len(B.nodes)), int)
	for i in A.keyroots:
		for j in B.keyroots:
			m = i - Al[i] + 2
			n = j - Bl[j] + 2
			fd = numpy.zeros((m, n), int)
			ioff = Al[i] - 1
			joff = Bl[j] - 1

			for x in range(1, m): # δ(l(i1)..i, θ) = δ(l(1i)..1-1, θ) + γ(v → λ)
				fd[x, 0] = fd[x-1, 0] + 1
			for y in range(1, n): # δ(θ, l(j1)..j) = δ(θ, l(j1)..j-1) + γ(λ → w)
				fd[0, y] = fd[0, y-1] + 1

			for x in range(1, m):
				for y in range(1, n):
					# only need to check if x is an ancestor of i
					# and y is an ancestor of j
					if Al[i] == Al[x+ioff] and Bl[j] == Bl[y+joff]:
						#                  +-
						#                  | δ(l(i1)..i-1, l(j1)..j) + γ(v → λ)
						# δ(F1 , F2) = min-+ δ(l(i1)..i , l(j1)..j-1) + γ(λ → w)
						#                  | δ(l(i1)..i-1, l(j1)..j-1) + γ(v → w)
						#                  +-
						labeldist = strdist(An[x+ioff].node, Bn[y+joff].node)
						fd[x, y] = min(fd[x-1, y] + 1, fd[x, y-1] + 1,
							fd[x-1, y-1] + labeldist)
						treedists[x+ioff, y+joff] = fd[x, y]
					else:
						#                  +-
						#                  | δ(l(i1)..i-1, l(j1)..j) + γ(v → λ)
						# δ(F1 , F2) = min-+ δ(l(i1)..i , l(j1)..j-1) + γ(λ → w)
						#                  | δ(l(i1)..l(i)-1, l(j1)..l(j)-1)
						#                  |                   + treedist(i1,j1)
						#                  +-
						p = Al[x+ioff]-1-ioff
						q = Bl[y+joff]-1-joff
						fd[x, y] = min(fd[x-1, y] + 1, fd[x, y-1] + 1,
							fd[p, q] + treedists[x+ioff, y+joff])
		if debug:
			if isinstance(An[i], Tree): astr = An[i].node #pprint(margin=9999)
			else: astr = str(An[i])
			j = treedists[i].argmin()
			if isinstance(Bn[j], Tree): bstr = Bn[j].node #pprint(margin=9999)
			else: bstr = str(Bn[j])
			if treedists[i, j]:
				print "%s[%d] %s[%d] %d" % (astr, i, bstr, j, treedists[i, j])
	return treedists[len(A.nodes)-1, len(B.nodes)-1]
# end Zhang-Shasha Tree Edit Distance Implementation.


# implementation as in Billie (2005), based on rparse code.
# slower but records edit script.
# should be converted to use a set of matrices as dynamic programming tables.
def newtreedist(A, B, debug=False):
	A = prepare(A).freeze(); B = prepare(B).freeze()
	for n, a in enumerate(A.subtrees()): a.idx = n
	for n, a in enumerate(B.subtrees()): a.idx = n
	result = geteditstats((A,), (B,))
	geteditstats.mem.clear()
	if debug: print result
	return result.distance

class EditStats(object):
	__slots__ = ('distance', 'matched', 'editscript')
	def __init__(self, distance=0, matched=0, editscript=None):
		if editscript is None: editscript = ()
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
		return ("%s(distance=%d, matched=%d, [\n\t%s])" % (
			self.__class__.__name__, self.distance, self.matched,
			",\n\t".join("%s(%s, %s)" % (a[0],
			"%s[%d]" % (a[1].node, a[1].idx) if isinstance(a[1], Tree) else a[1],
			"%s[%d]" % (a[2].node, a[2].idx) if isinstance(a[2], Tree) else a[2])
			for a in self.editscript)))

def geteditstats(forest1, forest2):
	try: return geteditstats.mem[forest1, forest2]
	except KeyError: pass
	flatforest1 = forest1 if forest1 == () else forest1[:-1] + tuple(forest1[-1][:])
	flatforest2 = forest2 if forest2 == () else forest2[:-1] + tuple(forest2[-1][:])
	if forest2 == ():
		if forest1 == (): result = EditStats(0, 0, ())
		else:
			tmp = geteditstats(flatforest1, ())
			result =  EditStats(tmp.distance + 1, tmp.matched,
				(('D', forest1[-1], None),) + tmp.editscript)
	elif forest1 == ():
		tmp = geteditstats((), flatforest2)
		result = EditStats(tmp.distance + 1, tmp.matched,
			(('I', None, forest2[-1]), ) + tmp.editscript)
	else:
		v = forest1[-1]; w = forest2[-1]
		tmp = geteditstats(flatforest1, forest2)
		deleteStats = EditStats(tmp.distance + 1, tmp.matched,
				(('D', v, None),) + tmp.editscript)
		tmp = geteditstats(forest1, flatforest2)
		insertStats = EditStats(tmp.distance + 1, tmp.matched,
				(('I', None, w), ) + tmp.editscript)
		matchOrSwapStats = (geteditstats(tuple(v[:]), tuple(w[:]))
				+ geteditstats(forest1[:-1], forest2[:-1]))
		if v.node == w.node:
			matchOrSwapStats = EditStats(matchOrSwapStats.distance,
				matchOrSwapStats.matched + 1, matchOrSwapStats.editscript)
		else:
			matchOrSwapStats = EditStats(matchOrSwapStats.distance + 1,
				matchOrSwapStats.matched,
				(('S', v, w), ) + matchOrSwapStats.editscript)
		result = min(deleteStats, insertStats, matchOrSwapStats)
	geteditstats.mem[forest1, forest2] = result
	return result
geteditstats.mem = {}

def main():
	a = Tree.parse("(f (d (a 0) (c (b 1))) (e 2))", parse_leaf=int)
	b = Tree.parse("(f (c (d (a 0) (b 1)) (e 2)))", parse_leaf=int)
	print '%s\n%s\ndistance: %d' % (a, b, treedist(a, b, debug=True))
	print '%s\n%s\ndistance: %d' % (a, b, newtreedist(a, b, debug=True))
	a = Tree.parse("(f (d (x (a 0)) (b 1) (c 2)) (z 3))", parse_leaf=int)
	b = Tree.parse("(f (c (d (a 0) (x (b 1)) (c 2)) (z 3)))", parse_leaf=int)
	print '%s\n%s\ndistance: %d' % (a, b, treedist(a, b, debug=True))
	print '%s\n%s\ndistance: %d' % (a, b, newtreedist(a, b, debug=True))

if __name__ == '__main__': main()
