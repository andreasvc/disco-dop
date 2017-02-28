"""Punctuation related functions."""
from __future__ import division, print_function, absolute_import, \
		unicode_literals
from .tree import Tree, ParentedTree
from .treetransforms import removeterminals


# fixme: treebank specific parameters for detecting punctuation.
PUNCTTAGS = {'.', ',', ':', "'", "''", '`', '``', '"',  # General
		'-LRB-', '-RRB-', '-LSB-', '-RSB-', '-LCB-', '-RCB-',  # PTB
		'$,', '$.', '$[', '$(',  # Negra/Tiger
		'let', 'LET[]', 'SPEC[symb]', 'TW[hoofd,vrij]',  # Alpino/Lassy
		'COMMA', 'PUNCT', 'PAREN'}  # Grammatical Framework

# NB: ' is not in this list of tokens, because if it occurs as a possesive
# marker it should be left alone; occurrences of ' as quotation marker may
# still be identified using tags.
PUNCTUATION = frozenset('.,():-";?/!*&`[]<>{}|=\xc2\xab\xc2\xbb\xb7\xad\\'
		) | {'&bullet;', '..', '...', '....', '.....', '......',
		'!!', '!!!', '??', '???', "''", '``', ',,',
		'--', '---', '----', '-LRB-', '-RRB-', '-LCB-', '-RCB-',
		'-LSB-', '-RSB-', '#LRB#', '#RRB#'}

# Punctuation that is pruned if it is leading or ending;
# cf. Collins (2003, sec 4.3). http://anthology.aclweb.org/J03-4003
PRUNEPUNCT = {'``', "''", '"', '.'}

# Punctuation that come in pairs (left: right).
BALANCEDPUNCTMATCH = {'"': '"', "'": "'", '``': "''",
		'[': ']', '(': ')', '-LRB-': '-RRB-', '-LSB-': '-RSB-',
		'-': '-', '\xc2\xab': '\xc2\xbb'}  # unicode << and >>


def applypunct(method, tree, sent):
	"""Apply punctuation strategy to tree (in-place).

	:param method: one of ``remove, removeall, move, moveall, prune,``
		or ``root``.
	"""
	if method == 'remove' or method == 'removeall':
		punctremove(tree, sent, method == 'removeall')
	elif method in ('move', 'moveall', 'prune'):
		if method == 'prune':
			punctprune(tree, sent)
		punctraise(tree, sent, method == 'moveall')
		balancedpunctraise(tree, sent)
		# restore linear precedence order
		for a in tree.postorder(
				lambda n: len(n) > 1 and isinstance(n[0], Tree)):
			a.children.sort(key=lambda n: n.leaves())
	elif method == 'root':
		punctroot(tree, sent)


def punctremove(tree, sent, rootpreterms=False):
	"""Remove any punctuation nodes, and any empty ancestors."""
	if rootpreterms:  # remove all preterminals under root indiscriminately.
		preterms = [node for node in tree if node and isinstance(node[0], int)]
		removeterminals(tree, sent, lambda _word, node: node in preterms)
	else:  # remove all punctuation preterminals anywhere in tree.
		removeterminals(tree, sent, ispunct)


def punctprune(tree, sent):
	"""Remove quotes and period at sentence beginning and end."""
	i = 0
	while i < len(sent) and sent[i] in PRUNEPUNCT:
		sent[i] = None
		i += 1
	i = len(sent) - 1
	while i < len(sent) and sent[i] in PRUNEPUNCT:
		sent[i] = None
		i -= 1
	if tree is None:
		sent[:] = [a for a in sent if a is not None]
	else:
		removeterminals(tree, sent, lambda a, _b: a is None)


def punctroot(tree, sent):
	"""Move punctuation directly under ROOT, as in the Negra annotation."""
	punct = []
	for a in reversed(tree.treepositions('leaves')):
		if ispunct(sent[tree[a]], tree[a[:-1]]):
			# store punctuation node
			punct.append(tree[a[:-1]])
			# remove this punctuation node and any empty ancestors
			for n in range(1, len(a)):
				del tree[a[:-n]]
				if tree[a[:-(n + 1)]]:
					break
	tree.extend(punct)


def punctlower(tree, sent):
	"""Find suitable constituent for punctuation marks and add it there.

	Initial candidate is the root node. Note that ``punctraise()`` performs
	better. Based on rparse code."""
	def lower(node, candidate):
		"""Lower a specific instance of punctuation in tree.

		Recurses top-down on suitable candidates."""
		num = node.leaves()[0]
		for i, child in enumerate(sorted(candidate, key=lambda x: x.leaves())):
			if not child or isinstance(child[0], int):
				continue
			termdom = child.leaves()
			if num < min(termdom):
				candidate.insert(i + 1, node)
				break
			elif num < max(termdom):
				lower(node, child)
				break

	for a in tree.treepositions('leaves'):
		if ispunct(sent[tree[a]], tree[a[:-1]]):
			b = tree[a[:-1]]
			del tree[a[:-1]]
			lower(b, tree)


def punctraise(tree, sent, rootpreterms=False):
	"""Attach punctuation nodes to an appropriate constituent.

	Trees in the Negra corpus have punctuation attached to the root;
	i.e., it is not part of the phrase-structure. This function moves the
	punctuation to an appropriate level in the tree. A punctuation node is a
	POS tag with a punctuation terminal. Modifies trees in-place.

	:param rootpreterms: if True, move all preterminals under root,
		instead of only recognized punctuation."""
	def phrasalnode(n):
		"""Test whether node is a phrasal node."""
		return n and isinstance(n[0], Tree)

	# punct = [node for node in tree.subtrees() if isinstance(node[0], int)
	punct = [node for node in tree if node and isinstance(node[0], int)
			and (rootpreterms or ispunct(sent[node[0]], node))]
	while punct:
		node = punct.pop()
		while node is not tree and len(node.parent) == 1:
			node = node.parent
		if node is tree:
			continue
		node.parent.pop(node.parent_index)
		for candidate in tree.subtrees(phrasalnode):
			# add punctuation mark to highest left/right neighbor
			# if any(node[0] - 1 == max(a.leaves()) for a in candidate):
			if any(node[0] + 1 == min(a.leaves()) for a in candidate):
				candidate.append(node)
				break
		else:
			tree.append(node)


def balancedpunctraise(tree, sent):
	"""Move balanced punctuation ``" ' - ( ) [ ]`` to a common constituent.

	Based on rparse code."""
	assert isinstance(tree, ParentedTree)
	# right punct str as key, mapped to left index as value
	punctmap = {}
	# punctuation indices mapped to preterminal nodes
	termparent = {a[0]: a for a in tree.subtrees()
			if a and isinstance(a[0], int) and ispunct(sent[a[0]], a)}
	for terminal in sorted(termparent):
		preterminal = termparent[terminal]
		# do we know the matching punctuation mark for this one?
		if preterminal.label in PUNCTTAGS and sent[terminal] in punctmap:
			right = terminal
			left = punctmap[sent[right]]
			rightparent = preterminal.parent
			leftparent = termparent[left].parent
			if max(leftparent.leaves()) == right - 1:
				node = termparent[right]
				leftparent.append(node.parent.pop(node.parent_index))
			elif min(rightparent.leaves()) == left + 1:
				node = termparent[left]
				rightparent.insert(0, node.parent.pop(node.parent_index))
			if sent[right] in punctmap:
				del punctmap[sent[right]]
		elif (sent[terminal] in BALANCEDPUNCTMATCH
				and preterminal.label in PUNCTTAGS):
			punctmap[BALANCEDPUNCTMATCH[sent[terminal]]] = terminal


def ispunct(word, tag):
	"""Test whether a word and/or tag is punctuation."""
	return tag.label in PUNCTTAGS or word in PUNCTUATION


__all__ = ['ispunct', 'punctremove', 'punctprune', 'punctroot', 'punctlower',
		'punctraise', 'balancedpunctraise', 'applypunct']
