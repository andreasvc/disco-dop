"""Treebank-indenpendent tree transformations.

This file contains three main transformations:

 - A straightforward binarization: binarize(), based on NLTK code.
   Provides some additional Markovization options.
 - An optimal binarization for LCFRS: optimalbinarize()
   Cf. Gildea (2010): Optimal parsing strategies for linear
   context-free rewriting systems. http://aclweb.org/anthology/N10-1118
 - Converting discontinuous trees to continuous trees and back:
   splitdiscnodes(). Cf. Boyd (2007): Discontinuity revisited.
   http://aclweb.org/anthology/W07-1506"""
# Original notice:
# Natural Language Toolkit: Tree Transformations
#
# Copyright (C) 2005-2007 Oregon Graduate Institute
# Author: Nathan Bodenstab <bodenstab@cslu.ogi.edu>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT
from __future__ import division, print_function, absolute_import, \
		unicode_literals
import re
import sys
from operator import attrgetter
from itertools import islice
from collections import defaultdict, Counter
if sys.version_info[0] > 2:
	unicode = str  # pylint: disable=redefined-builtin
from .tree import Tree, ImmutableTree, isdisc, bitfanout
from .util import ishead, OrderedSet

# e.g., 'VP_2*0' group 1: 'VP_2'; group 2: '0'; group 3: ''
SPLITLABELRE = re.compile(r'(.*)\*(?:([0-9]+)([^!]+![^!]+)?)?$')
MARKOVRE = re.compile(r'^(.*)\|<(.*;)?(.*)>(\^<.*>)?(.*)?$')


def binarize(tree, factor='right', horzmarkov=999, vertmarkov=1,
		revhorzmarkov=0, markhead=False, headoutward=False,
		childchar='|', parentchar='^', tailmarker='',
		leftmostunary=False, rightmostunary=False, threshold=2,
		artpa=True, ids=None, filterfuncs=(),
		labelfun=None, dot=False, abbrrepetition=False,
		direction=False):
	"""
	Binarize a Tree object.

	:param factor: "left" or "right". Determines whether binarization proceeds
		from left to right or vice versa.
	:param horzmarkov: amount of horizontal context in labels. Default is
		infinity, such that now new generalization are introduced by the
		binarization.
	:param vertmarkov: number of ancestors to include in labels.
		NB: 1 means only the direct parent, as in a normal tree.
	:param revhorzmarkov: like ``horzmarkov``, but looks backwards.
	:param headoutward: nodes are marked as head in their function tags;
		the direction of binarization will be switched when it is
		encountered, to enable a head-outward binarization.

	:param markhead: include label of the head child in all auxiliary labels.
	:param leftmostunary, rightmostunary: introduce a unary production for the
		first/last child. When h=1, this enables the same generalizations
		for the first & last non-terminals as for other siblings.
	:param tailmarker: when given a non-empty string, add this to artificial
		nodes introducing the last symbol. This is useful when the last
		symbol is the head node, ensuring that it is not exchangeable with
		other non-terminals.
	:param dot: if True, horizontal context will include all siblings not yet
		generated, separated with a dot from the siblings that have been.
	:param artpa: whether to add parent annotation to the artificial nodes
		introduced by the binarization.
	:param ids: abbreviate artificial node labels using numeric IDs from this
		object; must have dictionary-like interface.
	:param threshold: constituents with more than this number of children are
		factored; i.e., for a value of 2, do a normal binarization; for a
		value of 1, also factor binary productions to include an artificial
		node, etc.
	:param filterfuncs: n-ary branches contain children with grammatical
		functions for labels (optionally with parent annotation of the form
		``FUNC/PARENT``). Any function in the sequence ``filterfuncs`` will
		not become part of the horizontal context of the labels. Can be
		used to filter out adjunctions from this context.
	:param labelfun: a function to derive a label from a node to be used for
		the horizontal markovization context; the default is to use
		``child.label`` for a given child node.
	:param abbrrepetition: in horizontal context, reduce sequences of
		identical labels: e.g., ``<mwp,mwp,mwp,mwp>`` becomes ``<mwp+>``

	>>> from discodop.heads import sethead
	>>> tree = Tree('(S (VP (PDS 0) (ADV 3) (VVINF 4)) (VMFIN 1) (PIS 2))')
	>>> sethead(tree[1])
	>>> sent = 'das muss man jetzt machen'.split()
	>>> print(binarize(tree, horzmarkov=1, headoutward=True))
	(S (VP (PDS 0) (VP|<ADV> (ADV 3) (VVINF 4))) (S|<VMFIN> (VMFIN 1) (PIS 2)))
	>>> tree = Tree('(S (X (A 0) (B 3) (C 4)) (D 1) (E 2))')
	>>> sethead(tree[1])
	>>> print(binarize(tree, headoutward=True, leftmostunary=True,
	... rightmostunary=True))
	(S (S|<X,D,E> (X (X|<A,B,C> (A 0) (X|<B,C> (B 3) (X|<C> (C 4))))) \
(S|<D,E> (S|<D> (D 1)) (E 2))))
	"""
	# FIXME: combination of factor='left' and headoutward=True is broken.
	# assume all nodes have homogeneous children, terminals have no siblings
	if factor not in ('left', 'right'):
		raise ValueError("factor should be 'left' or 'right'.")
	if labelfun is None:
		labelfun = attrgetter('label')
	treeclass = tree.__class__
	origfactor = factor

	# Traverse tree depth-first keeping a list of ancestor nodes to the root.
	agenda = [(tree, [tree.label])]
	while agenda:
		node, parent = agenda.pop()
		if not isinstance(node, Tree):
			continue
		# parent annotation
		parents = ''
		origlabel = node.label if vertmarkov else '_'
		factor = origfactor
		if vertmarkov > 1 and node is not tree and isinstance(node[0], Tree):
			parents = '%s<%s>' % (parentchar, ','.join(parent))
			node.label += parents
			parent = [origlabel] + parent[:vertmarkov - 2]
			if not artpa:
				parents = ''
		# add children to the agenda before we mess with them
		agenda.extend((child, parent) for child in node)
		headidx = None
		if headoutward or markhead:
			for i, child in enumerate(node):
				if isinstance(child, Tree) and ishead(child):
					headidx = i
					break
		# binary form factorization
		if len(node) <= threshold:
			continue
		elif 1 <= len(node) <= 2:
			# insert an initial artificial nonterminal
			siblings = ''
			if isinstance(node[0], Tree):
				if direction and factor == 'left':
					siblings += 'r:'
				elif direction and factor == 'right':
					siblings += 'l:'
				if markhead and headidx is not None:
					siblings += node[headidx].label + ';'
				siblings += ','.join(labelfun(child)
						for child in node[:horzmarkov]
						if labelfun(child).split('/', 1)[0] not in filterfuncs)
			if dot:
				siblings += '.'
			mark = '<%s>%s' % (siblings, parents)
			if ids is not None:  # numeric identifier
				mark = '<%s>' % ids[mark]
			offset = 0 if factor == 'left' else 1
			childnodes = list(node[offset:offset + 1])
			node[offset:offset + 1] = []
			newnode = treeclass(
					'%s%s%s' % (origlabel, childchar, mark), childnodes)
			offset = -1 if factor == 'left' else 1
			node[offset:offset] = [newnode]
		else:
			if isinstance(node[0], Tree):
				childlabels = [labelfun(child) for child in node]
				if filterfuncs:
					childlabels = [x.split('/', 1)[0] for x in childlabels
							if x.split('/', 1)[0] not in filterfuncs]
				if abbrrepetition:
					childlabels = abbr(childlabels)
			else:
				childlabels = []
			childnodes = list(node)
			numchildren = len(childnodes)

			# insert an initial artificial nonterminal
			node[:] = []
			i = 0
			if headoutward and i == headidx:
				factor = 'right' if factor == 'left' else 'left'
			if leftmostunary:
				if factor == 'right':
					start = i
					end = i + horzmarkov
				else:  # factor == 'left'
					start = max(numchildren - i - horzmarkov + (headidx or 0),
							0)
					end = min(numchildren - i + (headidx or 0), numchildren)
				siblings = ''
				if direction and factor == 'left':
					siblings += 'r:'
				elif direction and factor == 'right':
					siblings += 'l:'
				if markhead and headidx is not None:
					siblings += childlabels[headidx] + ';'
				siblings += ','.join(childlabels[start:end])
				if dot:
					siblings += '.'
				mark = '<%s>%s' % (siblings, parents)
				if ids is not None:  # numeric identifier
					mark = '<%s>' % ids[mark]
				newnode = treeclass('%s%s%s' % (origlabel, childchar, mark), [])
				node.append(newnode)
				node = newnode
			curnode = node

			for i in range(1, numchildren - (not rightmostunary)):
				marktail = tailmarker if i + 1 == numchildren else ''
				newnode = treeclass('', [])
				if factor == 'right':
					start = i
					end = i + horzmarkov
				else:  # factor == 'left':
					start = max(numchildren - i - horzmarkov + (headidx or 0),
							(headidx or 0))
					end = min(numchildren - i + (headidx or 0),
							numchildren)
				if factor == 'right':
					curnode[:] = [childnodes.pop(0), newnode]
				else:  # factor == 'left':
					curnode[:] = [newnode, childnodes.pop()]
				siblings = ''
				if direction and factor == 'left':
					siblings += 'r:'
				elif direction and factor == 'right':
					siblings += 'l:'
				if markhead and headidx is not None:
					siblings += childlabels[headidx] + ';'
				if dot:
					siblings += ','.join(childlabels[:start]) + '.'
				if revhorzmarkov:
					if factor == 'right':
						siblings += ','.join(childlabels[
								max(start - revhorzmarkov, 0):start]) + ';'
					else:  # factor == 'left':
						siblings += ','.join(childlabels[
								end:end + revhorzmarkov]) + ';'
				siblings += ','.join(childlabels[start:end])
				mark = '<%s>%s' % (siblings, parents)
				if ids is not None:  # numeric identifier
					mark = '<%s>' % ids[mark]
				newnode.label = ''.join((origlabel, childchar, marktail, mark))
				curnode = newnode
				# switch direction upon encountering the head
				if headoutward and i == headidx:
					factor = 'right' if factor == 'left' else 'left'
				if (headoutward and direction and i == headidx
						and i + 1 != numchildren):
					# insert unary for switch of direction
					newnode = treeclass(curnode.label, curnode[:])
					curnode[:] = [newnode]
					# direction is 'm', no horz. markovization
					newnode.label = ''.join((origlabel, childchar, '<',
							'm:' if direction else '',
							childlabels[headidx] + ';'
								if markhead and headidx is not None else '',
							'>', parents))
					curnode = newnode
			assert len(childnodes) == 1 + (not rightmostunary)
			curnode.extend(childnodes)
	return tree


def unbinarize(tree, _sent=None, expandunary=True,
		childchar='|', parentchar='^', unarychar='+'):
	"""Restore a binarized tree to the original n-ary tree.

	Modifies tree in-place.
	NB: a malformed node such as ``(X|<Y> )`` which is not supposed to be empty
	will be silently discarded."""
	# increase robustness
	childchar += '<'
	parentchar += '<'
	treeclass = tree.__class__
	# Traverse the tree depth-first keeping a pointer to the parent for
	# modification purposes.
	agenda = [(tree, [])]
	while agenda:
		node, parent = agenda.pop()
		if isinstance(node, Tree):
			# if the node contains the 'childchar' character it means that it
			# is an artificial node and can be removed, although we still
			# need to move its children to its parent
			childindex = node.label.find(childchar)
			if childindex != -1 and node is not tree:
				# go by identity instead of equality
				for n, a in enumerate(parent):
					if a is node:
						# convert node to list so that its children may
						# get new parents.
						tmp = node[:]
						node[:] = []
						node = tmp
						parent[n:n + 1] = node
						break
				else:
					raise IndexError
			else:
				parentindex = node.label.find(parentchar)
				if parentindex != -1:
					# strip the node name of the parent annotation
					node.label = node.label[:parentindex]
				# expand collapsed unary productions
				if expandunary:
					unaryindex = node.label.find(unarychar)
					if unaryindex != -1:
						newnode = treeclass(
								node.label[unaryindex + 1:], node[:])
						node.label = node.label[:unaryindex]
						node[:] = [newnode]
				parent = node  # non-binarized node, move on to next parent
			agenda.extend((child, parent) for child in node)
	return tree


def collapseunary(tree, collapsepos=False, collapseroot=False, joinchar='+'):
	"""Collapse unary nodes into a new node indicated by 'joinchar'.

	For example``(NP (NN John))`` becomes ``(NP+NN John)``.
	The tree is modified in-place.

	:param collapsepos: when False (default), do not collapse preterminals
	:param collapseroot: when False (default) do not modify the root production
		if it is unary; e.g., TOP -> productions for the Penn WSJ treebank
	:param joinchar: A string used to connect collapsed node values"""
	agenda = [tree]
	if not collapseroot and isinstance(tree, Tree) and len(tree) == 1:
		agenda = [tree[0]]
	# depth-first traversal of tree
	while agenda:
		node = agenda.pop()
		if isinstance(node, Tree):
			if (len(node) == 1 and isinstance(node[0], Tree)
					and (collapsepos or isinstance(node[0, 0], Tree))):
				node.label += joinchar + node[0].label
				node[0:] = [child for child in node[0]]
				# since we assigned the child's children to the current node,
				# evaluate the current node again
				agenda.append(node)
			else:
				agenda.extend(node)
	return tree


def introducepreterminals(tree, sent, ids=None):
	"""Add preterminals with artificial POS-tags for terminals with siblings.

	:param ids: by default, artificial labels have the form
		``parent_label/terminal``. When an iterator is passed, its values are
		used in place of ``terminal``.

	>>> tree = Tree('(S (X 0 1 (CD 2 3) 4))')
	>>> print(introducepreterminals(tree, ['a', 'b', 'c', 'd', 'e']))
	(S (X (X/a 0) (X/b 1) (CD (CD/c 2) (CD/d 3)) (X/e 4)))
	>>> tree = Tree('(S (X 0 1 2))')
	>>> print(introducepreterminals(tree, [None, None, None], ids=iter('abc')))
	(S (X (X/a 0) (X/b 1) (X/c 2)))
	"""
	assert isinstance(tree, Tree)
	treeclass = tree.__class__
	agenda = [tree]
	while agenda:
		node = agenda.pop()
		hassiblings = len(node) > 1
		for n, child in enumerate(node):
			if isinstance(child, Tree):
				agenda.append(child)
			elif hassiblings:
				node[n] = treeclass('%s/%s' % (
							node.label,
							(sent[child] or '') if ids is None else next(ids)),
						[child])
	return tree


def handledisc(tree):
	"""Binarize discontinuous substitution sites.

	>>> print(handledisc(Tree('(S (X 0 2 4))')))
	(S (X 0 (X|<> 2 (X|<> 4))))
	>>> print(handledisc(Tree('(S (X 0 2))')))
	(S (X 0 (X|<> 2)))
	"""
	for a in tree.postorder(lambda n: len(n) > 1 and isinstance(n[0], int)):
		binarize(a, rightmostunary=True, threshold=1)
	return tree


def factorconstituent(node, sep='|', h=999, factor='right',
		markfanout=False, markyf=False, ids=None, threshold=2,
		filterfuncs=(), labelfun=attrgetter('label')):
	"""Binarize one constituent with a left/right factored binarization.

	Children remain unmodified. Nodes must be immutable and contain bitsets;
	use ``addbitsets()``. By default construct artificial labels using labels
	of child nodes. When markyf is True, each artificial label will include the
	yield function; this is necessary for a 'normal form' binarization that is
	equivalent to the original. When ids is given, it is used both as an
	interator (for new unique labels) and as a dictionary (to re-use labels).
	The first ID in a binarization will always be unique, while the others will
	be re-used for the same combination of labels and yield function."""
	if len(node) <= threshold:
		return node
	elif 1 <= len(node) <= 2:
		if ids is None:
			key = '%s%s' % (','.join(labelfun(child) for child in node[:h]
					if labelfun(child) not in filterfuncs),
					getyf(node[0], node[1] if len(node) > 1 else None)
					if markyf else '')
		else:
			key = next(ids)
		newlabel = '%s%s<%s>' % (node.label, sep, key)
		result = ImmutableTree(node.label, [ImmutableTree(newlabel, node)])
		result.bitset = node.bitset
	else:
		if factor == 'right':
			prev = node[-1]
			rng = range(len(node) - 2, 0, -1)
		elif factor == 'left':
			prev = node[0]
			rng = range(1, len(node) - 1)
		else:
			raise ValueError("factor should be 'left' or 'right'.")
		for i in rng:
			newbitset = node[i].bitset | prev.bitset
			if factor == 'right' and (ids is None or i > 1):
				key = ','.join(labelfun(child) for child in node[i:i + h]
						if labelfun(child) not in filterfuncs)
				if markyf:
					key += getyf(node[i], prev)
				if ids is not None:
					key = ids[key]
			elif factor == 'left' and (ids is None or i < len(node) - 2):
				key = ','.join(labelfun(child)
						for child in node[max(0, i - h + 1):i + 1]
						if labelfun(child) not in filterfuncs)
				if markyf:
					key += getyf(prev, node[i])
				if ids is not None:
					key = ids[key]
			else:
				key = next(ids)
			newlabel = '%s%s<%s>' % (node.label, sep, key)
			if markfanout:
				nodefanout = bitfanout(newbitset)
				if nodefanout > 1:
					newlabel += '_%d' % nodefanout
			prev = ImmutableTree(newlabel,
					[node[i], prev] if factor == 'right' else [prev, node[i]])
			prev.bitset = newbitset
		result = ImmutableTree(node.label,
				[node[0], prev] if factor == 'right' else [prev, node[-1]])
		result.bitset = (node[0].bitset if factor == 'right'
				else node[-1].bitset) | prev.bitset
	return result


def markovthreshold(trees, n, horzmarkov, vertmarkov):
	"""Reduce Markov order of binarization labels occurring < n times."""
	freqs = Counter(node.label for tree in trees
			for node in tree.subtrees()
			if MARKOVRE.match(node.label))
	newlabels = {}
	for label, freq in freqs.items():
		if freq < n:
			match = MARKOVRE.match(label)
			if not match:
				continue
			newlabel = '%s|<%s%s,>' % (
					match.group(1),
					match.group(2) or '',
					','.join(match.group(3).split(',')[:horzmarkov]))
			if match.group(4):
				newlabel += '^<%s>' % ','.join(
						match.group(4).split(',')[:vertmarkov])
			newlabels[label] = newlabel + match.group(5)
	for tree in trees:
		for node in tree.subtrees(lambda n: n.label in newlabels):
			node.label = newlabels[node.label]
	return ('markovization for labels with freq < %d reduced to h=%d v=%d.\n'
			'# labels before %d, after %d. %s' % (n, horzmarkov, vertmarkov,
			len(newlabels), len(set(newlabels.values())),
			', '.join('%s -> %s' % a for a in islice(newlabels.items(), 5))))


def splitdiscnodes(tree, markorigin=False):
	"""Boyd (2007): Discontinuity revisited. http://aclweb.org/anthology/W07-1506

	:markorigin=False: VP* (bare label)
	:markorigin=True: VP*1 (add index)

	>>> tree = Tree('(S (VP (VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4)'
	... '(VVPP 5)) (VAINF 6)) (VMFIN 3))')
	>>> print(splitdiscnodes(tree.copy(True)))
	...  # doctest: +NORMALIZE_WHITESPACE
	(S (VP* (VP* (PP (APPR 0) (ART 1) (NN 2)))) (VMFIN 3) (VP* (VP* (CARD 4)
		(VVPP 5)) (VAINF 6)))
	>>> print(splitdiscnodes(tree, markorigin=True))
	...  # doctest: +NORMALIZE_WHITESPACE
	(S (VP*0 (VP*0 (PP (APPR 0) (ART 1) (NN 2)))) (VMFIN 3) (VP*1 (VP*1
		(CARD 4) (VVPP 5)) (VAINF 6)))"""
	treeclass = tree.__class__
	for node in tree.postorder():
		nodes = list(node)
		node[:] = []
		for child in nodes:
			if isdisc(child):
				childnodes = list(child)
				child[:] = []
				for n, childsubset in enumerate(contsets(childnodes)):
					newlabel = ('%s*%d' % (child.label, n) if markorigin
							else '%s*' % child.label)
					newchild = treeclass(newlabel, childsubset)
					newchild.source = child.source
					node.append(newchild)
			else:
				node.append(child)
	return canonicalize(tree)


def mergediscnodes(tree):
	"""Reverse transformation of ``splitdiscnodes()``."""
	treeclass = tree.__class__
	for node in tree.subtrees():
		merge = defaultdict(list)  # a series of queues of nodes
		# e.g. merge['VP_2*'] = [Tree('VP_2', []), ...]
		# when origin is present (index after *), the node is moved to where
		# the next one is expected, e.g., VP_2*1 after VP_2*0 is added.
		nodes = list(node)  # the original, unmerged children
		node[:] = []  # the new, merged children
		for child in nodes:
			if not isinstance(child, Tree):
				node.append(child)
				continue
			match = SPLITLABELRE.search(child.label)
			if not match:
				node.append(child)
				continue
			label, part, _ = match.groups()
			grandchildren = list(child)
			child[:] = []
			if not merge[child.label]:
				merge[child.label].append(treeclass(label, []))
				node.append(merge[child.label][0])
			merge[child.label][0].extend(grandchildren)
			if part:
				nextlabel = '%s*%d' % (label, int(part) + 1)
				merge[nextlabel].append(merge[child.label].pop(0))
	return tree


def addfanoutmarkers(tree):
	"""Mark discontinuous constituents with '_n' where n = # gaps + 1."""
	for st in tree.subtrees():
		leaves = set(st.leaves())
		thisfanout = len([a for a in sorted(leaves) if a - 1 not in leaves])
		if thisfanout > 1 and not st.label.endswith('_%d' % thisfanout):
			st.label += '_%d' % thisfanout
	return tree


def removefanoutmarkers(tree):
	"""Remove fanout marks."""
	for a in tree.subtrees(lambda x: '_' in x.label):
		a.label = a.label.rsplit('_', 1)[0]
	return tree


def removeterminals(tree, sent, func):
	"""Remove any terminal for which ``func`` is True, and any empty ancestors.

	:param tree: a ParentedTree.
	:param func: a function with the signature (word, node) -> bool."""
	agenda = [tree]
	preterms = {}  # index => Tree object
	while agenda:
		node = agenda.pop()
		if not node:
			continue
		for n in range(len(node) - 1, -1, -1):
			child = node[n]
			if not child:
				continue
			elif isinstance(child[0], Tree):
				agenda.append(child)
			elif func(sent[child[0]], child):
				del node[n]
				# delete empty ancestors
				while not node and node is not tree:
					node, child = node.parent, node
					del node[child._get_parent_index()]
			else:
				preterms[child[0]] = child
	# renumber
	oldindices = sorted(preterms)
	newindices = {a: n for n, a in enumerate(oldindices)}
	for a, node in preterms.items():
		node[0] = newindices[a]
	sent[:] = [sent[a] for a in oldindices]


def removeemptynodes(tree, sent):
	"""Remove any empty nodes, and any empty ancestors."""
	removeterminals(tree, sent,
			lambda word, node: word in (None, '') or node.label == '-NONE-')


def treebankfanout(trees):
	"""Get maximal fan-out of a list of trees."""
	try:  # avoid max over empty sequence: 'treebank' may only have unary prods
		return max((fanout(a), n) for n, tree in enumerate(trees)
				for a in addbitsets(tree).subtrees(lambda x: len(x) > 1))
	except ValueError:
		return 1, 0


def canonicalize(tree):
	"""Restore canonical linear precedence order; tree is modified in-place."""
	for a in tree.postorder(lambda n: len(n) > 1 and isinstance(n[0], Tree)):
		a.children.sort(key=lambda n: n.leaves())
	return tree


def binarizetree(tree, binarization, relationalrealizational):
	"""Binarize a single tree."""
	if binarization.method == 'default':
		return binarize(tree, factor=binarization.factor,
				tailmarker=binarization.tailmarker,
				horzmarkov=binarization.h, vertmarkov=binarization.v,
				revhorzmarkov=binarization.revh,
				leftmostunary=binarization.leftmostunary,
				rightmostunary=binarization.rightmostunary,
				markhead=binarization.markhead,
				headoutward=binarization.headrules is not None,
				direction=binarization.headrules is not None,
				filterfuncs=(relationalrealizational['ignorefunctions']
					+ (relationalrealizational['adjunctionlabel'], ))
					if relationalrealizational else (),
				labelfun=binarization.labelfun)
	elif binarization.method == 'optimal':
		return Tree.convert(optimalbinarize(tree))
	elif binarization.method == 'optimalhead':
		return Tree.convert(optimalbinarize(
				tree, headdriven=True, h=binarization.h, v=binarization.v))
	return tree


def optimalbinarize(tree, sep='|', headdriven=False,
		h=None, v=1, fun=None):
	"""Recursively binarize a tree, optimizing for given function.

	``v=0`` is not implemented. Setting h to a nonzero integer restricts the
	possible binarizations to head driven binarizations."""
	if h is None:
		tree = canonicalize(Tree.convert(tree))
	return _optimalbinarize(addbitsets(tree), fun or complexityfanout, sep,
			headdriven, h or 999, v, ())


def _optimalbinarize(tree, fun, sep, headdriven, h, v, ancestors):
	"""Helper function for postorder / bottom-up binarization."""
	if not isinstance(tree, Tree):
		return tree
	parentstr = '^<%s>' % (','.join(ancestors[:v - 1])) if v > 1 else ''
	newtree = ImmutableTree(tree.label + parentstr,
		[_optimalbinarize(t, fun, sep, headdriven, h, v,
			(tree.label,) + ancestors) for t in tree])
	newtree.bitset = tree.bitset
	return minimalbinarization(newtree, fun, sep, parentstr=parentstr, h=h,
			head=(len(tree) - 1) if headdriven else None)


def minimalbinarization(tree, score, sep='|', head=None, parentstr='', h=999):
	"""Find optimal binarization according to a scoring function.

	Implementation of Gildea (2010): Optimal parsing strategies for linear
	context-free rewriting systems. http://aclweb.org/anthology/N10-1118

	:param tree: ImmutableTree for which the optimal binarization of its top
		production will be searched. Nodes need to have a .bitset attribute,
		as produced by ``addbitsets()``.
	:param score: a function from binarized trees to scores, where lower is
		better (the scores can be anything else which supports comparisons).
	:param head: an optional index of the head node, specifying it enables
		head-driven binarization (which constrains the possible binarizations).

	>>> tree = '(X (A 0) (B 1) (C 2) (D 3) (E 4))'
	>>> tree2 = binarize(Tree(tree))
	>>> minimalbinarization(addbitsets(tree), complexityfanout, head=2) == tree2
	True
	>>> tree = addbitsets('(A (B1 (t 6) (t 13)) (B2 (t 3) (t 7) (t 10)) '
	... '(B3 (t 1) (t 9) (t 11) (t 14) (t 16)) (B4 (t 0) (t 5) (t 8)))')
	>>> a = minimalbinarization(tree, complexityfanout)
	>>> b = minimalbinarization(tree, fanoutcomplexity)
	>>> print(max(map(complexityfanout, a.subtrees())))
	(14, 6)
	>>> print(max(map(complexityfanout, b.subtrees())))
	(15, 5)"""
	def newproduction(a, b):
		"""Return a new 'production' (here a tree) combining a and b."""
		if head is not None:
			siblings = (nonterms[a] | nonterms[b])[:h]
		else:
			siblings = getbits(nonterms[a] | nonterms[b])
		newlabel = '%s%s<%s>%s' % (tree.label, sep,
				','.join(labels[x] for x in siblings), parentstr)
		new = ImmutableTree(newlabel, [a, b])
		new.bitset = a.bitset | b.bitset
		return new
	if len(tree) <= 2:
		return tree
	# don't bother with optimality if this particular node is not discontinuous
	# do default right factored binarization instead
	elif fanout(tree) == 1 and all(fanout(a) == 1 for a in tree):
		return factorconstituent(tree, sep=sep, h=h)
	from .plcfrs import Agenda
	labels = [a.label for a in tree]
	# the four main datastructures:
	# the agenda is a priority queue of partial binarizations to explore
	# the first complete binarization that is dequeued is the optimal one
	agenda = Agenda()
	# the working set contains all the optimal partial binarizations
	# keys are binarizations, values are their scores
	workingset = {}
	# for each of the optimal partial binarizations, this dictionary has
	# a bitset that describes which non-terminals from the input it covers
	nonterms = {}
	# reverse lookup table for nonterms (from bitsets to binarizations)
	revnonterms = {}
	# the goal is a bitset that covers all non-terminals of the input
	goal = (1 << len(tree)) - 1
	if head is None:
		for n, a in enumerate(tree):
			nonterms[a] = 1 << n
			revnonterms[nonterms[a]] = a
			workingset[a] = score(a) + (0,)
			agenda[a] = workingset[a]
	else:
		# head driven binarization:
		# add all non-head nodes to the working set,
		# add all combinations of non-head nodes with head to agenda
		# caveat: Crescenzi et al. (2011) show that this problem is NP hard.
		# http://aclweb.org/anthology/P11-1046
		hd = tree[head]
		goal = OrderedSet(range(len(tree)))
		for n, a in enumerate(tree):
			nonterms[a] = OrderedSet([n])
			revnonterms[nonterms[a]] = a
			if n != head:
				workingset[a] = score(a) + (0,)
		for n, a in enumerate(tree):
			if n == head:
				continue
			# (add initial unary here)
			p = newproduction(a, hd)
			x = score(p)
			agenda[p] = workingset[p] = x + (x[0],)
			nonterms[p] = nonterms[a] | nonterms[hd]
			revnonterms[nonterms[p]] = p
	while agenda:
		p, x = agenda.popitem()
		if nonterms[p] == goal:
			# (add final unary here)
			p = ImmutableTree(tree.label, p[:])
			p.bitset = tree.bitset
			return p
		for p1, y in list(workingset.items()):
			if p1 not in workingset:
				continue
			# this is inefficient. we should have a single query for all
			# items not overlapping with p
			elif nonterms[p] & nonterms[p1]:
				continue
			# if we do head-driven binarization, add one nonterminal at a time
			if head is None:
				p2 = newproduction(p, p1)
				p2nonterms = nonterms[p] | nonterms[p1]
			elif len(nonterms[p1]) == 1:
				p2 = newproduction(p1, p)
				p2nonterms = nonterms[p1] | nonterms[p]
			elif len(nonterms[p]) == 1:
				p2 = newproduction(p, p1)
				p2nonterms = nonterms[p] | nonterms[p1]
			else:
				continue
			scorep2 = score(p2)
			# important: the score is the maximum score up till now
			x2 = max((scorep2, y[:-1], x[:-1]))
			# add the sum of all previous parsing complexities as last item
			x2 += (scorep2[0] + x[-1] + y[-1],)
			# if new or better:
			# should we allow item when score is equal?
			if (p2nonterms not in revnonterms
				or workingset[revnonterms[p2nonterms]] > x2):
				if p2nonterms in revnonterms:
					a = revnonterms[p2nonterms]
					del nonterms[a], workingset[a]
					if a in agenda:
						del agenda[a]
				nonterms[p2] = p2nonterms
				revnonterms[p2nonterms] = p2
				agenda[p2] = workingset[p2] = x2
	raise ValueError('agenda exhausted without finding binarization.')


def fanout(tree):
	"""Return fan-out of constituent. Requires ``bitset`` attribute."""
	return bitfanout(tree.bitset) if isinstance(tree, Tree) else 1


def complexity(tree):
	"""The degree of the time complexity of parsing with this rule.

	Cf. Gildea (2010). http://aclweb.org/anthology/N10-1118"""
	return fanout(tree) + sum(map(fanout, tree))


def complexityfanout(tree):
	"""Return a tuple with the complexity and fan-out of a subtree."""
	return (fanout(tree) + sum(map(fanout, tree)), fanout(tree))


def fanoutcomplexity(tree):
	"""Return a tuple with the fan-out and complexity of a subtree."""
	return (fanout(tree), fanout(tree) + sum(map(fanout, tree)))


def contsets(nodes):
	"""Partition children into continuous subsets.

	>>> tree = Tree('(VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) (VVPP 5))')
	>>> for a in contsets(tree):
	...		print(' / '.join('%s' % b for b in a))
	(PP (APPR 0) (ART 1) (NN 2))
	(CARD 4) / (VVPP 5)"""
	rng, subset = -1, []
	mins = {min(a.leaves()) if isinstance(a, Tree) else a: a for a in nodes}
	leaves = [a for child in nodes for a in child.leaves()]
	for a in sorted(leaves):
		if rng >= 0 and a != rng + 1:
			yield subset
			subset = []
		if a in mins:
			subset.append(mins[a])
		rng = a
	if subset:
		yield subset


def abbr(childlabels):
	"""Reduce sequences of identical labels.

	>>> print(' '.join(abbr(['mwp', 'mwp', 'mwp', 'mwp'])))
	mwp+"""
	result, inrun = [], ''
	for a, b in zip(childlabels, childlabels[1:] + [None]):
		if a == b:
			inrun = '+'
		else:
			result.append(a + inrun)
			inrun = ''
	return result


def getbits(bitset):
	"""Iterate over the indices of set bits in a bitset."""
	n = 0
	while bitset:
		if bitset & 1:
			yield n
		elif not bitset:
			break
		bitset >>= 1
		n += 1


def addbitsets(tree):
	"""Turn tree into an ImmutableTree and add bitset attribute.

	The bitset attribute is a Python integer corresponding to the information
	that leaves() would return for that node."""
	if isinstance(tree, (str, unicode)):
		result = ImmutableTree(tree)
	elif isinstance(tree, ImmutableTree):
		result = tree
	elif isinstance(tree, Tree):
		result = tree.freeze()
	else:
		raise ValueError('expected string or tree object')
	for a in result.subtrees():
		a.bitset = sum(1 << n for n in a.leaves())
	return result


def getyf(left, right):
	"""Return the yield function for two subtrees with bitsets.

	:returns: string representation of yield function; e.g., ';01,10'."""
	result = [';']
	cur = ','
	bits = left.bitset.bit_length()
	if right is not None:
		bits = max(bits, right.bitset.bit_length())
	for n in range(bits):
		mask = 1 << n
		if left.bitset & mask:
			if cur != '0':
				cur = '0'
				result.append(cur)
		elif right.bitset & mask:
			if cur != '1':
				cur = '1'
				result.append(cur)
		elif cur != ',':
			cur = ','
			result.append(cur)
	return ''.join(result)


__all__ = ['binarize', 'unbinarize', 'collapseunary', 'introducepreterminals',
		'factorconstituent', 'markovthreshold', 'splitdiscnodes',
		'mergediscnodes', 'addfanoutmarkers', 'removefanoutmarkers',
		'canonicalize', 'optimalbinarize', 'minimalbinarization',
		'fanout', 'complexity', 'complexityfanout', 'fanoutcomplexity',
		'contsets', 'abbr', 'getbits', 'addbitsets', 'getyf',
		'treebankfanout', 'handledisc', 'removeemptynodes', 'removeterminals',
		'binarizetree']
