"""Functions related to finding the linguistic head of a constituent."""
import io
import re
from collections import defaultdict, Counter
from .tree import Tree, HEAD, COMPLEMENT, MODIFIER
from .punctuation import ispunct

FIELDS = tuple(range(8))
WORD, LEMMA, TAG, MORPH, FUNC, PARENT, SECEDGETAG, SECEDGEPARENT = FIELDS
HEADRULERE = re.compile(r'^(\S+)\s+(LEFT-TO-RIGHT|RIGHT-TO-LEFT'
		r'|LEFT|RIGHT|LEFTDIS|RIGHTDIS|LIKE)(?:\s+(.*))?$')


def applyheadrules(tree, headrules, modifierrules=None):
	"""Apply head rules and set head attribute of nodes."""
	for node in tree.subtrees(
			lambda n: n and isinstance(n[0], Tree)):
		head = headfinder(node, headrules)
		if head is not None:
			head.type = HEAD
			if modifierrules is not None:
				markmodifiers(node, modifierrules)


def getheadpos(node):
	"""Get head word dominated by this node."""
	child = node
	while True:
		if not child:
			break
		if not isinstance(child[0], Tree):
			return child
		try:
			child = next(a for a in child if a.type == HEAD)
		except StopIteration:
			break
	return None


def readheadrules(filename):
	"""Read a file containing heuristic rules for head assignment.

	Example line: ``s right-to-left vmfin vafin vaimp``, which means
	traverse siblings of an S constituent from right to left, the first child
	with a label of vmfin, vafin, or vaimp will be marked as head."""
	headrules = {}
	with io.open(filename, encoding='utf8') as inp:
		for line in inp:
			line = line.strip().upper()
			if line and not line.startswith("%") and len(line.split()) > 2:
				try:
					label, direction, heads = HEADRULERE.match(line).groups()
				except AttributeError:
					print('no match:', line)
					raise
				if heads is None:
					heads = ''
				headrules.setdefault(label, [])
				if direction == 'LIKE':
					headrules[label].extend(headrules[heads])
				else:
					headrules[label].append((direction, heads.split()))
	return headrules


def headfinder(tree, headrules, headlabels=frozenset({'HD'})):
	"""Use head finding rules to select one child of tree node as head."""
	def find(heads, children):
		"""Match children with possible heads."""
		for head in heads:
			for child in children:
				if (isinstance(child, Tree)
						and child.label.split('[')[0] == head):
					return child

	def invfind(heads, children):
		"""Inverted version of find()."""
		for child in children:
			for head in heads:
				if (isinstance(child, Tree)
						and child.label.split('[')[0] == head):
					return child

	# check if we already have head information:
	for child in tree:
		if child.type == HEAD:
			return child
	for child in tree:
		if (child.source and not headlabels.isdisjoint(
				child.source[FUNC].upper().split('-'))):
			return child
	# apply heuristic rules:
	head = None
	children = tree
	for direction, heads in headrules.get(tree.label, []):
		if direction.startswith('LEFT'):
			children = tree
		elif direction.startswith('RIGHT'):
			children = tree[::-1]
		else:
			raise ValueError('expected RIGHT or LEFT.')
		if direction in ('LEFTDIS', 'RIGHTDIS'):
			head = invfind(heads, children)
		else:
			head = find(heads, children)
		if head is not None:
			break
	if head is None:
		# default head is initial/last nonterminal (depending on direction)
		for child in children:
			if (isinstance(child, Tree)
					and not ispunct(None, child)):
				return child
		return children[0]
	else:  # PTB-specific
		i = tree.index(head)
		if i >= 2 and tree[i - 1].label in {'CC', 'CONJP'}:
			for althead in tree[i - 2::-1]:
				if not ispunct(althead.label, althead):
					return althead
		return head


def readmodifierrules(filename):
	"""Read a file containing heuristic rules for marking modifiers.

	Example line: ``S *-MOD``, which means that for an S
	constituent, any child with the MOD function tag is a modifier.
	A default rule can be specified by using * as the first label, which
	always matches (in addition to another matching rule, if any).
	If none of the rules matches, a non-terminal is assumed to be a complement.
	"""
	modifierrules = {}
	with io.open(filename, encoding='utf8') as inp:
		for line in inp:
			line = line.strip().upper()
			if line and not line.startswith("%") and len(line.split()) > 2:
				label, modifiers = line.split(None, 1)
				if label in modifierrules:
					raise ValueError('duplicate rule for %r (each label'
							' should occur at most once in the file)' % label)
				modifierrules[label] = modifiers.split()
	return modifierrules


def markmodifiers(tree, modifierrules):
	"""Use heuristics to distinguish complements from modifiers.

	Should be applied after heads have been identified."""
	from discodop.treebanktransforms import function
	prev = None
	for child in tree:
		if child.type == HEAD:
			continue
		child.type = COMPLEMENT
		for mod in modifierrules.get(tree.label.split('-', 1)[0], []
				) + modifierrules.get('*', []):
			if ((child.label.split('-', 1)[0] == mod.split('-', 1)[0]
					or mod.split('-', 1)[0] == '*')
					and ('-' not in mod
						or mod.split('-', 1)[1] == '*'
						or function(child) == mod.split('-', 1)[1])):
				child.type = MODIFIER
				break
		if child.label == prev:  # mark enumerations/lists as modifiers
			child.type = MODIFIER
		prev = child.label


def saveheads(tree, tailmarker):
	"""Infer head from binarization and store."""
	if tailmarker:
		for node in tree.subtrees(lambda n: tailmarker in n.label):
			node.type = HEAD
	else:
		# assume head-outward binarization; the last binarized node has the head.
		for node in tree.subtrees(lambda n: '|<' in n.label
				and not any(child.label.startswith(
					n.label[:n.label.index('|<') + 2])
					for child in n)):
			node[-1].type = HEAD


def headstats(trees):
	"""Collect some information useful for writing headrules.

	- ``heads['NP']['NN'] ==`` number of times NN occurs as head of NP.
	- ``pos1['NP'][1] ==`` number of times head of NP is at position 1.
	- ``pos2`` is like pos1, but position is from the right.
	- ``unknown['NP']['NN'] ==`` number of times NP that does not have a head
		dominates an NN."""
	heads, unknown = defaultdict(Counter), defaultdict(Counter)
	pos1, pos2 = defaultdict(Counter), defaultdict(Counter)
	for tree in trees:
		for a in tree.subtrees(lambda x: len(x) > 1):
			for n, b in enumerate(a):
				if b.type == HEAD:
					heads[a.label][b.label] += 1
					pos1[a.label][n] += 1
					pos2[a.label][len(a) - (n + 2)] += 1
					break
			else:
				unknown[a.label].update(b.label for b in a)
	return heads, unknown, pos1, pos2


__all__ = ['getheadpos', 'readheadrules', 'headfinder', 'saveheads',
		'headstats', 'applyheadrules']
