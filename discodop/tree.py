"""Various Tree objects for representing syntax or morphological trees."""
# This is an adaptation of the original tree.py file from NLTK.
# Removed: probabilistic & multi-parented trees, binarization,
# reading off CFG productions, &c.
# Original notice:
# Natural Language Toolkit: Text Trees
#
# Copyright (C) 2001-2010 NLTK Project
# Author: Edward Loper <edloper@gradient.cis.upenn.edu>
#         Steven Bird <sb@csse.unimelb.edu.au>
#         Nathan Bodenstab <bodenstab@cslu.ogi.edu> (tree transforms)
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT
from __future__ import division, print_function, absolute_import, \
		unicode_literals
import re
import sys
from itertools import count
from operator import itemgetter
from collections import defaultdict, OrderedDict
if sys.version_info[0] == 2:
	from cgi import escape as htmlescape
else:
	from html import escape as htmlescape
	unicode = str  # pylint: disable=redefined-builtin
from .util import slice_bounds, ANSICOLOR

PTBPUNC = {'-LRB-', '-RRB-', '-LCB-', '-RCB-', '-LSB-', '-RSB-', '-NONE-'}
FRONTIERNTRE = re.compile(r' \)')
SUPERFLUOUSSPACERE = re.compile(r'\)\s+(?=\))')
INDEXRE = re.compile(r' ([0-9]+)\b')
# regex to check if the tree contains any terminals not prefixed by indices
STRTERMRE = re.compile(r' (?![0-9]+=)[^()]*\s*\)')

try:
	from .bit import fanout as bitfanout
except ImportError:
	def bitfanout(arg):
		"""Slower version."""
		prev, result = arg, 0
		while arg:
			arg &= arg - 1
			if ((prev - arg) << 1) & prev == 0:
				result += 1
			prev = arg
		return result


class Tree(object):
	"""A mutable, labeled, n-ary tree structure.

	Each Tree represents a single hierarchical grouping of leaves and subtrees.
	A tree's children are encoded as a list of leaves and subtrees, where
	a leaf is a basic (non-tree) value; and a subtree is a nested Tree. Any
	other properties that a Tree defines are known as node properties, and are
	used to add information about individual hierarchical groupings. For
	example, syntax trees use a label property to label syntactic constituents
	with phrase labels, such as "NP" and "VP".
	Several Tree methods use tree positions to specify children or descendants
	of a tree. Tree positions are defined as follows:

	- The tree position ``i`` specifies a Tree's ith child.
	- The tree position () specifies the Tree itself.
	- If ``p`` is the tree position of descendant ``d``, then
		``p + (i,)`` specifies the ith child of ``d``.

	i.e., every tree position is either a single index ``i``,
	specifying ``self[i]``; or a sequence ``(i1, i2, ..., iN)``,
	specifying ``self[i1][i2]...[iN]``.

	The constructor can be called in two ways:

	- ``Tree(label, children)`` constructs a new tree with the specified label
		and list of children.
	- ``Tree(s)`` constructs a new tree by parsing the string s. Equivalent to
		calling the class method ``Tree.parse(s)``.
		NB: expects integers as leaves by default;
		use ``Tree.parse(s, parse_leaf=None)`` to interpret leaves as strings.
	"""

	# NB: _parent is only used by ParentedTree subclass, but __slots__
	# does not work with multiple inheritance.
	# pylint believes unicode strings in slots attributes are illegal
	__slots__ = ('label', 'children', 'source', 'head', '_parent')

	def __new__(cls, label_or_str=None, children=None):
		if label_or_str is None:
			return object.__new__(cls)  # used by copy.deepcopy
		if children is None:
			if not isinstance(label_or_str, (str, unicode)):
				raise TypeError("%s: Expected a label and child list "
						"or a single string; got: %s" % (
						cls.__name__, type(label_or_str)))
			return cls.parse(label_or_str)
		if (isinstance(children, (str, unicode)) or
				not hasattr(children, '__iter__')):
			raise TypeError("%s() argument 2 should be a list, not a "
					"string" % cls.__name__)
		return object.__new__(cls)

	def __init__(self, label_or_str, children=None):
		# Because __new__ may delegate to Tree.parse(), the __init__
		# method may end up getting called more than once (once when
		# constructing the return value for Tree.parse; and again when
		# __new__ returns). We therefore check if `children` is None
		# (which will cause __new__ to call Tree.parse()); if so, then
		# __init__ has already been called once, so just return.
		if children is None:
			return
		# list.__init__(self, children)
		self.label = label_or_str
		self.children = list(children)
		self.source = None
		self.head = False

	# === Comparison operators ==================================
	def __eq__(self, other):
		if not isinstance(other, Tree):
			return False
		return (self.label == other.label
				and self.children == other.children)

	def __ne__(self, other):
		return not self.__eq__(other)

	def __lt__(self, other):
		if not isinstance(other, Tree):
			return False
		return (self.label < other.label
				or self.children < other.children)

	def __le__(self, other):
		if not isinstance(other, Tree):
			return False
		return (self.label <= other.label
				or self.children <= other.children)

	def __gt__(self, other):
		if not isinstance(other, Tree):
			return True
		return (self.label > other.label
				or self.children > other.children)

	def __ge__(self, other):
		if not isinstance(other, Tree):
			return False
		return (self.label >= other.label
				or self.children >= other.children)

	# === Delegated list operations ==============================
	def append(self, child):
		"""Append ``child`` to this node."""
		self.children.append(child)

	def extend(self, children):
		"""Extend this node's children with an iterable."""
		self.children.extend(children)

	def insert(self, index, child):
		"""Insert child at integer index."""
		self.children.insert(index, child)

	def pop(self, index=-1):
		"""Remove child at specified integer index (or default to last)."""
		return self.children.pop(index)

	def remove(self, child):
		"""Remove child, based on equality."""
		self.children.remove(child)

	def index(self, child):
		"""Return index of child, based on equality."""
		return self.children.index(child)

	def __iter__(self):
		return self.children.__iter__()

	def __len__(self):
		return self.children.__len__()

	# === Disabled list operations ==============================
	def __mul__(self, _):
		raise TypeError('Tree does not support multiplication')

	def __rmul__(self, _):
		raise TypeError('Tree does not support multiplication')

	def __add__(self, _):
		raise TypeError('Tree does not support addition')

	def __radd__(self, _):
		raise TypeError('Tree does not support addition')

	# === Indexing (with support for tree positions) ============
	def __getitem__(self, index):
		if isinstance(index, (int, slice)):
			return self.children.__getitem__(index)
		else:
			if len(index) == 0:
				return self
			elif len(index) == 1:
				return self[int(index[0])]
			return self[int(index[0])][index[1:]]

	def __setitem__(self, index, value):
		if isinstance(index, (int, slice)):
			return self.children.__setitem__(index, value)
		else:
			if len(index) == 0:
				raise IndexError('The tree position () may not be '
						'assigned to.')
			elif len(index) == 1:
				self[index[0]] = value
			else:
				self[index[0]][index[1:]] = value

	def __delitem__(self, index):
		if isinstance(index, (int, slice)):
			return self.children.__delitem__(index)
		else:
			if len(index) == 0:
				raise IndexError('The tree position () may not be deleted.')
			elif len(index) == 1:
				del self[index[0]]
			else:
				del self[index[0]][index[1:]]

	# === Basic tree operations =================================
	def leaves(self):
		""":returns: list containing this tree's leaves.

		The order reflects the order of the tree's hierarchical structure."""
		leaves = []
		for child in self.children:
			if isinstance(child, Tree):
				leaves.extend(child.leaves())
			else:
				leaves.append(child)
		return leaves

	def height(self):
		""":returns: The longest distance from this node to a leaf node.

		- The height of a tree containing no children is 1;
		- the height of a tree containing only leaves is 2;
		- the height of any other tree is one plus the maximum of its
			children's heights."""
		max_child_height = 0
		for child in self.children:
			if isinstance(child, Tree):
				max_child_height = max(max_child_height, child.height())
			else:
				max_child_height = max(max_child_height, 1)
		return 1 + max_child_height

	def subtrees(self, condition=None):
		"""Yield subtrees of this tree in depth-first, pre-order traversal.

		:param condition: a function ``Tree -> bool`` to filter which nodes are
			yielded (does not affect whether children are visited).

		NB: store traversal as list before any structural modifications."""
		# Non-recursive version
		agenda = [self]
		while agenda:
			node = agenda.pop()
			if isinstance(node, Tree):
				if condition is None or condition(node):
					yield node
				agenda.extend(node[::-1])

	def postorder(self, condition=None):
		"""A generator that does a post-order traversal of this tree.

		Similar to Tree.subtrees() which does a pre-order traversal.
		NB: store traversal as list before any structural modifications.

		:yields: Tree objects."""
		# Non-recursive; requires no parent pointers but uses O(n) space.
		agenda = [self]
		visited = set()
		while agenda:
			node = agenda[-1]
			if not isinstance(node, Tree):
				agenda.pop()
			elif id(node) in visited:
				agenda.pop()
				if condition is None or condition(node):
					yield node
			else:
				agenda.extend(node[::-1])
				visited.add(id(node))

	def pos(self, nodes=False):
		"""Collect preterminals (part-of-speech nodes).

		:param nodes: if True, return a sequence of preterminal Tree objects
			instead of tuples. NB: a preterminal that dominates multiple
			terminals is returned once.
		:returns: a list of tuples containing leaves and pre-terminals
			(part-of-speech tags). The order reflects the order of the tree's
			hierarchical structure. A preterminal that dominates multiple
			terminals generates a tuple for each terminal."""
		agenda = list(self[::-1])
		result = []
		while agenda:
			node = agenda.pop()
			if not isinstance(node, Tree):
				continue
			agenda.extend(node[::-1])
			for child in node:
				if not isinstance(child, Tree):
					if nodes:
						result.append(node)
						break
					result.append((child, node.label))
		return result

	def treepositions(self, order='preorder'):
		""":param order: One of preorder, postorder, bothorder, leaves."""
		positions = []
		if order in ('preorder', 'bothorder'):
			positions.append(())
		for i, child in enumerate(self.children):
			if isinstance(child, Tree):
				childpos = child.treepositions(order)
				positions.extend((i, ) + p for p in childpos)
			else:
				positions.append((i, ))
		if order in ('postorder', 'bothorder'):
			positions.append(())
		return positions

	def leaf_treeposition(self, index):
		"""
		:returns: The tree position of the index-th leaf in this tree;
			i.e., if ``tp=self.leaf_treeposition(i)``, then
			``self[tp]==self.leaves()[i]``.
		:raises IndexError: if this tree contains fewer than ``index + 1``
			leaves, or if ``index < 0``."""
		if index < 0:
			raise IndexError('index must be non-negative')
		stack = [(self, ())]
		while stack:
			value, treepos = stack.pop()
			if not isinstance(value, Tree):
				if index == 0:
					return treepos
				else:
					index -= 1
			else:
				for i in range(len(value) - 1, -1, -1):
					stack.append((value[i], treepos + (i, )))
		raise IndexError('index must be less than or equal to len(self)')

	def treeposition_spanning_leaves(self, start, end):
		"""
		:returns: The tree position of the lowest descendant of this tree
			that dominates ``self.leaves()[start:end]``.
		:raises ValueError: if ``end <= start``."""
		if end <= start:
			raise ValueError('end must be greater than start')
		# Find the tree positions of the start & end leaves,
		# and take the longest common subsequence.
		start_treepos = self.leaf_treeposition(start)
		end_treepos = self.leaf_treeposition(end - 1)
		# Find the first index where they mismatch:
		for i, a in enumerate(start_treepos):
			if i == len(end_treepos) or a != end_treepos[i]:
				return start_treepos[:i]
		return start_treepos

	# === Convert, copy =========================================
	@classmethod
	def convert(cls, val):
		"""Convert a tree between different subtypes of Tree.

		:param cls: the class that will be used for the new tree.
		:param val: The tree that should be converted."""
		if isinstance(val, Tree):
			children = [cls.convert(child) for child in val]
			tree = cls(val.label, children)
			tree.source = val.source
			tree.head = val.head
			if (isinstance(val, ImmutableTree)
					and isinstance(cls, ImmutableTree)):
				tree.bitset = val.bitset  # pylint: disable=W0201,E0237
			return tree
		return val

	def copy(self, deep=False):
		"""Create a copy of this tree."""
		if not deep:
			return self.__class__(self.label, self)
		return self.__class__.convert(self)

	def _frozen_class(self):
		"""The frozen version of this class."""
		return ImmutableTree

	def freeze(self, leaf_freezer=None):
		""":returns: an immutable version of this tree."""
		frozen_class = self._frozen_class()
		if leaf_freezer is None:
			newcopy = frozen_class.convert(self)
		else:
			newcopy = self.copy(deep=True)
			for pos in newcopy.treepositions('leaves'):
				newcopy[pos] = leaf_freezer(newcopy[pos])
			newcopy = frozen_class.convert(newcopy)
		hash(newcopy)  # Make sure the leaves are hashable.
		return newcopy

	# === Parsing ===============================================
	@classmethod
	def parse(cls, s, brackets='()', parse_label=None, parse_leaf=int,
			label_pattern=None, leaf_pattern=None):
		"""Parse a bracketed tree string and return the resulting tree.
		Trees are represented as nested bracketings, such as:
		``(S (NP (NNP John)) (VP (V runs)))``

		:param s: The string to parse
		:param brackets: The two bracket characters used to mark the
			beginning and end of trees and subtrees.
		:param parse_label, parse_leaf: If specified, these functions are
			applied to the substrings of s corresponding to labels and leaves
			(respectively) to obtain the values for those labels and leaves.
			They should have the following signature: parse_label(str) -> value
		:param label_pattern, leaf_pattern: Regular expression patterns used to
			find label and leaf substrings in s. By default, both label and
			leaf patterns are defined to match any sequence of non-whitespace
			non-bracket characters.
		:returns: A tree corresponding to the string representation s.
			If this class method is called using a subclass of Tree, then it
			will return a tree of that type."""
		if not isinstance(brackets, (str, unicode)) or len(brackets) != 2:
			raise TypeError('brackets must be a length-2 string')
		if re.search(r'\s', brackets):
			raise TypeError('whitespace brackets not allowed')
		# Construct a regexp that will tokenize the string.
		open_b, close_b = brackets[:1], brackets[1:]
		open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
		if label_pattern is None:
			label_pattern = r'[^\s%s%s]+' % (open_pattern, close_pattern)
		if leaf_pattern is None:
			leaf_pattern = r'[^\s%s%s]+' % (open_pattern, close_pattern)
		token_re = re.compile(r'%s\s*(%s)?|%s|(%s)' % (
				open_pattern, label_pattern, close_pattern, leaf_pattern))
		# Walk through each token, updating a stack of trees.
		stack = [(None, [])]  # list of (label, children) tuples
		for match in token_re.finditer(s):
			token = match.group()
			if token[0] == open_b:  # Beginning of a tree/subtree
				if len(stack) == 1 and len(stack[0][1]) > 0:
					cls._parse_error(s, match, 'end-of-string')
				label = token[1:].lstrip()
				if parse_label is not None:
					label = parse_label(label)
				stack.append((label, []))
			elif token == close_b:  # End of a tree/subtree
				if len(stack) == 1:
					if len(stack[0][1]) == 0:
						cls._parse_error(s, match, open_b)
					else:
						cls._parse_error(s, match, 'end-of-string')
				label, children = stack.pop()
				stack[-1][1].append(cls(label, children))
			else:  # Leaf node
				if len(stack) == 1:
					cls._parse_error(s, match, open_b)
				if parse_leaf is not None:
					token = parse_leaf(token)
				stack[-1][1].append(token)
		# check that we got exactly one complete tree.
		if len(stack) > 1:
			cls._parse_error(s, 'end-of-string', close_b)
		elif len(stack[0][1]) == 0:
			cls._parse_error(s, 'end-of-string', open_b)
		else:
			assert stack[0][0] is None and len(stack[0][1]) == 1
		tree = stack[0][1][0]
		return tree

	@classmethod
	def _parse_error(cls, orig, match, expecting):
		"""Display a friendly error message when parsing a tree string fails.

		:param orig: The string we're parsing.
		:param match: regexp match of the problem token.
		:param expecting: what we expected to see instead."""
		# Construct a basic error message
		if match == 'end-of-string':
			pos, token = len(orig), 'end-of-string'
		else:
			pos, token = match.start(), match.group()
		msg = '%s.parse(): expected %r but got %r\n%sat index %d.' % (
			cls.__name__, expecting, token, ' ' * 12, pos)
		# Add a display showing the error token itself:
		s = orig.replace('\n', ' ').replace('\t', ' ')
		offset = pos
		if len(s) > pos + 10:
			s = s[:pos + 10] + '...'
		if pos > 10:
			s = '...' + s[pos - 10:]
			offset = 13
		msg += '\n%s"%s"\n%s^' % (' ' * 16, s, ' ' * (17 + offset))
		msg += '\n%s' % orig
		raise ValueError(msg)

	# === String Representations ================================
	def __repr__(self):
		childstr = ", ".join(repr(c) for c in self)
		return '%s(%r, [%s])' % (self.__class__.__name__, self.label, childstr)

	def __str__(self):
		return self._pprint_flat('()')

	def pprint(self, margin=70, indent=0, brackets='()'):
		"""	:returns: A pretty-printed string representation of this tree.
		:param margin: The right margin at which to do line-wrapping.
		:param indent: The indentation level at which printing begins. This
			number is used to decide how far to indent subsequent lines."""
		# Try writing it on one line.
		s = self._pprint_flat(brackets)
		if len(s) + indent < margin:
			return s
		# If it doesn't fit on one line, then write it on multi-lines.
		if isinstance(self.label, (str, unicode)):
			s = '%s%s' % (brackets[0], self.label)
		else:
			s = '%s%r' % (brackets[0], self.label)
		for child in self.children:
			if isinstance(child, Tree):
				s += '\n' + ' ' * (indent + 2) + child.pprint(margin,
						indent + 2, brackets)
			elif isinstance(child, tuple):
				s += '\n' + ' ' * (indent + 2) + '/'.join(child)
			elif isinstance(child, (str, unicode)):
				s += '\n' + ' ' * (indent + 2) + '%s' % child
			else:
				s += '\n' + ' ' * (indent + 2) + '%r' % child
		return s + brackets[1]

	def _pprint_flat(self, brackets):
		"""Pretty-printing helper function."""
		childstrs = []
		for child in self.children:
			if isinstance(child, Tree):
				childstrs.append(child._pprint_flat(brackets))
			elif isinstance(child, (str, unicode)) or child is None:
				childstrs.append(child or '')
			else:
				childstrs.append(repr(child))
		if isinstance(self.label, (str, unicode)):
			return '%s%s %s%s' % (brackets[0], self.label,
									' '.join(childstrs), brackets[1])
		else:
			return '%s%r %s%s' % (brackets[0], self.label,
					' '.join(childstrs), brackets[1])

	def draw(self):
		""":returns: an ASCII art visualization of tree."""
		return DrawTree(self, ['%d' % a for a in self.leaves()]).text()

	def _repr_svg_(self):
		"""Return a rich representation for IPython notebook."""
		return DrawTree(self, ['%d' % a for a in self.leaves()]).svg()


class ImmutableTree(Tree):
	"""A tree which may not be modified.; has a hash() value.

	NB: the ``label`` and ``children`` attributes should not be modified, but
	this is not enforced.

	This class has the following optimizations compared to Tree objects:
		- precomputed ``hash()`` value
		- precomputed ``leaves()`` value of each node
		- a bitset attribute recording the leaf indices dominated by each node
	"""
	__slots__ = ('_hash', '_leaves', 'bitset')

	def __init__(self, label_or_str, children=None):
		if children is None:
			return  # see note in Tree.__init__()
		self._hash = self._leaves = None
		super(ImmutableTree, self).__init__(label_or_str, children)
		# Precompute our hash value. This ensures that we're really
		# immutable. It also means we only have to calculate it once.
		try:
			self._hash = hash((self.label, tuple(self)))
		except (TypeError, ValueError) as err:
			raise ValueError('ImmutableTree\'s label and children '
					'must be immutable:\n%s %r\n%r' % (self.label, self, err))
		else:
			# self._leaves = Tree.leaves(self)
			self._addleaves()
			try:
				self.bitset = sum(1 << n for n in self._leaves)
			except TypeError as err:
				self.bitset = None

	def _addleaves(self):
		"""Set leaves attribute of this node and its descendants."""
		leaves = []
		for child in self.children:
			if isinstance(child, Tree):
				if child._leaves is None:
					child._addleaves()
				leaves.extend(child._leaves)
			else:
				leaves.append(child)
		self._leaves = leaves

	def leaves(self):
		return self._leaves

	def __setitem__(self, _index, _value):
		raise ValueError('ImmutableTrees may not be modified')

	def __setslice__(self, _start, _stop, _value):
		raise ValueError('ImmutableTrees may not be modified')

	def __delitem__(self, _index):
		raise ValueError('ImmutableTrees may not be modified')

	def __delslice__(self, _start, _stop):
		raise ValueError('ImmutableTrees may not be modified')

	def __iadd__(self, _):
		raise ValueError('ImmutableTrees may not be modified')

	def __imul__(self, _):
		raise ValueError('ImmutableTrees may not be modified')

	def append(self, _):
		raise ValueError('ImmutableTrees may not be modified')

	def extend(self, _):
		raise ValueError('ImmutableTrees may not be modified')

	def pop(self, _=None):
		raise ValueError('ImmutableTrees may not be modified')

	def remove(self, _):
		raise ValueError('ImmutableTrees may not be modified')

	def __hash__(self):
		return self._hash


class ParentedTree(Tree):
	"""A Tree that maintains parent pointers for single-parented trees.

	The parent pointers are updated whenever any change is made to a tree's
	structure.

	The following read-only property values are automatically updated
	whenever the structure of a parented tree is modified: parent,
	parent_index, left_sibling, right_sibling, root, treeposition.
	Each ParentedTree may have at most one parent; i.e., subtrees may not be
	shared. Any attempt to reuse a single ParentedTree as a child of more than
	one parent (or as multiple children of the same parent) will cause a
	ValueError exception to be raised. ParentedTrees should never be used in
	the same tree as Trees or MultiParentedTrees. Mixing tree implementations
	may result in incorrect parent pointers and in TypeError exceptions.

	The ParentedTree class redefines all operations that modify a
	tree's structure to call two methods, which are used by subclasses to
	update parent information:

	- ``_setparent()`` is called whenever a new child is added.
	- ``_delparent()`` is called whenever a child is removed."""
	__slots__ = ()

	def __init__(self, label_or_str, children=None):
		if children is None:
			return  # see note in Tree.__init__()
		self._parent = None
		super(ParentedTree, self).__init__(label_or_str, children)
		# iterate over self.children, *not* children,
		# because children might be an iterator.
		for i, child in enumerate(self.children):
			if isinstance(child, Tree):
				self._setparent(child, i, dry_run=True)
		for i, child in enumerate(self.children):
			if isinstance(child, Tree):
				self._setparent(child, i)

	def _frozen_class(self):
		return ImmutableParentedTree

	# === Properties =================================================
	def _get_parent_index(self):
		"""The index of this tree in its parent.

		i.e., ptree.parent[ptree.parent_index] is ptree.
		Note that ptree.parent_index is not necessarily equal to
		ptree.parent.index(ptree), since the index() method
		returns the first child that is _equal_ to its argument."""
		if self._parent is None:
			return None
		for i, child in enumerate(self._parent):
			if child is self:
				return i
		raise ValueError('expected to find self in self._parent!')

	def _get_left_sibling(self):
		"""The left sibling of this tree, or None if it has none."""
		parent_index = self._get_parent_index()
		if self._parent and parent_index > 0:
			return self._parent[parent_index - 1]
		return None  # no left sibling

	def _get_right_sibling(self):
		"""The right sibling of this tree, or None if it has none."""
		parent_index = self._get_parent_index()
		if self._parent and parent_index < (len(self._parent) - 1):
			return self._parent[parent_index + 1]
		return None  # no right sibling

	def _get_treeposition(self):
		"""The tree position of this tree, relative to the root of the tree.

		i.e., ptree.root[ptree.treeposition] is ptree."""
		if self._parent is None:
			return ()
		return (self._parent._get_treeposition() +
				(self._get_parent_index(), ))

	def _get_root(self):
		""":returns: the root of this tree."""
		node = self
		while node._parent is not None:
			node = node._parent
		return node

	parent = property(lambda self: self._parent,
			doc="""The parent of this tree, or None if it has no parent.""")
	parent_index = property(_get_parent_index, doc=_get_parent_index.__doc__)
	left_sibling = property(_get_left_sibling, doc=_get_left_sibling.__doc__)
	right_sibling = property(_get_right_sibling, doc=_get_right_sibling.__doc__)
	root = property(_get_root, doc=_get_root.__doc__)
	treeposition = property(_get_treeposition, doc=_get_treeposition.__doc__)

	# === Parent Management ==========================================
	def _delparent(self, child, index):
		"""Update child's parent pointer to not point to self.

		This method is only called if child's type is Tree; i.e., it is not
		called when removing a leaf from a tree. This method is always called
		before the child is actually removed from self's child list.

		:param index: The index of child in self."""
		assert isinstance(child, ParentedTree)
		assert self[index] is child and child._parent is self
		child._parent = None

	def _setparent(self, child, _index, dry_run=False):
		"""Update child's parent pointer to point to self.

		This method is only called if child's type is Tree; i.e., it is not
		called when adding a leaf to a tree. This method is always called
		before the child is actually added to self's child list. Typically, if
		child is a tree, then its type needs to match self's type. This
		prevents mixing of different tree types (single-, multi-, and
		non-parented).

		:param index: The index of child in self.
		:param dry_run: If true, the don't actually set the child's parent
			pointer; just check for any error conditions, and raise an
			exception if one is found.
		:raises TypeError: if child is a tree with an inappropriate type."""
		if not isinstance(child, ParentedTree):
			raise TypeError('Cannot insert a non-ParentedTree '
					'into a ParentedTree')
		if child._parent is not None:
			raise ValueError('Cannot insert a subtree that already '
					'has a parent.')
		if not dry_run:
			child._parent = self

	# === Methods that add/remove children ======================
	# Every method that adds or removes a child must make
	# appropriate calls to _setparent() and _delparent().
	def __delitem__(self, index):
		if isinstance(index, slice):  # del ptree[start:stop]
			start, stop, _ = slice_bounds(self, index)
			# Clear all the children pointers.
			for i in range(start, stop):
				if isinstance(self[i], Tree):
					self._delparent(self[i], i)
			# Delete the children from our child list.
			super(ParentedTree, self).__delitem__(index)
		elif isinstance(index, int):  # del ptree[i]
			if index < 0:
				index += len(self)
			if index < 0:
				raise IndexError('index out of range')
			# Clear the child's parent pointer.
			if isinstance(self[index], Tree):
				self._delparent(self[index], index)
			# Remove the child from our child list.
			super(ParentedTree, self).__delitem__(index)
		elif len(index) == 0:  # del ptree[()]
			raise IndexError('The tree position () may not be deleted.')
		elif len(index) == 1:  # del ptree[(i, )]
			del self[index[0]]
		else:  # del ptree[i1, i2, i3]
			del self[index[0]][index[1:]]

	def __setitem__(self, index, value):
		if isinstance(index, slice):  # ptree[start:stop] = value
			start, stop, _ = slice_bounds(self, index)
			# make a copy of value, in case it's an iterator
			if not isinstance(value, (list, tuple)):
				value = list(value)
			# Check for any error conditions, so we can avoid ending
			# up in an inconsistent state if an error does occur.
			for i, child in enumerate(value):
				if isinstance(child, Tree):
					self._setparent(child, start + i, dry_run=True)
			# clear the child pointers of all parents we're removing
			for i in range(start, stop):
				if isinstance(self[i], Tree):
					self._delparent(self[i], i)
			# set the child pointers of the new children. We do this
			# after clearing *all* child pointers, in case we're e.g.
			# reversing the elements in a tree.
			for i, child in enumerate(value):
				if isinstance(child, Tree):
					self._setparent(child, start + i)
			# finally, update the content of the child list itself.
			super(ParentedTree, self).__setitem__(index, value)
		elif isinstance(index, int):  # ptree[i] = value
			if index < 0:
				index += len(self)
			if index < 0:
				raise IndexError('index out of range')
			# if the value is not changing, do nothing.
			if value is self[index]:
				return
			# Set the new child's parent pointer.
			if isinstance(value, Tree):
				self._setparent(value, index)
			# Remove the old child's parent pointer
			if isinstance(self[index], Tree):
				self._delparent(self[index], index)
			# Update our child list.
			super(ParentedTree, self).__setitem__(index, value)
		elif len(index) == 0:  # ptree[()] = value
			raise IndexError('The tree position () may not be assigned to.')
		elif len(index) == 1:  # ptree[(i, )] = value
			self[index[0]] = value
		else:  # ptree[i1, i2, i3] = value
			self[index[0]][index[1:]] = value

	def append(self, child):
		if isinstance(child, Tree):
			self._setparent(child, len(self))
		super(ParentedTree, self).append(child)

	def extend(self, children):
		for child in children:
			if isinstance(child, Tree):
				self._setparent(child, len(self))
			super(ParentedTree, self).append(child)

	def insert(self, index, child):
		# Handle negative indexes. Note that if index < -len(self),
		# we do *not* raise an IndexError, unlike __getitem__. This
		# is done for consistency with list.__getitem__ and list.index.
		if index < 0:
			index += len(self)
		if index < 0:
			index = 0
		# Set the child's parent, and update our child list.
		if isinstance(child, Tree):
			self._setparent(child, index)
		super(ParentedTree, self).insert(index, child)

	def pop(self, index=-1):
		if index < 0:
			index += len(self)
		if index < 0:
			raise IndexError('index out of range')
		if isinstance(self[index], Tree):
			self._delparent(self[index], index)
		return super(ParentedTree, self).pop(index)

	# NB: like `list`, this is done by equality, not identity!
	# To remove a specific child, use del ptree[i].
	def remove(self, child):
		index = self.index(child)
		if isinstance(self[index], Tree):
			self._delparent(self[index], index)
		super(ParentedTree, self).remove(child)


class ImmutableParentedTree(ImmutableTree, ParentedTree):
	"""Combination of an Immutable and Parented Tree."""
	__slots__ = ()

	def __init__(self, label_or_str, children=None):
		if children is None:
			return  # see note in Tree.__init__()
		super(ImmutableParentedTree, self).__init__(label_or_str, children)


class DiscTree(ImmutableTree):
	"""Wrap an immutable tree with indices as leaves and a sentence.

	Provides hash & equality."""
	__slots__ = ('sent', )

	def __init__(self, tree, sent):
		self.sent = tuple(sent)
		super(DiscTree, self).__init__(tree.label, tuple(
				DiscTree(child, sent) if isinstance(child, Tree) else child
				for child in tree))

	def __eq__(self, other):
		"""Test whether two discontinuous trees are equivalent.

		Assumes canonicalized() ordering."""
		if not isinstance(other, Tree):
			return False
		if self.label != other.label or len(self) != len(other):
			return False
		for a, b in zip(self.children, other.children):
			istree = isinstance(a, Tree)
			if istree != isinstance(b, Tree):
				return False
			elif not istree:
				return self.sent[a] == other.sent[b]
			elif not a.__eq__(b):
				return False
		return True

	def __hash__(self):
		return hash((self.label, ) + tuple(child.__hash__()
				if isinstance(child, Tree) else self.sent[child]
				for child in self.children))

	def __repr__(self):
		return "DisctTree(%r, %r)" % (
				super(DiscTree, self).__repr__(), self.sent)


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

	Interesting reference (not used for this code):
	T. Eschbach et al., Orth. Hypergraph Drawing, Journal of
	Graph Algorithms and Applications, 10(2) 141--157 (2006)149.
	http://jgaa.info/accepted/2006/EschbachGuentherBecker2006.10.2.pdf
	"""

	# each template is a tuple of strings ``(preamble, postamble)``.
	templates = dict(
			latex=(
				'''\\documentclass{article}
\\usepackage[landscape]{geometry}
\\usepackage[utf8]{inputenc}
\\usepackage{helvet,tikz,tikz-qtree}
\\usetikzlibrary{matrix,positioning}
\\tikzset{edge from parent/.style={draw, edge from parent path={
	(\\tikzparentnode.south) -- +(0,-3pt) -| (\\tikzchildnode)}}}
%% NB: preview is optional, to make a cropped pdf
\\usepackage[active,tightpage]{preview} \\setlength{\\PreviewBorder}{0.2cm}
\\begin{document}
\\pagestyle{empty}\\fontfamily{phv}\\selectfont
\\begin{preview}''',
				'\\end{preview}\n\\end{document}'),
			svg=(('<!doctype html>\n<html>\n<head>\n'
					'\t<meta http-equiv="Content-Type" '
					'content="text/html; charset=UTF-8">\n</head>\n<body>'),
					'</body></html>'),
			html=(('<!doctype html>\n<html>\n<head>\n'
					'\t<meta http-equiv="Content-Type" content="text/html; '
					'charset=UTF-8">\n</head>\n<body>\n<pre>'),
					'</pre></body></html>'))

	def __init__(self, tree, sent=None, abbr=False, highlight=(),
			highlightfunc=()):
		self.tree = tree
		self.sent = sent
		if isinstance(tree, (str, unicode)):
			if sent is None:
				self.tree, sent = brackettree(tree)
				self.sent = sent
			else:
				self.tree = Tree.parse(tree, parse_leaf=int)
		if sent is None:
			leaves = self.tree.leaves()
			if (leaves and not any(len(a) == 0 for a in self.tree.subtrees())
					and all(isinstance(a, int) for a in leaves)):
				self.sent = ['%d' % a for a in leaves]
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
				n.label = n.label[:4] + '\u2026'  # unicode '...' ellipsis
		self.sent = [ptbunescape(token) for token in self.sent]
		self.highlight = self.highlightfunc = None
		self.nodes, self.coords, self.edges = self.nodecoords(
				self.tree, self.sent, highlight, highlightfunc)

	def __str__(self):
		if sys.version_info[0] == 2:
			return self.text(unicodelines=True).encode('utf8')
		return self.text(unicodelines=True)

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
		tree = Tree.convert(tree)
		for a in tree.subtrees():
			a.children.sort(key=lambda n: min(n.leaves())
					if isinstance(n, Tree) else n)
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
					color, htmlescape(cat)))
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
		:param html: whether to wrap output in HTML code (default plain text).
		:param ansi: whether to produce colors with ANSI escape sequences
			(only effective when html==False).
		:param leafcolor, nodecolor: specify colors of leaves and phrasal
			nodes; effective when either html or ansi is True.
		:param funccolor, funcsep: if ``funcsep`` is a string, it is taken as a
			separator for function tags; when it occurs, the rest of the label
			is drawn with ``funccolor``.
		:param maxwidth: maximum number of characters before a label starts to
			wrap across multiple lines; pass None to disable."""
		if unicodelines:
			horzline = '\u2500'
			leftcorner = '\u250c'
			rightcorner = '\u2510'
			vertline = ' \u2502 '
			tee = horzline + '\u252C' + horzline
			bottom = horzline + '\u2534' + horzline
			cross = horzline + '\u253c' + horzline
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
					text = [htmlescape(a) for a in text]
				if (n in self.highlight or n in self.highlightfunc) and (
						html or ansi):
					newtext = []
					seensep = False
					# everything before funcsep in nodecolor,
					# after funcsep use funccolor.
					for line in text:
						if (funcsep and not seensep and isinstance(node, Tree)
								and node.label not in PTBPUNC
								and funcsep in line):
							cat, func = line.rsplit(funcsep, 1)
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
						else:
							newtext.append('')
						if func:
							if not seensep:
								newtext[-1] += funcsep
								seensep = True
							if html and n in self.highlightfunc:
								newtext[-1] += '<font color=%s>%s</font>' % (
										funccolor, func)
							elif ansi and n in self.highlightfunc:
								newtext[-1] += '\x1b[%d;1m%s\x1b[0m' % (
										ANSICOLOR[funccolor], func)
							else:
								newtext[-1] += func
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
		result = ['% ' + writediscbrackettree(self.tree, self.sent).rstrip(),
				r'''\begin{tikzpicture}[scale=0.75, align=center,
				text width=1.5cm, inner sep=0mm, node distance=1mm]''',
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
					func = ''
					if isinstance(node, Tree):
						cat = node.label
						color = nodecolor
						if (funcsep and funcsep in node.label
								and node.label not in PTBPUNC):
							cat, func = node.label.split(funcsep, 1)
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
		result = ['% ' + writediscbrackettree(self.tree, self.sent).rstrip(),
				r'''\begin{tikzpicture}[scale=0.75, align=center,
				text width=1.5cm, inner sep=0mm, node distance=1mm]''',
				r'\footnotesize\sffamily',
				r'\path']

		bottom = max(row for row, _ in self.coords.values())
		# write nodes with coordinates
		for n, (row, column) in self.coords.items():
			node = self.nodes[n]
			func = ''
			if isinstance(node, Tree):
				cat = node.label
				color = nodecolor
				if (funcsep and funcsep in node.label
						and node.label not in PTBPUNC):
					cat, func = node.label.split(funcsep, 1)
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
			if isdisc(self.nodes[parent]):
				result.append(
						'\\draw [white, -, line width=6pt] '
						'(n%d)  +(0, %g) -| (n%d);' % (parent, shift, child))
			result.append(
					'\\draw (n%d) -- +(0, %g) -| (n%d);' % (
					parent, shift, child))

		result += [r'\end{tikzpicture}']
		return result

	def tikzqtree(self, nodecolor='blue', leafcolor='red'):
		r"""Produce TiKZ-qtree code to draw a continuous tree.

		To get trees with straight edges, add this in the preamble::

			\tikzset{edge from parent/.style={draw, edge from parent path={
				(\tikzparentnode.south) -- +(0,-3pt) -| (\tikzchildnode)}}}
		"""
		reserved_chars = re.compile('([#$%&~_{}])')

		pprint = self.tree.pprint(indent=6, brackets=('[.', ' ]'))
		escaped = re.sub(reserved_chars, r'\\\1', pprint)
		return '\n'.join([
			'% ' + writebrackettree(self.tree, self.sent).rstrip(),
			'\\begin{tikzpicture}',
			'  \\tikzset{every node/.style={color=%s}, font=\\sf}' % nodecolor,
			'  \\tikzset{every leaf node/.style={color=%s}}' % leafcolor,
			'  \\Tree %s' % escaped,
			'\\end{tikzpicture}'])


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


def frontier(tree, sent, nodecolor='blue', leafcolor='red'):
	"""Return a representation of the frontier of a tree with ANSI colors."""
	return ' '.join(
			'\x1b[%d;1m%s\x1b[0m' % (ANSICOLOR[nodecolor], pos)
			if sent[idx] is None else
			'\x1b[%d;1m%s\x1b[0m' % (ANSICOLOR[leafcolor], sent[idx])
			for idx, pos in sorted(tree.pos()))


def brackettree(treestr):
	"""Parse a single tree presented in (disc)bracket format."""
	if STRTERMRE.search(treestr):  # bracket: terminals are not all indices
		sent, cnt = [], count()

		def substleaf(x):
			"""Collect word and return index."""
			sent.append(unescape(x))
			return next(cnt)

		tree = ParentedTree.parse(FRONTIERNTRE.sub(' #FRONTIER#)',
				SUPERFLUOUSSPACERE.sub(')', treestr)),
				parse_leaf=substleaf)
	else:  # discbracket: terminals have indices
		sent = {}

		def substleaf(x):
			"""Collect word and return index."""
			idx, word = x.split('=', 1)
			idx = int(idx)
			sent[idx] = unescape(word)
			return idx

		tree = ParentedTree.parse(
				SUPERFLUOUSSPACERE.sub(')', treestr),
				parse_leaf=substleaf)
		sent = [sent.get(n, None) for n in range(max(sent) + 1)]
	return tree, sent


def writebrackettree(tree, sent):
	"""Return a tree in bracket notation with words as leaves."""
	return INDEXRE.sub(
			lambda x: ' %s' % escape(sent[int(x.group(1))]),
			str(tree)) + '\n'


def writediscbrackettree(tree, sent):
	"""Return tree in bracket notation with leaves as ``index=word``."""
	return INDEXRE.sub(
			lambda x: ' %s=%s' % (x.group(1), escape(sent[int(x.group(1))])),
			str(tree)) + '\n'


def isdisc(node):
	"""Test whether a particular node has a discontinuous yield.

	i.e., test whether its yield contains two or more non-adjacent strings.
	Nodes can be continuous even if some of their children are
	discontinuous."""
	if not isinstance(node, Tree):
		return False
	elif isinstance(node, ImmutableTree):
		return bitfanout(node.bitset) > 1
	start = prev = None
	for a in sorted(node.leaves()):
		if start is None:
			start = prev = a
		elif a == prev + 1:
			prev = a
		else:
			return True
	return False


def escape(text):
	"""Escape all occurrences of parentheses and replace None with ''."""
	return '' if text is None else text.replace(
			'(', '#LRB#').replace(')', '#RRB#')


def unescape(text):
	"""Reverse escaping of parentheses, frontier spans."""
	return None if text in ('', '#FRONTIER#') else text.replace(
			'#LRB#', '(').replace('#RRB#', ')')


def ptbescape(token):
	"""Escape brackets according to PTB convention in a single token."""
	if token is None:
		return ''
	elif token == '{':
		return '-LCB-'
	elif token == '}':
		return '-RCB-'
	elif token == '[':
		return '-LSB-'
	elif token == ']':
		return '-RSB-'
	return token.replace('(', '-LRB-').replace(')', '-RRB-')


def ptbunescape(token):
	"""Unescape brackets in a single token, including PTB notation."""
	if token in ('', '#FRONTIER#', None):
		return None
	elif token == '-LCB-':
		return '{'
	elif token == '-RCB-':
		return '}'
	elif token == '-LSB-':
		return '['
	elif token == '-RSB-':
		return ']'
	return token.replace('-LRB-', '(').replace('-RRB-', ')').replace(
			'#LRB#', '(').replace('#RRB#', ')')

__all__ = ['Tree', 'ImmutableTree', 'ParentedTree', 'ImmutableParentedTree',
		'DiscTree', 'DrawTree', 'latexlabel', 'frontier', 'brackettree',
		'isdisc', 'escape', 'unescape', 'ptbescape', 'ptbunescape',
		'writebrackettree', 'writediscbrackettree']
