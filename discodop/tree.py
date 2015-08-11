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
if sys.version_info[0] > 2:
	unicode = str  # pylint: disable=redefined-builtin


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
		return not self == other

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

	def subtrees(self, condition=None):
		"""Traverse and generate subtrees of this tree in depth-first order.

		:param condition: a function to filter which nodes are generated.

		NB: store traversal as list before any structural modifications."""
		if condition is None or condition(self):
			yield self
		for child in self.children:
			if isinstance(child, Tree):
				for subtree in child.subtrees(condition):
					yield subtree

	def postorder(self, condition=None):
		"""A generator that does a postorder traversal of this tree.

		Similar to Tree.subtrees() which does a preorder traversal.
		NB: store traversal as list before any structural modifications."""
		for child in self.children:
			if isinstance(child, Tree):
				for subtree in child.postorder(condition):
					yield subtree
		if condition is None or condition(self):
			yield self

	def pos(self):
		"""
		:returns: a list of tuples containing leaves and pre-terminals
			(part-of-speech tags). The order reflects the order of the tree's
			hierarchical structure."""
		pos = []
		for child in self.children:
			if isinstance(child, Tree):
				pos.extend(child.pos())
			else:
				pos.append((child, self.label))
		return pos

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
		for i in range(len(start_treepos)):
			if i == len(end_treepos) or start_treepos[i] != end_treepos[i]:
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
		from discodop.treedraw import DrawTree
		return DrawTree(self, ['%d' % a for a in self.leaves()]).text()

	def _repr_svg_(self):
		"""Return a rich representation for IPython notebook."""
		from discodop.treedraw import DrawTree
		return DrawTree(self, ['%d' % a for a in self.leaves()]).svg()


class ImmutableTree(Tree):
	"""A tree which may not be modified.; has a hash() value.

	NB: the ``label`` and ``children`` attributes should not be modified, but
	this is not enforced."""
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

	def __iadd__(self):
		raise ValueError('ImmutableTrees may not be modified')

	def __imul__(self):
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
	structure. Two subclasses are defined: ParentedTree, MultiParentedTree

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
		while node.parent is not None:
			node = node.parent
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


def slice_bounds(seq, slice_obj, allow_step=False):
	"""Calculate the effective (start, stop) bounds of a slice.

	Takes into account ``None`` indices and negative indices.

	:returns: tuple ``(start, stop, 1)``, s.t. ``0 <= start <= stop <= len(seq)``
	:raises ValueError: if slice_obj.step is not None.
	:param allow_step: If true, then the slice object may have a non-None step.
		If it does, then return a tuple (start, stop, step)."""
	start, stop = (slice_obj.start, slice_obj.stop)
	if allow_step:
		slice_obj.step = 1 if slice_obj.step is None else slice_obj.step
		# Use a recursive call without allow_step to find the slice
		# bounds. If step is negative, then the roles of start and
		# stop (in terms of default values, etc), are swapped.
		if slice_obj.step < 0:
			start, stop, _ = slice_bounds(seq, slice(stop, start))
		else:
			start, stop, _ = slice_bounds(seq, slice(start, stop))
		return start, stop, slice_obj.step
	elif slice_obj.step not in (None, 1):
		raise ValueError('slices with steps are not supported by %s' %
				seq.__class__.__name__)
	start = 0 if start is None else start
	stop = len(seq) if stop is None else stop
	start = max(0, len(seq) + start) if start < 0 else start
	stop = max(0, len(seq) + stop) if stop < 0 else stop
	if stop > 0:  # Make sure stop doesn't go past the end of the list.
		try:  # Avoid calculating len(seq), may be expensive for lazy sequences
			seq[stop - 1]
		except IndexError:
			stop = len(seq)
	start = min(start, stop)
	return start, stop, 1

__all__ = ['Tree', 'ImmutableTree', 'ParentedTree', 'ImmutableParentedTree',
		'slice_bounds']
