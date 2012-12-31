# Natural Language Toolkit: Text Trees
#
# Copyright (C) 2001-2010 NLTK Project
# Author: Edward Loper <edloper@gradient.cis.upenn.edu>
#         Steven Bird <sb@csse.unimelb.edu.au>
#         Nathan Bodenstab <bodenstab@cslu.ogi.edu> (tree transforms)
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT
#
# This is an adaptation of the original tree.py file from NLTK.
# Probabilistic trees have been removed, as well as the possibility
# to read off CFG productions or draw trees.
# Remaining dependencies have been inlined.
""" Class for representing hierarchical language structures, such as syntax
trees and morphological trees. """

from __future__ import division, print_function, unicode_literals
from collections import defaultdict, deque, OrderedDict
import re, sys
if sys.version[0] >= '3':
	basestring = (str, bytes)

######################################################################
## Trees
######################################################################

class Tree(list):
	""" A hierarchical structure.

	Each Tree represents a single hierarchical grouping of
	leaves and subtrees.  For example, each constituent in a syntax
	tree is represented by a single Tree.

	A tree's children are encoded as a list of leaves and subtrees,
	where a leaf is a basic (non-tree) value; and a subtree is a
	nested Tree.

	Any other properties that a Tree defines are known as
	node properties, and are used to add information about
	individual hierarchical groupings.  For example, syntax trees use a
	label property to label syntactic constituents with phrase tags,
	such as \"NP\" and\"VP\".

	Several Tree methods use tree positions to specify
	children or descendants of a tree.  Tree positions are defined as
	follows:

	- The tree position i specifies a Tree's ith child.
	- The tree position () specifies the Tree itself.
	- If p is the tree position of descendant d, then
		p + (i,) specifies the ith child of d.

	I.e., every tree position is either a single index i,
	specifying self[i]; or a sequence (i1, i2, ...,
	iN), specifying
	self[i1][i2]...[iN]. """
	def __new__(cls, label_or_str=None, children=None):
		if label_or_str is None:
			return list.__new__(cls) # used by copy.deepcopy
		if children is None:
			if not isinstance(label_or_str, basestring):
				raise TypeError("%s: Expected a label and child list "
						"or a single string" % cls.__name__)
			return cls.parse(label_or_str)
		else:
			if (isinstance(children, basestring) or
				not hasattr(children, '__iter__')):
				raise TypeError("%s() argument 2 should be a list, not a "
						"string" % cls.__name__)
			return list.__new__(cls, label_or_str, children)

	def __init__(self, label_or_str, children=None):
		""" Construct a new tree.  This constructor can be called in one
		of two ways:

		- Tree(label, children) constructs a new tree with the
			specified label and list of children.

		- Tree(s) constructs a new tree by parsing the string
			s.  It is equivalent to calling the class method
			Tree.parse(s). """
		# Because __new__ may delegate to Tree.parse(), the __init__
		# method may end up getting called more than once (once when
		# constructing the return value for Tree.parse; and again when
		# __new__ returns).  We therefore check if `children` is None
		# (which will cause __new__ to call Tree.parse()); if so, then
		# __init__ has already been called once, so just return.
		if children is None:
			return

		list.__init__(self, children)
		self.label = label_or_str

	#////////////////////////////////////////////////////////////
	# Comparison operators
	#////////////////////////////////////////////////////////////

	def __eq__(self, other):
		if not isinstance(other, Tree):
			return False
		return self.label == other.label and list.__eq__(self, other)
	def __ne__(self, other):
		return not (self == other)
	def __lt__(self, other):
		if not isinstance(other, Tree):
			return False
		return self.label < other.label or list.__lt__(self, other)
	def __le__(self, other):
		if not isinstance(other, Tree):
			return False
		return self.label <= other.label or list.__le__(self, other)
	def __gt__(self, other):
		if not isinstance(other, Tree):
			return True
		return self.label > other.label or list.__gt__(self, other)
	def __ge__(self, other):
		if not isinstance(other, Tree):
			return False
		return self.label >= other.label or list.__ge__(self, other)

	#////////////////////////////////////////////////////////////
	# Disabled list operations
	#////////////////////////////////////////////////////////////

	def __mul__(self, v):
		raise TypeError('Tree does not support multiplication')
	def __rmul__(self, v):
		raise TypeError('Tree does not support multiplication')
	def __add__(self, v):
		raise TypeError('Tree does not support addition')
	def __radd__(self, v):
		raise TypeError('Tree does not support addition')

	#////////////////////////////////////////////////////////////
	# Indexing (with support for tree positions)
	#////////////////////////////////////////////////////////////

	def __getitem__(self, index):
		if isinstance(index, (int, slice)):
			return list.__getitem__(self, index)
		else:
			if len(index) == 0:
				return self
			elif len(index) == 1:
				return self[int(index[0])]
			else:
				return self[int(index[0])][index[1:]]

	def __setitem__(self, index, value):
		if isinstance(index, (int, slice)):
			return list.__setitem__(self, index, value)
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
			return list.__delitem__(self, index)
		else:
			if len(index) == 0:
				raise IndexError('The tree position () may not be deleted.')
			elif len(index) == 1:
				del self[index[0]]
			else:
				del self[index[0]][index[1:]]

	#////////////////////////////////////////////////////////////
	# Basic tree operations
	#////////////////////////////////////////////////////////////

	def leaves(self):
		""" @return: a list containing this tree's leaves.
			The order reflects the order of the
			leaves in the tree's hierarchical structure.
		@rtype: list """
		leaves = []
		for child in self:
			if isinstance(child, Tree):
				leaves.extend(child.leaves())
			else:
				leaves.append(child)
		return leaves

	#def leaves(self):
	#	""" @return: a list containing this tree's leaves.
	#		The order reflects the order of the
	#		leaves in the tree's hierarchical structure.
	#	@rtype: list
	#	Non-recursive version. Does not seem to have a speed advantage,
	#	but doesn't clutter profiles with enormous amounts of recursive
	#	calls ...
	#	"""
	#	queue = deque(self)
	#	theleaves = []
	#	while queue:
	#		node = queue.popleft()
	#		if isinstance(node, Tree):
	#			queue.extendleft(reversed(node))
	#		else:
	#			theleaves.append(node)
	#	return theleaves

	def flatten(self):
		""" @return: a tree consisting of this tree's root connected directly
			to its leaves, omitting all intervening non-terminal nodes.
		@rtype: Tree """
		return Tree(self.label, self.leaves())

	def height(self):
		""" @return: The height of this tree.  The height of a tree
			containing no children is 1; the height of a tree
			containing only leaves is 2; and the height of any other
			tree is one plus the maximum of its children's
			heights.
		@rtype: int """
		max_child_height = 0
		for child in self:
			if isinstance(child, Tree):
				max_child_height = max(max_child_height, child.height())
			else:
				max_child_height = max(max_child_height, 1)
		return 1 + max_child_height

	def depth(self):
		""" The depth of a tree is its height - 1. """
		return self.height() - 1

	def treepositions(self, order='preorder'):
		""" @param order: One of: preorder, postorder, bothorder,
			leaves. """
		positions = []
		if order in ('preorder', 'bothorder'):
			positions.append( () )
		for i, child in enumerate(self):
			if isinstance(child, Tree):
				childpos = child.treepositions(order)
				positions.extend((i, ) + p for p in childpos)
			else:
				positions.append((i, ))
		if order in ('postorder', 'bothorder'):
			positions.append(())
		return positions

	def subtrees(self, condition=None):
		""" Generate all the subtrees of this tree, optionally restricted
		to trees matching the condition function.
		@type condition: function
		@param condition: the function to filter all local trees """
		if condition is None or condition(self):
			yield self
		for child in self:
			if isinstance(child, Tree):
				for subtree in child.subtrees(condition):
					yield subtree

	def pos(self):
		""" @return: a list of tuples containing leaves and pre-terminals
			(part-of-speech tags).
			The order reflects the order of the
			leaves in the tree's hierarchical structure.
		@rtype: list of tuples """
		pos = []
		for child in self:
			if isinstance(child, Tree):
				pos.extend(child.pos())
			else:
				pos.append((child, self.label))
		return pos

	def leaf_treeposition(self, index):
		""" @return: The tree position of the index-th leaf in this
			tree.  I.e., if tp=self.leaf_treeposition(i), then
			self[tp]==self.leaves()[i].

		@raise IndexError: If this tree contains fewer than index+1
			leaves, or if index<0. """
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
					stack.append( (value[i], treepos + (i, )) )

		raise IndexError('index must be less than or equal to len(self)')

	def treeposition_spanning_leaves(self, start, end):
		""" @return: The tree position of the lowest descendant of this
			tree that dominates self.leaves()[start:end].
		@raise ValueError: if end <= start """
		if end <= start:
			raise ValueError('end must be greater than start')
		# Find the tree positions of the start & end leaves, and
		# take the longest common subsequence.
		start_treepos = self.leaf_treeposition(start)
		end_treepos = self.leaf_treeposition(end - 1)
		# Find the first index where they mismatch:
		for i in range(len(start_treepos)):
			if i == len(end_treepos) or start_treepos[i] != end_treepos[i]:
				return start_treepos[:i]
		return start_treepos

	#////////////////////////////////////////////////////////////
	# Transforms
	#////////////////////////////////////////////////////////////

	def chomsky_normal_form(self, factor="right", horzmarkov=None,
			vertmarkov=0, childchar="|", parentchar="^"):
		""" This method can modify a tree in three ways:

		1. Convert a tree into its Chomsky Normal Form (CNF)
			equivalent -- Every subtree has either two non-terminals
			or one terminal as its children.  This process requires
			the creation of more"artificial" non-terminal nodes.
		2. Markov (vertical) smoothing of children in new artificial
			nodes
		3. Horizontal (parent) annotation of nodes

		@param factor: Right or left factoring method (default = "right")
		@type  factor: string = [left|right]
		@param horzmarkov: Markov order for sibling smoothing in
			artificial nodes (None (default) = include all siblings)
		@type  horzmarkov: int | None
		@param vertmarkov: Markov order for parent smoothing
			(0 (default) = no vertical annotation)
		@type  vertmarkov: int | None
		@param childchar: A string used in construction of the
			artificial nodes, separating the head of the
			original subtree from the child nodes that have yet to be
			expanded (default = "|")
		@type  childchar: string
		@param parentchar: A string used to separate the node
			representation from its vertical annotation
		@type  parentchar: string """
		from treetransforms import binarize
		binarize(self, factor, horzmarkov, vertmarkov + 1, childchar,
				parentchar)

	def un_chomsky_normal_form(self, expandunary=True, childchar="|",
			parentchar="^", unarychar="+"):
		""" This method modifies the tree in three ways:

		1. Transforms a tree in Chomsky Normal Form back to its
			original structure (branching greater than two)
		2. Removes any parent annotation (if it exists)
		3. (optional) expands unary subtrees (if previously
			collapsed with collapseunary(...) )

		@param expandunary: Flag to expand unary or not (default = True)
		@type  expandunary: boolean
		@param childchar: A string separating the head node from its
			children in an artificial node (default = "|")
		@type  childchar: string
		@param parentchar: A sting separating the node label from its
			parent annotation (default = "^")
		@type  parentchar: string
		@param unarychar: A string joining two non-terminals in a unary
			production (default = "+")
		@type  unarychar: string """
		from treetransforms import unbinarize
		unbinarize(self, expandunary, childchar, parentchar, unarychar)

	def collapse_unary(self, collapsepos=False, collapseroot=False,
			joinchar="+"):
		""" Collapse subtrees with a single child (ie. unary productions)
		into a new non-terminal (Tree node) joined by 'joinchar'.
		This is useful when working with algorithms that do not allow
		unary productions, and completely removing the unary productions
		would require loss of useful information.  The Tree is modified
		directly (since it is passed by reference) and no value is returned.

		@param collapsepos: 'False' (default) will not collapse the
			parent of leaf nodes (ie., Part-of-Speech tags) since they
			are always unary productions
		@type  collapsepos: boolean
		@param collapseroot: 'False' (default) will not modify the root
			production if it is unary.  For the Penn WSJ treebank
			corpus, this corresponds to the TOP -> productions.
		@type collapseroot: boolean
		@param joinchar: A string used to connect collapsed node values
			(default = "+")
		@type  joinchar: string """
		from treetransforms import collapse_unary
		collapse_unary(self, collapsepos, collapseroot, joinchar)

	#////////////////////////////////////////////////////////////
	# Convert, copy
	#////////////////////////////////////////////////////////////

	# [classmethod]
	def convert(cls, val):
		""" Convert a tree between different subtypes of Tree.  cls
		determines which class will be used to encode the new tree.

		@type val: Tree
		@param val: The tree that should be converted.
		@return: The new Tree. """
		if isinstance(val, Tree):
			children = [cls.convert(child) for child in val]
			return cls(val.label, children)
		else:
			return val
	convert = classmethod(convert)

	def copy(self, deep=False):
		""" Create a copy of this tree. """
		if not deep:
			return self.__class__(self.label, self)
		else:
			return self.__class__.convert(self)

	def _frozen_class(self):
		return ImmutableTree
	def freeze(self, leaf_freezer=None):
		frozen_class = self._frozen_class()
		if leaf_freezer is None:
			newcopy = frozen_class.convert(self)
		else:
			newcopy = self.copy(deep=True)
			for pos in newcopy.treepositions('leaves'):
				newcopy[pos] = leaf_freezer(newcopy[pos])
			newcopy = frozen_class.convert(newcopy)
		hash(newcopy) # Make sure the leaves are hashable.
		return newcopy

	#////////////////////////////////////////////////////////////
	# Parsing
	#////////////////////////////////////////////////////////////

	@classmethod
	def parse(cls, s, brackets='()', parse_label=None, parse_leaf=None,
			label_pattern=None, leaf_pattern=None,
			remove_empty_top_bracketing=False):
		""" Parse a bracketed tree string and return the resulting tree.
		Trees are represented as nested brackettings, such as::

			(S (NP (NNP John)) (VP (V runs)))

		@type s: str
		@param s: The string to parse

		@type brackets: length-2 str
		@param brackets: The bracket characters used to mark the
			beginning and end of trees and subtrees.

		@type parse_label: function
		@type parse_leaf: function
		@param parse_label, parse_leaf: If specified, these functions
			are applied to the substrings of s corresponding to
			labels and leaves (respectively) to obtain the values for
			those labels and leaves.  They should have the following
			signature:

				parse_label(str) -> value

			For example, these functions could be used to parse labels
			and leaves whose values should be some type other than
			string (such as FeatStruct <nltk.featstruct.FeatStruct>).
			Note that by default, label strings and leaf strings are
			delimited by whitespace and brackets; to override this
			default, use the label_pattern and leaf_pattern
			arguments.

		@type label_pattern: str
		@type leaf_pattern: str
		@param label_pattern, leaf_pattern: Regular expression patterns
			used to find label and leaf substrings in s.  By
			default, both label and leaf patterns are defined to match any
			sequence of non-whitespace non-bracket characters.

		@type remove_empty_top_bracketing: bool
		@param remove_empty_top_bracketing: If the resulting tree has
			an empty node label, and is length one, then return its
			single child instead.  This is useful for treebank trees,
			which sometimes contain an extra level of bracketing.

		@return: A tree corresponding to the string representation s.
			If this class method is called using a subclass of Tree,
			then it will return a tree of that type.
		@rtype: Tree """
		if not isinstance(brackets, basestring) or len(brackets) != 2:
			raise TypeError('brackets must be a length-2 string')
		if re.search('\s', brackets):
			raise TypeError('whitespace brackets not allowed')
		# Construct a regexp that will tokenize the string.
		open_b, close_b = brackets
		open_pattern, close_pattern = (re.escape(open_b), re.escape(close_b))
		if label_pattern is None:
			label_pattern = '[^\s%s%s]+' % (open_pattern, close_pattern)
		if leaf_pattern is None:
			leaf_pattern = '[^\s%s%s]+' % (open_pattern, close_pattern)
		token_re = re.compile('%s\s*(%s)?|%s|(%s)' % (
			open_pattern, label_pattern, close_pattern, leaf_pattern))
		# Walk through each token, updating a stack of trees.
		stack = [(None, [])] # list of (label, children) tuples
		for match in token_re.finditer(s):
			token = match.group()
			# Beginning of a tree/subtree
			if token[0] == open_b:
				if len(stack) == 1 and len(stack[0][1]) > 0:
					cls._parse_error(s, match, 'end-of-string')
				label = token[1:].lstrip()
				if parse_label is not None:
					label = parse_label(label)
				stack.append((label, []))
			# End of a tree/subtree
			elif token == close_b:
				if len(stack) == 1:
					if len(stack[0][1]) == 0:
						cls._parse_error(s, match, open_b)
					else:
						cls._parse_error(s, match, 'end-of-string')
				label, children = stack.pop()
				stack[-1][1].append(cls(label, children))
			# Leaf node
			else:
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
			assert stack[0][0] is None
			assert len(stack[0][1]) == 1
		tree = stack[0][1][0]

		# If the tree has an extra level with label='', then get rid of
		# it.  E.g.: "((S (NP ...) (VP ...)))"
		if remove_empty_top_bracketing and tree.label == '' and len(tree) == 1:
			tree = tree[0]
		# return the tree.
		return tree

	@classmethod
	def _parse_error(cls, s, match, expecting):
		""" Display a friendly error message when parsing a tree string fails.
		@param s: The string we're parsing.
		@param match: regexp match of the problem token.
		@param expecting: what we expected to see instead. """
		# Construct a basic error message
		if match == 'end-of-string':
			pos, token = len(s), 'end-of-string'
		else:
			pos, token = match.start(), match.group()
		msg = '%s.parse(): expected %r but got %r\n%sat index %d.' % (
			cls.__name__, expecting, token, ' ' * 12, pos)
		# Add a display showing the error token itsels:
		s = s.replace('\n', ' ').replace('\t', ' ')
		offset = pos
		if len(s) > pos + 10:
			s = s[:pos + 10] + '...'
		if pos > 10:
			s = '...' + s[pos - 10:]
			offset = 13
		msg += '\n%s"%s"\n%s^' % (' ' * 16, s, ' ' * (17 + offset))
		raise ValueError(msg)

	#////////////////////////////////////////////////////////////
	# String Representation
	#////////////////////////////////////////////////////////////

	def __repr__(self):
		childstr = ", ".join(repr(c) for c in self)
		return '%s(%r, [%s])' % (self.__class__.__name__, self.label, childstr)

	def __str__(self):
		return self._pprint_flat('', '()', False)

	def pprint(self, margin=70, indent=0, labelsep='', parens='()',
			quotes=False):
		""" @return: A pretty-printed string representation of this tree.
		@rtype: string
		@param margin: The right margin at which to do line-wrapping.
		@type margin: int
		@param indent: The indentation level at which printing
			begins.  This number is used to decide how far to indent
			subsequent lines.
		@type indent: int
		@param labelsep: A string that is used to separate the label
			from the children.  E.g., the value ':' gives
			trees like (S: (NP: I) (VP: (V: saw) (NP: it))). """

		# Try writing it on one line.
		s = self._pprint_flat(labelsep, parens, quotes)
		if len(s) + indent < margin:
			return s

		# If it doesn't fit on one line, then write it on multi-lines.
		if isinstance(self.label, basestring):
			s = '%s%s%s' % (parens[0], self.label, labelsep)
		else:
			s = '%s%r%s' % (parens[0], self.label, labelsep)
		for child in self:
			if isinstance(child, Tree):
				s += '\n' + ' ' * (indent + 2) + child.pprint(margin,
						indent + 2, labelsep, parens, quotes)
			elif isinstance(child, tuple):
				s += '\n' + ' ' * (indent + 2) + "/".join(child)
			elif isinstance(child, basestring) and not quotes:
				s += '\n' + ' ' * (indent + 2) +  '%s' % child
			else:
				s += '\n' + ' ' * (indent + 2) + '%r' % child
		return s + parens[1]

	def pprint_latex_qtree(self):
		r""" Returns a representation of the tree compatible with the
		LaTeX qtree package. This consists of the string \Tree
		followed by the parse tree represented in bracketed notation.

		For example, the following result was generated from a parse tree of
		the sentence The announcement astounded us::

		\Tree [.I'' [.N'' [.D The ] [.N' [.N announcement ] ] ]
			[.I' [.V'' [.V' [.V astounded ] [.N'' [.N' [.N us ] ] ] ] ] ] ]

		See http://www.ling.upenn.edu/advice/latex.html for the LaTeX
		style file for the qtree package.

		@return: A latex qtree representation of this tree.
		@rtype: string """
		return r'\Tree ' + self.pprint(indent=6, labelsep='',
				parens=('[.', ' ]'))

	def _pprint_flat(self, labelsep, parens, quotes):
		childstrs = []
		for child in self:
			if isinstance(child, Tree):
				childstrs.append(child._pprint_flat(labelsep, parens, quotes))
			elif isinstance(child, tuple):
				childstrs.append("/".join(child))
			elif isinstance(child, basestring) and not quotes:
				childstrs.append('%s' % child)
			else:
				childstrs.append('%r' % child)
		if isinstance(self.label, basestring):
			return '%s%s%s %s%s' % (parens[0], self.label, labelsep,
									" ".join(childstrs), parens[1])
		else:
			return '%s%r%s %s%s' % (parens[0], self.label, labelsep,
									" ".join(childstrs), parens[1])
	def draw(self):
		return DrawTree(self, self.leaves()).text()

class ImmutableTree(Tree):
	""" A tree which may not be modified.
	Has a hash() value. """
	def __init__(self, label_or_str, children=None):
		if children is None:
			return # see note in Tree.__init__()
		super(ImmutableTree, self).__init__(label_or_str, children)
		# Precompute our hash value.  This ensures that we're really
		# immutable.  It also means we only have to calculate it once.
		try:
			self._hash = hash( (self.label, tuple(self)) )
		except (TypeError, ValueError):
			raise ValueError("ImmutableTree's label and children "
					"must be immutable")
		else:
			self._leaves = Tree.leaves(self)
			self._subtrees = tuple(Tree.subtrees(self))
	def leaves(self):
		return self._leaves
	def subtrees(self, condition=None):
		if condition is None:
			return self._subtrees
		else:
			return filter(condition, self._subtrees)
	def __setitem__(self):
		raise ValueError('ImmutableTrees may not be modified')
	def __setslice__(self):
		raise ValueError('ImmutableTrees may not be modified')
	def __delitem__(self):
		raise ValueError('ImmutableTrees may not be modified')
	def __delslice__(self):
		raise ValueError('ImmutableTrees may not be modified')
	def __iadd__(self):
		raise ValueError('ImmutableTrees may not be modified')
	def __imul__(self):
		raise ValueError('ImmutableTrees may not be modified')
	def append(self, v):
		raise ValueError('ImmutableTrees may not be modified')
	def extend(self, v):
		raise ValueError('ImmutableTrees may not be modified')
	def pop(self, v=None):
		raise ValueError('ImmutableTrees may not be modified')
	def remove(self, v):
		raise ValueError('ImmutableTrees may not be modified')
	def reverse(self):
		raise ValueError('ImmutableTrees may not be modified')
	def sort(self):
		raise ValueError('ImmutableTrees may not be modified')
	def __hash__(self):
		return self._hash

	def _set_label(self, label):
		"""Set self._label.  This will only succeed the first time the
		label is set, which should occur in Tree.__init__()."""
		if hasattr(self, 'label'):
			raise ValueError('ImmutableTrees may not be modified')
		self._label = label
	def _get_label(self):
		return self._label
	label = property(_get_label, _set_label)

######################################################################
## Parented trees
######################################################################

class AbstractParentedTree(Tree):
	""" An abstract base class for Trees that automatically maintain
	pointers to their parents.  These parent pointers are updated
	whenever any change is made to a tree's structure.  Two subclasses
	are currently defined:

	- ParentedTree is used for tree structures where each subtree
		has at most one parent.  This class should be used in cases
		where there is no"sharing" of subtrees.

	- MultiParentedTree is used for tree structures where a
		subtree may have zero or more parents.  This class should be
		used in cases where subtrees may be shared.

	Subclassing
	===========
	The AbstractParentedTree class redefines all operations that
	modify a tree's structure to call two methods, which are used by
	subclasses to update parent information:

	- _setparent() is called whenever a new child is added.
	- _delparent() is called whenever a child is removed. """
	def __init__(self, label_or_str, children=None):
		if children is None:
			return # see note in Tree.__init__()
		super(AbstractParentedTree, self).__init__(label_or_str, children)
		# iterate over self, and *not* children, because children
		# might be an iterator.
		for i, child in enumerate(self):
			if isinstance(child, Tree):
				self._setparent(child, i, dry_run=True)
		for i, child in enumerate(self):
			if isinstance(child, Tree):
				self._setparent(child, i)

	#////////////////////////////////////////////////////////////
	# Parent management
	#////////////////////////////////////////////////////////////

	def _setparent(self, child, index, dry_run=False):
		""" Update child's parent pointer to point to self.  This
		method is only called if child's type is Tree; i.e., it
		is not called when adding a leaf to a tree.  This method is
		always called before the child is actually added to self's
		child list.

		@type child: Tree
		@type index: int
		@param index: The index of child in self.
		@raise TypeError: If child is a tree with an impropriate
			type.  Typically, if child is a tree, then its type needs
			to match self's type.  This prevents mixing of
			different tree types (single-parented, multi-parented, and
			non-parented).
		@param dry_run: If true, the don't actually set the child's
			parent pointer; just check for any error conditions, and
			raise an exception if one is found. """
		raise AssertionError('Abstract base class')

	def _delparent(self, child, index):
		""" Update child's parent pointer to not point to self.  This
		method is only called if child's type is Tree; i.e., it
		is not called when removing a leaf from a tree.  This method
		is always called before the child is actually removed from
		self's child list.

		@type child: Tree
		@type index: int
		@param index: The index of child in self. """
		raise AssertionError('Abstract base class')

	#////////////////////////////////////////////////////////////
	# Methods that add/remove children
	#////////////////////////////////////////////////////////////
	# Every method that adds or removes a child must make
	# appropriate calls to _setparent() and _delparent().

	def __delitem__(self, index):
		# del ptree[start:stop]
		if isinstance(index, slice):
			start, stop = slice_bounds(self, index)
			# Clear all the children pointers.
			for i in range(start, stop):
				if isinstance(self[i], Tree):
					self._delparent(self[i], i)
			# Delete the children from our child list.
			super(AbstractParentedTree, self).__delitem__(index)

		# del ptree[i]
		elif isinstance(index, int):
			if index < 0:
				index += len(self)
			if index < 0:
				raise IndexError('index out of range')
			# Clear the child's parent pointer.
			if isinstance(self[index], Tree):
				self._delparent(self[index], index)
			# Remove the child from our child list.
			super(AbstractParentedTree, self).__delitem__(index)

		# del ptree[()]
		elif len(index) == 0:
			raise IndexError('The tree position () may not be deleted.')

		# del ptree[(i, )]
		elif len(index) == 1:
			del self[index[0]]

		# del ptree[i1, i2, i3]
		else:
			del self[index[0]][index[1:]]

	def __setitem__(self, index, value):
		# ptree[start:stop] = value
		if isinstance(index, slice):
			start, stop = slice_bounds(self, index)
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
			# set the child pointers of the new children.  We do this
			# after clearing *all* child pointers, in case we're e.g.
			# reversing the elements in a tree.
			for i, child in enumerate(value):
				if isinstance(child, Tree):
					self._setparent(child, start + i)
			# finally, update the content of the child list itself.
			super(AbstractParentedTree, self).__setitem__(index, value)

		# ptree[i] = value
		elif isinstance(index, int):
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
			super(AbstractParentedTree, self).__setitem__(index, value)

		# ptree[()] = value
		elif len(index) == 0:
			raise IndexError('The tree position () may not be assigned to.')

		# ptree[(i, )] = value
		elif len(index) == 1:
			self[index[0]] = value

		# ptree[i1, i2, i3] = value
		else:
			self[index[0]][index[1:]] = value

	def append(self, child):
		if isinstance(child, Tree):
			self._setparent(child, len(self))
		super(AbstractParentedTree, self).append(child)

	def extend(self, children):
		for child in children:
			if isinstance(child, Tree):
				self._setparent(child, len(self))
			super(AbstractParentedTree, self).append(child)

	def insert(self, index, child):
		# Handle negative indexes.  Note that if index < -len(self),
		# we do *not* raise an IndexError, unlike __getitem__.  This
		# is done for consistency with list.__getitem__ and list.index.
		if index < 0:
			index += len(self)
		if index < 0:
			index = 0
		# Set the child's parent, and update our child list.
		if isinstance(child, Tree):
			self._setparent(child, index)
		super(AbstractParentedTree, self).insert(index, child)

	def pop(self, index=-1):
		if index < 0:
			index += len(self)
		if index < 0:
			raise IndexError('index out of range')
		if isinstance(self[index], Tree):
			self._delparent(self[index], index)
		return super(AbstractParentedTree, self).pop(index)

	# n.b.: like `list`, this is done by equality, not identity!
	# To remove a specific child, use del ptree[i].
	def remove(self, child):
		index = self.index(child)
		if isinstance(self[index], Tree):
			self._delparent(self[index], index)
		super(AbstractParentedTree, self).remove(child)

	# We need to implement __getslice__ and friends, even though
	# they're deprecated, because otherwise list.__getslice__ will get
	# called (since we're subclassing from list).  Just delegate to
	# __getitem__ etc., but use max(0, start) and max(0, stop) because
	# because negative indices are already handled *before*
	# __getslice__ is called; and we don't want to double-count them.
	if hasattr(list, '__getslice__'):
		def __getslice__(self, start, stop):
			return self.__getitem__(slice(max(0, start), max(0, stop)))
		def __delslice__(self, start, stop):
			return self.__delitem__(slice(max(0, start), max(0, stop)))
		def __setslice__(self, start, stop, value):
			return self.__setitem__(slice(max(0, start), max(0, stop)), value)

class ParentedTree(AbstractParentedTree):
	""" A Tree that automatically maintains parent pointers for
	single-parented trees.  The following read-only property values
	are automatically updated whenever the structure of a parented
	tree is modified: parent, parent_index, left_sibling,
	right_sibling, root, treeposition.

	Each ParentedTree may have at most one parent.  In
	particular, subtrees may not be shared.  Any attempt to reuse a
	single ParentedTree as a child of more than one parent (or
	as multiple children of the same parent) will cause a
	ValueError exception to be raised.

	ParentedTrees should never be used in the same tree as Trees
	or MultiParentedTrees.  Mixing tree implementations may result
	in incorrect parent pointers and in TypeError exceptions. """
	def __init__(self, label_or_str, children=None):
		if children is None:
			return # see note in Tree.__init__()

		self._parent = None
		"""The parent of this Tree, or None if it has no parent."""

		super(ParentedTree, self).__init__(label_or_str, children)

	def _frozen_class(self):
		return ImmutableParentedTree

	#/////////////////////////////////////////////////////////////////
	# Properties
	#/////////////////////////////////////////////////////////////////

	def _get_parent_index(self):
		if self._parent is None:
			return None
		for i, child in enumerate(self._parent):
			if child is self:
				return i
		assert False, 'expected to find self in self._parent!'

	def _get_left_sibling(self):
		parent_index = self._get_parent_index()
		if self._parent and parent_index > 0:
			return self._parent[parent_index - 1]
		return None # no left sibling

	def _get_right_sibling(self):
		parent_index = self._get_parent_index()
		if self._parent and parent_index < (len(self._parent) - 1):
			return self._parent[parent_index + 1]
		return None # no right sibling

	def _get_treeposition(self):
		if self._parent is None:
			return ()
		else:
			return (self._parent._get_treeposition() +
					(self._get_parent_index(), ))

	def _get_root(self):
		if self._parent is None:
			return self
		else:
			return self._parent._get_root()

	parent = property(lambda self: self._parent, doc="""
		The parent of this tree, or None if it has no parent.""")

	parent_index = property(_get_parent_index, doc="""
		The index of this tree in its parent.  I.e.,
		ptree.parent[ptree.parent_index] is ptree.  Note that
		ptree.parent_index is not necessarily equal to
		ptree.parent.index(ptree), since the index() method
		returns the first child that is _equal_ to its argument.""")

	left_sibling = property(_get_left_sibling, doc="""
		The left sibling of this tree, or None if it has none.""")

	right_sibling = property(_get_right_sibling, doc="""
		The right sibling of this tree, or None if it has none.""")

	root = property(_get_root, doc="""
		The root of this tree.  I.e., the unique ancestor of this tree
		whose parent is None.  If ptree.parent is None, then
		ptree is its own root.""")

	treeposition = property(_get_treeposition, doc="""
		The tree position of this tree, relative to the root of the
		tree.  I.e., ptree.root[ptree.treeposition] is ptree.""")
	treepos = treeposition # [xx] alias -- which name should we use?

	#/////////////////////////////////////////////////////////////////
	# Parent Management
	#/////////////////////////////////////////////////////////////////

	def _delparent(self, child, index):
		# Sanity checks
		assert isinstance(child, ParentedTree)
		assert self[index] is child
		assert child._parent is self

		# Delete child's parent pointer.
		child._parent = None

	def _setparent(self, child, index, dry_run=False):
		# If the child's type is incorrect, then complain.
		if not isinstance(child, ParentedTree):
			raise TypeError('Can not insert a non-ParentedTree '
							'into a ParentedTree')

		# If child already has a parent, then complain.
		if child._parent is not None:
			raise ValueError('Can not insert a subtree that already '
					'has a parent.')

		# Set child's parent pointer & index.
		if not dry_run:
			child._parent = self

class MultiParentedTree(AbstractParentedTree):
	""" A Tree that automatically maintains parent pointers for
	multi-parented trees.  The following read-only property values are
	automatically updated whenever the structure of a multi-parented
	tree is modified: parents, parent_indices, left_siblings,
	right_siblings, roots, treepositions.

	Each MultiParentedTree may have zero or more parents.  In
	particular, subtrees may be shared.  If a single
	MultiParentedTree is used as multiple children of the same
	parent, then that parent will appear multiple times in its
	parents property.

	MultiParentedTrees should never be used in the same tree as
	Trees or ParentedTrees.  Mixing tree implementations may
	result in incorrect parent pointers and in TypeError exceptions. """
	def __init__(self, label_or_str, children=None):
		if children is None:
			return # see note in Tree.__init__()

		self._parents = []
		"""A list of this tree's parents.  This list should not
			contain duplicates, even if a parent contains this tree
			multiple times."""

		super(MultiParentedTree, self).__init__(label_or_str, children)

	def _frozen_class(self):
		return ImmutableMultiParentedTree

	#/////////////////////////////////////////////////////////////////
	# Properties
	#/////////////////////////////////////////////////////////////////

	def _get_parent_indices(self):
		return [(parent, index)
				for parent in self._parents
				for index, child in enumerate(parent)
				if child is self]

	def _get_left_siblings(self):
		return [parent[index - 1]
				for (parent, index) in self._get_parent_indices()
				if index > 0]

	def _get_right_siblings(self):
		return [parent[index + 1]
				for (parent, index) in self._get_parent_indices()
				if index < (len(parent) - 1)]

	def _get_roots(self):
		return list(self._get_roots_helper({}).values())

	def _get_roots_helper(self, result):
		if self._parents:
			for parent in self._parents:
				parent._get_roots_helper(result)
		else:
			result[id(self)] = self
		return result

	parents = property(lambda self: list(self._parents), doc="""
		The set of parents of this tree.  If this tree has no parents,
		then parents is the empty set.  To check if a tree is used
		as multiple children of the same parent, use the
		parent_indices property.

		@type: list of MultiParentedTree""")

	left_siblings = property(_get_left_siblings, doc="""
		A list of all left siblings of this tree, in any of its parent
		trees.  A tree may be its own left sibling if it is used as
		multiple contiguous children of the same parent.  A tree may
		appear multiple times in this list if it is the left sibling
		of this tree with respect to multiple parents.

		@type: list of MultiParentedTree""")

	right_siblings = property(_get_right_siblings, doc="""
		A list of all right siblings of this tree, in any of its parent
		trees.  A tree may be its own right sibling if it is used as
		multiple contiguous children of the same parent.  A tree may
		appear multiple times in this list if it is the right sibling
		of this tree with respect to multiple parents.

		@type: list of MultiParentedTree""")

	roots = property(_get_roots, doc="""
		The set of all roots of this tree.  This set is formed by
		tracing all possible parent paths until trees with no parents
		are found.

		@type: list of MultiParentedTree""")

	def parent_indices(self, parent):
		"""
		Return a list of the indices where this tree occurs as a child
		of parent.  If this child does not occur as a child of
		parent, then the empty list is returned.  The following is
		always true::

		for parent_index in ptree.parent_indices(parent):
			parent[parent_index] is ptree
		"""
		if parent not in self._parents:
			return []
		else:
			return [index for (index, child) in enumerate(parent)
					if child is self]

	def treepositions(self, root):
		""" Return a list of all tree positions that can be used to reach
		this multi-parented tree starting from root.  I.e., the
		following is always true::

		for treepos in ptree.treepositions(root):
			root[treepos] is ptree """
		if self is root:
			return [()]
		else:
			return [treepos + (index, )
					for parent in self._parents
					for treepos in parent.treepositions(root)
					for (index, child) in enumerate(parent) if child is self]


	#/////////////////////////////////////////////////////////////////
	# Parent Management
	#/////////////////////////////////////////////////////////////////

	def _delparent(self, child, index):
		# Sanity checks
		assert isinstance(child, MultiParentedTree)
		assert self[index] is child
		assert len([p for p in child._parents if p is self]) == 1

		# If the only copy of child in self is at index, then delete
		# self from child's parent list.
		for i, c in enumerate(self):
			if c is child and i != index:
				break
		else:
			child._parents.remove(self)

	def _setparent(self, child, index, dry_run=False):
		# If the child's type is incorrect, then complain.
		if not isinstance(child, MultiParentedTree):
			raise TypeError('Can not insert a non-MultiParentedTree '
							'into a MultiParentedTree')

		# Add self as a parent pointer if it's not already listed.
		if not dry_run:
			for parent in child._parents:
				if parent is self:
					break
			else:
				child._parents.append(self)

class ImmutableParentedTree(ImmutableTree, ParentedTree):
	def __init__(self, label_or_str, children=None):
		if children is None:
			return # see note in Tree.__init__()
		super(ImmutableParentedTree, self).__init__(label_or_str, children)

class ImmutableMultiParentedTree(ImmutableTree, MultiParentedTree):
	def __init__(self, label_or_str, children=None):
		if children is None:
			return # see note in Tree.__init__()
		super(ImmutableMultiParentedTree, self).__init__(label_or_str, children)

## discontinuous trees ##
def eqtree(tree1, sent1, tree2, sent2):
	""" Test whether two discontinuous trees are equivalent;
	assumes canonicalized() ordering. """
	if tree1.label != tree2.label or len(tree1) != len(tree2):
		return False
	for a, b in zip(tree1, tree2):
		istree = isinstance(a, Tree)
		if istree != isinstance(b, Tree):
			return False
		elif istree:
			if not a.__eq__(b):
				return False
		else:
			return sent1[a] == sent2[b]
	return True

class DiscTree(ImmutableTree):
	""" Wrap an immutable tree with indices as leaves
	and a sentence. """
	def __init__(self, tree, sent):
		super(DiscTree, self).__init__(tree.label,
				tuple(DiscTree(a, sent) if isinstance(a, Tree) else a
				for a in tree))
		self.sent = sent
	def __eq__(self, other):
		return isinstance(other, Tree) and eqtree(self, self.sent,
				other, other.sent)
	def __hash__(self):
		return hash((self.label, ) + tuple(a.__hash__()
				if isinstance(a, Tree) else self.sent[a] for a in self))
	def __repr__(self):
		return "DisctTree(%r, %r)" % (
				super(DiscTree, self).__repr__(), self.sent)


######################################################################
## Discontinuous tree drawing
######################################################################

class DrawTree(object):
	""" Visualize a discontinuous tree in various formats. """
	def __init__(self, tree, sent):
		self.nodes, self.coords, self.edges = self.nodecoords(tree, sent)
	def __str__(self):
		return self.text()
	def __repr__(self):
		return "\n".join("%d: coord=%r, parent=%r, node=%s" % (
						n, self.coords[n], self.edges.get(n), self.nodes[n])
					for n in sorted(self.nodes))

	def nodecoords(self, tree, sent):
		""" Resolve placement of nodes for drawing discontinuous trees
		programmatically.
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
		Coordinates are (row, column); the origin (0, 0) is at the top left; the
		root node is on row 0. Coordinates do not consider the size of a node
		(which depends on font, &c), so the width of a column of the grid
		should be automatically determined by the element with the greatest
		width in that column. Alternatively, the integer coordinates could be
		converted to coordinates in which the distances between adjacent nodes
		are non-uniform.
		Produces tuple (nodes, coords, edges) where:
		- nodes[id] == Tree object for the node with this integer id
		- coords[id] == (n, m) coordinate where to draw node with id in the grid
		- edges[id] == parent id of node with this id (ordered dictionary)
		"""
		def findcell(rows, children):
			""" find vacant row, column index
			iterate over current rows for this level (try lowest first)
			and look for cell between first and last child of this node,
			add new row to level if no free row available. """
			l = [a for a in children[m]]
			center = min(l) + (max(l) - min(l)) // 2
			i = min(center + 1, max(l))
			j = center
			initi, initj = i, j

			for idx, row in enumerate(rows):
				i, j = initi, initj
				if all(a is None for a in
						row[min(children[m]):max(children[m])+1]):
					# find free column
					while zeroindex < j or i < scale * len(sent):
						if scale * len(sent) > i and rows[idx][i] is None:
							return idx, i
						elif zeroindex < j and rows[idx][j] is None:
							return idx, j
						i += 1
						j -= 1
			# end of for loop without finding candidate
			rows.append([None] * (len(sent) * scale))
			rowindex = len(rows) - 1
			return rowindex, initi

		assert all(isinstance(n, int) and 0 <= n < len(sent)
				for n in tree.leaves()), (tree.leaves(), sent)
		blocked = 1 # constant
		tree = tree.copy(True)
		for a in tree.subtrees(lambda n: n and isinstance(n[0], Tree)):
			a.sort(key=lambda n: min(n.leaves()))
		scale = 10
		crossed = set()
		zeroindex = min(tree.leaves())
		# internal nodes and lexical nodes (no frontiers)
		positions = tree.treepositions()
		#positions = [a for a in tree.treepositions()
		#		if (not isinstance(tree[a], int))] # or sent[tree[a]] != None]
		depth = max(map(len, positions)) + 1
		childcols = defaultdict(list)
		matrix = [[None] * (len(sent) * scale) for _ in range(2)]
		nodes = {}
		ids = {a: n for n, a in enumerate(positions)}
		levels = {n: [] for n in range(depth)}
		preterminals = []
		terminals = []
		for a in positions:
			node = tree[a]
			if isinstance(node, Tree):
				if node and not isinstance(node[0], Tree):
					preterminals.append(a)
				else:
					levels[len(a)].append(a)
			else:
				terminals.append(a)

		for n in levels:
			levels[n].sort(key=lambda n: (max(tree[n].leaves())
					- min(tree[n].leaves()), len(n)), reverse=True)
		preterminals.sort()
		terminals.sort()
		positions = set(positions)

		for m in terminals:
			i = int(tree[m]) * scale
			assert matrix[1][i] == None, (
					matrix[1][i], m, i)
			matrix[1][i] = ids[m]
			nodes[ids[m]] = sent[tree[m]]
			if nodes[ids[m]] is None:
				nodes[ids[m]] = "..."
				#matrix[0][i] = True
			positions.remove(m)
			childcols[m[:-1]].append(i)
		# preterminals directly above (if any)
		for m in preterminals:
			candidates = [a * scale for a in tree[m].leaves()]
			i = min(candidates) + (max(candidates) - min(candidates)) // 2
			assert matrix[0][i] in (None, True), (matrix[0][i], m, i)
			matrix[0][i] = ids[m]
			nodes[ids[m]] = tree[m]
			positions.remove(m)
			childcols[m[:-1]].append(i)

		# add other nodes centered on their children,
		# if the center is already taken, back off
		# to the left and right alternately, until an empty cell is found.
		for n in sorted(levels, reverse=True):
			nodesatdepth = levels[n]
			rows = [[None] * (len(sent) * scale)]
			for m in nodesatdepth: #[::-1]:
				if n < depth - 1 and childcols[m]:
					pivot = min(childcols[m])
					if (set(a[:-1] for row in rows[1:] for a in row[:pivot]
							if isinstance(a, tuple)) &
						(set(a[:-1] for row in rows[1:] for a in row[pivot:]
							if isinstance(a, tuple)))):
						crossed.add(m)

				rowindex, i = findcell(rows, childcols)
				positions.remove(m)

				# block positions where children of this node branch out
				for x in childcols[m]:
					assert rows[rowindex][x] is None, (rows[rowindex][x], m)
					rows[rowindex][x] = blocked
				# node itself
				assert rows[rowindex][i] in (None, blocked), (
						rows[rowindex][i], m)
				rows[rowindex][i] = ids[m]
				nodes[ids[m]] = tree[m]
				ids[m] = ids[m]
				# add column to the set of children for its parent
				if m != ():
					childcols[m[:-1]].append(i)
			matrix[:0] = rows

		assert len(positions) == 0

		# remove unused columns, right to left
		for m in range(scale * len(sent) - 1, -1, -1):
			if not any(isinstance(row[m], (Tree, int))
					for row in matrix):
				for row in matrix:
					del row[m]

		# remove unused rows
		matrix = [row for row in matrix if not all(a is None for a in row)]

		# collect coordinates of nodes
		coords = {}
		for n, _ in enumerate(matrix):
			for m, i in enumerate(matrix[n]):
				if isinstance(i, (Tree, int)):
					coords[i] = n, m

		#move crossed edges last
		positions = sorted([a for level in levels.values()
				for a in level] + preterminals,
				key=lambda a: a[:-1] in crossed)

		# collect edges from node to node
		edges = OrderedDict()
		for i in reversed(positions):
			for j, _ in enumerate(tree[i]):
				edges[ids[i + (j, )]] = ids[i]
		return nodes, coords, edges

	def svg(self, nodecolor='blue', leafcolor='red'):
		""" Return SVG representation of a discontinuous tree. """
		fontsize = 12
		hscale = 40
		vscale = 25
		hstart = vstart = 20
		width = max(col for _, col in self.coords.values())
		height = max(row for row, _ in self.coords.values())
		result = ['<svg version="1.1" xmlns="http://www.w3.org/2000/svg" '
				'width="%dem" height="%dem" viewBox="%d %d %d %d">' % (
						width * 3,
						height * 2.5,
						-hstart, -vstart,
						width * hscale + 3 * hstart,
						height * vscale + 3 * vstart)]

		children = defaultdict(set)
		for n in self.nodes:
			if n:
				children[self.edges[n]].add(n)

		# horizontal branches from nodes to children
		for node in self.nodes:
			if not children[node]:
				continue
			py, px = self.coords[node]
			px *= hscale
			py *= vscale
			px += hstart
			py += vstart + fontsize // 2
			childx = [self.coords[c][1] for c in children[node]]
			xmin = hstart + hscale * min(childx)
			xmax = hstart + hscale * max(childx)
			result.append(
				'\t<polyline style="stroke:black; stroke-width:1; fill:none;" '
					'points="%g,%g %g,%g" />' % (
					xmin, py, xmax, py))
			result.append(
				'\t<polyline style="stroke:black; stroke-width:1; fill:none;" '
					'points="%g,%g %g,%g" />' % (
					px, py, px, py - fontsize // 3))

		# vertical branches from children to parents
		for child, parent in self.edges.items():
			py, _ = self.coords[parent]
			py *= vscale
			py += vstart + fontsize // 2
			y, x = self.coords[child]
			x *= hscale
			y *= vscale
			x += hstart
			y += vstart - fontsize
			result.append(
				'\t<polyline style="stroke:white; stroke-width:10; fill:none;" '
				'points="%g,%g %g,%g" />' % (x, y, x, py + 5))
			result.append(
				'\t<polyline style="stroke:black; stroke-width:1; fill:none;" '
				'points="%g,%g %g,%g" />' % (x, y, x, py))

		# write nodes with coordinates
		for n, (row, column) in self.coords.items():
			node = self.nodes[n]
			x = column * hscale + hstart
			y = row * vscale + vstart
			result.append('\t<text style="text-anchor: middle; fill: %s; '
					'font-size: %dpx;" x="%g" y="%g">%s</text>' % (
					nodecolor if isinstance(node, Tree) else leafcolor,
					fontsize, x, y, node.label
					if isinstance(node, Tree) else node))

		result += ['</svg>']
		return "\n".join(result)

	def text(self, nodedist=1, unicodelines=False, html=False,
				nodecolor="blue", leafcolor="red"):
		""" Return ASCII art for a discontinuous tree.
		unicodelines: whether to use Unicode line drawing characters
			instead of plain (7-bit) ASCII.
		html: whether to wrap output in html code (default plain text).
		leafcolor, nodecolor: specify colors of leaves and phrasal nodes.
			Only applicable when html is True (use ANSI codes?). """

		if unicodelines:
			horzline = '\u2500'
			leftcorner = '\u250c'
			rightcorner = '\u2510'
			vertline = ' \u2502 '
			tee = horzline + '\u252C' + horzline
			bottom = horzline + '\u2534' + horzline
			cross = horzline + '\u253c' + horzline
		else:
			horzline = "_"
			leftcorner = rightcorner = " "
			vertline = " | "
			tee = 3 * horzline
			cross = bottom = "_|_"
		def crosscell(cur, x=vertline):
			""" Overwrite center of this cell with a vertical branch. """
			splitl = len(cur) - len(cur) // 2 - len(x) // 2 - 1
			lst = list(cur)
			lst[splitl:splitl+len(x)] = list(x)
			return ''.join(lst)

		result = []
		matrix = defaultdict(dict)
		maxnode = defaultdict(lambda: 3)
		maxcol = 0
		minchildcol = {}
		maxchildcol = {}
		childcols = defaultdict(set)
		for a in self.nodes:
			row, column = self.coords[a]
			matrix[row][column] = a
			maxcol = max(maxcol, column)
			maxnode[column] = max(maxnode[column], len(self.nodes[a].label
					if isinstance(self.nodes[a], Tree) else self.nodes[a]))
			if a not in self.edges:
				continue # e.g. root
			parent = self.edges[a]
			childcols[parent].add(column)
			minchildcol[parent] = min(minchildcol.get(parent, column), column)
			maxchildcol[parent] = max(maxchildcol.get(parent, column), column)
		for col in range(maxcol + 1):
			maxnode[col] += nodedist
		# bottom up level order traversal
		for row in sorted(matrix, reverse=True):
			noderow = ["".center(maxnode[col]) for col in range(maxcol + 1)]
			branchrow = ["".center(maxnode[col]) for col in range(maxcol + 1)]
			for col in matrix[row]:
				n = matrix[row][col]
				node = self.nodes[n]
				if isinstance(node, Tree):
					text = node.label
					n = matrix[row][col]
					#horizontal branch towards children for this node
					if n in minchildcol and minchildcol[n] < maxchildcol[n]:
						i, j = minchildcol[n], maxchildcol[n]
						l, m = (maxnode[i] + 1) // 2 - 1, maxnode[j] // 2
						branchrow[i] = ((' ' * l) + leftcorner).ljust(
								maxnode[i], horzline)
						branchrow[j] = (rightcorner + (' ' * m)).rjust(
								maxnode[j], horzline)
						for i in range(minchildcol[n] + 1, maxchildcol[n]):
							if i == col and i in childcols[n]:
								line = cross
							elif i == col:
								line = bottom
							elif i in childcols[n]:
								line = tee
							else:
								line = horzline
							branchrow[i] = line.center(maxnode[i], horzline)
					else: #if n and n in minchildcol:
						branchrow[col] = crosscell(branchrow[col])
				else:
					text = node
				text = text.center(maxnode[col])
				if html:
					if isinstance(node, Tree):
						text = "<font color=%s>%s</font>" % (nodecolor, text)
					else:
						text = "<font color=%s>%s</font>" % (leafcolor, text)
				noderow[col] = text
			#for each column, if there is a node below us which has a parent
			#above us, draw a vertical branch in that column.
			if row != max(matrix):
				for n, (childrow, col) in self.coords.items():
					if n and self.coords[self.edges[n]][0] < row < childrow:
						branchrow[col] = crosscell(branchrow[col])
						if col not in matrix[row]:
							noderow[col] = crosscell(noderow[col])
				noderow.append("\n")
				result.append("".join(noderow + branchrow))
			else:
				result.append("".join(noderow))
		return "\n".join(reversed(result)) + "\n"

	def tikzmatrix(self, nodecolor="blue", leafcolor="red"):
		""" Produce TiKZ code for use with LaTeX. PDF can be produced with
		pdflatex. Uses TiKZ matrices meaning that nodes are put into a fixed
		grid. Where the cells of each column all have the same width."""
		result = [r"""\begin{tikzpicture}[scale=1, minimum height=1.25em,
			text height=1.25ex, text depth=.25ex,
			inner sep=0mm, node distance=1mm]""",
		r"\footnotesize\sffamily",
		r"\matrix[row sep=0.5cm,column sep=0.1cm] {"]

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
					i = matrix[row][col]
					node = self.nodes[i]
					if isinstance(node, Tree):
						color = nodecolor
						label = latexlabel(node.label)
					else:
						color = leafcolor
						label = node
					line.append(r"\node [%s] (n%d) { %s };" % (
							color, i, label))
				line.append("&")
			# new row: skip last column char "&", add newline
			result.append(" ".join(line[:-1]) + r" \\")
		result += ["};"]

		shift = -0.5
		# write branches from node to node
		for child, parent in self.edges.items():
			result.append(
				"\draw [white, -, line width=6pt] (n%d)  +(0, %g) -| (n%d);"
				"	\draw (n%d) -- +(0, %g) -| (n%d);" % (
				parent, shift, child, parent, shift, child))

		result += [r"\end{tikzpicture}"]
		return "\n".join(result)

	def tikznode(self, nodecolor="blue", leafcolor="red"):
		""" Produce TiKZ code to draw a tree. Nodes are drawn with the
		\\node command so they can have arbitrary coordinates. """
		result = [r"""\begin{tikzpicture}[scale=0.75, minimum height=1.25em,
			text height=1.25ex, text depth=.25ex,
			inner sep=0mm, node distance=1mm]""",
		r"\footnotesize\sffamily",
		r"\path"]

		bottom = max(row for row, _ in self.coords.values())
		# write nodes with coordinates
		for n, (row, column) in self.coords.items():
			node = self.nodes[n]
			result.append("\t(%d, %d) node [%s] (n%d) {%s}"
					% (column, bottom - row,
					nodecolor if isinstance(node, Tree) else leafcolor,
					n, latexlabel(node.label)
					if isinstance(node, Tree) else node))
		result += [";"]

		shift = -0.5
		# write branches from node to node
		for child, parent in self.edges.items():
			result.append(
				"\draw [white, -, line width=6pt] (n%d)  +(0, %g) -| (n%d);"
				"	\draw (n%d) -- +(0, %g) -| (n%d);" % (
				parent, shift, child, parent, shift, child))

		result += [r"\end{tikzpicture}"]
		return "\n".join(result)

def latexlabel(label):
	""" quote/format label for latex """
	l = label.replace("$", r"\$").replace("[", "(").replace("_", "\_")
	# underscore => math mode
	if "|" in l:
		x, y = l.split("|", 1)
		y = y.replace("<", "").replace(">", "")
		if "^" in y:
			y, z = y.split("^")
			y = y[1:-1].replace("-", ",")
			l = "$ \\textsf{%s}_\\textsf{%s}^\\textsf{%s} $" % (x, y, z)
		else:
			l = "$ \\textsf{%s}_\\textsf{%s} $" % (x, y.replace("-",","))
	return l

######################################################################
## Utilitiy functions
######################################################################

def slice_bounds(sequence, slice_obj, allow_step=False):
	""" Given a slice, return the corresponding (start, stop) bounds, taking
	into account None indices and negative indices. The following guarantees
	are made for the returned start and stop values:

	- 0 <= start <= len(sequence)
	- 0 <= stop <= len(sequence)
	- start <= stop

	@raise ValueError: If slice_obj.step is not None.
	@param allow_step: If true, then the slice object may have a
		non-None step.  If it does, then return a tuple
		(start, stop, step).
	"""
	start, stop = (slice_obj.start, slice_obj.stop)

	# If allow_step is true, then include the step in our return
	# value tuple.
	if allow_step:
		if slice_obj.step is None:
			slice_obj.step = 1
		# Use a recursive call without allow_step to find the slice
		# bounds.  If step is negative, then the roles of start and
		# stop (in terms of default values, etc), are swapped.
		if slice_obj.step < 0:
			start, stop = slice_bounds(sequence, slice(stop, start))
		else:
			start, stop = slice_bounds(sequence, slice(start, stop))
		return start, stop, slice_obj.step

	# Otherwise, make sure that no non-default step value is used.
	elif slice_obj.step not in (None, 1):
		raise ValueError('slices with steps are not supported by %s' %
				sequence.__class__.__name__)

	# Supply default offsets.
	if start is None:
		start = 0
	if stop is None:
		stop = len(sequence)

	# Handle negative indices.
	if start < 0:
		start = max(0, len(sequence) + start)
	if stop < 0:
		stop = max(0, len(sequence) + stop)

	# Make sure stop doesn't go past the end of the list.  Note that
	# we avoid calculating len(sequence) if possible, because for lazy
	# sequences, calculating the length of a sequence can be expensive.
	if stop > 0:
		try:
			sequence[stop - 1]
		except IndexError:
			stop = len(sequence)

	# Make sure start isn't past stop.
	start = min(start, stop)

	# That's all folks!
	return start, stop

######################################################################
## Demonstration
######################################################################

def main():
	"""
	A demonstration showing how Trees and Trees can be
	used.  This demonstration creates a Tree, and loads a
	Tree from the treebank<nltk.corpus.treebank> corpus,
	and shows the results of calling several of their methods.
	"""

	# Demonstrate tree parsing.
	s = '(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN cookie))))'
	t = Tree(s)
	print("Convert bracketed string into tree:")
	print(t)
	print(t.__repr__())

	print("Display tree properties:")
	print(t.label)		# tree's constituent type
	print(t[0])			# tree's first child
	print(t[1])			# tree's second child
	print(t.height())
	print(t.leaves())
	print(t[1])
	print(t[1, 1])
	print(t[1, 1, 0])

	# Demonstrate tree modification.
	the_cat = t[0]
	the_cat.insert(1, Tree.parse('(JJ big)'))
	print("Tree modification:")
	print(t)
	t[1, 1, 1] = Tree.parse('(NN cake)')
	print(t)
	print()

	# Tree transforms
	print("Collapse unary:")
	t.collapse_unary()
	print(t)
	print("Chomsky normal form:")
	t.chomsky_normal_form()
	print(t)
	print()

	# Demonstrate parsing of treebank output format.
	t = Tree.parse(t.pprint())
	print("Convert tree to bracketed string and back again:")
	print(t)
	print()

	# Demonstrate LaTeX output
	print("LaTeX output:")
	print(t.pprint_latex_qtree())
	print()

	# Demonstrate tree nodes containing objects other than strings
	t.label = ('test', 3)
	print(t)

	trees = """(ROOT (S (ADV 0) (VVFIN 1) (NP (PDAT 2) (NN 3)) (PTKNEG 4) \
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
			(top (punct 10) (punct 11) (punct 16) (smain (np (det 0) (noun 1)) \
				(verb 2) (np (det 3) (noun 4)) (adv 5) (du (cp (comp 6) (ssub \
				(noun 7) (verb 8) (inf (verb 9)))) (du (smain (noun 12) \
				(verb 13) (adv 14) (part 15)) (noun 17)))) \
				(punct 18) (punct 19))
			(top (smain (noun 0) (verb 1) (inf (verb 8) (inf (verb 9) (inf \
				(adv 2) (pp (prep 3) (noun 4)) (pp (prep 5) (np (det 6) \
				(noun 7))) (verb 10))))) (punct 11))
			(top (smain (noun 0) (verb 1) (pp (prep 2) (np (det 3) (adj 4) \
				(noun 5) (rel (noun 6) (ssub (noun 7) (verb 10) (ppart (adj 8) \
				(part 9) (verb 11))))))) (punct 12))
			(top (smain (np (det 0) (noun 1)) (verb 2) (ap (adv 3) (num 4) \
				(cp (comp 5) (np (det 6) (adj 7) (noun 8) (rel (noun 9) (ssub \
				(noun 10) (verb 11) (pp (prep 12) (np (det 13) (adj 14) \
				(adj 15) (noun 16))))))))) (punct 17))
			(top (smain (np (det 0) (noun 1)) (verb 2) (adv 3) (pp (prep 4) \
				(np (det 5) (noun 6)) (part 7))) (punct 8))
			(top (punct 7) (conj (smain (noun 0) (verb 1) (np (det 2) \
				(noun 3)) (pp (prep 4) (np (det 5) (noun 6)))) (smain (verb 8) \
				(np (det 9) (num 10) (noun 11)) (part 12)) (vg 13) (smain \
				(verb 14) (noun 15) (pp (prep 16) (np (det 17) (noun 18) \
				(pp (prep 19) (np (det 20) (noun 21))))))) (punct 22))
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
			(top (smain (verb 2) (noun 3) (adv 4) (ppart (np (det 0) (noun 1)) \
				(verb 5))) (punct 6))
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
			(VP (VP 0 3) (NP (PRP 1) (NN 2)))"""
	sents = """Leider stehen diese Fragen nicht im Vordergrund der \
				augenblicklichen Diskussion .
			is Mary happy there
			das muss man jetzt machen
			Of ze had gewoon met haar vriendinnen rond kunnen slenteren in de \
				buurt van Trafalgar Square .
			Het had een prachtige dag kunnen zijn in Londen .
			Cathy zag hen wild zwaaien .
			Het was een spel geworden , zij en haar vriendinnen kozen iemand \
				uit en probeerden zijn of haar nationaliteit te raden .
			Elk jaar in het hoogseizoen trokken daar massa's toeristen voorbij \
				, hun fototoestel in de aanslag , pratend , gillend en lachend \
				in de vreemdste talen .
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
			Ze had met haar moeder kunnen gaan winkelen , zwemmen of terrassen .
			Dat werkwoord had ze zelf uitgevonden .
			De middagzon hing klein tussen de takken en de schaduwen van de \
				wolken drentelden over het gras .
			Zij zou mams rug ingewreven hebben en mam de hare .
			0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
			0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"""
	trees = [Tree.parse(a, parse_leaf=int) for a in trees.splitlines()]
	sents = [a.split() for a in sents.splitlines()]
	sents.extend([['Wake', None, 'up'],
		[None, 'your', 'friend', None]])
	#for tree, sent in zip(trees, sents):
	#	nodes, coords, edges = nodecoords(tree, sent)
	#	print("tikz node:\n", tikznode(nodes, coords, edges))
	#	print("tikz matrix:\n", tikzmatrix(nodes, coords, edges))
	#req = open("/tmp/t.html", "w")
	#req.write("""<!doctype html><html><body>""")
	for n, (tree, sent) in enumerate(zip(trees, sents)):
		dt = DrawTree(tree, sent)
		#req.write("<div>%s</div>\n\n" % dt.svg().encode('utf-8'))
		#open("/tmp/t%d.svg" % n, "w").writelines(dt.svg())
		print("\ntree, sent", tree,
				" ".join("..." if a is None else a for a in sent),
				#repr(dt),
				sep='\n')
		try:
			print(dt.text(unicodelines=True, html=False), sep='\n')
		except (UnicodeDecodeError, UnicodeEncodeError):
			print(dt.text(unicodelines=False, html=False), sep='\n')
	#req.write("</body></html>")
	#req.close()

if __name__ == '__main__':
	main()
