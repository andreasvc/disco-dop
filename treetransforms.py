r""" This file contains three main transformations:
 - A straightforward binarization: binarize(), based on NLTK code.
   Provides some additional Markovization options.
 - An optimal binarization for LCFRS: optimalbinarize()
   Cf. Gildea (2010): Optimal parsing strategies for linear
   context-free rewriting systems.
 - Converting discontinuous trees to continuous trees and back:
   splitdiscnodes(). Cf. Boyd (2007): Discontinuity revisited.

# Original notice:
# Natural Language Toolkit: Tree Transformations
#
# Copyright (C) 2005-2007 Oregon Graduate Institute
# Author: Nathan Bodenstab <bodenstab@cslu.ogi.edu>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT


A collection of methods for tree (grammar) transformations used
in parsing natural language.

Although many of these methods are technically grammar transformations
(ie. Chomsky Norm Form), when working with treebanks it is much more
natural to visualize these modifications in a tree structure.  Hence,
we will do all transformation directly to the tree itself.
Transforming the tree directly also allows us to do parent annotation.
A grammar can then be simply induced from the modified tree.

The following is a short tutorial on the available transformations.

 1. Chomsky Normal Form (binarization)

    It is well known that any grammar has a Chomsky Normal Form (CNF)
    equivalent grammar where CNF is defined by every production having
    either two non-terminals or one terminal on its right hand side.
    When we have hierarchically structured data (ie. a treebank), it is
    natural to view this in terms of productions where the root of every
    subtree is the head (left hand side) of the production and all of
    its children are the right hand side constituents.  In order to
    convert a tree into CNF, we simply need to ensure that every subtree
    has either two subtrees as children (binarization), or one leaf node
    (non-terminal).  In order to binarize a subtree with more than two
    children, we must introduce artificial nodes.

    There are two popular methods to convert a tree into CNF: left
    factoring and right factoring.  The following example demonstrates
    the difference between them.  Example::

      Original    Right-Factored       Left-Factored

          A              A                      A
        / | \          /   \                  /   \
       B  C  D   ==>  B    A|<C-D>   OR   A|<B-C>  D
                            /  \          /  \
                           C    D        B    C

 2. Parent Annotation

    In addition to binarizing the tree, there are two standard
    modifications to node labels we can do in the same traversal: parent
    annotation and Markov order-N smoothing (or sibling smoothing).

    The purpose of parent annotation is to refine the probabilities of
    productions by adding a small amount of context.  With this simple
    addition, a CYK (inside-outside, dynamic programming chart parse)
    can improve from 74% to 79% accuracy.  A natural generalization from
    parent annotation is to grandparent annotation and beyond.  The
    tradeoff becomes accuracy gain vs. computational complexity.  We
    must also keep in mind data sparsity issues.  Example:

     Original       Parent Annotation

          A                A^<?>
        / | \             /   \
       B  C  D   ==>  B^<A>    A|<C-D>^<?>     where ? is the
                                 /  \          parent of A
                             C^<A>   D^<A>


 3. Markov order-N smoothing

    Markov smoothing combats data sparsity issues as well as decreasing
    computational requirements by limiting the number of children
    included in artificial nodes.  In practice, most people use an order
    2 grammar.  Example::

      Original       No Smoothing       Markov order 1   Markov order 2   etc.

       __A__            A                      A                A
      / /|\ \         /   \                  /   \            /   \
     B C D E F  ==>  B    A|<C-D-E-F>  ==>  B   A|<C>  ==>   B  A|<C-D>
                            /   \               /   \            /   \
                           C    ...            C    ...         C    ...



    Annotation decisions can be thought about in the vertical direction
    (parent, grandparent, etc) and the horizontal direction (number of
    siblings to keep).  Parameters to the following functions specify
    these values.  For more information see:

    Dan Klein and Chris Manning (2003) "Accurate Unlexicalized
    Parsing", ACL-03.  http://www.aclweb.org/anthology/P03-1054

 4. Unary Collapsing

    Collapse unary productions (ie. subtrees with a single child) into a
    new non-terminal (Tree node).  This is useful when working with
    algorithms that do not allow unary productions, yet you do not wish
    to lose the parent information.  Example::

       A
       |
       B   ==>   A+B
      / \        / \
     C   D      C   D

"""
from __future__ import print_function
import re, sys
from itertools import count, repeat
from collections import defaultdict, Set, Iterable
if sys.version[0] >= '3':
	basestring = (str, bytes) # pylint: disable=W0622,C0103
from tree import Tree, ImmutableTree
from grammar import ranges
try:
	from bit import fanout as bitfanout
except ImportError:
	def bitfanout(arg):
		""" Slower version. """
		prev = arg
		result = 0
		while arg:
			arg &= arg - 1
			if ((prev - arg) << 1) & prev == 0:
				result += 1
			prev = arg
		return result

USAGE = """Treebank binarization and conversion
usage: %s [options] action input output
where input and output are treebanks, and action is one of:
    none
    binarize [-h x] [-v x] [--factor left|right]
    optimalbinarize [-h x] [-v x]
    unbinarize
    introducepreterminals
    splitdisc [--markorigin]
    mergedisc

options may consist of (* marks default option):
  --inputfmt|--outputfmt [*export|discbracket|bracket|alpino]
  --inputenc|--ounpuntenc [*UTF-8|ISO-8859-1|...]
  --slice n:m    select a range of sentences from input starting with n,
                 up to but not including m; as in Python, n or m can be left
                 out or negative, and the first index is 0.
  --factor [left|*right]
                 whether binarization factors to the left or right
  -h n           horizontal markovization. default: infinite
  -v n           vertical markovization. default: 1
  --headrules x  turn on head finding; affects binarization.
                 reads rules from file "x" (e.g., "negra.headrules").
  --markheads    mark heads with '^' in phrasal labels.
  --morphaspos   for input, use morphology tags instead of POS tags
  --functions [add|remove]
                 for input, add or remove function tags to phrasal labels
  --punct x      possible options:
                 remove: remove any punctuation.
                 move: re-attach punctuation to nearest constituent to minimize
                       discontinuity.
                 restore: attach punctuation under root node.
Note: some of these transformations are specific to discontinuous treebanks,
    specifically the Negra/Tiger treebanks. In the output only POS & phrasal
    labels are guaranteed to be retained.
    The formats 'conll' and 'mst' do an unlabeled dependency conversion and
    require all constituents to have a child with HD as one of its function
    tags, or the use of heuristic head rules. """ % sys.argv[0]


def binarize(tree, factor="right", horzmarkov=None, vertmarkov=1,
	childchar="|", parentchar="^", headmarked=None, tailmarker="",
	leftmostunary=False, rightmostunary=False, threshold=2,
	pospa=False, artpa=True, reverse=False, ids=None):
	""" Binarize an NLTK Tree object. Parameters:
	factor: "left" or "right". Determines whether binarization proceeds from
			left to right or vice versa.
	horzmarkov: amount of horizontal context in labels. Default is infinity,
			such that now new generalization are introduced by the
			binarization.
	vertmarkov: number of ancestors to include in labels.
			NB: 1 means only the direct parent, as in a normal tree.
	headmarked: when given a string, signifies that a node is the head node;
			the direction of binarization will be switched when it is
			encountered, to enable a head-outward binarization.
			NB: for discontinuous trees this is not necessary, as the order of
			children can be freely adjusted to achieve the same effect of a
			head-outward binarization.
	leftmostunary: see below
	rightmostunary: introduce a new unary production for the first/last element
			in the RHS. This enables the same generalizations for the
			first & last non-terminals as with other siblings.
	tailmarker: when given a non-empty string, add this to artificial nodes
			introducing the last symbol. This is useful if the last symbol is
			the head node, ensuring that it is not exchangeable with other
			non-terminals.
	reverse: reverse direction of the horizontal markovization; e.g.:
			(A (B ) (C ) (D )) becomes:
			left:  (A (A|<D> (A|<C-D> (A|<B-C> (B )) (C )) (D )))
			right: (A (A|<B> (B ) (A|<B-C> (C ) (A|<C-D> (D )))))
			in this way the markovization represents the history of the
			nonterminals that have *already* been parsed, instead of those
			still to come (assuming bottom-up parsing).
	pospa: whether to add parent annotation to POS nodes.
	artpa: whether to add parent annotation to the artificial nodes introduced
			by the binarization.
	ids: a function to provide artificial node labels, instead of combining
			labels of sibling nodes. Disables Markovization.
	threshold: constituents with more than this number of children are factored;
		i.e., for a value of 2, do a normal binarization; for a value of 1, also
		factor binary productions to include an artificial node, etc.

	>>> sent = "das muss man jetzt machen".split()
	>>> origtree = Tree("(S (VP (PDS 0) (ADV 3) (VVINF 4)) (PIS 2) (VMFIN 1))")
	>>> tree = Tree("(S (VP (PDS 0) (ADV 3) (VVINF 4)) (PIS 2) (VMFIN 1))")
	>>> tree = binarize(tree, horzmarkov=0, tailmarker='')
	>>> print(tree)
	(S (VP (PDS 0) (VP|<> (ADV 3) (VVINF 4))) (S|<> (PIS 2) (VMFIN 1)))
	>>> tree = unbinarize(tree); assert tree == origtree
	>>> print(tree)
	(S (VP (PDS 0) (ADV 3) (VVINF 4)) (PIS 2) (VMFIN 1))

	>>> tree = binarize(tree, horzmarkov=1, tailmarker='')
	>>> print(tree)
	(S (VP (PDS 0) (VP|<ADV> (ADV 3) (VVINF 4))) (S|<PIS> (PIS 2) (VMFIN 1)))

	>>> tree = unbinarize(tree); assert tree == origtree
	>>> tree = binarize(tree, horzmarkov=1, leftmostunary=False, \
		rightmostunary=True, tailmarker='')
	>>> print(tree)
	(S (VP (PDS 0) (VP|<ADV> (ADV 3) (VP|<VVINF> (VVINF 4)))) (S|<PIS> \
		(PIS 2) (S|<VMFIN> (VMFIN 1))))

	>>> tree = unbinarize(tree); assert tree == origtree
	>>> tree = binarize(tree, horzmarkov=1, leftmostunary=True, \
		rightmostunary=False, tailmarker='')
	>>> print(tree)
	(S (S|<VP> (VP (VP|<PDS> (PDS 0) (VP|<ADV> (ADV 3) (VVINF 4)))) (S|<PIS> \
		(PIS 2) (VMFIN 1))))

	>>> tree = unbinarize(tree); assert tree == origtree
	>>> tree = binarize(tree, horzmarkov=1, leftmostunary=False, \
		rightmostunary=False, tailmarker='')
	>>> print(tree)
	(S (VP (PDS 0) (VP|<ADV> (ADV 3) (VVINF 4))) (S|<PIS> (PIS 2) (VMFIN 1)))

	>>> tree = unbinarize(tree); assert tree == origtree
	>>> tree = binarize(tree, horzmarkov=2, tailmarker='', reverse=True)
	>>> print(tree)
	(S (VP (PDS 0) (VP|<PDS-ADV> (ADV 3) (VVINF 4))) (S|<VP-PIS> (PIS 2) \
		(VMFIN 1)))

	>>> tree = unbinarize(tree); assert tree == origtree
	>>> tree = binarize(tree, factor="left", horzmarkov=2, tailmarker='')
	>>> print(tree)
	(S (S|<PIS-VMFIN> (VP (VP|<ADV-VVINF> (PDS 0) (ADV 3)) (VVINF 4)) (PIS 2)) \
		(VMFIN 1))

	>>> tree = unbinarize(tree); assert tree == origtree
	>>> tree = binarize(tree, horzmarkov=1, vertmarkov=2, leftmostunary=True, \
		rightmostunary=False, tailmarker='')
	>>> print(tree)
	(S (S|<VP> (VP^<S> (VP|<PDS>^<S> (PDS 0) (VP|<ADV>^<S> (ADV 3) \
		(VVINF 4)))) (S|<PIS> (PIS 2) (VMFIN 1))))

	>>> tree = unbinarize(tree); assert tree == origtree
	>>> tree = binarize(tree, horzmarkov=1, vertmarkov=2, leftmostunary=True, \
		rightmostunary=False, tailmarker='', pospa=True)
	>>> print(tree)
	(S (S|<VP> (VP^<S> (VP|<PDS>^<S> (PDS^<VP> 0) (VP|<ADV>^<S> (ADV^<VP> 3) \
			(VVINF^<VP> 4)))) (S|<PIS> (PIS^<S> 2) (VMFIN^<S> 1))))

	>>> tree = unbinarize(tree); assert tree == origtree
	>>> tree = binarize(tree, horzmarkov=1, vertmarkov=3, leftmostunary=True, \
		rightmostunary=False, tailmarker='', pospa=True)
	>>> print(tree)
	(S (S|<VP> (VP^<S> (VP|<PDS>^<S> (PDS^<VP-S> 0) (VP|<ADV>^<S> \
		(ADV^<VP-S> 3) (VVINF^<VP-S> 4)))) (S|<PIS> (PIS^<S> 2) (VMFIN^<S> 1))))

	>>> tree = Tree("(S (NN 2) (VP (PDS 0) (ADV 3) (VAINF 4)) (VMFIN 1))")
	>>> tree = binarize(tree, horzmarkov=2, tailmarker='', reverse=True)
	>>> print(tree)
	(S (NN 2) (S|<NN-VP> (VP (PDS 0) (VP|<PDS-ADV> (ADV 3) (VAINF 4))) \
		(VMFIN 1)))

	>>> tree = Tree("(S (A 0) (B 1) (C 2) (D 3) (E 4) (F 5))")
	>>> tree = binarize(tree, tailmarker='', reverse=False)
	>>> print(tree)
	(S (A 0) (S|<B-C-D-E-F> (B 1) (S|<C-D-E-F> (C 2) (S|<D-E-F> (D 3) \
		(S|<E-F> (E 4) (F 5))))))

	"""
	# assume all subtrees have homogeneous children
	# assume all terminals have no siblings

	# A semi-hack to have elegant looking code below.  As a result, any subtree
	# with a branching factor greater than 999 will be incorrectly truncated.
	if horzmarkov == None:
		horzmarkov = 999

	# Traverse tree depth-first keeping a list of ancestor nodes to the root.
	# I chose not to use the tree.treepositions() method since it requires
	# two traversals of the tree (one to get the positions, one to iterate
	# over them) and node access time is proportional to the height of the node.
	# This method is 7x faster which helps when binarizing 40,000 sentences.
	assert factor in ("left", "right")
	treeclass = tree.__class__
	leftmostunary = 1 if leftmostunary else 0
	agenda = [(tree, [tree.label])]
	while agenda:
		node, parent = agenda.pop()
		if not isinstance(node, Tree):
			continue
		# parent annotation
		parentstring = ""
		originallabel = node.label if vertmarkov else ""
		if vertmarkov > 1 and node is not tree and (
				pospa or isinstance(node[0], Tree)):
			parentstring = "%s<%s>" % (parentchar, "-".join(parent))
			node.label += parentstring
			parent = [originallabel] + parent[:vertmarkov - 2]
			if not artpa:
				parentstring = ""

		# add children to the agenda before we mess with them
		agenda.extend((child, parent) for child in node)

		# binary form factorization
		if len(node) <= threshold:
			continue
		elif 1 <= len(node) <= 2:
			if not isinstance(node[0], Tree):
				continue
			# insert an initial artificial nonterminal
			if ids is None:
				siblings = "-".join(child.label for child in node[:horzmarkov])
			else:
				siblings = str(next(ids))
			newnode = treeclass("%s%s<%s>%s" % (originallabel, childchar,
					siblings, parentstring), node)
			node[:] = [newnode]
		else:
			if isinstance(node[0], Tree):
				childlabels = [child.label for child in node]
			else:
				childlabels = []
			childnodes = list(node)
			numchildren = len(childnodes)
			headidx = 0

			# insert an initial artificial nonterminal
			if factor == "right":
				start = 0
				end = min(1, horzmarkov) if reverse else horzmarkov
			else: # factor == "left"
				start = ((numchildren - min(1, horzmarkov))
						if reverse else horzmarkov)
				end = numchildren
			if ids is None:
				siblings = "-".join(childlabels[start:end])
			else:
				siblings = str(next(ids))
			newnode = treeclass("%s%s<%s>%s" % (originallabel, childchar,
											siblings, parentstring), [])
			node[:] = []
			if leftmostunary:
				node.append(newnode)
				node = newnode
			curnode = node

			for i in range(1, numchildren - (not rightmostunary)):
				marktail = tailmarker if i + 1 == numchildren else ''
				newnode = treeclass('', [])
				if factor == "right":
					if reverse:
						start = max(i - horzmarkov + 1, 0)
						end = i + 1
					else:
						start = i
						end = i + horzmarkov
					curnode[:] = [childnodes.pop(0), newnode]
				else: # factor == "left":
					start = headidx + numchildren - i - 1
					end = start + horzmarkov
					curnode[:] = [newnode, childnodes.pop()]
				# switch direction upon encountering the head
				if headmarked and headmarked in childlabels[i]:
					headidx = i
					factor = "right" if factor == "left" else "left"
					start = headidx + numchildren - i - 1
					end = start + horzmarkov
				if ids is None:
					siblings = "-".join(childlabels[start:end])
				else:
					siblings = str(next(ids))
				newnode.label = "%s%s<%s>%s%s" % (originallabel, childchar,
						siblings, parentstring, marktail)
				curnode = newnode
			assert len(childnodes) == 1 + (not rightmostunary)
			curnode.extend(childnodes)
	return tree

def unbinarize(tree, expandunary=True, childchar="|", parentchar="^",
		unarychar="+"):
	""" Restore a binarized tree to the original n-ary tree.
	Modifies tree in-place.
	NB: a malformed node such as (X|<Y> ) which is not supposed to be empty
	will be silently discarded. """
	# increase robustness
	childchar += "<"
	parentchar += "<"
	treeclass =  tree.__class__
	# Traverse the tree-depth first keeping a pointer to the parent for
	# modification purposes.
	agenda = [(tree, [])]
	while agenda:
		node, parent = agenda.pop()
		if isinstance(node, Tree):
			# if the node contains the 'childchar' character it means that it
			# is an artificial node and can be removed, although we still
			# need to move its children to its parent
			childindex = node.label.find(childchar)
			if childindex != -1:
				# go by identity instead of equality
				n = [id(a) for a in parent].index(id(node))
				parent[n:n + 1] = node
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
				# non-binarized constituent, so move on to next parent
				parent = node

			for child in node:
				agenda.append((child, parent))
	return tree

def collapse_unary(tree, collapsepos=False, collapseroot=False, joinchar="+"):
	"""
	Collapse subtrees with a single child (ie. unary productions)
	into a new non-terminal (Tree node) joined by 'joinchar'.
	This is useful when working with algorithms that do not allow
	unary productions, and completely removing the unary productions
	would require loss of useful information.  The Tree is modified
	directly (since it is passed by reference) and no value is returned.

	@param tree: The Tree to be collapsed
	@type  tree: C{Tree}
	@param collapsepos: 'False' (default) will not collapse the parent of leaf
						nodes (i.e., Part-of-Speech tags) since they are always
						unary productions
	@type  collapsepos: C{boolean}
	@param collapseroot: 'False' (default) will not modify the root production
						if it is unary.  For the Penn WSJ treebank corpus, this
						corresponds to the TOP -> productions.
	@type collapseroot: C{boolean}
	@param joinchar: A string used to connect collapsed node values (default: "+")
	@type  joinchar: C{string}
	"""
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
				for child in node:
					agenda.append(child)
	return tree

def introducepreterminals(tree, ids=None):
	""" Introduce preterminals with artificial POS-tags where needed
	(i.e., for every terminal with siblings.)

	>>> tree = Tree("(S (X a b (CD c d) e))")
	>>> tree = introducepreterminals(tree)
	>>> print(tree)
	(S (X (X/a a) (X/b b) (CD (CD/c c) (CD/d d)) (X/e e)))

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
				node[n] = treeclass("%s/%s" % (
					node.label if ids is None else next(ids), child), [child])
	return tree

def getbits(bitset):
	""" Iterate over the indices of set bits in a bitset. """
	n = 0
	while bitset:
		if bitset & 1:
			yield n
		elif not bitset:
			break
		bitset >>= 1
		n += 1

def slowfanout(tree):
	""" Get the fan-out of a constituent. Slow because call to leaves() is
	recursive and not cached. """
	if isinstance(tree, Tree):
		return len(list(ranges(sorted(tree.leaves()))))
	else:
		return 1

def fastfanout(tree):
	""" Get the fan-out of a constituent. Requires the presence of a bitset
	attribute. """
	return bitfanout(tree.bitset) if isinstance(tree, Tree) else 1

#can only be used with ImmutableTrees because of memoization.
#fanout = memoize(slowfanout)
fanout = fastfanout

def complexity(tree):
	""" The degree of the time complextiy of parsing with this rule.
	Cf. Gildea (2011). """
	return fanout(tree) + sum(map(fanout, tree))

def complexityfanout(tree):
	""" Combination of complexity and fan-out, where the latter is used
	to break ties in the former. """
	return (fanout(tree) + sum(map(fanout, tree)),
			fanout(tree))

def fanoutcomplexity(tree):
	""" Combination of fan-out and complexity, where the latter is used
	to break ties in the former. """
	return (fanout(tree),
			fanout(tree) + sum(map(fanout, tree)))

def addbitsets(tree):
	""" Turn tree into an ImmutableTree and add a bitset attribute to each
	constituent to avoid slow calls to leaves(). """
	if isinstance(tree, basestring):
		result = ImmutableTree.parse(tree, parse_leaf=int)
	elif isinstance(tree, ImmutableTree):
		result = tree
	elif isinstance(tree, Tree):
		result = tree.freeze()
	else:
		raise ValueError("expected string or tree object")
	for a in result.subtrees():
		a.bitset = sum(1 << n for n in a.leaves())
	return result

def defaultleftbin(node, sep="|", h=999, markfanout=False, ids=None,
		threshold=2):
	""" Binarize one constituent with a left factored binarization.
	Children remain unmodified. Bottom-up version. Nodes must contain
	bitsets (use addbitsets()).
	By default construct artificial labels using labels of child nodes.
	When iterator ids is specified, use identifiers from that instead. """
	if len(node) <= threshold:
		return node
	elif 1 <= len(node) <= 2:
		if ids is None:
			childlabels = [child.label for child in node]
			newlabel = "%s%s<%s>" % (node.label, sep, "-".join(childlabels[:h]))
		else:
			newlabel = "%s%s<%d>" % (node.label, sep, next(ids))
		result = ImmutableTree(node.label, [ImmutableTree(newlabel, node)])
		result.bitset = node.bitset
	else:
		childlabels = [child.label for child in node]
		prev = node[0]
		for i in range(1, len(node) - 1):
			newbitset = prev.bitset | node[i].bitset
			if ids is None:
				newlabel = "%s%s<%s>" % (node.label, sep,
						"-".join(childlabels[max(0, i-h+1):i+1]))
			else:
				newlabel = "%s%s<%d>" % (node.label, sep, next(ids))
			if markfanout:
				nodefanout = bitfanout(newbitset)
				if nodefanout > 1:
					newlabel += "_" + str(nodefanout)
			prev = ImmutableTree(newlabel, [prev, node[i]])
			prev.bitset = newbitset
		result = ImmutableTree(node.label, [prev, node[-1]])
		result.bitset = prev.bitset | node[-1].bitset
	return result

def defaultrightbin(node, sep="|", h=999, markfanout=False, ids=None,
		threshold=2):
	""" Binarize one constituent with a right factored binarization.
	Children remain unmodified. Bottom-up version. Nodes must contain
	bitsets (use addbitsets()).
	By default construct artificial labels using labels of child nodes.
	When iterator ids is specified, use identifiers from that instead. """
	if len(node) <= threshold:
		return node
	elif 1 <= len(node) <= 2:
		if ids is None:
			childlabels = [child.label for child in node]
			newlabel = "%s%s<%s>" % (node.label, sep, "-".join(childlabels[:h]))
		else:
			newlabel = "%s%s<%d>" % (node.label, sep, next(ids))
		result = ImmutableTree(node.label, [ImmutableTree(newlabel, node)])
		result.bitset = node.bitset
	else:
		childlabels = [child.label for child in node]
		prev = node[-1]
		for i in range(len(node) - 2, 0, -1):
			newbitset = node[i].bitset | prev.bitset
			if ids is None:
				newlabel = "%s%s<%s>" % (node.label, sep,
						"-".join(childlabels[i:i+h]))
			else:
				newlabel = "%s%s<%d>" % (node.label, sep, next(ids))
			if markfanout:
				nodefanout = bitfanout(newbitset)
				if nodefanout > 1:
					newlabel += "_" + str(nodefanout)
			prev = ImmutableTree(newlabel, [node[i], prev])
			prev.bitset = newbitset
		result = ImmutableTree(node.label, [node[0], prev])
		result.bitset = node[0].bitset | prev.bitset
	return result

def minimalbinarization(tree, score, sep="|", head=None, parentstr="", h=999):
	""" Implementation of Gildea (2010): Optimal parsing strategies for
	linear context-free rewriting systems.  Expects an immutable tree where
	the terminals are integers corresponding to indices, with a special
	bitset attribute to avoid having to call leaves() repeatedly.
	The bitset attribute can be added with addbitsets()

	- tree is the tree for which the optimal binarization of its top
      production will be searched.
	- score is a function from binarized trees to some value, where lower is
      better (the value can be numeric or anything else which supports
      comparisons)
	- head is an optional index of the head node, specifying it enables
      head-driven binarization (which constrains the possible binarizations)

	>>> tree = "(X (A 0) (B 1) (C 2) (D 3) (E 4))"
	>>> tree1=addbitsets(tree)
	>>> tree2=Tree.parse(tree, parse_leaf=int)
	>>> tree2.chomsky_normal_form()
	>>> minimalbinarization(tree1, complexityfanout, head=2) == tree2
	True
	>>> tree = "(X (A 0) (B 3) (C 5) (D 7) (E 8))"
	>>> print(minimalbinarization(addbitsets(tree), complexityfanout, head=2))
	(X (A 0) (X|<B-E-D-C> (B 3) (X|<E-D-C> (E 8) (X|<D-C> (D 7) (C 5)))))
	>>> tree = "(X (A 0) (B 3) (C 5) (D 7) (E 8))"
	>>> print(minimalbinarization(addbitsets(tree), complexityfanout, head=2, h=1))
	(X (A 0) (X|<B> (B 3) (X|<E> (E 8) (X|<D> (D 7) (C 5)))))
	>>> tree = "(A (B1 (t 6) (t 13)) (B2 (t 3) (t 7) (t 10))  (B3 (t 1) \
		(t 9) (t 11) (t 14) (t 16)) (B4 (t 0) (t 5) (t 8)))"
	>>> a = minimalbinarization(addbitsets(tree), complexityfanout)
	>>> b = minimalbinarization(addbitsets(tree), fanoutcomplexity)
	>>> print(max(map(complexityfanout, a.subtrees())))
	(14, 6)
	>>> print(max(map(complexityfanout, b.subtrees())))
	(15, 5)
	"""
	def newproduction(a, b):
		""" return a new `production' (here a tree) combining a and b """
		if head is not None:
			siblings = (nonterms[a] | nonterms[b])[:h]
		else:
			siblings = getbits(nonterms[a] | nonterms[b])
		newlabel = "%s%s<%s>%s" % (tree.label, sep,
				"-".join(labels[x] for x in siblings), parentstr)
		new = ImmutableTree(newlabel, [a, b])
		new.bitset = a.bitset | b.bitset
		return new
	if len(tree) <= 2:
		return tree
	#don't bother with optimality if this particular node is not discontinuous
	#do default right factored binarization instead
	elif fanout(tree) == 1 and all(fanout(a) == 1 for a in tree):
		return defaultrightbin(tree, sep, h)
	from agenda import Agenda
	labels = [a.label for a in tree]
	#the four main datastructures:
	#the agenda is a priority queue of partial binarizations to explore
	#the first complete binarization that is dequeued is the optimal one
	agenda = Agenda()
	#the working set contains all the optimal partial binarizations
	#keys are binarizations, values are their scores
	workingset = {}
	#for each of the optimal partial binarizations, this dictionary has
	#a bitset that describes which non-terminals from the input it covers
	nonterms = {}
	# reverse lookup table for nonterms (from bitsets to binarizations)
	revnonterms = {}
	#the goal is a bitset that covers all non-terminals of the input
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
			#if new or better:
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
	raise ValueError

def optimalbinarize(tree, sep="|", headdriven=False, h=None, v=1):
	""" Recursively binarize a tree optimizing for complexity.
	v=0 is not implemented. """
	if headdriven:
		return recbinarizetreehd(addbitsets(tree), sep, h, v, ())
	else:
		assert h is None, "Horizontal Markovization requires headdriven=True"
	tree = Tree.convert(tree)
	for a in list(tree.subtrees(lambda x: len(x) > 1))[::-1]:
		a.sort(key=lambda x: x.leaves())
	return recbinarizetree(addbitsets(tree), sep, v, ())

def recbinarizetree(tree, sep, v, ancestors):
	""" postorder / bottom-up binarization """
	if not isinstance(tree, Tree):
		return tree
	parentstr = "^<%s>" % ("-".join(ancestors[:v-1])) if v > 1 else ""
	newtree = ImmutableTree(tree.label + parentstr,
		[recbinarizetree(t, sep, v, (tree.label,) + ancestors) for t in tree])
	newtree.bitset = tree.bitset
	return minimalbinarization(newtree, complexityfanout, sep,
			parentstr=parentstr)

def recbinarizetreehd(tree, sep, h, v, ancestors):
	""" postorder / bottom-up binarization """
	if not isinstance(tree, Tree):
		return tree
	parentstr = "^<%s>" % ("-".join(ancestors[:v-1])) if v > 1 else ""
	newtree = ImmutableTree(tree.label + parentstr,
		[recbinarizetreehd(t, sep, h, v, (tree.label,) + ancestors)
														for t in tree])
	newtree.bitset = tree.bitset
	return minimalbinarization(newtree, complexityfanout, sep,
			head=len(tree) - 1, parentstr=parentstr, h=h)

def disc(node):
	""" This function evaluates whether a particular node is locally
	discontinuous.  The root node will, by definition, be continuous.
	Nodes can be continuous even if some of their children are discontinuous.
	"""
	if not isinstance(node, Tree):
		return False
	return len(list(ranges(sorted(node.leaves())))) > 1

def addfanoutmarkers(tree):
	""" Modifies tree so that the label of each node with a fanout > 1 contains
	a marker "_n" indicating its fanout. """
	for st in tree.subtrees():
		leaves = set(st.leaves())
		thisfanout = len([a for a in sorted(leaves) if a - 1 not in leaves])
		if thisfanout > 1 and not st.label.endswith("_%d" % thisfanout):
			st.label += "_%d" % thisfanout
	return tree

def removefanoutmarkers(tree):
	""" Remove fanout marks. """
	for a in tree.subtrees(lambda x: "_" in x.label):
		a.label = a.label.rsplit("_", 1)[0]
	return tree

def postorder(tree, f=None):
	""" Do a postorder traversal of tree; similar to Tree.subtrees(),
	but Tree.subtrees() does a preorder traversal. """
	for child in tree:
		if isinstance(child, Tree):
			for a in postorder(child):
				if not f or f(a):
					yield a
	if not f or f(tree):
		yield tree


def canonicalize(tree):
	""" canonical linear precedence (of first component of each node) order """
	for a in postorder(tree, lambda n: isinstance(n[0], Tree)):
		a.sort(key=lambda n: n.leaves())
	return tree

def canonicalized(tree):
	""" canonical linear precedence (of first component of each node) order.
	returns a new tree. """
	if not isinstance(tree, Tree):
		return tree
	children = list(map(canonicalized, tree))
	if len(children) > 1:
		children.sort(key=lambda n: n.leaves())
	return Tree(tree.label, children)

def contsets(nodes):
	""" partition children into continuous subsets

	>>> tree = Tree.parse( \
		"(VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) (VVPP 5))", parse_leaf=int)
	>>> print(list(contsets(tree)))
	[[Tree('PP', [Tree('APPR', [0]), Tree('ART', [1]), Tree('NN', [2])])],
	[Tree('CARD', [4]), Tree('VVPP', [5])]]
	"""
	rng = -1
	subset = []
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

def splitdiscnodes(tree, markorigin=False):
	""" Boyd (2007): Discontinuity revisited.

	>>> tree = Tree.parse("(S (VP (VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) \
		(VVPP 5)) (VAINF 6)) (VMFIN 3))", parse_leaf=int)
	>>> print(splitdiscnodes(tree.copy(True)))
	(S (VP* (VP* (PP (APPR 0) (ART 1) (NN 2)))) (VMFIN 3) (VP* (VP* (CARD 4) \
		(VVPP 5)) (VAINF 6)))
	>>> print(splitdiscnodes(tree, markorigin=True))
	(S (VP*0 (VP*0 (PP (APPR 0) (ART 1) (NN 2)))) (VMFIN 3) (VP*1 (VP*1 \
		(CARD 4) (VVPP 5)) (VAINF 6)))
	"""
	treeclass = tree.__class__
	for node in postorder(tree):
		nodes = list(node)
		node[:] = []
		for child in nodes:
			if disc(child):
				childnodes = list(child)
				child[:] = []
				node.extend(treeclass((("%s*%d" % (child.label, n))
					if markorigin else '%s*' % child.label), childsubset)
					for n, childsubset in enumerate(contsets(childnodes)))
			else:
				node.append(child)
	return canonicalize(tree)

splitlabel = re.compile(r"(.*)\*([0-9]*)$")
def mergediscnodes(tree):
	""" Reverse transformation of splitdiscnodes()

	>>> tree = Tree.parse("(S (VP (VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) \
		(VVPP 5)) (VAINF 6)) (VMFIN 3))", parse_leaf=int)
	>>> print(mergediscnodes(splitdiscnodes(tree)))
	(S (VP (VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) (VVPP 5)) (VAINF 6)) \
		(VMFIN 3))
	>>> print(mergediscnodes(splitdiscnodes(tree, markorigin=True)))
	(S (VP (VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) (VVPP 5)) (VAINF 6)) \
	(VMFIN 3))
	>>> tree = Tree.parse("(S (X (A 0) (A 2)) (X (A 1) (A 3)))", parse_leaf=int)
	>>> print(mergediscnodes(splitdiscnodes(tree, markorigin=True)))
	(S (X (A 0) (A 2)) (X (A 1) (A 3)))
	>>> tree = Tree.parse("(S (X (A 0) (A 2)) (X (A 1) (A 3)))", parse_leaf=int)
	>>> print(splitdiscnodes(tree, markorigin=True))
	(S (X*0 (A 0)) (X*0 (A 1)) (X*1 (A 2)) (X*1 (A 3)))
	>>> tree = Tree.parse("(S (X (A 0) (A 2)) (X (A 1) (A 3)))", parse_leaf=int)
	>>> print(mergediscnodes(splitdiscnodes(tree)))
	(S (X (A 0) (A 1) (A 2) (A 3)))
	"""
	treeclass = tree.__class__
	for node in tree.subtrees():
		merge = defaultdict(list)
		nodes = list(node)
		node[:] = []
		for child in nodes:
			if not isinstance(child, Tree):
				node.append(child)
				continue
			match = splitlabel.search(child.label)
			if not match:
				node.append(child)
				continue
			label = match.group(1)
			grandchildren = list(child)
			child[:] = []
			if not merge[child.label]:
				merge[child.label].append(treeclass(label, []))
				node.append(merge[child.label][0])
			merge[child.label][0].extend(grandchildren)
			if match.group(2):
				nextlabel = "%s*%d" % (
						match.group(1), int(match.group(2)) + 1)
				merge[nextlabel].append(merge[child.label].pop(0))
	return tree

class OrderedSet(Set):
	""" A frozen, ordered set which maintains a regular list/tuple and set.
		The set is indexable. Equality is defined _without_ regard for order.
	"""
	def __init__(self, iterable=None):
		if iterable:
			self.seq = tuple(iterable)
			self.theset = frozenset(self.seq)
		else:
			self.seq = ()
			self.theset = frozenset()
	def __hash__(self):
		return hash(self.theset)
	def __contains__(self, value):
		return value in self.theset
	def __len__(self):
		return len(self.theset)
	def __iter__(self):
		return iter(self.seq)
	def __getitem__(self, n):
		return self.seq[n]
	def __reversed__(self):
		return reversed(self.seq)
	def __repr__(self):
		if not self.seq:
			return '%s()' % self.__class__.__name__
		return '%s(%r)' % (self.__class__.__name__, self.seq)
	def __eq__(self, other):
		#if isinstance(other, (OrderedSet, Sequence)):
		#	return len(self) == len(other) and list(self) == list(other)
		# equality is defined _without_ regard for order
		return self.theset == set(other)
	def __and__(self, other):
		""" maintain the order of the left operand. """
		if not isinstance(other, Iterable):
			return NotImplemented
		return self._from_iterable(value for value in self if value in other)


#################################################################
# Demonstration & tests
#################################################################

def demo():
	"""
	A demonstration showing how each tree transform can be used.
	"""

	# original tree from WSJ bracketed text
	sentence = """(TOP (S
	(S
      (VP
        (VBN Turned)
        (ADVP (RB loose))
        (PP
          (IN in)
          (NP
			(NP (NNP Shane) (NNP Longman) (POS 's))
			(NN trading)
			(NN room)))))
	(, ,)
	(NP (DT the) (NN yuppie) (NNS dealers))
	(VP (AUX do) (NP (NP (RB little)) (ADJP (RB right))))
	(. .)))"""
	tree = Tree(sentence)
	print("original", tree)

	# collapse subtrees with only one child
	collapsedtree = tree.copy(True)
	#collapse_unary(collapsedtree)

	# convert the tree to CNF
	cnftree = collapsedtree.copy(True)
	lcnftree = collapsedtree.copy(True)
	binarize(cnftree, factor="right", horzmarkov=2)
	binarize(lcnftree, factor="left", horzmarkov=2)

	# convert the tree to CNF with parent annotation
	# (one level) and horizontal smoothing of order two
	parenttree = collapsedtree.copy(True)
	binarize(parenttree, horzmarkov=2, vertmarkov=1)

	# convert the tree back to its original form
	original = cnftree.copy(True)
	original2 = lcnftree.copy(True)
	unbinarize(original)
	unbinarize(original2)

	print("binarized", cnftree)
	print("Sentences the same? ", tree == original, tree == original2)
	assert tree == original and tree == original2

def testminbin():
	""" Verify that all optimal parsing complexities are lower than or equal
	to the complexities of right-to-left binarizations. """
	from treebank import NegraCorpusReader
	import time
	#corpus = NegraCorpusReader("../rparse", "negraproc.export",
	#corpus = NegraCorpusReader("..", "negra-corpus.export",
	#	encoding="iso-8859-1", punct="move", headrules="negra.headrules",
	#	headfinal=True, headreverse=False)
	corpus = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1",
			punct="move", headrules=None, headfinal=True, headreverse=False)
	total = violations = violationshd = 0
	for n, tree, sent in zip(count(), list(
			corpus.parsed_sents().values())[:-2000], corpus.sents().values()):
		#if len(tree.leaves()) <= 25: continue
		begin = time.clock()
		t = addbitsets(tree)
		if all(fanout(x) == 1 for x in t.subtrees()):
			continue
		print(n, tree, '\n', " ".join(sent))
		total += 1
		optbin = optimalbinarize(tree.copy(True), headdriven=False, h=None, v=1)
		print(time.clock() - begin, "s\n")
		continue

		normbin = Tree.convert(tree)
		# undo head-ordering to get a normal right-to-left binarization
		for a in list(normbin.subtrees(lambda x: len(x) > 1))[::-1]:
			a.sort(key=lambda x: x.leaves())
		normbin.chomsky_normal_form()
		normbin = addbitsets(normbin)
		if (max(map(complexityfanout, optbin.subtrees()))
				> max(map(complexityfanout, normbin.subtrees()))):
			print("non-hd")
			print(tree)
			print(max(map(complexityfanout, optbin.subtrees())), optbin)
			print(max(map(complexityfanout, normbin.subtrees())), normbin)
			print('\n')
			violations += 1
			assert False

		optbin = optimalbinarize(tree.copy(True), headdriven=True, h=1, v=1)
		normbin = Tree.convert(tree)
		normbin.chomsky_normal_form(horzmarkov=1)
		#binarize(normbin, horzmarkov=1)
		normbin = addbitsets(normbin)
		if (max(map(complexityfanout, optbin.subtrees()))
				> max(map(complexityfanout, normbin.subtrees()))):
			print("hd")
			print(tree)
			print(max(map(complexityfanout, optbin.subtrees())), optbin)
			print(max(map(complexityfanout, normbin.subtrees())), normbin)
			print('\n')
			violationshd += 1
	print("violations: %d / %d" % (violations, total))
	print("violationshd: %d / %d" % (violationshd, total))
	assert violations == violationshd == 0

def testsplit():
	"""" Verify that splitting and merging discontinuties gives the
	same trees for a treebank"""
	from treebank import NegraCorpusReader
	correct = wrong = 0
	#n = NegraCorpusReader("../rparse", "tigerproc.export")
	n = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1")
	for tree in n.parsed_sents().values():
		if mergediscnodes(splitdiscnodes(tree)) == tree:
			correct += 1
		else:
			wrong += 1
	total = len(n.sents())
	print("correct", correct, "=", 100 * correct / total, "%")
	print("wrong", wrong, "=", 100 * wrong / total, "%")

def test():
	""" Run all examples. """
	demo()
	testminbin()
	testsplit()

def main():
	""" Command line interface for applying tree(bank) transforms. """
	import io
	from getopt import gnu_getopt, GetoptError
	from treebank import getreader, writetree, readheadrules
	flags = ('markorigin', 'markheads', 'morphaspos')
	options = ('factor=', 'headrules=', 'markorigin=', 'inputfmt=',
			'outputfmt=', 'inputenc=', 'outputenc=', 'slice=', 'punct=',
			'functions=')
	try:
		opts, args = gnu_getopt(sys.argv[1:], "h:v:", flags + options)
		action, infilename, outfilename = args
	except (GetoptError, ValueError) as err:
		print("error: %r\n%s" % (err, USAGE))
		sys.exit(2)
	opts = dict(opts)

	# read input
	Reader = getreader(opts.get('--inputfmt', 'export'))
	corpus = Reader(".", infilename,
			encoding=opts.get('--inputenc', 'utf-8'),
			headrules=opts.get("--headrules"),
			markheads='--markheads' in opts,
			morphaspos='--morphaspos' in opts,
			functiontags={'add': True, 'remove': False,
					None: None}[opts.get('--functions')],
			punct=opts.get("--punct"))
	start, end = opts.get('--slice', ':').split(':')
	start = int(start) if start else None
	end = int(end) if end else None
	trees = list(corpus.parsed_sents().values())[start:end]
	sents = list(corpus.sents().values())[start:end]
	keys = list(corpus.sents().keys())[start:end]
	print("read %d trees from %s" % (len(trees), infilename))

	# apply transformation
	actions = ("binarize unbinarize optimalbinarize introducepreterminals "
			"splitdisc mergedisc none").split()
	assert action in actions, ("unrecognized action: %r\n"
			"available actions: %r" % (action, actions))
	if action == "binarize":
		factor = opts.get('--factor', 'right')
		h = int(opts['-h']) if 'h' in opts else None
		v = int(opts.get('-v', 1))
		for a in trees:
			binarize(a, factor, h, v)
	elif action == "unbinarize":
		for a in trees:
			unbinarize(a, factor, h, v)
	elif action == "optimalbinarize":
		sep = "|"
		headdriven = "--headrules" in opts
		h = int(opts['-h']) if 'h' in opts else None
		v = int(opts.get('-v', 1))
		trees = [optimalbinarize(a, sep, headdriven, h, v) for a in trees]
	elif action == "introducepreterminals":
		for a in trees:
			introducepreterminals(a)
	elif action == "splitdisc":
		for a in trees:
			splitdiscnodes(a, '--markorigin' in opts)
	elif action == "mergedisc":
		for a in trees:
			mergediscnodes(a)
	if action != "none":
		print("transformed %d trees with action %r" % (len(trees), action))

	# write output
	if opts.get('--outputfmt') in ('mst', 'conll'):
		assert opts.get("--headrules"), (
				"need head rules for dependency conversion")
		headrules = readheadrules(opts.get("--headrules"))
	else:
		headrules = None

	print("going to write %d trees to %s" % (len(trees), outfilename))
	encoding = opts.get('outputenc', 'utf-8')
	with io.open(outfilename, "w", encoding=encoding) as outfile:
		if action == 'none' and (None == opts.get("--headrules")
				== opts.get('--markheads') == opts.get("--punct")
				and opts.get('--inputfmt') == opts.get('--outputfmt')):
			# copy treebank verbatim. useful when only taking a slice
			# or converting encoding.
			outfile.writelines(block for block in corpus.blocks().values())
		else:
			outfile.writelines(writetree(*x) for x in zip(
						trees, sents, keys,
						repeat(opts.get('--outputfmt', 'export')),
						repeat(headrules)))
	print("done")

__all__ = ["binarize", "unbinarize", "collapse_unary", "introducepreterminals",
	"splitdiscnodes", "mergediscnodes", "optimalbinarize", "defaultrightbin",
	"addfanoutmarkers", "removefanoutmarkers"]

if __name__ == '__main__':
	main()
