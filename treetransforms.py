"""This file contains three main transformations:
 - A straightforward binarization: binarize(), based on NLTK code.
   Modified to introduce a new unary production for the first/last
   element in the RHS.
   It is possible to change the markovization direction:
   (A (B ) (C ) (D )) becomes:
     left:  (A (A|<D> (A|<C-D> (A|<B-C> (B )) (C )) (D )))
     right: (A (A|<B> (B ) (A|<B-C> (C ) (A|<C-D> (D )))))
   in this way the markovization represents the history of the nonterminals
   that have *already* been parsed, instead of those still to come
   (assuming bottom-up parsing).
 - An optimal binarization for LCFRS: optimalbinarize()
   Cf. Gildea (2010): Optimal parsing strategies for linear
   context-free rewriting systems.
 - Converting discontinuous trees to continuous trees and back:
   splitdiscnodes(). Cf. Boyd (2007): Discontinuity revisited.
"""

# Original notice:
# Natural Language Toolkit: Tree Transformations
#
# Copyright (C) 2005-2007 Oregon Graduate Institute
# Author: Nathan Bodenstab <bodenstab@cslu.ogi.edu>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT


"""
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

    Markov smoothing combats data sparcity issues as well as decreasing
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
from itertools import count
from nltk import Tree, ImmutableTree
from grammar import ranges, canonicalize
from containers import OrderedSet
from collections import defaultdict
import re

def binarize(tree, factor="right", horzMarkov=None, vertMarkov=0,
	childChar="|", parentChar="^", headMarked=None,
	rightMostUnary=True, leftMostUnary=True,
	tailMarker="$", reverse=True):
	"""
	>>> sent = "das muss man jetzt machen".split()
	>>> tree = Tree("(S (VP (PDS 0) (ADV 3) (VVINF 4)) (PIS 2) (VMFIN 1))")
	>>> tree = binarize(tree, horzMarkov=0, tailMarker='')
	>>> print tree.pprint(margin=999)
	(S (S|<> (VP (VP|<> (PDS 0) (VP|<> (ADV 3) (VP|<> (VVINF 4))))) (S|<> (PIS 2) (S|<> (VMFIN 1)))))
	>>> tree = unbinarize(tree)
	>>> print tree
	(S (VP (PDS 0) (ADV 3) (VVINF 4)) (PIS 2) (VMFIN 1))

	>>> tree = binarize(tree, horzMarkov=1, tailMarker='')
	>>> print tree.pprint(margin=999)
	(S (S|<VP> (VP (VP|<PDS> (PDS 0) (VP|<ADV> (ADV 3) (VP|<VVINF> (VVINF 4))))) (S|<PIS> (PIS 2) (S|<VMFIN> (VMFIN 1)))))

	>>> tree = unbinarize(tree)
	>>> tree = binarize(tree, horzMarkov=1, leftMostUnary=False, rightMostUnary=True, tailMarker='')
	>>> print tree.pprint(margin=999)
	(S (VP (PDS 0) (VP|<ADV> (ADV 3) (VP|<VVINF> (VVINF 4)))) (S|<PIS> (PIS 2) (S|<VMFIN> (VMFIN 1))))

	>>> tree = unbinarize(tree)
	>>> tree = binarize(tree, horzMarkov=1, leftMostUnary=True, rightMostUnary=False, tailMarker='')
	>>> print tree.pprint(margin=999)
	(S (S|<VP> (VP (VP|<PDS> (PDS 0) (VP|<ADV> (ADV 3) (VVINF 4)))) (S|<PIS> (PIS 2) (VMFIN 1))))

	>>> tree = unbinarize(tree)
	>>> tree = binarize(tree, horzMarkov=1, leftMostUnary=False, rightMostUnary=False, tailMarker='')
	>>> print tree.pprint(margin=999)
	(S (VP (PDS 0) (VP|<ADV> (ADV 3) (VVINF 4))) (S|<PIS> (PIS 2) (VMFIN 1)))

	>>> tree = unbinarize(tree)
	>>> tree = binarize(tree, horzMarkov=2, tailMarker='')
	>>> print tree.pprint(margin=999)
	(S (S|<VP> (VP (VP|<PDS> (PDS 0) (VP|<PDS-ADV> (ADV 3) (VP|<ADV-VVINF> (VVINF 4))))) (S|<VP-PIS> (PIS 2) (S|<PIS-VMFIN> (VMFIN 1)))))

	>>> tree = unbinarize(tree)
	>>> tree = binarize(tree, factor="left", horzMarkov=2, tailMarker='')
	>>> print tree.pprint(margin=999)
	(S (S|<VMFIN> (S|<PIS-VMFIN> (S|<VP-PIS> (VP (VP|<VVINF> (VP|<ADV-VVINF> (VP|<PDS-ADV> (PDS 0)) (ADV 3)) (VVINF 4)))) (PIS 2)) (VMFIN 1)))

	>>> tree = Tree("(S (NN 2) (VP (PDS 0) (ADV 3) (VAINF 4)) (VMFIN 1))")
	>>> tree = binarize(tree, horzMarkov=2, tailMarker='')
	>>> print tree.pprint(margin=999)
	(S (S|<NN> (NN 2) (S|<NN-VP> (VP (VP|<PDS> (PDS 0) (VP|<PDS-ADV> (ADV 3) (VP|<ADV-VAINF> (VAINF 4))))) (S|<VP-VMFIN> (VMFIN 1)))))

	>>> tree = Tree("(S (A 0) (B 1) (C 2) (D 3) (E 4) (F 5))")
	>>> tree = binarize(tree, tailMarker='', reverse=False)
	>>> print tree.pprint(margin=999)
	(S (S|<A-B-C-D-E-F> (A 0) (S|<B-C-D-E-F> (B 1) (S|<C-D-E-F> (C 2) (S|<D-E-F> (D 3) (S|<E-F> (E 4) (S|<F> (F 5))))))))

	"""
	# assume all subtrees have homogeneous children
	# assume all terminals have no siblings

	# A semi-hack to have elegant looking code below.  As a result,
	# any subtree with a branching factor greater than 999 will be incorrectly truncated.
	if horzMarkov == None: horzMarkov = 999

	# Traverse the tree depth-first keeping a list of ancestor nodes to the root.
	# I chose not to use the tree.treepositions() method since it requires
	# two traversals of the tree (one to get the positions, one to iterate
	# over them) and node access time is proportional to the height of the node.
	# This method is 7x faster which helps when parsing 40,000 sentences.  
	leftMostUnary = 1 if leftMostUnary else 0
	nodeList = [(tree, [tree.node])]
	while nodeList != []:
		node, parent = nodeList.pop()
		if isinstance(node, Tree):

			# parent annotation
			parentString = ""
			originalNode = node.node
			if vertMarkov != 0 and node != tree and isinstance(node[0],Tree):
				parentString = "%s<%s>" % (parentChar, "-".join(parent))
				#node.node += parentString
				parent = [originalNode] + parent[:vertMarkov - 1]

			# add children to the agenda before we mess with them
			for child in node:
				nodeList.append((child, parent))

			# binary form factorization
			if len(node) > 2:
				childNodes = [child.node for child in node]
				nodeCopy = node.copy()
				numChildren = len(nodeCopy)
				headidx = 0

				# insert an initial artificial nonterminal
				if factor == "right":
					start = 0
					end = min(1, horzMarkov) if reverse else horzMarkov
				else: # factor == "left"
					start = (numChildren - min(1, horzMarkov)) if reverse else horzMarkov
					end = numChildren
				siblings = "-".join(childNodes[start:end])
				newNode = Tree("%s%s<%s>%s" % (originalNode, childChar,
												siblings, parentString), [])
				node[:] = []
				if leftMostUnary:
					node.append(newNode)
					node = newNode
				curNode = node

				for i in range(1, numChildren):
					marktail = tailMarker if i + 1 == numChildren else ''
					newNode = Tree('', [])
					if factor == "right":
						if reverse:
							start = max(i - horzMarkov + 1, 0)
							end = i + 1
						else:
							start = i
							end = i + horzMarkov
						curNode[:] = [nodeCopy.pop(0), newNode]
					else: # factor == "left":
						start = headidx + numChildren - i - 1
						end = start + horzMarkov
						curNode[:] = [newNode, nodeCopy.pop()]
					# switch direction upon encountering the head
					if headMarked and headMarked in childNodes[i]:
						headidx = i
						factor = "right" if factor == "left" else "left"
						start = headidx + numChildren - i - 1
						end = start + horzMarkov
					siblings = "-".join(childNodes[start:end])
					newNode.node = "%s%s%s<%s>%s" % (originalNode, childChar,
											marktail, siblings, parentString)
					curNode = newNode
				assert len(nodeCopy) == 1
				if rightMostUnary:
					curNode.append(nodeCopy.pop())
				else:
					curNode.node = nodeCopy[0].node
					curNode[:] = nodeCopy.pop()
	return tree

def unbinarize(tree, expandUnary=True, childChar="|", parentChar="^",
	unaryChar="+"):
	# Traverse the tree-depth first keeping a pointer to the parent for
	# modification purposes.
	agenda = [(tree, [])]
	while agenda:
		node, parent = agenda.pop()
		if isinstance(node, Tree):
			# if the node contains the 'childChar' character it means that it
			# is an artificial node and can be removed, although we still
			# need to move its children to its parent
			childIndex = node.node.find(childChar)
			if childIndex != -1:
				nodeIndex = parent.index(node)
				# replace node with children of node
				parent[nodeIndex:nodeIndex+1] = node

			else:
				parentIndex = node.node.find(parentChar)
				if parentIndex != -1:
					# strip the node name of the parent annotation
					node.node = node.node[:parentIndex]

				# expand collapsed unary productions
				if expandUnary:
					unaryIndex = node.node.find(unaryChar)
					if unaryIndex != -1:
						newNode = Tree(node.node[unaryIndex + 1:], node[:])
						node.node = node.node[:unaryIndex]
						node[0:] = [newNode]

				# non-binarized constituent, so move on to next parent
				parent = node

			for child in node:
				agenda.append((child, parent))
	return tree

def collapse_unary(tree, collapsePOS = False, collapseRoot = False, joinChar = "+"):
	"""
	Collapse subtrees with a single child (ie. unary productions)
	into a new non-terminal (Tree node) joined by 'joinChar'.
	This is useful when working with algorithms that do not allow
	unary productions, and completely removing the unary productions
	would require loss of useful information.  The Tree is modified
	directly (since it is passed by reference) and no value is returned.

	@param tree: The Tree to be collapsed
	@type  tree: C{Tree}
	@param collapsePOS: 'False' (default) will not collapse the parent of leaf nodes (ie. 
						Part-of-Speech tags) since they are always unary productions
	@type  collapsePOS: C{boolean}
	@param collapseRoot: 'False' (default) will not modify the root production
						if it is unary.  For the Penn WSJ treebank corpus, this corresponds
						to the TOP -> productions.
	@type collapseRoot: C{boolean}
	@param joinChar: A string used to connect collapsed node values (default = "+")
	@type  joinChar: C{string}
	"""

	if collapseRoot == False and isinstance(tree, Tree) and len(tree) == 1:
		nodeList = [tree[0]]
	else:
		nodeList = [tree]

	# depth-first traversal of tree
	while nodeList:
		node = nodeList.pop()
		if isinstance(node,Tree):
			if len(node) == 1 and isinstance(node[0], Tree) and (collapsePOS == True or isinstance(node[0,0], Tree)):
				node.node += joinChar + node[0].node
				node[0:] = [child for child in node[0]]
				# since we assigned the child's children to the current node, 
				# evaluate the current node again
				nodeList.append(node)
			else:
				for child in node:
					nodeList.append(child)
	return tree

def introducepreterminals(tree):
	""" Introduce preterminals with artificial POS-tags where needed
	(i.e., for every terminal with siblings.)

	>>> tree = Tree("(S (X a b (CD c d) e))")
	>>> tree = introducepreterminals(tree)
	>>> print tree.pprint(margin=999)
	(S (X (X/a a) (X/b b) (CD (CD/c c) (CD/d d)) (X/e e)))

	"""
	assert isinstance(tree, Tree)
	nodeList = [tree]
	while nodeList:
		node = nodeList.pop()
		hassiblings = len(node) > 1
		for n, child in enumerate(node):
			if isinstance(child, Tree):
				nodeList.append(child)
			elif hassiblings:
				node[n] = Tree("%s/%s" % (node.node, child), [child])
	return tree

def getbits(bitset):
	""" Iterate over the indices of set bits in a bitset. """
	for n in xrange(999):
		if bitset & 1: yield n
		elif not bitset: break
		bitset >>= 1


from bit import fanout as bitfanout
def slowfanout(tree):
	return len(list(ranges(sorted(tree.leaves())))) if isinstance(tree, Tree) else 1

def fastfanout(tree):
	return bitfanout(tree.bitset) if isinstance(tree, Tree) else 1

#can only be used with ImmutableTrees because of memoization.
#fanout = memoize(slowfanout)
fanout = fastfanout

def complexity(tree):
	return fanout(tree) + sum(map(fanout, tree))

def complexityfanout(tree):
	return (fanout(tree) + sum(map(fanout, tree)),
			fanout(tree))

def fanoutcomplexity(tree):
	return (fanout(tree),
			fanout(tree) + sum(map(fanout, tree)))

def addbitsets(tree):
	if isinstance(tree, basestring):
		result = ImmutableTree.parse(tree, parse_leaf=int)
	elif isinstance(tree, ImmutableTree): result = tree
	elif isinstance(tree, Tree): result = tree.freeze()
	else: raise ValueError("expected string or tree object")
	for a in result.subtrees():
		a.bitset = sum(1 << n for n in a.leaves())
	return result

def defaultrightbin(node, sep="|", h=999, fanout=False):
	""" Binarize one constituent with a right factored binarization.
	Children remain unmodified. Bottom-up version. Nodes must contain
	bitsets (use addbitsets()). """
	if len(node) < 3: return node
	label = node.node
	childlabels = [child.node for child in node]
	prev = node[-1]
	for i in range(len(node) - 2, 0, -1):
		newbitset = node[i].bitset | prev.bitset
		newlabel = "%s%s<%s>" % (label, sep, "-".join(childlabels[i:i+h]))
		if fanout:
			nodefanout = bitfanout(newbitset)
			if nodefanout > 1: newlabel += "_" + str(nodefanout)
		prev = ImmutableTree(newlabel, [node[i], prev])
		prev.bitset = newbitset
	result = ImmutableTree(label, [node[0], prev])
	result.bitset = node[0].bitset | prev.bitset
	return result

def minimalbinarization(tree, score, sep="|", head=None, h=999):
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
	>>> print minimalbinarization(addbitsets(tree), complexityfanout, head=2)
	(X (A 0) (X|<B-E-D-C> (B 3) (X|<E-D-C> (E 8) (X|<D-C> (D 7) (C 5)))))
	>>> tree = "(X (A 0) (B 3) (C 5) (D 7) (E 8))"
	>>> print minimalbinarization(addbitsets(tree), complexityfanout, head=2,h=1)
	(X (A 0) (X|<B> (B 3) (X|<E> (E 8) (X|<D> (D 7) (C 5)))))
	>>> tree = "(A (B1 (t 6) (t 13)) (B2 (t 3) (t 7) (t 10))  (B3 (t 1) (t 9) (t 11) (t 14) (t 16)) (B4 (t 0) (t 5) (t 8)))"
	>>> a = minimalbinarization(addbitsets(tree), complexityfanout)
	>>> b = minimalbinarization(addbitsets(tree), fanoutcomplexity)
	>>> print max(map(complexityfanout, a.subtrees()))
	(14, 6)
	>>> print max(map(complexityfanout, b.subtrees()))
	(15, 5)
	"""
	def newproduction(a, b):
		""" return a new `production' (here a tree) combining a and b """
		if head is not None: siblings = (nonterms[a] | nonterms[b])[:h]
		else: siblings = getbits(nonterms[a] | nonterms[b])
		newlabel = "%s%s<%s>" % (tree.node, sep,
				"-".join(labels[x] for x in siblings))
		new = ImmutableTree(newlabel, [a, b])
		new.bitset = a.bitset | b.bitset
		return new
	if len(tree) <= 2: return tree
	#don't bother with optimality if this particular node is not discontinuous
	#do default right factored binarization instead
	elif fanout(tree) == 1 and all(fanout(a) == 1 for a in tree):
		return defaultrightbin(tree, sep, h)
	from agenda import Agenda
	labels = [a.node for a in tree]
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
	goal = (1L << len(tree)) - 1
	if head is None:
		for n, a in enumerate(tree):
			nonterms[a] = 1L << n
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
			if n == head: continue
			# (add initial unary here)
			p = newproduction(a, hd)
			x = score(p)
			agenda[p] = workingset[p] = x + (x[0],)
			nonterms[p] = nonterms[a] | nonterms[hd]
			revnonterms[nonterms[p]] = p
	while agenda:
		entry = agenda.popentry()
		p = agenda.getkey(entry); x = agenda.getval(entry)
		if nonterms[p] == goal:
			# (add final unary here)
			p = ImmutableTree(tree.node, p[:])
			p.bitset = tree.bitset
			return p
		for p1, y in workingset.items():
			if p1 not in workingset: continue
			# this is inefficient. we should have a single query for all
			# items not overlapping with p
			elif nonterms[p] & nonterms[p1]: continue
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
					if a in agenda: del agenda[a]
				nonterms[p2] = p2nonterms
				revnonterms[p2nonterms] = p2
				agenda[p2] = workingset[p2] = x2
	raise ValueError

def optimalbinarize(tree, sep="|", headdriven=False, h=None, v=1, ancestors=()):
	""" Recursively binarize a tree optimizing for complexity.
	v=0 is not implemented. """
	if headdriven: return recbinarizetreehd(addbitsets(tree), sep, h, v, ())
	tree = Tree.convert(tree)
	for a in list(tree.subtrees(lambda x: len(x) > 1))[::-1]:
		a.sort(key=lambda x: x.leaves())
	return recbinarizetree(addbitsets(tree), sep)

def recbinarizetree(tree, sep="|"):
	""" postorder / bottom-up binarization """
	if not isinstance(tree, Tree): return tree
	newtree = ImmutableTree(tree.node,
		[recbinarizetree(t, sep) for t in tree])
	newtree.bitset = tree.bitset
	return minimalbinarization(newtree, complexityfanout, sep)

def recbinarizetreehd(tree, sep="|", h=None, v=1, ancestors=()):
	""" postorder / bottom-up binarization """
	if not isinstance(tree, Tree): return tree
	parentstr = "^<%s>" % ("-".join(ancestors[:v-1])) if v > 1 else ""
	newtree = ImmutableTree(tree.node + parentstr,
		[recbinarizetreehd(t, sep, h, v, (tree.node,) + ancestors)
														for t in tree])
	newtree.bitset = tree.bitset
	return minimalbinarization(newtree, complexityfanout, sep,
			head=len(tree) - 1, h=h)

def disc(node):
	""" This function evaluates whether a particular node is locally
	discontinuous.  The root node will, by definition, be continuous.
	Nodes can be continuous even if some of their children are discontinuous.
	"""
	if not isinstance(node, Tree): return False
	return len(list(ranges(sorted(node.leaves())))) > 1

def addfanoutmarkers(tree):
	""" Modifies tree so that the label of each node with a fanout > 1 contains
	a marker "_n" indicating its fanout. """
	for st in tree.subtrees():
		leaves = set(st.leaves())
		fanout = len([a for a in sorted(leaves) if a - 1 not in leaves])
		if fanout > 1 and not st.node.endswith("_%d" % fanout):
			st.node += "_%d" % fanout
	return tree

def contsets(tree):
	""" partition children of tree into continuous subsets
	>>> tree = Tree.parse("(VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) (VVPP 5))", parse_leaf=int)
	>>> print list(contsets(tree))
	[[Tree('PP', [Tree('APPR', [0]), Tree('ART', [1]), Tree('NN', [2])])], 
	[Tree('CARD', [4]), Tree('VVPP', [5])]]
	"""
	rng = -1
	subset = []
	mins = [min(a.leaves()) if isinstance(a, Tree) else a for a in tree]

	for a in sorted(tree.leaves()):
		if rng >= 0 and a != rng + 1:
			yield subset
			subset = []
		if a in mins:
			subset.append(tree[mins.index(a)])
		rng = a
	if subset: yield subset

def splitdiscnodes(tree, markorigin=False):
	""" Boyd (2007): Discontinuity revisited.

	>>> tree = Tree.parse("(S (VP (VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) (VVPP 5)) (VAINF 6)) (VMFIN 3))", parse_leaf=int)
	>>> print splitdiscnodes(tree.copy(True)).pprint(margin=999)
	(S (VP* (VP* (PP (APPR 0) (ART 1) (NN 2)))) (VMFIN 3) (VP* (VP* (CARD 4) (VVPP 5)) (VAINF 6)))
	>>> print splitdiscnodes(tree, markorigin=True).pprint(margin=999)
	(S (VP*0 (VP*0 (PP (APPR 0) (ART 1) (NN 2)))) (VMFIN 3) (VP*1 (VP*1 (CARD 4) (VVPP 5)) (VAINF 6)))
	"""
	from grammar import postorder
	for node in postorder(tree):
		result = []
		for child in node:
			if disc(child):
				result.extend(Tree((("%s*%d" % (child.node, n))
					if markorigin else '%s*' % child.node), childsubset)
					for n, childsubset in enumerate(contsets(child)))
			else: result.append(child)
		node[:] = result
	return canonicalize(tree)

splitlabel = re.compile(r"\*[0-9]*$")
def mergediscnodes(tree):
	""" Reverse of Boyd (2007): Discontinuity revisited.
	>>> tree = Tree.parse("(S (VP (VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) (VVPP 5)) (VAINF 6)) (VMFIN 3))", parse_leaf=int)
	>>> print mergediscnodes(splitdiscnodes(tree)).pprint(margin=999)
	(S (VP (VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) (VVPP 5)) (VAINF 6)) (VMFIN 3))
	>>> print mergediscnodes(splitdiscnodes(tree, markorigin=True)).pprint(margin=999)
	(S (VP (VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) (VVPP 5)) (VAINF 6)) (VMFIN 3))
	"""
	for node in tree.subtrees():
		merge = defaultdict(list)
		for child in node:
			if isinstance(child, Tree) and splitlabel.search(child.node):
				merge[splitlabel.sub("", child.node)].append(child)
		node[:] = [child for child in node if not isinstance(child, Tree)
						or not splitlabel.search(child.node)]
		for label, subsets in merge.iteritems():
			children = [child for component in subsets for child in component]
			node.append(Tree(label, children))
	return canonicalize(tree)

#################################################################
# Demonstration
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
	print "original", tree

	# collapse subtrees with only one child
	collapsedTree = tree.copy(True)
	#collapse_unary(collapsedTree)

	# convert the tree to CNF
	cnfTree = collapsedTree.copy(True)
	lcnfTree = collapsedTree.copy(True)
	binarize(cnfTree, factor="right", horzMarkov=2)
	binarize(lcnfTree, factor="left", horzMarkov=2)

	# convert the tree to CNF with parent annotation 
	# (one level) and horizontal smoothing of order two
	parentTree = collapsedTree.copy(True)
	binarize(parentTree, horzMarkov=2, vertMarkov=1)

	# convert the tree back to its original form
	original = cnfTree.copy(True)
	original2 = lcnfTree.copy(True)
	unbinarize(original)
	unbinarize(original2)

	print "binarized", cnfTree
	print "Sentences the same? ", tree == original, tree == original2
	assert tree == original and tree == original2

def testminbin():
	""" Verify that all optimal parsing complexities are lower than or equal
	to the complexities of right-to-left binarizations. """
	from treebank import NegraCorpusReader
	import time
	#corpus = NegraCorpusReader("../rparse", "negraproc.export",
	#corpus = NegraCorpusReader("..", "negra-corpus.export", encoding="iso-8859-1",
	#	movepunct=True, headorder=True, headfinal=True, headreverse=False)
	corpus = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1",
		movepunct=True, headorder=False, headfinal=True, headreverse=False)
	total = violations = violationshd = 0
	for n, tree, sent in zip(count(), corpus.parsed_sents()[:-2000], corpus.sents()):
		#if len(tree.leaves()) <= 25: continue
		begin = time.clock()
		t = addbitsets(tree)
		if all(fanout(x) == 1 for x in t.subtrees()): continue
		print n, tree, '\n', " ".join(sent)
		total += 1
		foo = optimalbinarize(tree.copy(True), headdriven=False, h=None, v=1)
		print time.clock() - begin, "s\n"
		continue

		bar = Tree.convert(tree)
		# undo head-ordering to get a normal right-to-left binarization
		for a in list(bar.subtrees(lambda x: len(x) > 1))[::-1]:
			a.sort(key=lambda x: x.leaves())
		bar.chomsky_normal_form()
		bar = addbitsets(bar)
		if max(map(complexityfanout, foo.subtrees())) > max(map(complexityfanout, bar.subtrees())):
			print "non-hd"
			print tree.pprint(margin=999)
			print max(map(complexityfanout, foo.subtrees())), foo
			print max(map(complexityfanout, bar.subtrees())), bar
			print '\n'
			violations += 1
			assert False

		foo = optimalbinarize(tree.copy(True), headdriven=True, h=1, v=1)
		bar = Tree.convert(tree)
		bar.chomsky_normal_form(horzMarkov=1)
		#binarize(bar, horzMarkov=1)
		bar = addbitsets(bar)
		if max(map(complexityfanout, foo.subtrees())) > max(map(complexityfanout, bar.subtrees())):
			print "hd"
			print tree.pprint(margin=999)
			print max(map(complexityfanout, foo.subtrees())), foo
			print max(map(complexityfanout, bar.subtrees())), bar
			print '\n'
			violationshd += 1
	print "violations: %d / %d" % (violations, total)
	print "violationshd: %d / %d" % (violationshd, total)
	assert violations == violationshd == 0

def testsplit():
	from treebank import NegraCorpusReader
	correct = wrong = 0
	#n = NegraCorpusReader("../rparse", "tigerproc.export")
	n = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1")
	for tree in n.parsed_sents():
		if mergediscnodes(splitdiscnodes(tree)) == tree:
			correct += 1
		else:
			wrong += 1
	total = float(len(n.sents()))
	print "correct", correct, "=", 100*correct/total, "%"
	print "wrong", wrong, "=", 100*wrong/total, "%"

def test():
	demo()
	testminbin()
	testsplit()

def main():
	import sys, codecs
	from getopt import gnu_getopt, GetoptError
	from treebank import NegraCorpusReader, DiscBracketCorpusReader, \
			BracketCorpusReader, export
	flags = ("headdriven", "markorigin", "removepunct", "movepunct")
	options = ('factor=', 'markorigin=', 'inputfmt=', 'outputfmt=',
			'inputenc=', 'outputenc=', 'slice=')
	try:
		opts, args = gnu_getopt(sys.argv[1:], "h:v:", flags + options)
		action, input, output = args
	except Exception, err:
		print "error:", err
		usage()
		exit(2)
	else: opts = dict(opts)

	# read input
	if opts.get('--inputfmt', 'export') == 'export':
		Reader = NegraCorpusReader
	elif opts.get('--inputfmt') == 'discbracket':
		Reader = DiscBracketCorpusReader
	elif opts.get('--inputfmt') == 'bracket':
		Reader = BracketCorpusReader
	else: raise ValueError("unrecognized format: %r" % opts.get('--inputfmt'))

	corpus = Reader(".", input,
			encoding=opts.get('--inputenc', 'utf-8'),
			headorder="--headdriven" in opts,
			headfinal=True, headreverse=False,
			removepunct="--removepunct" in opts,
			movepunct="--movepunct" in opts)
	start, end = opts.get('--slice', ':').split(':')
	start = int(start) if start else None
	end = int(end) if end else None
	trees, sents = corpus.parsed_sents()[start:end], corpus.sents()[start:end]

	# apply transformation
	if action == "binarize":
		factor = opts.get('--factor', 'right')
		h = int(opts['-h']) if 'h' in opts else None
		v = int(opts.get('-v', 0))
		for a in trees: binarize(a, factor, h, v)
	elif action == "unbinarize":
		for a in trees: unbinarize(a, factor, h, v)
	elif action == "optimalbinarize":
		sep="|"; headdriven = "--headdriven" in opts
		h = int(opts['-h']) if 'h' in opts else None
		v = int(opts.get('-v', 0))
		trees = [optimalbinarize(a, sep, headdriven, h, v) for a in trees]
	elif action == "introducepreterminals":
		map(introducepreterminals, trees)
	elif action == "splitdisc":
		for a in trees: splitdiscnodes(a, '--markorigin' in opts)
	elif action == "mergedisc":
		for a in trees: mergediscnodes(a)
	elif action in ("none"): pass
	else: raise ValueError("unrecognized action: %r" % action)

	# write output
	if opts.get('--outputfmt', 'export') == 'export':
		codecs.open(output, "w", encoding=opts.get('outputenc', 'utf-8')
			).writelines(export(*x) for x in zip(trees, sents, count(1)))
	elif opts.get('--outputfmt') == 'discbracket':
		codecs.open(output, "w", encoding=opts.get('outputenc', 'utf-8')
			).writelines("%s\t%s\n" % (tree.pprint(margin=9999), " ".join(sent))
			for tree, sent in zip(trees, sents))
	elif opts.get('--outputfmt') == 'bracket':
		codecs.open(output, "w", encoding=opts.get('outputenc', 'utf-8')
			).writelines("%s\n" % Tree.parse(tree.pprint(margin=9999),
				parse_leaf=lambda l: sent[int(l)]).pprint(margin=9999)
			for tree, sent in zip(trees, sents))
	else: raise ValueError("unrecognized format: %r" % opts.get('--outputfmt'))

def usage():
	import sys
	print """Treebank binarization and conversion
usage: %s [options] action input output
where input and output are treebanks, and action is one of:
    binarize [-h x] [-v x] [--factor left|right]
    optimalbinarize [-h x] [-v x]
    unbinarize
    introducepreterminals
    splitdisc [--markorigin]
    mergedisc
    none

options may consist of (* marks default option):
  --inputfmt [*export|discbracket|bracket]
  --outputfmt [*export|discbracket|bracket]
  --inputenc [*UTF-8|ISO-8859-1|...]
  --outputenc [*UTF-8|ISO-8859-1|...]
  --slice n:m select a range of sentences from input starting with n, up to but
              not including m; n or m can be left out or negative, as in Python
  --factor [left|*right] whether binarization factors to the left or right
  -h n           horizontal markovization. default: infinite
  -v n           vertical markovization. default: 1
  --headdriven   turn on marking of heads; also affects binarization.
                 requires the file "negra.headrules".
  --removepunct  remove any punctuation.
  --movepunct    re-attach punctuation to nearest constituent to minimize
                 discontinuity.
Note: some of these transformations are specific to discontinuous treebanks,
    specifically the Negra/Tiger treebanks. In the output only POS & phrasal
    labels are retained.""" % sys.argv[0]

__all__ = ["binarize", "unbinarize", "collapse_unary", "introducepreterminals",
	"splitdiscnodes", "mergediscnodes", "optimalbinarize", "defaultrightbin",
	"addfanoutmarkers"]

if __name__ == '__main__': main()
