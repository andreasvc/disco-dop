# Natural Language Toolkit: Tree Transformations
#
# Copyright (C) 2005-2007 Oregon Graduate Institute
# Author: Nathan Bodenstab <bodenstab@cslu.ogi.edu>
# URL: <http://www.nltk.org/>
# For license information, see LICENSE.TXT

# andreasvc: modified to introduce a new unary production for the first/last
# element in the RHS.
# The markovization direction is changed:
# (A (B ) (C ) (D )) becomes:
#	left:  (A (A|<D> (A|<C-D> (A|<B-C> (B )) (C )) (D )))
#   right: (A (A|<B> (B ) (A|<B-C> (C ) (A|<C-D> (D )))))
# in this way the markovization represents the history of the nonterminals that
# have *already* been parsed, instead of those still to come. 

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
from heapq import heappush, heappop
from itertools import chain
from nltk import Tree, ImmutableTree, memoize
from grammar import rangeheads

def collinize(tree, factor="right", horzMarkov=None, vertMarkov=0, childChar="|", parentChar="^", tailMarker="$", headMarked=None, minMarkov=3):
	"""
	>>> sent = "das muss man jetzt machen".split()
	>>> tree = Tree("(S (VP (PDS 0) (ADV 3) (VVINF 4)) (PIS 2) (VMFIN 1))")
	>>> collinize(tree, horzMarkov=0, tailMarker=''); print tree.pprint(margin=999)
	(S (S|<> (VP (VP|<> (PDS 0) (VP|<> (ADV 3) (VP|<> (VVINF 4))))) (S|<> (PIS 2) (S|<> (VMFIN 1)))))
	>>> un_collinize(tree); print tree
	(S (VP (PDS 0) (ADV 3) (VVINF 4)) (PIS 2) (VMFIN 1))
	>>> collinize(tree, horzMarkov=1, tailMarker=''); print tree.pprint(margin=999)
	(S (S|<VP> (VP (VP|<PDS> (PDS 0) (VP|<ADV> (ADV 3) (VP|<VVINF> (VVINF 4))))) (S|<PIS> (PIS 2) (S|<VMFIN> (VMFIN 1)))))
	>>> un_collinize(tree); collinize(tree, horzMarkov=2, tailMarker=''); print tree.pprint(margin=999)
	(S (S|<VP> (VP (VP|<PDS> (PDS 0) (VP|<PDS-ADV> (ADV 3) (VP|<ADV-VVINF> (VVINF 4))))) (S|<VP-PIS> (PIS 2) (S|<PIS-VMFIN> (VMFIN 1)))))
	>>> un_collinize(tree); collinize(tree, factor="left", horzMarkov=2, tailMarker=''); print tree.pprint(margin=999)
	(S (S|<VMFIN> (S|<PIS-VMFIN> (S|<VP-PIS> (VP (VP|<VVINF> (VP|<ADV-VVINF> (VP|<PDS-ADV> (PDS 0)) (ADV 3)) (VVINF 4)))) (PIS 2)) (VMFIN 1)))

	>>> tree = Tree("(S (NN 2) (VP (PDS 0) (ADV 3) (VAINF 4)) (VMFIN 1))")
	>>> un_collinize(tree); collinize(tree, horzMarkov=2, tailMarker=''); print tree.pprint(margin=999)
	(S (S|<NN> (NN 2) (S|<NN-VP> (VP (VP|<PDS> (PDS 0) (VP|<PDS-ADV> (ADV 3) (VP|<ADV-VAINF> (VAINF 4))))) (S|<VP-VMFIN> (VMFIN 1)))))
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

			# chomsky normal form factorization
			if len(node) > (minMarkov - 1) and isinstance(node[0], Tree):
				childNodes = [child.node for child in node]
				nodeCopy = node.copy()
				numChildren = len(nodeCopy)
				headidx = 0

				# insert an initial artificial nonterminal
				if factor == "right":
					start = 0
					end = min(1, horzMarkov)
				else: # factor == "left"
					start = numChildren - min(1, horzMarkov)
					end = numChildren
				siblings = "-".join(childNodes[start:end])
				newNode = Tree("%s%s<%s>%s" % (originalNode, childChar,
												siblings, parentString), [])
				node[:] = [newNode]
				node = newNode

				curNode = node
				for i in range(1, numChildren):
					marktail = tailMarker if i + 1 == numChildren else ''
					newNode = Tree('', [])
					if factor == "right":
						start = max(i - horzMarkov + 1, 0)
						end = i + 1
						curNode[:] = [nodeCopy.pop(0), newNode]
					else: # factor == "left":
						start = headidx + numChildren - i - 1
						end = start + horzMarkov
						curNode[:] = [newNode, nodeCopy.pop()]
					# switch direction upon encountering the head
					if (headMarked and headMarked in childNodes[i] 
						and factor == "right"):
						headidx = i
						factor = "left"
						start = headidx + numChildren - i - 1
						end = start + horzMarkov
					siblings = "-".join(childNodes[start:end])
					newNode.node = "%s%s%s<%s>%s" % (originalNode, childChar,
											marktail, siblings, parentString)

					curNode = newNode
				curNode.append(nodeCopy.pop())
	

def un_collinize(tree, expandUnary=True, childChar="|", parentChar="^", unaryChar="+"):
	# Traverse the tree-depth first keeping a pointer to the parent for modification purposes.
	agenda = [(tree, [])]
	while agenda:
		node, parent = agenda.pop()
		if isinstance(node, Tree):
			# if the node contains the 'childChar' character it means that
			# it is an artificial node and can be removed, although we still need
			# to move its children to its parent
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
	while nodeList != []:
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

@memoize
def fanout(tree):
	if not isinstance(tree, Tree): return 1
	return len(rangeheads(sorted(tree.leaves())))

def complexityfanout(tree):
	return (fanout(tree) + sum(map(fanout, tree)), fanout(tree))

def fanoutcomplexity(tree):
	return (fanout(tree), fanout(tree) + sum(map(fanout, tree)))

def random(tree):
	from random import randint
	return randint(1, 25),

def minimalbinarization(tree, score, sep="|", head=None):
	"""
	Implementation of Gildea (2010): Optimal parsing strategies for linear
	context-free rewriting systems.
	Expects an immutable tree where the terminals are integers corresponding to
	indices.

	tree is the tree for which the optimal binarization of its top production
	will be searched. score is a function from binarized trees to some value,
	where lower is better (the value can be numeric or a tuple or anything
	else which supports comparisons)

	>>> tree=ImmutableTree.parse("(NP (ART 0) (ADJ 2) (NN 1))", parse_leaf=int)
	>>> print minimalbinarization(tree, complexityfanout)
	(NP (NP|<ART-NN> (ART 0) (NN 1)) (ADJ 2))

	>>> tree = "(X (A 0) (B 1) (C 2) (D 3) (E 4))"
	>>> tree=ImmutableTree.parse(tree, parse_leaf=int)
	>>> head = ImmutableTree("C", [2])
	>>> print minimalbinarization(tree, complexityfanout, head=head)
	(X (X|<A-B-C-D> (X|<A-B-C> (A 0) (X|<B-C> (B 1) (C 2))) (D 3)) (E 4))

	>>> tree = "(X (A 0) (B 3) (C 5) (D 7) (E 8))"
	>>> tree=ImmutableTree.parse(tree, parse_leaf=int)
	>>> head = ImmutableTree("C", [5])
	>>> print minimalbinarization(tree, complexityfanout, head=head)
	(X (X|<A-C-D-E> (A 0) (X|<C-D-E> (X|<C-D> (C 5) (D 7)) (E 8))) (B 3))

	>>> tree = "(A (B1 (t 6) (t 13)) (B2 (t 3) (t 7) (t 10))  (B3 (t 1) (t 9) (t 11) (t 14) (t 16)) (B4 (t 0) (t 5) (t 8)))"
	>>> tree=ImmutableTree.parse(tree, parse_leaf=int)
	>>> a = minimalbinarization(tree, complexityfanout)
	>>> b = minimalbinarization(tree, fanoutcomplexity)
	>>> print max(map(complexityfanout, a.subtrees()))
	(14, 6)
	>>> print max(map(complexityfanout, b.subtrees()))
	(15, 5)
	"""
	def newproduction(a, b):
		""" return a new `production' (here a tree) combining a and b """
		# swap a and b according to linear precedence
		#if min(a.leaves()) > min(b.leaves()): a, b = b, a
		if (min(z for x, y in nonterms[a] for z in y) >
			min(z for x, y in nonterms[b] for z in y)): a, b = b, a
		newlabel = "%s%s<%s>" % (tree.node, sep, "-".join(x.node for x,y
				in sorted(nonterms[a] | nonterms[b], key=lambda z: z[1])))
		return ImmutableTree(newlabel, [a, b])
	if len(tree) <= 2: return tree
	workingset = set()
	agenda = []
	nonterms = {}
	goal = frozenset((a, tuple(a.leaves())) for a in tree)
	if head:
		# head driven binarization:
		# add all non-head nodes to the working set,
		# add all combination of non-head nodes with head to agenda
		# caveat: Gildea (2011) shows that this problem is NP hard.
		for a in (a for a in tree if a != head):
			workingset.add((score(a), a))
			nonterms[a] = frozenset([(a, tuple(a.leaves()))])
		nonterms[head] = frozenset([(head, tuple(head.leaves()))])
		for a in (a for a in tree if a != head):
			p = newproduction(a, head); x = score(p)
			workingset.add((x, p))
			heappush(agenda, (x, p))
			nonterms[p] = nonterms[a] | nonterms[head]
	else:
		for a in tree:
			workingset.add((score(a), a))
			heappush(agenda, (score(a), a))
			nonterms[a] = frozenset([(a, tuple(a.leaves()))])
	while agenda:
		x, p = item = heappop(agenda)
		if item not in workingset: continue
		if nonterms[p] == goal:
			p = ImmutableTree(tree.node, p[:])
			return p
		for y, p1 in list(workingset):
			if (y, p1) not in workingset or nonterms[p] & nonterms[p1]:
				continue
			p2 = newproduction(p, p1)
			p2nonterms = nonterms[p] | nonterms[p1]
			x2 = tuple(max(a) for a in zip(score(p2), y, x))
			# enumerate binarizations which dominate the same subset 
			# of the right hand side, but have a lower score
			inferior = set((z, p3) for z, p3 in workingset
							if nonterms[p3] == p2nonterms and x2 < z)
			if inferior or p2nonterms not in nonterms.values():
				workingset -= inferior
				workingset.add((x2, p2))
				heappush(agenda, (x2, p2))
				nonterms[p2] = p2nonterms
				for _, p3 in inferior: del nonterms[p3]

def binarizetree(tree, sep="|"):
	""" Recursively binarize a tree. Tree needs to be immutable."""
	if not isinstance(tree, Tree): return tree
	# bypass algorithm when there are no discontinuities:
	elif len(rangeheads(tree.leaves())) == 1:
		newtree = Tree(tree.node, map(lambda t: binarizetree(t, sep), tree))
		newtree.chomsky_normal_form(childChar=sep)
		return newtree
	return Tree(tree.node, map(lambda t: binarizetree(t, sep),
				minimalbinarization(tree, complexityfanout, sep)))

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
	collinize(cnfTree, factor="right", horzMarkov=2, minMarkov=1)
	collinize(lcnfTree, factor="left", horzMarkov=2, minMarkov=1)
	
	# convert the tree to CNF with parent annotation (one level) and horizontal smoothing of order two
	parentTree = collapsedTree.copy(True)
	collinize(parentTree, horzMarkov=2, vertMarkov=1)
	
	# convert the tree back to its original form (used to make CYK results comparable)
	original = cnfTree.copy(True)
	original2 = lcnfTree.copy(True)
	un_collinize(original)
	un_collinize(original2)
	
	print "binarized", cnfTree
	print "Sentences the same? ", tree == original, tree == original2
	assert tree == original and tree == original2
	
	#draw_trees(tree, collapsedTree, cnfTree, parentTree, original)

if __name__ == '__main__':
	from doctest import testmod, NORMALIZE_WHITESPACE, ELLIPSIS
	# do doctests, but don't be pedantic about whitespace (I suspect it is the
	# militant anti-tab faction who are behind this obnoxious default)
	fail, attempted = testmod(verbose=False, optionflags=NORMALIZE_WHITESPACE | ELLIPSIS)
	#demo()
	if attempted and not fail: print "%d doctests succeeded!" % attempted

__all__ = ["collinize", "un_collinize", "collapse_unary"]
