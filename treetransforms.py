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
#   left:  (A (A|<D> (A|<C-D> (A|<B-C> (B )) (C )) (D )))
#   right: (A (A|<B> (B ) (A|<B-C> (C ) (A|<C-D> (D )))))
# in this way the markovization represents the history of the nonterminals that
# have *already* been parsed, instead of those still to come (assuming
# bottom-up parsing).

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
from grammar import ranges, canonicalize
from orderedset import OrderedSet
import re

def collinize(tree, factor="right", horzMarkov=None, vertMarkov=0,
	childChar="|", parentChar="^", headMarked=None,
	rightMostUnary=True, leftMostUnary=True,
	tailMarker="$"):
	"""
	>>> sent = "das muss man jetzt machen".split()
	>>> tree = Tree("(S (VP (PDS 0) (ADV 3) (VVINF 4)) (PIS 2) (VMFIN 1))")
	>>> collinize(tree, horzMarkov=0, tailMarker=''); print tree.pprint(margin=999)
	(S (S|<> (VP (VP|<> (PDS 0) (VP|<> (ADV 3) (VP|<> (VVINF 4))))) (S|<> (PIS 2) (S|<> (VMFIN 1)))))
	>>> un_collinize(tree); print tree
	(S (VP (PDS 0) (ADV 3) (VVINF 4)) (PIS 2) (VMFIN 1))

	>>> collinize(tree, horzMarkov=1, tailMarker=''); print tree.pprint(margin=999)
	(S (S|<VP> (VP (VP|<PDS> (PDS 0) (VP|<ADV> (ADV 3) (VP|<VVINF> (VVINF 4))))) (S|<PIS> (PIS 2) (S|<VMFIN> (VMFIN 1)))))

	>>> un_collinize(tree); collinize(tree, horzMarkov=1, leftMostUnary=False, rightMostUnary=True, tailMarker=''); print tree.pprint(margin=999)
	(S (VP (PDS 0) (VP|<ADV> (ADV 3) (VP|<VVINF> (VVINF 4)))) (S|<PIS> (PIS 2) (S|<VMFIN> (VMFIN 1))))

	>>> un_collinize(tree); collinize(tree, horzMarkov=1, leftMostUnary=True, rightMostUnary=False, tailMarker=''); print tree.pprint(margin=999)
	(S (S|<VP> (VP (VP|<PDS> (PDS 0) (VP|<ADV> (ADV 3) (VVINF 4)))) (S|<PIS> (PIS 2) (VMFIN 1))))
	
	>>> un_collinize(tree); collinize(tree, horzMarkov=1, leftMostUnary=False, rightMostUnary=False, tailMarker=''); print tree.pprint(margin=999)
	(S (VP (PDS 0) (VP|<ADV> (ADV 3) (VVINF 4))) (S|<PIS> (PIS 2) (VMFIN 1)))

	>>> un_collinize(tree); collinize(tree, horzMarkov=2, tailMarker=''); print tree.pprint(margin=999)
	(S (S|<VP> (VP (VP|<PDS> (PDS 0) (VP|<PDS-ADV> (ADV 3) (VP|<ADV-VVINF> (VVINF 4))))) (S|<VP-PIS> (PIS 2) (S|<PIS-VMFIN> (VMFIN 1)))))

	>>> un_collinize(tree); collinize(tree, factor="left", horzMarkov=2, tailMarker=''); print tree.pprint(margin=999)
	(S (S|<VMFIN> (S|<PIS-VMFIN> (S|<VP-PIS> (VP (VP|<VVINF> (VP|<ADV-VVINF> (VP|<PDS-ADV> (PDS 0)) (ADV 3)) (VVINF 4)))) (PIS 2)) (VMFIN 1)))

	>>> tree = Tree("(S (NN 2) (VP (PDS 0) (ADV 3) (VAINF 4)) (VMFIN 1))")
	>>> collinize(tree, horzMarkov=2, tailMarker=''); print tree.pprint(margin=999)
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

			# chomsky normal form factorization
			if len(node) > 2 and isinstance(node[0], Tree):
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
				node[:] = []
				if leftMostUnary:
					node.append(newNode)
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
					if headMarked and headMarked in childNodes[i]:
						headidx = i
						# although it doesn't make any sense to switch from
						# left to right, it's supported anyway
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
					curNode[:] = nodeCopy.pop()[:]
	

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

def fanout(tree):
	if not isinstance(tree, Tree): return 1
	return len(list(ranges(sorted(tree.leaves()))))

#can only be used with ImmutableTrees because of memoization.
cachedfanout = memoize(fanout)

def complexityfanout(tree):
	return (cachedfanout(tree) + sum(map(cachedfanout, tree)),
			cachedfanout(tree))

def fanoutcomplexity(tree):
	return (cachedfanout(tree),
			cachedfanout(tree) + sum(map(cachedfanout, tree)))

def minimalbinarization(tree, score, sep="|", head=None, h=999):
	"""
	Implementation of Gildea (2010): Optimal parsing strategies for linear
	context-free rewriting systems.
	Expects an immutable tree where the terminals are integers corresponding to
	indices.

	tree is the tree for which the optimal binarization of its top production
	will be searched.
	score is a function from binarized trees to some value, where lower is
	better (the value can be numeric or anything else which supports
	comparisons)
	head is an optional index of the head node, specifying enables head-driven
	binarization

	>>> tree = "(X (A 0) (B 1) (C 2) (D 3) (E 4))"
	>>> tree1=ImmutableTree.parse(tree, parse_leaf=int)
	>>> tree2=Tree.parse(tree, parse_leaf=int)
	>>> tree2.chomsky_normal_form()
	>>> minimalbinarization(tree1, complexityfanout, head=2) == tree2
	True
	>>> tree = "(X (A 0) (B 3) (C 5) (D 7) (E 8))"
	>>> tree=ImmutableTree.parse(tree, parse_leaf=int)
	>>> print minimalbinarization(tree, complexityfanout, head=2)
	(X (D 7) (X|<A-B-E-C> (A 0) (X|<B-E-C> (B 3) (X|<E-C> (E 8) (C 5)))))
	>>> tree = "(X (A 0) (B 3) (C 5) (D 7) (E 8))"
	>>> tree=ImmutableTree.parse(tree, parse_leaf=int)
	>>> print minimalbinarization(tree, complexityfanout, head=2, h=1)
	(X (D 7) (X|<A> (A 0) (X|<B> (B 3) (X|<E> (E 8) (C 5)))))

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
		if head is not None: siblings = (nonterms[a] | nonterms[b])[:h]
		else: siblings =  sorted(nonterms[a] | nonterms[b])
		# swap a and b according to linear precedence
		#if min(a.leaves()) > min(b.leaves()): a, b = b, a
		#if (min(z for x, y in nonterms[a] for z in y) >
		#	min(z for x, y in nonterms[b] for z in y)): a, b = b, a
		# (disabled, do as postprocessing step instead).
		newlabel = "%s%s<%s>" % (tree.node, sep, "-".join(tree[x].node for x
			in siblings))
		return ImmutableTree(newlabel, [a, b])
	if len(tree) <= 2: return tree
	#don't bother with optimality if there are no discontinuities
	elif cachedfanout(tree) == 1:
		# do default right factored binarization
		node = Tree(tree.node, [a for a in tree])
		nodeCopy = node.copy(True)
		#fix parent annotation
		childnodes = [child.node for child in node]
		numChildren = len(tree)
		curNode = node
		for i in range(1, numChildren):
			newNode = Tree('', [])
			curNode[:] = [nodeCopy.pop(0), newNode]
			siblings = "-".join(childnodes[i:i+h])
			newNode.node = "%s%s<%s>" % (tree.node, sep, siblings)
			curNode = newNode
		curNode.node = nodeCopy[0].node
		curNode[:] = nodeCopy.pop()[:]
		return node.freeze() #suboptimal to convert back and forth but well
	agenda = []
	workingset = {}
	nonterms = {}
	revnonterms = {} # reverse mapping of nonterms ... better name anyone?
	#goal = frozenset((a, tuple(a.leaves())) for a in tree)
	goal = frozenset(range(len(tree)))	#optimization: use bitsets instead
	if head is None:
		for n, a in enumerate(tree):
			workingset[a] = score(a)
			agenda.append((workingset[a], a))
			nonterms[a] = frozenset([n])
			revnonterms[nonterms[a]] = a
	else:
		# head driven binarization:
		# add all non-head nodes to the working set,
		# add all combination of non-head nodes with head to agenda
		# caveat: Crescenzi et al. (2011) show that this problem is NP hard.
		hd = tree[head]
		for n, a in enumerate(tree):
			nonterms[a] = OrderedSet([n])
			revnonterms[nonterms[a]] = a
			if n != head:
				workingset[a] = score(a)
		for n, a in enumerate(tree):
			if n == head: continue
			# (add initial unary here)
			p = newproduction(a, hd)
			workingset[p] = score(p)
			agenda.append((workingset[p], p))
			nonterms[p] = nonterms[a] | nonterms[hd]
			revnonterms[nonterms[p]] = p
	# heapify agenda, use sort() to get stable sort 
	#[but heappop is not stable, so does it matter??]
	agenda.sort()
	while agenda:
		x, p = heappop(agenda)
		if p not in workingset or workingset[p] != x: continue
		if nonterms[p] == goal:
			# (add final unary here)
			p = ImmutableTree(tree.node, p[:])
			return p
		for p1, y in workingset.items():
			if p1 not in workingset or workingset[p1] != y or nonterms[p] & nonterms[p1]:
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
			# important: the score is the maximum score up till now
			x2 = tuple(max(a) for a in zip(score(p2), y, x))
			#if new or better:
			if p2nonterms not in revnonterms or revnonterms[p2nonterms] > x2:
				for a,b in nonterms.items():
					if b == p2nonterms:
						del nonterms[a]
						del workingset[a]
						#might as well remove it from agenda as well if that
						#were supported
				workingset[p2] = x2
				heappush(agenda, (x2, p2))
				nonterms[p2] = p2nonterms
				revnonterms[p2nonterms] = p2
	raise ValueError

def binarizetree(tree, sep="|", headdriven=False, h=None, v=1, ancestors=()):
	""" Recursively binarize a tree optimizing for complexity.
	Tree needs to be immutable. v=0 is not implemented. """
	if not isinstance(tree, Tree): return tree
	if not headdriven:
		return Tree(tree.node, sorted(map(
			lambda t: binarizetree(t, sep),
			minimalbinarization(tree, complexityfanout, sep)),
				key=lambda n: min(n.leaves()) if isinstance(n, Tree) else 1))
	parentstr = "^<%s>" % ("-".join(ancestors[:v-1])) if v > 1 else ""
	return Tree(tree.node + parentstr, sorted(map(
		lambda t: binarizetree(t, sep, headdriven, h=h, v=v,
				ancestors=(tree.node,)+ancestors),
		minimalbinarization(tree,complexityfanout,sep,head=len(tree)-1,h=h)),
		key=lambda n: min(n.leaves()) if isinstance(n, Tree) else 1))

def disc(tree):
	if not isinstance(tree, Tree): return False
	return len(list(ranges(sorted(tree.leaves())))) > 1

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
				if markorigin:
					result.extend(Tree("%s*%d" % (child.node, n), childsubset)
						for n, childsubset in enumerate(contsets(child)))
				else:
					result.extend(Tree(child.node + "*", childsubset)
						for n, childsubset in enumerate(contsets(child)))
			else: result.append(child)
		node[:] = result
	return canonicalize(tree)

removesplit = re.compile("\*[0-9]*")
def mergediscnodes(tree):
	""" Reverse of Boyd (2007): Discontinuity revisited.
	>>> tree = Tree.parse("(S (VP (VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) (VVPP 5)) (VAINF 6)) (VMFIN 3))", parse_leaf=int)
	>>> print mergediscnodes(splitdiscnodes(tree)).pprint(margin=999)
	(S (VP (VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) (VVPP 5)) (VAINF 6)) (VMFIN 3))
	>>> print mergediscnodes(splitdiscnodes(tree, markorigin=True)).pprint(margin=999)
	(S (VP (VP (PP (APPR 0) (ART 1) (NN 2)) (CARD 4) (VVPP 5)) (VAINF 6)) (VMFIN 3))
	"""
	for node in tree.subtrees():
		merge = {}
		for child in node:
			if isinstance(child, Tree) and "*" in child.node:
				merge.setdefault(removesplit.sub("", child.node),
							[]).append(child)
		node[:] = [child for child in node if not isinstance(child, Tree)
						or "*" not in child.node]
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
	collinize(cnfTree, factor="right", horzMarkov=2)
	collinize(lcnfTree, factor="left", horzMarkov=2)
	
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
	demo()
	if attempted and not fail:
		print "%s: %d doctests succeeded!" % (__file__, attempted)
	from negra import NegraCorpusReader
	correct = wrong = 0
	#n = NegraCorpusReader("../rparse", "tigerproc\.export")
	n = NegraCorpusReader(".", "sample2\.export", encoding="iso-8859-1")
	for tree in n.parsed_sents():
		if mergediscnodes(splitdiscnodes(tree)) == tree:
			correct += 1
		else:
			wrong += 1
	total = float(len(n.sents()))
	print "correct", correct, "=", 100*correct/total, "%"
	print "wrong", wrong, "=", 100*wrong/total, "%"

__all__ = ["collinize", "un_collinize", "collapse_unary"]
