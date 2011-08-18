""" Implementation of Sangati et al., Efficiently extract recurring tree
fragments from large treebanks. """
from collections import defaultdict
from grammar import cartpi
from itertools import count
from nltk import Tree, ImmutableTree
from grammar import srcg_productions, canonicalize,\
					alpha_normalize, rangeheads

#def dtrees(trees, sents):
#	results = []
#	for tree, sent in zip(trees, sents):
#		results.append(make_DTree(tree, sent))
#	return results
#
#def make_DTree(tree, sent):
#	vec = sum(1 << n for n in tree.leaves())
#	if all(lambda x: isinstance(x, Tree)) and 1 <= len(tree) <= 2:
#		return new_DTree(tree.prod, vec, False,
#				make_DTree(tree[0]),
#				make_DTree(tree[1]) if len(tree) == 2 else None)
#	elif len(tree) == 1:
#		return new_DTree(tree.prod, vec, True, None, None)
#	else: raise ValueError
		
def extractfragments(trees, sents):
	""" Seeks the largest fragments occurring at least twice in the
	corpus.  """
	fraglist = defaultdict(set)
	trees = [indices(add_srcg_rules(canonicalize(aa).freeze(), bb))
				for aa, bb in zip(trees, sents)]
	lentrees = len(trees)
	for n in range(lentrees):
		a = trees[n]; asent = sents[n]
		for m in range(n + 1, lentrees):
			b = trees[m]; bsent = sents[m]
			for x in extractfrom(a, trees[m], asent, sents[m]):
				if x: fraglist[x].update((n, m))
	return dict([(aa, len(bb)) for aa, bb in fraglist.iteritems()])

def extractfrom(a, b, asent, bsent):
	""" Return the set of fragments that a and b have in common.
	A fragment is a connected subset of nodes where each node either
	has zero children, or as much as in the original tree.""" 
	mem = {}; l = set()
	for i in a:
		for j in b:
			try: x = frozenset(mem[i, j])
			except KeyError:
				x = frozenset(extractmaxfragments(a, b, i, j,
									asent, bsent, mem))
			if len(x) < 2 or x in l: continue
			# disjoint-set datastructure here?
			for y in l:
				if x < y: break
				elif x > y:
					l.remove(y)
					l.add(x)
					break
			else: l.add(x)
	return set([fragmentfromindices(a, asent, x) for x in l])

def extractmaxfragments(a, b, i, j, asent, bsent, mem):
	""" Get common fragments starting from positions i and j. """
	# compare label & arity / terminal
	if type(a[i]) != type(b[j]):
		mem[i, j] = set()
		return set()
	elif isinstance(a[i], Tree):
		# assume presence of arity markers
		if a[i].node != b[j].node:
			mem[i, j] = set()
			return set()
	elif asent[a[i]] != bsent[b[j]]:
		mem[i, j] = set()
		return set()

	nodeset = set([i])
	if not isinstance(a[i], Tree):
		mem[i, j] = nodeset
		return nodeset
	if a[i].prod == b[j].prod:
		for n in range(len(a[i])):
			ii,  jj = i+(n,), j+(n,)
			nodeset.update(mem[ii, jj] if (ii, jj) in mem
					else extractmaxfragments(a, b, ii, jj, asent, bsent, mem))
	mem[i, j] = nodeset
	return nodeset

#def extractmaxpartialfragments(a, b):
#	""" partial fragments allow difference in number of children
#	not tested. """
#	if not same((a,b)): return set()
#	mappingsset = maxsetmappings(a,b,0,0,True)
#	if not mappingsset: return [ImmutableTree(a.node, [])]
#	partfragset = set()
#	for mapping in mappingsset:
#		maxpartialfragmentpairs = [set() for x in mapping]
#		for i, (n1, n2) in enumerate(mapping):
#			maxpartialfragmentpairs[i] = extractmaxpartialfragments(n1, n2)
#		for nodeset in cartpi(maxpartialfragmentpairs):
#			nodeset.union(a)
#			partfragset.add(nodeset)
#	return partfragset
#
#def maxsetmappings(a,b,x,y,firstCall=False):
#	mappings = []
#	startx = x if firstCall else x + 1
#	starty = y if firstCall else y + 1
#	endx = len(a) - 1
#	endy = len(b) - 1
#	startxexists = startx < len(a)
#	startyexists = starty < len(b)
#	while startxexists or startyexists:
#		if startxexists:
#			for celly in range(endy, starty + 1, -1):
#				if a[startx] == b[celly]:
#					endy = celly
#					submappings = maxsetmappings(a,b,startx,celly)
#					if not firstCall:
#						for mapping in submappings:
#							mapping.add((a[startx],b[celly]))
#					mappings.extend(submappings)
#		if startyexists:
#			for cellx in range(endx, startx + 1, -1):
#				if a[startx] == b[starty]:
#					endx = cellx
#					submappings = maxsetmappings(a,b,cellx,starty)
#					if not firstCall:
#						for mapping in submappings:
#							mapping.add((a[cellx],b[starty]))
#					mappings.extend(submappings)
#		if startxexists and startyexists and a[startx] == b[starty]:
#			submappings = maxsetmappings(a,b,startx,starty,False)
#			if not firstCall:
#				for mapping in submappings:
#					mapping.add((a[startx],b[starty]))
#			mappings.extend(submappings)
#			break
#		if startx+1 <= endx: startx += 1
#		else: startxexists = False
#		if starty+1 <= endy: starty += 1
#		else: startyexists = False
#	return set(mappings)
#
#def same(a, b, asent, bsent):
#	if type(a) != type(b): return False
#	elif isinstance(a, Tree): return a.node == b.node
#	else: return asent[a] == bsent[b]

def fragmentfromindices(tree, sent, indices):
	""" Given a tree and a set of connected indices, return the fragment
	described by these indices.
	>>> np = Tree('NP', [Tree('DET', [2]), Tree('NN', [3])])
	>>> tree = Tree('S', [Tree('NP', [0]), Tree('VP', [Tree('V', [1]), np])])
	>>> sent = "Mary sees a unicorn".split()
	>>> indices = set([(1,), (1, 0), (1, 0, 0), (1, 1)])
	>>> print "%s, %s" % fragmentfromindices(tree, sent, indices)
	(VP (V 0) (NP 1)), ('sees', None)
	"""
	if not indices: return
	froot = min(indices, key=len)
	if not isinstance(tree[froot], Tree): return
	result = Tree(tree[froot].node, [])
	sent = list(sent[:])
	agenda = [(result, a, froot+(n,)) for n,a in enumerate(tree[froot])]
	while agenda:
		current, node, idx = agenda.pop(0)
		if idx in indices:
			if isinstance(node, Tree):
				child = Tree(node.node, [])
				current.append(child)
				new = [(child, a, idx+(n,)) for n,a in enumerate(node)
							if idx+(n,) in indices]
				if new: agenda += new
				else:
					child[:] = rangeheads(sorted(node.leaves()))
					for a in child: sent[a] = None
			else:
				current.append(node)
	# frontier nodes and leaves should both receive index
	# with frontier nodes pointing to None in sent
	# discontinous frontier nodes get multiple indices (ranges)
	# frontier nodes should get rangeheads(a.leaves())
	# renumber
	leaves = result.leaves()
	sent = tuple([a for n,a in enumerate(sent) if n in leaves])
	leafmap = dict(zip(sorted(leaves), count()))
	for n, a in enumerate(result.treepositions('leaves')):
		result[a] = leafmap[result[a]]
	return (result.freeze(), sent) if len(result) else None

def add_srcg_rules(tree, sent):
	for a, b in zip(tree.subtrees(), srcg_productions(tree, sent, False)):
		a.prod = alpha_normalize(zip(*b))
	return tree

def indices(tree):
	return dict((idx, tree[idx]) for idx in tree.treepositions())

def main():
	from negra import NegraCorpusReader
	import sys, codecs
	# this fixes utf-8 output when piped through e.g. less
	# won't help if the locale is not actually utf-8, of course
	sys.stdout = codecs.getwriter('utf8')(sys.stdout)
	corpus = NegraCorpusReader(".", "sample2\.export", encoding="iso-8859-1", headorder=True, headfinal=True, headreverse=False)
	#corpus = NegraCorpusReader("../rparse", "tigerproc\.export", headorder=True, headfinal=True, headreverse=False)
	treebank = """(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (JJ hungry) (NN dog))))
(S (NP (DT The) (NN cat)) (VP (VBP saw) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN mouse)) (VP (VBP saw) (NP (DT the) (JJ yellow) (NN cat))))
(S (NP (DT The) (JJ little) (NN mouse)) (VP (VBP saw) (NP (DT the) (NN cat))))
(S (NP (DT The) (NN cat)) (VP (VBP ate) (NP (DT the) (NN dog))))
(S (NP (DT The) (NN mouse)) (VP (VBP ate) (NP (DT the) (NN cat))))"""
	treebank = map(Tree, treebank.splitlines())
	sents = [tree.leaves() for tree in treebank]
	for tree, sent in zip(treebank, sents):
		for n, idx in enumerate(tree.treepositions('leaves')): tree[idx] = n
	for (a,aa),b in sorted(extractfragments(treebank, sents).items()):
		print a.pprint(margin=999), aa,b
	print
	trees = list(corpus.parsed_sents()[:10])
	sents = corpus.sents()[:100]
	for tree in trees:
		for idx in tree.treepositions('leaves'): tree[idx] = int(tree[idx])
	print "fragments",
	for (a,aa),b in sorted(extractfragments(trees, sents).items()):
		print a.pprint(margin=999),aa,b
	print

if __name__ == '__main__': main()
