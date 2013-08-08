# -*- coding: UTF-8 -*-
""" Treebank transformations:
- generic transforms listed by name
- Relational-realizational transform
- reattaching punctuation """
from __future__ import division, print_function, unicode_literals
from itertools import count, islice
from .tree import Tree, ParentedTree
#from .treebank import TAG, MORPH, FUNC

FIELDS = tuple(range(8))
WORD, LEMMA, TAG, MORPH, FUNC, PARENT, SECEDGETAG, SECEDGEPARENT = FIELDS
STATESPLIT = '^'


def transform(tree, sent, transformations):
	""" Perform some transformations, specific to Negra and WSJ treebanks.
	State-splits are preceded by '^'.
	negra:  transformations=('S-RC', 'VP-GF', 'NP', 'PUNCT')
	wsj:    transformations=('S-WH', 'VP-HD', 'S-INF')
	alpino: transformations=('PUNCT', ) """
	for name in transformations:
		# negra / tiger
		if name == 'S-RC':  # relative clause => S becomes SRC
			for s in tree.subtrees(lambda n: n.label == 'S'
					and function(n) == 'RC'):
				s.label += STATESPLIT + 'RC'
		elif name == 'NP':  # case
			for np in tree.subtrees(lambda n: n.label == 'NP'):
				np.label += STATESPLIT + function(np)
		elif name == 'PUNCT':  # distinguish . ? !
			for punct in tree.subtrees(lambda n: n.label.upper() in (
					'$.', 'PUNCT', 'LET[]')):
				punct.label += STATESPLIT + sent[punct[0]].replace(
						'(', '[').replace(')', ']').encode('unicode-escape')
		elif name == 'PP-NP':  # un-flatten PPs by introducing NPs
			addtopp = ('AC', )
			for pp in tree.subtrees(lambda n: n.label == 'PP'):
				ac = [a for a in pp if function(a) in addtopp]
				# anything before an initial preposition goes to the PP
				# (modifiers, punctuation), otherwise it goes to the NP;
				# mutatis mutandis for postpositions.
				functions = [function(x) for x in pp]
				if 'AC' in functions and 'NK' in functions:
					if functions.index('AC') < functions.index('NK'):
						ac[:0] = pp[:functions.index('AC')]
					if rindex(functions, 'AC') > rindex(functions, 'NK'):
						ac += pp[rindex(functions, 'AC') + 1:]
				#else:
				#	print('PP but no AC or NK', ' '.join(functions))
				nk = [a for a in pp if a not in ac]
				# introduce a PP unless there is already an NP in the PP
				# (annotation mistake?), or there is a PN and we want to avoid
				# a cylic unary of NP -> PN -> NP.
				if ac and nk and (len(nk) > 1
						or nk[0].label not in s('NP', 'PN')):
					pp[:] = []
					pp[:] = ac + [ParentedTree('NP', nk)]
		elif name == 'DP':  # introduce determiner phrases (DPs)
			#determiners = set('ART PDS PDAT PIS PIAT PPOSAT PRELS PRELAT '
			#	'PWS PWAT PWAV'.split())
			determiners = {'ART'}
			for np in list(tree.subtrees(lambda n: n.label == 'NP')):
				if np[0].label in determiners:
					np.label = 'DP'
					if len(np) > 1 and np[1].label != 'PN':
						np1 = np[1:]
						np[1:] = []
						np[1:] = [ParentedTree('NP', np1)]
		elif name == 'VP-GF':  # VP category split based on head
			for vp in tree.subtrees(lambda n: n.label == 'VP'):
				vp.label += STATESPLIT + function(vp)
		elif name == 'VP-FIN_NEGRA':  # introduce finite VP at S level
			# collect objects and modifiers
			# introduce new S level for discourse markers
			newlevel = 'DM'.split()
			addtovp = 'HD AC DA MO NG OA OA2 OC OG PD VO SVP'.split()

			def finitevp(s):
				""" Introduce finite VPs grouping verbs and their objects. """
				if any(x.label.startswith('V') and x.label.endswith('FIN')
						for x in s if isinstance(x, Tree)):
					vp = [a for a in s if function(a) in addtovp]
					# introduce a VP unless it would lead to a unary
					# VP -> VP production
					if len(vp) != 1 or vp[0].label != 'VP':
						s[:] = [pop(a) for a in s if function(a) not in addtovp
								] + [pop(a) for a in vp]
			toplevel_s = []
			if 'S' in labels(tree):
				toplevel_s = [a for a in tree if a.label == 'S']
				for s in toplevel_s:
					while function(s[0]) in newlevel:
						s[:] = [s[0], ParentedTree('S', s[1:])]
						s = s[1]
						toplevel_s = [s]
			elif 'CS' in labels(tree):
				cs = tree[labels(tree).index('CS')]
				toplevel_s = [a for a in cs if a.label == 'S']
			for a in toplevel_s:
				finitevp(a)
		elif name == 'POS-PART':  # introduce POS tag for particle verbs
			for a in tree.subtrees(
					lambda n: any(function(x) == 'SVP' for x in n)):
				svp = [x for x in a if function(x) == 'SVP'].pop()
				# apparently there can be a _verb_ particle without a verb.
				# headlines? annotation mistake?
				if any(map(ishead, a)):
					hd = [x for x in a if ishead(x)].pop()
					if hd.label != a.label:
						particleverb = ParentedTree(hd.label, [hd, svp])
						a[:] = [particleverb if ishead(x) else x
											for x in a if function(x) != 'SVP']
		elif name == 'SBAR':  # introduce SBAR level
			sbarfunc = 'CP'.split()
			# in the annotation, complementizers belong to the first S
			# in S conjunctions, even when they appear to apply to the whole
			# conjunction.
			for s in list(tree.subtrees(lambda n: n.label == 'S'
					and function(n[0]) in sbarfunc and len(n) > 1)):
				s.label = 'SBAR'
				s[:] = [s[0], ParentedTree('S', s[1:])]
		elif name == 'NEST':  # introduce nested structures for modifiers
			# (iterated adjunction instead of sister adjunction)
			adjunctable = set('NP'.split())  # PP AP VP
			for a in list(tree.subtrees(lambda n: n.label in adjunctable
					and any(function(x) == 'MO' for x in n))):
				modifiers = [x for x in a if function(x) == 'MO']
				if min(n for n, x in enumerate(a) if function(x) == 'MO') == 0:
					modifiers[:] = modifiers[::-1]
				while modifiers:
					modifier = modifiers.pop()
					a[:] = [ParentedTree(a.label,
							[x for x in a if x != modifier]), modifier]
					a = a[0]
		elif name == 'UNARY':  # introduce phrasal projections for single tokens
			# currently only adds NPs.
			tagtoconst, _ = getgeneralizations()
			for a in tree.treepositions('leaves'):
				tag = tree[a[:-1]]   # e.g. NN
				const = tree[a[:-2]]  # e.g. S
				if (tag.label in tagtoconst
						and const.label != tagtoconst[tag.label]):
					# NN -> NN -> word
					tag[:] = [ParentedTree(tag.label, [tag[0]])]
					# NP -> NN -> word
					tag.source = 8 * ['--']
					tag.source[TAG] = tag.label = tagtoconst[tag.label]
					tag.source[FUNC] = function(tag[0])
					#tag[0].source[FUNC] = 'NK'

		# wsj
		elif name == 'S-WH':
			for sbar in tree.subtrees(lambda n: n.label == 'SBAR'):
				for s in sbar:
					if (s.label == 'S'
							and any(a.label.startswith('WH') for a in s)):
						s.label += STATESPLIT + 'WH'
		elif name == 'VP-HD':  # VP category split based on head
			for vp in tree.subtrees(lambda n: n.label == 'VP'):
				hd = [x for x in vp if ishead(x)].pop()
				if hd.label == 'VB':
					vp.label += STATESPLIT + 'HINF'
				elif hd.label == 'TO':
					vp.label += STATESPLIT + 'HTO'
				elif hd.label in ('VBN', 'VBG'):
					vp.label += STATESPLIT + 'HPART'
		elif name == 'S-INF':
			for s in tree.subtrees(lambda n: n.label == 'S'):
				hd = [x for x in s if ishead(x)].pop()
				if hd.label in ('VP' + STATESPLIT + 'HINF',
						'VP' + STATESPLIT + 'HTO'):
					s.label += STATESPLIT + 'INF'
		elif name == 'VP-FIN_WSJ':  # add disc. finite VP when verb is under S
			for s in tree.subtrees(lambda n: n.label == 'S'):
				if not any(a.label.startswith('VP') for a in s):
					raise NotImplementedError
		# alpino?
		# ...
		else:
			raise ValueError('unrecognized transformation %r' % name)
	for a in reversed(list(tree.subtrees(lambda x: len(x) > 1))):
		a.sort(key=Tree.leaves)
	return tree


def reversetransform(tree, transformations):
	""" Undo specified transformations, as well as removing any hyphen-marked
	state splits. Do not apply twice (might remove VPs which shouldn't be). """
	tagtoconst, _ = getgeneralizations()
	# Generic state-split removal
	for node in tree.subtrees(lambda n: STATESPLIT in n.label[1:]):
		node.label = node.label[:node.label.index(STATESPLIT, 1)]

	# restore linear precedence ordering
	for a in tree.subtrees(lambda n: len(n) > 1):
		a.sort(key=lambda n: n.leaves())
	for name in reversed(transformations):
		if name == 'DP':  # remove DPs
			for dp in tree.subtrees(lambda n: n.label == 'DP'):
				dp.label = 'NP'
				if len(dp) > 1 and dp[1].label == 'NP':
					#dp1 = dp[1][:]
					#dp[1][:] = []
					#dp[1:] = dp1
					dp[1][:], dp[1:] = [], dp[1][:]
		elif name == 'NEST':  # flatten adjunctions
			nkonly = set('PDAT CAP PPOSS PPOSAT ADJA FM PRF NM NN NE PIAT '
					'PRELS PN TRUNC CH CNP PWAT PDS VP CS CARD ART PWS PPER'
					''.split())
			probably_nk = set('AP PIS'.split()) | nkonly
			for np in tree.subtrees(lambda n: len(n) == 2
					and n.label == 'NP'
					and [x.label for x in n].count('NP') == 1
					and not set(labels(n)) & probably_nk):
				np.sort(key=lambda n: n.label == 'NP')
				np[:] = np[:1] + np[1][:]
		elif name == 'PP-NP':  # flatten PPs
			for pp in tree.subtrees(lambda n: n.label == 'PP'):
				if 'NP' in labels(pp) and 'NN' not in labels(pp):
					# ensure NP is in last position
					pp.sort(key=lambda n: n.label == 'NP')
					pp[-1][:], pp[-1:] = [], pp[-1][:]
		elif name == 'SBAR':  # merge extra S level
			for sbar in list(tree.subtrees(lambda n: n.label == 'SBAR'
					or (n.label == 'S' and len(n) == 2
						and labels(n) == ['PTKANT', 'S']))):
				sbar.label = 'S'
				if sbar[0].label == 'S':
					sbar[:] = sbar[1:] + sbar[0][:]
				else:
					sbar[:] = sbar[:1] + sbar[1][:]
		elif name == 'VP-FIN_NEGRA':
			def mergevp(s):
				""" merge finite VP with S level """
				for vp in (n for n, a in enumerate(s) if a.label == 'VP'):
					if any(a.label.endswith('FIN') for a in s[vp]):
						s[vp][:], s[vp:vp + 1] = [], s[vp][:]
			#if any(a.label == 'S' for a in tree):
			#	map(mergevp, [a for a in tree if a.label == 'S'])
			#elif any(a.label == 'CS' for a in tree):
			#	map(mergevp, [s for cs in tree for s in cs if cs.label == 'CS'
			#		and s.label == 'S'])
			for s in tree.subtrees(lambda n: n.label == 'S'):
				mergevp(s)
		elif name == 'POS-PART':
			# remove constituents for particle verbs
			# get the grandfather of each verb particle
			hasparticle = lambda n: any('PTKVZ' in (x.label
					for x in m if isinstance(x, Tree)) for m in n
					if isinstance(m, Tree))
			for a in list(tree.subtrees(hasparticle)):
				for n, b in enumerate(a):
					if (len(b) == 2 and b.label.startswith('V')
						and 'PTKVZ' in (c.label for c in b
							if isinstance(c, Tree))
						and any(c.label == b.label for c in b)):
						a[n:n + 1] = b[:]
		elif name == 'UNARY':
			# remove phrasal projections for single tokens
			for a in tree.treepositions('leaves'):
				tag = tree[a[:-1]]    # NN
				const = tree[a[:-2]]  # NP
				parent = tree[a[:-3]]  # PP
				if (len(const) == 1 and tag.label in tagtoconst
						and const.label == tagtoconst[tag.label]):
					parent[a[-3]] = const.pop(0)
					del const
	# restore linear precedence ordering
	for a in tree.subtrees(lambda n: len(n) > 1):
		a.sort(key=lambda n: n.leaves())
	return tree


def getgeneralizations():
	""" Some tree transforms for Negra/Tiger.  """
	# generalizations suggested by SyntaxGeneralizer of TigerAPI
	# however, instead of renaming, we introduce unary productions
	# POS tags
	tonp = 'NN NNE PNC PRF PDS PIS PPER PPOS PRELS PWS'.split()
	#topp = 'PROAV PWAV'.split()  # ??
	#toap = 'ADJA PDAT PIAT PPOSAT PRELAT PWAT PRELS'.split()
	#toavp = 'ADJD ADV'.split()

	tagtoconst = {}
	for label in tonp:
		tagtoconst[label] = 'NP'
	#for label in toap:
	#	tagtoconst[label] = 'AP'
	#for label in toavp:
	#	tagtoconst[label] = 'AVP'

	# phrasal categories
	tonp = 'CNP NM PN'.split()
	#topp = 'CPP'.split()
	#toap = 'MTA CAP'.split()
	#toavp = 'AA CAVP'.split()
	unaryconst = {}
	for label in tonp:
		unaryconst[label] = 'NP'
	return tagtoconst, unaryconst


def unifymorphfeat(feats, percolatefeatures=None):
	""" Treat a sequence of strings as feature vectors, either
	comma or dot separated, and produce the sorted union of their features.

	:param percolatefeatures: if a set is given, select only these features;
		by default all features are used.

	>>> print(unifymorphfeat({'Def.*.*', '*.Sg.*', '*.*.Akk'}))
	Akk.Def.Sg
	>>> print(unifymorphfeat({'LID[bep,stan,rest]', 'N[soort,ev,zijd,stan]'}))
	bep,ev,rest,soort,stan,zijd """
	sep = '.' if any('.' in a for a in feats) else ','
	result = set()
	for a in feats:
		if '[' in a:
			a = a[a.index('[') + 1:a.index(']')]
		result.update(a.split(sep))
	if percolatefeatures:
		result.intersection_update(percolatefeatures)
	return sep.join(sorted(result - {'*', '--'}))


def rrtransform(tree, morphlevels=0, percolatefeatures=None,
		adjunctionlabel=None, ignorefunctions=None, ignorecategories=None):
	""" Relational-realizational tree transformation.
	Every constituent node is expanded to three levels:

	1) syntactic category, e.g., S
	2) unordered functional argument structure of children, e.g., S/<SBJ,HD,OBJ>
	3) for each child:
		grammatical function + parent syntactic category, e.g., OBJ/S

	Example::

		(NP-SBJ (NN-HD ...)) => (NP (<HD>/NP (HD/NP (NN ...))))

	:param adjunctionlabel: a grammatical function label identifying
		adjunctions. They will not be part of argument structures, and their
		grammatical function will be replaced with their neighboring
		non-adjunctive functions.
	:param ignorefunctions: function labels that do not go into argument
		structure, but keep their function in their realization to make
		backtransform possible.
	:param morphlevels: if nonzero, percolate morphological features this many
		levels upwards. For a given node, the union of the features of its
		children are collected, and the result is appended to its syntactic
		category.
	:param percolatefeatures: if a sequence is given, percolate only these
		morphological features; by default all features are used.
	:returns: a new, transformed tree. """
	def realize(child, prevfunc, nextfunc):
		""" Generate realization of a child node by recursion. """
		newchild, morph, lvl = rrtransform(child, morphlevels,
				percolatefeatures, adjunctionlabel, ignorefunctions,
				ignorecategories)
		result = tree.__class__('%s/%s' % (('%s:%s' % (prevfunc, nextfunc)
				if child.source[FUNC] == adjunctionlabel
				else child.source[FUNC]), tree.label), [newchild])
		return result, morph, lvl

	if not isinstance(tree[0], Tree):
		morph = tree.source[MORPH].replace('(', '[').replace(')', ']')
		preterminal = tree.__class__('%s/%s' % (tree.label, morph), tree)
		if morphlevels:
			return preterminal, morph, morphlevels
		return preterminal, None, 0
	# for each node, collect the functions of closest non-adjunctive sibling
	childfuncsl = (prevfunc, ) = ['']
	for child in tree:
		if (isinstance(child, Tree) and child.source[FUNC]
				and child.source[FUNC] != adjunctionlabel
				and child.source[FUNC] not in ignorefunctions
				and child.label not in ignorecategories):
			prevfunc = child.source[FUNC]
		childfuncsl.append(prevfunc)
	childfuncsr = (nextfunc, ) = ['']
	for child in reversed(tree[1:]):
		if (isinstance(child, Tree) and child.source[FUNC]
				and child.source[FUNC] != adjunctionlabel
				and child.source[FUNC] not in ignorefunctions
				and child.label not in ignorecategories):
			nextfunc = child.source[FUNC]
		childfuncsr.insert(0, prevfunc)
	funcstr = ','.join(sorted(child.source[FUNC] for child in tree
			if isinstance(child, Tree) and child.source[FUNC]
					and child.source[FUNC] != adjunctionlabel
					and child.source[FUNC] not in ignorefunctions
					and child.label not in ignorecategories))
	children, feats, levels = [], [], []
	for child, prevfunc, nextfunc in zip(tree, childfuncsl, childfuncsr):
		newchild, morph, lvl = realize(child, prevfunc, nextfunc)
		children.append(newchild)
		if morph and lvl:
			feats.append(morph)
			levels.append(lvl)
	morph, lvl = None, 0
	if feats and max(levels) and tree.label != 'ROOT':
		morph, lvl = unifymorphfeat(feats, percolatefeatures), max(levels) - 1
	configuration = tree.__class__('%s/<%s>' % (tree.label, funcstr),
			children)
	projection = tree.__class__(('%s-%s' % (tree.label, morph)) if morph
			else tree.label, [configuration])
	return projection, morph, lvl


def rrbacktransform(tree, adjunctionlabel=None, func=None):
	""" Reverse the relational-realizational transformation, conserving
	grammatical functions.

	:param adjunctionlabel: used to assign a grammatical function to
		adjunctions that have been converted to contextual labels 'next:prev'.
	:param func: used internally to percolate functional labels.
	:returns: a new tree. """
	morph = None
	if not isinstance(tree[0], Tree):
		tag, morph = tree.label.split('/')
		result = tree.__class__(tag, tree)
	elif '/' not in tree[0].label:
		result = tree.__class__(tree.label,
				[rrbacktransform(child, adjunctionlabel) for child in tree])
	else:
		result = tree.__class__(tree.label.split('-')[0],
				[rrbacktransform(
						child[0],
						adjunctionlabel,
						child.label.split('/')[0])
					for child in tree[0]])
	result.source = ['--'] * 8
	result.source[TAG] = result.label
	if morph:
		result.source[MORPH] = morph.replace('[', '(').replace(']', ')')
	if func and adjunctionlabel and ':' in func:
		result.source[FUNC] = adjunctionlabel
	elif func:
		result.source[FUNC] = func
	return result


# relative frequencies of punctuation in Negra: (is XY an annotation error?)
#1/1             $,      ,
#14793/17269     $.      .
#8598/13345      $[      "
#1557/13345      $[      )
#1557/13345      $[      (
#1843/17269      $.      :
#232/2669        $[      -
#343/13345       $[      /
#276/17269       $.      ?
#249/17269       $.      ;
#89/13345        $[      '
#101/17269       $.      !
#2/513           XY      :
#41/13345        $[      ...
#1/513           XY      -
#1/2467          $.      ·      # NB this is not a period but a \cdot ...

PUNCTUATION = frozenset(',."()&:-/!!!??;\'```....[]|\xc2\xab\xc2\xbb\\'
		) | {'&bullet;'}


def ispunct(word, tag):
	""" Test whether a word and/or tag is punctuation. """
	# fixme: treebank specific parameters for detecting punctuation.
	return tag in ('$,', '$.', '$[', '$(',) or word in PUNCTUATION


def punctremove(tree, sent):
	""" Remove any punctuation nodes, and any empty ancestors. """
	for a in reversed(tree.treepositions('leaves')):
		if ispunct(sent[tree[a]], tree[a[:-1]].label):
			# remove this punctuation node and any empty ancestors
			for n in range(1, len(a)):
				del tree[a[:-n]]
				if tree[a[:-(n + 1)]]:
					break
	# renumber
	oldleaves = sorted(tree.leaves())
	newleaves = {a: n for n, a in enumerate(oldleaves)}
	for a in tree.treepositions('leaves'):
		tree[a] = newleaves[tree[a]]
	assert sorted(tree.leaves()) == list(range(len(tree.leaves()))), tree


def punctroot(tree, sent):
	""" Move punctuation directly under the ROOT node, as in the
	original Negra/Tiger treebanks. """
	punct = []
	for a in reversed(tree.treepositions('leaves')):
		if ispunct(sent[tree[a]], tree[a[:-1]].label):
			# store punctuation node
			punct.append(tree[a[:-1]])
			# remove this punctuation node and any empty ancestors
			for n in range(1, len(a)):
				del tree[a[:-n]]
				if tree[a[:-(n + 1)]]:
					break
	tree.extend(punct)


def punctlower(tree, sent):
	""" Find suitable constituent for punctuation marks and add it there;
	removal at previous location is up to the caller.  Based on rparse code.
	Initial candidate is the root node."""
	def lower(node, candidate):
		""" Lower a specific instance of punctuation in tree,
		recursing top-down on suitable candidates. """
		num = node.leaves()[0]
		for i, child in enumerate(sorted(candidate, key=lambda x: x.leaves())):
			if not isinstance(child[0], Tree):
				continue
			termdom = child.leaves()
			if num < min(termdom):
				print('moving', node, 'under', candidate.label)
				candidate.insert(i + 1, node)
				break
			elif num < max(termdom):
				lower(node, child)
				break

	for a in tree.treepositions('leaves'):
		if ispunct(sent[tree[a]], tree[a[:-1]].label):
			b = tree[a[:-1]]
			del tree[a[:-1]]
			lower(b, tree)


def punctraise(tree, sent):
	""" Trees in the Negra corpus have punctuation attached to the root;
	i.e., it is not part of the phrase-structure.  This function attaches
	punctuation nodes (that is, a POS tag with punctuation terminal) to an
	appropriate constituent. """
	#punct = [node for node in tree.subtrees() if isinstance(node[0], int)
	punct = [node for node in tree if isinstance(node[0], int)
			and ispunct(sent[node[0]], node.label)]
	while punct:
		node = punct.pop()
		# dedicated punctation node (??)
		if all(isinstance(a[0], int) and ispunct(sent[a[0]], a)
				for a in node.parent):
			continue
		node.parent.pop(node.parent_index)
		phrasalnode = lambda x: len(x) and isinstance(x[0], Tree)
		for candidate in tree.subtrees(phrasalnode):
			# add punctuation mark next to biggest constituent which it borders
			#if any(node[0] - 1 in borders(sorted(a.leaves())) for a in candidate):
			#if any(node[0] - 1 == max(a.leaves()) for a in candidate):
			if any(node[0] + 1 == min(a.leaves()) for a in candidate):
				candidate.append(node)
				break
		else:
			tree.append(node)

BALANCEDPUNCTMATCH = {'"': '"', '[': ']', '(': ')', '-': '-', "'": "'",
		'\xc2\xab': '\xc2\xbb'}  # the last ones are unicode for << and >>.


def balancedpunctraise(tree, sent):
	""" Move balanced punctuation marks " ' - ( ) [ ] together in the same
	constituent. Based on rparse code.

	>>> tree = ParentedTree.parse('(ROOT ($, 3) ($[ 7) ($[ 13) ($, 14) ($, 20)'
	... ' (S (NP (ART 0) (ADJA 1) (NN 2) (NP (CARD 4) (NN 5) (PP (APPR 6) '
	... '(CNP (NN 8) (ADV 9) (ISU ($. 10) ($. 11) ($. 12))))) (S (PRELS 15) '
	... '(MPN (NE 16) (NE 17)) (ADJD 18) (VVFIN 19))) (VVFIN 21) (ADV 22) '
	... '(NP (ADJA 23) (NN 24))) ($. 25))', parse_leaf=int)
	>>> sent = ("Die zweite Konzertreihe , sechs Abende mit ' Orgel plus "
	... ". . . ' , die Hayko Siemens musikalisch leitet , bietet wieder "
	... "ungewoehnliche Kombinationen .".split())
	>>> punctraise(tree, sent)
	>>> balancedpunctraise(tree, sent)
	>>> from .treetransforms import addbitsets, fanout
	>>> max(map(fanout, addbitsets(tree).subtrees()))
	1
	>>> nopunct = Tree.parse('(ROOT (S (NP (ART 0) (ADJA 1) (NN 2) (NP (CARD 3)'
	... ' (NN 4) (PP (APPR 5) (CNP (NN 6) (ADV 7)))) (S (PRELS 8) (MPN (NE 9) '
	... '(NE 10)) (ADJD 11) (VVFIN 12))) (VVFIN 13) (ADV 14) (NP (ADJA 15) '
	... '(NN 16))))', parse_leaf=int)
	>>> max(map(fanout, addbitsets(nopunct).subtrees()))
	1
	"""
	assert isinstance(tree, ParentedTree)
	# right punct str as key, mapped to left index as value
	punctmap = {}
	# punctuation indices mapped to preterminal nodes
	termparent = {a[0]: a for a in tree.subtrees()
			if a and isinstance(a[0], int) and ispunct(sent[a[0]], a.label)}
	for terminal in sorted(termparent):
		preterminal = termparent[terminal]
		# do we know the matching punctuation mark for this one?
		if sent[terminal] in punctmap:
			right = terminal
			left = punctmap[sent[right]]
			rightparent = preterminal.parent
			leftparent = termparent[left].parent
			if max(leftparent.leaves()) == right - 1:
				node = termparent[right]
				leftparent.append(node.parent.pop(node.parent_index))
			elif min(rightparent.leaves()) == left + 1:
				node = termparent[left]
				rightparent.insert(0, node.parent.pop(node.parent_index))
			if sent[right] in punctmap:
				del punctmap[sent[right]]
		elif sent[terminal] in BALANCEDPUNCTMATCH:
			punctmap[BALANCEDPUNCTMATCH[sent[terminal]]] = terminal


def function(tree):
	""" Return grammatical function for node, or an empty string. """
	if hasattr(tree, 'source'):
		return tree.source[FUNC].split('-')[0]
	else:
		return ''


def ishead(tree):
	""" Test whether this node is the head of the parent constituent. """
	if hasattr(tree, 'source'):
		return 'HD' in tree.source[FUNC].upper().split('-')
	else:
		return False


def rindex(l, v):
	""" Like list.index(), but go from right to left. """
	return len(l) - 1 - l[::-1].index(v)


def labels(tree):
	""" Return the labels of the children of this node. """
	return [a.label for a in tree if isinstance(a, Tree)]


def pop(a):
	""" Remove this node from its parent node, if it has one.
	Convenience function for ParentedTrees."""
	try:
		return a.parent.pop(a.parent_index)
	except AttributeError:
		return a


def bracketings(tree):
	""" Labelled bracketings of a tree. """
	return [(a.label, tuple(sorted(a.leaves())))
		for a in tree.subtrees(lambda t: t and isinstance(t[0], Tree))]


def testpunct():
	""" Verify that punctuation movement does not increase fan-out. """
	from .treetransforms import addbitsets, fanout
	from .treebank import NegraCorpusReader
	filename = 'sample2.export'  # 'negraproc.export'
	mangledtrees = NegraCorpusReader('.', filename, headrules=None,
			encoding='iso-8859-1', punct='move')
	nopunct = list(NegraCorpusReader('.', filename, headrules=None,
			encoding='iso-8859-1', punct='remove').parsed_sents().values())
	originals = list(NegraCorpusReader('.', filename, headrules=None,
			encoding='iso-8859-1').parsed_sents().values())
	phrasal = lambda x: len(x) and isinstance(x[0], Tree)
	for n, mangled, sent, nopunct, original in zip(count(),
			mangledtrees.parsed_sents().values(), mangledtrees.sents().values(),
			nopunct, originals):
		print(n, end='')
		for a, b in zip(sorted(addbitsets(mangled).subtrees(phrasal),
				key=lambda n: min(n.leaves())),
				sorted(addbitsets(nopunct).subtrees(phrasal),
				key=lambda n: min(n.leaves()))):
			if fanout(a) != fanout(b):
				print(' '.join(sent))
				print(mangled)
				print(nopunct)
				print(original)
			assert fanout(a) == fanout(b), '%d %d\n%s\n%s' % (
				fanout(a), fanout(b), a, b)
	print()


def testtransforms():
	""" Test whether the Tiger transformations (transform / reversetransform)
	are reversible. """
	from .treetransforms import canonicalize
	from .treebank import NegraCorpusReader, handlefunctions
	headrules = None  # 'negra.headrules'
	n = NegraCorpusReader('.', 'sample2.export', encoding='iso-8859-1',
			headrules=headrules)
	nn = NegraCorpusReader('.', 'sample2.export', encoding='iso-8859-1',
			headrules=headrules)
	transformations = ('S-RC', 'VP-GF', 'NP')
	trees = [transform(tree, sent, transformations)
			for tree, sent in zip(nn.parsed_sents().values(),
				nn.sents().values())]
	print('\ntransformed')
	correct = exact = d = 0
	for a, b, c in islice(zip(n.parsed_sents().values(),
			trees, n.sents().values()), 100):
		transformb = reversetransform(b.copy(True), transformations)
		b1 = bracketings(canonicalize(a))
		b2 = bracketings(canonicalize(transformb))
		z = -1  # 825
		if b1 != b2 or d == z:
			precision = len(set(b1) & set(b2)) / len(set(b1))
			recall = len(set(b1) & set(b2)) / len(set(b2))
			if precision != 1.0 or recall != 1.0 or d == z:
				print(d, ' '.join(':'.join((str(n),
					a.encode('unicode-escape'))) for n, a in enumerate(c)))
				print('no match', precision, recall)
				print(len(b1), len(b2), 'gold-transformed', set(b2) - set(b1),
						'transformed-gold', set(b1) - set(b2))
				print(a)
				print(transformb)
				handlefunctions('add', a)
				print(a)
				print(b)
				print()
			else:
				correct += 1
		else:
			exact += 1
			correct += 1
		d += 1
	print('matches', correct, '/', d, 100 * correct / d, '%')
	print('exact', exact)

if __name__ == '__main__':
	main()


def main():
	""" Just tests. """
	testpunct()
	testtransforms()

if __name__ == '__main__':
	main()
