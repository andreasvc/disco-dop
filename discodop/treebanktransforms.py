# -*- coding: UTF-8 -*-
"""Treebank transformations.

- generic transforms listed by name
- Relational-realizational transform
- reattaching punctuation"""
from __future__ import division, print_function
import re
from itertools import islice, repeat
from collections import defaultdict, Counter as multiset
from discodop.tree import Tree, ParentedTree

FIELDS = tuple(range(8))
WORD, LEMMA, TAG, MORPH, FUNC, PARENT, SECEDGETAG, SECEDGEPARENT = FIELDS
STATESPLIT = '^'
LABELRE = re.compile("[^^|<>-]+")
HEADRULERE = re.compile(r'^(\w+)\s+(LEFT-TO-RIGHT|RIGHT-TO-LEFT)(?:\s+(.*))?$')
NEGRATAGUNARY = dict(zip(
		'NN NNE PNC PRF PDS PIS PPER PPOS PRELS PWS'.split(), repeat('NP')))
NEGRACONSTUNARY = dict(zip('CNP NM PN'.split(), repeat('NP')))
NUMBERRE = re.compile('^[0-9]+(?:[,.][0-9]+)*$')
DERE = re.compile("^([Dd]es?|du|d')$")
PPORNP = re.compile('^(NP|PP)+PP$')


def transform(tree, sent, transformations):
	"""Perform specified sequence of transformations on a tree.

	State-splits are preceded by '^'. ``transformations`` is a sequence of
	transformation names that will be performed on the given tree (in-place).
	Presets for particular treebanks:

	:negra:  ``transformations=('S-RC', 'VP-GF', 'NP', 'PUNCT')``
	:wsj:    ``transformations=('S-WH', 'VP-HD', 'S-INF')``
	:alpino: ``transformations=('PUNCT', )``
	:ftb:
			``transformations=('markinf markpart de2 markp1 mwadvs mwadvsel1 '
			'mwadvsel2 mwnsel1 mwnsel2 PUNCT tagpa').split()``"""
	for name in transformations:
		if name == 'APPEND-FUNC':  # add function to phrasal label
			for a in tree.subtrees():
				func = function(a)
				if func and func != '--':
					a.label += '-' + func
		elif name == 'FUNC-NODE':  # insert node w/function above phrasal label
			from discodop.treetransforms import postorder
			for a in postorder(tree):
				func = function(a)
				if func and func != '--':
					a[:] = [a.__class__(a.label,
							[a.pop() for _ in range(len(a))][::-1])]
					a.label = '-' + func
		elif name == 'MORPH-NODE':  # insert node w/morph. features above POS
			from discodop.treetransforms import postorder
			for a in postorder(tree, lambda n: n and isinstance(n[0], int)):
				morph = '--'
				if getattr(a, 'source', None):
					morph = a.source[MORPH].replace('(', '[').replace(')', ']')
				a[:] = [a.__class__(a.label,
						[a.pop() for _ in range(len(a))][::-1])]
				a.label = morph
		elif name == 'LEMMA-NODE':  # insert node w/lemma above terminal
			from discodop.treetransforms import postorder
			from discodop.treebank import quote
			for a in postorder(tree, lambda n: n and isinstance(n[0], int)):
				lemma = '--'
				if getattr(a, 'source', None):
					lemma = quote(a.source[LEMMA])
				a[:] = [a.__class__(lemma,
						[a.pop() for _ in range(len(a))][::-1])]
		elif name == 'NP-PP':  # mark PPs under NPs
			for pp in tree.subtrees(lambda n: n.label == 'PP'
					and n.parent.label == 'NP'):
				pp.label += STATESPLIT + 'NP'
		elif name == 'FOLD-NUMBERS':
			sent[:] = ['000' if NUMBERRE.match(a) else a for a in sent]
		elif name == 'PUNCT':  # distinguish sentence-ending punctuation.
			for punct in tree.subtrees(lambda n: isinstance(n[0], int)
					and sent[n[0]] in '.?!'):
				punct.label += STATESPLIT + sent[punct[0]]
		elif (negratransforms(name, tree, sent)
				or wsjtransforms(name, tree, sent)
				or ftbtransforms(name, tree, sent)):
			pass
		else:
			raise ValueError('unrecognized transformation %r' % name)
	for a in reversed(list(tree.subtrees(lambda x: len(x) > 1))):
		a.sort(key=Tree.leaves)
	return tree


def negratransforms(name, tree, _sent):
	"""Negra / Tiger transforms."""
	if name == 'S-RC':  # relative clause => S becomes SRC
		for s in tree.subtrees(lambda n: n.label == 'S'
				and function(n) == 'RC'):
			s.label += STATESPLIT + 'RC'
	elif name == 'NP':  # case
		for np in tree.subtrees(lambda n: n.label == 'NP'):
			np.label += STATESPLIT + function(np)
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
			"""Introduce finite VPs grouping verbs and their objects."""
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
		for a in tree.treepositions('leaves'):
			tag = tree[a[:-1]]   # e.g. NN
			const = tree[a[:-2]]  # e.g. S
			if (tag.label in NEGRATAGUNARY
					and const.label != NEGRATAGUNARY[tag.label]):
				# NN -> NN -> word
				tag[:] = [ParentedTree(tag.label, [tag[0]])]
				# NP -> NN -> word
				tag.source = 8 * ['--']
				tag.source[TAG] = tag.label = NEGRATAGUNARY[tag.label]
				tag.source[FUNC] = function(tag[0])
				#tag[0].source[FUNC] = 'NK'
	else:
		return False
	return True


def wsjtransforms(name, tree, _sent):
	"""Transforms for WSJ section of Penn treebank."""
	if name == 'S-WH':
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
	elif name == 'MARK-UNARY':  # add -U to unary nodes to avoid cycles
		for unary in tree.subtrees(lambda n: len(n) == 1
				and isinstance(n[0], Tree)):
			unary.label += STATESPLIT + 'U'
	else:
		return False
	return True


def ftbtransforms(name, tree, sent):
	"""Port of manual FTB enrichments specified in Stanford parser.

	cf. ``FrenchTreebankParserParams.java``"""
	if name == 'markinf':
		for t in tree.subtrees(lambda n: strip(n.label) == "V"
				and isinstance(n.parent, Tree)
				and isinstance(n.parent.parent, Tree)
				and strip(n.parent.label) == "VN"
				and strip(n.parent.parent.label) == "VPinf"):
			t.label += "^infinitive"
	elif name == 'markpart':
		for t in tree.subtrees(lambda n: strip(n.label) == "V"
				and isinstance(n.parent, Tree)
				and isinstance(n.parent.parent, Tree)
				and strip(n.parent.label) == "VN"
				and strip(n.parent.parent.label) == "VPpart"):
			t.label += "^participle"
	elif name == 'markvn':
		for t in tree.subtrees(lambda n: strip(n.label) == "VN"):
			for sub in islice(t.subtrees(), 1, None):
				sub.label += "^withVN"
	elif name == 'tagpa':  # Add parent annotation to POS tags
		for t in tree.subtrees(lambda n:
				not isinstance(n[0], Tree)
				and strip(n.label) != "PUNC"):
			t.label += "^" + t.parent.label
	elif name == 'coord1':
		for t in tree.subtrees(lambda n: strip(n.label) == 'COORD'
				and len(n) >= 2):
			t.label += "^" + t[1].label
	elif name == 'de2':
		for t in tree.subtrees(lambda n: strip(n.label) == 'P'
				and DERE.match(sent[n[0]])):
			t.label += "^de2"
	elif name == 'de3':
		# @NP|PP|COORD >+(@NP|PP) (@PP <, (@P < /^([Dd]es?|du|d')$/))
		for t in tree.subtrees(lambda n:
				strip(n.label) in ("PP", "COORD")):
			a = list(ancestors(t))
			for n in range(2, len(a)):
				if PPORNP.match("".join(strip(x.label) for x in a[:n])):
					if (strip(a[n - 1][0].label) == "P"
							and DERE.match(sent[a[n - 1][0][0]])):
						t.label += "^de3"
						break
	elif name == 'markp1':
		for t in tree.subtrees(lambda n: strip(n.label) == "P"
				and strip(n.parent.label) == "PP"
				and strip(n.parent.parent.label) == "NP"):
			t.label += "^n"
	elif name == 'mwadvs':
		for t in tree.subtrees(lambda n: strip(n.label) == "MWADV"
				and "S" in n.parent.label):
			t.label += "^mwadv-s"
	elif name == 'mwadvsel1':
		for t in tree.subtrees(lambda n: strip(n.label) == "MWADV"
				and len(n) == 2
				and strip(n[0].label) == "P"
				and strip(n[1].label) == "N"):
			t.label += "^mwadv1"
	elif name == 'mwadvsel2':
		for t in tree.subtrees(lambda n: strip(n.label) == "MWADV"
				and len(n) == 3
				and strip(n[0].label) == "P"
				and strip(n[1].label) == "D"
				and strip(n[2].label) == "N"):
			t.label += "^mwadv2"
	elif name == 'mwnsel1':
		for t in tree.subtrees(lambda n: strip(n.label) == "MWN"
				and len(n) == 2
				and strip(n[0].label) == "N"
				and strip(n[1].label) == "A"):
			t.label += "^mwn1"
	elif name == 'mwnsel2':
		for t in tree.subtrees(lambda n: strip(n.label) == "MWN"
				and len(n) == 3
				and strip(n[0].label) == "N"
				and strip(n[1].label) == "P"
				and strip(n[2].label) == "N"):
			t.label += "^mwn2"
	elif name == 'mwnsel3':  # noun-noun compound joined with dash.
		for t in tree.subtrees(lambda n: strip(n.label) == "MWN"
				and len(n) == 3
				and strip(n[0].label) == "N"
				and sent[n[1][0]] == "-"
				and strip(n[2].label) == "N"):
			t.label += "^mwn3"
	else:
		return False
	return True


def reversetransform(tree, transformations):
	"""Undo specified transformations and remove state splits marked by ``^``.

	Do not apply twice (might remove VPs which shouldn't be)."""
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
				"""Merge finite VP with S level."""
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
				if (len(const) == 1 and tag.label in NEGRATAGUNARY
						and const.label == NEGRATAGUNARY[tag.label]):
					parent[a[-3]] = const.pop(0)
					del const
		elif name == 'APPEND-FUNC':  # functions appended to phrasal labels
			for a in tree.subtrees():
				if '-' in a.label:
					label, func = a.label.split('-', 1)
					a.source = ['--'] * 8
					a.source[TAG] = a.label = label
					a.source[FUNC] = func
		elif name == 'FUNC-NODE':  # nodes with function above phrasal labels
			from discodop.treetransforms import postorder
			for a in postorder(tree, lambda n: n.label.startswith('-')):
				a.source = ['--'] * 8
				a.source[FUNC] = a.label[1:]
				a.source[TAG] = a.label = a[0].label
				a[:] = [a[0].pop() for _ in range(len(a[0]))][::-1]
		elif name == 'MORPH-NODE':  # nodes with morph. above preterminals
			from discodop.treetransforms import postorder
			for a in postorder(tree, lambda n: n and isinstance(n[0], Tree)
					and n[0] and isinstance(n[0][0], int)):
				a.source = ['--'] * 8
				a.source[MORPH] = a.label
				a.source[TAG] = a.label = a[0].label
				a[:] = [a[0].pop() for _ in range(len(a[0]))][::-1]
		elif name == 'LEMMA-NODE':  # nodes with lemmas above words
			from discodop.treetransforms import postorder
			from discodop.treebank import unquote
			for a in postorder(tree, lambda n: n and isinstance(n[0], Tree)
					and n[0] and isinstance(n[0][0], int)):
				a.source = ['--'] * 8
				a.source[LEMMA] = unquote(a[0].label)
				a.source[TAG] = a.label
				a[:] = [a[0].pop() for _ in range(len(a[0]))][::-1]
	# restore linear precedence ordering
	for a in tree.subtrees(lambda n: len(n) > 1):
		a.sort(key=lambda n: n.leaves())
	return tree


def collapselabels(trees, _sents, mapping=None):
	"""Collapse non-root phrasal labels with specified mapping.

	The mapping is of the form::

		{coarselabel1: {finelabel1, finelabel2, ...}, ...}

	For example following Charniak et al. (2006),
	multi-level coarse-to-fine parsing:
	:level 0: single label P
	:level 1: HP, MP (arguments, modifiers)
	:level 2: S, N, A, P (verbal, nominal, adjectival, prepositional)
	:level 3: no-op, return original treebank labels"""
	def collapse(orig):
		"""Collapse labels of a single tree; returns a new Tree object."""
		tree = orig.copy(True)
		for subtree in tree.subtrees():
			if subtree.label != "ROOT" and isinstance(subtree[0], Tree):
				subtree.label = LABELRE.sub(revmapping.get, subtree.label)

	revmapping = {finelabel: coarselabel for coarselabel in mapping
			for finelabel in mapping[coarselabel]}
	return [collapse(tree) for tree in trees]


def unifymorphfeat(feats, percolatefeatures=None):
	"""Get the sorted union of features for a sequence of feature vectors.

	:param feats: a sequence of strings of comma/dot separated feature vectors.
	:param percolatefeatures: if a set is given, select only these features;
		by default all features are used.

	>>> print(unifymorphfeat({'Def.*.*', '*.Sg.*', '*.*.Akk'}))
	Akk.Def.Sg
	>>> print(unifymorphfeat({'LID[bep,stan,rest]', 'N[soort,ev,zijd,stan]'}))
	bep,ev,rest,soort,stan,zijd"""
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
		adjunctionlabel=None, ignorefunctions=None, ignorecategories=None,
		adjleft=True, adjright=True):
	"""Relational-realizational tree transformation.

	Every constituent node is expanded to three levels:

	1) syntactic category, e.g., S
	2) unordered functional argument structure of children, e.g., S/<SBJ,HD,OBJ>
	3) for each child:
		grammatical function + parent syntactic category, e.g., OBJ/S

	Example::

		(NP-SBJ (NN-HD ...)) => (NP (<HD>/NP (HD/NP (NN ...))))

	:param adjunctionlabel: a grammatical function label identifying
			adjunctions. They will not be part of argument structures, and
			their grammatical function will be replaced with their neighboring
			non-adjunctive functions.
	:param adjleft, adjright: whether to include the left and right sibling,
			respectively, when replacing the function label for
			``adjunctionlabel``.
	:param ignorefunctions: function labels that do not go into argument
			structure, but keep their function in their realization to make
			backtransform possible.
	:param morphlevels: if nonzero, percolate morphological features this many
			levels upwards. For a given node, the union of the features of its
			children are collected, and the result is appended to its syntactic
			category.
	:param percolatefeatures: if a sequence is given, percolate only these
			morphological features; by default all features are used.
	:returns: a new, transformed tree."""
	def realize(child, prevfunc, nextfunc):
		"""Generate realization of a child node by recursion."""
		newchild, morph, lvl = rrtransform(child, morphlevels,
				percolatefeatures, adjunctionlabel, ignorefunctions,
				ignorecategories, adjleft, adjright)
		result = tree.__class__('%s/%s' % (('%s:%s' % (prevfunc, nextfunc)
				if child.source[FUNC] == adjunctionlabel
				and (prevfunc or nextfunc)
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
		newchild, morph, lvl = realize(child,
				prevfunc if adjleft else '',
				nextfunc if adjright else '')
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
	"""Reverse relational-realizational transformation.

	:param adjunctionlabel: used to assign a grammatical function to
		adjunctions that have been converted to contextual labels 'next:prev'.
	:param func: used internally to percolate functional labels.
	:returns: a new tree."""
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


def removeterminals(tree, sent, func):
	"""Remove any empty nodes, and any empty ancestors."""
	for a in reversed(tree.treepositions('leaves')):
		if func(sent[tree[a]], tree[a[:-1]].label):
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


def removeemptynodes(tree, sent):
	"""Remove any empty nodes, and any empty ancestors."""
	removeterminals(tree, sent, lambda x, _: x in (None, '', '-NONE-'))


# fixme: treebank specific parameters for detecting punctuation.
PUNCTTAGS = {"''", "``", "-LRB-", "-RRB-", ".", ":", ",",  # PTB
		'$,', '$.', '$[', '$(',  # Negra/Tiger
		'let', 'LET[]', 'SPEC[symb]', 'TW[hoofd,vrij]'}  # Alpino/Lassy
PUNCTUATION = frozenset(u'.,():\'-";?/!*&`[]<>{}|=\'\xc2\xab\xc2\xbb\xb7\xad\\'
		) | {u'&bullet;', u'..', u'...', u'....', u'.....', u'......',
		u'!!', u'!!!', u'??', u'???', u"''", u'``', u',,',
		u'--', u'---', u'----', u'-LRB-', u'-RRB-', u'-LCB-', u'-RCB-'}


def ispunct(word, tag):
	"""Test whether a word and/or tag is punctuation."""
	return tag in PUNCTTAGS or word in PUNCTUATION


def punctremove(tree, sent):
	"""Remove any punctuation nodes, and any empty ancestors."""
	removeterminals(tree, sent, ispunct)


def punctroot(tree, sent):
	"""Move punctuation directly under ROOT, as in the Negra annotation."""
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
	"""Find suitable constituent for punctuation marks and add it there.

	Initial candidate is the root node. Note that ``punctraise()`` performs
	better. Based on rparse code."""
	def lower(node, candidate):
		"""Lower a specific instance of punctuation in tree.

		Recurses top-down on suitable candidates."""
		num = node.leaves()[0]
		for i, child in enumerate(sorted(candidate, key=lambda x: x.leaves())):
			if not isinstance(child[0], Tree):
				continue
			termdom = child.leaves()
			if num < min(termdom):
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


def punctraise(tree, sent, rootpreterms=False):
	"""Attach punctuation nodes to an appropriate constituent.

	Trees in the Negra corpus have punctuation attached to the root;
	i.e., it is not part of the phrase-structure. This function moves the
	punctuation to an appropriate level in the tree. A punctuation node is a
	POS tag with a punctuation terminal. Modifies trees in-place.

	:param rootpreterms: if True, move all preterminals under root,
		instead of only recognized punctuation."""
	#punct = [node for node in tree.subtrees() if isinstance(node[0], int)
	punct = [node for node in tree if isinstance(node[0], int)
			and (rootpreterms or ispunct(sent[node[0]], node.label))]
	while punct:
		node = punct.pop()
		while node is not tree and len(node.parent) == 1:
			node = node.parent
		if node is tree:
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
	"""Move balanced punctuation ``" ' - ( ) [ ]`` to a common constituent.

	Based on rparse code."""
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
	""":returns: grammatical function for node, or an empty string."""
	if getattr(tree, 'source', None):
		return tree.source[FUNC].split('-')[0]
	return ''


def ishead(tree):
	"""Test whether this node is the head of the parent constituent."""
	if getattr(tree, 'source', None):
		return 'HD' in tree.source[FUNC].upper().split('-')
	return False


def rindex(l, v):
	"""Like list.index(), but go from right to left."""
	return len(l) - 1 - l[::-1].index(v)


def labels(tree):
	""":returns: the labels of the children of this node."""
	return [a.label for a in tree if isinstance(a, Tree)]


def pop(a):
	"""Remove this node from its parent node, if it has one.

	Convenience function for ParentedTrees."""
	try:
		return a.parent.pop(a.parent_index)
	except AttributeError:
		return a


def strip(label):
	"""Equivalent to the effect of the @ operator in tregex."""
	return label[:label.index("^")] if "^" in label else label


def ancestors(node):
	"""Yield ancestors of node from direct parent to root node."""
	while node:
		node = node.parent
		yield node


def bracketings(tree):
	"""Labeled bracketings of a tree."""
	return [(a.label, tuple(sorted(a.leaves())))
		for a in tree.subtrees(lambda t: t and isinstance(t[0], Tree))]


def readheadrules(filename):
	"""Read a file containing heuristic rules for head assignment.

	Example line: ``s right-to-left vmfin vafin vaimp``, which means
	traverse siblings of an S constituent from right to left, the first child
	with a label of vmfin, vafin, or vaimp will be marked as head."""
	headrules = {}
	for line in open(filename):
		line = line.strip().upper()
		if line and not line.startswith("%") and len(line.split()) > 2:
			label, lr, heads = HEADRULERE.match(line).groups()
			if heads is None:
				heads = ''
			headrules.setdefault(label, []).append((lr, heads.split()))
	return headrules


def headfinder(tree, headrules, headlabels=frozenset({'HD'})):
	"""Use head finding rules to select one child of tree as head."""
	candidates = [a for a in tree if getattr(a, 'source', None)
			and headlabels.intersection(a.source[FUNC].upper().split('-'))]
	if candidates:
		return candidates[0]
	children = tree
	for lr, heads in headrules.get(tree.label, []):
		if lr == 'LEFT-TO-RIGHT':
			children = tree
		elif lr == 'RIGHT-TO-LEFT':
			children = tree[::-1]
		else:
			raise ValueError
		for head in heads:
			for child in children:
				if (isinstance(child, Tree)
						and child.label.split('[')[0] == head):
					return child
	# default head is initial/last nonterminal (depending on direction lr)
	for child in children:
		if isinstance(child, Tree):
			return child


def sethead(child):
	"""Mark node as head in an auxiliary field."""
	child.source = getattr(child, "source")
	if child.source is None:
		child.source = 6 * ['']
		child.source[TAG] = child.label
	if 'HD' not in child.source[FUNC].upper().split("-"):
		x = list(child.source)
		if child.source[FUNC] in (None, '', '--'):
			x[FUNC] = '-HD'
		else:
			x[FUNC] = x[FUNC] + '-HD'
		child.source = tuple(x)


def headmark(tree):
	"""Add marker to label of head node."""
	head = [a for a in tree if getattr(a, 'source', None)
			and 'HD' in a.source[FUNC].upper().split('-')]
	if not head:
		return
	head[-1].label += '-HD'


def headorder(tree, headfinal, reverse):
	"""Order constituents based on head (identified with function tag)."""
	head = [n for n, a in enumerate(tree)
		if getattr(a, 'source', None)
		and 'HD' in a.source[FUNC].upper().split("-")]
	if not head:
		return
	headidx = head.pop()
	# everything until the head is reversed and prepended to the rest,
	# leaving the head as the first element
	nodes = tree[:]
	tree[:] = []
	if headfinal:
		if reverse:  # head final, reverse rhs: A B C^ D E => A B E D C^
			tree[:] = nodes[:headidx] + nodes[headidx:][::-1]
		else:  # head final, reverse lhs:  A B C^ D E => E D A B C^
			tree[:] = nodes[headidx + 1:][::-1] + nodes[:headidx + 1]
	else:
		if reverse:  # head first, reverse lhs: A B C^ D E => C^ B A D E
			tree[:] = nodes[:headidx + 1][::-1] + nodes[headidx + 1:]
		else:  # head first, reverse rhs: A B C^ D E => C^ D E B A
			tree[:] = nodes[headidx:] + nodes[:headidx][::-1]


def saveheads(tree, tailmarker):
	"""Store head as grammatical function when inferrable from binarization."""
	if not tailmarker:
		return
	for node in tree.subtrees(lambda n: tailmarker in n.label):
		node.source = ['--'] * 6
		node.source[FUNC] = 'HD'


def headstats(trees):
	"""Collect some information useful for writing headrules.

	- ``heads['NP']['NN'] ==`` number of times NN occurs as head of NP.
	- ``pos1['NP'][1] ==`` number of times head of NP is at position 1.
	- ``pos2`` is like pos1, but position is from the right.
	- ``unknown['NP']['NN'] ==`` number of times NP that does not have a head
		dominates an NN."""
	heads, unknown = defaultdict(multiset), defaultdict(multiset)
	pos1, pos2 = defaultdict(multiset), defaultdict(multiset)
	for tree in trees:
		for a in tree.subtrees(lambda x: len(x) > 1):
			for n, b in enumerate(a):
				if 'hd' in b.source[FUNC].lower():
					heads[a.label][b.label] += 1
					pos1[a.label][n] += 1
					pos2[a.label][len(a) - (n + 2)] += 1
					break
			else:
				unknown[a.label].update(b.label for b in a)
	return heads, unknown, pos1, pos2
