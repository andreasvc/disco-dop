# -*- coding: UTF-8 -*-
"""Treebank transformations.

- Transforms (primarily state splits) listed by name
- Relational-realizational transform
"""
from __future__ import division, print_function, absolute_import, \
		unicode_literals
import re
from itertools import islice
from .tree import Tree, ParentedTree, escape, unescape, ptbescape
from .treebank import EXPORTNONTERMINAL
from .treetransforms import addfanoutmarkers, removefanoutmarkers
from .punctuation import punctprune, PUNCTUATION
from .util import ishead

FIELDS = tuple(range(6))
WORD, LEMMA, TAG, MORPH, FUNC, PARENT = FIELDS
STATESPLIT = '^'
LABELRE = re.compile("[^^|<>,;:_-]+")
CASERE = re.compile(r'\b(Nom|Acc|Gen|Dat)\b')
DERE = re.compile("^([Dd]es?|du|d')$")
PPORNP = re.compile('^(NP|PP)+PP$')
YEARRE = re.compile('^(?:19|20)[0-9]{2}$')
PRESETS = {
		# basic state splits, German, English, Dutch:
		'negra': ('S-RC', 'VP-GF', 'NP', 'PUNCT'),
		'wsj': ('PTBbrackets', 'S-WH', 'VP-HD', 'S-INF'),
		'alpino': ('PUNCT', ),
		# extensive state splits following particular papers:
		# French
		'green2013ftb': ('markinf,markpart,de2,markp1,mwadvs,mwadvsel1,'
			'mwadvsel2,mwnsel1,mwnsel2,PUNCT,TAGPA').split(','),
		# English
		# These are the "-goodPCFG" options of the Stanford Parser
		'km2003wsj': ('PTBbrackets,splitIN4,splitPercent,splitPoss,splitCC,'
			'unaryDT,unaryRB,splitAux2,splitVP3,splitSGapped,splitTMP,'
			'splitBaseNP,dominatesV,splitNPADV,markDitransV,MARK-YEAR'
			).split(','),
		# a simpler variant mentioned in Bansal & Klein 2010
		'km2003simple': ('PTBbrackets,splitIN4,splitPercent,splitPoss,splitCC,'
			'unaryDT,unaryRB,splitAux2,splitSGapped,splitBaseNP,dominatesV,'
			'splitNPADV,markDitransV,MARK-YEAR').split(','),
		# German
		'fraser2013tiger': ('elimNKCJ,addUnary,APPEND-FUNC,addCase,lexPrep,'
			'PUNCT,adjAttach,relPath,whFeat,nounSeq,properChunks,markAP,'
			'subConjType,VPfeat,noHead,noSubj,MARK-YEAR').split(','),
		# Dutch
		'lassy': ('nladdunary,nlelimcnj,nlselectmorph,PUNCT,'
			'MARK-YEAR,nlpercolatemorph,nlmwuhead').split(','),
		# this variant adds function tags to non-terminal labels
		'lassy-func': ('nladdunary,nlelimcnj,APPEND-FUNC,nlselectmorph,PUNCT,'
			'MARK-YEAR,nlpercolatemorph,nlmwuhead').split(','),
		}

# Mappings for multi-level coarse-to-fine parsing
# following Charniak et al. (2006), multi-level coarse-to-fine parsing.
# http://aclweb.org/anthology/N06-1022
MAPPINGS = {
		'ptb': {
			# level 0: P (all phrase labels)
			0: {'P': {'S', 'VP', 'UCP', 'SQ', 'SBAR', 'SBARQ', 'SINV',
					'NP', 'NAC', 'NX', 'LST', 'X', 'FRAG', 'PRT|ADVP',
					'ADJP', 'QP', 'CONJP', 'ADVP', 'INTJ', 'PRN', 'PRT',
					'PP', 'RRC', 'WHADJP', 'WHADVP', 'WHNP', 'WHPP'}},
			# level 1: HP (arguments), MP (modifiers)
			1: {'HP': {'S', 'VP', 'UCP', 'SQ', 'SBAR', 'SBARQ', 'SINV',
					'NP', 'NAC', 'NX', 'LST', 'X', 'FRAG'},
				'MP': {'ADJP', 'QP', 'CONJP', 'ADVP', 'INTJ', 'PRN', 'PRT',
					'PRT|ADVP', 'PP', 'RRC', 'WHADJP', 'WHADVP', 'WHNP',
					'WHPP'}},
			# level 2: S (verbal), N (nominal), A (adjectival),
			# 	P (prepositional)
			# note: PRT is part of both A_ and P_ in the paper;
			# UCP is part of both S_ and N_
			2: {'S_': {'S', 'VP', 'SQ', 'SBAR', 'SBARQ', 'SINV'},
				'N_': {'NP', 'NAC', 'NX', 'LST', 'X', 'UCP', 'FRAG'},
				'A_': {'ADJP', 'QP', 'CONJP', 'ADVP', 'INTJ', 'PRN', 'PRT',
					'PRT|ADVP'},
				'P_': {'PP', 'RRC', 'WHADJP', 'WHADVP', 'WHNP', 'WHPP'}},
			# level 3: no-op, return original treebank labels
		},
		'negra': {
			# level 0: P (all phrase labels)
			0: {'P': {'--', 'AA', 'AP', 'AVP', 'CAC', 'CAP', 'CAVP', 'CCP',
				'CH', 'CNP', 'CO', 'CPP', 'CS', 'CVP', 'CVZ', 'DL', 'ISU',
				'MPN', 'MTA', 'NM', 'NP', 'PN', 'PP', 'QL', 'S', 'VP', 'VZ'}},
			# level 1: HP (arguments), MP (modifiers)
			1: {'HP': {'NP', 'S', 'VP', 'VZ', 'CO', 'AA', 'CNP', 'CS', 'CVP',
					'CVZ', 'PN', 'MPN', 'NM', 'CH', 'CCP', 'DL', 'ISU', 'QL'},
				'MP': {'--', 'AP', 'PP', 'AVP', 'CAP', 'CPP', 'CAVP', 'CAC',
					'MTA'}},
			# level 2: S (verbal), N (nominal), A (adjectival),
			# 	P (prepositional)
			2: {'S_': {'S', 'VP', 'VZ', 'CO', 'AA', 'CS', 'CVP',
					'CVZ', 'CCP', 'DL', 'ISU', 'QL'},
				'N_': {'NP', 'CNP', 'PN', 'MPN', 'NM', 'CH'},
				'A_': {'--', 'AP', 'AVP', 'CAP', 'CAVP', 'MTA'}},
				'P_': {'PP', 'CPP', 'CAC'}
			# level 3: no-op, return original treebank labels
		},
		'alpino': {
			# level 0: P (all phrase labels)
			0: {'P': {'ADVP', 'AHI', 'AP', 'CONJ', 'CP', 'DETP', 'DU', 'INF',
					'MWU', 'NP', 'OTI', 'PP', 'PPART', 'PPRES', 'REL', 'SMAIN',
					'SSUB', 'SV1', 'SVAN', 'TI', 'WHQ', 'WHREL', 'WHSUB'}},
			# level 1: HP (arguments), MP (modifiers)
			1: {'HP': {'AHI', 'CONJ', 'CP', 'DETP', 'DU', 'INF', 'MWU', 'NP',
					'OTI', 'PPART', 'PPRES', 'REL', 'SMAIN', 'SSUB', 'SVAN',
					'SV1', 'TI', 'WHSUB', 'WHQ'},
				'MP': {'AP', 'ADVP', 'PP', 'REL', 'WHREL'}},
			# level 2: S (verbal), N (nominal), A (adjectival),
			# 	P (prepositional)
			2: {'S_': {'AHI', 'CP', 'DU', 'INF', 'OTI', 'PPART', 'PPRES',
					'SMAIN', 'SSUB', 'SVAN', 'SV1', 'TI', 'WHSUB', 'WHQ'},
				'N_': {'CONJ', 'DETP', 'MWU', 'NP'},
				'A_': {'AP', 'ADVP', 'REL', 'WHREL'},
				'P_': {'PP'}},
			# level 3: no-op, return original treebank labels
		},
	}


def expandpresets(transformations):
	"""Expand aliases for presets."""
	return [a for name in transformations
			for a in PRESETS.get(name, [name])]


def transform(tree, sent, transformations):
	"""Perform specified sequence of transformations on a tree.

	State-splits are preceded by '^'. ``transformations`` is a sequence of
	transformation names (order matters) that will be performed on the given
	tree (in-place). There are presets for particular treebanks. The name of a
	preset can be used as an alias that expands to a sequence of
	transformations; see the variable ``PRESETS``."""
	# unfreeze attributes so that they can be modified
	for a in tree.subtrees(lambda n: isinstance(
			getattr(n, 'source', None), tuple)):
		a.source = list(a.source)
	for name in transformations:
		if name == 'APPEND-FUNC':  # add function to phrasal label
			for a in tree.subtrees():
				func = functions(a)
				if func and not a.label.endswith('-'):  # -LRB-
					a.label += '-' + '-'.join(func)
		elif name == 'FUNC-NODE':  # insert node w/function above phrasal label
			for a in tree.postorder():
				func = functions(a)
				if func and not a.label.endswith('-'):  # -LRB-
					a[:] = [a.__class__(a.label,
							[a.pop() for _ in range(len(a))][::-1])]
					a.label = '-' + '-'.join(func)
		elif name == 'APPEND-MORPH':  # Append morph. features to POS tag
			for a in tree.subtree(lambda n: n and isinstance(n[0], int)):
				morph = '--'
				if getattr(a, 'source', None):
					morph = a.source[MORPH].replace('(', '[').replace(')', ']')
				a.label += STATESPLIT + morph
		elif name == 'MORPH-NODE':  # insert node w/morph. features above POS
			for a in tree.postorder(lambda n: n and isinstance(n[0], int)):
				morph = '--'
				if getattr(a, 'source', None):
					morph = a.source[MORPH].replace('(', '[').replace(')', ']')
				a[:] = [a.__class__(morph,
						[a.pop() for _ in range(len(a))][::-1])]
		elif name == 'LEMMA-NODE':  # insert node w/lemma above terminal
			for a in tree.postorder(lambda n: n and isinstance(n[0], int)):
				lemma = '--'
				if getattr(a, 'source', None):
					lemma = escape(a.source[LEMMA])
				a[:] = [a.__class__(lemma,
						[a.pop() for _ in range(len(a))][::-1])]
		elif name == 'MARK-YEAR':  # mark POS label of year terminals
			for node in tree.subtrees(lambda n: n and isinstance(n[0], int)
					and YEARRE.match(sent[n[0]])):
				node.label += STATESPLIT + 'year'
		elif name == 'PUNCT':  # distinguish sentence-ending punctuation.
			for punct in tree.subtrees(lambda n: n and isinstance(n[0], int)
					and sent[n[0]] in '.?!'):
				punct.label += STATESPLIT + sent[punct[0]]
		elif name == 'PUNCT-PRUNE':  # remove initial/ending quotes & period
			punctprune(tree, sent)
		elif name == 'FANOUT':  # add fan-out markers
			addfanoutmarkers(tree)
		elif name == 'PARENT':  # add one level of parent annotation
			# Useful to do here to add the parent annotations before
			# adding any other annotations to the labels.
			# Skips preterminals.
			for node in islice(tree.subtrees(
					lambda n: n and isinstance(n[0], Tree)), 1, None):
				node.label += STATESPLIT + strip(node.parent.label)
		elif name == 'TAGPA':  # Add parent annotation to non-punct. POS tags
			for node in tree.subtrees(lambda n:
					n and isinstance(n[0], int)
					and sent[n[0]] not in PUNCTUATION):
				node.label += STATESPLIT + strip(node.parent.label)
		elif name == 'NP-PP':  # mark PPs under NPs
			for pp in tree.subtrees(lambda n: n.label == 'PP'
					and n.parent.label == 'NP'):
				pp.label += STATESPLIT + 'NP'
		elif (negratransforms(name, tree, sent)
				or lassytransforms(name, tree, sent)
				or ptbtransforms(name, tree, sent)
				or ftbtransforms(name, tree, sent)):
			pass
		else:
			raise ValueError('unrecognized transformation %r' % name)
	for a in reversed(list(tree.subtrees(lambda x: len(x) > 1))):
		a.children.sort(key=Tree.leaves)
	return tree


def negratransforms(name, tree, sent):
	"""Negra / Tiger transforms."""
	if name == 'S-RC':  # relative clause => S becomes S^RC
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
			funcs = [function(x) for x in pp]
			if 'AC' in funcs and 'NK' in funcs:
				if funcs.index('AC') < funcs.index('NK'):
					ac[:0] = pp[:funcs.index('AC')]
				if rindex(funcs, 'AC') > rindex(funcs, 'NK'):
					ac += pp[rindex(funcs, 'AC') + 1:]
			# else:
			# 	print('PP but no AC or NK', ' '.join(funcs))
			nk = [a for a in pp if a not in ac]
			# introduce a PP unless there is already an NP in the PP
			# (annotation mistake?), or there is a PN and we want to avoid
			# a cylic unary of NP -> PN -> NP.
			if ac and nk and (len(nk) > 1
					or nk[0].label not in s('NP', 'PN')):
				pp[:] = []
				pp[:] = ac + [ParentedTree('NP', nk)]
	elif name == 'DP':  # introduce determiner phrases (DPs)
		# determiners = {'ART', 'PDS', 'PDAT', 'PIS', 'PIAT', 'PPOSAT',
		# 	'PRELS', 'PRELAT', 'PWS', 'PWAT', 'PWAV'}
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
		adjunctable = {'NP'}  # PP AP VP
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
	# The following transformations as described in Fraser et al (CL, 2013)
	# http://aclweb.org/anthology/J13-1005
	elif name == 'addUnary':  # introduce unary NPs
		maxid = getmaxid(tree)
		for node in tree.postorder(lambda n: strip(n.label) in
					{'NN', 'PPER', 'PDS', 'PIS', 'PRELS', 'CARD', 'PN'}
					and strip(n.parent.label) in {'S', 'VP', 'ROOT', 'DL'}):
			if node.label == 'PN' and len(node) == 1:  # only complex PNs
				continue
			children = node[:]
			node[:] = []
			tag = ParentedTree(node.label, children)
			tag.source = node.source[:]
			node[:] = [tag]
			node.source[TAG] = node.label = 'NP'
			node.source[FUNC] = tag.source[FUNC]
			maxid += 1
			node.source[WORD] = '#%d' % maxid
			node.source[LEMMA] = node.source[MORPH] = '--'
			tag.source[FUNC] = 'HD'
	elif name == 'addCase':  # add case features to POS tags
		for node in tree.subtrees(lambda n: n and isinstance(n[0], int)):
			case = CASERE.match(node.source[MORPH])
			if case:
				node.label += '/' + case.group(1)
	elif name == 'elimNKCJ':  # eliminate NK and CJ functions
		for node in tree.subtrees(lambda n: function(n) in {'NK', 'CJ'}):
			if function(node) == 'NK':
				node.source[FUNC] = 'HD'
			else:  # elif function(node) == 'CJ':
				node.source[FUNC] = function(node.parent) or '--'
	elif name == 'lexPrep':  # lexicalize frequent prepositions/conjunctions
		for node in tree.subtrees(lambda n: n and isinstance(n[0], int)):
			word = sent[node[0]].lower()
			if base(node, 'APPR') and word in {
					'in', 'von', 'auf', 'durch', 'um',
					'unter', 'unters', 'unterm'}:
				node.label += STATESPLIT + word
			elif base(node, 'KON') and function(node) == 'CD' and (word in {
					'sowohl', 'als', 'weder', 'entweder', 'noch'}
					or (word == 'oder'  # 'oder' if preceded by entweder
						and any(a.lower() == 'entweder'
							for a in sent[:node[0]]))):
				node.label += STATESPLIT + word
	elif name == 'adjAttach':  # annotate attachments of adjuncts
		for node in tree.subtrees(
				lambda n: strip(n.label) in {'PP', 'AVP', 'ADV', 'ADJD'}):
			if strip(node.parent.label) in {'S', 'VP'}:
				annot = 'V'
			elif strip(node.parent.label) in {'NP', 'PP'}:
				annot = 'N'
			else:
				annot = '0'
			node.label += STATESPLIT + annot
			if strip(node.label) == 'AVP':  # propagate to head child
				for child in node:
					if ishead(child):
						child.label += STATESPLIT + annot
						break
	elif name == 'relPath':  # mark path from relative clause to rel. pronoun
		for node in tree.subtrees(lambda n: base(n, 'S')
				and function(n) == 'RC'):
			for child in node.subtrees(lambda n: strip(n.label) in
					{'PRELS', 'PRELAT', 'PWAV', 'PWS'}):
				child = child.parent
				while child is not node:
					child.label += STATESPLIT + 'rel'
					child = child.parent
				node.label += STATESPLIT + 'rel'
				break
			else:  # no rel. pronoun found
				node.label += STATESPLIT + 'norel'
	elif name == 'whFeat':  # mark NP/PP that immediately dominates WH-pronoun
		for node in tree.subtrees(lambda n: strip(n.label) in {'NP', 'PP'} and
				any(strip(a.label) in {'PWAT', 'PWS', 'PWAV'} for a in n)):
			node.label += STATESPLIT + 'wh'
	elif name == 'nounSeq':  # consecutive nouns in NP
		for node in tree.subtrees(lambda n: base(n, 'NP')):
			for n, a in enumerate(node[:-1]):
				if base(a, 'NN') and base(node[n + 1], 'NN'):
					a.label += STATESPLIT + 'seq'
					break
	elif name == 'properChunks':  # mark POS tags in proper noun chunks
		for node in tree.subtrees(lambda n: base(n, 'NP')
				and function(n) == 'PNC'):
			for tag in node:
				tag.label += STATESPLIT + 'name'
	elif name == 'markAP':  # mark predicative APs, APs with nominal head
		for node in tree.subtrees(lambda n: base(n, 'AP')):
			if any(base(a, 'ADJD') for a in node.subtrees()):
				node.label += STATESPLIT + 'pred'
			if any(ishead(child) and strip(child.label) in {'NN', 'NP'}
					for child in node):
				node.label += STATESPLIT + 'nom'
	elif name == 'subConjType':  # mark type of subordinating conj.
		for node in tree.subtrees(lambda n: base(n, 'S')
				and function(n) in {'SB', 'OC', 'MO', 'RE'}):
			for child in node:
				if base(child, 'KOUS'):
					child.label += STATESPLIT + function(node)
					break
	elif name == 'VPfeat':  # mark object VPs with head label
		for node in tree.subtrees(lambda n: base(n, 'VP')
				and function(n) == 'OC'):
			for child in node:
				if ishead(child):
					node.label += STATESPLIT + strip(child.label)
					break
	elif name == 'noHead':  # constituents without head child
		for node in tree.subtrees(lambda n: n is not tree
				and n and isinstance(n[0], Tree)):
			# The heuristically found heads do not count.
			if not any(function(child) in {'HD', 'PNC', 'AC', 'AVC', 'NMC',
					'PH', 'PD', 'ADC', 'UC', 'DH'}
					for child in node):
				node.label += STATESPLIT + 'nohead'
	elif name == 'noSubj':  # conjunct clauses without subject
		for node in tree.subtrees(lambda n: n and isinstance(n[0], Tree)
				and base(n, 'S') and function(n) == 'CJ'):
			if not any(function(child) in {'SB', 'EP'}
					or strip(child.label) in {'VVIMP', 'VAIMP'}
					for child in node):
				node.label += STATESPLIT + 'nosubj'
	else:
		return False
	return True


def ptbtransforms(name, tree, sent):
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
		# this counters the flattening when a VP is not possible because of
		# non-standard word order; e.g. is John happy
		for s in tree.postorder(lambda n: n.label == 'S'):
			if not any(a.label.startswith('VP') for a in s):
				vp = ParentedTree('VP', [])
				for child in list(s):
					# FIXME: check which functions should not go in the VP
					# (pre)modifiers unclear.
					if 'SBJ' not in functions(child):
						vp.append(s.pop(child))
				s.append(vp)
	elif name == 'MARK-UNARY':  # add -U to unary nodes to avoid cycles
		for unary in tree.subtrees(lambda n: len(n) == 1
				and isinstance(n[0], Tree)):
			unary.label += STATESPLIT + 'U'
	# The following transformations are translations of
	# the Stanford Parser state splits described in
	# Accurate Unlexicalized Parsing (ACL 2003).
	# http://aclweb.org/anthology/P03-1054
	elif name == 'splitIN':  # Stanford Parser splitIN=3
		for node in tree.subtrees(lambda n: base(n, 'IN')):
			if base(node.parent.parent, 'N') and (
					base(node.parent, 'P') or
					base(node.parent, 'A')):
				node.label += STATESPLIT + 'N'
			elif base(node.parent, 'Q') and (
					base(node.parent.parent, 'N') or
					base(node.parent.parent, 'ADJP')):
				node.label += STATESPLIT + 'Q'
			elif base(node.parent.parent, 'S'):
				if base(node.parent, 'SBAR'):
					node.label += STATESPLIT + 'SCC'
				else:
					node.label += STATESPLIT + 'SC'
			elif base(node.parent, 'SBAR') or base(
					node.parent, 'WHNP'):
				node.label += STATESPLIT + 'T'
	elif name == 'splitIN4':  # Stanford Parser splitIN=4
		for node in tree.subtrees(lambda n: base(n, 'IN')):
			if base(node.parent.parent, 'N') and (
					base(node.parent, 'P') or
					base(node.parent, 'A')):
				node.label += STATESPLIT + 'N'
			elif base(node.parent, 'Q') and (
					base(node.parent.parent, 'N') or
					base(node.parent.parent, 'ADJP')):
				node.label += STATESPLIT + 'Q'
			elif node.parent.parent.label[0] == 'S' and not base(
					node.parent.parent, 'SBAR'):
				if base(node.parent, 'SBAR'):
					node.label += STATESPLIT + 'SCC'
				elif not base(node.parent, 'NP') and not base(
						node.parent, 'ADJP'):
					node.label += STATESPLIT + 'SC'
			elif base(node.parent, 'SBAR') or base(
					node.parent, 'WHNP') or base(node.parent, 'WHADVP'):
				node.label += STATESPLIT + 'T'
	elif name == 'splitPercent':  # Stanford Parser splitPercent=1
		for node in tree.subtrees(lambda n: n and isinstance(n[0], int)
				and sent[n[0]] == '%'):
			node.label += STATESPLIT + r'%'
	elif name == 'splitPoss':  # Stanford Parser splitPoss=1
		for node in tree.subtrees(lambda n: base(n, 'NP')
				and n[-1].label.startswith('POS')):
			node.label += STATESPLIT + 'P'
	elif name == 'splitCC':  # Stanford Parser splitCC=2
		for node in tree.subtrees(lambda n: base(n, 'CC')):
			if sent[node[0]].lower() == 'but':
				node.label += STATESPLIT + 'B'
			elif sent[node[0]] == '&':
				node.label += STATESPLIT + 'A'
	elif name == 'unaryDT':  # Stanford Parser unaryDT=true
		for node in tree.subtrees(lambda n: base(n, 'DT')
				and len(n.parent) == 1):
			node.label += STATESPLIT + 'U'
	elif name == 'unaryRB':  # Stanford Parser unaryRB=true
		for node in tree.subtrees(lambda n: base(n, 'RB')
				and len(n.parent) == 1):
			node.label += STATESPLIT + 'U'
	elif name == 'splitAux':  # Stanford Parser splitAux=1
		for node in tree.subtrees(lambda n: strip(n.label)
				in {'VBZ', 'VBP', 'VBD', 'VBN', 'VBG', 'VB'}):
			if sent[node[0]].lower() in {
					'is', 'am', 'are', 'was', 'were', "'m", "'re", "'s",
					'being', 'be', 'been'}:
				node.label += STATESPLIT + 'BE'
			elif sent[node[0]].lower() in {
					'have', "'ve", 'having', 'has', 'had', "'d"}:
				node.label += STATESPLIT + 'HV'
	elif name == 'splitAux2':  # Stanford Parser splitAux=2
		for node in tree.subtrees(lambda n: strip(n.label)
				in {'VBZ', 'VBP', 'VBD', 'VBN', 'VBG', 'VB'}):
			if sent[node[0]].lower() in {"'s", "s"}:
				# 's can be a contraction of both "is" and "have"
				foundAux = False
				for sibling in node.parent:
					if foundAux:
						if base(sibling, 'VP') and any(strip(a.label)
								in {'VBD', 'VBN'} for a in sibling):
							node.label += STATESPLIT + 'HV'
							break
					elif sibling.label.startswith('VBZ'):
						foundAux = True
				else:
					node.label += STATESPLIT + 'BE'
			if sent[node[0]].lower() in {'am', 'is', 'are', 'was', 'were',
					"'m", "'re", 'be', 'being', 'been', 'ai'}:
				node.label += STATESPLIT + 'BE'
			elif sent[node[0]].lower() in {
					'have', "'ve", 'having', 'has', 'had', "'d"}:
				node.label += STATESPLIT + 'HV'
	elif name == 'splitVP':  # Stanford Parser splitVP=2
		for node in tree.subtrees(lambda n: base(n, 'VP')):
			for child in node:
				if ishead(child):
					if strip(child.label) in {'VBZ', 'VBP', 'VBD', 'MD'}:
						node.label += STATESPLIT + 'VBF'
					else:
						node.label += STATESPLIT + strip(child.label)
					break
	elif name == 'splitVP3':  # Stanford Parser splitVP=3
		for node in tree.subtrees(lambda n: base(n, 'VP')):
			for child in node:
				if ishead(child):
					if strip(child.label) in {'VBZ', 'VBP', 'VBD', 'MD'}:
						node.label += STATESPLIT + 'VBF'
					elif strip(child.label) in {'TO', 'VBG', 'VBN', 'VB'}:
						node.label += STATESPLIT + strip(child.label)
					break
	elif name == 'splitSGapped':  # Stanford Parser splitSGapped=3
		seenPredCat = seenCC = seenS = False
		seenNP = 0
		for node in tree.subtrees(lambda n: base(n, 'S')):
			for child in node:
				cat2 = child.label
				if cat2.startswith('NP'):
					seenNP += 1
				elif strip(cat2) in {'VP', 'ADJP', 'PP', 'UCP'}:
					seenPredCat = True
				elif cat2.startswith('CC'):
					seenCC = True
				elif cat2.startswith('S'):
					seenS = True
			if (not (seenCC and seenS)) and (
					seenNP == 0 or (seenNP == 1 and not seenPredCat)):
				node.label += STATESPLIT + 'G'
	elif name == 'splitTMP':  # Stanford Parser splitTMP=TEMPORAL_ACL03PCFG
		for node in tree.postorder(lambda n: 'TMP' in functions(n)):
			child = node
			hd = None
			while node and isinstance(node[0], Tree):
				try:
					i, hd = next(a for a in enumerate(child) if ishead(a[1]))
				except StopIteration:
					break
				if strip(hd) == 'POS' and i > 0:
					hd = child[i - 1]
				child = hd
			if 'TMP' in functions(node):
				node.label += STATESPLIT + 'TMP'
			if hd and hd.label.startswith('N'):
				hd.label += STATESPLIT + 'TMP'
	elif name == 'splitBaseNP':  # Stanford Parser splitBaseNP=1
		# Mark NPs that only dominate preterminals
		for node in tree.subtrees(lambda n: base(n, 'NP')):
			if all(a and isinstance(a[0], int) for a in node):
				node.label += STATESPLIT + 'B'
	elif name == 'dominatesV':  # Stanford Parser dominatesV=1
		for node in tree.subtrees(lambda n: base(n, 'VP')):
			if any(tag.startswith('V') or tag.startswith('MD')
					for _, tag in node.pos()):
				node.label += STATESPLIT + 'v'
	elif name == 'splitNPADV':  # Stanford Parser splitNPADV=1
		for node in tree.subtrees(lambda n:
				base(n, 'NP') and 'ADV' in functions(n)):
			node.label += STATESPLIT + 'ADV'
			try:
				hd = next(a for a in node if ishead(a))
			except StopIteration:
				continue
			if base(hd, 'POS') and hd.parent_index > 0:
				hd = node[hd.parent_index - 1]
			while base(hd, 'NP'):
				hd.label += STATESPLIT + 'ADV'
				try:
					hd = next(a for a in hd if ishead(a))
				except StopIteration:
					break
	elif name == 'markDitransV':  # Stanford Parser markDitransV=2
		for node in tree.subtrees(lambda n: n.label.startswith('VB')):
			npargs = sum(1 for a in node.parent if base(a, 'NP')
					and 'TMP' not in functions(a))
			if npargs >= 2:
				node.label += STATESPLIT + '2Arg'
	elif name == 'PTBbrackets':  # ensure that brackets are in PTB format
		sent[:] = [ptbescape(token) for token in sent]
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
			t.label += STATESPLIT + "infinitive"
	elif name == 'markpart':
		for t in tree.subtrees(lambda n: strip(n.label) == "V"
				and isinstance(n.parent, Tree)
				and isinstance(n.parent.parent, Tree)
				and strip(n.parent.label) == "VN"
				and strip(n.parent.parent.label) == "VPpart"):
			t.label += STATESPLIT + "participle"
	elif name == 'markvn':
		for t in tree.subtrees(lambda n: strip(n.label) == "VN"):
			for sub in islice(t.subtrees(), 1, None):
				sub.label += STATESPLIT + "withVN"
	elif name == 'coord1':
		for t in tree.subtrees(lambda n: strip(n.label) == 'COORD'
				and len(n) >= 2):
			t.label += STATESPLIT + strip(t[1].label)
	elif name == 'de2':
		for t in tree.subtrees(lambda n: strip(n.label) == 'P'
				and DERE.match(sent[n[0]])):
			t.label += STATESPLIT + "de2"
	elif name == 'de3':
		# @NP|PP|COORD >+(@NP|PP) (@PP <, (@P < /^([Dd]es?|du|d')$/))
		for t in tree.subtrees(lambda n:
				strip(n.label) in ("PP", "COORD")):
			a = list(ancestors(t))
			for n in range(2, len(a)):
				if PPORNP.match("".join(strip(x.label) for x in a[:n])):
					if (strip(a[n - 1][0].label) == "P"
							and DERE.match(sent[a[n - 1][0][0]])):
						t.label += STATESPLIT + "de3"
						break
	elif name == 'markp1':
		for t in tree.subtrees(lambda n: strip(n.label) == "P"
				and strip(n.parent.label) == "PP"
				and strip(n.parent.parent.label) == "NP"):
			t.label += STATESPLIT + "n"
	elif name == 'mwadvs':
		for t in tree.subtrees(lambda n: strip(n.label) == "MWADV"
				and "S" in n.parent.label):
			t.label += STATESPLIT + "mwadv-s"
	elif name == 'mwadvsel1':
		for t in tree.subtrees(lambda n: strip(n.label) == "MWADV"
				and len(n) == 2
				and strip(n[0].label) == "P"
				and strip(n[1].label) == "N"):
			t.label += STATESPLIT + "mwadv1"
	elif name == 'mwadvsel2':
		for t in tree.subtrees(lambda n: strip(n.label) == "MWADV"
				and len(n) == 3
				and strip(n[0].label) == "P"
				and strip(n[1].label) == "D"
				and strip(n[2].label) == "N"):
			t.label += STATESPLIT + "mwadv2"
	elif name == 'mwnsel1':
		for t in tree.subtrees(lambda n: strip(n.label) == "MWN"
				and len(n) == 2
				and strip(n[0].label) == "N"
				and strip(n[1].label) == "A"):
			t.label += STATESPLIT + "mwn1"
	elif name == 'mwnsel2':
		for t in tree.subtrees(lambda n: strip(n.label) == "MWN"
				and len(n) == 3
				and strip(n[0].label) == "N"
				and strip(n[1].label) == "P"
				and strip(n[2].label) == "N"):
			t.label += STATESPLIT + "mwn2"
	elif name == 'mwnsel3':  # noun-noun compound joined with dash.
		for t in tree.subtrees(lambda n: strip(n.label) == "MWN"
				and len(n) == 3
				and strip(n[0].label) == "N"
				and sent[n[1][0]] == "-"
				and strip(n[2].label) == "N"):
			t.label += STATESPLIT + "mwn3"
	else:
		return False
	return True


def lassytransforms(name, tree, _sent):
	"""Transformations for the Dutch Lassy & Alpino treebanks."""
	if name == 'nlselectmorph':  # add select morph. feats to coarse POS tags
		SELECTMORPH = {'eigen', 'det', 'pron', 'init', 'fin', 'neven', 'onder',
				'prenom', 'nom', 'vrij', 'pv', 'inf', 'vd', 'od'}
		for pos in tree.subtrees(lambda n: n and isinstance(n[0], int)):
			tag = pos.source[MORPH].split('(')[0]
			selected = [feat for feat in morphfeats(pos)
					if feat in SELECTMORPH]
			pos.label += '/%s[%s]' % (tag, ','.join(selected))
	elif name == 'nlpercolatemorph':  # percolate select morph tags upwards
		PERCOLATE = {'pv': 2, 'inf': 2}
		for feat in sorted(PERCOLATE):
			lvl = PERCOLATE[feat]
			for pos in tree.subtrees(lambda n, f=feat: n
					and isinstance(n[0], int) and f in morphfeats(n)):
				cnt = 0
				node = pos.parent
				while (cnt < lvl and node is not None
						and node.parent is not None):
					if not node.label.endswith(STATESPLIT + feat):
						node.label += STATESPLIT + feat
					node = node.parent
					cnt += 1
	elif name == 'nlmwuhead':  # add label of head child to MWU nodes
		EXPANDCAT = {'MWU'}
		for node in tree.subtrees(lambda n: strip(n.label) in EXPANDCAT):
			node.label += STATESPLIT + next(
					iter(strip(a.label) for a in node
					if ishead(a) or a is node[-1]))
	elif name == 'nladdunary':  # introduce unary NPs
		maxid = getmaxid(tree)
		for node in tree.postorder(lambda n: strip(n.label) in {'n', 'vnw'}
				and strip(n.parent.label) in {'SMAIN', 'PP', 'INF'}):
			children = node[:]
			node[:] = []
			tag = ParentedTree(node.label, children)
			tag.source = node.source[:]
			node[:] = [tag]
			node.source[TAG] = node.label = 'NP'
			node.source[FUNC] = tag.source[FUNC]
			if node.source[FUNC] and node.source[FUNC][0].isupper():
				tag.source[FUNC] = 'HD'
			else:
				tag.source[FUNC] = 'hd'
			maxid += 1
			node.source[WORD] = '#%d' % maxid
			node.source[LEMMA] = node.source[MORPH] = '--'
	elif name == 'nlelimcnj':  # assign conjuncts the function of the parent
		for node in tree.subtrees(lambda n: function(n) == 'cnj'):
			node.source[FUNC] = function(node.parent) or '--'
	else:
		return False
	return True


def reversetransform(tree, transformations):
	"""Undo specified transformations and remove state splits marked by ``^``.

	Do not apply twice (might remove VPs which shouldn't be)."""
	# Generic state-split removal
	for node in tree.subtrees(lambda n: STATESPLIT in n.label[1:]):
		node.label = node.label[:node.label.index(STATESPLIT, 1)]
	# restore linear precedence order
	for a in tree.subtrees(lambda n: len(n) > 1):
		a.children.sort(key=lambda n: min(n.leaves())
				if isinstance(n, Tree) else n)
	# unfreeze attributes so that they can be modified
	for a in tree.subtrees():
		if isinstance(getattr(a, 'source', None), tuple):
			a.source = list(a.source)

	for name in reversed(transformations):
		if name == 'FANOUT':
			removefanoutmarkers(tree)
		elif name == 'DP':  # remove DPs
			for dp in tree.subtrees(lambda n: n.label == 'DP'):
				dp.label = 'NP'
				if len(dp) > 1 and dp[1].label == 'NP':
					# dp1 = dp[1][:]
					# dp[1][:] = []
					# dp[1:] = dp1
					dp[1][:], dp[1:] = [], dp[1][:]
		elif name == 'NEST':  # flatten adjunctions
			nkonly = {'PDAT', 'CAP', 'PPOSS', 'PPOSAT', 'ADJA', 'FM', 'PRF',
					'NM', 'NN', 'NE', 'PIAT', 'PRELS', 'PN', 'TRUNC', 'CH',
					'CNP', 'PWAT', 'PDS', 'VP', 'CS', 'CARD', 'ART', 'PWS',
					'PPER'}
			probably_nk = {'AP', 'PIS'} | nkonly
			for n in tree.subtrees():
				if (len(n) == 2 and n.label == 'NP'
						and [x.label for x in n].count('NP') == 1
						and not set(labels(n)) & probably_nk):
					n.children.sort(key=lambda n: n.label == 'NP')
					n[:] = n[:1] + n[1][:]
		elif name == 'PP-NP':  # flatten PPs
			for pp in tree.subtrees(lambda n: n.label == 'PP'):
				if 'NP' in labels(pp) and 'NN' not in labels(pp):
					# ensure NP is in last position
					pp.children.sort(key=lambda n: n.label == 'NP')
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
			# if any(a.label == 'S' for a in tree):
			# 	map(mergevp, [a for a in tree if a.label == 'S'])
			# elif any(a.label == 'CS' for a in tree):
			# 	map(mergevp, [s for cs in tree for s in cs if cs.label == 'CS'
			# 		and s.label == 'S'])
			for s in tree.subtrees(lambda n: n.label == 'S'):
				mergevp(s)
		elif name == 'POS-PART':
			# remove constituents for particle verbs
			# get the grandfather of each verb particle
			def hasparticle(n):
				"""Test whether node has a PTKVZ node."""
				return any('PTKVZ' in (x.label
					for x in m if isinstance(x, Tree)) for m in n
					if isinstance(m, Tree))
			for a in list(tree.subtrees(hasparticle)):
				for n, b in enumerate(a):
					if (len(b) == 2 and b.label.startswith('V')
						and 'PTKVZ' in (c.label for c in b
							if isinstance(c, Tree))
						and any(c.label == b.label for c in b)):
						a[n:n + 1] = b[:]
		elif name == 'addUnary':
			# remove phrasal projections for single tokens
			# e.g. S => NP => NN becomes S => NN
			for node in tree.subtrees(lambda n:
					strip(n.label) in {'S', 'VP', 'ROOT', 'DL'}):
				for child in node:
					if (len(child) == 1 and base(child, 'NP')
							and strip(child[0].label) in {'NN', 'PPER', 'PDS',
								'PIS', 'PRELS', 'CARD', 'PN'}):
						child.label = child[0].label
						origfunc = function(child)
						child.source = getattr(child[0], 'source', None)
						if child.source:
							child.source[FUNC] = origfunc or '--'
						children = child[0][:]
						child[0][:] = []
						child[:] = children
		elif name == 'elimNKCJ':  # restore NK and CJ functions
			for node in tree.subtrees(lambda n: strip(n.label)
					in {'NP', 'PP'}):
				for child in node:
					if not getattr(child, 'source', None):
						child.source = ['--'] * 6
					if function(child) == 'HD':
						child.source[FUNC] = 'NK'
			for node in tree.subtrees(lambda n: strip(n.label)
					in {'CS', 'CNP', 'CVP', 'CAP', 'CAVP', 'CAC'}):
				for child in node:
					if not getattr(child, 'source', None):
						child.source = ['--'] * 6
					if function(child) != 'CD':
						child.source[FUNC] = 'CJ'
		elif name == 'nladdunary':  # remove unary node
			for node in tree.subtrees(lambda n:
					strip(n.label) in {'SMAIN', 'PP', 'INF'}):
				for child in node:
					if (len(child) == 1 and isinstance(child[0], Tree)
							and base(child, 'NP')
							and strip(child[0].label) in {'n', 'vnw'}):
						child.label = child[0].label
						origfunc = function(child)
						child.source = getattr(child[0], 'source', None)
						if child.source:
							child.source[FUNC] = origfunc or '--'
						children = child[0][:]
						child[0][:] = []
						child[:] = children
		elif name == 'nlelimcnj':  # restore cnj function
			for node in tree.subtrees(lambda n: base(n, 'CONJ')):
				for child in node:
					if not getattr(child, 'source', None):
						child.source = ['--'] * 6
					if function(child) != 'crd' and not base(child, 'let'):
						child.source[FUNC] = 'cnj'
		elif name == 'APPEND-FUNC':  # functions appended to phrasal labels
			for a in tree.subtrees(lambda n: '-' in n.label
					and not n.label.startswith('-')
					and not n.label.endswith('-')):  # -LRB-
				label, func = a.label.split('-', 1)
				if not getattr(a, 'source', None):
					a.source = ['--'] * 6
				a.source[TAG] = a.label = label
				a.source[FUNC] = func
		# morphological features appended to phrasal labels
		elif name in {'APPEND-MORPH', 'addCase', 'nlselectmorph'}:
			for a in tree.subtrees(lambda n: n and isinstance(n[0], int)):
				if '/' in a.label:
					label, morph = a.label.split('/', 1)
					if not getattr(a, 'source', None):
						a.source = ['--'] * 6
					a.source[TAG] = a.label = label
					a.source[MORPH] = morph.replace('[', '(').replace(']', ')')
		elif name == 'FUNC-NODE':  # nodes with function above phrasal labels
			for a in list(tree.postorder(lambda n: n.label.startswith('-')
					and not n.label.endswith('-')  # -LRB-
					and n and isinstance(n[0], Tree))):
				a.source = ['--'] * 6
				a.source[FUNC] = a.label[1:]
				a.source[TAG] = a.label = a[0].label
				a[:] = [a[0].pop() for _ in range(len(a[0]))][::-1]
		elif name == 'MORPH-NODE':  # nodes with morph. above preterminals
			for a in list(tree.postorder(lambda n: n and isinstance(n[0], Tree)
					and n[0] and isinstance(n[0][0], int))):
				a.source = ['--'] * 6
				a.source[MORPH] = a.label
				a.source[TAG] = a.label = a[0].label
				a[:] = [a[0].pop() for _ in range(len(a[0]))][::-1]
		elif name == 'LEMMA-NODE':  # nodes with lemmas above words
			for a in list(tree.postorder(lambda n: n and isinstance(n[0], Tree)
					and n[0] and isinstance(n[0][0], int))):
				a.source = ['--'] * 6
				a.source[LEMMA] = unescape(a[0].label)
				a.source[TAG] = a.label
				a[:] = [a[0].pop() for _ in range(len(a[0]))][::-1]

	# restore linear precedence order
	for a in tree.subtrees(lambda n: len(n) > 1):
		a.children.sort(key=lambda n: min(n.leaves())
				if isinstance(n, Tree) else n)
	return tree


def collapselabels(trees, _sents=None, tbmapping=None):
	"""Collapse non-root phrasal labels with specified mapping.

	Trees are modified in-place.

	:param tbmapping: a mapping of treebank labels of the form::

			{coarselabel1: {finelabel1, finelabel2, ...}, ...}

		Cf. ``treebanktransforms.MAPPINGS``
	:returns: a tuple ``(trees, mapping)`` with the transformed trees
		and a mapping of their original labels to the collapsed labels.
	"""
	def collapse(tree):
		"""Collapse labels of a single tree."""
		for node in islice(tree.subtrees(), 1, None):
			if node and isinstance(node[0], Tree):
				# anything not part of the mapping is stripped
				# (state splits, function tags, &c.)
				mapping[node.label] = LABELRE.sub(
						lambda x: revmapping.get(x.group(), ''),
						# lambda x: revmapping[x.group()],
						node.label).replace('-', '').rstrip('^')
				assert (mapping[node.label]
						and mapping[node.label][0].isalpha()), node.label
				node.label = mapping[node.label]

	# maps original treebank labels to coarser labels; e.g. NP => X
	revmapping = {finelabel: coarselabel for coarselabel in tbmapping
			for finelabel in tbmapping[coarselabel]}
	# maps labels after binarization and other transformations,
	# e.g., NP<DT,JJ,NN> => X<X,X,X>
	mapping = {'Epsilon': 'Epsilon', trees[0].label: trees[0].label}
	# collect POS tags, will not be changed
	for tree in trees:
		for node in tree.subtrees(lambda n: not isinstance(n[0], Tree)):
			mapping[node.label] = node.label
			revmapping[node.label] = node.label
	for tree in trees:
		collapse(tree)
	return trees, mapping


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

	if isinstance(tree[0], int):
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
	if isinstance(tree[0], int):
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
	result.source = ['--'] * 6
	result.source[TAG] = result.label
	if morph:
		result.source[MORPH] = morph.replace('[', '(').replace(']', ')')
	if func and adjunctionlabel and ':' in func:
		result.source[FUNC] = adjunctionlabel
	elif func:
		result.source[FUNC] = func
	return result


def dlevel(tree, lang='nl'):
	"""Return the D-level measure of syntactic complexity.

	Original version:
	Rosenberg & Abbeduto (1987), https://doi.org/10.1017/S0142716400000047
	Covington et al. (2006), http://ai1.ai.uga.edu/caspr/2006-01-Covington.pdf
	Dutch version implemented here: Appendix A of T-Scan manual,
	https://github.com/proycon/tscan/raw/master/docs/tscanhandleiding.pdf

	:param tree: A tree from the Alpino parser (i.e., not binarized, with
		function and morphological tags).
	:returns: integer 0-7; 7 is most complex."""
	if lang != 'nl':
		raise NotImplementedError
	poslist = []
	pv_counter = neven_counter = 0
	for pos in tree.subtrees(lambda n: n and isinstance(n[0], int)):
		poslist.append(pos)
		if strip(pos.label) == 'ww' and 'pv' in morphfeats(pos):
			pv_counter += 1
		elif strip(pos.label) == 'vg' and 'neven' in morphfeats(pos):
			neven_counter += 1

	# 7: sentence with multiple subordinate clauses
	# (disregarding clauses in conjunctions)
	if pv_counter - neven_counter > 2:
		return 7
	# 6: a subordinate clause modifying the subject
	for node in tree.subtrees():
		if (strip(node.label) == 'REL' and function(node) == 'mod'
				and function(node.parent) == 'su'):
			return 6
		elif (strip(node.label) in ('CP', 'WHSUB', 'WHREL', 'TI', 'OTI', 'INF')
				and function(node) == 'su'):
			return 6
		elif (strip(node.label) == 'ww' and function(node.parent) == 'su'
				and strip(node.parent.label) == 'NP'):
			return 6
	# 5: subordinate clause
	for pos in poslist:
		if (strip(pos.label) == 'vg' and 'onder' in morphfeats(pos)
				and pos.source[LEMMA] != 'dat'):
			return 5
	# 4: non-finite clause as object with overt subject
	for node in tree.subtrees():
		if function(node) == 'obcomp':
			return 4
	for node in tree.subtrees(lambda n: function(n) == 'vc'):
		if strip(node.label) in ('TI', 'OTI', 'INF'):
			vcid = node.source[WORD].lstrip('#')
			for sib in node.parent:
				if function(sib) == 'obj1' and hassecedge(sib, 'su', vcid):
					return 4
	# 3: finite clause as objects (and equivalents)
	for node in tree.subtrees():
		if strip(node.label) == 'REL' and function(node) == 'mod':
			if function(node.parent) == 'obj1':
				return 3
		elif strip(node.label) == 'ww':
			if strip(node.parent.label) == 'NP' and function(
					node.parent) == 'obj1':
				return 3
		elif strip(node.label) in ('CP', 'WHSUB') and function(node) == 'vc':
			return 3
		elif function(node) == 'sup':
			return 3
	# 2: coordinated structure
	for pos in poslist:
		if strip(pos.label) == 'vg' and 'neven' in morphfeats(pos):
			return 2
	# 1: non-finite clause with subject coindexed from main clause
	for node in tree.subtrees(lambda n: function(n) == 'vc'):
		if strip(node.label) in ('TI', 'OTI', 'INF'):
			vcid = node.source[WORD].lstrip('#')
			for sib in node.parent:
				if function(sib) == 'su' and hassecedge(sib, 'su', vcid):
					return 2
	# 0: simple sentence
	return 0


def rindex(l, v):
	"""Like list.index(), but go from right to left."""
	return len(l) - 1 - l[::-1].index(v)


def labels(tree):
	""":returns: the labels of the children of this node."""
	return [a.label for a in tree if isinstance(a, Tree)]


def pop(node):
	"""Remove this node from its parent node, if it has one.

	Convenience function for ParentedTrees."""
	try:
		return node.parent.pop(node.parent_index)
	except AttributeError:
		return node


def base(node, match):
	"""Test whether ``node.label`` equals ``match`` after stripping features."""
	return (node.label == match
			or node.label.startswith(match + STATESPLIT)
			or node.label.startswith(match + '-'))


def strip(label):
	"""Equivalent to the effect of the @ operator in tregex."""
	if '-' in label:
		return label[:label.index('-')]
	elif STATESPLIT in label:
		return label[:label.index(STATESPLIT)]
	return label


def ancestors(node):
	"""Yield ancestors of node from direct parent to root node."""
	while node:
		node = node.parent
		yield node


def bracketings(tree):
	"""Labeled bracketings of a tree."""
	return [(a.label, tuple(sorted(a.leaves())))
		for a in tree.subtrees(lambda t: t and isinstance(t[0], Tree))]


# morphological features
def morphfeats(node):
	"""Return the morphological features of a preterminal node.

	Features may be separated by dots or commas."""
	try:
		morph = node.source[MORPH].replace('(', '[').replace(')', ']')
		morph = morph[morph.index('[') + 1:morph.index(']')]
	except (TypeError, ValueError):
		return ()
	return morph.replace('.', ',').split(',')


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


# Function tags
def function(node):
	""":returns: The first function tag for node, or the empty string."""
	if getattr(node, 'source', None) is None:
		return ''
	return node.source[FUNC].split('-')[0]


def functions(node):
	""":returns: list of function tags for node, or an empty list."""
	if getattr(node, 'source', None) is None:
		return []
	a = node.source[FUNC]
	if a == '--' or a == '' or a is None:
		return []
	return a.split('-')


# Secondary edges
def hassecedge(node, func, parentid):
	"""Test whether this node has a secondary edge ``(func, parentid)``."""
	if getattr(node, 'source', None) is None:
		return False
	return any(f == func and pid == parentid
			for f, pid in zip(node.source[6::2], node.source[7::2]))


def getmaxid(tree):
	"""Return highest export non-terminal ID in tree."""
	return max((int(node.source[WORD].lstrip('#'))
			for n, node in enumerate(
				tree.subtrees(lambda n: n.source
					and EXPORTNONTERMINAL.match(n.source[WORD])),
				500)),
			default=500)

__all__ = ['expandpresets', 'transform', 'reversetransform', 'collapselabels',
		'dlevel', 'rrtransform', 'rrbacktransform', 'rindex', 'labels', 'pop',
		'strip', 'ancestors', 'bracketings', 'morphfeats', 'unifymorphfeat',
		'function', 'functions', 'hassecedge']
