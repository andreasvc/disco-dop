from nltk import Tree
from itertools import count, repeat
from os import path
from glob import glob
import re, codecs, logging

BOS = re.compile("^#BOS.*\n")
EOS = re.compile("^#EOS")
WORD, LEMMA, TAG, MORPH, FUNC, PARENT = range(6)

class NegraCorpusReader():
	""" Read a corpus in the Negra export format. """
	def __init__(self, root, fileids, encoding="utf-8", headorder=False,
	headfinal=False, headreverse=False, unfold=False, functiontags=False,
	removepunct=False, movepunct=False):
		""" headorder: whether to order constituents according to heads
			headfinal: whether to put the head in final or in frontal position
			headreverse: the head is made final/frontal by reversing everything
			before or after the head. When true, the side on which the head is
				will be the reversed side.
			unfold: whether to apply corpus transformations
			functiontags: whether to add function tags to node labels e.g. NP+OA
			removepunct: eliminate punctuation
			movepunct: move punctuation to appropriate constituents"""
		self.headorder = headorder; self.reverse = headreverse
		self.headfinal = headfinal;	self.unfold = unfold
		self.functiontags = functiontags;
		self.removepunct = removepunct; self.movepunct = movepunct
		self.headrules = readheadrules() if headorder else {}
		self._encoding = encoding
		self._filenames = glob(path.join(root, fileids))
		self._block_cache = self._read_blocks()
		self._sents_cache = None
		self._tagged_sents_cache = None
		self._parsed_sents_cache = None
	def parsed_sents(self, fileids=None):
		if not self._parsed_sents_cache:
			self._parsed_sents_cache = map(self._parse, self._block_cache)
		return self._parsed_sents_cache
	def tagged_sents(self, fileids=None):
		if not self._tagged_sents_cache:
			self._tagged_sents_cache = map(self._tag, self._block_cache)
		return self._tagged_sents_cache
	def sents(self, fileids=None):
		if not self._sents_cache:
			self._sents_cache = map(self._word, self._block_cache)
		return self._sents_cache
	def blocks(self):
		""" Return a list of blocks from the treebank file,
		with any transformations applied."""
		return [export(*x) for x in zip(self.parsed_sents(),
			self.tagged_sents(), repeat(None))]
		return ['\n'.join('\t'.join(a) for a in block[1:-1]) + '\n'
			for block in self._block_cache]
	def _read_blocks(self):
		def sixelements(a):
			""" take a line and add dummy lemma if that field is not present """
			if "%%" in a: a[a.index("%%"):] = []
			lena = len(a)
			if lena == 5: return a[:1] + [''] + a[1:]
			elif lena == 6: return a
			else: raise ValueError("expected 5 or 6 columns: %r" % a)
		result = []
		started = False
		for filename in self._filenames:
			for line in codecs.open(filename, encoding=self._encoding):
				if line.startswith("#BOS"):
					if not started:
						started = True
						lines = []
					else: raise ValueError("beginning of sentence marker while "
							"previous one still open: %s" % line)
				elif started and line.startswith("#EOS"):
					started = False
					result.append(lines)
				elif started: lines.append(sixelements(line.split()))
		return result
	def _parse(self, s):
		def getchildren(parent, children):
			results = []
			for n, a in children[parent]:
				# n is the index in the block to record word indices
				if a[WORD].startswith("#"): #nonterminal
					results.append(Tree(a[TAG], getchildren(a[WORD][1:],
						children)))
					results[-1].source = a
				else: #terminal
					results.append(Tree(a[TAG].replace("$(", "$["), [n]))
					results[-1].source = a
			return results
		children = {}
		for n,a in enumerate(s):
			children.setdefault(a[PARENT], []).append((n,a))
		result = Tree("ROOT", getchildren("0", children))
		# roughly order constituents by order in sentence
		for a in reversed(list(result.subtrees(lambda x: len(x) > 1))):
			a.sort(key=lambda x: x.leaves())
		if self.removepunct: doremovepunct(result)
		elif self.movepunct:
			punctraise(result)
			balancedpunctraise(result, self._word(s))
			# restore order
			for a in reversed(list(result.subtrees(lambda x: len(x) > 1))):
				a.sort(key=lambda x: x.leaves())
		if self.unfold: result = unfold(result)
		if self.headorder:
			map(lambda x: headfinder(x, self.headrules), result.subtrees())
			map(lambda x: headorder(x, self.headfinal, self.reverse),
													result.subtrees())
		if self.functiontags: addfunctions(result)
		return result
	def _word(self, s):
		if self.removepunct:
			return [a[WORD] for a in s if a[WORD][0] != "#"
				and not a[TAG].startswith("$")]
		return [a[WORD] for a in s if a[WORD][0] != "#"]
	def _tag(self, s):
		if self.removepunct:
			return [(a[WORD], a[TAG].replace("$(", "$[")) for a in s
				if a[WORD][0] != "#" and not a[TAG].startswith("$")]
		return [(a[WORD], a[TAG].replace("$(", "$[")) for a in s
				if a[WORD][0] != "#"]

class DiscBracketCorpusReader():
	""" A corpus reader where the phrase-structure is represented by a tree in
	bracket notation, where the leaves are indices pointing to words in a
	separately represented sentence; e.g.:
	(S (NP 1) (VP (VB 0) (JJ 2)) (? 3))	is John rich ?
	Note that the tree and the sentence are separated by a tab, while the words
	in the sentence are separated by spaces. There is one tree plus sentence
	per line.
	Compared to Negra's export format, this format lacks morphology, lemmas and
	functional edges. On the other hand, it is very close to the internal
	representation employed here, so it is quick to load.
	"""
	def __init__(self, root, fileids, encoding="utf-8", headorder=False,
	headfinal=False, headreverse=False, unfold=False, functiontags=False,
	removepunct=False, movepunct=False):
		""" headorder: whether to order constituents according to heads
			headfinal: whether to put the head in final or in frontal position
			headreverse: the head is made final/frontal by reversing everything
			before or after the head. When true, the side on which the head is
				will be the reversed side.
			unfold: whether to apply corpus transformations
			functiontags: whether to add function tags to node labels e.g. NP+OA
			removepunct: eliminate punctuation
			movepunct: move punctuation to appropriate constituents"""
		self.headorder = headorder; self.reverse = headreverse
		self.headfinal = headfinal;	self.unfold = unfold
		self.functiontags = functiontags;
		self.removepunct = removepunct; self.movepunct = movepunct
		self.headrules = readheadrules() if headorder else {}
		self._encoding = encoding
		self._filenames = glob(path.join(root, fileids))
		self._parsed_sents_cache = None
	def sents(self, fileids=None):
		return [line.split("\t", 1)[1].rstrip("\n\r").split(" ")
			for filename in self._filenames
				for line in codecs.open(filename, encoding=self._encoding)]
	def tagged_sents(self, fileids=None):
		# for each line, zip its words & tags together in a list.
		return [zip(line.split("\t", 1)[1].rstrip("\n\r").split(" "),
				map(itemgetter(1), sorted(Tree.parse(line.split("\t", 1)[0],
					parse_leaf=int).pos())))
			for filename in self._filenames
				for line in codecs.open(filename, encoding=self._encoding)]
	def parsed_sents(self, fileids=None):
		if not self._parsed_sents_cache:
			self._parsed_sents_cache = map(self._parse, self.blocks())
		return self._parsed_sents_cache
	def blocks(self):
		""" Return a list of strings containing the raw representation of
		trees in the treebank, with any transformations applied."""
		return [line for filename in self._filenames
			for line in codecs.open(filename, encoding=self._encoding)]
	def _parse(self, s):
		result = Tree.parse(s.split("\t", 1)[0], parse_leaf=int)
		# roughly order constituents by order in sentence
		for a in reversed(list(result.subtrees(lambda x: len(x) > 1))):
			a.sort(key=lambda x: x.leaves())
		if self.removepunct: doremovepunct(result)
		elif self.movepunct:
			punctraise(result)
			balancedpunctraise(result, self._word(s))
			# restore order
			for a in reversed(list(result.subtrees(lambda x: len(x) > 1))):
				a.sort(key=lambda x: x.leaves())
		if self.unfold: result = unfold(result)
		if self.headorder:
			map(lambda x: headfinder(x, self.headrules), result.subtrees())
			map(lambda x: headorder(x, self.headfinal, self.reverse),
													result.subtrees())
		if self.functiontags: addfunctions(result)
		return result

def export(tree, sent, n):
	""" Convert a tree with indices as leafs and a sentence with the
	corresponding non-terminals to a single string in Negra's export format."""
	result = []
	if n: result.append("#BOS %d" % n)
	wordsandpreterminals = tree.treepositions('leaves') + [a[:-1]
		for a in tree.treepositions('leaves')]
	phrasalnodes = list(sorted([a for a in tree.treepositions()
		if a not in wordsandpreterminals and a != ()], key=len, reverse=True))
	wordids = dict((tree[a], a) for a in tree.treepositions('leaves'))
	for i, word in enumerate(sent):
		idx = wordids[i]
		result.append("\t".join((word[0],
				tree[idx[:-1]].node.replace("$[","$("),
				"--", "--",
				str(500+phrasalnodes.index(idx[:-2]) if len(idx) > 2 else 0))))
	for idx in phrasalnodes:
		result.append("\t".join(("#%d" % (500 + phrasalnodes.index(idx)),
				tree[idx].node,
				"--", "--",
				str(500+phrasalnodes.index(idx[:-1]) if len(idx) > 1 else 0))))
	if n: result.append("#EOS %d" % n)
	return "\n".join(result) #.encode("utf-8")

def addfunctions(tree):
	# FIXME: pos tags probably shouldn't get a function tag.
	for a in tree.subtrees():
		if a.source[FUNC].split("-")[0]:
			a.node += "+%s" % a.source[FUNC].split("-")[0]

def readheadrules():
	headrules = {}
	# this file containing head assignment rules is part of rparse,
	# under src/de/tuebingen/rparse/treebank/constituent/negra/
	try: rulefile = open("negra.headrules")
	except IOError:
		logging.warning("negra head rules not found! no head annotation will be performed.")
		return headrules
	for a in rulefile:
		if a.strip() and not a.strip().startswith("%") and len(a.split()) > 2:
			label, lr, heads = a.upper().split(None, 2)
			headrules.setdefault(label, []).append((lr, heads.split()))
	headrules["ROOT"] = headrules["VROOT"]
	return headrules

def sethead(child):
	child.source = getattr(child, "source", 6 * [''])
	if "-" in child.source[FUNC]: pass
	elif child.source[FUNC]: child.source[FUNC] += "-HD"
	else: child.source[FUNC] = "HD"

def headfinder(tree, headrules):
	""" use head finding rules to mark one child of tree as head. """
	if any(a.source[FUNC].split("-")[-1] == "HD" for a in tree
		if hasattr(a, "source")): return
	for lr, heads in headrules.get(tree.node, []):
		if lr == "LEFT-TO-RIGHT": children = tree
		elif lr == "RIGHT-TO-LEFT": children = tree[::-1]
		else: raise ValueError
		for head in heads:
			for child in children:
				if isinstance(child, Tree) and child.node == head:
					sethead(child)
					return
	# default head is initial nonterminal
	for a in tree:
		if isinstance(a, Tree):
			sethead(a)
			return

def headorder(tree, headfinal, reverse):
	""" change order of constituents based on head (identified with
	function tag). """
	head = [n for n,a in enumerate(tree)
		if hasattr(a, "source") and "HD" in a.source[FUNC].split("-")]
	if not head: return
	headidx = head.pop()
	# everything until the head is reversed and prepended to the rest,
	# leaving the head as the first element
	if headfinal:
		if reverse:
			# head final, reverse rhs: A B C^ D E => A B E D C^
			tree[:] = tree[:headidx] + tree[headidx:][::-1]
		else:
			# head final, reverse lhs:  A B C^ D E => E D A B C^
			tree[:] = tree[headidx+1:][::-1] + tree[:headidx+1]
	else:
		if reverse:
			# head first, reverse lhs: A B C^ D E => C^ B A D E
			tree[:] = tree[:headidx+1][::-1] + tree[headidx+1:]
		else:
			# head first, reverse rhs: A B C^ D E => C^ D E B A
			tree[:] = tree[headidx:] + tree[:headidx][::-1]

# generalizations suggested by SyntaxGeneralizer of TigerAPI
# however, instead of renaming, we introduce unary productions
# POS tags
tonp = "NN NNE PNC PRF PDS PIS PPER PPOS PRELS PWS".split()
topp = "PROAV PWAV".split()  # ??
toap = "ADJA PDAT PIAT PPOSAT PRELAT PWAT PRELS".split()
toavp = "ADJD ADV".split()

tagtoconst = {}
for a in tonp: tagtoconst[a] = "NP"
#for a in toap: tagtoconst[a] = "AP"
#for a in toavp: tagtoconst[a] = "AVP"

# phrasal categories
tonp = "CNP NM PN".split()
topp = "CPP".split()
toap = "MTA CAP".split()
toavp = "AA CAVP".split()
unaryconst = {}
for a in tonp: unaryconst[a] = "NP"

def function(tree):
	if hasattr(tree, "source"): return tree.source[FUNC].split("-")[0]
	else: return ''

def ishead(tree):
	if hasattr(tree, "source"): return "HD" in tree.source[FUNC]
	else: return False

def rindex(l, v):
	return len(l) - 1 - l[::-1].index(v)

def labels(tree):
	return [a.node for a in tree if isinstance(a, Tree)]

def unfold(tree):
	""" Unfold redundancies and perform other transformations introducing
	more hierarchy in the phrase-structure annotation, based on
	grammatical functions and syntactic categories.
	"""
	#original = tree.copy(Tree); current = tree # for debugging

	# un-flatten PPs
	addtopp = "AC".split()
	for pp in tree.subtrees(lambda n: n.node == "PP"):
		ac = [a for a in pp if function(a) in addtopp]
		# anything before an initial preposition goes to the PP (modifiers,
		# punctuation), otherwise it goes to the NP; mutatis mutandis for
		# postpositions
		functions = map(function, pp)
		if "AC" in functions and "NK" in functions:
			if functions.index("AC") < functions.index("NK"):
				ac[:0] = pp[:functions.index("AC")]
			if rindex(functions, "AC") > rindex(functions, "NK"):
				ac += pp[rindex(functions, "AC")+1:]
		#else: print "PP but no AC or NK", " ".join(functions)
		nk = [a for a in pp if a not in ac]
		# introduce a PP unless there is already an NP in the PP (annotation
		# mistake), or there is a PN and we want to avoid a cylic unary of 
		# NP -> PN -> NP
		if ac and nk and (len(nk) > 1 or nk[0].node not in "NP PN".split()):
			pp[:] = ac + [Tree("NP", nk)]

	# introduce DPs
	#determiners = set("ART PDS PDAT PIS PIAT PPOSAT PRELS PRELAT PWS PWAT PWAV".split())
	determiners = set("ART".split())
	for np in list(tree.subtrees(lambda n: n.node == "NP")):
		if np[0].node in determiners:
			np.node = "DP"
			if len(np) > 1 and np[1].node != "PN":
				np[:] = [np[0], Tree("NP", np[1:])]

	# VP category split based on head
	for vp in tree.subtrees(lambda n: n.node == "VP"):
		hd = [x for x in vp if ishead(x)].pop()
		vp.node += "-" + hd.node

	# introduce finite VP at S level, collect objects and modifiers
	# introduce new S level for discourse markers
	newlevel = "DM".split()
	addtovp = "HD AC DA MO NG OA OA2 OC OG PD VO SVP".split()
	labels = [a.node for a in tree]
	def finitevp(s):
		if any(x.node.startswith("V") and x.node.endswith("FIN")
			for x in s if isinstance(x, Tree)):
			vp = Tree("VP", [a for a in s if function(a) in addtovp])
			# introduce a VP unless it would lead to a unary VP -> VP production
			if len(vp) != 1 or vp[0].node != "VP":
				s[:] = [a for a in s if function(a) not in addtovp] + [vp]

	# relative clause => S becomes SRC
	for s in tree.subtrees(lambda n: n.node == "S" and function(n) == "RC"):
		s.node = "SRC"
	toplevel_s = []
	if "S" in labels:
		toplevel_s = [a for a in tree if a.node == "S"]
		for s in toplevel_s:
			while function(s[0]) in newlevel:
				s[:] = [s[0], Tree("S", s[1:])]
				s = s[1]
				toplevel_s = [s]
	elif "CS" in labels:
		cs = tree[labels.index("CS")]
		toplevel_s = [a for a in cs if a.node == "S"]
	map(finitevp, toplevel_s)

	# introduce POS tag for particle verbs
	for a in tree.subtrees(lambda n: "SVP" in map(function, n)):
		svp = [x for x in a if function(x) == "SVP"].pop()
		#apparently there can be a _verb_ particle without a verb. headlines? annotation mistake?
		if any(map(ishead, a)):
			hd = [x for x in a if ishead(x)].pop()
			if hd.node != a.node:
				particleverb = Tree(hd.node, [hd, svp])
				a[:] = [particleverb if ishead(x) else x
									for x in a if function(x) != "SVP"]

	# introduce SBAR level
	sbarfunc = "CP".split()
	# in the annotation, complementizers belong to the first S
	# in S conjunctions, even when they appear to apply to the whole
	# conjunction.
	for s in list(tree.subtrees(lambda n: n.node == "S"
			and function(n[0]) in sbarfunc and len(n) > 1)):
		s.node = "SBAR"
		s[:] = [s[0], Tree("S", s[1:])]

	# introduce nested structures for modifiers
	# (iterated adjunction instead of sister adjunction)
	#adjunctable = set("NP".split()) # PP AP VP
	#for a in list(tree.subtrees(lambda n: n.node in adjunctable and any(function(x) == "MO" for x in n))):
	#	modifiers = [x for x in a if function(x) == "MO"]
	#	if min(n for n,x in enumerate(a) if function(x) =="MO") == 0:
	#		modifiers[:] = modifiers[::-1]
	#	while modifiers:
	#		modifier = modifiers.pop()
	#		a[:] = [Tree(a.node, [x for x in a if x != modifier]), modifier]
	#		a = a[0]

	# restore linear precedence ordering
	for a in tree.subtrees(lambda n: len(n) > 1): a.sort(key=lambda n: n.leaves())
	# introduce phrasal projections for single tokens (currently only adds NPs)
	for a in tree.treepositions("leaves"):
		tag = tree[a[:-1]]   # e.g. NN
		const = tree[a[:-2]] # e.g. S
		if tag.node in tagtoconst and const.node != tagtoconst[tag.node]:
			newconst = Tree(tagtoconst[tag.node], [tag])
			const[a[-2]] = newconst
	return tree

def fold(tree):
	""" Undo the transformations performed by unfold. Do not apply twice (might
	remove VPs which shouldn't be). """
	# restore linear precedence ordering
	for a in tree.subtrees(lambda n: len(n) > 1): a.sort(key=lambda n: n.leaves())
	global original, current
	original = tree.copy(True)
	current = tree

	# remove DPs
	for dp in tree.subtrees(lambda n: n.node == "DP"):
		dp.node = "NP"
		if len(dp) > 1 and dp[1].node == "NP":
			dp[:] = dp[:1] + dp[1][:]

	# flatten adjunctions
	#nkonly = set("PDAT CAP PPOSS PPOSAT ADJA FM PRF NM NN NE PIAT PRELS PN TRUNC CH CNP PWAT PDS VP CS CARD ART PWS PPER".split())
	#probably_nk = set("AP PIS".split()) | nkonly
	#for np in tree.subtrees(lambda n: len(n) == 2 
	#								and n.node == "NP" 
	#								and [x.node for x in n].count("NP") == 1
	#								and not set(labels(n)) & probably_nk):
	#	np.sort(key=lambda n: n.node == "NP")
	#	np[:] = np[:1] + np[1][:]

	# flatten PPs
	for pp in tree.subtrees(lambda n: n.node == "PP"):
		if "NP" in labels(pp) and "NN" not in labels(pp):
			#ensure NP is in last position
			pp.sort(key=lambda n: n.node == "NP")
			pp[:] = pp[:-1] + pp[-1][:]
	# SRC => S, VP-* => VP
	for s in tree.subtrees(lambda n: n.node == "SRC"): s.node = "S"
	for vp in tree.subtrees(lambda n: n.node.startswith("VP-")): vp.node = "VP"

	# merge extra S level
	for sbar in list(tree.subtrees(lambda n: n.node == "SBAR"
		or (n.node == "S" and len(n) == 2 and labels(n) == ["PTKANT", "S"]))):
		sbar.node = "S"
		if sbar[0].node == "S":
			sbar[:] = sbar[1:] + sbar[0][:]
		else: sbar[:] = sbar[:1] + sbar[1][:]

	# merge finite VP with S level
	def mergevp(s):
		for vp in (n for n,a in enumerate(s) if a.node == "VP"):
			if any(a.node.endswith("FIN") for a in s[vp]):
				s[:] = s[:vp] + s[vp][:] + s[vp+1:]
	#if any(a.node == "S" for a in tree):
	#	map(mergevp, [a for a in tree if a.node == "S"])
	#elif any(a.node == "CS" for a in tree):
	#	map(mergevp, [s for cs in tree for s in cs if cs.node == "CS" and s.node == "S"])
	for s in tree.subtrees(lambda n: n.node == "S"): mergevp(s)

	# remove constituents for particle verbs
	# get the grandfather of each verb particle
	for a in list(tree.subtrees(lambda n: any("PTKVZ" in (x.node for x in m if isinstance(x, Tree)) for m in n if isinstance(m, Tree)))):
		for n,b in enumerate(a):
			if len(b) == 2 and b.node.startswith("V") and "PTKVZ" in (c.node for c in b if isinstance(c, Tree)) and any(c.node == b.node for c in b):
				a[n:n+1] = b[:]

	# remove phrasal projections for single tokens
	for a in tree.treepositions("leaves"):
		tag = tree[a[:-1]]    # NN
		const = tree[a[:-2]]  # NP
		parent = tree[a[:-3]] # PP
		if len(const) == 1 and tag.node in tagtoconst and const.node == tagtoconst[tag.node]:
			parent[a[-3]] = tag
			del const
	# restore linear precedence ordering
	for a in tree.subtrees(lambda n: len(n) > 1): a.sort(key=lambda n: n.leaves())
	return tree

def bracketings(tree):
	return [(a.node, tuple(sorted(a.leaves()))) for a in tree.subtrees(lambda t: t.height() > 2)]

def labelfunc(tree):
	for a in tree.subtrees(): a.node += "-" + function(a)
	return tree

def doremovepunct(tree):
	for a in reversed(tree.treepositions("leaves")):
		if tree[a[:-1]].node.startswith("$"):
			# remove this punctuation node and any empty ancestors
			for n in range(1, len(a)):
				del tree[a[:-n]]
				if tree[a[:-(n+1)]]: break
	#renumber
	oldleaves = sorted(tree.leaves())
	newleaves = dict((a,n) for n, a in enumerate(oldleaves))
	for a in tree.treepositions("leaves"):
		tree[a] = newleaves[tree[a]]
	assert sorted(tree.leaves()) == range(len(tree.leaves())), tree

def renumber(tree):
	leaves = sorted(tree.leaves())
	newleaves = {}
	shift = 0
	for n, a in enumerate(leaves):
		if a > 1 and a - 1 not in leaves:
			shift += 1
		newleaves[a] = n + shift
	for a in tree.treepositions("leaves"):
		tree[a] = newleaves[tree[a]]

def punctlower(tree):
	""" Find suitable constituent for punctuation marks and add it there;
	removal at previous location is up to the caller.  Based on rparse code.
	Initial candidate is the root node."""
	def lower(node, candidate):
		num = node.leaves()[0]
		for i, child in enumerate(sorted(candidate, key=lambda x: x.leaves())):
			if not isinstance(child[0], Tree): continue
			termdom = child.leaves()
			if num < min(termdom):
				print "moving", node, "under", candidate.node
				candidate.insert(i + 1, node)
				break
			elif num < max(termdom):
				lower(node, child)
				break
	for a in sorted(tree):
		if a.node.startswith("$"): #punctuation
			tree.remove(a)
			lower(a, tree)

def punctraise(tree):
	""" Trees in the Negra corpus have punctuation attached to the root;
	i.e., it is not part of the phrase-structure.  This function attaches
	punctuation nodes (that is, a POS tag with punctuation terminal) to an
	appropriate constituent """
	punct = [a for a in tree if a.node.startswith("$")]
	tree[:] = [a for a in tree if not a.node.startswith("$")]
	for node in reversed(punct):
		num = node.leaves()[0]
		phrasalnode = lambda x: len(x) and isinstance(x[0], Tree)
		for candidate in tree.subtrees(phrasalnode):
			# add punctuation mark next to biggest constituent which it borders
			#if any(num - 1 in borders(sorted(a.leaves())) for a in candidate):
			#if any(num - 1 == max(leaves(a)) for a in candidate):
			if any(num + 1 == min(leaves(a)) for a in candidate):
				candidate.append(node)
				break
		else: tree.append(node)

def balancedpunctraise(tree, sent):
	""" Move balanced punctuation marks " ' ( ) [ ] together in the same
	constituent. Based on rparse code. """
	match = { '"' : '"', '[':']', '(':')', "-":"-", "'" : "'" }
	punctmap = {}

	def preterm(idx): # sent idx -> tag + terminal node
		return tree.leaf_treeposition(tree.leaves().index(idx))[:-1]
	def termparent(idx): # sent idx -> parent node
		return tree[tree.leaf_treeposition(tree.leaves().index(idx))[:-2]]

	for terminal in sorted(leaves(tree)):
		if not tree[preterm(terminal)].node.startswith("$"): continue
		if sent[terminal] in punctmap:
			right = terminal
			left = punctmap[sent[right]]
			leftparent = termparent(left)
			rightparent = termparent(terminal)
			if max(leftparent.leaves()) == right - 1:
				node = tree[preterm(terminal)]
				del tree[preterm(terminal)]
				leftparent.append(node)
			elif min(rightparent.leaves()) == left + 1:
				node = tree[preterm(left)]
				del tree[preterm(left)]
				rightparent.insert(0, node)
			if sent[right] in punctmap: del punctmap[sent[right]]
		elif sent[terminal] in match: punctmap[match[sent[terminal]]] = terminal


from collections import deque
def leaves(node):
	"""Return the leaves of the tree. Non-recursive version."""
	queue, leaves = deque(node), []
	while queue:
		node = queue.popleft()
		if isinstance(node, Tree): queue.extend(node)
		else: leaves.append(node)
	return leaves

def puncttest():
	from treetransforms import slowfanout as fanout
	#corpus = NegraCorpusReader("..", "negra-corpus.export", headorder=False, encoding="iso-8859-1", movepunct=True)
	#corpust2 = NegraCorpusReader("..", "negra-corpus.export", headorder=False, encoding="iso-8859-1", removepunct=True).parsed_sents()
	#corpust3 = NegraCorpusReader("..", "negra-corpus.export", headorder=False, encoding="iso-8859-1").parsed_sents()
	#corpust3 = NegraCorpusReader("../rparse", "negraproc.export", headorder=False, encoding="iso-8859-1").parsed_sents()
	corpus = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1", movepunct=True)
	corpust2 = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1", removepunct=True).parsed_sents()
	corpust3 = range(len(corpust2))
	for n, tree, sent, t2, t3 in zip(count(), corpus.parsed_sents(), corpus.sents(), corpust2, corpust3):
		print n,
		for a, b in zip(tree.subtrees(lambda x: isinstance(x[0], Tree)), t2.subtrees(lambda x: len(x) and isinstance(x[0], Tree))):
			if fanout(a) != fanout(b):
				print " ".join(sent)
				print tree.pprint(margin=999)
				print t2.pprint(margin=999)
			assert fanout(a) == fanout(b), "%d %d\n%s\n%s" % (fanout(a), fanout(b), a.pprint(margin=999), b.pprint(margin=999))

def main():
	from grammar import canonicalize
	import sys, codecs
	# this fixes utf-8 output when piped through e.g. less
	# won't help if the locale is not actually utf-8, of course
	sys.stdout = codecs.getwriter('utf8')(sys.stdout)

	n = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1", headorder=True)
	nn = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1", headorder=True, unfold=True)
	#n = NegraCorpusReader("../rparse", "tigerproc.export", headorder=False)
	#nn = NegraCorpusReader("../rparse", "tigerproc.export", headorder=True, unfold=True)
	print "\nunfolded"
	correct = exact = d = 0
	nk = set(); mo = set()
	from nltk import FreqDist
	fnk = FreqDist(); fmo = FreqDist()
	for a,b,c in zip(n.parsed_sents()[:100], nn.parsed_sents()[:100], n.sents()[:100]):
		#if len(c) > 15: continue
		for x in a.subtrees(lambda n: n.node == "NP"):
			nk.update(y.node for y in x if function(y) == "NK")
			mo.update(y.node for y in x if function(y) == "MO")
			fnk.update(y.node for y in x if function(y) == "NK")
			fmo.update(y.node for y in x if function(y) == "MO")
		foldb = fold(b.copy(True))
		b1 = bracketings(canonicalize(a))
		b2 = bracketings(canonicalize(foldb))
		z = -1 #825
		if b1 != b2 or d == z:
			precision = len(set(b1) & set(b2)) / float(len(set(b1)))
			recall = len(set(b1) & set(b2)) / float(len(set(b2)))
			if precision != 1.0 or recall != 1.0 or d == z:
				print d, " ".join(":".join((str(n), a)) for n,a in enumerate(c))
				print "no match", precision, recall
				print len(b1), len(b2), "gold-fold", set(b2) - set(b1), "fold-gold", set(b1) - set(b2)
				print labelfunc(a)
				print foldb
				print b
			else: correct += 1
		else: exact += 1; correct += 1
		d += 1
	print "matches", correct, "/", d, 100 * correct / float(d), "%"
	print "exact", exact
	print
	print "nk & mo", " ".join(nk & mo)
	print "nk - mo", " ".join(nk - mo)
	print "mo - nk", " ".join(mo - nk)
	for x in nk & mo: print x, "as nk", fnk[x], "as mo", fmo[x]
if __name__ == '__main__': main()
