from nltk.corpus.reader.api import CorpusReader, SyntaxCorpusReader
from nltk.corpus.reader.util import read_regexp_block, StreamBackedCorpusView, concat
from nltk import Tree
import re

BOS = re.compile("^#BOS.*\n")
EOS = re.compile("^#EOS")
WORD, LEMMA, TAG, MORPH, FUNC, PARENT = range(6)

class NegraCorpusReader(SyntaxCorpusReader):
	def __init__(self, root, fileids, encoding=None, n=6, headorder=False, headfinal=False, reverse=False, unfold=False):
		""" n=6 for files with 6 columns, n=5 for files with 5 columns (no lemmas)
			headfinal: whether to put the head in final or in frontal position
			reverse: the head is made final/frontal by reversing everything before or after the head. 
				when true, the side on which the head is will be the reversed side"""
		if n == 5: self.d = 1	#fixme: autodetect this
		else: self.d = 0
		self.headorder = headorder; self.headfinal = headfinal; 
		self.reverse = reverse; self.unfold = unfold
		CorpusReader.__init__(self, root, fileids, encoding)
	def _parse(self, s):
		d = self.d
		def getchildren(parent, children):
			results = []; head = None
			for n,a in children[parent]:
				# n is the index in the block to record word indices
				if a[WORD][0] == "#": #nonterminal
					results.append(Tree(a[TAG-d], getchildren(a[WORD][1:], children)))
					results[-1].source = a
				else: #terminal
					results.append(Tree(a[TAG-d], [n]))
					results[-1].source = a
				if head is None and "HD" in a[FUNC-d].split("-"): head = results[-1]
			# roughly order constituents by order in sentence
			results.sort(key=lambda a: a.leaves()[0])
			if head is None or not self.headorder: return results
			head = results.index(head)
			# everything until the head is reversed and prepended to the rest,
			# leaving the head as the first element
			if self.headfinal:
				if self.reverse:
					# head final, reverse rhs: A B C^ D E => A B E D C^
					return results[:head] + results[head:][::-1]
				else:
					# head final, no reverse:  A B C^ D E => D E A B C^
					#return sorted(results[head+1:] + results[:head]) + results[head:head+1]
					# head final, reverse lhs:  A B C^ D E => E D A B C^
					return results[head+1:][::-1] + results[:head+1]
			else:
				if self.reverse:
					# head first, reverse lhs: A B C^ D E => C^ B A D E
					return results[:head+1][::-1] + results[head+1:]
				else:
					# head first, reverse rhs: A B C^ D E => C^ D E B A
					return results[head:] + results[:head][::-1]
		children = {}
		for n,a in enumerate(s):
			children.setdefault(a[PARENT-d], []).append((n,a))
		if self.unfold:
			return unfold(Tree("ROOT", getchildren("0", children)))
		return Tree("ROOT", getchildren("0", children))
	def _word(self, s):
		return [a[WORD] for a in s if a[WORD][0] != "#"]
	def _tag(self, s, ignore):
		return [(a[WORD], a[TAG-self.d]) for a in s if a[WORD][0] != "#"]
	def _read_block(self, stream):
		return [[line.split() for line in block.splitlines()[1:]] 
				for block in read_regexp_block(stream, BOS, EOS)]
			# didn't seem to help:
			#for b in map(lambda x: read_regexp_block(stream, BOS, EOS), range(1000)) for block in b]
	def blocks(self):
		""" Return a list of blocks, taken verbatim from the treebank file."""
		def reader(stream):
			result = read_regexp_block(stream, BOS, EOS)
			return [re.sub(BOS,"", result[0])] if result else []
		return concat([StreamBackedCorpusView(fileid, reader, encoding=enc)
					for fileid, enc in self.abspaths(self._fileids, True)])

# generalizations suggested by SyntaxGeneralizer of TigerAPI
# however, instead of renaming, we introduce unary productions
# POS tags
tonp = "NN NE NNE PNC PRF PDS PIS PPER PPOS PRELS PWS".split()
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

def unfold(tree):
	""" Unfold redundancies and perform other transformations introducing
	more hierarchy in the phrase-structure annotation. """
	def function(tree):
		if hasattr(tree, "source"): return tree.source[FUNC].split("-")[0]
	# introduce DPs
	for np in tree.subtrees(lambda n: n.node == "NP"):
		if np[0].node == "ART":
			np.node = "DP"
			np[:] = [np[0], Tree("NP", np[1:])]
	# un-flatten PPs
	for pp in tree.subtrees(lambda n: n.node == "PP"):
		if (len(pp) == 2 and pp[1].node != "NP" or len(pp) > 2):
			np = Tree("NP", pp[1:])
			if np[0].node == "ART": 
				np = Tree("DP", [np[0], Tree("NP", np[1:])])
			pp[:] = [pp[0], np]
	# introduce finite VP at S level, collect objects and modifiers
	addtovp = "HD AC DA MO NG OA OA2 OC OG PD VO".split()
	s = tree[[a.node for a in tree].index("S")]
	if "HD" in map(function, s):
		vp = Tree("VP", [a for a in s if function(a) in addtovp])
		s[:] = [a for a in s if function(a) not in addtovp] + [vp]
	# introduce new S level for discourse markers and complementizers
	newlevel = "DM CP".split()
	if s[0].source[FUNC] in newlevel:
		s[:] = [tree[0], Tree("S", tree[1:])]
	# introduce POS tag for particle verbs
	for a in tree.subtrees(lambda n: "SVP" in map(function, n)):
		hd = [x for x in a if function(x) == "HD"][0]
		svp = [x for x in a if function(x) == "SVP"][0]
		particleverb = Tree(hd.node, [hd, svp])
		a[:] = [particleverb if function(x) == "HD" else x for x in a if function(x) != "SVP"]
	# introduce SBAR level
	# TODO
	# introduce phrasal projections for single tokens
	for a in tree.treepositions("leaves"):
		tag = tree[a[:-1]]   # e.g. NN
		const = tree[a[:-2]] # e.g. S
		if tag.node in tagtoconst and const.node != tagtoconst[tag.node]:
			newconst = Tree(tagtoconst[tag.node], [tag])
			const[a[-2]] = newconst
	return tree

def fold(tree):
	""" Undo the transformations performed by unfold. Do not apply twice (would remove VPs which shouldn't be). """
	# remove DPs
	for dp in tree.subtrees(lambda n: n.node == "DP"):
			dp.node = dp[1].node
			dp[:] = dp[:1] + dp[1][:]
	# flatten PPs
	for pp in tree.subtrees(lambda n: n.node == "PP"):
		if len(pp) == 2 and pp[1].node == "NP":
			pp[:] = pp[:1] + pp[1][:]
	# merge extra S level
	s = tree[[a.node for a in tree].index("S")]
	if len(s) == 2 and s[1].node == "S":
		s = s[:1] + s[1][:]	
	# merge finite VP with S level
	s = tree[[a.node for a in tree].index("S")]
	vp = [a.node for a in s].index("VP")
	s[:] = s[:vp] + s[vp][:] + s[vp+1:]
	# remove constituents for particle verbs
	for a in tree.subtrees(lambda n: any(isinstance(x, Tree) and x.node == "VP" for x in n)):
		for n,b in enumerate(a):
			if isinstance(b, Tree) and b.node == "VP" and "PTKVZ" in (c.node for c in b):
				a[n:n+1] = b[:]
				# break out of the for loop because the list/tree was modified in-place
				# assumes there is only one particle verb in a VP
				break 
	# remove phrasal projections for single tokens
	for a in tree.treepositions("leaves"):
		tag = tree[a[:-1]]    # NN
		const = tree[a[:-2]]  # NP
		parent = tree[a[:-3]] # PP
		if len(const) == 1 and tag.node in tagtoconst and const.node == tagtoconst[tag.node]:
			parent[a[-3]] = tag
			del const
	return tree

def bracketings(tree):
        return [(a.node, tuple(sorted(a.leaves()))) for a in tree.subtrees(lambda t: t.height() > 2)]

def main():
	from rcgrules import canonicalize
	print "normal"
	n = NegraCorpusReader(".", "sample2\.export")
	for a in n.parsed_sents(): print a
	for a in n.tagged_sents(): print " ".join("/".join(x) for x in a)
	for a in n.sents(): print " ".join(a)
	for a in n.blocks(): print a
	print "\nunfolded"
	nn = NegraCorpusReader(".", "sample2\.export", unfold=True)
	for a,b,c in zip(n.parsed_sents(), nn.parsed_sents(), n.sents()):
		print b
		print " ".join(":".join((str(n),a)) for n,a in enumerate(c))
		foldb = fold(b)
		if bracketings(canonicalize(a)) == bracketings(canonicalize(foldb)): 
			print "match"
		else: 
			b1 = bracketings(canonicalize(a))
			b2 = bracketings(canonicalize(foldb))
			print "no match", len(set(b1) & set(b2)) / float(len(set(b1))), len(set(b1) & set(b2)) / float(len(set(b2)))
			print len(b1), len(b2), set(b2) - set(b1), set(b1) - set(b2)

if __name__ == '__main__': main()
