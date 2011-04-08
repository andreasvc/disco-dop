from nltk.corpus.reader.api import CorpusReader, SyntaxCorpusReader
from nltk.corpus.reader.util import read_regexp_block, StreamBackedCorpusView, concat
from nltk import Tree
import re

BOS = re.compile("^#BOS.*\n")
EOS = re.compile("^#EOS")
WORD, LEMMA, TAG, MORPH, FUNC, PARENT = range(6)

class NegraCorpusReader(SyntaxCorpusReader):
	def __init__(self, root, fileids, encoding=None, n=6):
		if n == 5: self.d = 1
		else: self.d = 0
		CorpusReader.__init__(self, root, fileids, encoding)
	def _parse(self, s):
		d = self.d
		def getchildren(parent, children):
			results = []; head = None
			for n,a in children[parent]:
				# n is the index in the block to record word indices
				if a[WORD][0] == "#":
					results.append(Tree(a[TAG-d], getchildren(a[WORD][1:], children)))
				else:
					results.append(Tree(a[TAG-d], [n]))
				if head is None and "HD" in a[FUNC-d].split("-"): head = results[-1]
			if head is None: return results
			# roughly order by order in sentence
			results.sort(key=lambda a: a.leaves()[0])
			head = results.index(head)
			# everything until the head is reversed and prepended to the rest,
			# leaving the head as the first element
			# head final:
			return results[head+1:][::-1] + results[:head+1]
			#return (results[:head+1][::-1] + results[head+1:])[::-1]
			#return results[:head] + results[head:][::-1]
		children = {}
		for n,a in enumerate(s):
			children.setdefault(a[PARENT-d], []).append((n,a))
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
		def reader(stream):
			result = read_regexp_block(stream, BOS, EOS)
			return [re.sub(BOS,"", result[0])] if result else []
	        return concat([StreamBackedCorpusView(fileid, reader, encoding=enc)
        	               for fileid, enc in self.abspaths(self._fileids, True)])

def main():
	n = NegraCorpusReader(".", "sample2\.export")
	for a in n.parsed_sents(): print a
	for a in n.tagged_sents(): print a
	for a in n.sents(): print a
	for a in n.blocks(): print a

if __name__ == '__main__': main()
