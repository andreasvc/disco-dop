from nltk.corpus.reader.api import CorpusReader, SyntaxCorpusReader
from nltk.corpus.reader.util import read_regexp_block
from nltk import Tree
import re

BOS = re.compile("^#BOS")
EOS = re.compile("^#EOS")
WORD, LEMMA, TAG, MORPH, FUNC, PARENT = range(6)

class NegraCorpusReader(SyntaxCorpusReader):
	def __init__(self, root, fileids, encoding=None, n=6):
		if n == 5: self.d = 1
		else: self.d = 0
		CorpusReader.__init__(self, root, fileids, encoding)
	def _parse(self, s):
		def getchildren(parent):
			for n,a in ((n,a) for n,a in enumerate(s) if a[PARENT-self.d] == parent):
				if a[WORD][0] == "#":
					yield Tree(a[TAG-self.d], getchildren(a[WORD][1:]))
				else:
					#yield Tree(a[TAG], [a[WORD]])
					yield Tree(a[TAG-self.d], [n])
		return Tree("ROOT", getchildren("0"))
	def _word(self, s):
		return [a[WORD] for a in s if a[WORD][0] != "#"]
	def _tag(self, s, ignore):
		return [(a[WORD], a[TAG-self.d]) for a in s if a[WORD][0] != "#"]
	def _read_block(self, stream):
		return [[line.split() for line in block.splitlines()[1:]] for block in read_regexp_block(stream, BOS, EOS)]

def main():
	n = NegraCorpusReader(".", "sample2\.export")
	print n.parsed_sents()
	print n.tagged_sents()
	print n.sents()

if __name__ == '__main__': main()
