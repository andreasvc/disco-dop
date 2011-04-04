# -*- coding: UTF-8 -*-
from negra import NegraCorpusReader
from rcgrules import srcg_productions, dop_srcg_rules, induce_srcg, enumchart, extractfragments, splitgrammar
from nltk import FreqDist, Tree
from nltk.metrics import precision, recall, f_measure, accuracy
from itertools import islice, chain
from math import log, e
from functools import partial
from pprint import pprint
import cPickle
import re
try: 
	from plcfrs_cython import parse, mostprobableparse
except:
	from plcfrs import parse, mostprobableparse
	print "running non-cython code"
import pyximport; pyximport.install()
from estimates import getestimates, getoutside

def rem_marks(tree):
	for a in tree.subtrees(lambda x: "_" in x.node):
		a.node = a.node.rsplit("_", 1)[0]
	for a in tree.treepositions('leaves'):
		tree[a] = int(tree[a])
	return tree

def escapetree(tree):
	result = Tree(re.sub("\$\(", "$[", tree))
	for a in result.subtrees(lambda x: x.node == "$["):
		a.node = "$("
	return result
	
def bracketings(tree):
	# sorted or not?
	return [(a.node, tuple(sorted(a.leaves()))) for a in tree.subtrees(lambda t: t.height() > 2)]

def harmean(seq):
	try: return float(len([a for a in seq if a])) / sum(1/a if a else 0 for a in seq)
	except: return "zerodiv"

def mean(seq):
	return sum(seq) / float(len(seq)) if seq else "zerodiv"

def export(tree, sent, n):
	""" Convert a tree with indices as leafs and a sentence with the
	corresponding non-terminals to a single string in Negra's export format """
	result = ["#BOS %d" % n]
	wordsandpreterminals = tree.treepositions('leaves') + [a[:-1] for a in tree.treepositions('leaves')]
	nonpreterminals = list(sorted([a for a in tree.treepositions() if a not in wordsandpreterminals and a != ()], key=len, reverse=True))
	wordids = dict((tree[a], a) for a in tree.treepositions('leaves'))
	for i, word in enumerate(sent):
		idx = wordids[i]
		result.append("\t".join((word[0],
				tree[idx[:-1]].node, 
				"--", "--", 
				str(500+nonpreterminals.index(idx[:-2]) if len(idx) > 2 else 0))))
	for idx in nonpreterminals:
		result.append("\t".join(("#%d" % (500 + nonpreterminals.index(idx)),
				tree[idx].node,
				"--", "--",
				str(500+nonpreterminals.index(idx[:-1]) if len(idx) > 1 else 0))))
	result.append("#EOS %d" % n)
	return "\n".join(result)

# Tiger treebank version 2 sample:
# http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/annotation/sample2.export
corpus = NegraCorpusReader(".", "sample2\.export")
#corpus = NegraCorpusReader("../rparse", "tigerproc.export", n=5)
trees, sents = corpus.parsed_sents()[:3600], corpus.sents()[:3600]

dop = True
grammar = splitgrammar(induce_srcg(list(trees), sents, h=1, v=1))
#trees=list(trees)
#for a in trees: a.chomsky_normal_form(vertMarkov=1, horzMarkov=1)
if dop: dopgrammar = splitgrammar(dop_srcg_rules(list(trees), list(sents), normalize=True, shortestderiv=False))
print "getting outside estimates"
outside = getestimates(dopgrammar, 15, dopgrammar[4]["ROOT"])
print "done"

for a in trees: a.chomsky_normal_form()
nodes = sum(len(list(a.subtrees())) for a in trees)
nts = nodes - len(trees) + len(set(b.node for a in trees for b in a.subtrees()))
if dop: print "DOP model based on", len(trees), "sentences,", nodes, "nodes,", nts, "nonterminals"
#for a,b in extractfragments(trees).items():
#	print a,b
#exit()
#trees, sents, blocks = corpus.parsed_sents()[3600:], corpus.tagged_sents()[3600:], corpus.blocks()[3600:]
trees, sents, blocks = corpus.parsed_sents(), corpus.tagged_sents(), corpus.blocks()
maxlen = 45
maxsent = 360
viterbi = True
sample = False
n = 1    #number of top-derivations to parse (should become n-best)
m = 10000  #number of derivations to sample/enumerate
nsent = 0
exact, exacts = 0, 0
snoparse, dnoparse = 0, 0
lp, lr, lf = [], [], []
lps, lrs, lfs = [], [], []
sresults = []
dresults = []
gold = []
gsent = []
gconst = 0
scorrect = 0
dcorrect = 0
sconst = 0
dconst = 0
for tree, sent, block in zip(trees, sents, blocks):
	if len(sent) > maxlen: continue
	if nsent >= maxsent: break
	nsent += 1
	print "%d. [len=%d] %s" % (nsent, len(sent), " ".join(a[0]+"/"+a[1] for a in sent))
	goldb = set(bracketings(tree))
	gconst += len(goldb)
	gold.append(block)
	gsent.append(sent)
	print "SRCG:",
	chart, start = parse([a[0] for a in sent], grammar, tags=[a[1] for a in sent], start=grammar[4]['ROOT'], viterbi=True)
	for result, prob in enumchart(chart, start, grammar[5]) if chart else ():
		result = rem_marks(escapetree(result))
		result.un_chomsky_normal_form()
		print "p =", e**prob,
		candb = set(bracketings(result))
		prec = precision(goldb, candb)
		rec = recall(goldb, candb)
		f1 = f_measure(goldb, candb)
		scorrect += len(candb & goldb)
		sconst += len(candb)
		if result == tree:
			print "exact match"
			exacts += 1
		else:
			print "labeled precision", prec, "recall", rec, "f-measure", f1
			print result.pprint(margin=1000)
		sresults.append(result)
		break
	else:
		print "no parse"
		result = Tree("ROOT", [Tree("PN", [i]) for i in range(len(sent))])
		candb = set(bracketings(result))
		prec = precision(goldb, candb)
		rec = recall(goldb, candb)
		f1 = f_measure(goldb, candb)
		snoparse += 1
		sresults.append(result)
	lps.append(prec)
	lrs.append(rec)
	lfs.append(f1)
	if not dop: continue
	print "DOP:",
	estimate = partial(getoutside, outside, 15, len(sent))
	chart, start = parse([a[0] for a in sent], dopgrammar, tags=[a[1] for a in sent], start=dopgrammar[4]['ROOT'], viterbi=viterbi, n=n, estimate=estimate)
	print "viterbi =", viterbi, "n=%d" % n if viterbi else '',
	if chart:
		mpp = mostprobableparse(chart, start, dopgrammar[5], n=m, sample=sample).items()
		dresult, prob = max(mpp, key=lambda x: x[1])
		#for a,b in mpp: print a,b
		print "p =", prob,
		dresult = rem_marks(escapetree(dresult))
		dresult.un_chomsky_normal_form()
		candb = set(bracketings(dresult))
		prec = precision(goldb, candb)
		rec = recall(goldb, candb)
		f1 = f_measure(goldb, candb)
		dcorrect += len(candb & goldb)
		dconst += len(candb)
		if dresult == tree:
			print "exact match"
			exact += 1
		else: 
			print "labeled precision", prec, "recall", rec, "f-measure", f1
			print dresult.pprint(margin=1000)
		dresults.append(dresult)
	else:
		print "no parse"
		dresult = Tree("ROOT", [Tree("PN", [i]) for i in range(len(sent))])
		candb = set(bracketings(dresult))
		prec = precision(goldb, candb)
		rec = recall(goldb, candb)
		f1 = f_measure(goldb, candb)
		dnoparse += 1
		dresults.append(dresult)
	lp.append(prec) 
	lr.append(rec)
	lf.append(f1)
	try:
		print "srcg em", lfs.count(1.0) / float(len(lfs)), "lp", scorrect / float(sconst), "lr", scorrect / float(gconst), 
		print "lf1", harmean((scorrect / float(sconst), scorrect / float(gconst)))
		print "dop  em", lf.count(1.0) / float(len(lf)), "lp", dcorrect / float(dconst), "lr", dcorrect / float(gconst), 
		print "lf1", harmean((dcorrect / float(dconst), dcorrect / float(gconst)))
	except ZeroDivisionError: print "zerodiv"
	print

open("test.srcg", "w").writelines("%s\n" % export(a,b,n) for n,(a,b) in enumerate(zip(sresults, gsent)))
open("test.dop", "w").writelines("%s\n" % export(a,b,n) for n,(a,b) in enumerate(zip(dresults, gsent)))
open("test.gold", "w").writelines("#BOS %d\n%s\n#EOS %d\n" % (n,a,n) for n,a in enumerate(gold))
print "SRCG:"
print "exact match", lfs.count(1.0), "/", len(lfs), "=", lfs.count(1.0) / float(len(lfs)) if lfs else "zerodiv"
print "lp", scorrect / float(sconst), "lr", scorrect / float(gconst), 
print "lf1", harmean((scorrect / float(sconst), scorrect / float(gconst)))
print "coverage", (len(lps) - snoparse), "/", len(lps), "=", (len(lps) - snoparse) / float(len(lps)) if lps else "zerodiv"
print
print "DOP:"
print "exact match", lf.count(1.0), "/", len(lf), "=", lf.count(1.0) / float(len(lf)) if lf else "zerodiv"
print "lp", dcorrect / float(dconst), "lr", dcorrect / float(gconst), 
print "lf1", harmean((dcorrect / float(dconst), dcorrect / float(gconst)))
print "coverage", (len(lp) - dnoparse), "/", len(lp), "=", (len(lp) - dnoparse) / float(len(lp)) if lp else "zerodiv"
