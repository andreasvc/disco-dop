# -*- coding: UTF-8 -*-
from negra import NegraCorpusReader
from rcgrules import srcg_productions, dop_srcg_rules, induce_srcg, enumchart, extractfragments
from nltk import FreqDist, Tree
from nltk.metrics import precision, recall, f_measure, accuracy
from itertools import islice, chain
from math import log, e
from pprint import pprint
import cPickle
try: import pyximport
except:
	from plcfrs import parse, mostprobableparse
else:
	pyximport.install()
	from plcfrs_cython import parse, mostprobableparse

def rem_marks(tree):
	for a in tree.subtrees(lambda x: "_" in x.node):
		a.node = a.node.rsplit("_", 1)[0]
	for a in tree.treepositions('leaves'):
		tree[a] = int(tree[a])
	return tree

def bracketings(tree):
	# sorted or not?
	return [(a.node, tuple(sorted(a.leaves()))) for a in tree.subtrees(lambda t: t.height() > 2)]

def harmean(seq):
	try: return float(len([a for a in lps if a])) / sum(1/a if a else 0 for a in seq)
	except: return "zerodiv"

def mean(seq):
	return sum(seq) / float(len(seq)) if seq else "zerodiv"

def export(tree, sent, n):
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
corpus = NegraCorpusReader(".", "sample2.export")
#corpus = NegraCorpusReader("../rparse", ".*\.export", n=5)
trees, sents = corpus.parsed_sents()[:3600], corpus.sents()[:3600]

dop = True
grammar = induce_srcg(list(trees), sents, h=1, v=1)
if dop: dopgrammar = dop_srcg_rules(list(trees), list(sents))

nodes = sum(len(list(a.subtrees())) for a in trees)
if dop: print "DOP model based on", len(trees), "sentences,", nodes, "nodes, max", nodes*8, "nonterminals"
#for a,b in extractfragments(trees).items():
#	print a,b
#exit()
#trees, sents, blocks = corpus.parsed_sents()[3600:], corpus.tagged_sents()[3600:], corpus.blocks()[3600:]
trees, sents, blocks = corpus.parsed_sents(), corpus.tagged_sents(), corpus.blocks()
maxlen = 99
maxsent = 360
viterbi = True
sample = False
n = 1
nsent = 0
exact, exacts = 0, 0
snoparse, dnoparse = 0, 0
lp, lr, lf = [], [], []
lps, lrs, lfs = [], [], []
sresults = []
dresults = []
gold = []
gsent = []
for tree, sent, block in zip(trees, sents, blocks):
	if len(sent) > maxlen: continue
	nsent += 1
	if nsent > maxsent: break
	print "%d. [len=%d] %s" % (nsent, len(sent), " ".join(a[0]+"/"+a[1] for a in sent))
	goldb = set(bracketings(tree))
	gold.append(block)
	gsent.append(sent)
	print "SRCG:",
	chart, start = parse([a[0] for a in sent], grammar, tags=[a[1] for a in sent], start='ROOT', viterbi=True)
	for result, prob in enumchart(chart, start) if chart else ():
		result = rem_marks(Tree.convert(result))
		result.un_chomsky_normal_form()
		print "p =", e**prob,
		if result == tree:
			print "exact match"
			exacts += 1
			prec, rec, f1 = 1.0, 1.0, 1.0
		else: 
			candb = set(bracketings(result))
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
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
	chart, start = parse([a[0] for a in sent], dopgrammar, tags=[a[1] for a in sent], start='ROOT', viterbi=viterbi, n=n)
	print "viterbi =", viterbi, "n=%d" % n if viterbi else '',
	for dresult, prob in mostprobableparse(chart, start,n=10000,sample=sample).items() if chart else ():
		print "p =", prob,
		dresult = rem_marks(Tree.convert(dresult))
		dresult.un_chomsky_normal_form()
		if dresult == tree:
			print "exact match"
			exact += 1
			prec, rec, f1 = 1.0, 1.0, 1.0
		else: 
			candb = set(bracketings(dresult))
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			print "labeled precision", prec, "recall", rec, "f-measure", f1
			print dresult.pprint(margin=1000)
		dresults.append(dresult)
		break
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
	print

open("test.srcg", "w").writelines("%s\n" % export(a,b,n) for n,(a,b) in enumerate(zip(sresults, gsent)))
open("test.dop", "w").writelines("%s\n" % export(a,b,n) for n,(a,b) in enumerate(zip(dresults, gsent)))
open("test.gold", "w").writelines("#BOS %d\n%s\n#EOS %d\n" % (n,a,n) for n,a in enumerate(gold))
print "SRCG:"
print "exact match", lps.count(1.0), "/", nsent, "=", exacts / float(nsent)
print "harm lp", harmean(lps), "lr", harmean(lrs), "lf1", harmean(lfs)
print "mean lp", mean(lps), "lr", mean(lrs), "lf1", mean(lfs)
print "coverage", (nsent - snoparse), "/", nsent, "=", (nsent - snoparse) / float(nsent)
print
print "DOP:"
print "exact match", lp.count(1.0), "/", nsent, "=", exact / float(nsent)
print "harm lp", harmean(lp), "lr", harmean(lr), "lf1", harmean(lf)
print "mean lp", mean(lp), "lr", mean(lr), "lf1", mean(lf)
print "coverage", (nsent - dnoparse), "/", nsent, "=", (nsent - dnoparse) / float(nsent)
