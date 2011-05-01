# -*- coding: UTF-8 -*-
from negra import NegraCorpusReader, fold, unfold
from rcgrules import srcg_productions, dop_srcg_rules, induce_srcg, enumchart, extractfragments, splitgrammar, binarizetree
from treetransforms import collinize, un_collinize
from nltk import FreqDist, Tree
from nltk.metrics import precision, recall, f_measure, accuracy
from itertools import islice, chain
from math import log, exp
from functools import partial
from pprint import pprint
import cPickle
import re, time
#try:
#from plcfrs_cython import parse, mostprobableparse
#except Exception as e:
#	print e
from plcfrs import parse, mostprobableparse
#try: import plcfrs.cython; assert plcfrs.cython
#except: print "running non-cython code"
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

def alterbinarization(tree):
	# converts the binarization of rparse to the format that NLTK expects
	# S1 is the constituent, CS1 the parent, CARD1 the current sibling/child
	# @^S1^CS1-CARD1X1   -->  S1|<CARD1>^CS1
	#how to optionally add \2 if nonempty?
	tree = re.sub("@\^([A-Z.,()$]+)\d+(\^[A-Z.,()$]+\d+)*(?:-([A-Z.,()$]+)\d+)*X\d+", r"\1|<\3>", tree)
	# remove arity markers
	tree = re.sub(r"([A-Z.,()$]+)\d+", r"\1", tree)
	tree = re.sub("VROOT", r"ROOT", tree)
	assert "@" not in tree
	return tree

def testgrammar(grammar):
	for a,b in grammar.bylhs.items():
		if abs(sum(exp(-w) for rule,w in b) - 1.0) > 0.01:
			print "Does not sum to 1:", grammar.tolabel[a], sum(exp(-w) for rule,w in b)
			break
	else: print "All left hand sides sum to 1"

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

def read_rparse_grammar(file):
	result = []
	for line in open(file):
		yf = eval(line[line.index("[[["):].replace("false","0").replace("true","1"))[0]
		line = line[:line.index("[[[")].split()
		line.pop(0) #freq?
		prob, lhs = line.pop(0).split(":")
		line.pop(0) # -->
		result.append(((tuple([lhs] + line), tuple(map(tuple, yf))), log(float(prob))))
	return result

# Tiger treebank version 2 sample:
# http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/annotation/sample2.export
corpus = NegraCorpusReader(".", "sample2\.export")
#corpus = NegraCorpusReader("../rparse", "tiger3600proc.export", n=5, headfinal=True, reverse=False)
#corpus = NegraCorpusReader("../rparse", "tigerproc.export", n=5, headorder=False)
trees, sents, blocks = corpus.parsed_sents()[:7200], corpus.sents()[:7200], corpus.blocks()[:7200]
print "read training corpus"

srcg = True
dop = True
unfolded = False
maxlen = 99
trees, sents, blocks = zip(*[sent for sent in zip(trees, sents, blocks) if len(sent[1]) <= maxlen])
factor = "right"
v = 1
h = 1
for a,b in zip(trees, sents):
	pass
	#unfold(a); unfolded = True
	#foo = srcg_productions(a, b)	#adds arity markers
	#collinize(a, factor=factor, horzMarkov=h)
	collinize(a, factor=factor, vertMarkov=v-1, horzMarkov=h)
	#a.chomsky_normal_form(factor="left", vertMarkov=v-1, horzMarkov=1)
	#a.chomsky_normal_form()
#trees = [binarizetree(tree.freeze()) for n, tree in enumerate(trees)]
#print "binarized corpus %s branching with v = %d, h = %d" % (factor, v, h)

#trees = trees[:10]; sents = sents[:10]
#seen = set()
#for n, (tree, sent) in enumerate(zip(trees, sents)):
#	rules = [a for a,b in induce_srcg([tree], [sent]) if a not in seen]
#	seen.update(rules)
#	match = False
#	for rule,yf in rules:
#		if len(rule) == 2 and rule[1] != "Epsilon":
#			print rule[0], "-->", rule[1], "\t\t", [list(a) for a in yf]
#			match = True
#	if False and match:
#		print tree
#		print n, sent
#for (rule,yf),w in sorted(induce_srcg(list(trees), sents), key=lambda x:-x[1]):
#	if len(rule) != 2 and rule[1] != "Epsilon"#:
#		print exp(w), rule[0], "-->", " ".join(rule[1:]), "\t\t", [list(a) for a in yf]
#exit()

if srcg:
	grammar = induce_srcg(list(trees), sents)
	#for (rule,yf),w in sorted(grammar, key=lambda x: x[0][0][0]):
	#	if len(rule) == 2 and rule[1] != "Epsilon":
	#		print exp(w), rule[0], "-->", " ".join(rule[1:]), "\t\t", [list(a) for a in yf]
	#grammar = read_rparse_grammar("../rparse/bin3600")
	lhs = set(rule[0] for (rule,yf),w in grammar)
	print "SRCG based on", len(trees), "sentences"
	l = len(grammar)
	print "labels:", len(set(rule[a] for (rule,yf),w in grammar for a in range(3) if len(rule) > a)), "of which nonterminals:", len(set(rule[0] for (rule,yf),w in grammar if rule[1] == "Epsilon")) or len(set(rule[a] for (rule,yf),w in grammar for a in range(1,3) if len(rule) > a and rule[a] not in lhs))
	print "max arity:", max((len(yf), rule, yf, w) for (rule,yf),w in grammar)
	grammar = splitgrammar(grammar)
	ll=sum(len(b) for a,b in grammar.lexical.items())
	print "clauses:",l, "lexical clauses:", ll, "non-lexical clauses:", l - ll
	testgrammar(grammar)

trees=list(trees)
print "induced srcg grammar"
if dop:
	dopgrammar = dop_srcg_rules(list(trees), list(sents), normalize=False, shortestderiv=False)
	nodes = sum(len(list(a.subtrees())) for a in trees)
	l = len(dopgrammar)
	print "labels:", len(set(rule[a] for (rule,yf),w in dopgrammar for a in range(3) if len(rule) > a)), "of which nonterminals:", len(set(rule[0] for (rule,yf),w in dopgrammar if rule[1] == "Epsilon")) or len(set(rule[a] for (rule,yf),w in dopgrammar for a in range(1,3) if len(rule) > a and rule[a] not in lhs))
	print "max arity:", max((len(yf), rule, yf, w) for (rule,yf),w in dopgrammar)
	dopgrammar = splitgrammar(dopgrammar)
	ll=sum(len(b) for a,b in dopgrammar.lexical.items())
	print "clauses:",l, "lexical clauses:", ll, "non-lexical clauses:", l - ll
	testgrammar(dopgrammar)
	print "DOP model based on", len(trees), "sentences,", nodes, "nodes,", len(dopgrammar.toid), "nonterminals"
#print "getting outside estimates"
#begin = time.clock()
#outside = getestimates(dopgrammar, maxlen, dopgrammar.toid["ROOT"])
#print "done. time elapsed: ", time.clock() - begin,
#cPickle.dump(outside, open("outside.pickle", "wb"))
#outside = cPickle.load(open("outside.pickle", "rb"))
#print "pickled"

#for a,b in extractfragments(trees).items():
#	print a,b
#exit()
trees, sents, blocks = corpus.parsed_sents(), corpus.tagged_sents(), corpus.blocks()
#corpus = NegraCorpusReader("../rparse", "tigerproc.export", n=5)
#trees, sents, blocks = corpus.parsed_sents()[7200:9000], corpus.tagged_sents()[7200:9000], corpus.blocks()[7200:9000]
print "read test corpus"
maxsent = 360
maxlen = 15
viterbi = False
sample = True
n = 0      #number of top-derivations to parse (1 for 1-best, 0 to parse exhaustively)
m = 100000  #number of derivations to sample/enumerate
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
estimate = lambda a,b: 0.0
for tree, sent, block in zip(trees, sents, blocks):
	if len(sent) > maxlen: continue
	if nsent >= maxsent: break
	nsent += 1
	print "%d. [len=%d] %s" % (nsent, len(sent), " ".join(a[0]+"/"+a[1] for a in sent))
	goldb = set(bracketings(tree))
	gconst += len(goldb)
	gold.append(block)
	gsent.append(sent)
	if srcg:
		print "SRCG:",
		chart, start = parse([a[0] for a in sent], grammar, tags=[a[1] for a in sent], start=grammar.toid['ROOT'], viterbi=True)
		#print
	else: chart = ()
	for a in chart: chart[a].sort(key=lambda x: x[0])
	for result, prob in enumchart(chart, start, grammar.tolabel) if chart else ():
		#result = rem_marks(escapetree(alterbinarization(result)))
		print result
		result = rem_marks(escapetree(result))
		un_collinize(result)
		if unfolded: fold(result)
		print "p =", exp(-prob),
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
		if srcg: print "no parse"
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
	if dop:
		print "DOP:",
		#estimate = partial(getoutside, outside, maxlen, len(sent))
		chart, start = parse([a[0] for a in sent], dopgrammar, tags=[a[1] for a in sent], start=dopgrammar.toid['ROOT'], viterbi=viterbi, n=n, estimate=estimate)
		#print "viterbi =", viterbi, "n=%d" % n if viterbi else '',
	else: chart = ()
	if chart:
		mpp = mostprobableparse(chart, start, dopgrammar.tolabel, n=m, sample=sample, both=False).items()
		dresult, prob = max(mpp, key=lambda x: x[1])
		#for a,b in mpp: print a,b
		print "p =", prob, "(%d parsetrees)" % len(mpp)
		dresult = rem_marks(escapetree(dresult))
		un_collinize(dresult)
		if unfolded: fold(dresult)
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
		if dop: print "no parse"
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
	if srcg:
		try:
			print "srcg em", lfs.count(1.0) / float(len(lfs)), "lp", scorrect / float(sconst), "lr", scorrect / float(gconst),
			print "lf1", harmean((scorrect / float(sconst), scorrect / float(gconst)))
		except ZeroDivisionError: print "zerodiv"
	if dop:
		try:
			print "dop  em", lf.count(1.0) / float(len(lf)), "lp", dcorrect / float(dconst), "lr", dcorrect / float(gconst),
			print "lf1", harmean((dcorrect / float(dconst), dcorrect / float(gconst)))
		except ZeroDivisionError: print "zerodiv"
	print

if srcg: open("test1.srcg", "w").writelines("%s\n" % export(a,b,n) for n,(a,b) in enumerate(zip(sresults, gsent)))
if dop: open("test1.dop", "w").writelines("%s\n" % export(a,b,n) for n,(a,b) in enumerate(zip(dresults, gsent)))
open("test1.gold", "w").writelines("#BOS %d\n%s\n#EOS %d\n" % (n,a,n) for n,a in enumerate(gold))
if srcg:
	print "SRCG:"
	print "exact match", lfs.count(1.0), "/", len(lfs), "=", lfs.count(1.0) / float(len(lfs)) if lfs else "zerodiv"
	if sconst and gconst:
		print "lp", scorrect / float(sconst), "lr", scorrect / float(gconst),
		print "lf1", harmean((scorrect / float(sconst), scorrect / float(gconst)))
	print "coverage", (len(lps) - snoparse), "/", len(lps), "=", (len(lps) - snoparse) / float(len(lps)) if lps else "zerodiv"
	print
if dop:
	print "DOP:"
	print "exact match", lf.count(1.0), "/", len(lf), "=", lf.count(1.0) / float(len(lf)) if lf else "zerodiv"
	print "lp", dcorrect / float(dconst), "lr", dcorrect / float(gconst),
	print "lf1", harmean((dcorrect / float(dconst), dcorrect / float(gconst)))
	print "coverage", (len(lp) - dnoparse), "/", len(lp), "=", (len(lp) - dnoparse) / float(len(lp)) if lp else "zerodiv"
