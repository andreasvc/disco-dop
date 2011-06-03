# -*- coding: UTF-8 -*-
from negra import NegraCorpusReader, fold, unfold
from rcgrules import srcg_productions, dop_srcg_rules, induce_srcg, enumchart,\
		extractfragments, binarizetree, ranges, export, \
		read_rparse_grammar, mean, harmean, testgrammar, bracketings, \
		printbrackets, rem_marks, alterbinarization, varstoindices, \
		read_bitpar_grammar
from rcgrules import newsplitgrammar as splitgrammar
from treetransforms import collinize, un_collinize
from nltk import FreqDist, Tree
from nltk.metrics import precision, recall, f_measure, accuracy
from collections import defaultdict
from itertools import islice, chain, count
from operator import itemgetter
from functools import partial
from pprint import pprint
from math import log, exp
import cPickle, re, time, codecs
#import plac
#from estimates import getestimates, getoutside
from kbest import lazykbest
#from plcfrs_cython import mostprobableparse
#from plcfrs import parse
try:
	from plcfrs_cython import parse, mostprobableparse, mostprobablederivation, filterchart
	print "running cython"
except: from plcfrs import parse, mostprobableparse; print "running non-cython"

def read_penn_format(corpus, maxlen=15, n=7200):
	trees = [a for a in islice((Tree(a) for a in codecs.open(corpus, encoding='iso-8859-1')), n)
				if len(a.leaves()) <= maxlen]
	sents = [a.pos() for a in trees]
	for tree in trees:
		for n, a in enumerate(tree.treepositions('leaves')):
			tree[a] = n
	return trees, sents

def terminals(tree, sent):
	tree = Tree(tree)
	tree.node = "VROOT"
	for a, (w, t) in zip(tree.treepositions('leaves'), sent):
		tree[a] = w
	return tree.pprint(margin=999)

def main(
	#parameters. parameters. PARAMETERS!!
	srcg = True,
	dop = True,
	unfolded = False,
	maxlen = 99,	# max number of words for sentences in training & test corpus
	bintype = "collinize", # choices: collinize, nltk, optimal
	estimator = "dop1", # choices: dop1, ewe, shortest, sl-dop
	factor = "right",
	v = 1,
	h = 1,
	minMarkov = 3,
	tailmarker = "$",
	maxsent = 360,	# number of sentences to parse
	viterbi = True,
	sample = False,
	both = False,
	arity_marks = True,
	arity_marks_before_bin = False,
	interpolate = 1.0,
	wrong_interpolate = False,
	n = 0,     #number of top-derivations to parse (1 for 1-best, 0 to parse exhaustively)
	m = 1000  #number of derivations to sample/enumerate
	):
	# Tiger treebank version 2 sample:
	# http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/annotation/sample2.export
	corpus = NegraCorpusReader(".", "sample2\.export", encoding="iso-8859-1"); maxlen = 99
	#corpus = NegraCorpusReader("../rparse", "tiger3600proc.export", headfinal=True, headreverse=False)
	#corpus = NegraCorpusReader("../rparse", "tigerproc.export",
	#		headorder=(bintype=="collinize"), headfinal=True,
	#		headreverse=False, unfold=unfolded)

	assert bintype in ("optimal", "collinize", "nltk")
	assert estimator in ("dop1", "ewe", "shortest", "sl-dop")
	trees, sents, blocks = corpus.parsed_sents()[:7200], corpus.sents()[:7200], corpus.blocks()[:7200]
	trees, sents, blocks = zip(*[sent for sent in zip(trees, sents, blocks) if len(sent[1]) <= maxlen])
	# parse training corpus as a "soundness check"
	test = corpus.parsed_sents(), corpus.tagged_sents(), corpus.blocks()
	#test = NegraCorpusReader("../rparse", "tigerproc.export")
	#test = test.parsed_sents()[7200:9000], test.tagged_sents()[7200:9000], test.blocks()[7200:9000]
	print "read training & test corpus"
	if arity_marks_before_bin: [srcg_productions(a, b) for a, b in zip(trees, sents)]
	if bintype == "collinize":
		bintype += " %s h=%d v=%d %s markovize rank > %d" % (factor, h, v, "tailmarker" if tailmarker else '', minMarkov)
		[collinize(a, factor=factor, vertMarkov=v-1, horzMarkov=h, tailMarker=tailmarker, minMarkov=minMarkov) for a in trees]
	if bintype == "nltk":
		bintype += " %s h=%d v=%d" % (factor, h, v)
		for a in trees: a.chomsky_normal_form(factor="left", vertMarkov=v-1, horzMarkov=1)
	if bintype == "optimal": trees = [binarizetree(tree.freeze()) for tree in trees]
	print "binarized", bintype

	#trees = trees[:10]; sents = sents[:10]
	seen = set()
	v = set(); e = {}; weights = {}
	for n, (tree, sent) in enumerate(zip(trees, sents)):
		rules = [(a,b) for a,b in induce_srcg([tree], [sent]) if a not in seen]
		seen.update(map(lambda (a,b): a, rules))
		match = False
		for (rule,yf), w in rules:
			if len(rule) == 2 and rule[1] != "Epsilon":
				#print n, rule[0], "-->", rule[1], "\t\t", [list(a) for a in yf]
				match = True
				v.add(rule[0])
				e.setdefault(rule[0], set()).add(rule[1])
				weights[rule[0],rule[1]] = w
		if False and match:
			print tree
			print n, sent

	def visit(current, edges, visited):
		""" depth-first cycle detection """
		for a in edges.get(current, set()):
			if a in visited:
				visit.mem.add(current)
				yield visited[visited.index(a):] + [a]
			elif a not in visit.mem:
				for b in visit(a, edges, visited + [a]): yield b
	visit.mem = set()
	for a in v:
		for b in visit(a, e, []):
			print "cycle", b, "cost", sum(weights[c,d] for c,d in zip(b, b[1:]))

	for interp in range(0, 1): #disable interpolation
		interpolate = interp / 10.0
		print "INTERPOLATE", interpolate
		grammar = []; dopgrammar = []
		if srcg:
			grammar = induce_srcg(list(trees), sents)
			#for (rule,yf),w in sorted(grammar, key=lambda x: x[0][0][0]):
			#	if len(rule) == 2 and rule[1] != "Epsilon":
			#		print exp(w), rule[0], "-->", " ".join(rule[1:]), "\t\t", [list(a) for a in yf]
			#grammar = read_rparse_grammar("../rparse/bin3600")
			lhs = set(rule[0] for (rule,yf),w in grammar)
			print "SRCG based on", len(trees), "sentences"
			l = len(grammar)
			print "labels:", len(set(rule[a] for (rule,yf),w in grammar for a in range(3) if len(rule) > a)), "of which preterminals:", len(set(rule[0] for (rule,yf),w in grammar if rule[1] == "Epsilon")) or len(set(rule[a] for (rule,yf),w in grammar for a in range(1,3) if len(rule) > a and rule[a] not in lhs))
			print "max arity:", max((len(yf), rule, yf, w) for (rule,yf),w in grammar)
			grammar = splitgrammar(grammar)
			ll=sum(len(b) for a,b in grammar.lexical.items())
			print "clauses:",l, "lexical clauses:", ll, "non-lexical clauses:", l - ll
			testgrammar(grammar)
			print "induced srcg grammar"

		if dop:
			if estimator == "shortest":
				# the secondary model is used to resolve ties for the shortest derivation
				dopgrammar, secondarymodel = dop_srcg_rules(list(trees), list(sents), normalize=False,
								shortestderiv=True,	arity_marks=arity_marks)
			else:
				dopgrammar = dop_srcg_rules(list(trees), list(sents), normalize=(estimator in ("ewe", "sl-dop")),
								shortestderiv=False, arity_marks=arity_marks,
								interpolate=interpolate, wrong_interpolate=wrong_interpolate)
			nodes = sum(len(list(a.subtrees())) for a in trees)
			l = len(dopgrammar)
			print "labels:", len(set(rule[a] for (rule,yf),w in dopgrammar for a in range(3) if len(rule) > a)), "of which preterminals:", len(set(rule[0] for (rule,yf),w in dopgrammar if rule[1] == "Epsilon")) or len(set(rule[a] for (rule,yf),w in dopgrammar for a in range(1,3) if len(rule) > a and rule[a] not in lhs))
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
		#exit(
		doparse(srcg, dop, estimator, unfolded, bintype, viterbi, sample, both, arity_marks, arity_marks_before_bin, interpolate, wrong_interpolate, n, m, grammar, dopgrammar, test, maxlen, maxsent)

def doparse(srcg, dop, estimator, unfolded, bintype, viterbi, sample, both, arity_marks, arity_marks_before_bin, interpolate, wrong_interpolate, n, m, grammar, dopgrammar, test, maxlen, maxsent, top='ROOT', tags=True):
	sresults = []; dresults = []
	serrors1 = FreqDist(); serrors2 = FreqDist()
	derrors1 = FreqDist(); derrors2 = FreqDist()
	gold = []; gsent = []
	scandb = set(); dcandb = set(); goldbrackets = set()
	nsent = exact = exacts = snoparse = dnoparse =  0
	estimate = lambda a,b: 0.0
	removeids = re.compile("@[0-9]+")
	#if srcg: derivout = codecs.open("srcgderivations", "w", encoding='utf-8')
	for tree, sent, block in zip(*test):
		if len(sent) > maxlen: continue
		if nsent >= maxsent: break
		nsent += 1
		print "%d. [len=%d] %s" % (nsent, len(sent), " ".join(a[0]+"/"+a[1] for a in sent))
		goldb = bracketings(tree)
		gold.append(block)
		gsent.append(sent)
		goldbrackets.update((nsent, a) for a in goldb)
		if srcg:
			print "SRCG:",
			chart, start = parse([w for w,t in sent], grammar,
								[t for w,t in sent] if tags else [],
								grammar.toid[top], True, 1, None)
			print
		else: chart = {}; start = False
		#for a in chart: chart[a].sort()
		#for result, prob in enumchart(chart, start, grammar.tolabel) if start else ():
		if start:
			result, prob = mostprobablederivation(chart, start, grammar.tolabel)
			#result = rem_marks(Tree(alterbinarization(result)))
			#print result
			#derivout.write("vitprob=%.6g\n%s\n\n" % (
			#				exp(-prob), terminals(result,  sent)))
			result = Tree(result)
			un_collinize(result)
			rem_marks(result)
			if unfolded: fold(result)
			print "p = %.4e" % (exp(-prob),),
			candb = bracketings(result)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			if result == tree or f1 == 1.0:
				assert result != tree or f1 == 1.0
				print "exact match"
				exacts += 1
			else:
				print "LP", round(100 * prec, 2), "LR", round(100 * rec, 2), "LF", round(100 * f1, 2)
				print "cand-gold", printbrackets(candb - goldb),
				print "gold-cand", printbrackets(goldb - candb)
				print "     ", result.pprint(margin=1000)
				serrors1.update(a[0] for a in candb - goldb)
				serrors2.update(a[0] for a in goldb - candb)
			sresults.append(result)
		else:
			if srcg: print "no parse"
			#derivout.write("Failed to parse\nparse_failure.\n\n")
			result = Tree(top, [Tree("PN", [i]) for i in range(len(sent))])
			candb = bracketings(result)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			snoparse += 1
			sresults.append(result)
		scandb.update((nsent, a) for a in candb)
		if dop:
			print "DOP:",
			#estimate = partial(getoutside, outside, maxlen, len(sent))
			if srcg: srcgchart = filterchart(chart, start)
			else: srcgchart = {}
			print "srcgchart filtered", len(srcgchart)
			chart, start = parse([a[0] for a in sent], dopgrammar,
								[a[1] for a in sent] if tags else [],
								dopgrammar.toid[top], viterbi, n, None,
								prune=frozenset(srcgchart.keys()),
								prunetoid=grammar.toid)
		else: chart = {}; start = False
		if start:
			if nsent == 1:
				codecs.open("dopderivations", "w",
					encoding="utf-8").writelines(
						"vitprob=%#.6g\n%s\n" % (exp(-p),
							re.sub(r'([{}\[\]<>\^$\'])', r'\\\1',
								terminals(t, sent).replace(') (', ')(')))
						for t, p in lazykbest(dict(chart), start, m,
													dopgrammar.tolabel))
			if estimator == "shortest": # equal to ls-dop with n=1 ?
				mpp = mostprobableparse(chart, start, dopgrammar.tolabel, n=m,
						sample=sample, both=both, shortest=True,
						secondarymodel=secondarymodel).items()
			elif estimator == "sl-dop":
				# get n most likely derivations
				derivations = lazykbest(chart, start, m, dopgrammar.tolabel)
				# sum over Goodman derivations to get DOP derivations
				mpp = FreqDist()
				for t, p in derivations: mpp[removeids.sub("@", t)] += exp(-p)
				# count subtrees used (negative, because less is better), use probabilities for tie breaking
				mpp = [(t.replace("@", ""), (t.count("@") - t.count("("), p)) for t, p in mpp.items()[:14]]
				print "(%d derivations, %d parsetrees)" % (len(derivations), len(mpp))
			else:
				mpp = mostprobableparse(chart, start, dopgrammar.tolabel, n=m, sample=sample, both=both).items()
			prob = max(b for a,b in mpp)
			tieresults = [Tree(a) for a,b in mpp if b == prob]
			tieresults.sort(key=lambda t: len(list(t.subtrees())))
			dresult = tieresults[0]
			if len(tieresults) > 1:
				if len(list(tieresults[0].subtrees())) < len(list(tieresults[1].subtrees())):
					print "tie broken"
				else: print "tie not broken"
			if isinstance(prob, tuple):
				print "subtrees = %d, p = %.4e" % (abs(prob[0]), prob[1]),
			else:
				print "p = %.4e" % (prob,),
			un_collinize(dresult)
			rem_marks(dresult)
			if unfolded: fold(dresult)
			candb = bracketings(dresult)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			if dresult == tree or f1 == 1.0:
				print "exact match"
				exact += 1
			else:
				print "LP", round(100 * prec, 2), "LR", round(100 * rec, 2), "LF", round(100 * f1, 2)
				print "cand-gold", printbrackets(candb - goldb),
				print "gold-cand", printbrackets(goldb - candb)
				print "     ", dresult.pprint(margin=1000)
				derrors1.update(a[0] for a in candb - goldb)
				derrors2.update(a[0] for a in goldb - candb)
			dresults.append(dresult)
		else:
			if dop: print "\nno parse"
			dresult = Tree(top, [Tree("PN", [i]) for i in range(len(sent))])
			candb = bracketings(dresult)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			dnoparse += 1
			dresults.append(dresult)
		print "GOLD:", tree.pprint(margin=1000)
		dcandb.update((nsent, a) for a in candb)
		if srcg:
			print "srcg ex", 100 * round(exacts / float(nsent), 4),
			print "lp", 100 * round(precision(goldbrackets, scandb), 4),
			print "lr", 100 * round(recall(goldbrackets, scandb), 4),
			print "lf1", 100 * round(f_measure(goldbrackets, scandb), 4)
		if dop:
			print "dop  ex", 100 * round(exact / float(nsent), 4),
			print "lp", 100 * round(precision(goldbrackets, dcandb), 4),
			print "lr", 100 * round(recall(goldbrackets, dcandb), 4),
			print "lf1", 100 * round(f_measure(goldbrackets, dcandb), 4)
		print

	if srcg:
		#derivout.close()
		codecs.open("test1.srcg", "w", encoding='utf-8').writelines(
			"%s\n" % export(a,b,n)
			for n,(a,b) in enumerate(zip(sresults, gsent)))
		codecs.open("test.cf.srcg", "w", encoding='utf-8').writelines(
			a.pprint(margin=999)+'\n' for a in sresults)
	if dop:
		codecs.open("test1.dop", "w", encoding='utf-8').writelines(
			"%s\n" % export(a, b, n)
			for n,(a, b) in enumerate(zip(dresults, gsent)))
		codecs.open("test.cf.dop", "w", encoding='utf-8').writelines(
			a.pprint(margin=999)+'\n' for a in dresults)
	#if dop: open("interp%d.dop" % interp, "w").writelines("%s\n" % export(a,b,n) for n,(a,b) in enumerate(zip(dresults, gsent)))
	codecs.open("test1.gold", "w", encoding='utf-8').write(''.join(
		"#BOS %d\n%s\n#EOS %d\n" % (n, a, n) for n, a in enumerate(gold)))
	codecs.open("test.cf.gold", "w", encoding='utf-8').writelines(
		a.pprint(margin=999)+'\n' for a in test[0])
	print "maxlen", maxlen, "unfolded", unfolded, "arity marks", arity_marks, "binarized", bintype, "estimator", estimator
	if interpolate != 1.0: print "interpolate", interpolate, "wrong_interpolate", wrong_interpolate
	print "error breakdown, first 10 categories."
	if srcg and dop: print "SRCG (not in gold, missing from candidate), DOP (idem)"
	elif srcg: print "SRCG (not in gold, missing from candidate)"
	elif dop: print "DOP (not in gold, missing from candidate)"
	z = ((serrors1.items(), serrors2.items()) if srcg else ()) + ((derrors1.items(), derrors2.items()) if dop else ())
	for a in zip(*z)[:10]:
		print "\t".join(map(lambda x: ": ".join(map(str, x)), a))
	if srcg and nsent:
		print "SRCG:"
		print "coverage", (nsent - snoparse), "/", nsent, "=", 100 * (nsent - snoparse) / float(nsent), "%",
		print "exact match", exacts, "/", nsent, "=", 100 * exacts / float(nsent)
		print "srcg lp", 100 * precision(goldbrackets, scandb),
		print "lr", 100 * recall(goldbrackets, scandb), "lf1", 100 * f_measure(goldbrackets, scandb)
		print
	if dop and nsent:
		print "DOP:"
		print "coverage", (nsent - dnoparse), "/", nsent, "=", 100 * (nsent - dnoparse) / float(nsent), "%",
		print "exact match", exact, "/", nsent, "=", 100 * exact / float(nsent)
		print "dop  lp", 100 * precision(goldbrackets, dcandb),
		print "lr", 100 * recall(goldbrackets, dcandb), "lf1", 100 * f_measure(goldbrackets, dcandb)

def root(tree):
	if tree.node == "VROOT": tree.node = "ROOT"
	else: tree = Tree("ROOT",[tree])
	return tree

def cftiger():
	#read_penn_format('../tiger/corpus/tiger_release_aug07.mrg')
	grammar = read_bitpar_grammar('/tmp/gtigerpcfg.pcfg', '/tmp/gtigerpcfg.lex')
	dopgrammar = read_bitpar_grammar('/tmp/gtiger.pcfg', '/tmp/gtiger.lex')
	testgrammar(grammar)
	testgrammar(dopgrammar)
	dop = True; srcg = True; unfolded = False; bintype = "collinize h=1 v=1"
	viterbi = True; sample = False; both = False; arity_marks = True
	arity_marks_before_bin = False; estimator = 'dop1'; interpolate = 1.0
	wrong_interpolate = False; n = 0; m = 1000; maxlen = 15; maxsent = 1
	top = "ROOT"; tags = False
	trees = list(islice((a for a in islice((root(Tree(a))
					for a in codecs.open(
							'../tiger/corpus/tiger_release_aug07.mrg',
							encoding='iso-8859-1')), 7200, 9600)
				if len(a.leaves()) <= maxlen), maxsent))
	lex = set(wt for tree in (root(Tree(a))
					for a in islice(codecs.open(
						'../tiger/corpus/tiger_release_aug07.mrg',
						encoding='iso-8859-1'), 7200))
				if len(tree.leaves()) <= maxlen for wt in tree.pos())
	sents = [[(t + '_' + (w if (w, t) in lex else ''), t)
						for w, t in a.pos()] for a in trees]
	for tree in trees:
		for nn, a in enumerate(tree.treepositions('leaves')):
			tree[a] = nn
	blocks = [export(*a) for a in zip(trees, sents, count())]
	test = trees, sents, blocks
	doparse(srcg, dop, estimator, unfolded, bintype, viterbi, sample, both, arity_marks, arity_marks_before_bin, interpolate, wrong_interpolate, n, m, grammar, dopgrammar, test, maxlen, maxsent, top, tags)

def foo(a):
	result = Tree(a)
	un_collinize(result)
	for n, a in enumerate(result.treepositions('leaves')):
		result[a] = n
	return result.pprint(margin=999) + '\n'

if __name__ == '__main__':
	import sys
	sys.stdout = codecs.getwriter('utf8')(sys.stdout)
	#cftiger()
	#plac.call(main)
	main()
