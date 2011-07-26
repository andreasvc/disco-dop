# -*- coding: UTF-8 -*-
from collections import defaultdict
from itertools import islice, chain, count
from operator import itemgetter
from functools import partial
from pprint import pprint
from math import log, exp, fsum
from heapq import nlargest
import cPickle, re, time, codecs
from nltk import FreqDist, Tree
from nltk.metrics import precision, recall, f_measure, accuracy
#import plac
from kbest import lazykbest
from negra import NegraCorpusReader, fold, unfold
from grammar import srcg_productions, dop_srcg_rules, induce_srcg, enumchart,\
		export, read_rparse_grammar, mean, harmean, testgrammar,\
		bracketings, printbrackets, rem_marks, alterbinarization, terminals,\
		varstoindices, read_bitpar_grammar, read_penn_format, newsplitgrammar,\
		coarse_grammar, grammarinfo
from fragmentseeker import extractfragments
from treetransforms import collinize, un_collinize, binarizetree,\
							splitdiscnodes, mergediscnodes
from plcfrs_cython import parse
from coarsetofine import prunelist_fromchart
from disambiguation import mostprobableparse, mostprobablederivation,\
							sldop, sldop_simple

def main(
	#parameters. parameters. PARAMETERS!!
	srcg = True,
	dop = True,
	unfolded = False,
	maxlen = 40,  # max number of words for sentences in training & test corpus
	train = 7200, maxsent = 100,	# number of sentences to parse
	#train = 18602, maxsent = 1000, #9999999,	# number of sentences to parse
	skip=1000, #skip test set to get dev set
	bintype = "nltk", # choices: collinize, nltk, optimal, optimalhead
	factor = "right",
	v = 1,
	h = 1,
	arity_marks = True,
	arity_marks_before_bin = False,
	tailmarker = "",
	sample=False, both=False,
	m = 10000,		#number of derivations to sample/enumerate
	estimator = "ewe", # choices: dop1, ewe, shortest, sl-dop[-simple]
	sldop_n=7,
	k = 1000,		#number of coarse derivations to prune with
	prune=True,	#whether to use srcg chart to prune parsing of dop
	getestimates=False, #compute & store estimates
	useestimates=False,  #load & use estimates
	removeparentannotation=False,
	mergesplitnodes=True,
	neverblockmarkovized=True):
	# Tiger treebank version 2 sample:
	# http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/annotation/sample2.export
	#corpus = NegraCorpusReader(".", "sample2\.export", encoding="iso-8859-1"); maxlen = 99
	assert bintype in ("optimal", "optimalhead", "collinize", "nltk")
	assert estimator in ("dop1", "ewe", "shortest", "sl-dop", "sl-dop-simple")
	if isinstance(train, float) or train > 7200:
		#corpus = NegraCorpusReader("../rparse", "tigerprocfull.export",
		#	headorder=(bintype=="collinize"), headfinal=True,
		#	headreverse=False, unfold=unfolded)
		corpus = NegraCorpusReader("../rparse", "negraproc.export",
			headorder=(bintype=="collinize"), headfinal=True,
			headreverse=False, unfold=unfolded)
	else:
		corpus = NegraCorpusReader("../rparse", "tigerproc.export",
			headorder=(bintype=="collinize"), headfinal=True,
			headreverse=False, unfold=unfolded)
	if isinstance(train, float):
		train = int(train * len(corpus.sents()))
	trees, sents, blocks = corpus.parsed_sents()[:train], corpus.sents()[:train], corpus.blocks()[:train]
	trees, sents, blocks = zip(*[sent for sent in zip(trees, sents, blocks)]) # if len(sent[1]) <= maxlen])
	# parse training corpus as a "soundness check"
	#test = corpus.parsed_sents(), corpus.tagged_sents(), corpus.blocks()
	if isinstance(train, float) or train > 7200:
		#test = NegraCorpusReader("../rparse", "tigerprocfull.export")
		test = NegraCorpusReader("../rparse", "negraproc.export")
	else:
		test = NegraCorpusReader("../rparse", "tigerproc.export")
	test = test.parsed_sents()[train+skip:], test.tagged_sents()[train+skip:], test.blocks()[train+skip:]
	print "read training & test corpus"
	begin = time.clock()
	if mergesplitnodes:
		splittrees = [splitdiscnodes(a.copy(True)) for a in trees]
		print "splitted discontinuous nodes"
		for a in splittrees:
			a.chomsky_normal_form(factor="right", vertMarkov=v-1, horzMarkov=h)
	if arity_marks_before_bin: [srcg_productions(a, b) for a, b in zip(trees, sents)]
	if bintype == "collinize":
		bintype += " %s h=%d v=%d %s markovize" % (factor, h, v, "tailmarker" if tailmarker else '')
		#for tree in trees:
		#	for a in tree.subtrees(lambda n: len(n) > 1):
		#		a[-1].node += "*"
		#		a.sort(key=lambda n: n.leaves())
		[collinize(a, factor=factor, vertMarkov=v-1, horzMarkov=h, tailMarker=tailmarker, leftMostUnary=True, rightMostUnary=True) for a in trees]
		#for tree in trees:
		#	for a in tree.subtrees(lambda n: "*" in n.node):
		#		a.node = a.node.replace("*", "")
	if bintype == "nltk":
		bintype += " %s h=%d v=%d" % (factor, h, v)
		for a in trees: a.chomsky_normal_form(factor="right", vertMarkov=v-1, horzMarkov=h)
	ln = len(trees) / 100.0
	def timeprinteval(n,a,f):
		#print n,n/ln,'%',a,
		#begin = time.clock()
		x = f()
		#print time.clock() - begin,'s'
		return x
	if bintype == "optimal": trees = [timeprinteval(n,tree,lambda: binarizetree(tree.freeze())) for n,tree in enumerate(trees)]
	if bintype == "optimalhead": trees = [timeprinteval(n,tree,lambda: binarizetree(tree.freeze(), headdriven=True, h=h, v=v)) for n,tree in enumerate(trees)]
	print "binarized", bintype,
	print "time elapsed: ", time.clock() - begin

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
				weights[rule[0], rule[1]] = abs(w)
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
			print "cycle (cost %5.2f): %s" % (
				sum(weights[c,d] for c,d in zip(b, b[1:])), " => ".join(b))

	srcggrammar = []; dopgrammar = []; secondarymodel = []
	if srcg:
		if mergesplitnodes:
			srcggrammar = induce_srcg(splittrees, sents)
			print "induced CFG based on", len(trees), "sentences"
			#srcggrammar = dop_srcg_rules(splittrees, sents)
			#print "induced DOP CFG based on", len(trees), "sentences"
		else:
			srcggrammar = induce_srcg(trees, sents)
			print "induced SRCG based on", len(trees), "sentences"
		#srcggrammar = coarse_grammar(trees, sents)
		#srcggrammar = read_rparse_grammar("../rparse/bin3600")
		grammarinfo(srcggrammar)
		srcggrammar = newsplitgrammar(srcggrammar)
		testgrammar(srcggrammar)

	if dop:
		if estimator == "shortest":
			# the secondary model is used to resolve ties for the shortest derivation
			dopgrammar, secondarymodel = dop_srcg_rules(list(trees), list(sents), normalize=False,
							shortestderiv=True,	arity_marks=arity_marks)
		elif "sl-dop" in estimator:
			dopgrammar = dop_srcg_rules(list(trees), list(sents), normalize=True,
							shortestderiv=False,	arity_marks=arity_marks)
			dopshortest, _ = dop_srcg_rules(list(trees), list(sents),
							normalize=False, shortestderiv=True,
							arity_marks=arity_marks)
			secondarymodel = newsplitgrammar(dopshortest)
		else:
			dopgrammar = dop_srcg_rules(list(trees), list(sents), normalize=(estimator in ("ewe", "sl-dop", "sl-dop-simple")),
							shortestderiv=False, arity_marks=arity_marks)
			#dopgrammar = dop_srcg_rules(list(trees), list(sents), normalize=(estimator in ("ewe", "sl-dop")),
			#				shortestderiv=False, arity_marks=arity_marks)
		nodes = sum(len(list(a.subtrees())) for a in trees)
		dopgrammar1 = newsplitgrammar(dopgrammar)
		print "DOP model based on", len(trees), "sentences,", nodes, "nodes,",
		print len(dopgrammar1.toid), "nonterminals"
		grammarinfo(dopgrammar)
		dopgrammar = dopgrammar1
		testgrammar(dopgrammar)

	if getestimates:
		from estimates import getestimates
		import numpy as np
		print "computing estimates"
		begin = time.clock()
		outside = getestimates(srcggrammar, maxlen, srcggrammar.toid["ROOT"])
		print "done. time elapsed: ", time.clock() - begin,
		np.savez("outside.npz", outside=outside)
		#cPickle.dump(outside, open("outside.pickle", "wb"))
		print "saved estimates"
	if useestimates:
		import numpy as np
		#outside = cPickle.load(open("outside.pickle", "rb"))
		outside = np.load("outside.npz")['outside']
		print "loaded estimates"
	else: outside = None

	#for a,b in extractfragments(trees).items():
	#	print a,b
	#exit()
	begin = time.clock()
	results = doparse(srcg, dop, estimator, unfolded, bintype, sample,
			both, arity_marks, arity_marks_before_bin, m,
			srcggrammar, dopgrammar, secondarymodel, test, maxlen, maxsent,
			prune, k, sldop_n, useestimates, outside, "ROOT", True,
			removeparentannotation, mergesplitnodes, neverblockmarkovized)
	print "time elapsed during parsing: ", time.clock() - begin
	doeval(*results)

def doparse(srcg, dop, estimator, unfolded, bintype, sample, both, arity_marks,
		arity_marks_before_bin, m, srcggrammar, dopgrammar, secondarymodel,
		test, maxlen, maxsent, prune, k, sldop_n=14, useestimates=False,
		outside=None, top='ROOT', tags=True, removeparentannotation=False,
		mergesplitnodes=False, neverblockmarkovized=False, filename="results",
		sentinit=0, doph=999):
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
		print "%d. [len=%d] " % (nsent, len(sent)),
		myprint(u" ".join(a[0]+u"/"+a[1] for a in sent))
		goldb = bracketings(tree)
		gold.append(block)
		gsent.append(sent)
		goldbrackets.update((nsent, a) for a in goldb)
		if srcg:
			print "SRCG:",
			begin = time.clock()
			chart, start = parse(
						[w for w,t in sent], srcggrammar,
						tags=[t for w,t in sent] if tags else [],
						start=srcggrammar.toid[top], exhaustive=prune,
						estimate=(outside, maxlen) if useestimates else None,
						) #beamwidth=50)
		else: chart = {}; start = False
		if start:
			result, prob = mostprobablederivation(chart, start, srcggrammar.tolabel)
			#derivout.write("vitprob=%.6g\n%s\n\n" % (
			#				exp(-prob), terminals(result,  sent)))
			result = Tree(result)
			un_collinize(result)
			if mergesplitnodes: mergediscnodes(result)
			rem_marks(result)
			if unfolded: fold(result)
			print "p = %.4e" % (exp(-prob),),
			candb = bracketings(result)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			if result == tree or f1 == 1.0:
				assert result != tree or f1 == 1.0
				print "\nexact match"
				exacts += 1
			else:
				print "\nLP %5.2f LR %5.2f LF %5.2f" % (
								100 * prec, 100 * rec, 100 * f1)
				print "cand-gold", printbrackets(candb - goldb)
				print "gold-cand", printbrackets(goldb - candb)
				print "     ", result.pprint(margin=1000)
				serrors1.update(a[0] for a in candb - goldb)
				serrors2.update(a[0] for a in goldb - candb)
			sresults.append(result)
		else:
			if srcg: print "\nno parse"
			#derivout.write("Failed to parse\nparse_failure.\n\n")
			result = baseline([(n,t) for n,(w,t) in enumerate(sent)])
			result = Tree.parse("(%s %s)" % (top, result), parse_leaf=int)
			candb = bracketings(result)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			snoparse += 1
			sresults.append(result)
		if srcg: print " %.2fs cpu time elapsed" % (time.clock() - begin)
		scandb.update((nsent, a) for a in candb)
		if dop and (start or not prune):
			print " DOP:",
			begin = time.clock()
			if srcg and prune and start:
				prunelist = prunelist_fromchart(chart, start, srcggrammar,
								dopgrammar, k, removeparentannotation,
								mergesplitnodes, doph)
			else: prunelist = []
			chart, start = parse(
								[w for w,t in sent], dopgrammar,
								[t for w,t in sent] if tags else [],
								dopgrammar.toid[top], True, None,
								prunelist, neverblockmarkovized)
		else: chart = {}; start = False
		if dop and start:
			if False and nsent == 1:
				codecs.open("dopderivations", "w",
					encoding="utf-8").writelines(
						"vitprob=%#.6g\n%s\n" % (exp(-p),
							re.sub(r'([{}\[\]<>\^$\'])', r'\\\1',
								terminals(t, sent).replace(') (', ')(')))
						for t, p in lazykbest(dict(chart), start, m,
													dopgrammar.tolabel))
			if estimator == "shortest":
				mpp = mostprobableparse(chart, start, dopgrammar.tolabel, n=m,
						sample=sample, both=both, shortest=True,
						secondarymodel=secondarymodel).items()
			elif estimator == "sl-dop":
				mpp = sldop(chart, start, sent, tags, dopgrammar,
							secondarymodel, m, sldop_n, sample, both)
			elif estimator == "sl-dop-simple":
				# old method, estimate shortest derivation directly from number
				# of addressed nodes
				mpp = sldop_simple(chart, start, dopgrammar, m, sldop_n)
			else: #dop1, ewe
				mpp = mostprobableparse(chart, start, dopgrammar.tolabel,
									n=m, sample=sample, both=both).items()
			dresult, prob = max(mpp, key=itemgetter(1))
			dresult = Tree(dresult)
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
				print "\nexact match"
				exact += 1
			else:
				print "\nLP %5.2f LR %5.2f LF %5.2f" % (
								100 * prec, 100 * rec, 100 * f1)
				print "cand-gold", printbrackets(candb - goldb)
				print "gold-cand", printbrackets(goldb - candb)
				print "     ", dresult.pprint(margin=1000)
				derrors1.update(a[0] for a in candb - goldb)
				derrors2.update(a[0] for a in goldb - candb)
			dresults.append(dresult)
		else:
			if dop: print "\nno parse"
			dresult = baseline([(n,t) for n,(w,t) in enumerate(sent)])
			dresult = Tree.parse("(%s %s)" % (top, result), parse_leaf=int)
			candb = bracketings(dresult)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			dnoparse += 1
			dresults.append(dresult)
		if dop: print " %.2fs cpu time elapsed" % (time.clock() - begin)
		print "GOLD:", tree.pprint(margin=1000)
		dcandb.update((nsent, a) for a in candb)
		if srcg:
			print "srcg ex %5.2f lp %5.2f lr %5.2f lf %5.2f" % (
								100 * (exacts / float(nsent)),
								100 * precision(goldbrackets, scandb),
								100 * recall(goldbrackets, scandb),
								100 * f_measure(goldbrackets, scandb))
		if dop:
			print "dop  ex %5.2f lp %5.2f lr %5.2f lf %5.2f (delta %5.2f)" % (
								100 * (exact / float(nsent)),
								100 * precision(goldbrackets, dcandb),
								100 * recall(goldbrackets, dcandb),
								100 * f_measure(goldbrackets, dcandb),
								100 * (f_measure(goldbrackets, dcandb)
									- f_measure(goldbrackets, scandb)))
		print

	if srcg:
		codecs.open("%s.srcg" % filename, "w", encoding='utf-8').writelines(
			"%s\n" % export(a,b,n + 1)
			for n,a,b in zip(count(sentinit), sresults, gsent))
	if dop:
		codecs.open("%s.dop" % filename, "w", encoding='utf-8').writelines(
			"%s\n" % export(a, b, n + 1)
			for n,a,b in zip(count(sentinit), dresults, gsent))
	codecs.open("%s.gold" % filename, "w", encoding='utf-8').write(''.join(
		"#BOS %d\n%s\n#EOS %d\n" % (n + 1, a, n + 1) for n, a in zip(count(sentinit), gold)))

	return (srcg, dop, serrors1, serrors2, derrors1, derrors2, nsent, maxlen,
		exact, exacts, snoparse, dnoparse, goldbrackets, scandb, dcandb, unfolded,
		arity_marks, bintype, estimator, sldop_n)

def doeval(srcg, dop, serrors1, serrors2, derrors1, derrors2, nsent, maxlen,
		exact, exacts, snoparse, dnoparse, goldbrackets, scandb, dcandb,
		unfolded, arity_marks, bintype, estimator, sldop_n):
	print "maxlen", maxlen, "unfolded", unfolded, "arity marks", arity_marks,
	print "binarized", bintype, "estimator", estimator, sldop_n if 'sl-dop' in estimator else ''
	print "error breakdown, first 10 categories."
	if srcg and dop: print "SRCG (not in gold, missing from candidate), DOP (idem)"
	elif srcg: print "SRCG (not in gold, missing from candidate)"
	elif dop: print "DOP (not in gold, missing from candidate)"
	z = ((serrors1.items(), serrors2.items()) if srcg else ()) + ((derrors1.items(), derrors2.items()) if dop else ())
	for a in zip(*z)[:10]:
		print "\t".join(map(lambda x: ": ".join(map(str, x)), a))
	if srcg and nsent:
		print "srcg lp %5.2f lr %5.2f lf %5.2f" % (
				100 * precision(goldbrackets, scandb),
				100 * recall(goldbrackets, scandb),
				100 * f_measure(goldbrackets, scandb))
		print "coverage %d / %d = %5.2f %%  " %(
				nsent - snoparse, nsent, 100.0 * (nsent - snoparse) / nsent),
		print "exact match %d / %d = %5.2f %%\n" % (
				exacts, nsent, 100.0 * exacts / nsent)
	if dop and nsent:
		print "dop  lp %5.2f lr %5.2f lf %5.2f" % (
				100 * precision(goldbrackets, dcandb),
				100 * recall(goldbrackets, dcandb),
				100 * f_measure(goldbrackets, dcandb))
		print "coverage %d / %d = %5.2f %%  " %(
				nsent - dnoparse, nsent, 100.0 * (nsent - dnoparse) / nsent),
		print "exact match %d / %d = %5.2f %%\n" % (
				exact, nsent, 100.0 * exact / nsent)

def baseline(wordstags):
	""" a right branching baseline parse (NP like (NP this (NP example (NP here)))). """
	if wordstags == []: return ''
	return "(%s (%s %s) %s)" % ("NP", wordstags[0][1],
			wordstags[0][0], baseline(wordstags[1:]))

def root(tree):
	if tree.node == "VROOT": tree.node = "ROOT"
	else: tree = Tree("ROOT",[tree])
	return tree

def cftiger():
	#read_penn_format('../tiger/corpus/tiger_release_aug07.mrg')
	grammar = read_bitpar_grammar('/tmp/gtigerpcfg.pcfg', '/tmp/gtigerpcfg.lex')
	dopgrammar = read_bitpar_grammar('/tmp/gtiger.pcfg', '/tmp/gtiger.lex', ewe=False)
	testgrammar(grammar)
	testgrammar(dopgrammar)
	dop = True; srcg = True; unfolded = False; bintype = "collinize h=1 v=1"
	viterbi = True; sample = False; both = False; arity_marks = True
	arity_marks_before_bin = False; estimator = 'sl-dop'; n = 0; m = 10000;
	maxlen = 15; maxsent = 360
	prune = False; top = "ROOT"; tags = False; sldop_n = 5
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
	doparse(srcg, dop, estimator, unfolded, bintype, sample, both, arity_marks,
		arity_marks_before_bin, m, grammar, dopgrammar, test, maxlen, maxsent,
		prune, sldop_n, top, tags)

def readtepacoc():
	tepacocids = set()
	tepacocsents = defaultdict(list)
	cat = "undefined"
	tepacoc = codecs.open("../tepacoc.txt", encoding="utf8")
	for line in tepacoc.read().splitlines():
		fields = line.split("\t") # = [id, '', sent]
		if len(fields) == 3:
			if fields[0].strip():
				# subtract one because our ids are zero-based, tepacoc 1-based
				sentid = int(fields[0]) - 1
				tepacocids.add(sentid)
				tepacocsents[cat].append((sentid, fields[2].split()))
			else:
				cat = fields[2]
				if cat.startswith("CUC"): cat = "CUC"
		elif fields[0] == "TuBa": break
	return tepacocids, tepacocsents

def parsetepacoc():
	dop = True; srcg = True; estimator = 'ewe'; unfolded = False;
	bintype = "nltk"; h=1; v=1; factor = "right"; doph=1
	arity_marks = True; arity_marks_before_bin = False;
	sample = False; both = False; m = 10000;
	maxlen = 40; maxsent = 999; k = 1000; prune=True; sldop_n=7
	removeparentannotation=False; mergesplitnodes=True
	neverblockmarkovized=True

	tepacocids, tepacocsents = readtepacoc()
	#corpus = NegraCorpusReader("../rparse", "tigerprocfull.export",
	#		headorder=(bintype in ("collinize", "nltk")), headfinal=True,
	#		headreverse=False, unfold=unfolded)
	#corpus_sents = list(corpus.sents())
	#corpus_taggedsents = list(corpus.tagged_sents())
	#corpus_trees = list(corpus.parsed_sents())
	#corpus_blocks = list(corpus.blocks())
	#thecorpus = [a for a in zip(corpus_sents, corpus_taggedsents, corpus_trees, corpus_blocks)]
	#cPickle.dump(thecorpus, open("tiger.pickle", "wb"), protocol=-1)
	corpus_sents, corpus_taggedsents, corpus_trees, corpus_blocks = zip(*cPickle.load(open("tiger.pickle", "rb")))
	train = 25005 #int(0.9 * len(corpus_sents))
	trees, sents, blocks = zip(*[sent for n, sent in 
				enumerate(zip(corpus_trees, corpus_sents,
							corpus_blocks)) if #len(sent[1]) <= maxlen and 
							n not in tepacocids][:train])
	begin = time.clock()
	if mergesplitnodes:
		trees = [splitdiscnodes(a.copy(True)) for a in trees]
		print "splitted discontinuous nodes"
	if bintype == "optimal":
		def timeprinteval(n,a,f): return f()
		trees = [timeprinteval(n,tree,lambda: binarizetree(tree.freeze()))
					for n, tree in enumerate(trees)]
	elif bintype == "nltk":
		for a in trees: a.chomsky_normal_form(factor="right", vertMarkov=v-1, horzMarkov=h)
	elif bintype == "collinize":
		[collinize(a, factor=factor, vertMarkov=v-1, horzMarkov=h, tailMarker="$",
					leftMostUnary=True, rightMostUnary=True) for a in trees]
	print "time elapsed during binarization: ", time.clock() - begin
	srcggrammar = induce_srcg(list(trees), sents)
	print "induced srcg grammar of", len(sents), "sentences"
	grammarinfo(srcggrammar)
	srcggrammar = newsplitgrammar(srcggrammar)
	testgrammar(srcggrammar)
	
	if removeparentannotation:
		for a in trees:
			a.un_chomsky_normal_form()
			a.chomsky_normal_form(factor="right", horzMarkov=doph)
	if mergesplitnodes:
		trees, sents, blocks = zip(*[sent for n, sent in 
				enumerate(zip(corpus_trees, corpus_sents,
							corpus_blocks)) if #len(sent[1]) <= maxlen and 
							n not in tepacocids][:train])
		for a in trees: a.chomsky_normal_form(factor="right", vertMarkov=v-1, horzMarkov=h)
	dopgrammar = dop_srcg_rules(list(trees), list(sents),
				normalize=(estimator in ("ewe", "sl-dop", "sl-dop-simple")),
				shortestderiv=False, arity_marks=arity_marks)
	print "induced dop reduction of", len(sents), "sentences"
	grammarinfo(dopgrammar)
	dopgrammar = newsplitgrammar(dopgrammar)
	testgrammar(dopgrammar)
	secondarymodel = []

	results = {}
	testset = {}
	cnt = 0
	for cat, catsents in tepacocsents.iteritems():
		print "category:", cat,
		trees, sents, blocks = [], [], []
		test = trees, sents, blocks
		for n, sent in catsents:
			corpus_sent = corpus_sents[n]
			if len(corpus_sent) <= maxlen:
				if sent != corpus_sent:
					print "mismatch\nnot in corpus",
					print [a for a,b in zip(sent, corpus_sent) if a != b]
					print "not in tepacoc",
					print [b for a,b in zip(sent, corpus_sent) if a != b]
				sents.append(corpus_taggedsents[n])
				trees.append(corpus_trees[n])
				blocks.append(corpus_blocks[n])
		print len(test[0]), "of", len(catsents), "sentences"
		testset[cat] = test
	del corpus_sents, corpus_taggedsents, corpus_trees, corpus_blocks
	for cat, test in sorted(testset.items()):
		print "category:", cat
		begin = time.clock()
		results[cat] = doparse(srcg, dop, estimator, unfolded, bintype, sample,
					both, arity_marks, arity_marks_before_bin, m, srcggrammar,
					dopgrammar, secondarymodel, test, maxlen, maxsent, prune,
					k, sldop_n, False, None, "ROOT", True,
					removeparentannotation, mergesplitnodes,
					neverblockmarkovized, filename="tepacoc-split40/%s" % cat,
					sentinit=cnt) #, doph=doph if doph != h else 999)
		cnt += len(test[0])
		print "time elapsed during parsing: ", time.clock() - begin
	goldbrackets = set(); scandb = set(); dcandb = set()
	exact = exacts = snoparse = dnoparse = 0
	for cat, res in results.iteritems():
		print "category:", cat
		exact += res[8]
		exacts += res[9]
		snoparse += res[10]
		dnoparse += res[11]
		goldbrackets |= res[12]
		scandb |= res[13]
		dcandb |= res[14]
		doeval(*res)
	print "TOTAL"
	doeval(True, True, {}, {}, {}, {}, cnt, maxlen, exact, exacts, snoparse,
			dnoparse, goldbrackets, scandb, dcandb, False, arity_marks,
			bintype, estimator, sldop_n)

def myprint(a):
	sys.stdout.write(a)
	print

if __name__ == '__main__':
	import sys
	sys.stdout = codecs.getwriter('utf8')(sys.stdout)
	#plac.call(main)
	#cftiger()
	#parsetepacoc()
	main()
