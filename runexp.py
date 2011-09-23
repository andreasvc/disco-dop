# -*- coding: UTF-8 -*-
import logging
from collections import defaultdict
from itertools import islice, count
from operator import itemgetter
from math import exp
import cPickle, re, time, codecs
from nltk import FreqDist, Tree
from nltk.metrics import precision, recall, f_measure, accuracy
#import plac
from negra import NegraCorpusReader, fold, unfold
from grammar import srcg_productions, dop_srcg_rules, induce_srcg, enumchart,\
		read_rparse_grammar, testgrammar, rem_marks, alterbinarization,\
		terminals, varstoindices, read_bitpar_grammar, read_penn_format,\
		splitgrammar, coarse_grammar, grammarinfo, baseline, write_srcg_grammar
from eval import bracketings, printbrackets, export, mean, harmean
from fragmentseeker import extractfragments
from treetransforms import collinize, un_collinize, optimalbinarize,\
							splitdiscnodes, mergediscnodes
from plcfrs import parse
from coarsetofine import prunelist_fromchart
from disambiguation import mostprobableparse, mostprobablederivation,\
							sldop, sldop_simple

# todo:
# - command line interface
# - split off evaluation code into new module

def main(
	#parameters. parameters. PARAMETERS!!
	srcg = True,
	bitpardop = False,
	dop = True,
	unfolded = False,
	corpusdir="../rparse",
	corpusfile="negraproc.export",
	#corpusfile="tigerprocfull.export",
	maxlen = 40,  # max number of words for sentences in test corpus
	trainmaxlen = 40, # max number of words for sentences in train corpus
	#train = 7200, maxsent = 100,	# number of sentences to parse
	#train = 0.9, maxsent = 9999,	# percent of sentences to parse
	train = 18602, maxsent = 1000, # number of sentences to parse
	skip=0,
	#skip=1000, #skip dev set to get test set
	bintype = "collinize", # choices: collinize, nltk, optimal, optimalhead
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
	k = 50,		#number of coarse derivations to prune with; k=0 => filter only
	prune=True,	#whether to use srcg chart to prune parsing of dop
	getestimates=False, #compute & store estimates
	useestimates=False,  #load & use estimates
	mergesplitnodes=True, #coarse grammar is PCFG with splitted nodes eg. VP*
	markorigin=True, #when splitting nodes, mark origin: VP_2 => {VP*1, VP*2}
	splitprune=True, #VP_2[101] is treated as { VP*[100], VP*[001] } during parsing
	removeparentannotation=False, # VP^<S> is treated as VP
	neverblockmarkovized=False, #do not prune intermediate nodes of binarization
	neverblockdiscontinuous=False, #never block discontinuous nodes.
	usebitpar=False,
	quiet=False, reallyquiet=False #quiet=no per sentence results
	):
	# Tiger treebank version 2 sample:
	# http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/annotation/sample2.export
	#corpus = NegraCorpusReader(".", "sample2\.export", encoding="iso-8859-1"); maxlen = 99
	assert bintype in ("optimal", "optimalhead", "collinize", "nltk")
	assert estimator in ("dop1", "ewe", "shortest", "sl-dop", "sl-dop-simple")
	#format = "%(levelname)s %(asctime)s %(funcName)s %(lineno)d %(message)s"
	# Log everything, and send it to stderr, in a format with just the message.
	format = '%(message)s'
	if reallyquiet: logging.basicConfig(level=logging.WARNING, format=format)
	elif quiet: logging.basicConfig(level=logging.INFO, format=format)
	else: logging.basicConfig(level=logging.DEBUG, format=format)

	corpus = NegraCorpusReader(corpusdir, corpusfile,
		headorder=(bintype in ("collinize", "optimalhead")),
		headfinal=True, headreverse=False, unfold=unfolded)
	logging.info("%d sentences in corpus %s/%s" % (
			len(corpus.parsed_sents()), corpusdir, corpusfile))
	if isinstance(train, float):
		train = int(train * len(corpus.sents()))
	trees, sents, blocks = corpus.parsed_sents()[:train], corpus.sents()[:train], corpus.blocks()[:train]
	logging.info("%d training sentences before length restriction" % len(trees))
	trees, sents, blocks = zip(*[sent for sent in zip(trees, sents, blocks) if len(sent[1]) <= trainmaxlen])
	logging.info("%d training sentences after length restriction" % len(trees))

	# parse training corpus as a "soundness check"
	#test = corpus.parsed_sents(), corpus.tagged_sents(), corpus.blocks()

	test = NegraCorpusReader(corpusdir, corpusfile)
	test = test.parsed_sents()[train+skip:], test.tagged_sents()[train+skip:], test.blocks()[train+skip:]

	logging.info("%d test sentences (before length restriction)" % len(test[0]))
	test = zip(*((a,b,c) for a,b,c in zip(*test) if len(b) <= maxlen))
	logging.info("%d test sentences (after length restriction)" % len(test[0]))
	logging.info("read training & test corpus")

	# binarization
	begin = time.clock()
	if arity_marks_before_bin: [srcg_productions(a, b) for a, b in zip(trees, sents)]
	if bintype == "nltk":
		bintype += " %s h=%d v=%d" % (factor, h, v)
		for a in trees: a.chomsky_normal_form(factor="right", vertMarkov=v-1, horzMarkov=h)
	elif bintype == "collinize":
		bintype += " %s h=%d v=%d %s" % (factor, h, v, "tailmarker" if tailmarker else '')
		[collinize(a, factor=factor, vertMarkov=v-1, horzMarkov=h, tailMarker=tailmarker, leftMostUnary=True, rightMostUnary=True) for a in trees]
	elif bintype == "optimal":
		trees = [Tree.convert(optimalbinarize(tree))
						for n, tree in enumerate(trees)]
	elif bintype == "optimalhead":
		trees = [Tree.convert(
					optimalbinarize(tree, headdriven=True, h=h, v=v))
						for n,tree in enumerate(trees)]
	logging.info("binarized %s cpu time elapsed: %g s" % (
						bintype, time.clock() - begin))
	
	if mergesplitnodes:
		splittrees = [splitdiscnodes(a.copy(True), markorigin) for a in trees]
		logging.info("splitted discontinuous nodes")
		# second layer of binarization:
		for a in splittrees:
			a.chomsky_normal_form(childChar=":")

	# cycle detection
	seen = set()
	v = set(); e = {}; weights = {}
	for n, (tree, sent) in enumerate(zip(trees, sents)):
		rules = [(a,b) for a,b in induce_srcg([tree], [sent]) if a not in seen]
		seen.update(map(lambda (a,b): a, rules))
		for (rule,yf), w in rules:
			if len(rule) == 2 and rule[1] != "Epsilon":
				v.add(rule[0])
				e.setdefault(rule[0], set()).add(rule[1])
				weights[rule[0], rule[1]] = abs(w)

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
			logging.debug("cycle (cost %5.2f): %s" % (
				sum(weights[c,d] for c,d in zip(b, b[1:])), " => ".join(b)))

	srcggrammar = []; dopgrammar = []; secondarymodel = []
	if srcg:
		if mergesplitnodes:
			#corpus = codecs.open("../tiger/corpus/tiger_release_aug07.mrg", encoding="iso-8859-1").read().splitlines()
			#splittrees = map(Tree, corpus[:7200])
			#for a in splittrees:
			#	a.node = "ROOT"
			#	a.chomsky_normal_form(horzMarkov=1)
			#	for n, x in enumerate(a.treepositions('leaves')): a[x] = n
			srcggrammar = induce_srcg(splittrees, sents)
			logging.info("induced CFG based on %d sentences" % len(splittrees))
			#srcggrammar = dop_srcg_rules(splittrees, sents)
			#logging.info("induced DOP CFG based on %d sentences" % len(trees))
			if usebitpar:
				pass # FIXME
				# write grammar
		else:
			srcggrammar = induce_srcg(trees, sents)
			logging.info("induced SRCG based on %d sentences" % len(trees))
		#srcggrammar = coarse_grammar(trees, sents)
		#srcggrammar = read_rparse_grammar("../rparse/bin3600")
		#write_srcg_grammar(srcggrammar, "rules.srcg", "lexicon.srcg")
		grammarinfo(srcggrammar)
		srcggrammar = splitgrammar(srcggrammar)
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
			secondarymodel = splitgrammar(dopshortest)
		else:
			dopgrammar = dop_srcg_rules(list(trees), list(sents), normalize=(estimator in ("ewe", "sl-dop", "sl-dop-simple")),
							shortestderiv=False, arity_marks=arity_marks)
			#dopgrammar = dop_srcg_rules(list(trees), list(sents), normalize=(estimator in ("ewe", "sl-dop")),
			#				shortestderiv=False, arity_marks=arity_marks)
		nodes = sum(len(list(a.subtrees())) for a in trees)
		dopgrammar1 = splitgrammar(dopgrammar)
		logging.info("DOP model based on %d sentences, %d nodes, %d nonterminals"
						% (len(trees), nodes, len(dopgrammar1.toid)))
		grammarinfo(dopgrammar)
		dopgrammar = dopgrammar1
		testgrammar(dopgrammar)

	if getestimates:
		from estimates import getestimates
		import numpy as np
		logging.info("computing estimates")
		begin = time.clock()
		outside = getestimates(srcggrammar, maxlen, srcggrammar.toid["ROOT"])
		logging.info("estimates done. cpu time elapsed: %g s"
					% (time.clock() - begin))
		np.savez("outside.npz", outside=outside)
		#cPickle.dump(outside, open("outside.pickle", "wb"))
		logging.info("saved estimates")
	if useestimates:
		import numpy as np
		#outside = cPickle.load(open("outside.pickle", "rb"))
		outside = np.load("outside.npz")['outside']
		logging.info("loaded estimates")
	else: outside = None

	#for a,b in extractfragments(trees).items():
	#	print a,b
	#exit()
	begin = time.clock()
	results = doparse(srcg, dop, estimator, unfolded, bintype, sample,
			both, arity_marks, arity_marks_before_bin, m,
			srcggrammar, dopgrammar, secondarymodel, test, maxlen, maxsent,
			prune, k, sldop_n, useestimates, outside, "ROOT", True,
			removeparentannotation, splitprune, mergesplitnodes, markorigin,
			neverblockmarkovized, neverblockdiscontinuous, usebitpar)
	logging.info("time elapsed during parsing: %g s" % (time.clock() - begin))
	doeval(*results)

def doparse(srcg, dop, estimator, unfolded, bintype, sample, both, arity_marks,
		arity_marks_before_bin, m, srcggrammar, dopgrammar, secondarymodel,
		test, maxlen, maxsent, prune, k, sldop_n=14, useestimates=False,
		outside=None, top='ROOT', tags=True, removeparentannotation=False,
		splitprune=False, mergesplitnodes=False, markorigin=False,
		neverblockmarkovized=False, neverblockdiscontinuous=False,
		usebitpar=False, filename="results", sentinit=0, doph=999):
	sresults = []; dresults = []
	serrors1 = FreqDist(); serrors2 = FreqDist()
	derrors1 = FreqDist(); derrors2 = FreqDist()
	gold = []; gsent = []
	scandb = set(); dcandb = set(); goldbrackets = set()
	nsent = exact = exacts = snoparse = dnoparse =  0
	estimate = lambda a,b: 0.0
	removeids = re.compile("@[0-9]+")
	#if srcg: derivout = codecs.open("srcgderivations", "w", encoding='utf-8')
	#if bitpardop:	
	# parse w/bitpar
	# get MPPs
	# get prunelist
	# ???
	# profit
	for tree, sent, block in zip(*test):
		if len(sent) > maxlen: continue
		if nsent >= maxsent: break
		nsent += 1
		logging.debug("%d. [len=%d] %s" % (nsent, len(sent),
					u" ".join(a[0]+u"/"+a[1] for a in sent)))
		goldb = bracketings(tree)
		gold.append(block)
		gsent.append(sent)
		goldbrackets.update((nsent, a) for a in goldb)
		if srcg and not usebitpar:
			msg = "SRCG: "
			begin = time.clock()
			chart, start = parse(
						[w for w,t in sent], srcggrammar,
						tags=[t for w,t in sent] if tags else [],
						start=srcggrammar.toid[top], exhaustive=dop and prune,
						estimate=(outside, maxlen) if useestimates else None,
						) #beamwidth=50)
		elif usebitpar:
			msg = "PCFG: "
			begin = time.clock()
			# FIXME
			# bitpar
			# read derivations
			# build list of items in k-best derivs
		else: chart = {}; start = False
		if start:
			resultstr, prob = mostprobablederivation(chart, start, srcggrammar.tolabel)
			#derivout.write("vitprob=%.6g\n%s\n\n" % (
			#				exp(-prob), terminals(result,  sent)))
			result = Tree(resultstr)
			if mergesplitnodes:
				result.un_chomsky_normal_form(childChar=":")
				mergediscnodes(result)
			un_collinize(result)
			rem_marks(result)
			if unfolded: fold(result)
			msg += "p = %.4e " % exp(-prob)
			candb = bracketings(result)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			if result == tree or f1 == 1.0:
				assert result != tree or f1 == 1.0
				msg += "exact match"
				exacts += 1
			else:
				msg += "LP %5.2f LR %5.2f LF %5.2f\n" % (
								100 * prec, 100 * rec, 100 * f1)
				if candb - goldb:
					msg += "cand-gold %s " % printbrackets(candb - goldb)
				if goldb - candb:
					msg += "gold-cand %s" % printbrackets(goldb - candb)
				if (candb - goldb) or (goldb - candb): msg += '\n'
				msg += "      %s" % result.pprint(margin=1000)
				serrors1.update(a[0] for a in candb - goldb)
				serrors2.update(a[0] for a in goldb - candb)
			sresults.append(result)
		else:
			if srcg: msg += "no parse"
			#derivout.write("Failed to parse\nparse_failure.\n\n")
			result = baseline([(n,t) for n,(w,t) in enumerate(sent)])
			result = Tree.parse("(%s %s)" % (top, result), parse_leaf=int)
			candb = bracketings(result)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			snoparse += 1
			sresults.append(result)
		if srcg:
			logging.debug(msg+"\n%.2fs cpu time elapsed" % (
						time.clock() - begin))
		scandb.update((nsent, a) for a in candb)
		msg = ""
		if dop and start:
			msg = " DOP: "
			begin = time.clock()
			if srcg and prune and start:
				prunelist = prunelist_fromchart(chart, start, srcggrammar,
								dopgrammar, k, removeparentannotation,
								mergesplitnodes and not splitprune, doph)
			else: prunelist = []
			chart, start = parse(
							[w for w,t in sent], dopgrammar,
							tags=[t for w,t in sent] if tags else None,
							start=dopgrammar.toid[top],
							exhaustive=True,
							estimate=None,
							prunelist=prunelist,
							prunetoid=srcggrammar.toid,
							splitprune=splitprune,
							markorigin=markorigin,
							neverblockmarkovized=neverblockmarkovized,
							neverblockdiscontinuous=neverblockdiscontinuous)
		else: chart = {}; start = False
		if dop and start:
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
				msg += "subtrees = %d, p = %.4e " % (abs(prob[0]), prob[1])
			else:
				msg += "p = %.4e " % prob
			un_collinize(dresult)
			rem_marks(dresult)
			if unfolded: fold(dresult)
			candb = bracketings(dresult)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			if dresult == tree or f1 == 1.0:
				msg += "exact match"
				exact += 1
			else:
				msg += "LP %5.2f LR %5.2f LF %5.2f\n" % (
								100 * prec, 100 * rec, 100 * f1)
				if candb - goldb:
					msg += "cand-gold %s " % printbrackets(candb - goldb)
				if goldb - candb:
					msg += "gold-cand %s " % printbrackets(goldb - candb)
				if (candb - goldb) or (goldb - candb): msg += '\n'
				msg += "      %s" % dresult.pprint(margin=1000)
				derrors1.update(a[0] for a in candb - goldb)
				derrors2.update(a[0] for a in goldb - candb)
			dresults.append(dresult)
		else:
			if dop: msg += "no parse"
			dresult = baseline([(n,t) for n,(w,t) in enumerate(sent)])
			dresult = Tree.parse("(%s %s)" % (top, dresult), parse_leaf=int)
			candb = bracketings(dresult)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			dnoparse += 1
			dresults.append(dresult)
		if dop:
			logging.debug(msg+"\n%.2fs cpu time elapsed" % (
						time.clock() - begin))
		logging.debug("GOLD: %s" % tree.pprint(margin=1000))
		dcandb.update((nsent, a) for a in candb)
		if srcg:
			logging.debug("srcg cov %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f%s" % (
								100 * (1 - snoparse/float(len(sresults))),
								100 * (exacts / float(nsent)),
								100 * precision(goldbrackets, scandb),
								100 * recall(goldbrackets, scandb),
								100 * f_measure(goldbrackets, scandb),
								('' if dop else '\n')))
		if dop:
			logging.debug("dop  cov %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f (%+5.2f)\n" % (
								100 * (1 - dnoparse/float(len(dresults))),
								100 * (exact / float(nsent)),
								100 * precision(goldbrackets, dcandb),
								100 * recall(goldbrackets, dcandb),
								100 * f_measure(goldbrackets, dcandb),
								100 * (f_measure(goldbrackets, dcandb)
									- f_measure(goldbrackets, scandb))))

	if srcg:
		codecs.open("%s.srcg" % filename, "w", encoding='utf-8').writelines(
			"%s\n" % export(a,b,n + 1)
			for n,a,b in zip(count(sentinit), sresults, gsent))
	if dop:
		codecs.open("%s.dop" % filename, "w", encoding='utf-8').writelines(
			"%s\n" % export(a, b, n + 1)
			for n,a,b in zip(count(sentinit), dresults, gsent))
	codecs.open("%s.gold" % filename, "w", encoding='utf-8').write(''.join(
		"#BOS %d\n%s#EOS %d\n" % (n + 1, a, n + 1) for n, a in zip(count(sentinit), gold)))

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

def parsetepacoc(dop=True, srcg=True, estimator='ewe', unfolded=False,
	bintype="collinize", h=1, v=1, factor="right", doph=1, arity_marks=True,
	arity_marks_before_bin=False, sample=False, both=False, m=10000,
	trainmaxlen=999, maxlen=40, maxsent=999, k=50, prune=True, sldop_n=7,
	removeparentannotation=False, splitprune=True, mergesplitnodes=True,
	neverblockmarkovized=False, markorigin = True, resultdir="tepacoc-40"):

	format = '%(message)s'
	logging.basicConfig(level=logging.DEBUG, format=format)
	tepacocids, tepacocsents = readtepacoc()
	#corpus = NegraCorpusReader("../rparse", "tigerprocfullnew.export",
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
							corpus_blocks)) if len(sent[1]) <= trainmaxlen
							and n not in tepacocids][:train])
	begin = time.clock()
	if bintype == "optimal":
		trees = [optimalbinarize(tree) for n, tree in enumerate(trees)]
	elif bintype == "nltk":
		for a in trees: a.chomsky_normal_form(factor="right", vertMarkov=v-1, horzMarkov=h)
	elif bintype == "collinize":
		[collinize(a, factor=factor, vertMarkov=v-1, horzMarkov=h, tailMarker="",
					leftMostUnary=True, rightMostUnary=True) for a in trees]
	print "time elapsed during binarization: ", time.clock() - begin
	coarsetrees = trees
	if mergesplitnodes:
		coarsetrees = [splitdiscnodes(a.copy(True), markorigin) for a in trees]
		for a in coarsetrees: a.chomsky_normal_form(childChar=":")
		print "splitted discontinuous nodes"
	srcggrammar = induce_srcg(list(coarsetrees), sents)
	print "induced", "pcfg" if mergesplitnodes else "srcg",
	print "of", len(sents), "sentences"
	grammarinfo(srcggrammar)
	srcggrammar = splitgrammar(srcggrammar)
	testgrammar(srcggrammar)
	
	if removeparentannotation:
		for a in trees:
			a.chomsky_normal_form(factor="right", horzMarkov=doph)
	dopgrammar = dop_srcg_rules(list(trees), list(sents),
				normalize=(estimator in ("ewe", "sl-dop", "sl-dop-simple")),
				shortestderiv=False, arity_marks=arity_marks)
	print "induced dop reduction of", len(sents), "sentences"
	grammarinfo(dopgrammar)
	dopgrammar = splitgrammar(dopgrammar)
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
					removeparentannotation, splitprune, mergesplitnodes,
					markorigin, neverblockmarkovized,
					filename="/".join((resultdir, cat)),
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

if __name__ == '__main__':
	import sys
	sys.stdout = codecs.getwriter('utf8')(sys.stdout)
	#plac.call(main)
	#cftiger()
	parsetepacoc()
	#main()
