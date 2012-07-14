# -*- coding: UTF-8 -*-
import os, re, time, logging, codecs, cPickle
import multiprocessing
from collections import defaultdict, Counter as multiset
from itertools import islice, count
from operator import itemgetter
from math import exp
from nltk import Tree
from treebank import NegraCorpusReader, fold, export
from grammar import srcg_productions, dop_srcg_rules, induce_srcg, rem_marks, \
	read_bitpar_grammar, grammarinfo, baseline, write_srcg_grammar, \
	read_srcg_grammar, doubledop
from containers import Grammar, maxbitveclen
from treetransforms import binarize, unbinarize, optimalbinarize,\
							splitdiscnodes, mergediscnodes
from coarsetofine import prunechart, kbest_items
from plcfrs import parse, cfgparse
from disambiguation import marginalize, viterbiderivation, sldop, sldop_simple
from eval import bracketings, printbrackets, precision, recall, f_measure

def main(
	#parameters. parameters. PARAMETERS!!
	splitpcfg = False,
	srcg = True,
	dop = True,
	usedoubledop = False,	# when False, use DOP reduction instead
	corpusdir="..",
	corpusfile="negra-corpus.export",
	encoding="iso-8859-1",
	movepunct=False,
	removepunct=False,
	unfolded = False,
	testmaxwords = 25,  # max number of words for sentences in test corpus
	trainmaxwords = 25, # max number of words for sentences in train corpus
	#trainsents = 0.9, testsents = 9999,	# percent of sentences to parse
	#trainsents = 18602, testsents = 1000, # number of sentences to parse
	trainsents = 7200, testsents = 100, # number of sentences to parse
	skip=0,	# dev set
	#skip=1000, #skip dev set to get test set
	bintype = "binarize", # choices: binarize, nltk, optimal, optimalhead
	factor = "right",
	revmarkov = True,
	v = 1,
	h = 2,
	arity_marks = True,
	arity_marks_before_bin = False,
	tailmarker = "",
	sample=False, both=False,
	usecfgparse=False, #whether to use the dedicated CFG parser for splitpcfg
	m = 10000,		#number of derivations to sample/enumerate
	estimator = "ewe", # choices: dop1, ewe, shortest, sl-dop[-simple]
	sldop_n=7,
	splitk = 50, #number of coarse pcfg derivations to prune with; k=0 => filter only
	k = 50,		#number of coarse srcg derivations to prune with; k=0 => filter only
	prune=True,	#whether to use srcg chart to prune parsing of dop
	getestimates=False, #compute & store estimates
	useestimates=False,  #load & use estimates
	mergesplitnodes=True, #coarse grammar is PCFG with splitted nodes eg. VP*
	markorigin=True, #when splitting nodes, mark origin: VP_2 => {VP*1, VP*2}
	splitprune=False, #VP_2[101] is treated as { VP*[100], VP*[001] } during parsing
	removeparentannotation=False, # VP^<S> is treated as VP
	neverblockre=None, #do not prune nodes with label that match regex
	quiet=False, reallyquiet=False, #quiet=no per sentence results
	numproc=1,
	resultdir="output"
	):
	from treetransforms import slowfanout
	def treebankfanout(trees):
		return max((slowfanout(a), n) for n, tree in enumerate(trees)
			for a in tree.subtrees(lambda x: len(x) > 1))

	# Tiger treebank version 2 sample:
	# http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/annotation/sample2.export
	#corpus = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1"); testmaxwords = 99
	assert bintype in ("optimal", "optimalhead", "binarize", "nltk")
	assert estimator in ("dop1", "ewe", "shortest", "sl-dop", "sl-dop-simple", "srcg")
	os.mkdir(resultdir)
	# Log everything, and send it to stderr, in a format with just the message.
	format = '%(message)s'
	if reallyquiet: logging.basicConfig(level=logging.WARNING, format=format)
	elif quiet: logging.basicConfig(level=logging.INFO, format=format)
	else: logging.basicConfig(level=logging.DEBUG, format=format)

	# log up to INFO to a results log file
	file = logging.FileHandler(filename='%s/output.log' % resultdir)
	file.setLevel(logging.INFO)
	file.setFormatter(logging.Formatter(format))
	logging.getLogger('').addHandler(file)

	corpus = NegraCorpusReader(corpusdir, corpusfile, encoding=encoding,
		headorder=(bintype in ("binarize", "optimalhead")),
		headfinal=True, headreverse=False, unfold=unfolded,
		movepunct=movepunct, removepunct=removepunct)
	logging.info("%d sentences in corpus %s/%s" % (
			len(corpus.parsed_sents()), corpusdir, corpusfile))
	if isinstance(trainsents, float):
		trainsents = int(trainsents * len(corpus.sents()))
	trees, sents, blocks = corpus.parsed_sents()[:trainsents], corpus.sents()[:trainsents], corpus.blocks()[:trainsents]
	logging.info("%d training sentences before length restriction" % len(trees))
	trees, sents, blocks = zip(*[sent for sent in zip(trees, sents, blocks) if len(sent[1]) <= trainmaxwords])
	logging.info("%d training sentences after length restriction <= %d" % (len(trees), trainmaxwords))

	# parse training corpus as a "soundness check"
	#test = corpus.parsed_sents(), corpus.tagged_sents(), corpus.blocks()
	test = NegraCorpusReader(corpusdir, corpusfile, encoding=encoding,
			removepunct=removepunct, movepunct=movepunct)
	test = (test.parsed_sents()[trainsents+skip:trainsents+skip+testsents],
			test.tagged_sents()[trainsents+skip:trainsents+skip+testsents],
			test.blocks()[trainsents+skip:trainsents+skip+testsents])
	assert len(test[0]), "test corpus should be non-empty"

	f, n = treebankfanout(trees)
	logging.info("%d test sentences before length restriction" % len(test[0]))
	test = zip(*((a,b,c) for a,b,c in zip(*test) if len(b) <= testmaxwords))
	logging.info("%d test sentences after length restriction <= %d" % (len(test[0]), testmaxwords))
	logging.info("treebank fan-out before binarization: %d #%d" % (f, n))
	logging.info("read training & test corpus")

	# binarization
	begin = time.clock()
	if arity_marks_before_bin: [srcg_productions(a, b) for a, b in zip(trees, sents)]
	if bintype == "nltk":
		bintype += " %s h=%d v=%d" % (factor, h, v)
		for a in trees: a.chomsky_normal_form(factor="right", vertMarkov=v-1, horzMarkov=h)
	elif bintype == "binarize":
		bintype += " %s h=%d v=%d %s" % (factor, h, v, "tailmarker" if tailmarker else '')
		[binarize(a, factor=factor, vertMarkov=v-1, horzMarkov=h,
				tailMarker=tailmarker,
				leftMostUnary=True, rightMostUnary=True,
				#leftMostUnary=False, rightMostUnary=False,
				reverse=revmarkov) for a in trees]
	elif bintype == "optimal":
		trees = [Tree.convert(optimalbinarize(tree))
						for n, tree in enumerate(trees)]
	elif bintype == "optimalhead":
		trees = [Tree.convert(
					optimalbinarize(tree, headdriven=True, h=h, v=v))
						for n,tree in enumerate(trees)]
	logging.info("binarized %s cpu time elapsed: %gs" % (
						bintype, time.clock() - begin))
	logging.info("binarized treebank fan-out: %d #%d" % treebankfanout(trees))

	#cycledetection()
	pcfggrammar = []; srcggrammar = []; dopgrammar = []; secondarymodel = []
	backtransform = None
	if splitpcfg:
		splittrees = [splitdiscnodes(a.copy(True), markorigin) for a in trees]
		logging.info("splitted discontinuous nodes")
		# second layer of binarization:
		for a in splittrees:
			a.chomsky_normal_form(childChar=":")
		pcfggrammar = induce_srcg(splittrees, sents)
		logging.info("induced CFG based on %d sentences" % len(splittrees))
		#srcggrammar = dop_srcg_rules(splittrees, sents)
		#logging.info("induced DOP CFG based on %d sentences" % len(trees))
		logging.info(grammarinfo(pcfggrammar))
		pcfggrammar = Grammar(pcfggrammar)
	if srcg:
		srcggrammar = induce_srcg(trees, sents)
		logging.info("induced SRCG based on %d sentences" % len(trees))
		logging.info(grammarinfo(srcggrammar, dump="%s/pcdist.txt" % resultdir))
		write_srcg_grammar(srcggrammar,
			"%s/grammar.srcg"   % resultdir,
			"%s/grammar.lex" % resultdir,
		encoding='utf-8')
		srcggrammar = Grammar(srcggrammar)
		srcggrammar.testgrammar()
		if neverblockre: srcggrammar.getdonotprune(re.compile(neverblockre))
		logging.info("wrote grammar to %s/grammar.{srcg,lex}" % resultdir)
	if dop:
		if estimator == "shortest":
			# the secondary model is used to resolve ties for the shortest derivation
			dopgrammar, secondarymodel = dop_srcg_rules(list(trees), list(sents),
				normalize=False, shortestderiv=True,	arity_marks=arity_marks)
		elif "sl-dop" in estimator:
			dopgrammar = dop_srcg_rules(list(trees), list(sents), normalize=True,
							shortestderiv=False,	arity_marks=arity_marks)
			dopshortest, _ = dop_srcg_rules(list(trees), list(sents),
							normalize=False, shortestderiv=True,
							arity_marks=arity_marks)
			secondarymodel = Grammar(dopshortest)
		elif estimator == "srcg":
			# hack to have srcg instead of dop grammar as fine stage
			dopgrammar = induce_srcg(trees, sents)
			logging.info("induced SRCG based on %d sentences" % len(trees))
		elif usedoubledop:
			assert estimator not in ("ewe", "sl-dop", "sl-dop-simple", "shortest")
			dopgrammar, backtransform = doubledop(list(trees), list(sents),
					stroutput=True, debug=False)
		else:
			dopgrammar = dop_srcg_rules(list(trees), list(sents),
				normalize=(estimator in ("ewe", "sl-dop", "sl-dop-simple")),
				shortestderiv=False, arity_marks=arity_marks)
		nodes = sum(len(list(a.subtrees())) for a in trees)
		dopgrammar1 = Grammar(dopgrammar)
		if estimator != "srcg":
			logging.info("DOP model based on %d sentences, %d nodes, %d nonterminals"
						% (len(trees), nodes, len(dopgrammar1.toid)))
		logging.info(grammarinfo(dopgrammar))
		dopgrammar = dopgrammar1
		dopgrammar.testgrammar()
		if neverblockre: dopgrammar.getdonotprune(re.compile(neverblockre))

	if getestimates:
		from estimates import getestimates
		import numpy as np
		logging.info("computing estimates")
		begin = time.clock()
		outside = getestimates(srcggrammar, testmaxwords, srcggrammar.toid["ROOT"])
		logging.info("estimates done. cpu time elapsed: %gs"
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
	if numproc == 1: xdoparse = doparse
	else: xdoparse = multidoparse
	results = xdoparse(splitpcfg, srcg, dop, estimator, unfolded, bintype,
			sample, both, arity_marks, arity_marks_before_bin, m, pcfggrammar,
			srcggrammar, dopgrammar, secondarymodel, test, testmaxwords, testsents,
			prune, splitk, k, sldop_n, useestimates, outside, "ROOT", True,
			removeparentannotation, splitprune, mergesplitnodes, markorigin,
			resultdir, usecfgparse, backtransform)
	logging.info("time elapsed during parsing: %gs" % (time.clock() - begin))
	doeval(*results)

def doparse(splitpcfg, srcg, dop, estimator, unfolded, bintype,
			sample, both, arity_marks, arity_marks_before_bin, m, pcfggrammar,
			srcggrammar, dopgrammar, secondarymodel, test, testmaxwords, testsents,
			prune, splitk, k, sldop_n=14, useestimates=False, outside=None,
			top='ROOT',	tags=True, removeparentannotation=False,
			splitprune=False, mergesplitnodes=False, markorigin=False,
			resultdir="results", usecfgparse=False, backtransform=None,
			category=None, sentinit=0, doph=999):
	presults = []; sresults = []; dresults = []
	gold = []; gsent = []
	pcandb = multiset(); scandb = multiset(); dcandb = multiset()
	goldbrackets = multiset()
	nsent = exactp = exactd = exacts = pnoparse = snoparse = dnoparse =  0
	#if srcg: derivout = codecs.open("srcgderivations", "w", encoding='utf-8')
	timesfile = open("%s/parsetimes.txt" % resultdir, "w")
	# main parse loop over each sentence in test corpus
	for tree, sent, block in zip(*test):
		if len(sent) > testmaxwords: continue
		if nsent >= testsents: break
		nsent += 1
		logging.debug("%d. [len=%d] %s" % (nsent, len(sent),
					u" ".join(a[0] for a in sent)))			# words only
					#u" ".join(a[0]+u"/"+a[1] for a in sent))) word/TAG
		goldb = bracketings(tree)
		gold.append(block)
		gsent.append(sent)
		goldbrackets.update((nsent, a) for a in goldb.elements())
		msg = ''
		if splitpcfg and len(sent) < maxbitveclen: # hard limit of word sized bit vectors
			msg = "PCFG: "
			begin = time.clock()
			if usecfgparse:
				chart, start = cfgparse([w for w, t in sent],
						pcfggrammar,
						tags=[t for w, t in sent] if tags else [],
						start=pcfggrammar.toid[top])
			else:
				chart, start, msg1 = parse([w for w, t in sent],
						pcfggrammar,
						tags=[t for w, t in sent] if tags else [],
						start=pcfggrammar.toid[top],
						exhaustive=True)
				msg += msg1
		else: chart = {}; start = False
		if start:
			resultstr, prob = viterbiderivation(chart, start,
								pcfggrammar.tolabel)
			result = Tree(resultstr)
			result.un_chomsky_normal_form(childChar=":")
			mergediscnodes(result)
			unbinarize(result)
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
				exactp += 1
			else:
				msg += "LP %5.2f LR %5.2f LF %5.2f\n" % (
								100 * prec, 100 * rec, 100 * f1)
				if candb - goldb:
					msg += "cand-gold=%s " % printbrackets(candb - goldb)
				if goldb - candb:
					msg += "gold-cand=%s" % printbrackets(goldb - candb)
				if (candb - goldb) or (goldb - candb): msg += '\n'
				msg += "      %s" % result.pprint(margin=1000)
			presults.append(result)
		else:
			if splitpcfg: msg += "no parse"
			result = baseline([(n,t) for n,(w, t) in enumerate(sent)])
			result = Tree.parse("(%s %s)" % (top, result), parse_leaf=int)
			candb = bracketings(result)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			pnoparse += 1
			presults.append(result)
		if splitpcfg:
			pcfgtime = time.clock() - begin
			logging.debug(msg+"\n%.2fs cpu time elapsed" % (pcfgtime))
		pcandb.update((nsent, a) for a in candb.elements())
		msg = ""
		if srcg and (not splitpcfg or start) and len(sent) < maxbitveclen: # hard limit
			msg = "SRCG: "
			begin = time.clock()
			whitelist = []
			if splitpcfg and start:
				if splitprune:
					pchart = kbest_items(chart, start, splitk)
					logging.debug("whitelist obtained: %gs; before: %d; after: %d" % (
						time.clock() - begin, len(chart), len(pchart)))
				else:
					whitelist = prunechart(chart, start, pcfggrammar,
								srcggrammar, splitk, removeparentannotation,
								mergesplitnodes and not splitprune, 999)
					logging.debug(
						"whitelist obtained: %gs; before: %d; after: %d" % (
						time.clock() - begin, len(chart), sum(map(len, whitelist))))
			chart, start, msg1 = parse(
						[w for w, t in sent], srcggrammar,
						tags=[t for w, t in sent] if tags else [],
						start=srcggrammar.toid[top],
						whitelist=whitelist,
						coarsechart=pchart if (splitprune and splitpcfg) else None,
						coarsegrammar=pcfggrammar if splitpcfg else None,
						splitprune=splitprune and splitpcfg,
						markorigin=markorigin,
						exhaustive=dop and prune,
						estimate=(outside, testmaxwords) if useestimates else None,
						)
			msg += msg1
		elif srcg: chart = {}; start = False
		if start and srcg:
			resultstr, prob = viterbiderivation(chart, start, srcggrammar.tolabel)
			result = Tree(resultstr)
			unbinarize(result)
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
					msg += "cand-gold=%s " % printbrackets(candb - goldb)
				if goldb - candb:
					msg += "gold-cand=%s" % printbrackets(goldb - candb)
				if (candb - goldb) or (goldb - candb): msg += '\n'
				msg += "      %s" % result.pprint(margin=1000)
			sresults.append(result)
		else:
			if srcg: msg += "no parse"
			result = baseline([(n,t) for n,(w, t) in enumerate(sent)])
			result = Tree.parse("(%s %s)" % (top, result), parse_leaf=int)
			candb = bracketings(result)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			snoparse += 1
			sresults.append(result)
		if srcg:
			srcgtime = time.clock() - begin
			logging.debug(msg+"\n%.2fs cpu time elapsed" % (srcgtime))
		scandb.update((nsent, a) for a in candb.elements())
		msg = ""
		if (dop and (not (splitpcfg or srcg) or start)
				and len(sent) < maxbitveclen): # hard limit
			msg = " DOP: "
			begin = time.clock()
			whitelist = []
			if (splitpcfg or srcg) and prune and start:
				if splitprune and not srcg:
					chart = kbest_items(chart, start, k)
					logging.debug("whitelist obtained: %gs; before: %d; after: %d" % (
						time.clock() - begin, len(chart), len(chart)))
				else:
					whitelist = prunechart(chart, start, srcggrammar,
								dopgrammar, k, removeparentannotation,
								mergesplitnodes and not srcg, doph)
					logging.debug(
						"whitelist obtained: %gs; before: %d; after: %d" % (
						time.clock() - begin, len(chart), sum(map(len, whitelist))))
				# dump prune list
				#for a, b in enumerate(whitelist):
				#	print dopgrammar.tolabel[a],
				#	if b:
				#		for c,d in b.items():
				#			print bin(c), exp(-d)
				#	else: print {}
			chart, start, msg1 = parse(
							[w for w, _ in sent], dopgrammar,
							tags=[t for _, t in sent] if tags else None,
							start=dopgrammar.toid[top],
							exhaustive=True,
							estimate=None,
							whitelist=whitelist,
							coarsechart=chart if (splitprune and not srcg) else None,
							coarsegrammar=srcggrammar if srcg else (
								pcfggrammar if splitpcfg else None),
							splitprune=splitprune and not srcg,
							markorigin=markorigin)
			msg += msg1
			if srcg and not start:
				from plcfrs import pprint_chart
				pprint_chart(chart,
						[w.encode('unicode-escape') for w, _ in sent],
						dopgrammar.tolabel)
				raise ValueError("expected successful parse")
		else: chart = {}; start = False
		if dop and start:
			if estimator == "shortest":
				mpp, msg1 = marginalize(chart, start, dopgrammar.tolabel, n=m,
						sample=sample, both=both, shortest=True,
						secondarymodel=secondarymodel)
				mpp = mpp.items()
			elif estimator == "sl-dop":
				mpp, msg1 = sldop(chart, start, sent, tags, dopgrammar,
							secondarymodel, m, sldop_n, sample, both)
			elif estimator == "sl-dop-simple":
				# old method, estimate shortest derivation directly from number
				# of addressed nodes
				mpp, msg1 = sldop_simple(chart, start, dopgrammar, m, sldop_n)
			elif backtransform is not None:
				mpp, msg1 = marginalize(chart, start, dopgrammar.tolabel, n=m,
					sample=sample, both=both, backtransform=backtransform)
				mpp = mpp.items()
			else: #dop1, ewe
				mpp, msg1 = marginalize(chart, start, dopgrammar.tolabel,
									n=m, sample=sample, both=both)
				mpp = mpp.items()
			logging.debug(msg1)
			dresult, prob = max(mpp, key=itemgetter(1))
			dresult = Tree(dresult)
			if isinstance(prob, tuple):
				msg += "subtrees = %d, p = %.4e " % (abs(prob[0]), prob[1])
			else:
				msg += "p = %.4e " % prob
			unbinarize(dresult)
			rem_marks(dresult)
			if unfolded: fold(dresult)
			candb = bracketings(dresult)
			try: prec = precision(goldb, candb)
			except ZeroDivisionError:
				prec = 0.0
				logging.warning("empty candidate brackets?\n%s" % dresult.pprint())
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			if dresult == tree or f1 == 1.0:
				msg += "exact match"
				exactd += 1
			else:
				msg += "LP %5.2f LR %5.2f LF %5.2f\n" % (
								100 * prec, 100 * rec, 100 * f1)
				if candb - goldb:
					msg += "cand-gold=%s " % printbrackets(candb - goldb)
				if goldb - candb:
					msg += "gold-cand=%s " % printbrackets(goldb - candb)
				if (candb - goldb) or (goldb - candb): msg += '\n'
				msg += "      %s" % dresult.pprint(margin=1000)
			dresults.append(dresult)
		else:
			if dop:
				msg += "no parse"
			dresult = baseline([(n,t) for n,(w, t) in enumerate(sent)])
			dresult = Tree.parse("(%s %s)" % (top, dresult), parse_leaf=int)
			candb = bracketings(dresult)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			dnoparse += 1
			dresults.append(dresult)
		doptime = 0
		if dop:
			doptime = time.clock() - begin
			logging.debug(msg+"\n%.2fs cpu time elapsed" % (doptime))
		timesfile.write("%d\t%d" % (nsent, len(sent)))
		if splitpcfg: timesfile.write("\t%g" % pcfgtime)
		if srcg: timesfile.write("\t%g" % srcgtime)
		if dop: timesfile.write("\t%g" % doptime)
		timesfile.write("\n")
		logging.debug("GOLD: %s" % tree.pprint(margin=1000))
		dcandb.update((nsent, a) for a in candb.elements())
		if splitpcfg:
			logging.debug("pcfg cov %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f%s" % (
								100 * (1 - pnoparse/float(len(presults))),
								100 * (exactp / float(nsent)),
								100 * precision(goldbrackets, pcandb),
								100 * recall(goldbrackets, pcandb),
								100 * f_measure(goldbrackets, pcandb),
								('' if srcg or dop else '\n')))
		if srcg:
			logging.debug("srcg cov %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f%s" % (
								100 * (1 - snoparse/float(len(sresults))),
								100 * (exacts / float(nsent)),
								100 * precision(goldbrackets, scandb),
								100 * recall(goldbrackets, scandb),
								100 * f_measure(goldbrackets, scandb),
								('' if dop else '\n')))
		if dop:
			logging.debug("dop  cov %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f"
					" (%+5.2f)\n" % (
								100 * (1 - dnoparse/float(len(dresults))),
								100 * (exactd / float(nsent)),
								100 * precision(goldbrackets, dcandb),
								100 * recall(goldbrackets, dcandb),
								100 * f_measure(goldbrackets, dcandb),
								100 * (f_measure(goldbrackets, dcandb) -
									max(f_measure(goldbrackets, pcandb),
									f_measure(goldbrackets, scandb)))))

	if splitpcfg:
		codecs.open("%s/%s.pcfg" % (resultdir, category or "results"),
			"w", encoding='utf-8').writelines(export(a, [w for w,_ in b],n + 1)
			for n,a,b in zip(count(sentinit), presults, gsent))
	if srcg:
		codecs.open("%s/%s.srcg" % (resultdir, category or "results"),
			"w", encoding='utf-8').writelines(export(a, [w for w,_ in b],n + 1)
			for n,a,b in zip(count(sentinit), sresults, gsent))
	if dop:
		codecs.open("%s/%s.dop" % (resultdir, category or "results"),
			"w", encoding='utf-8').writelines(export(a, [w for w,_ in b], n + 1)
			for n,a,b in zip(count(sentinit), dresults, gsent))
	codecs.open("%s/%s.gold" % (resultdir, category or "results"),
			"w", encoding='utf-8').write(''.join(
				"#BOS %d\n%s#EOS %d\n" % (n + 1, a, n + 1)
				for n, a in zip(count(sentinit), gold)))
	timesfile.close()
	logging.info("wrote results to %s/%s.{gold,srcg,dop}" % (resultdir, category or "results"))

	return (splitpcfg, srcg, dop, nsent, testmaxwords, exactp, exacts, exactd,
		pnoparse, snoparse, dnoparse,goldbrackets, pcandb, scandb, dcandb,
		unfolded, arity_marks, bintype, estimator, sldop_n)

def doeval(splitpcfg, srcg, dop, nsent, testmaxwords, exactp, exacts, exactd,
		pnoparse, snoparse, dnoparse, goldbrackets, pcandb, scandb, dcandb,
		unfolded, arity_marks, bintype, estimator, sldop_n):
	print "testmaxwords", testmaxwords, "unfolded", unfolded, "arity marks", arity_marks,
	print "binarized", bintype, "estimator", estimator,
	if dop:
		if'sl-dop' in estimator: print sldop_n
		if doubledop: print "doubledop"
	if splitpcfg and nsent:
		logging.info("pcfg lp %5.2f lr %5.2f lf %5.2f\n"
			"coverage %d / %d = %5.2f %%  exact match %d / %d = %5.2f %%\n" % (
				100 * precision(goldbrackets, pcandb),
				100 * recall(goldbrackets, pcandb),
				100 * f_measure(goldbrackets, pcandb),
				nsent - pnoparse, nsent, 100.0 * (nsent - pnoparse) / nsent,
				exactp, nsent, 100.0 * exactp / nsent))
	if srcg and nsent:
		logging.info("srcg lp %5.2f lr %5.2f lf %5.2f\n"
			"coverage %d / %d = %5.2f %%  exact match %d / %d = %5.2f %%\n" % (
				100 * precision(goldbrackets, scandb),
				100 * recall(goldbrackets, scandb),
				100 * f_measure(goldbrackets, scandb),
				nsent - snoparse, nsent, 100.0 * (nsent - snoparse) / nsent,
				exacts, nsent, 100.0 * exacts / nsent))
	if dop and nsent:
		logging.info("dop  lp %5.2f lr %5.2f lf %5.2f\n"
			"coverage %d / %d = %5.2f %%  exact match %d / %d = %5.2f %%\n" % (
				100 * precision(goldbrackets, dcandb),
				100 * recall(goldbrackets, dcandb),
				100 * f_measure(goldbrackets, dcandb),
				nsent - dnoparse, nsent, 100.0 * (nsent - dnoparse) / nsent,
				exactd, nsent, 100.0 * exactd / nsent))

def root(tree):
	if tree.node == "VROOT": tree.node = "ROOT"
	else: tree = Tree("ROOT",[tree])
	return tree

def cftiger():
	#read_penn_format('../tiger/corpus/tiger_release_aug07.mrg')
	grammar = read_bitpar_grammar('/tmp/gtigerpcfg.pcfg', '/tmp/gtigerpcfg.lex')
	dopgrammar = read_bitpar_grammar('/tmp/gtiger.pcfg', '/tmp/gtiger.lex',
			ewe=False)
	grammar.testgrammar()
	dopgrammar.testgrammar()
	dop = True; srcg = True; unfolded = False; bintype = "binarize h=1 v=1"
	sample = False; both = False; arity_marks = True
	arity_marks_before_bin = False; estimator = 'sl-dop'; m = 10000;
	testmaxwords = 15; testsents = 360
	prune = False; top = "ROOT"; tags = False; sldop_n = 5
	trees = list(islice((a for a in islice((root(Tree(a))
					for a in codecs.open(
							'../tiger/corpus/tiger_release_aug07.mrg',
							encoding='iso-8859-1')), 7200, 9600)
				if len(a.leaves()) <= testmaxwords), testsents))
	lex = set(wt for tree in (root(Tree(a))
					for a in islice(codecs.open(
						'../tiger/corpus/tiger_release_aug07.mrg',
						encoding='iso-8859-1'), 7200))
				if len(tree.leaves()) <= testmaxwords for wt in tree.pos())
	sents = [[(t + '_' + (w if (w, t) in lex else ''), t)
						for w, t in a.pos()] for a in trees]
	for tree in trees:
		for nn, a in enumerate(tree.treepositions('leaves')):
			tree[a] = nn
	blocks = [export(*a) for a in zip(trees,
		([w for w,_ in s] for s in sents), count())]
	test = trees, sents, blocks
	doparse(srcg, dop, estimator, unfolded, bintype, sample, both, arity_marks,
		arity_marks_before_bin, m, grammar, dopgrammar, test, testmaxwords, testsents,
		prune, sldop_n, top, tags)

def cycledetection(trees, sents):
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
	bintype="binarize", h=1, v=1, factor="right", doph=1, arity_marks=True,
	arity_marks_before_bin=False, sample=False, both=False, m=10000,
	trainmaxwords=999, testmaxwords=40, testsents=999, k=1000, prune=True, sldop_n=7,
	removeparentannotation=False, splitprune=False, mergesplitnodes=True,
	neverblockre=None, markorigin = True, resultdir="tepacoc-40k1000"):

	format = '%(message)s'
	logging.basicConfig(level=logging.DEBUG, format=format)
	tepacocids, tepacocsents = readtepacoc()
	#corpus = NegraCorpusReader("../rparse", "tigerprocfullnew.export",
	#		headorder=(bintype in ("binarize", "nltk")), headfinal=True,
	#		headreverse=False, unfold=unfolded)
	#corpus_sents = list(corpus.sents())
	#corpus_taggedsents = list(corpus.tagged_sents())
	#corpus_trees = list(corpus.parsed_sents())
	#corpus_blocks = list(corpus.blocks())
	#thecorpus = [a for a in zip(corpus_sents, corpus_taggedsents, corpus_trees,
	#	corpus_blocks)]
	#cPickle.dump(thecorpus, open("tiger.pickle", "wb"), protocol=-1)
	corpus_sents, corpus_taggedsents, corpus_trees, corpus_blocks = zip(
		*cPickle.load(open("tiger.pickle", "rb")))
	trainsents = 25005 #int(0.9 * len(corpus_sents))
	trees, sents, blocks = zip(*[sent for n, sent in
				enumerate(zip(corpus_trees, corpus_sents,
							corpus_blocks)) if len(sent[1]) <= trainmaxwords
							and n not in tepacocids][:trainsents])
	begin = time.clock()
	if bintype == "optimal":
		trees = [optimalbinarize(tree) for n, tree in enumerate(trees)]
	elif bintype == "nltk":
		for a in trees:
			a.chomsky_normal_form(factor="right", vertMarkov=v-1, horzMarkov=h)
	elif bintype == "binarize":
		[binarize(a, factor=factor, vertMarkov=v-1, horzMarkov=h, tailMarker="",
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
	logging.info(grammarinfo(srcggrammar))
	srcggrammar = Grammar(srcggrammar)
	srcggrammar.testgrammar()

	if removeparentannotation:
		for a in trees:
			a.chomsky_normal_form(factor="right", horzMarkov=doph)
	dopgrammar = dop_srcg_rules(list(trees), list(sents),
				normalize=(estimator in ("ewe", "sl-dop", "sl-dop-simple")),
				shortestderiv=False, arity_marks=arity_marks)
	print "induced dop reduction of", len(sents), "sentences"
	logging.info(grammarinfo(dopgrammar))
	dopgrammar = Grammar(dopgrammar)
	dopgrammar.testgrammar()
	if neverblockre: dopgrammar.getdonotprune(re.compile(neverblockre))
	secondarymodel = []
	os.mkdir(resultdir)

	results = {}
	testset = {}
	cnt = 0
	for cat, catsents in tepacocsents.iteritems():
		print "category:", cat,
		trees, sents, blocks = [], [], []
		test = trees, sents, blocks
		for n, sent in catsents:
			corpus_sent = corpus_sents[n]
			if len(corpus_sent) <= testmaxwords:
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
					dopgrammar, secondarymodel, test, testmaxwords, testsents, prune,
					50, k, sldop_n, False, None, "ROOT", True,
					removeparentannotation, splitprune, mergesplitnodes,
					markorigin, resultdir=resultdir, category=cat,
					sentinit=cnt) #, doph=doph if doph != h else 999)
		cnt += len(test[0])
		print "time elapsed during parsing: ", time.clock() - begin
	goldbrackets = multiset(); scandb = multiset(); dcandb = multiset()
	exactd = exacts = snoparse = dnoparse = 0
	for cat, res in results.iteritems():
		print "category:", cat
		exactd += res[8]
		exacts += res[9]
		snoparse += res[10]
		dnoparse += res[11]
		goldbrackets |= res[12]
		scandb |= res[13]
		dcandb |= res[14]
		doeval(*res)
	print "TOTAL"
	doeval(True, True, {}, {}, {}, {}, cnt, testmaxwords, exactd, exacts, snoparse,
			dnoparse, goldbrackets, scandb, dcandb, False, arity_marks,
			bintype, estimator, sldop_n)

def worker((nsent, tree, sent, block)):
	""" parse a sentence using pcfg, lcfrs, dop """
	d = internalparams
	goldb = bracketings(tree)
	pnoparse = snoparse = dnoparse = False
	exactp = exacts = exactd = False
	msg = ''
	if d.splitpcfg:
		msg += "PCFG: "
		begin = time.clock()
		if d.usecfgparse:
			chart, start = cfgparse([w for w, t in sent],
					d.pcfggrammar,
					tags=[t for w, t in sent] if d.tags else [],
					start=d.pcfggrammar.toid[d.top])
		else:
			chart, start, msg1 = parse([w for w, t in sent],
					d.pcfggrammar,
					tags=[t for w, t in sent] if d.tags else [],
					start=d.pcfggrammar.toid[d.top],
					exhaustive=True)
			msg += msg1
	else: chart = {}; start = False
	if start:
		resultstr, prob = viterbiderivation(chart, start,
							d.pcfggrammar.tolabel)
		result = Tree(resultstr)
		result.un_chomsky_normal_form(childChar=":")
		mergediscnodes(result)
		unbinarize(result)
		rem_marks(result)
		if d.unfolded: fold(result)
		msg += "\np = %.4e " % exp(-prob)
		candb = bracketings(result)
		prec = precision(goldb, candb)
		rec = recall(goldb, candb)
		f1 = f_measure(goldb, candb)
		if result == tree or f1 == 1.0:
			assert result != tree or f1 == 1.0
			msg += "exact match"
			exactp = True
		else:
			msg += "LP %5.2f LR %5.2f LF %5.2f\n" % (
							100 * prec, 100 * rec, 100 * f1)
			if candb - goldb:
				msg += "cand-gold=%s " % printbrackets(candb - goldb)
			if goldb - candb:
				msg += "gold-cand=%s" % printbrackets(goldb - candb)
			if (candb - goldb) or (goldb - candb): msg += '\n'
			msg += "      %s" % result.pprint(margin=1000)
	else:
		if d.splitpcfg: msg += "no parse"
		result = baseline([(n,t) for n,(w, t) in enumerate(sent)])
		result = Tree.parse("(%s %s)" % (d.top, result), parse_leaf=int)
		candb = bracketings(result)
		prec = precision(goldb, candb)
		rec = recall(goldb, candb)
		f1 = f_measure(goldb, candb)
		pnoparse = True
	presult = result
	pcandb = candb
	if d.splitpcfg:
		pcfgtime = time.clock() - begin
		msg += "\n%.2fs cpu time elapsed\n" % (pcfgtime)
	if d.srcg and (not d.splitpcfg or start):
		begin = time.clock()
		whitelist = []
		if d.splitpcfg and start:
			if d.splitprune:
				pchart = kbest_items(chart, start, d.splitk)
				msg += "whitelist obtained: %gs; before: %d; after: %d\n" % (
					time.clock() - begin, len(chart), len(pchart))
			else:
				whitelist = prunechart(chart, start, d.pcfggrammar,
							d.srcggrammar, d.splitk, d.removeparentannotation,
							d.mergesplitnodes and not d.splitprune, 999)
				msg += "whitelist obtained: %gs; before: %d; after: %d\n" % (
					time.clock() - begin, len(chart), sum(map(len, whitelist)))
		chart, start, msg1 = parse(
					[w for w, t in sent], d.srcggrammar,
					tags=[t for w, t in sent] if d.tags else [],
					start=d.srcggrammar.toid[d.top],
					whitelist=whitelist,
					coarsechart=pchart if (d.splitprune and d.splitpcfg) else None,
					coarsegrammar=d.pcfggrammar if d.splitpcfg else None,
					splitprune=d.splitprune and d.splitpcfg,
					markorigin=d.markorigin,
					exhaustive=d.dop and d.prune,
					estimate=(d.outside, d.testmaxwords) if d.useestimates else None,
					)
		msg += "SRCG: " + msg1
	elif d.srcg: chart = {}; start = False
	if start and d.srcg:
		resultstr, prob = viterbiderivation(chart, start, d.srcggrammar.tolabel)
		result = Tree(resultstr)
		unbinarize(result)
		rem_marks(result)
		if d.unfolded: fold(result)
		msg += "\np = %.4e " % exp(-prob)
		candb = bracketings(result)
		prec = precision(goldb, candb)
		rec = recall(goldb, candb)
		f1 = f_measure(goldb, candb)
		if result == tree or f1 == 1.0:
			assert result != tree or f1 == 1.0
			msg += "exact match"
			exacts = True
		else:
			msg += "LP %5.2f LR %5.2f LF %5.2f\n" % (
							100 * prec, 100 * rec, 100 * f1)
			if candb - goldb:
				msg += "cand-gold=%s " % printbrackets(candb - goldb)
			if goldb - candb:
				msg += "gold-cand=%s" % printbrackets(goldb - candb)
			if (candb - goldb) or (goldb - candb): msg += '\n'
			msg += "      %s" % result.pprint(margin=1000)
	else:
		if d.srcg: msg += "no parse"
		result = baseline([(n,t) for n,(w, t) in enumerate(sent)])
		result = Tree.parse("(%s %s)" % (d.top, result), parse_leaf=int)
		candb = bracketings(result)
		prec = precision(goldb, candb)
		rec = recall(goldb, candb)
		f1 = f_measure(goldb, candb)
		snoparse = True
	sresult = result
	scandb = candb
	if d.srcg:
		msg += "\n%.2fs cpu time elapsed\n" % (time.clock() - begin)
	if d.dop and (not (d.splitpcfg or d.srcg) or start):
		begin = time.clock()
		whitelist = []
		if (d.splitpcfg or d.srcg) and d.prune and start:
			if d.splitprune and not d.srcg:
				chart = kbest_items(chart, start, d.k)
				msg += "whitelist obtained: %gs; before: %d; after: %d\n" % (
					time.clock() - begin, len(chart), len(chart))
			else:
				whitelist = prunechart(chart, start, d.srcggrammar,
							d.dopgrammar, d.k, d.removeparentannotation,
							d.mergesplitnodes and not d.srcg, d.doph)
				msg += "whitelist obtained: %gs; before: %d; after: %d\n" % (
					time.clock() - begin, len(chart), sum(map(len, whitelist)))
		chart, start, msg1 = parse(
						[w for w, _ in sent], d.dopgrammar,
						tags=[t for _, t in sent] if d.tags else None,
						start=d.dopgrammar.toid[d.top],
						exhaustive=True,
						estimate=None,
						whitelist=whitelist,
						coarsechart=chart if (d.splitprune and not d.srcg) else None,
						coarsegrammar=d.srcggrammar if d.srcg else (
							d.pcfggrammar if d.splitpcfg else None),
						splitprune=d.splitprune and not d.srcg,
						markorigin=d.markorigin)
		msg += " DOP: " + msg1
		if d.srcg and not start:
			from plcfrs import pprint_chart
			pprint_chart(chart,
					[w.encode('unicode-escape') for w, _ in sent],
					d.dopgrammar.tolabel)
			raise ValueError("expected successful parse")
	else: chart = {}; start = False
	if d.dop and start:
		if d.estimator == "shortest":
			mpp, msg1 = marginalize(chart, start, d.dopgrammar.tolabel, n=d.m,
					sample=d.sample, both=d.both, shortest=True,
					secondarymodel=d.secondarymodel)
			mpp = mpp.items()
		elif d.estimator == "sl-dop":
			mpp, msg1 = sldop(chart, start, sent, d.tags, d.dopgrammar,
						d.secondarymodel, d.m, d.sldop_n, d.sample, d.both)
		elif d.estimator == "sl-dop-simple":
			# old method, estimate shortest derivation directly from number
			# of addressed nodes
			mpp, msg1 = sldop_simple(chart, start, d.dopgrammar, d.m, d.sldop_n)
		elif d.backtransform is not None:
			mpp, msg1 = marginalize(chart, start, d.dopgrammar.tolabel, n=d.m,
				sample=d.sample, both=d.both, backtransform=d.backtransform)
			mpp = mpp.items()
		else: #dop1, ewe
			mpp, msg1 = marginalize(chart, start, d.dopgrammar.tolabel,
								n=d.m, sample=d.sample, both=d.both)
			mpp = mpp.items()
		msg += msg1
		dresult, prob = max(mpp, key=itemgetter(1))
		dresult = Tree(dresult)
		if isinstance(prob, tuple):
			msg += "\nsubtrees = %d, p = %.4e " % (abs(prob[0]), prob[1])
		else:
			msg += "\np = %.4e " % prob
		unbinarize(dresult)
		rem_marks(dresult)
		if d.unfolded: fold(dresult)
		candb = bracketings(dresult)
		try: prec = precision(goldb, candb)
		except ZeroDivisionError:
			prec = 0.0
			logging.warning("empty candidate brackets?\n%s" % dresult.pprint())
		rec = recall(goldb, candb)
		f1 = f_measure(goldb, candb)
		if dresult == tree or f1 == 1.0:
			msg += "exact match"
			exactd = True
		else:
			msg += "LP %5.2f LR %5.2f LF %5.2f\n" % (
							100 * prec, 100 * rec, 100 * f1)
			if candb - goldb:
				msg += "cand-gold=%s " % printbrackets(candb - goldb)
			if goldb - candb:
				msg += "gold-cand=%s " % printbrackets(goldb - candb)
			if (candb - goldb) or (goldb - candb): msg += '\n'
			msg += "      %s" % dresult.pprint(margin=1000)
	else:
		if d.dop:
			msg += "no parse\n"
		dresult = baseline([(n,t) for n,(w, t) in enumerate(sent)])
		dresult = Tree.parse("(%s %s)" % (d.top, dresult), parse_leaf=int)
		candb = bracketings(dresult)
		prec = precision(goldb, candb)
		rec = recall(goldb, candb)
		f1 = f_measure(goldb, candb)
		dnoparse = True
	dresult = dresult
	dcandb = candb
	doptime = 0
	if d.dop:
		doptime = time.clock() - begin
		msg += "\n%.2fs cpu time elapsed\n" % (doptime)
	msg += "GOLD: %s" % tree.pprint(margin=1000)
	return (nsent, msg, (pcandb, presult, pnoparse, exactp),
			(scandb, sresult, snoparse, exacts),
			(dcandb, dresult, dnoparse, exactd))

class Params:
	def __init__(self, **kw): self.__dict__.update(kw)
internalparams = None
def initworker(params):
	""" Set global parameter object """
	global internalparams
	internalparams = params

def multidoparse(splitpcfg, srcg, dop, estimator, unfolded, bintype,
		sample, both, arity_marks, arity_marks_before_bin, m, pcfggrammar,
		srcggrammar, dopgrammar, secondarymodel, test, testmaxwords, testsents,
		prune, splitk, k, sldop_n=14, useestimates=False, outside=None,
		top='ROOT', tags=True, removeparentannotation=False,
		splitprune=False, mergesplitnodes=False, markorigin=False,
		resultdir="results", usecfgparse=False, backtransform=None,
		category=None, sentinit=0, doph=999):
	""" Like doparse but use multiprocessing. """
	import tempfile
	params = Params(
			splitpcfg=splitpcfg, srcg=srcg, dop=dop,
			pcfggrammar=pcfggrammar, srcggrammar=srcggrammar,
			dopgrammar=dopgrammar,
			estimator=estimator, unfolded=unfolded,
			bintype=bintype, sample=sample, both=both, arity_marks=arity_marks,
			arity_marks_before_bin=arity_marks_before_bin, m=m,
			secondarymodel=secondarymodel, test=test,
			testmaxwords=testmaxwords, testsents=testsents, prune=prune,
			splitk=splitk, k=k, sldop_n=sldop_n, useestimates=useestimates,
			outside=outside, top=top, tags=tags,
			removeparentannotation=removeparentannotation,
			splitprune=splitprune, mergesplitnodes=mergesplitnodes,
			markorigin=markorigin, resultdir=resultdir,
			usecfgparse=usecfgparse, backtransform=backtransform,
			category=category, sentinit=sentinit, doph=doph)
	pool = multiprocessing.Pool(processes=None, initializer=initworker,
			initargs=(params,))
	maxlen = min(testmaxwords, maxbitveclen)
	work = [a for a in zip(count(1), *test) if len(a[2]) <= maxlen][:testsents]
	print "going to parse %d sentences." % len(test)
	dowork = pool.imap_unordered(worker, work)

	presults = []; sresults = []; dresults = []
	gold = []; gsent = []
	pcandb = multiset(); scandb = multiset(); dcandb = multiset()
	goldbrackets = multiset()
	exactp = exactd = exacts = pnoparse = snoparse = dnoparse =  0
	# main parse loop over each sentence in test corpus
	for data in dowork:
		nsent, msg, (p,pr,pp,pe), (s,sr,sp,se), (d,dr,dp,de) = data
		_, tree, sent, block = work[nsent-1]
		logging.debug("%d. [len=%d] %s\n%s" % (nsent, len(sent),
					u" ".join(a[0] for a in sent),	# words only
					#u" ".join(a[0]+u"/"+a[1] for a in sent))) word/TAG
					msg))
		goldb = bracketings(tree)
		gold.append(block)
		gsent.append(sent)
		goldbrackets.update((nsent, a) for a in goldb.elements())
		if splitpcfg:
			pcandb.update((nsent, a) for a in p.elements())
			presults.append(pr)
			if pp: pnoparse += 1
			if pe: exactp += 1
		if srcg:
			scandb.update((nsent, a) for a in s.elements())
			sresults.append(sr)
			if sp: snoparse += 1
			if se: exacts += 1
		if dop:
			dcandb.update((nsent, a) for a in d.elements())
			dresults.append(dr)
			if dp: dnoparse += 1
			if de: exactd += 1
		msg = ''
		if splitpcfg:
			logging.debug("pcfg cov %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f%s" % (
								100 * (1 - pnoparse/float(len(presults))),
								100 * (exactp / float(nsent)),
								100 * precision(goldbrackets, pcandb),
								100 * recall(goldbrackets, pcandb),
								100 * f_measure(goldbrackets, pcandb),
								('' if srcg or dop else '\n')))
		if srcg:
			logging.debug("srcg cov %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f%s" % (
								100 * (1 - snoparse/float(len(sresults))),
								100 * (exacts / float(nsent)),
								100 * precision(goldbrackets, scandb),
								100 * recall(goldbrackets, scandb),
								100 * f_measure(goldbrackets, scandb),
								('' if dop else '\n')))
		if dop:
			logging.debug("dop  cov %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f"
					" (%+5.2f)\n" % (
								100 * (1 - dnoparse/float(len(dresults))),
								100 * (exactd / float(nsent)),
								100 * precision(goldbrackets, dcandb),
								100 * recall(goldbrackets, dcandb),
								100 * f_measure(goldbrackets, dcandb),
								100 * (f_measure(goldbrackets, dcandb) -
									max(f_measure(goldbrackets, pcandb) if splitpcfg else -1,
									f_measure(goldbrackets, scandb) if srcg else -1))))

	if splitpcfg:
		codecs.open("%s/%s.pcfg" % (resultdir, category or "results"),
			"w", encoding='utf-8').writelines(export(a, [w for w,_ in b],n + 1)
			for n,a,b in zip(count(sentinit), presults, gsent))
	if srcg:
		codecs.open("%s/%s.srcg" % (resultdir, category or "results"),
			"w", encoding='utf-8').writelines(export(a, [w for w,_ in b],n + 1)
			for n,a,b in zip(count(sentinit), sresults, gsent))
	if dop:
		codecs.open("%s/%s.dop" % (resultdir, category or "results"),
			"w", encoding='utf-8').writelines(export(a, [w for w,_ in b], n + 1)
			for n,a,b in zip(count(sentinit), dresults, gsent))
	codecs.open("%s/%s.gold" % (resultdir, category or "results"),
			"w", encoding='utf-8').write(''.join(
				"#BOS %d\n%s#EOS %d\n" % (n + 1, a, n + 1)
				for n, a in zip(count(sentinit), gold)))
	logging.info("wrote results to %s/%s.{gold,srcg,dop}" % (
		resultdir, category or "results"))

	return (splitpcfg, srcg, dop, nsent, testmaxwords, exactp, exacts, exactd,
		pnoparse, snoparse, dnoparse,goldbrackets, pcandb, scandb, dcandb,
		unfolded, arity_marks, bintype, estimator, sldop_n)

if __name__ == '__main__':
	import sys
	sys.stdout = codecs.getwriter('utf8')(sys.stdout)
	#cftiger()
	#parsetepacoc(); exit()
	if len(sys.argv) > 1:
		paramstr = open(sys.argv[1]).read()
		params = eval("dict(%s)" % paramstr)
		params['resultdir'] = sys.argv[1].rsplit(".", 1)[0]
		main(**params)
		# copy parameter file to result dir
		open("%s/params.prm" % params['resultdir'], "w").write(paramstr)
	else: main()
