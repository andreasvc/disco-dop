# -*- coding: UTF-8 -*-
import os, re, time, gzip, logging, codecs, cPickle
import multiprocessing
from collections import defaultdict, Counter as multiset
from itertools import islice, imap, count
from operator import itemgetter
from math import exp
from nltk import Tree
from treebank import NegraCorpusReader, fold, export
from grammar import induce_plcfrs, dop_lcfrs_rules, doubledop, grammarinfo, \
	rem_marks, read_bitpar_grammar, defaultparse, canonicalize
from containers import Grammar, maxbitveclen
from treetransforms import binarize, unbinarize, optimalbinarize,\
	splitdiscnodes, mergediscnodes, addfanoutmarkers
from coarsetofine import prunechart, kbest_items
from parser import parse, cfgparse, pprint_chart
from disambiguation import marginalize, viterbiderivation, sldop, sldop_simple
from eval import bracketings, printbrackets, precision, recall, f_measure

def main(
	#parameters. parameters. PARAMETERS!!
	splitpcfg = True,
	plcfrs = True,
	dop = True,
	usedoubledop = False,	# when False, use DOP reduction instead
	corpusdir=".",
	corpusfile="sample2.export",
	encoding="iso-8859-1",
	movepunct=False,
	removepunct=False,
	unfolded = False,
	testmaxwords = 40,  # max number of words for sentences in test corpus
	trainmaxwords = 40, # max number of words for sentences in train corpus
	#trainsents = 0.9, testsents = 9999,	# percent of sentences to parse
	#trainsents = 18602, testsents = 1000, # number of sentences to parse
	#trainsents = 7200, testsents = 100, # number of sentences to parse
	trainsents = 2, testsents = 1, # number of sentences to parse
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
	k = 50,		#number of coarse plcfrs derivations to prune with; k=0 => filter only
	prune=True,	#whether to use plcfrs chart to prune parsing of dop
	getestimates=False, #compute & store estimates
	useestimates=False,  #load & use estimates
	markorigin=True, #when splitting nodes, mark origin: VP_2 => {VP*1, VP*2}
	splitprune=False, #VP_2[101] is treated as { VP*[100], VP*[001] } during parsing
	neverblockre=None, #do not prune nodes with label that match regex
	quiet=False, reallyquiet=False, #quiet=no per sentence results
	numproc=1,	#increase to use multiple CPUs. Set to None to use all CPUs.
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
	assert estimator in ("dop1", "ewe", "shortest", "sl-dop", "sl-dop-simple")
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
	trees = corpus.parsed_sents()[:trainsents]
	sents = corpus.sents()[:trainsents]
	blocks = corpus.blocks()[:trainsents]
	logging.info("%d training sentences before length restriction" % len(trees))
	trees, sents, blocks = zip(*[sent for sent in zip(trees, sents, blocks) if len(sent[1]) <= trainmaxwords])
	logging.info("%d training sentences after length restriction <= %d" % (len(trees), trainmaxwords))

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
	if arity_marks_before_bin: trees = map(addfanoutmarkers, trees)
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
	for a in trees: canonicalize(a)

	#cycledetection()
	pcfggrammar = []; plcfrsgrammar = []; dopgrammar = []; secondarymodel = []
	backtransform = None
	if splitpcfg:
		splittrees = [splitdiscnodes(a.copy(True), markorigin) for a in trees]
		logging.info("splitted discontinuous nodes")
		# second layer of binarization:
		for a in splittrees:
			a.chomsky_normal_form(childChar=":")
		pcfggrammar = induce_plcfrs(splittrees, sents)
		logging.info("induced CFG based on %d sentences" % len(splittrees))
		#pcfggrammar = dop_lcgrs_rules(splittrees, sents)
		#logging.info("induced DOP CFG based on %d sentences" % len(trees))
		logging.info(grammarinfo(pcfggrammar))
		pcfggrammar = Grammar(pcfggrammar)
		if usecfgparse: pcfggrammar.getunaryclosure()
		pcfggrammar.testgrammar(logging)
		pcfggrammar.write_bitpar_grammar(
			open(resultdir + "/pcfg.rules", "w"),
			codecs.open(resultdir + "/pcfg.lex", "w", "utf-8"))

	if arity_marks: trees = map(addfanoutmarkers, trees)
	if plcfrs:
		plcfrsgrammar = induce_plcfrs(trees, sents)
		logging.info("induced PLCFRS based on %d sentences" % len(trees))
		logging.info(grammarinfo(plcfrsgrammar, dump="%s/pcdist.txt" % resultdir))
		plcfrsgrammar = Grammar(plcfrsgrammar)
		plcfrsgrammar.testgrammar(logging)
		plcfrsgrammar.write_lcfrs_grammar(
			open(resultdir + "/plcfrs.rules", "w"),
			codecs.open(resultdir + "/plcfrs.lex", "w", "utf-8"))
		if splitpcfg and prune:
			plcfrsgrammar.getmapping(re.compile(r"_[0-9]+$"), #None
				re.compile(neverblockre) if neverblockre else None,
				pcfggrammar, splitprune, markorigin)
		logging.info("wrote grammar to %s/plcfrs.{rules,lex}" % resultdir)
	if dop:
		if estimator == "shortest":
			# the secondary model is used to resolve ties for the shortest derivation
			dopgrammar, secondarymodel = dop_lcfrs_rules(trees, sents,
				normalize=False, shortestderiv=True)
		elif "sl-dop" in estimator:
			dopgrammar = dop_lcfrs_rules(trees, sents, normalize=True,
							shortestderiv=False)
			dopshortest, _ = dop_lcfrs_rules(trees, sents,
							normalize=False, shortestderiv=True)
			secondarymodel = Grammar(dopshortest)
		elif usedoubledop:
			assert estimator not in ("ewe", "sl-dop", "sl-dop-simple",
					"shortest"), "Not implemented."
			dopgrammar, backtransform = doubledop(trees, sents, numproc)
		else:
			dopgrammar = dop_lcfrs_rules(trees, sents,
				normalize=(estimator in ("ewe", "sl-dop", "sl-dop-simple")),
				shortestderiv=False)
		nodes = sum(len(list(a.subtrees())) for a in trees)
		msg = grammarinfo(dopgrammar)
		dopgrammar = Grammar(dopgrammar)
		logging.info("DOP model based on %d sentences, %d nodes, %d nonterminals"
					% (len(trees), nodes, len(dopgrammar.toid)))
		logging.info(msg)
		dopgrammar.testgrammar(logging)
		dopgrammar.write_lcfrs_grammar(
			gzip.open(resultdir + "/dop.rules.gz", "w"),
			codecs.getwriter('utf-8')(gzip.open(resultdir + "/dop.lex.gz", "w")))
		if usedoubledop:
			gzip.open(resultdir + "/dop.backtransform.gz", "w").writelines(
				"%s\t%s\n" % a for a in backtransform.iteritems())
			if prune:
				dopgrammar.getmapping(re.compile("@.+$") if plcfrs else
					re.compile("(?:_[0-9]+)?(?:@.+)?$"),
					re.compile(r'^#[0-9]+|.+}<'), # + neverblockre?
					plcfrsgrammar if plcfrs else pcfggrammar,
					splitprune and not plcfrs, markorigin)
		elif prune:
			dopgrammar.getmapping(re.compile("@[-0-9]+$") if plcfrs else
					re.compile("(?:_[0-9]+)?(?:@[-0-9]+)?$"),
					re.compile(neverblockre) if neverblockre else None,
					plcfrsgrammar if plcfrs else pcfggrammar,
					splitprune and not plcfrs, markorigin)
		logging.info("wrote grammar to %s/dop.{rules,lex%s}.gz" % (
			resultdir, ".backtransform" if usedoubledop else ''))

	if getestimates == 'SX' and splitpcfg:
		from estimates import getpcfgestimates
		import numpy as np
		logging.info("computing PCFG estimates")
		begin = time.clock()
		outside = getpcfgestimates(pcfggrammar, testmaxwords,
				pcfggrammar.toid["ROOT"])
		logging.info("estimates done. cpu time elapsed: %gs"
					% (time.clock() - begin))
		np.savez("pcfgoutside.npz", outside=outside)
		logging.info("saved PCFG estimates")
	elif getestimates == 'SXlrgaps' and plcfrs:
		from estimates import getestimates
		import numpy as np
		logging.info("computing PLCFRS estimates")
		begin = time.clock()
		outside = getestimates(plcfrsgrammar, testmaxwords,
				plcfrsgrammar.toid["ROOT"])
		logging.info("estimates done. cpu time elapsed: %gs"
					% (time.clock() - begin))
		np.savez("outside.npz", outside=outside)
		#cPickle.dump(outside, open("outside.pickle", "wb"))
		logging.info("saved estimates")
	if useestimates == 'SX' and splitpcfg:
		if not getestimates:
			import numpy as np
			assert not cfgparse, "estimates require agenda-based parser."
			outside = np.load("pcfgoutside.npz")['outside']
			logging.info("loaded PCFG estimates")
	elif useestimates == 'SXlrgaps' and plcfrs:
		if not getestimates:
			import numpy as np
			#outside = cPickle.load(open("outside.pickle", "rb"))
			outside = np.load("outside.npz")['outside']
			logging.info("loaded PLCFRS estimates")
	else: outside = None

	#for a,b in extractfragments(trees).items():
	#	print a,b
	#exit()
	begin = time.clock()
	results = doparse(splitpcfg, plcfrs, dop, estimator, unfolded, bintype,
			sample, both, arity_marks, arity_marks_before_bin, m, pcfggrammar,
			plcfrsgrammar, dopgrammar, secondarymodel, test, testmaxwords,
			testsents, prune, splitk, k, sldop_n, useestimates, outside,
			"ROOT", True, splitprune, markorigin, resultdir, usecfgparse,
			backtransform, numproc)
	if numproc == 1:
		logging.info("time elapsed during parsing: %gs" % (time.clock() - begin))
	doeval(*results)

class Params:
	def __init__(self, **kw): self.__dict__.update(kw)
internalparams = None
def initworker(params):
	""" Set global parameter object """
	global internalparams
	internalparams = params


class Result:
	def __init__(self, candb, result, noparse, exact):
		self.candb = candb; self.result = result
		self.noparse = noparse; self.exact = exact

def doparse(splitpcfg, plcfrs, dop, estimator, unfolded, bintype,
		sample, both, arity_marks, arity_marks_before_bin, m, pcfggrammar,
		plcfrsgrammar, dopgrammar, secondarymodel, test, testmaxwords,
		testsents, prune, splitk, k, sldop_n=14, useestimates=False,
		outside=None, top='ROOT', tags=True, splitprune=False,
		markorigin=False, resultdir="results", usecfgparse=False,
		backtransform=None, numproc=None, category=None, sentinit=0):
	# FIXME: use multiprocessing namespace:
	#mgr = multiprocessing.Manager()
	#namespace = mgr.Namespace()
	params = Params(
			splitpcfg=splitpcfg, plcfrs=plcfrs, dop=dop,
			pcfggrammar=pcfggrammar, plcfrsgrammar=plcfrsgrammar,
			dopgrammar=dopgrammar, estimator=estimator, unfolded=unfolded,
			bintype=bintype, sample=sample, both=both, arity_marks=arity_marks,
			arity_marks_before_bin=arity_marks_before_bin, m=m,
			secondarymodel=secondarymodel, test=test,
			testmaxwords=testmaxwords, testsents=testsents, prune=prune,
			splitk=splitk, k=k, sldop_n=sldop_n, useestimates=useestimates,
			outside=outside, top=top, tags=tags, splitprune=splitprune,
			markorigin=markorigin, resultdir=resultdir,
			usecfgparse=usecfgparse, backtransform=backtransform,
			category=category, sentinit=sentinit)
	pcandb = multiset(); scandb = multiset(); dcandb = multiset()
	goldbrackets = multiset()
	exactp = exactd = exacts = pnoparse = snoparse = dnoparse =  0

	maxlen = min(testmaxwords, maxbitveclen)
	work = [a for a in zip(count(1), *test) if len(a[2]) <= maxlen][:testsents]
	gold = [None] * len(work)
	gsent = [None] * len(work)
	presults = [None] * len(work)
	sresults = [None] * len(work)
	dresults = [None] * len(work)
	if numproc == 1:
		initworker(params)
		dowork = imap(worker, work)
	else:
		pool = multiprocessing.Pool(processes=numproc, initializer=initworker,
				initargs=(params,))
		dowork = pool.imap_unordered(worker, work)
	print "going to parse %d sentences." % len(work)
	# main parse loop over each sentence in test corpus
	for nsent, data in enumerate(dowork, 1):
		sentid, msg, p, s, d = data
		thesentid, tree, sent, block = work[sentid-1]
		assert thesentid == sentid
		logging.debug("%d/%d%s. [len=%d] %s\n%s" % (nsent, len(work),
					(' (%d)' % sentid) if numproc != 1 else '', len(sent),
					u" ".join(a[0] for a in sent),	# words only
					#u" ".join(a[0]+u"/"+a[1] for a in sent))) word/TAG
					msg))
		goldb = bracketings(tree)
		gold[sentid-1] = block
		gsent[sentid-1] = sent
		goldbrackets.update((sentid, a) for a in goldb.elements())
		if splitpcfg:
			pcandb.update((sentid, a) for a in p.candb.elements())
			presults[sentid-1] = p.result
			if p.noparse: pnoparse += 1
			if p.exact: exactp += 1
		if plcfrs:
			scandb.update((sentid, a) for a in s.candb.elements())
			sresults[sentid-1] = s.result
			if s.noparse: snoparse += 1
			if s.exact: exacts += 1
		if dop:
			dcandb.update((sentid, a) for a in d.candb.elements())
			dresults[sentid-1] = d.result
			if d.noparse: dnoparse += 1
			if d.exact: exactd += 1
		msg = ''
		if splitpcfg:
			logging.debug("pcfg   cov %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f%s" % (
								100 * (1 - pnoparse/float(len(presults))),
								100 * (exactp / float(nsent)),
								100 * precision(goldbrackets, pcandb),
								100 * recall(goldbrackets, pcandb),
								100 * f_measure(goldbrackets, pcandb),
								('' if plcfrs or dop else '\n')))
		if plcfrs:
			logging.debug("plcfrs cov %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f%s" % (
								100 * (1 - snoparse/float(len(sresults))),
								100 * (exacts / float(nsent)),
								100 * precision(goldbrackets, scandb),
								100 * recall(goldbrackets, scandb),
								100 * f_measure(goldbrackets, scandb),
								('' if dop else '\n')))
		if dop:
			logging.debug("dop    cov %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f"
					" (%+5.2f)\n" % (
								100 * (1 - dnoparse/float(len(dresults))),
								100 * (exactd / float(nsent)),
								100 * precision(goldbrackets, dcandb),
								100 * recall(goldbrackets, dcandb),
								100 * f_measure(goldbrackets, dcandb),
								100 * (f_measure(goldbrackets, dcandb) -
									max(f_measure(goldbrackets, pcandb) if splitpcfg else -1,
									f_measure(goldbrackets, scandb) if plcfrs else -1))))
	if numproc != 1:
		pool.terminate()
		pool.join()
		del dowork, pool

	codecs.open("%s/%s.gold" % (resultdir, category or "results"),
			"w", encoding='utf-8').write(''.join(
				"#BOS %d\n%s#EOS %d\n" % (n + 1, a, n + 1)
				for n, a in zip(count(sentinit), gold)))
	if splitpcfg:
		codecs.open("%s/%s.pcfg" % (resultdir, category or "results"),
			"w", encoding='utf-8').writelines(export(a, [w for w, _ in b], n + 1)
			for n,a,b in zip(count(sentinit), presults, gsent))
	if plcfrs:
		codecs.open("%s/%s.plcfrs" % (resultdir, category or "results"),
			"w", encoding='utf-8').writelines(export(a, [w for w, _ in b], n + 1)
			for n,a,b in zip(count(sentinit), sresults, gsent))
	if dop:
		codecs.open("%s/%s.dop" % (resultdir, category or "results"),
			"w", encoding='utf-8').writelines(export(a, [w for w, _ in b], n + 1)
			for n,a,b in zip(count(sentinit), dresults, gsent))
	logging.info("wrote results to %s/%s.{gold,plcfrs,dop}" % (
		resultdir, category or "results"))

	return (splitpcfg, plcfrs, dop, len(work), testmaxwords, exactp, exacts,
			exactd, pnoparse, snoparse, dnoparse,goldbrackets, pcandb, scandb,
			dcandb, unfolded, arity_marks, bintype, estimator, sldop_n,
			bool(backtransform))

def worker(args):
	""" parse a sentence using pcfg, plcfrs, dop """
	nsent, tree, sent, block = args
	d = internalparams
	goldb = bracketings(tree)
	pnoparse = snoparse = dnoparse = False
	exactp = exacts = exactd = False
	msg = ''
	if d.splitpcfg:
		msg += "PCFG:\t"
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
					exhaustive=d.prune and (d.plcfrs or d.dop),
					estimates=('SX', d.outside)
						if d.useestimates=='SX' else None)
			msg += msg1 + '\n'
	else: chart = {}; start = False
	if start:
		resultstr, prob = viterbiderivation(chart, start,
							d.pcfggrammar.tolabel)
		presult = Tree(resultstr)
		presult.un_chomsky_normal_form(childChar=":")
		mergediscnodes(presult)
		unbinarize(presult)
		rem_marks(presult)
		if d.unfolded: fold(presult)
		msg += "\tp=%.4e " % exp(-prob)
		pcandb = bracketings(presult)
		prec = precision(goldb, pcandb)
		rec = recall(goldb, pcandb)
		f1 = f_measure(goldb, pcandb)
		if presult == tree or f1 == 1.0:
			assert presult != tree or f1 == 1.0
			msg += "exact match"
			exactp = True
		else:
			msg += "LP %5.2f LR %5.2f LF %5.2f\n" % (
							100 * prec, 100 * rec, 100 * f1)
			if pcandb - goldb:
				msg += "\tcand-gold=%s " % printbrackets(pcandb - goldb)
			if goldb - pcandb:
				msg += "gold-cand=%s" % printbrackets(goldb - pcandb)
			if (pcandb - goldb) or (goldb - pcandb): msg += '\n'
			msg += "\t%s" % presult.pprint(margin=1000)
	else:
		if d.splitpcfg: msg += "\tno parse"
		presult = defaultparse([(n,t) for n,(w, t) in enumerate(sent)])
		presult = Tree.parse("(%s %s)" % (d.top, presult), parse_leaf=int)
		pcandb = bracketings(presult)
		prec = precision(goldb, pcandb)
		rec = recall(goldb, pcandb)
		f1 = f_measure(goldb, pcandb)
		pnoparse = True
		#pprint_chart(chart,
		#		[w.encode('unicode-escape') for w, _ in sent],
		#		d.pcfggrammar.tolabel)
	if d.splitpcfg:
		pcfgtime = time.clock() - begin
		msg += "\n\t%.2fs cpu time elapsed\n" % (pcfgtime)
	msg1 = ''
	if d.plcfrs and (not d.splitpcfg or start):
		begin = time.clock()
		whitelist = []
		if d.splitpcfg and start:
			whitelist, items = prunechart(chart, start, d.pcfggrammar,
					d.plcfrsgrammar, d.splitk, d.splitprune, d.markorigin)
			msg += "\tcoarse items before pruning: %d; after: %d\n" % (
				len(chart), items)
		chart, start, msg1 = parse(
					[w for w, t in sent], d.plcfrsgrammar,
					tags=[t for w, t in sent] if d.tags else [],
					start=d.plcfrsgrammar.toid[d.top],
					whitelist=whitelist if d.splitpcfg else None,
					splitprune=d.splitprune and d.splitpcfg,
					markorigin=d.markorigin,
					exhaustive=d.dop and d.prune,
					estimates=('SXlrgaps', d.outside)
						if d.useestimates=='SXlrgaps' else None)
	elif d.plcfrs: chart = {}; start = False
	if d.plcfrs: msg += "PLCFRS: " + msg1
	if d.plcfrs and start:
		resultstr, prob = viterbiderivation(chart, start, d.plcfrsgrammar.tolabel)
		sresult = Tree(resultstr)
		unbinarize(sresult)
		rem_marks(sresult)
		if d.unfolded: fold(sresult)
		msg += "\n\tp=%.4e " % exp(-prob)
		scandb = bracketings(sresult)
		prec = precision(goldb, scandb)
		rec = recall(goldb, scandb)
		f1 = f_measure(goldb, scandb)
		if sresult == tree or f1 == 1.0:
			assert sresult != tree or f1 == 1.0
			msg += "exact match"
			exacts = True
		else:
			msg += "LP %5.2f LR %5.2f LF %5.2f\n" % (
							100 * prec, 100 * rec, 100 * f1)
			if scandb - goldb:
				msg += "\tcand-gold=%s " % printbrackets(scandb - goldb)
			if goldb - scandb:
				msg += "gold-cand=%s" % printbrackets(goldb - scandb)
			if (scandb - goldb) or (goldb - scandb): msg += '\n'
			msg += "\t%s" % sresult.pprint(margin=1000)
	else:
		if d.plcfrs: msg += " no parse"
		sresult = defaultparse([(n,t) for n,(w, t) in enumerate(sent)])
		sresult = Tree.parse("(%s %s)" % (d.top, sresult), parse_leaf=int)
		scandb = bracketings(sresult)
		prec = precision(goldb, scandb)
		rec = recall(goldb, scandb)
		f1 = f_measure(goldb, scandb)
		snoparse = True
	if d.plcfrs:
		msg += "\n\t%.2fs cpu time elapsed\n" % (time.clock() - begin)
	if d.dop and (not (d.splitpcfg or d.plcfrs) or start):
		begin = time.clock()
		whitelist = []
		if (d.splitpcfg or d.plcfrs) and d.prune and start:
			whitelist, items = prunechart(chart, start,
					d.plcfrsgrammar if d.plcfrs else d.pcfggrammar,
					d.dopgrammar, d.k, d.splitprune and not d.plcfrs,
					d.markorigin)
			msg += "\tcoarse items before pruning: %d; after: %d\n" % (
				len(chart), items)
		chart, start, msg1 = parse(
						[w for w, _ in sent], d.dopgrammar,
						tags=[t for _, t in sent] if d.tags else None,
						start=d.dopgrammar.toid[d.top],
						exhaustive=True,
						estimates=None,
						whitelist=whitelist,
						splitprune=d.splitprune and not d.plcfrs,
						markorigin=d.markorigin)
		if d.plcfrs and not start:
			pprint_chart(chart,
					[w.encode('unicode-escape') for w, _ in sent],
					d.dopgrammar.tolabel)
			raise ValueError("expected successful parse")
	else: chart = {}; start = False; msg1 = ""
	if d.dop: msg += "DOP:\t%s" % msg1
	if d.dop and start:
		begindisamb = time.clock()
		if d.estimator == "shortest":
			mpp, msg1 = marginalize(chart, start, d.dopgrammar.tolabel, n=d.m,
					sample=d.sample, both=d.both, shortest=True,
					secondarymodel=d.secondarymodel)
		elif d.estimator == "sl-dop":
			mpp, msg1 = sldop(chart, start, sent, d.tags, d.dopgrammar,
						d.secondarymodel, d.m, d.sldop_n, d.sample, d.both)
			mpp = dict(mpp)
		elif d.estimator == "sl-dop-simple":
			# old method, estimate shortest derivation directly from number
			# of addressed nodes
			mpp, msg1 = sldop_simple(chart, start, d.dopgrammar, d.m, d.sldop_n)
			mpp = dict(mpp)
		elif d.backtransform is not None:
			mpp, msg1 = marginalize(chart, start, d.dopgrammar.tolabel, n=d.m,
				sample=d.sample, both=d.both, backtransform=d.backtransform)
		else: #dop1, ewe
			mpp, msg1 = marginalize(chart, start, d.dopgrammar.tolabel,
								n=d.m, sample=d.sample, both=d.both)
		msg += "\n\tdisambiguation: %s, %gs\n" % (msg1, time.clock() - begindisamb)
		dresult, prob = max(mpp.iteritems(), key=itemgetter(1))
		dresult = Tree(dresult)
		if isinstance(prob, tuple):
			msg += " subtrees = %d, p=%.4e " % (abs(prob[0]), prob[1])
		else: msg += "        p=%.4e " % prob
		unbinarize(dresult)
		rem_marks(dresult)
		if d.unfolded: fold(dresult)
		dcandb = bracketings(dresult)
		try: prec = precision(goldb, dcandb)
		except ZeroDivisionError:
			prec = 0.0
			logging.warning("empty candidate brackets?\n%s" % dresult.pprint())
		rec = recall(goldb, dcandb)
		f1 = f_measure(goldb, dcandb)
		if dresult == tree or f1 == 1.0:
			msg += "exact match"
			exactd = True
		else:
			msg += "LP %5.2f LR %5.2f LF %5.2f\n" % (
							100 * prec, 100 * rec, 100 * f1)
			if dcandb - goldb:
				msg += "\tcand-gold=%s " % printbrackets(dcandb - goldb)
			if goldb - dcandb:
				msg += "gold-cand=%s " % printbrackets(goldb - dcandb)
			if (dcandb - goldb) or (goldb - dcandb): msg += '\n'
			msg += "        %s" % dresult.pprint(margin=1000)
	else:
		if d.dop: msg += " no parse"
		dresult = defaultparse([(n,t) for n,(w, t) in enumerate(sent)])
		dresult = Tree.parse("(%s %s)" % (d.top, dresult), parse_leaf=int)
		dcandb = bracketings(dresult)
		prec = precision(goldb, dcandb)
		rec = recall(goldb, dcandb)
		f1 = f_measure(goldb, dcandb)
		dnoparse = True
	if d.dop:
		doptime = time.clock() - begin
		msg += "\n\t%.2fs cpu time elapsed\n" % (doptime)
	else: doptime = 0
	msg += "GOLD:   %s" % tree.pprint(margin=1000)
	return (nsent, msg,
			Result(pcandb, presult, pnoparse, exactp),
			Result(scandb, sresult, snoparse, exacts),
			Result(dcandb, dresult, dnoparse, exactd))

def doeval(splitpcfg, plcfrs, dop, nsent, testmaxwords, exactp, exacts, exactd,
		pnoparse, snoparse, dnoparse, goldbrackets, pcandb, scandb, dcandb,
		unfolded, arity_marks, bintype, estimator, sldop_n, usedoubledop):
	print "testmaxwords", testmaxwords, "unfolded", unfolded,
	print "arity marks", arity_marks, "binarized", bintype,
	print "estimator", estimator,
	if dop:
		if 'sl-dop' in estimator: print sldop_n
		if usedoubledop: print "doubledop"
	if splitpcfg and nsent:
		logging.info("pcfg lp %5.2f lr %5.2f lf %5.2f\n"
			"coverage %d / %d = %5.2f %%  exact match %d / %d = %5.2f %%\n" % (
				100 * precision(goldbrackets, pcandb),
				100 * recall(goldbrackets, pcandb),
				100 * f_measure(goldbrackets, pcandb),
				nsent - pnoparse, nsent, 100.0 * (nsent - pnoparse) / nsent,
				exactp, nsent, 100.0 * exactp / nsent))
	if plcfrs and nsent:
		logging.info("plcfrs lp %5.2f lr %5.2f lf %5.2f\n"
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
	grammar.testgrammar(logging)
	dopgrammar.testgrammar(logging)
	dop = True; plcfrs = True; unfolded = False; bintype = "binarize h=1 v=1"
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
	doparse(plcfrs, dop, estimator, unfolded, bintype, sample, both, arity_marks,
			arity_marks_before_bin, m, grammar, dopgrammar, test, testmaxwords,
			testsents, prune, sldop_n, top, tags)

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

def parsetepacoc(dop=True, plcfrs=True, estimator='ewe', unfolded=False,
	bintype="binarize", h=1, v=1, factor="right", arity_marks=True,
	arity_marks_before_bin=False, sample=False, both=False, m=10000,
	trainmaxwords=999, testmaxwords=40, testsents=999, k=1000, prune=True,
	sldop_n=7, splitprune=False, neverblockre=None, markorigin=True,
	resultdir="tepacoc-40k1000"):

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
	if splitprune:
		coarsetrees = [splitdiscnodes(a.copy(True), markorigin) for a in trees]
		for a in coarsetrees: a.chomsky_normal_form(childChar=":")
		print "splitted discontinuous nodes"
	plcfrsgrammar = induce_plcfrs(list(coarsetrees), sents)
	print "induced", "pcfg" if splitprune else "plcfrs",
	print "of", len(sents), "sentences"
	logging.info(grammarinfo(plcfrsgrammar))
	plcfrsgrammar = Grammar(plcfrsgrammar)
	plcfrsgrammar.testgrammar(logging)

	dopgrammar = dop_lcfrs_rules(list(trees), list(sents),
				normalize=(estimator in ("ewe", "sl-dop", "sl-dop-simple")),
				shortestderiv=False)
	print "induced dop reduction of", len(sents), "sentences"
	logging.info(grammarinfo(dopgrammar))
	dopgrammar = Grammar(dopgrammar)
	dopgrammar.testgrammar(logging)
	dopgrammar.getmapping(re.compile("@[-0-9]+$"),
			re.compile(neverblockre) if neverblockre else None,
			plcfrsgrammar if plcfrs else pcfggrammar,
			not plcfrs and splitprune, markorigin)
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
		results[cat] = doparse(plcfrs, dop, estimator, unfolded, bintype, sample,
					both, arity_marks, arity_marks_before_bin, m, plcfrsgrammar,
					dopgrammar, secondarymodel, test, testmaxwords, testsents, prune,
					50, k, sldop_n, False, None, "ROOT", True, splitprune,
					markorigin, resultdir=resultdir, category=cat, sentinit=cnt)
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

def cycledetection(trees, sents):
	seen = set()
	v = set(); e = {}; weights = {}
	for n, (tree, sent) in enumerate(zip(trees, sents)):
		rules = [(a,b) for a,b in induce_plcfrs([tree], [sent]) if a not in seen]
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

def test():
	from doctest import testmod, NORMALIZE_WHITESPACE, ELLIPSIS
	import bit, demos, kbest, parser, grammar, treebank, estimates, _fragments
	import agenda, coarsetofine, treetransforms, disambiguation
	modules = (bit, demos, kbest, parser, grammar, treebank, estimates,
			_fragments, agenda, coarsetofine, treetransforms, disambiguation)
	results = {}
	for mod in modules:
		results[mod] = fail, _ = testmod(mod, verbose=False,
				optionflags=NORMALIZE_WHITESPACE | ELLIPSIS)
		assert fail == 0, mod.__file__
	for mod in modules:
		mod.test() if hasattr(mod, 'test') else mod.main()
	print "no doctests:"
	for mod, (fail, attempted) in results.iteritems():
		if not attempted: print mod.__file__,
	print
	for mod, (fail, attempted) in sorted(results.iteritems(), key=itemgetter(1)):
		if attempted: print '%s: %d doctests succeeded!' % (mod.__file__, attempted)
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
