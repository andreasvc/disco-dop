# -*- coding: UTF-8 -*-
import os, re, sys, time, gzip, string, codecs, logging, cPickle, tempfile
import multiprocessing
from collections import defaultdict, Counter as multiset
from itertools import islice, imap, count, izip_longest
from operator import itemgetter
from subprocess import Popen, PIPE
from math import exp
from nltk import Tree
from nltk.metrics import accuracy
import numpy as np
from treebank import NegraCorpusReader, fold, export
from fragments import getfragments
from grammar import induce_plcfrs, dop_lcfrs_rules, doubledop, grammarinfo, \
	rem_marks, read_bitpar_grammar, defaultparse, canonicalize, doubledop_new
from containers import Grammar, maxbitveclen
from treetransforms import binarize, unbinarize, optimalbinarize,\
	splitdiscnodes, mergediscnodes, addfanoutmarkers, slowfanout
from coarsetofine import prunechart
from parser import parse, cfgparse, pprint_chart
from disambiguation import marginalize, viterbiderivation, sldop, sldop_simple
from eval import bracketings, printbrackets, precision, recall, f_measure

def main(
	#parameters. parameters. PARAMETERS!!
	splitpcfg = True,
	plcfrs = True,
	dop = True,
	usecfgparse=False, #whether to use the dedicated CFG parser for splitpcfg
	usedoubledop = False,	# when False, use DOP reduction instead
	usetagger=None,	#default is to use tags from treebank.
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
	newdd=False, #use experimental, more efficient double dop algorithm
	iterate=False, #for double dop, whether to include fragments of fragments
	complement=False, #for double dop, whether to include fragments which form
			#the complement of the maximal recurring fragments extracted
	quiet=False, reallyquiet=False, #quiet=no per sentence results
	numproc=1,	#increase to use multiple CPUs. Set to None to use all CPUs.
	resultdir="output"
	):

	# Tiger treebank version 2 sample:
	# http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/annotation/sample2.export
	#corpus = NegraCorpusReader(".", "sample2.export", encoding="iso-8859-1"); testmaxwords = 99
	assert bintype in ("optimal", "optimalhead", "binarize", "nltk")
	assert estimator in ("dop1", "ewe", "shortest", "sl-dop", "sl-dop-simple")
	assert usetagger in (None, "treetagger", "stanford")
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
	logging.info("%d sentences in corpus %s/%s",
			len(corpus.parsed_sents()), corpusdir, corpusfile)
	if isinstance(trainsents, float):
		trainsents = int(trainsents * len(corpus.sents()))
	trees = corpus.parsed_sents()[:trainsents]
	sents = corpus.sents()[:trainsents]
	blocks = corpus.blocks()[:trainsents]
	logging.info("%d training sentences before length restriction", len(trees))
	trees, sents, blocks = zip(*[sent for sent in zip(trees, sents, blocks)
		if len(sent[1]) <= trainmaxwords])
	logging.info("%d training sentences after length restriction <= %d",
		len(trees), trainmaxwords)

	test = NegraCorpusReader(corpusdir, corpusfile, encoding=encoding,
			removepunct=removepunct, movepunct=movepunct)
	test = (test.parsed_sents()[trainsents+skip:trainsents+skip+testsents],
			test.tagged_sents()[trainsents+skip:trainsents+skip+testsents],
			test.blocks()[trainsents+skip:trainsents+skip+testsents])
	assert len(test[0]), "test corpus should be non-empty"

	if usetagger:
		if usetagger == 'treetagger':
			# these two tags are never given by tree-tagger,
			# so collect words whose tag needs to be overriden
			overridetags = ("PTKANT", "PIDAT")
		else: overridetags = ("PTKANT", )
		taglex = defaultdict(set)
		for sent in corpus.tagged_sents()[:trainsents]:
			for word, tag in sent: taglex[word].add(tag)
		overridetagdict = dict((tag,
			set(word for word, tags in taglex.iteritems() if tags == set([tag])))
			for tag in overridetags)
		tagmap = { "$(": "$[", "PAV": "PROAV" }
		dotagging(usetagger, test[1], overridetagdict, tagmap)

	f, n = treebankfanout(trees)
	logging.info("%d test sentences before length restriction", len(test[0]))
	test = zip(*((a, b, c) for a, b, c in zip(*test) if len(b) <= testmaxwords))
	logging.info("%d test sentences after length restriction <= %d",
		len(test[0]), testmaxwords)
	logging.info("treebank fan-out before binarization: %d #%d", f, n)
	logging.info("read training & test corpus")

	(pcfggrammar, plcfrsgrammar,
			dopgrammar, backtransform, secondarymodel) = getgrammars(
			trees, sents, splitpcfg, plcfrs, dop, estimator, bintype, h, v,
			factor, tailmarker, revmarkov, arity_marks, arity_marks_before_bin,
			trainmaxwords, trainsents, neverblockre, prune, splitprune,
			markorigin, resultdir, usedoubledop, newdd, iterate, complement,
			numproc)

	if getestimates == 'SX' and splitpcfg:
		from estimates import getpcfgestimates
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
		logging.info("computing PLCFRS estimates")
		begin = time.clock()
		outside = getestimates(plcfrsgrammar, testmaxwords,
				plcfrsgrammar.toid["ROOT"])
		logging.info("estimates done. cpu time elapsed: %gs"
					% (time.clock() - begin))
		np.savez("outside.npz", outside=outside)
		logging.info("saved estimates")
	if useestimates == 'SX' and splitpcfg:
		if not getestimates:
			assert not cfgparse, "estimates require agenda-based parser."
			outside = np.load("pcfgoutside.npz")['outside']
			logging.info("loaded PCFG estimates")
	elif useestimates == 'SXlrgaps' and plcfrs:
		if not getestimates:
			outside = np.load("outside.npz")['outside']
			logging.info("loaded PLCFRS estimates")
	else: outside = None

	begin = time.clock()
	results = doparse(splitpcfg, plcfrs, dop, estimator, unfolded, bintype,
			sample, both, arity_marks, arity_marks_before_bin, m, pcfggrammar,
			plcfrsgrammar, dopgrammar, secondarymodel, test, testmaxwords,
			testsents, prune, splitk, k, sldop_n, useestimates, outside,
			"ROOT", True, splitprune, markorigin, resultdir, usecfgparse,
			newdd, backtransform, numproc)
	if numproc == 1:
		logging.info("time elapsed during parsing: %gs", time.clock() - begin)
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
		markorigin=False, resultdir="results", usecfgparse=False, newdd=False,
		backtransform=None, numproc=None, category=None, sentinit=0,
		deletelabel=("ROOT", "$.", "$,", "$(", "$[")):
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
			usecfgparse=usecfgparse, newdd=newdd, backtransform=backtransform,
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
		logging.debug("%d/%d%s. [len=%d] %s\n%s", nsent, len(work),
					(' (%d)' % sentid) if numproc != 1 else '', len(sent),
					#u" ".join(a[0] for a in sent),	# words only
					u" ".join(a[0]+u"/"+a[1] for a in sent), # word/TAG
					msg)
		goldb = bracketings(tree)
		gold[sentid-1] = block
		gsent[sentid-1] = sent
		goldbrackets.update((sentid, (label, span)) for label, span
				in goldb.elements() if label not in deletelabel)
		if splitpcfg:
			pcandb.update((sentid, (label, span)) for label, span
					in p.candb.elements() if label not in deletelabel)
			presults[sentid-1] = p.result
			if p.noparse: pnoparse += 1
			if p.exact: exactp += 1
		if plcfrs:
			scandb.update((sentid, (label, span)) for label, span
					in s.candb.elements() if label not in deletelabel)
			sresults[sentid-1] = s.result
			if s.noparse: snoparse += 1
			if s.exact: exacts += 1
		if dop:
			dcandb.update((sentid, (label, span)) for label, span
					in d.candb.elements() if label not in deletelabel)
			dresults[sentid-1] = d.result
			if d.noparse: dnoparse += 1
			if d.exact: exactd += 1
		msg = ''
		if splitpcfg and pcandb:
			logging.debug("pcfg   cov %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f%s",
								100 * (1 - pnoparse/float(nsent)),
								100 * (exactp / float(nsent)),
								100 * precision(goldbrackets, pcandb),
								100 * recall(goldbrackets, pcandb),
								100 * f_measure(goldbrackets, pcandb),
								('' if plcfrs or dop else '\n'))
		if plcfrs and scandb:
			logging.debug("plcfrs cov %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f%s",
								100 * (1 - snoparse/float(nsent)),
								100 * (exacts / float(nsent)),
								100 * precision(goldbrackets, scandb),
								100 * recall(goldbrackets, scandb),
								100 * f_measure(goldbrackets, scandb),
								('' if dop else '\n'))
		if dop and dcandb:
			logging.debug("dop    cov %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f"
					" (%+5.2f)\n",
								100 * (1 - dnoparse/float(nsent)),
								100 * (exactd / float(nsent)),
								100 * precision(goldbrackets, dcandb),
								100 * recall(goldbrackets, dcandb),
								100 * f_measure(goldbrackets, dcandb),
								100 * (f_measure(goldbrackets, dcandb) -
									max(f_measure(goldbrackets, pcandb) if splitpcfg else -1,
									f_measure(goldbrackets, scandb) if plcfrs else -1)))
	if numproc != 1:
		pool.terminate()
		pool.join()
		del dowork, pool
	writeresults(splitpcfg, plcfrs, dop, gold, gsent, presults, sresults,
			dresults, resultdir, category, sentinit)

	return (splitpcfg, plcfrs, dop, len(work), testmaxwords, exactp, exacts,
			exactd, pnoparse, snoparse, dnoparse, goldbrackets, pcandb, scandb,
			dcandb, unfolded, arity_marks, bintype, estimator, sldop_n,
			bool(backtransform))

def worker(args):
	""" parse a sentence using pcfg, plcfrs, dop """
	nsent, tree, sent, _ = args
	d = internalparams
	goldb = bracketings(tree)
	pnoparse = snoparse = dnoparse = False
	exactp = exacts = exactd = False
	msg = ''
	if d.splitpcfg:
		msg += "PCFG:"
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
			msg += '\t' + msg1 + '\n'
	else: chart = {}; start = None
	if start:
		try:
			resultstr, prob = viterbiderivation(chart, start,
							d.pcfggrammar.tolabel)
		except KeyError: pprint_chart(chart, sent, d.pcfggrammar.tolabel)
		presult = Tree(resultstr)
		presult.un_chomsky_normal_form(childChar=":")
		mergediscnodes(presult)
		unbinarize(presult)
		rem_marks(presult)
		if d.unfolded: fold(presult)
		msg += "\tp=%.4e " % exp(-prob)
		pcandb = bracketings(presult)
		if goldb and pcandb:
			prec = precision(goldb, pcandb)
			rec = recall(goldb, pcandb)
			f1 = f_measure(goldb, pcandb)
		else: prec = rec = f1 = 0
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
		presult = defaultparse([(n, t) for n, (w, t) in enumerate(sent)])
		presult = Tree.parse("(%s %s)" % (d.top, presult), parse_leaf=int)
		pcandb = bracketings(presult)
		prec = precision(goldb, pcandb)
		rec = recall(goldb, pcandb)
		f1 = f_measure(goldb, pcandb)
		pnoparse = True
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
				(sum(len(a) for x in chart for a in x)
				if d.usecfgparse else len(chart)), items)
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
	elif d.plcfrs: chart = {}; start = None
	if d.plcfrs: msg += "PLCFRS: " + msg1
	if d.plcfrs and start:
		resultstr, prob = viterbiderivation(chart, start, d.plcfrsgrammar.tolabel)
		sresult = Tree(resultstr)
		unbinarize(sresult)
		rem_marks(sresult)
		if d.unfolded: fold(sresult)
		msg += "\n\tp=%.4e " % exp(-prob)
		scandb = bracketings(sresult)
		if goldb and scandb:
			prec = precision(goldb, scandb)
			rec = recall(goldb, scandb)
			f1 = f_measure(goldb, scandb)
		else: prec = rec = f1 = 0
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
		sresult = defaultparse([(n, t) for n, (w, t) in enumerate(sent)])
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
	else: chart = {}; start = None; msg1 = ""
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
				sample=d.sample, both=d.both, backtransform=d.backtransform,
				newdd=d.newdd)
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
		if goldb and dcandb:
			prec = precision(goldb, dcandb)
			rec = recall(goldb, dcandb)
			f1 = f_measure(goldb, dcandb)
		else: prec = rec = f1 = 0
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
		dresult = defaultparse([(n, t) for n,(w, t) in enumerate(sent)])
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

def writeresults(splitpcfg, plcfrs, dop, gold, gsent, presults, sresults,
		dresults, resultdir, category, sentinit):
	codecs.open("%s/%s.gold" % (resultdir, category or "results"),
			"w", encoding='utf-8').write(''.join(
				"#BOS %d\n%s#EOS %d\n" % (n + 1, a, n + 1)
				for n, a in zip(count(sentinit), gold)))
	if splitpcfg:
		codecs.open("%s/%s.pcfg" % (resultdir, category or "results"),
			"w", encoding='utf-8').writelines(export(a, [w for w, _ in b], n+1)
			for n, a, b in zip(count(sentinit), presults, gsent))
	if plcfrs:
		codecs.open("%s/%s.plcfrs" % (resultdir, category or "results"),
			"w", encoding='utf-8').writelines(export(a, [w for w, _ in b], n+1)
			for n, a, b in zip(count(sentinit), sresults, gsent))
	if dop:
		codecs.open("%s/%s.dop" % (resultdir, category or "results"),
			"w", encoding='utf-8').writelines(export(a, [w for w, _ in b], n+1)
			for n, a, b in zip(count(sentinit), dresults, gsent))
	logging.info("wrote results to %s/%s.{gold,plcfrs,dop}",
		resultdir, category or "results")

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
	grammar.testgrammar()
	dopgrammar.testgrammar()
	dop = True; plcfrs = True; unfolded = False; bintype = "binarize h=1 v=1"
	sample = False; both = False; arity_marks = True
	arity_marks_before_bin = False; estimator = 'sl-dop'; m = 10000
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
	doparse(False, plcfrs, dop, estimator, unfolded, bintype,
		sample, both, arity_marks, arity_marks_before_bin, m, None,
		grammar, dopgrammar, None, test, testmaxwords,
		testsents, prune, 0, 50, sldop_n=sldop_n, top=top, tags=tags)

def getgrammars(trees, sents, splitpcfg, plcfrs, dop, estimator, bintype, h, v,
		factor, tailmarker, revmarkov, arity_marks, arity_marks_before_bin,
		trainmaxwords, trainsents, neverblockre, prune, splitprune, markorigin,
		resultdir, usedoubledop, newdd, iterate, complement, numproc):
	# binarization
	begin = time.clock()
	if arity_marks_before_bin: trees = map(addfanoutmarkers, trees)
	if bintype == "nltk":
		bintype += " %s h=%d v=%d" % (factor, h, v)
		for a in trees: a.chomsky_normal_form(factor="right", vertMarkov=v-1, horzMarkov=h)
	elif bintype == "binarize":
		bintype += " %s h=%d v=%d %s" % (factor, h, v, "tailmarker" if tailmarker else '')
		for a in trees:
			binarize(a, factor=factor, vertMarkov=v-1, horzMarkov=h,
					tailMarker=tailmarker, leftMostUnary=True,
					rightMostUnary=True, reverse=revmarkov)
					#fixme: leftMostUnary=False, rightMostUnary=False, 
	elif bintype == "optimal":
		trees = [Tree.convert(optimalbinarize(tree))
						for n, tree in enumerate(trees)]
	elif bintype == "optimalhead":
		trees = [Tree.convert(
					optimalbinarize(tree, headdriven=True, h=h, v=v))
						for n, tree in enumerate(trees)]
	logging.info("binarized %s cpu time elapsed: %gs",
						bintype, time.clock() - begin)
	logging.info("binarized treebank fan-out: %d #%d", *treebankfanout(trees))
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
		logging.info("induced CFG based on %d sentences", len(splittrees))
		#pcfggrammar = dop_lcgrs_rules(splittrees, sents)
		#logging.info("induced DOP CFG based on %d sentences", len(trees))
		logging.info(grammarinfo(pcfggrammar))
		pcfggrammar = Grammar(pcfggrammar)
		pcfggrammar.testgrammar()
		pcfggrammar.write_bitpar_grammar(
			open(resultdir + "/pcfg.rules", "w"),
			codecs.open(resultdir + "/pcfg.lex", "w", "utf-8"))

	if arity_marks: trees = map(addfanoutmarkers, trees)
	if plcfrs:
		plcfrsgrammar = induce_plcfrs(trees, sents)
		logging.info("induced PLCFRS based on %d sentences", len(trees))
		logging.info(grammarinfo(plcfrsgrammar, dump="%s/pcdist.txt" % resultdir))
		plcfrsgrammar = Grammar(plcfrsgrammar)
		plcfrsgrammar.testgrammar()
		plcfrsgrammar.write_lcfrs_grammar(
			open(resultdir + "/plcfrs.rules", "w"),
			codecs.open(resultdir + "/plcfrs.lex", "w", "utf-8"))
		if splitpcfg and prune:
			plcfrsgrammar.getmapping(re.compile(r"_[0-9]+$"), #None
				re.compile(neverblockre) if neverblockre else None,
				pcfggrammar, splitprune, markorigin)
		logging.info("wrote grammar to %s/plcfrs.{rules,lex}", resultdir)
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
			# find recurring fragments in treebank,
			# as well as depth-1 'cover' fragments
			fragments = getfragments(trees, sents, numproc,
					iterate=iterate, complement=complement)
			if newdd:
				dopgrammar, backtransform = doubledop_new(fragments)
			else:
				dopgrammar, backtransform = doubledop(fragments)
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
		dopgrammar.testgrammar()
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
		logging.info("wrote grammar to %s/dop.{rules,lex%s}.gz",
			resultdir, ".backtransform" if usedoubledop else '')
	return pcfggrammar, plcfrsgrammar, dopgrammar, backtransform, secondarymodel

def readtepacoc():
	tepacocids = set()
	tepacocsents = defaultdict(list)
	cat = "undefined"
	tepacoc = codecs.open("../tepacoc.txt", encoding="utf8")
	for line in tepacoc.read().splitlines():
		fields = line.split("\t") # = [id, '', sent]
		if line.strip() and len(fields) == 3:
			if fields[0].strip():
				# subtract one because our ids are zero-based, tepacoc 1-based
				sentid = int(fields[0]) - 1
				tepacocids.add(sentid)
				tepacocsents[cat].append((sentid, fields[2].split()))
			else: # new category
				cat = fields[2]
				if cat.startswith("CUC"): cat = "CUC"
		elif fields[0] == "TuBa": break
	return tepacocids, tepacocsents

def parsetepacoc(splitpcfg=True, plcfrs=True, dop=True, estimator='dop1',
		unfolded=False, bintype="binarize", h=1, v=2, factor="right",
		tailmarker='', revmarkov=False, arity_marks=True,
		arity_marks_before_bin=False, sample=False, both=False, m=10000,
		trainmaxwords=999, testmaxwords=999, testsents=2000, splitk=10000,
		k=5000, prune=True, sldop_n=7, neverblockre=None, splitprune=True,
		markorigin=True, iterate=False, complement=False, usecfgparse=True,
		usedoubledop=True, newdd=False, usetagger='stanford',
		resultdir="tepacoc-sfdv2", numproc=8):
	trainsents = 25005
	os.mkdir(resultdir)
	# Log everything, and send it to stderr, in a format with just the message.
	format = '%(message)s'
	logging.basicConfig(level=logging.DEBUG, format=format)
	# log up to INFO to a results log file
	file = logging.FileHandler(filename='%s/output.log' % resultdir)
	file.setLevel(logging.INFO)
	file.setFormatter(logging.Formatter(format))
	logging.getLogger('').addHandler(file)
	tepacocids, tepacocsents = readtepacoc()
	try:
		(corpus_sents, corpus_taggedsents,
				corpus_trees, corpus_blocks) = cPickle.load(
					gzip.open("tiger.pickle.gz", "rb"))
	except IOError: # file not found
		corpus = NegraCorpusReader("../tiger/corpus", "tiger_release_aug07.export",
				headorder=(bintype in ("binarize", "nltk")), headfinal=True,
				headreverse=False, unfold=unfolded, movepunct=True,
				removepunct=False, encoding='iso-8859-1')
		corpus_sents = list(corpus.sents())
		corpus_taggedsents = list(corpus.tagged_sents())
		corpus_trees = list(corpus.parsed_sents())
		corpus_blocks = list(corpus.blocks())
		cPickle.dump((corpus_sents, corpus_taggedsents, corpus_trees,
			corpus_blocks), gzip.open("tiger.pickle.gz", "wb"), protocol=-1)

	# test set
	testset = {}
	allsents = []
	for cat, catsents in tepacocsents.iteritems():
		test = trees, sents, blocks = [], [], []
		for n, sent in catsents:
			if sent != corpus_sents[n]:
				logging.error("mismatch. sent %d:\n%r\n%r\n"
					"not in corpus %r\nnot in tepacoc %r",
					n + 1, sent, corpus_sents[n],
					[a for a, b in izip_longest(sent, corpus_sents[n]) if a and a != b],
					[b for a, b in izip_longest(sent, corpus_sents[n]) if b and a != b])
			elif len(corpus_sents[n]) <= testmaxwords:
				sents.append(corpus_taggedsents[n])
				trees.append(corpus_trees[n])
				blocks.append(corpus_blocks[n])
		allsents.extend(sents)
		logging.info("category: %s, %d of %d sentences",
				cat, len(test[0]), len(catsents))
		testset[cat] = test
	testset['baseline'] = zip(*[sent for n, sent in
				enumerate(zip(corpus_trees, corpus_taggedsents, corpus_blocks))
				if len(sent[1]) <= trainmaxwords
				and n not in tepacocids][trainsents:trainsents+2000])
	allsents.extend(testset['baseline'][1])

	if usetagger:
		overridetags = ("PTKANT", "VAIMP")
		taglex = defaultdict(set)
		for sent in corpus_taggedsents[:trainsents]:
			for word, tag in sent: taglex[word].add(tag)
		overridetagdict = dict((tag,
			set(word for word, tags in taglex.iteritems() if tags == set([tag])))
			for tag in overridetags)
		tagmap = { "$(": "$[", "PAV": "PROAV", "PIDAT": "PIAT" }
		# the sentences in the list allsents are modified in-place so that
		# the relevant copy in testset[cat][1] is updated as well.
		dotagging(usetagger, allsents, overridetagdict, tagmap)

	# training set
	trees, sents, blocks = zip(*[sent for n, sent in
				enumerate(zip(corpus_trees, corpus_sents,
							corpus_blocks)) if len(sent[1]) <= trainmaxwords
							and n not in tepacocids][:trainsents])
	(pcfggrammar, plcfrsgrammar,
			dopgrammar, backtransform, secondarymodel) = getgrammars(
			trees, sents, splitpcfg, plcfrs, dop, estimator, bintype, h, v,
			factor, tailmarker, revmarkov, arity_marks, arity_marks_before_bin,
			trainmaxwords, trainsents, neverblockre, prune, splitprune,
			markorigin, resultdir, usedoubledop, newdd, iterate, complement,
			numproc)

	del corpus_sents, corpus_taggedsents, corpus_trees, corpus_blocks
	results = {}
	cnt = 0
	for cat, test in sorted(testset.items()):
		if cat == 'baseline': continue
		logging.info("category: %s", cat)
		begin = time.clock()
		results[cat] = doparse(splitpcfg, plcfrs, dop, estimator, unfolded, bintype,
				sample, both, arity_marks, arity_marks_before_bin, m, pcfggrammar,
				plcfrsgrammar, dopgrammar, secondarymodel, test, testmaxwords,
				testsents, prune, splitk, k, sldop_n, False, None,
				"ROOT", True, splitprune, markorigin, resultdir, usecfgparse,
				newdd, backtransform, numproc, category=cat, sentinit=cnt)
		cnt += len(test[0])
		logging.info("time elapsed during parsing: %g", time.clock() - begin)
	goldbrackets = multiset(); pcandb = multiset(); scandb = multiset(); dcandb = multiset()
	exactp = exacts = exactd = pnoparse = snoparse = dnoparse = 0
	glines = []; plines = []; slines = []; dlines = []
	for cat, res in results.iteritems():
		logging.info("category: %s", cat)
		for ext, b in zip("gold pcgf plcfrs dop".split(),
				(glines, plines, slines, dlines)):
			if os.path.exists("%s/%s.%s" % (resultdir, cat, ext)):
				b.append(codecs.open("%s/%s.%s" % (resultdir, cat, ext),
					encoding='utf-8').read())
		exactp += res[5]
		exacts += res[6]
		exactd += res[7]
		snoparse += res[8]
		snoparse += res[9]
		dnoparse += res[10]
		goldbrackets |= res[11]
		pcandb |= res[12]
		scandb |= res[13]
		dcandb |= res[14]
		doeval(*res)
	logging.info("TOTAL")
	# write TOTAL results file with all tepacoc sentences (not the baseline)
	for ext, b in zip("gold pcgf plcfrs dop".split(),
			(glines, plines, slines, dlines)):
		if b:
			codecs.open("%s/TOTAL.%s" % (resultdir, ext), "w",
					encoding='utf-8').writelines(b)
	doeval(splitpcfg, plcfrs, dop, cnt, testmaxwords, exactp, exacts, exactd,
		pnoparse, snoparse, dnoparse, goldbrackets, pcandb, scandb, dcandb,
		False, arity_marks, bintype, estimator, sldop_n, usedoubledop)
	# do baseline separately because it shouldn't count towards the total score
	cat = 'baseline'
	logging.info("category: %s", cat)
	doeval(*doparse(splitpcfg, plcfrs, dop, estimator, unfolded, bintype,
				sample, both, arity_marks, arity_marks_before_bin, m, pcfggrammar,
				plcfrsgrammar, dopgrammar, secondarymodel, testset[cat], testmaxwords,
				testsents, prune, splitk, k, sldop_n, False, None,
				"ROOT", True, splitprune, markorigin, resultdir, usecfgparse,
				newdd, backtransform, numproc, category=cat, sentinit=cnt))

def cycledetection(trees, sents):
	seen = set()
	v = set(); e = {}; weights = {}
	for tree, sent in zip(trees, sents):
		rules = [(a, b) for a, b in induce_plcfrs([tree], [sent]) if a not in seen]
		seen.update(map(lambda (a, b): a, rules))
		for (rule, _), w in rules:
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
			logging.debug("cycle (cost %5.2f): %s",
				sum(weights[c, d] for c, d in zip(b, b[1:])), " => ".join(b))

def dotagging(usetagger, sents, overridetag, tagmap):
	""" Use an external tool to tag a list of tagged sentences, overwriting the
	original tags in-place. """
	logging.info("Start tagging.")
	goldtags = [t for sent in sents for _, t in sent]
	if usetagger == "treetagger": # Tree-tagger
		# ftp://ftp.ims.uni-stuttgart.de/pub/corpora/tree-tagger-linux-3.2.tar.gz
		# ftp://ftp.ims.uni-stuttgart.de/pub/corpora/german-par-linux-3.2-utf8.bin.gz
		infile, inname = tempfile.mkstemp(text=True)
		with os.fdopen(infile, 'w') as infile:
			for tagsent in sents:
				sent = map(itemgetter(0), tagsent)
				infile.write("\n".join(wordmangle(w, n, sent)
					for n, w in enumerate(sent)) + "\n<S>\n")
		#tagger = Popen(executable="tree-tagger/cmd/tree-tagger-german",
		#		args=["tree-tagger/cmd/tree-tagger-german"],
		#		stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=False)
		tagger = Popen("tree-tagger/bin/tree-tagger -token -sgml"
				" tree-tagger/lib/german-par-linux-3.2-utf8.bin"
				" %s | tree-tagger/cmd/filter-german-tags" % inname,
				stdout=PIPE, shell=True)
		tagout = tagger.stdout.read().decode('utf-8').split("<S>")[:-1]
		os.unlink(inname)
		taggedsents = [[tagmangle(a, None, overridetag, tagmap)
					for a in tags.splitlines() if a.strip()]
					for tags in tagout]
	elif usetagger == "stanford": # Stanford Tagger
		# http://nlp.stanford.edu/software/stanford-postagger-full-2012-07-09.tgz
		infile, inname = tempfile.mkstemp(text=True)
		with os.fdopen(infile, 'w') as infile:
			for tagsent in sents:
				sent = map(itemgetter(0), tagsent)
				infile.write(" ".join(wordmangle(w, n, sent)
					for n, w in enumerate(sent)) + "\n")
		tagger = Popen(args=(
				"/usr/bin/java -mx2G -classpath stanford-postagger.jar "
				"edu.stanford.nlp.tagger.maxent.MaxentTagger "
				"-model models/german-hgc.tagger -tokenize false "
				"-encoding utf-8 -textFile %s" % inname).split(),
				cwd="../src/stanford-postagger-full-2012-07-09",
				shell=False, stdout=PIPE)
		tagout = tagger.stdout.read().decode('utf-8').splitlines()
		os.unlink(inname)
		taggedsents = [[tagmangle(a, "_", overridetag, tagmap)
			for a in tags.split()] for tags in tagout]
	assert len(taggedsents) == len(sents), (
			"mismatch in number of sentences after tagging.")
	for n, tags in enumerate(taggedsents):
		assert len(sents[n]) == len(tags), (
				"mismatch in number of tokens after tagging.\n"
				"before: %r\nafter: %r" % (sents[n], tags))
		sents[n][:] = tags
	newtags = [t for sent in taggedsents for _, t in sent]
	logging.info("Tag accuracy: %5.2f\ngold - cand: %r\ncand - gold %r",
		(100 * accuracy(goldtags, newtags)),
		set(goldtags) - set(newtags), set(newtags) - set(goldtags))

sentend = "(\"'!?..." # ";/-"
def wordmangle(w, n, sent):
	#if n > 0 and w[0] in string.uppercase and not sent[n-1] in sentend:
	#	return ("%s\tNE\tNN\tFM" % w).encode('utf-8')
	return w.encode('utf-8')

def tagmangle(a, splitchar, overridetag, tagmap):
	word, tag = a.rsplit(splitchar, 1)
	for newtag in overridetag:
		if word in overridetag[newtag]: tag = newtag
	return word, tagmap.get(tag, tag)

def treebankfanout(trees):
	return max((slowfanout(a), n) for n, tree in enumerate(trees)
		for a in tree.subtrees(lambda x: len(x) > 1))

def test():
	from doctest import testmod, NORMALIZE_WHITESPACE, ELLIPSIS
	import bit, demos, kbest, parser, grammar, treebank, estimates, _fragments
	import agenda, coarsetofine, treetransforms, disambiguation
	modules = (bit, demos, kbest, parser, grammar, treebank, estimates,
			_fragments, agenda, coarsetofine, treetransforms, disambiguation)
	results = {}
	for mod in modules:
		print 'running doctests of', mod.__file__
		results[mod] = fail, _ = testmod(mod, verbose=False,
			optionflags=NORMALIZE_WHITESPACE | ELLIPSIS)
		assert fail == 0, mod.__file__
	for mod in modules:
		if hasattr(mod, 'test'): mod.test()
		else: mod.main()
	print "no doctests:"
	for mod, (fail, attempted) in results.iteritems():
		if not attempted: print mod.__file__,
	print
	for mod, (fail, attempted) in sorted(results.iteritems(), key=itemgetter(1)):
		if attempted: print '%s: %d doctests succeeded!' % (mod.__file__, attempted)

usage = """Usage: %s [--test|parameter file]
--test	run tests on all modules
If a parameter file is given, an experiment is run. See the file sample.prm
for an example parameter file. Note that to repeat an experiment, the directory
with the previous results must be moved somewhere else manually, to avoid
accidentally overwriting results. """ % sys.argv[0]

if __name__ == '__main__':
	sys.stdout = codecs.getwriter('utf8')(sys.stdout)
	if len(sys.argv) == 1: print usage
	elif '--test' in sys.argv: test()
	elif '--tepacoc' in sys.argv: parsetepacoc()
	elif '--cftiger' in sys.argv: cftiger()
	else:
		paramstr = open(sys.argv[1]).read()
		params = eval("dict(%s)" % paramstr)
		params['resultdir'] = sys.argv[1].rsplit(".", 1)[0]
		main(**params)
		# copy parameter file to result dir
		open("%s/params.prm" % params['resultdir'], "w").write(paramstr)
