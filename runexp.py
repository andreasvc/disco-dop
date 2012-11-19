# -*- coding: UTF-8 -*-
import os, re, sys, time, gzip, codecs, logging, cPickle, tempfile
import multiprocessing
from collections import defaultdict, Counter as multiset
from itertools import imap, count, izip_longest
from operator import itemgetter
from subprocess import Popen, PIPE
from math import exp
from nltk import Tree
from nltk.metrics import accuracy
import numpy as np
from treebank import NegraCorpusReader, DiscBracketCorpusReader, \
		BracketCorpusReader, fold, export
from fragments import getfragments
from grammar import induce_plcfrs, dop_lcfrs_rules, doubledop, grammarinfo, \
	rem_marks, defaultparse, canonicalize, doubledop_new
from containers import Grammar, maxbitveclen
from treetransforms import binarize, unbinarize, optimalbinarize,\
	splitdiscnodes, mergediscnodes, addfanoutmarkers, slowfanout
from coarsetofine import prunechart
from parser import parse, cfgparse, pprint_chart
from disambiguation import marginalize, viterbiderivation, sldop, sldop_simple
from eval import bracketings, precision, recall, f_measure, printbrackets

class DictObj(object):
	""" A class to wrap a dictionary. """
	def __init__(self, **kw): self.__dict__.update(kw)
	def __repr__(self):
		return "%s(%s)" % (self.__class__.__name__,
			",\n".join("%s=%r" % a for a in self.__dict__.items()))


internalparams = None
def initworker(params):
	""" Set global parameter object """
	global internalparams
	internalparams = params

def main(
		stages=(), # see variable 'defaults' below
		corpusfmt="export", # choices: export, discbracket, bracket
		corpusdir=".",
		corpusfile="sample2.export",
		encoding="iso-8859-1",
		movepunct=False,
		removepunct=False,
		unfolded = False,
		testmaxwords = 40,
		trainmaxwords = 40,
		trainsents = 2,
		testsents = 1, # number of sentences to parse
		skip=0,	# dev set
		#skip=1000, #skip dev set to get test set
		usetagger=None,	#default is to use gold tags from treebank.
		bintype = "binarize", # choices: binarize, nltk, optimal, optimalhead
		factor = "right",
		revmarkov = True,
		v = 1,
		h = 2,
		arity_marks = True,
		arity_marks_before_bin = False,
		tailmarker = "",
		quiet=False, reallyquiet=False, #quiet=no per sentence results
		numproc=1,	#increase to use multiple CPUs. Set to None to use all CPUs.
		resultdir='results'):

	assert bintype in ("optimal", "optimalhead", "binarize", "nltk")
	assert usetagger in (None, "treetagger", "stanford")

	defaults = dict(
			name='stage1', # identifier, used for filenames
			mode='plcfrs', # use the agenda-based PLCFRS parser
			prune=False,	#whether to use previous chart to prune parsing of this stage
			split=False, #split disc. nodes VP_2[101] as { VP*[100], VP*[001] }
			splitprune=False, #VP_2[101] is treated as { VP*[100], VP*[001] } for pruning
			markorigin=False, #when splitting nodes, mark origin: VP_2 => {VP*1, VP*2}
			k = 50, #number of coarse pcfg derivations to prune with; k=0 => filter only
			neverblockre=None, #do not prune nodes with label that match regex
			getestimates=None, #compute & store estimates
			useestimates=None,  #load & use estimates
			dop=False, # enable DOP mode (DOP reduction / double DOP)
			usedoubledop = False,	# when False, use DOP reduction instead
			newdd=False, #use experimental, more efficient double dop algorithm
			iterate=False, #for double dop, whether to include fragments of fragments
			complement=False, #for double dop, whether to include fragments which form
					#the complement of the maximal recurring fragments extracted
			sample=False, both=False,
			m = 10000,		#number of derivations to sample/enumerate
			estimator = "ewe", # choices: dop1, ewe, shortest, sl-dop[-simple]
			sldop_n=7)
	for n, a in enumerate(stages):
		tmp = defaults.copy()
		tmp.update(a)
		assert tmp['estimator'] in (
				"dop1", "ewe", "shortest", "sl-dop", "sl-dop-simple")
		stages[n] = DictObj(**tmp)

	os.mkdir(resultdir)
	# Log everything, and send it to stderr, in a format with just the message.
	formatstr = '%(message)s'
	if reallyquiet: logging.basicConfig(level=logging.WARNING, format=formatstr)
	elif quiet: logging.basicConfig(level=logging.INFO, format=formatstr)
	else: logging.basicConfig(level=logging.DEBUG, format=formatstr)

	# log up to INFO to a results log file
	fileobj = logging.FileHandler(filename='%s/output.log' % resultdir)
	fileobj.setLevel(logging.INFO)
	fileobj.setFormatter(logging.Formatter(formatstr))
	logging.getLogger('').addHandler(fileobj)

	if corpusfmt == 'export': CorpusReader = NegraCorpusReader
	elif corpusfmt == 'bracket': CorpusReader = BracketCorpusReader
	elif corpusfmt == 'discbracket': CorpusReader = DiscBracketCorpusReader
	corpus = CorpusReader(corpusdir, corpusfile, encoding=encoding,
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

	test = CorpusReader(corpusdir, corpusfile, encoding=encoding,
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

	logging.info("%d test sentences before length restriction", len(test[0]))
	test = zip(*((a, b, c) for a, b, c in zip(*test) if len(b) <= testmaxwords))
	logging.info("%d test sentences after length restriction <= %d",
		len(test[0]), testmaxwords)
	logging.info("read training & test corpus")

	getgrammars(trees, sents, stages, bintype, h, v, factor, tailmarker,
			revmarkov, arity_marks, arity_marks_before_bin, testmaxwords,
			resultdir, numproc)

	begin = time.clock()
	results = doparse(stages, unfolded, bintype, arity_marks,
			arity_marks_before_bin, test, testmaxwords, testsents,
			trees[0].node, True, resultdir, numproc)
	if numproc == 1:
		logging.info("time elapsed during parsing: %gs", time.clock() - begin)
	print "testmaxwords", testmaxwords, "unfolded", unfolded,
	print "arity marks", arity_marks, "binarized", bintype,
	if stages[-1].dop:
		print "estimator", stages[-1].estimator,
		if 'sl-dop' in stages[-1].estimator: print stages[-1].sldop_n
		if stages[-1].usedoubledop: print "doubledop"
	doeval(*results)

def getgrammars(trees, sents, stages, bintype, h, v, factor, tailmarker,
		revmarkov, arity_marks, arity_marks_before_bin, testmaxwords, resultdir,
		numproc):
	f, n = treebankfanout(trees)
	logging.info("treebank fan-out before binarization: %d #%d", f, n)
	# binarization
	begin = time.clock()
	if arity_marks_before_bin: trees = map(addfanoutmarkers, trees)
	if bintype == "nltk":
		bintype += " %s h=%d v=%d" % (factor, h, v)
		for a in trees:
			a.chomsky_normal_form(factor="right", vertMarkov=v-1, horzMarkov=h)
	elif bintype == "binarize":
		bintype += " %s h=%d v=%d %s" % (factor, h, v,
			"tailmarker" if tailmarker else '')
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
	if arity_marks: trees = map(addfanoutmarkers, trees)
	logging.info("binarized %s cpu time elapsed: %gs",
						bintype, time.clock() - begin)
	logging.info("binarized treebank fan-out: %d #%d", *treebankfanout(trees))
	for a in trees: canonicalize(a)

	#cycledetection()
	if any(stage.split for stage in stages):
		splittrees = [splitdiscnodes(a.copy(True), stages[0].markorigin)
				for a in trees]
		logging.info("splitted discontinuous nodes")
		for a in splittrees: a.chomsky_normal_form(childChar=":")
	for n, stage in enumerate(stages):
		assert stage.mode in ("plcfrs", "pcfg")
		if stage.prune: assert stage.mode == "plcfrs", (
			"pruning requires agenda-based parser")
		if stage.split: traintrees = splittrees
		else: traintrees = trees
		if stage.dop:
			stages[n].backtransform = None
			if stage.estimator == "shortest":
				# the secondary model is used to resolve ties
				# for the shortest derivation
				grammar, secondarymodel = dop_lcfrs_rules(traintrees, sents,
					normalize=False, shortestderiv=True)
				stages[n].secondarymodel = secondarymodel
			elif "sl-dop" in stage.estimator:
				grammar = dop_lcfrs_rules(traintrees, sents, normalize=True,
								shortestderiv=False)
				dopshortest, _ = dop_lcfrs_rules(traintrees, sents,
								normalize=False, shortestderiv=True)
				secondarymodel = Grammar(dopshortest)
				stages[n].secondarymodel = secondarymodel
			elif stage.usedoubledop:
				assert stage.estimator not in ("ewe", "sl-dop", "sl-dop-simple",
						"shortest"), "Not implemented."
				# find recurring fragments in treebank,
				# as well as depth-1 'cover' fragments
				fragments = getfragments(traintrees, sents, numproc,
						iterate=stage.iterate, complement=stage.complement)
				if stage.newdd: grammar, backtransform = doubledop_new(fragments)
				else: grammar, backtransform = doubledop(fragments)
				stages[n].backtransform = backtransform
			else:
				grammar = dop_lcfrs_rules(traintrees, sents,
					normalize=(stage.estimator
						in ("ewe", "sl-dop", "sl-dop-simple")),
					shortestderiv=False)
			nodes = sum(len(list(a.subtrees())) for a in traintrees)
			msg = grammarinfo(grammar)
			grammar = Grammar(grammar)
			logging.info("DOP model based on %d sentences, %d nodes, "
				"%d nonterminals",  len(traintrees), nodes, len(grammar.toid))
			logging.info(msg)
			grammar.testgrammar()
			if stage.usedoubledop:
				gzip.open(resultdir + "/dop.backtransform.gz", "w").writelines(
					"%s\t%s\n" % a for a in backtransform.iteritems())
				if stage.prune:
					grammar.getmapping(re.compile("(?:_[0-9]+)?(?:@.+)?$")
						if stages[n-1].split else re.compile("@.+$"),
						re.compile(r'^#[0-9]+|.+}<'), # + neverblockre?
						stages[n-1].grammar,
						stage.splitprune and stages[n-1].split,
						stages[n-1].markorigin)
			elif stage.prune:
				grammar.getmapping(re.compile("(?:_[0-9]+)?(?:@[-0-9]+)?$")
					if stages[n-1].split else re.compile("@[-0-9]+$"),
					re.compile(stage.neverblockre)
					if stage.neverblockre else None,
					stages[n-1].grammar,
					stage.splitprune and stages[n-1].split,
					stages[n-1].markorigin)
		else: # not stage.dop
			grammar = induce_plcfrs(traintrees, sents)
			logging.info("induced %s based on %d sentences",
				("PCFG" if f == 1 or stage.split else "PLCFRS"), len(traintrees))
			if stage.split or os.path.exists("%s/pcdist.txt" % resultdir):
				logging.info(grammarinfo(grammar))
			else:
				logging.info(grammarinfo(grammar,
						dump="%s/pcdist.txt" % resultdir))
			grammar = Grammar(grammar)
			grammar.testgrammar()
			if stage.prune:
				assert n > 0, (
					"need previous stage to prune, but this stage is first.")
				grammar.getmapping(None, #re.compile(r"_[0-9]+$"),
					re.compile(stage.neverblockre)
					if stage.neverblockre else None,
					stages[n-1].grammar,
					stage.splitprune and stages[n-1].split,
					stages[n-1].markorigin)

		stages[n].grammar = grammar
		rules = gzip.open("%s/%s.rules.gz" % (resultdir, stages[n].name), "w")
		lexicon = codecs.getwriter('utf-8')(gzip.open("%s/%s.lex.gz" % (
			resultdir, stages[n].name), "w"))
		if f == 1 or stage.split: grammar.write_bitpar_grammar(rules, lexicon)
		else: grammar.write_lcfrs_grammar(rules, lexicon)
		logging.info("wrote grammar to %s/%s.{rules,lex%s}.gz",
			resultdir, stage.name, ",backtransform" if stage.usedoubledop else '')

		outside = None
		if stage.getestimates == 'SX':
			assert f == 1 or stage.split, "SX estimate requires PCFG."
			from estimates import getpcfgestimates
			logging.info("computing PCFG estimates")
			begin = time.clock()
			outside = getpcfgestimates(grammar, testmaxwords,
					grammar.toid[trees[0].node])
			logging.info("estimates done. cpu time elapsed: %gs",
					time.clock() - begin)
			np.savez("pcfgoutside.npz", outside=outside)
			logging.info("saved PCFG estimates")
		elif stage.useestimates == 'SX':
			assert f == 1 or stage.split, "SX estimate requires PCFG."
			assert stage.mode != 'pcfg', (
				"estimates require agenda-based parser.")
			outside = np.load("pcfgoutside.npz")['outside']
			logging.info("loaded PCFG estimates")
		if stage.getestimates == 'SXlrgaps':
			from estimates import getestimates
			logging.info("computing PLCFRS estimates")
			begin = time.clock()
			outside = getestimates(grammar, testmaxwords,
					grammar.toid[trees[0].node])
			logging.info("estimates done. cpu time elapsed: %gs",
						time.clock() - begin)
			np.savez("outside.npz", outside=outside)
			logging.info("saved estimates")
		elif stage.useestimates == 'SXlrgaps':
			outside = np.load("outside.npz")['outside']
			logging.info("loaded PLCFRS estimates")
		stages[n].outside = outside

#def doparse(**params):
#	params = DictObj(**params)
def doparse(stages, unfolded, bintype, arity_marks, arity_marks_before_bin,
		test, testmaxwords, testsents, top, tags=True,
		resultdir="results", numproc=None, category=None, sentinit=0,
		deletelabel=("ROOT", "VROOT", "TOP", "$.", "$,", "$(", "$[")):
	params = DictObj(stages=stages, unfolded=unfolded, bintype=bintype,
			arity_marks=arity_marks,
			arity_marks_before_bin=arity_marks_before_bin, test=test,
			testmaxwords=testmaxwords, testsents=testsents, top=top, tags=tags,
			resultdir=resultdir, category=category, sentinit=sentinit)
	goldbrackets = multiset()
	maxlen = min(testmaxwords, maxbitveclen)
	work = [a for a in zip(count(1), *test) if len(a[2]) <= maxlen][:testsents]
	gold = [None] * len(work)
	gsent = [None] * len(work)
	results = [DictObj(name=stage.name) for stage in stages]
	for result in results:
		result.elapsedtime = [None] * len(work)
		result.parsetrees = [None] * len(work)
		result.brackets = multiset()
		result.exact = result.noparse = 0
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
		sentid, msg, sentresults = data
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
		for n, r in enumerate(sentresults):
			results[n].brackets.update((sentid, (label, span)) for label, span
					in r.candb.elements() if label not in deletelabel)
			results[n].parsetrees[sentid-1] = r.parsetree
			results[n].elapsedtime[sentid-1] = r.elapsedtime
			if r.noparse: results[n].noparse += 1
			if r.exact: results[n].exact += 1
			logging.debug(
				"%s cov %5.2f ex %5.2f lp %5.2f lr %5.2f lf %5.2f%s",
					r.name.ljust(7),
					100 * (1 - results[n].noparse/float(nsent)),
					100 * (results[n].exact / float(nsent)),
					100 * precision(goldbrackets, results[n].brackets),
					100 * recall(goldbrackets, results[n].brackets),
					100 * f_measure(goldbrackets, results[n].brackets),
					('' if n + 1 < len(sentresults) else '\n'))
	if numproc != 1:
		pool.terminate()
		pool.join()
		del dowork, pool

	writeresults(results, gold, gsent, resultdir, category, sentinit)
	return results, len(work), goldbrackets, gold, gsent

def worker(args):
	""" parse a sentence using specified stages (pcfg, plcfrs, dop, ...) """
	nsent, tree, sent, _ = args
	d = internalparams
	goldb = bracketings(tree)
	results = []
	msg = ''
	chart = {}; start = None
	for n, stage in enumerate(d.stages):
		begin = time.clock()
		exact = noparse = False
		msg += "%s:\t" % stage.name.upper()
		if n == 0 or start:
			if n != 0 and stage.prune:
				whitelist, items = prunechart(chart, start,
						d.stages[n-1].grammar, stage.grammar, stage.k,
						stage.splitprune, d.stages[n-1].markorigin)
				msg += "coarse items before pruning: %d; after: %d\n\t" % (
					(sum(len(a) for x in chart for a in x if a)
					if d.stages[n-1].mode == 'pcfg' else len(chart)), items)
			else: whitelist = None
			if stage.mode == 'pcfg':
				chart, start = cfgparse([w for w, _ in sent],
						stage.grammar,
						tags=[t for _, t in sent] if d.tags else None,
						start=stage.grammar.toid[d.top],
						) #chart=whitelist if stage.prune else None)
			else:
				chart, start, msg1 = parse([w for w, _ in sent],
						stage.grammar,
						tags=[t for _, t in sent] if d.tags else None,
						start=stage.grammar.toid[d.top],
						exhaustive=stage.dop or (
							n+1 != len(d.stages) and d.stages[n+1].prune),
						whitelist=whitelist,
						splitprune=stage.splitprune and d.stages[n-1].split,
						markorigin=d.stages[n-1].markorigin,
						estimates=(stage.useestimates, stage.outside)
							if stage.useestimates in ('SX', 'SXlrgaps') else None)
				msg += "%s\n\t" % msg1
			if (n != 0 and not start and not results[-1].noparse
					and stage.split == d.stages[n-1].split):
				#pprint_chart(chart,
				#		[w.encode('unicode-escape') for w, _ in sent],
				#		stage.grammar.tolabel)
				logging.error("expected successful parse. "
						"sent %d, %s." % (nsent, stage.name))
				raise ValueError
		# store & report result
		if start:
			if stage.dop:
				begindisamb = time.clock()
				if stage.estimator == "shortest":
					mpp, msg1 = marginalize(chart, start, stage.grammar.tolabel,
							n=stage.m, sample=stage.sample, both=stage.both,
							shortest=True, secondarymodel=stage.secondarymodel)
				elif stage.estimator == "sl-dop":
					mpp, msg1 = sldop(chart, start, sent, d.tags, stage.grammar,
							stage.secondarymodel, stage.m, stage.sldop_n,
							stage.sample, stage.both)
					mpp = dict(mpp)
				elif stage.estimator == "sl-dop-simple":
					# old method, estimate shortest derivation directly
					# from number of addressed nodes
					mpp, msg1 = sldop_simple(chart, start, stage.grammar,
							stage.m, stage.sldop_n)
					mpp = dict(mpp)
				elif stage.backtransform is not None:
					mpp, msg1 = marginalize(chart, start, stage.grammar.tolabel,
							n=stage.m, sample=stage.sample, both=stage.both,
							backtransform=stage.backtransform, newdd=stage.newdd)
				else: #dop1, ewe
					mpp, msg1 = marginalize(chart, start, stage.grammar.tolabel,
							n=stage.m, sample=stage.sample, both=stage.both)
				resultstr, prob = max(mpp.iteritems(), key=itemgetter(1))
				msg += "disambiguation: %s, %gs\n\t" % (
						msg1, time.clock() - begindisamb)
				if isinstance(prob, tuple):
					msg += "subtrees = %d, p=%.4e " % (abs(prob[0]), prob[1])
				else: msg += "p=%.4e " % prob
			elif not stage.dop:
				resultstr, prob = viterbiderivation(chart, start,
						stage.grammar.tolabel)
				msg += "p=%.4e " % exp(-prob)
			parsetree = Tree(resultstr)
			if stage.split:
				parsetree.un_chomsky_normal_form(childChar=":")
				mergediscnodes(parsetree)
			unbinarize(parsetree)
			rem_marks(parsetree)
			if d.unfolded: fold(parsetree)
			candb = bracketings(parsetree)
			if goldb and candb:
				prec = precision(goldb, candb)
				rec = recall(goldb, candb)
				f1 = f_measure(goldb, candb)
			else: prec = rec = f1 = 0
			if parsetree == tree or f1 == 1.0:
				assert parsetree != tree or f1 == 1.0
				msg += "exact match "
				exact = True
			else:
				msg += "LP %5.2f LR %5.2f LF %5.2f\n" % (
								100 * prec, 100 * rec, 100 * f1)
				if candb - goldb:
					msg += "\tcand-gold=%s " % printbrackets(candb - goldb)
				if goldb - candb:
					msg += "gold-cand=%s" % printbrackets(goldb - candb)
				if (candb - goldb) or (goldb - candb): msg += '\n'
				msg += "\t%s\n\t" % parsetree.pprint(margin=1000)
		elif not start:
			msg += "no parse. "
			parsetree = defaultparse([(n, t) for n, (w, t) in enumerate(sent)])
			parsetree = Tree.parse("(%s %s)" % (d.top, parsetree), parse_leaf=int)
			candb = bracketings(parsetree)
			prec = precision(goldb, candb)
			rec = recall(goldb, candb)
			f1 = f_measure(goldb, candb)
			noparse = True
		elapsedtime= time.clock() - begin
		msg += "%.2fs cpu time elapsed\n" % (elapsedtime)
		results.append(DictObj(name=stage.name, candb=candb, parsetree=parsetree,
				noparse=noparse, exact=exact, elapsedtime=elapsedtime))
	msg += "GOLD:   %s" % tree.pprint(margin=1000)
	return (nsent, msg, results)

def writeresults(results, gold, gsent, resultdir, category, sentinit):
	codecs.open("%s/%s.export" % (resultdir,
			".".join(category, "gold") if category else "gold"),
			"w", encoding='utf-8').write(''.join(
				"#BOS %d\n%s#EOS %d\n" % (n + 1, a, n + 1)
				for n, a in zip(count(sentinit), gold)))
	for result in results:
		codecs.open("%s/%s.export" % (resultdir,
			".".join(category, result.name) if category else result.name),
			"w", encoding='utf-8').writelines(export(a, [w for w, _ in b], n+1)
			for n, a, b in zip(count(sentinit), result.parsetrees, gsent))
	with open("%s/parsetimes.txt" % resultdir, "w") as f:
		f.write("# id\tlen\t%s\n" % "\t".join(result.name for result in results))
		f.writelines(
			"%d\t%d\t%s\n" % (n + 1, len(gsent[n]),
					"\t".join(str(result.elapsedtime[n]) for result in results))
				for n, _ in enumerate(results[0].elapsedtime))
	logging.info("wrote results to %s/%s.{gold,plcfrs,dop}",
		resultdir, category or "results")

def doeval(results, nsent, goldbrackets, gold, gsent):
	for n, result in enumerate(results):
		if nsent == 0: break
		logging.info("%s lp %5.2f lr %5.2f lf %5.2f\n"
			"coverage %d / %d = %5.2f %%  exact match %d / %d = %5.2f %%\n",
				result.name,
				100 * precision(goldbrackets, result.brackets),
				100 * recall(goldbrackets, result.brackets),
				100 * f_measure(goldbrackets, result.brackets),
				nsent - result.noparse, nsent,
				100.0 * (nsent - result.noparse) / nsent,
				result.exact, nsent, 100.0 * result.exact / nsent)

def root(tree):
	if tree.node == "VROOT": tree.node = "ROOT"
	else: tree = Tree("ROOT",[tree])
	return tree

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

def parsetepacoc(
		stages=(
		dict(mode='pcfg', # use the dedicated PCFG parser
			split=True,
			markorigin=True, #when splitting nodes, mark origin: VP_2 => {VP*1, VP*2}
		),
		dict(mode='plcfrs', # the agenda-based PLCFRS parser
			prune=True,	#whether to use previous chart to prune parsing this stage
			k = 10000, #number of coarse pcfg derivations to prune with; k=0 => filter only
			splitprune=True,
		),
		dict(mode='plcfrs', # the agenda-based PLCFRS parser
			prune=True,	#whether to use previous chart to prune parsing of this stage
			k = 5000, #number of coarse plcfrs derivations to prune with; k=0 => filter only
			dop=True,
			usedoubledop=True,	# when False, use DOP reduction instead
			newdd=False, #use experimental, more efficient double dop algorithm
					#the complement of the maximal recurring fragments extracted
			m = 10000,		#number of derivations to sample/enumerate
			estimator = "dop1", # choices: dop1, ewe, shortest, sl-dop[-simple]
			sample=False, both=False,
			iterate=False, #for double dop, whether to include fragments of fragments
			complement=False, #for double dop, whether to include fragments which form
		)),
		unfolded=False, bintype="binarize", h=1, v=1, factor="right",
		tailmarker='', revmarkov=False, arity_marks=True,
		arity_marks_before_bin=False,
		trainmaxwords=999, testmaxwords=999, testsents=2000,
		usetagger='stanford', resultdir="tepacoc", numproc=1):
	trainsents = 25005
	os.mkdir(resultdir)
	# Log everything, and send it to stderr, in a format with just the message.
	formatstr = '%(message)s'
	logging.basicConfig(level=logging.DEBUG, format=formatstr)
	# log up to INFO to a results log file
	fileobj = logging.FileHandler(filename='%s/output.log' % resultdir)
	fileobj.setLevel(logging.INFO)
	fileobj.setFormatter(logging.Formatter(formatstr))
	logging.getLogger('').addHandler(fileobj)
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
					[a for a, b in izip_longest(sent, corpus_sents[n])
							if a and a != b],
					[b for a, b in izip_longest(sent, corpus_sents[n])
							if b and a != b])
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
	getgrammars(trees, sents, stages, bintype, h, v, factor, tailmarker,
			revmarkov, arity_marks, arity_marks_before_bin, testmaxwords,
			resultdir, numproc)

	del corpus_sents, corpus_taggedsents, corpus_trees, corpus_blocks
	results = {}
	cnt = 0
	for cat, test in sorted(testset.items()):
		if cat == 'baseline': continue
		logging.info("category: %s", cat)
		begin = time.clock()
		results[cat] = doparse(stages, unfolded, bintype, arity_marks,
				arity_marks_before_bin, test, testmaxwords, testsents,
				trees[0].node, True, resultdir, numproc, category=cat,
				sentinit=cnt)
		cnt += len(test[0])
		if numproc == 1:
			logging.info("time elapsed during parsing: %g", time.clock() - begin)
		#else: # wall clock time here
	goldbrackets = multiset()
	totresults = [DictObj(name=stage.name) for stage in stages]
	for result in totresults:
		result.elapsedtime = [None] * cnt
		result.parsetrees = [None] * cnt
		result.brackets = multiset()
		result.exact = result.noparse = 0
	gold = []; gsent = []
	for cat, res in results.iteritems():
		logging.info("category: %s", cat)
		goldbrackets |= res[2]
		gold.extend(res[3])
		gsent.extend(res[4])
		for result, totresult in zip(res[0], totresults):
			totresult.exact += result.exact
			totresult.noparse += result.noparse
			totresult.brackets |= result.brackets
			totresult.elapsedtime.extend(result.elapsedtime)
		doeval(*res)
	logging.info("TOTAL")
	# write TOTAL results file with all tepacoc sentences (not the baseline)
	writeresults(totresults, gold, gsent, resultdir, "TOTAL", 0) #sentinit??
	doeval(totresults, cnt, goldbrackets, gold, gsent)
	# do baseline separately because it shouldn't count towards the total score
	cat = 'baseline'
	logging.info("category: %s", cat)
	doeval(*doparse(stages, unfolded, bintype, arity_marks,
			arity_marks_before_bin, testset[cat], testmaxwords, testsents,
			trees[0].node, True, resultdir, numproc, category=cat,
			sentinit=cnt))

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

def testmain():
	# Tiger treebank version 2 sample:
	# http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/annotation/sample2.export
	main(
		stages=[dict(
			mode='pcfg', # use the dedicated PCFG parser
			split=True,
			markorigin=True, #when splitting nodes, mark origin: VP_2 => {VP*1, VP*2}
			getestimates=False, #compute & store estimates
			useestimates=False,  #load & use estimates
		),
		dict(
			mode='plcfrs', # the agenda-based PLCFRS parser
			prune=True,	#whether to use previous chart to prune parsing of this stage
			splitprune=True, #VP_2[101] is treated as { VP*[100], VP*[001] } during parsing
			k = 50, #number of coarse pcfg derivations to prune with; k=0 => filter only
			neverblockre=None, #do not prune nodes with label that match regex
			getestimates=False, #compute & store estimates
			useestimates=False,  #load & use estimates
		),
		dict(
			mode='plcfrs', # the agenda-based PLCFRS parser
			prune=True,	#whether to use previous chart to prune parsing of this stage
			k = 50,		#number of coarse plcfrs derivations to prune with; k=0 => filter only
			dop=True,
			usedoubledop = False,	# when False, use DOP reduction instead
			newdd=False, #use experimental, more efficient double dop algorithm
			iterate=False, #for double dop, whether to include fragments of fragments
			complement=False, #for double dop, whether to include fragments which form
					#the complement of the maximal recurring fragments extracted
			sample=False, both=False,
			m = 10000,		#number of derivations to sample/enumerate
			estimator = "ewe", # choices: dop1, ewe, shortest, sl-dop[-simple]
			sldop_n=7,
			neverblockre=None, #do not prune nodes with label that match regex
		)],
		corpusdir=".",
		corpusfile="sample2.export",
		encoding="iso-8859-1",
		movepunct=False,
		removepunct=False,
		unfolded = False,
		testmaxwords = 40,
		trainmaxwords = 40,
		trainsents = 2,
		testsents = 1, # number of sentences to parse
		skip=0,	# dev set
		#skip=1000, #skip dev set to get test set
		usetagger=None,	#default is to use gold tags from treebank.
		bintype = "binarize", # choices: binarize, nltk, optimal, optimalhead
		factor = "right",
		revmarkov = True,
		v = 1,
		h = 2,
		arity_marks = True,
		arity_marks_before_bin = False,
		tailmarker = "",
		quiet=False, reallyquiet=False, #quiet=no per sentence results
		numproc=1,	#increase to use multiple CPUs. Set to None to use all CPUs.
	)

def test():
	from doctest import testmod, NORMALIZE_WHITESPACE, ELLIPSIS
	import bit, demos, kbest, parser, grammar, treebank, estimates, _fragments
	import agenda, coarsetofine, treetransforms, disambiguation
	modules = (bit, demos, kbest, parser, grammar, treebank, estimates,
			_fragments, agenda, coarsetofine, treetransforms, disambiguation)
	results = {}
	for mod in modules:
		print 'running doctests of', mod.__file__
		results[mod] = fail, attempted = testmod(mod, verbose=False,
			optionflags=NORMALIZE_WHITESPACE | ELLIPSIS)
		assert fail == 0, mod.__file__
	if any(not attempted for fail, attempted in results.values()):
		print "no doctests:"
		for mod, (fail, attempted) in results.iteritems():
			if not attempted: print mod.__file__,
		print
	for mod in modules:
		if hasattr(mod, 'test'): mod.test()
		else: mod.main()
	#testmain() # test this module (runexp)
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
	else:
		paramstr = open(sys.argv[1]).read()
		params = eval("dict(%s)" % paramstr)
		params['resultdir'] = sys.argv[1].rsplit(".", 1)[0]
		main(**params)
		# copy parameter file to result dir
		open("%s/params.prm" % params['resultdir'], "w").write(paramstr)
