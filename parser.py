""" Simple command line interface to parse with grammar(s) in text format.  """
import os, re, sys, time, gzip, codecs
from math import exp
from getopt import gnu_getopt, GetoptError
from heapq import nlargest
from operator import itemgetter
from _parser import parse, cfgparse
from grammar import read_bitpar_grammar, read_lcfrs_grammar, FORMAT
from containers import Grammar
from kbest import lazykbest
from coarsetofine import prunechart
from disambiguation import marginalize

USAGE = """
usage: %s [options] rules lexicon [input [output]]
or: %s [options] coarserules coarselexicon finerules finelexicon \
[input [output]]

Grammars need to be binarized, and are in bitpar or LCFRS format.
When no file is given, output is written to standard output;
when additionally no input is given, it is read from standard input.
Files must be encoded in UTF-8.

    Options:
    -b k          Return the k-best parses instead of just 1.
    -s x          Use "x" as start symbol instead of default "TOP".
    --kbestctf k  Use k-best coarse-to-fine;
                  prune items not in k-best derivations (default 50).
    --prob        Print probabilities as well as parse trees.
    --mpd         In coarse-to-fine mode, produce the most probable
                  derivation (MPD) instead of the most probable parse (MPP).

%s
""" % (sys.argv[0], sys.argv[0], FORMAT)

def main():
	""" Handle command line arguments. """
	print >> sys.stderr, "PLCFRS parser - Andreas van Cranenburgh"
	options = "kbestctf= prob mpd".split()
	try:
		opts, args = gnu_getopt(sys.argv[1:], "u:b:s:", options)
		assert 2 <= len(args) <= 6, "incorrect number of arguments"
	except (GetoptError, AssertionError) as err:
		print err, USAGE
		return
	for n, filename in enumerate(args):
		assert os.path.exists(filename), (
				"file %d not found: %r" % (n + 1, filename))
	opts = dict(opts)
	k = int(opts.get("-b", 1))
	top = opts.get("-s", "TOP")
	prob = "--prob" in opts
	rules = (gzip.open if args[0].endswith(".gz") else open)(args[0])
	lexicon = codecs.getreader('utf-8')((gzip.open if args[1].endswith(".gz")
			else open)(args[1]))
	try:
		coarse = Grammar(read_lcfrs_grammar(rules, lexicon))
	except ValueError:
		rules.seek(0)
		lexicon.seek(0)
		coarse = Grammar(read_bitpar_grammar(rules, lexicon))
		lcfrs = False
	else:
		lcfrs = True
	assert top in coarse.toid, "Start symbol %r not in grammar." % top
	if 2 <= len(args) <= 4:
		infile = (codecs.open(args[2], encoding='utf-8')
				if len(args) >= 3 else sys.stdin)
		out = (codecs.open(args[3], "w", encoding='utf-8')
				if len(args) == 4 else sys.stdout)
		simple(coarse, lcfrs, infile, out, k, prob, top)
	elif 4 <= len(args) <= 6:
		threshold = int(opts.get("--kbestctf", 50))
		rules = (gzip.open if args[2].endswith(".gz") else open)(args[2])
		lexicon = codecs.getreader('utf-8')((gzip.open
				if args[3].endswith(".gz") else open)(args[3]))
		try:
			fine = Grammar(read_lcfrs_grammar(rules, lexicon))
		except ValueError:
			rules.seek(0)
			lexicon.seek(0)
			fine = Grammar(read_bitpar_grammar(rules, lexicon))
			lcfrs |= False
		else:
			lcfrs = True
		fine.getmapping(coarse, striplabelre=re.compile("@.+$"))
		assert top in fine.toid, "Start symbol %r not in fine grammar." % top
		infile = (codecs.open(args[4], encoding='utf-8')
				if len(args) >= 5 else sys.stdin)
		out = (codecs.open(args[5], "w", encoding='utf-8')
				if len(args) == 6 else sys.stdout)
		ctf(coarse, fine, lcfrs, infile, out, k, prob, top, threshold,
				"--mpd" in opts)

def simple(grammar, lcfrs, infile, out, k, printprob, top):
	""" Parse with a single grammar. """
	times = [time.clock()]
	for n, a in enumerate(infile.read().split("\n\n")):
		if not a.strip():
			continue
		sent = a.splitlines()
		assert not set(sent) - grammar.lexical.viewkeys(), (
			"unknown words and no open class tags supplied: %r" % (
			list(set(sent) - grammar.lexical.viewkeys())))
		print >> sys.stderr, "parsing:", n, " ".join(sent)
		sys.stdout.flush()
		if lcfrs:
			chart, start, _ = parse(sent, grammar, start=grammar.toid[top])
		else:
			chart, start, _ = cfgparse(sent, grammar, start=grammar.toid[top])
		if start:
			derivations, _ = lazykbest(chart, start, k, grammar.tolabel)
			if printprob:
				out.writelines("vitprob=%.16g\n%s\n" % (exp(-prob), tree)
					for tree, prob in derivations)
			else:
				out.writelines("%s\n" % tree for tree, _ in derivations)
		elif True: # baseline parse:
			out.write("(NP %s)\n" % "".join("(%s %s)" % (a, a) for a in sent))
		#else:
		#	out.write("No parse for \"%s\"\n" % " ".join(sent))
		#out.write("\n")
		out.flush()
		times.append(time.clock())
		print >> sys.stderr, times[-1] - times[-2], "s"
	print >> sys.stderr, "raw cpu time", time.clock() - times[0]
	times = [a - b for a, b in zip(times[1::2], times[::2])]
	print >> sys.stderr, "average time per sentence", sum(times) / len(times)
	print >> sys.stderr, "finished"
	out.close()

def ctf(coarse, fine, lcfrs, infile, out, k, printprob, top, threshold, mpd):
	""" Do coarse-to-fine parsing with two grammars.
	Assumes state splits in fine grammar are marked with '@'; e.g., 'NP@2'.
	Sums probabilities of derivations producing the same tree. """
	m = 10000 # number of derivations from fine grammar to marginalize
	maxlen = 999 #65
	unparsed = 0
	times = [time.clock()]

	for n, a in enumerate(infile.read().split("\n\n")):
		if not a.strip():
			continue
		sent = a.splitlines()
		if len(sent) > maxlen:
			continue
		unknown = (set(sent) - coarse.lexical.viewkeys()
			| set(sent) - fine.lexical.viewkeys())
		assert not unknown, (
			"unknown words and no open class tags supplied: %r" % list(unknown))
		print >> sys.stderr, "parsing:", n, " ".join(sent),
		if lcfrs:
			chart, start, msg = parse(sent, coarse, start=coarse.toid[top],
					exhaustive=True)
		else:
			chart, start, msg = cfgparse(sent, coarse, start=coarse.toid[top])
		print >> sys.stderr, msg
		if start:
			print >> sys.stderr, "pruning ...",
			sys.stdout.flush()
			whitelist, _ = prunechart(chart, start, coarse, fine, threshold,
					False, False, not lcfrs)
			if lcfrs:
				chart, start, _ = parse(sent, fine, start=fine.toid[top],
						whitelist=whitelist)
			else:
				chart, start, _ = cfgparse(sent, fine, start=fine.toid[top],
						chart=whitelist)
			print >> sys.stderr, msg

			assert start, (
				"sentence covered by coarse grammar could not be parsed "\
				"by fine grammar")
			print >> sys.stderr, "disambiguating ...",
			parsetrees, msg = marginalize("mpd" if mpd else "mpp", chart,
					start, fine, m, sample=False, kbest=True, sent=sent)
			print >> sys.stderr, msg
			results = nlargest(k, parsetrees, key=itemgetter(1))
			# print k-best parsetrees
			if printprob:
				label = "derivprob" if mpd else "parseprob"
				out.writelines("%s=%.16g\n%s\n" % (label, parsetrees[tree],
					tree) for tree in results)
			else:
				out.writelines("%s\n" % tree for tree in results)
		else:
			unparsed += 1
			print >> sys.stderr, "No parse"
			out.write("No parse for \"%s\"\n" % " ".join(sent))
		out.write("\n")
		times.append(time.clock())
		print >> sys.stderr, times[-1] - times[-2], "s"
		out.flush()
	print >> sys.stderr, "raw cpu time", time.clock() - times[0]
	times = [a - b for a, b in zip(times[1::2], times[::2])]
	print >> sys.stderr, "average time per sentence", sum(times) / len(times)
	print >> sys.stderr, "unparsed sentences:", unparsed
	print >> sys.stderr, "finished"
	out.close()

if __name__ == '__main__':
	main()
