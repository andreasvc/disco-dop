""" Simple command line interface to parse with grammar(s) in text format.  """
from __future__ import print_function
import io
import os
import re
import sys
import time
import gzip
import codecs
import string  # pylint: disable=W0402
from math import exp
from getopt import gnu_getopt, GetoptError
from heapq import nlargest
from operator import itemgetter
from . import plcfrs, pcfg
from .grammar import FORMAT
from .containers import Grammar
from .kbest import lazykbest
from .coarsetofine import prunechart
from .disambiguation import marginalize

USAGE = """
usage: %s [options] rules lexicon [input [output]]
or: %s [options] coarserules coarselexicon finerules finelexicon \
[input [output]]

Grammars need to be binarized, and are in bitpar or PLCFRS format.
When no file is given, output is written to standard output;
when additionally no input is given, it is read from standard input.
Files must be encoded in UTF-8.
Input should contain one token per line, with sentences delimited by two
newlines. Output consists of bracketed trees, with discontinuities indicated
through indices pointing to words in the original sentence.

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
	print("PLCFRS parser - Andreas van Cranenburgh", file=sys.stderr)
	options = "kbestctf= prob mpd".split()
	try:
		opts, args = gnu_getopt(sys.argv[1:], "u:b:s:", options)
		assert 2 <= len(args) <= 6, "incorrect number of arguments"
	except (GetoptError, AssertionError) as err:
		print(err, USAGE)
		return
	for n, filename in enumerate(args):
		assert os.path.exists(filename), (
				"file %d not found: %r" % (n + 1, filename))
	opts = dict(opts)
	k = int(opts.get("-b", 1))
	top = opts.get("-s", "TOP")
	prob = "--prob" in opts
	rules = (gzip.open if args[0].endswith(".gz") else open)(args[0]).read()
	lexicon = codecs.getreader('utf-8')((gzip.open if args[1].endswith(".gz")
			else open)(args[1])).read()
	bitpar = rules[0] in string.digits
	coarse = Grammar(rules, lexicon, start=top, bitpar=bitpar)
	if 2 <= len(args) <= 4:
		infile = (io.open(args[2], encoding='utf-8')
				if len(args) >= 3 else sys.stdin)
		out = (io.open(args[3], "w", encoding='utf-8')
				if len(args) == 4 else sys.stdout)
		simple(coarse, bitpar, infile, out, k, prob)
	elif 4 <= len(args) <= 6:
		threshold = int(opts.get("--kbestctf", 50))
		rules = (gzip.open if args[2].endswith(".gz") else open)(args[2]).read()
		lexicon = codecs.getreader('utf-8')((gzip.open
				if args[3].endswith(".gz") else open)(args[3])).read()
		# detect bitpar format
		bitpar = rules[0] in string.digits
		fine = Grammar(rules, lexicon, start=top, bitpar=bitpar)
		fine.getmapping(coarse, striplabelre=re.compile(b"@.+$"))
		infile = (io.open(args[4], encoding='utf-8')
				if len(args) >= 5 else sys.stdin)
		out = (io.open(args[5], "w", encoding='utf-8')
				if len(args) == 6 else sys.stdout)
		ctf(coarse, fine, bitpar, infile, out, k, prob, threshold,
				"--mpd" in opts)


def simple(grammar, ispcfg, infile, out, k, printprob):
	""" Parse with a single grammar. """
	times = [time.clock()]
	for n, a in enumerate(infile.read().split("\n\n")):
		if not a.strip():
			continue
		sent = a.splitlines()
		assert not set(sent) - set(grammar.lexical), (
			"unknown words and no open class tags supplied: %r" % (
			list(set(sent) - set(grammar.lexical))))
		print("parsing %d: %s" % (n, ' '.join(sent)), file=sys.stderr)
		sys.stdout.flush()
		if ispcfg:
			chart, start, _ = pcfg.parse(sent, grammar)
		else:
			chart, start, _ = plcfrs.parse(sent, grammar, exhaustive=k > 1)
		if start:
			derivations = lazykbest(chart, start, k, grammar.tolabel)[0]
			if printprob:
				out.writelines("vitprob=%.16g\n%s\n" % (exp(-prob), tree)
					for tree, prob in derivations)
			else:
				out.writelines("%s\n" % tree for tree, _ in derivations)
		elif True:  # baseline parse:
			out.write("(NP %s)\n" % "".join("(%s %s)" % (a, a) for a in sent))
		#else:
		#	out.write("No parse for \"%s\"\n" % " ".join(sent))
		#out.write("\n")
		out.flush()
		times.append(time.clock())
		print(times[-1] - times[-2], "s", file=sys.stderr)
	print("raw cpu time", time.clock() - times[0], file=sys.stderr)
	times = [a - b for a, b in zip(times[1::2], times[::2])]
	print("average time per sentence", sum(times) / len(times), file=sys.stderr)
	print("finished", file=sys.stderr)
	out.close()


def ctf(coarse, fine, ispcfg, infile, out, k, printprob, threshold, mpd):
	""" Do coarse-to-fine parsing with two grammars.
	Assumes state splits in fine grammar are marked with '@'; e.g., 'NP@2'.
	Sums probabilities of derivations producing the same tree. """
	m = 10000  # number of derivations from fine grammar to marginalize
	maxlen = 999  # 65
	unparsed = 0
	times = [time.clock()]

	for n, a in enumerate(infile.read().split("\n\n")):
		if not a.strip():
			continue
		sent = a.splitlines()
		if len(sent) > maxlen:
			continue
		unknown = (set(sent) - set(coarse.lexical)
			| set(sent) - set(fine.lexical))
		assert not unknown, (
			"unknown words and no open class tags supplied: %r" % list(unknown))
		print("parsing %d: %s" % (n, ' '.join(sent)), file=sys.stderr)
		if ispcfg:
			chart, start, msg = pcfg.parse(sent, coarse)
		else:
			chart, start, msg = plcfrs.parse(sent, coarse, exhaustive=True)
		print(msg, file=sys.stderr)
		if start:
			print("pruning ...", file=sys.stderr, end='')
			sys.stdout.flush()
			whitelist, _ = prunechart(chart, start, coarse, fine, threshold,
					False, False, ispcfg)
			if ispcfg:
				chart, start, _ = pcfg.parse(sent, fine, chart=whitelist)
			else:
				chart, start, _ = plcfrs.parse(sent, fine, whitelist=whitelist,
						exhaustive=k > 1)
			print(msg, file=sys.stderr)

			assert start, (
				"sentence covered by coarse grammar could not be parsed "
				"by fine grammar")
			print("disambiguating ...", file=sys.stderr, end='')
			parsetrees, msg = marginalize("mpd" if mpd else "mpp", chart,
					start, fine, m, sample=False, kbest=True, sent=sent)
			print(msg, file=sys.stderr)
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
			print("No parse", file=sys.stderr)
			out.write("No parse for \"%s\"\n" % " ".join(sent))
		out.write("\n")
		times.append(time.clock())
		print(times[-1] - times[-2], "s", file=sys.stderr)
		out.flush()
	times = [a - b for a, b in zip(times[1::2], times[::2])]
	print("raw cpu time", time.clock() - times[0],
			"\naverage time per sentence", sum(times) / len(times),
			"\nunparsed sentences:", unparsed,
			"\nfinished",
			file=sys.stderr)
	out.close()

if __name__ == '__main__':
	main()
