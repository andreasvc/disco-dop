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
from .runexp import Parser, DictObj, DEFAULTSTAGE

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
	stages = []
	infile = (io.open(args[2], encoding='utf-8')
			if len(args) >= 3 else sys.stdin)
	out = (io.open(args[3], "w", encoding='utf-8')
			if len(args) == 4 else sys.stdout)
	stage = DEFAULTSTAGE.copy()
	stage.update(
			name='coarse',
			mode='pcfg' if bitpar else 'plcfrs',
			grammar=coarse,
			secondarymodel=None,
			backtransform=None,
			m=k)
	stages.append(DictObj(stage))
	if 4 <= len(args) <= 6:
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
		stage = DEFAULTSTAGE.copy()
		stage.update(
				name='fine',
				mode='pcfg' if bitpar else 'plcfrs',
				grammar=fine,
				secondarymodel=None,
				backtransform=None,
				m=k,
				k=threshold,
				objective='mpd' if '--mpd' in opts else 'mpp')
		stages.append(DictObj(stage))
	doparsing(Parser(stages), infile, out, prob)


def doparsing(parser, infile, out, printprob):
	""" Parse sentences from file and write results to file, log to stdout. """
	times = [time.clock()]
	unparsed = 0
	for n, a in enumerate(infile.read().split("\n\n")):
		if not a.strip():
			continue
		sent = a.splitlines()
		lexicon = parser.stages[0].grammar.lexical
		assert not set(sent) - set(lexicon), (
			"unknown words and no open class tags supplied: %r" % (
			list(set(sent) - set(lexicon))))
		print("parsing %d: %s" % (n, ' '.join(sent)), file=sys.stderr)
		sys.stdout.flush()
		result = list(parser.parse(sent))[-1]
		if result.noparse:
			unparsed += 1
		if printprob:
			out.writelines("vitprob=%.16g\n%s\n" % (exp(-prob), tree)
					for tree, prob in sorted(result.parsetrees.items(),
						key=itemgetter(1)))
		else:
			out.writelines("%s\n" % tree
					for tree in sorted(result.parsetrees,
						key=result.parsetrees.get))
		out.flush()
		times.append(time.clock())
		print(times[-1] - times[-2], "s", file=sys.stderr)
	times = [a - b for a, b in zip(times[1::2], times[::2])]
	print("raw cpu time", time.clock() - times[0],
			"\naverage time per sentence", sum(times) / len(times),
			"\nunparsed sentences:", unparsed,
			"\nfinished",
			file=sys.stderr)
	out.close()

if __name__ == '__main__':
	main()
