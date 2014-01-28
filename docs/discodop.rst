discodop command line options
=============================

A parser for Probalistic Linear Context-Free Rewriting Systems (LCFRS) and
Probabilistic Context-Free Grammars (PCFG), as well as facilities to extract
and parse with data-oriented parsing (DOP) grammars.

Usage: discodop <command> [arguments]

Command is one of:

:runexp:           Run experiment: grammar extraction, parsing & evaluation.
:fragments:        Extract recurring fragments from treebanks.
:eval:             Evaluate discontinuous parse trees; similar to EVALB.
:treetransforms:   Apply tree transformations and convert between formats.
:treedraw:         Visualize (discontinuous) trees
:grammar:          Read off grammars from treebanks.
:parser:           Simple command line parser.
:demos:            Show some demonstrations of formalisms encoded in LCFRS.
:gen:              Generate sentences from a PLCFRS.

for additional instructions issue: discodop <command> --help

runexp
------
Usage: discodop runexp <parameter file> [--rerun]

If a parameter file is given, an experiment is run. See the file sample.prm for
an example parameter file. To repeat an experiment with an existing grammar,
pass the option --rerun. The directory with the name of the parameter file
without extension must exist in the current path; its results will be
overwritten.

Example invocation::

    discodop runexp filename.prm

This will create a new directory with the base name of the parameter file, i.e.,
``filename/`` in this case. This directory must not exist yet, to avoid
accidentally overwriting previous results. The directory will contain the
grammar rules and lexicon in a text format, as well as the parsing results and
the gold standard file in Negra's export format.

fragments
---------
Usage: discodop fragments <treebank1> [treebank2] [options]

or: discodop fragments --batch=<dir> <treebank1> <treebank2>... [options]

If only one treebank is given, fragments occurring at least twice are sought.
If two treebanks are given, finds common fragments between first & second.
Input is in Penn treebank format (S-expressions), one tree per line.
Output contains lines of the form "tree<TAB>frequency".
Frequencies refer to the first treebank by default.
Output is sent to stdout; to save the results, redirect to a file.

Options:

  --fmt=<export|bracket|discbracket|tiger|alpino|dact>
                when format is not 'bracket', work with discontinuous trees;
                output is in 'discbracket' format:
                tree<TAB>sentence<TAB>frequency
                where "tree' has indices as leaves, referring to elements of
                "sentence", a space separated list of words.
  --indices     report sets of indices instead of frequencies.
  --cover       include all depth-1 fragments of first treebank corresponding
                to single productions.
  --complete    find complete matches of fragments from treebank1 (needle) in
                treebank2 (haystack); frequencies are from haystack.
  --batch=dir   enable batch mode; any number of treebanks > 1 can be given;
                first treebank will be compared to all others.
                Results are written to filenames of the form dir/A_B.
  --numproc=n   use n independent processes, to enable multi-core usage
                (default: 1); use 0 to detect the number of CPUs.
  --numtrees=n  only read first n trees from first treebank
  --encoding=x  use x as treebank encoding, e.g. utf-8, iso-8859-1, etc.
  --approx      report approximate frequencies (lower bound)
  --nofreq      do not report frequencies.
  --relfreq     report relative frequencies wrt. root node of fragments.
  --quadratic   use the slower, quadratic algorithm for finding fragments.
  --alt         alternative output format: (NP (DT "a") NN)
                default: (NP (DT a) (NN ))
  --debug       extra debug information, ignored when numproc > 1.
  --quiet       disable all messages.


eval
----
Evaluation of (discontinuous) parse trees, following EVALB as much as possible.

Usage: discodop eval <gold> <parses> [param] [options]

where gold and parses are files with parse trees, param is an EVALB parameter
file, and options may consist of:

--cutofflen=n    Overrides the sentence length cutoff of the parameter file.
--verbose        Print table with per sentence information.
--debug          Print debug information with per sentence bracketings etc.
--disconly       Only evaluate bracketings of discontinuous constituents
                 (only affects Parseval measures).

--goldfmt, --parsesfmt=<*export|bracket|discbracket|tiger|alpino|dact>
                 Specify corpus format.

--goldenc, --parsesenc=<*utf-8|iso-8859-1|...>
                 Specify a different encoding than the default utf-8.

--ted            Enable tree-edit distance evaluation.
--headrules=x    Specify file with rules for head assignment of constituents
                 that do not already have a child marked as head; this
                 enables dependency evaluation.

--functions=x    'remove'=default: strip functions off labels,
                 'leave': leave syntactic labels as is,
                 'add': evaluate both syntactic categories and functions,
                 'replace': only evaluate grammatical functions.

--morphology=x   'no'=default: only evaluate POS tags,
                 'add': concatenate morphology tags to POS tags,
                 'replace': replace POS tags with morphology tags,
                 'between': add morphological node between POS tag and word.

The parameter file should be encoded in utf-8 and supports the following
options (in addition to those described in the README of EVALB):

:DISC_ONLY:      only consider discontinuous constituents for F-scores.
:TED:            when enabled, give tree-edit distance scores; disabled by
                 default as these are slow to compute.
:DEBUG:
                 :-1: only print summary table
                 :0: additionally, print category / tag breakdowns (default)
                   (after application of cutoff length).
                 :1: give per-sentence results ('--verbose')
                 :2: give detailed information for each sentence ('--debug')
:MAX_ERROR:      this values is ignored, no errors are tolerated.
                 the parameter is accepted to support usage of unmodified
                 EVALB parameter files.


treetransforms
--------------
Treebank binarization and conversion

Usage: discodop treetransforms [options] <action> [input [output]]

where input and output are treebanks; standard in/output is used if not given.
action is one of::

    none
    binarize [-h x] [-v x] [--factor=left|*right]
    optimalbinarize [-h x] [-v x]
    unbinarize
    introducepreterminals
    splitdisc [--markorigin]
    mergedisc

options may consist of (* marks default option):

--inputfmt=<*export|bracket|discbracket|tiger|alpino|dact>
                Input treebank format.
--outputfmt=<*export|bracket|discbracket|dact|conll|mst|tokens|wordpos>
                Output treebank format.
--inputenc, --outputenc=<*utf-8|iso-8859-1|...>
                Treebank encoding.
--slice=<n:m>   select a range of sentences from input starting with n,
                up to but not including m; as in Python, n or m can be left
                out or negative, and the first index is 0.
--punct=x       possible options:

                :'remove': remove any punctuation.
                :'move': re-attach punctuation to nearest constituent
                      to minimize discontinuity.
                :'restore': attach punctuation under root node.
--functions=x   :'leave': (default): leave syntactic labels as is,
                :'remove': strip away hyphen-separated function labels
                :'add': concatenate syntactic categories with functions,
                :'replace': replace syntactic labels w/grammatical functions.
--morphology=x  :'no' (default): use POS tags as preterminals
                :'add': concatenate morphological information to POS tags,
                    e.g., DET/sg.def
                :'replace': use morphological information as preterminal label
                :'between': insert node with morphological information between
                    POS tag and word, e.g., (DET (sg.def the))
--lemmas        insert node with lemma between word and POS tag.
--ensureroot=x  add root node labeled 'x' to trees if not already present.
--factor=<left|*right>
                whether binarization factors to the left or right
-h n            horizontal markovization. default: infinite (all siblings)
-v n            vertical markovization. default: 1 (immediate parent only)
--leftunary     make initial / final productions of binarized constituents
--rightunary    ... unary productions.
--tailmarker    mark rightmost child (the head if headrules are applied), to
                avoid cyclic rules when --leftunary and --rightunary are used.
--headrules=x   turn on head finding; affects binarization.
                reads rules from file "x" (e.g., "negra.headrules").
--markheads     mark heads with '^' in phrasal labels.


Note: selecting the formats 'conll' or 'mst' results in an unlabeled dependency
    conversion and requires the use of heuristic head rules (--headrules),
    to ensure that all constituents have a child marked as head.
    A command line interface to perform transformations on
    treebanks such as binarization.

grammar
-------
Read off grammars from treebanks.
Usage::

   discodop grammar <type> <input> <output> [options]

type is one of:
   pcfg
   plcfrs
   ptsg
   dopreduction
   doubledop

input is a binarized treebank, or in the ptsg case, weighted fragments
in the same format as the output of the discodop fragments command;
output is the base name for the filenames to write the grammar to.

Options (* marks default option):

--inputfmt=<*export|bracket|discbracket|tiger|alpino|dact>
          The treebank format.

--inputenc=<\*utf-8|iso-8859-1|...>
          Treebank encoding.

--dopestimator=<*rfe|ewe|shortest|...>
          The DOP estimator to use with dopreduction/doubledop.

--numproc=<*1|2|...>
          only relevant for double dop fragment extraction

--gzip
          compress output with gzip, view with zless &c.

--packed
          use packed graph encoding for DOP reduction

--bitpar
          produce an unbinarized grammar for use with bitpar

-s X
          start symbol to use for PTSG.

When a PCFG is requested, or the input format is 'bracket' (Penn format), the
output will be in bitpar format. Otherwise the grammar is written as a PLCFRS.
The encoding of the input treebank may be specified. Output encoding will be
ASCII for the rules, and utf-8 for the lexicon.

The PLCFRS format is as follows. Rules are delimited by newlines.
Fields are separated by tabs. The fields are::

    LHS	RHS1	[RHS2]	yield-function	weight

The yield function defines how the spans of the RHS nonterminals
are combined to form the spans of the LHS nonterminal. Components of the yield
function are comma-separated, 0 refers to a component of the first RHS
nonterminal, and 1 from the second. Weights are expressed as rational
fractions.
The lexicon is defined in a separate file. Lines start with a single word,
followed by pairs of possible tags and their probabilities::

    WORD	TAG1	PROB1	[TAG2	PROB2 ...]

Example::

    rules:   S	NP	VP	010	1/2
             VP_2	VB	NP	0,1	2/3
             NP	NN	0	1/4
    lexicon: Haus	NN	3/10	JJ	1/9


parser
------
A basic command line interface to the parser comparable to bitpar.
Reads grammars from text files.

usage: discodop parser [options] <rules> <lexicon> [input [output]]

or:    discodop parser [options] --ctf k <coarserules> <coarselex>
          <finerules> <finelex> [input [output]]

Grammars need to be binarized, and are in bitpar or PLCFRS format.
When no file is given, output is written to standard output;
when additionally no input is given, it is read from standard input.
Files must be encoded in utf-8.
Input should contain one token per line, with sentences delimited by two
newlines. Output consists of bracketed trees, with discontinuities indicated
through indices pointing to words in the original sentence.

Options:

  -b k           Return the k-best parses instead of just 1.
  -s x           Use "x" as start symbol instead of default "TOP".
  -z             Input is one sentence per line, space-separated tokens.
  --ctf=k        Use k-best coarse-to-fine; prune items not in top k derivations
  --prob         Print probabilities as well as parse trees.
  --mpp=k        By default, the output consists of derivations, with the most
                 probable derivation (MPD) ranked highest. With a PTSG such as
                 DOP, it is possible to aim for the most probable parse (MPP)
                 instead, whose probability is the sum of any number of the
                 k-best derivations.
  --bt=file      backtransform table to recover TSG derivations.
  --bitpar       use bitpar to parse with an unbinarized grammar.
  --numproc=k    launch k processes, to exploit multiple cores.

The PLCFRS format is as follows. Rules are delimited by newlines.
Fields are separated by tabs. The fields are::

    LHS	RHS1	[RHS2]	yield-function	weight

The yield function defines how the spans of the RHS nonterminals
are combined to form the spans of the LHS nonterminal. Components of the yield
function are comma-separated, 0 refers to a component of the first RHS
nonterminal, and 1 from the second. Weights are expressed as rational
fractions.
The lexicon is defined in a separate file. Lines start with a single word,
followed by pairs of possible tags and their probabilities::

    WORD	TAG1	PROB1	[TAG2	PROB2 ...]

Example::

    rules:   S	NP	VP	010	1/2
             VP_2	VB	NP	0,1	2/3
             NP	NN	0	1/4
    lexicon: Haus	NN	3/10	JJ	1/9

treedraw
--------
Usage: discodop treedraw [<treebank>...] [options]

Options (* marks default option):

--fmt=<*export|bracket|discbracket|tiger|alpino|dact>
                 Specify corpus format.

--encoding=enc   Specify a different encoding than the default utf-8.
--functions=x    :'leave'=default: leave syntactic labels as is,
                 :'remove': strip functions off labels,
                 :'add': show both syntactic categories and functions,
                 :'replace': only show grammatical functions.

--morphology=x   :'no'=default: only show POS tags,
                 :'add': concatenate morphology tags to POS tags,
                 :'replace': replace POS tags with morphology tags,
                 :'between': add morphological node between POS tag and word.

--abbr           abbreviate labels longer than 5 characters.
--plain          disable ANSI colors.

If no treebank is given, input is read from standard input; format is detected.
If more than one treebank is specified, trees will be displayed in parallel.
Pipe the output through 'less -R' to preserve the colors.

gen
---
Generate random sentences with a PLCFRS or PCFG.
Reads grammar from a text file in PLCFRS or bitpar format.
Usage: discodop gen [--verbose] <rules> <lexicon>
or: discodop gen --test

Grammar is assumed to be in utf-8; may be gzip'ed (.gz extension).


Web interfaces
--------------
There are three web based tools in the ``web/`` directory. These require Flask to
be installed.

``parse.py``
    A web interface to the parser. Expects a series of grammars
    in subdirectories of ``web/grammars/``, each containing grammar files
    as produced by running ``discodop runexp``.
    `Download grammars <http://staff.science.uva.nl/~acranenb/grammars/>`_
    for English, German, and Dutch, as used in the 2013 IWPT paper.

``treesearch.py``
    A web interface for searching through treebanks. Expects
    one or more treebanks with the ``.mrg`` or ``.dact`` extension in the
    directory ``web/corpus/`` (sample included). Depends on
    `tgrep2 <http://tedlab.mit.edu/~dr/Tgrep2/>`_,
    `alpinocorpus <https://github.com/rug-compling/alpinocorpus-python>`_, and
    `style <http://www.gnu.org/software/diction/diction.html>`_.

``treedraw.py``
    A web interface for drawing discontinuous trees in various
    formats.
