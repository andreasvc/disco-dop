Command line options
====================

A parser for Probalistic Linear Context-Free Rewriting Systems (LCFRS) and
Probabilistic Context-Free Grammars (PCFG), as well as facilities to extract
and parse with data-oriented parsing (DOP) grammars.

Usage: ``discodop <command> [arguments]``

Command is one of:

:`runexp`_:           Run experiment: grammar extraction, parsing & evaluation.
:`fragments`_:        Extract recurring fragments from treebanks.
:`eval`_:             Evaluate discontinuous parse trees; similar to EVALB.
:`treetransforms`_:   Apply tree transformations and convert between formats.
:`treedraw`_:         Visualize (discontinuous) trees
:`treesearch`_:       Search through treebanks with queries.
:`grammar`_:          Read off grammars from treebanks.
:`parser`_:           Simple command line parser.
:demos:               Show some demonstrations of formalisms encoded in LCFRS.
:`gen`_:              Generate sentences from a PLCFRS.

for additional instructions issue: ``discodop <command> --help``

runexp
------
Usage: ``discodop runexp <parameter file> [--rerun]``

If a parameter file is given, an experiment is run. See the file sample.prm for
an example parameter file. To repeat an experiment with an existing grammar,
pass the option ``--rerun``. The directory with the name of the parameter file
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
| Usage: ``discodop fragments <treebank1> [treebank2] [options]``
| or: ``discodop fragments --batch=<dir> <treebank1> <treebank2>... [options]``

If only one treebank is given, fragments occurring at least twice are sought.
If two treebanks are given, finds common fragments between first & second.
Input is in Penn treebank format (S-expressions), one tree per line.
Output contains lines of the form "tree<TAB>frequency".
Frequencies refer to the first treebank by default.
Output is sent to stdout; to save the results, redirect to a file.

Options:

--fmt=(export|bracket|discbracket|tiger|alpino|dact)
              when format is not ``bracket``, work with discontinuous trees;
              output is in ``discbracket`` format:
              tree<TAB>sentence<TAB>frequency
              where ``tree`` has indices as leaves, referring to elements of
              ``sentence``, a space separated list of words.

-o file       Write output to ``file`` instead of stdout.
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
--nofreq      do not report frequencies.
--approx      report counts of occurrence as maximal fragment (lower bound)
--relfreq     report relative frequencies wrt. root node of fragments.
--debin       debinarize fragments.
--twoterms    only consider fragments with at least two lexical terminals.
--adjacent    only consider pairs of adjacent fragments (n, n + 1).
--alt         alternative output format: (NP (DT "a") NN)
              default: (NP (DT a) (NN ))
--debug       extra debug information, ignored when numproc > 1.
--quiet       disable all messages.


eval
----
Evaluation of (discontinuous) parse trees, following EVALB as much as possible.

Usage: ``discodop eval <gold> <parses> [param] [options]``

where gold and parses are files with parse trees, param is an EVALB parameter
file, and options may consist of:

--cutofflen=n    Overrides the sentence length cutoff of the parameter file.
--verbose        Print table with per sentence information.
--debug          Print debug information with per sentence bracketings etc.
--disconly       Only evaluate bracketings of discontinuous constituents
                 (only affects Parseval measures).

--goldfmt, --parsesfmt=(export|bracket|discbracket|tiger|alpino|dact)
                 Specify corpus format [default: export].

--fmt=[...]      Shorthand for setting both ``--goldfmt`` and ``--parsesfmt``.

--goldenc, --parsesenc=(utf-8|iso-8859-1|...)
                 Specify encoding [default: utf-8].

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

:DELETE_ROOT_PRETERMS:
                 when enabled, preterminals directly under the root in
                 gold trees are ignored for scoring purposes.
:DISC_ONLY:      only consider discontinuous constituents for F-scores.
:TED:            when enabled, give tree-edit distance scores; disabled by
                 default as these are slow to compute.
:DEBUG:
                 :-1: only print summary table
                 :0: additionally, print category / tag breakdowns (default)
                   (after application of cutoff length).
                 :1: give per-sentence results (``--verbose``)
                 :2: give detailed information for each sentence (``--debug``)
:MAX_ERROR:      this values is ignored, no errors are tolerated.
                 the parameter is accepted to support usage of unmodified
                 EVALB parameter files.


treetransforms
--------------
Treebank binarization and conversion

Usage: ``discodop treetransforms [options] <action> [input [output]]``

where input and output are treebanks; standard in/output is used if not given.
action is one of::

    none
    binarize [-h x] [-v x] [--factor=left|right]
    optimalbinarize [-h x] [-v x]
    unbinarize
    introducepreterminals
    splitdisc [--markorigin]
    mergedisc
    transform [--reverse] [--transforms=<NAME1,NAME2...>]

options may consist of:

--inputfmt=(export|bracket|discbracket|tiger|alpino|dact)
                Input treebank format [default: export].

--outputfmt=(export|bracket|discbracket|dact|conll|mst|tokens|wordpos)
                Output treebank format [default: export].

--fmt=x         Shortcut to specify both input and output format.

--inputenc, --outputenc, --enc=(utf-8|iso-8859-1|...)
                Treebank encoding [default: utf-8].

--slice=<n:m>   select a range of sentences from input starting with *n*,
                up to but not including *m*; as in Python, *n* or *m* can be left
                out or negative, and the first index is 0.

--renumber      Replace sentence IDs with numbers starting from 1,
                padded with 8 spaces.

--maxlen=n      only select sentences with up to *n* tokens.
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
--lemmas=x      :'no' (default): do not use lemmas.
                :'add': concatenate lemmas to terminals, e.g., word/lemma
                :'replace': use lemma instead of terminals
                :'between': insert node with lemma between POS tag and word,
                    e.g., (NN (man men))
--ensureroot=x  add root node labeled ``x`` to trees if not already present.

--factor=(left|right)
                specify left- or right-factored binarization [default: right].

-h n            horizontal markovization. default: infinite (all siblings)
-v n            vertical markovization. default: 1 (immediate parent only)
--leftunary     make initial / final productions of binarized constituents
--rightunary    ... unary productions.
--tailmarker    mark rightmost child (the head if headrules are applied), to
                avoid cyclic rules when ``--leftunary`` and ``--rightunary``
                are used.
--headrules=x   turn on head finding; affects binarization.
                reads rules from file ``x`` (e.g., "negra.headrules").
--markheads     mark heads with ``^`` in phrasal labels.
--reverse       reverse the transformations given by ``--transform``
--transforms    specify names of tree transformations to apply; for possible
                names, cf. :mod:`discodop.treebanktransforms` module.

.. note::
    selecting the formats ``conll`` or ``mst`` results in an unlabeled
    dependency conversion and requires the use of heuristic head rules
    (``--headrules``), to ensure that all constituents have a child marked as
    head. A command line interface to perform transformations on treebanks such
    as binarization.

grammar
-------
Read off grammars from treebanks.

| Usage: ``discodop grammar <type> <input> <output> [options]``
| or: ``discodop param <parameter-file> <output-directory>``
| or: ``discodop info <rules-file>``
| or: ``discodop merge (rules|lexicon|fragments) <input1> <input2>... <output>``

``type`` is one of:

:pcfg:            Probabilistic Context-Free Grammar (treebank grammar).
:plcfrs:
                  Probabilistic Linear Context-Free Rewriting System
                  (discontinuous treebank grammar).

:ptsg:            Probabilistic Tree-Substitution Grammar.
:dopreduction:    All-fragments PTSG using Goodman's reduction.
:doubledop:       PTSG from recurring fragmensts.
:param:           Extract a series of grammars according to parameters.
:info:            Print statistics for PLCFRS/bitpar rules.
:merge:
                  Interpolate given sorted grammars into a single grammar.
                  Input can be a rules, lexicon or fragment file.

``input`` is a binarized treebank, or in the ``ptsg`` case, weighted fragments
in the same format as the output of the ``discodop fragments`` command;
``input`` may contain discontinuous constituents, except for the ``pcfg`` case.
``output`` is the base name for the filenames to write the grammar to; the
filenames will be ``<output>.rules`` and ``<output>.lex``. NB: both the
``info`` and ``merge`` commands expect grammars to be sorted by LHS, such as
the ones created by this tool.

Options:

--inputfmt=(export|bracket|discbracket|tiger|alpino|dact)
          The treebank format [default: export].

--inputenc=(utf-8|iso-8859-1|...)
          Treebank encoding [default: utf-8].

--dopestimator=(rfe|ewe|shortest|...)
          The DOP estimator to use with dopreduction/doubledop [default: rfe].

--numproc=(1|2|...)
          Number of processes to start [default: 1].
          Only relevant for double dop fragment extraction.

--gzip
          compress output with gzip, view with ``zless`` &c.

--packed
          use packed graph encoding for DOP reduction

--bitpar
          produce an unbinarized grammar for use with bitpar

-s X
          start symbol to use for PTSG.

When a PCFG is requested, or the input format is ``bracket`` (Penn format), the
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
A command line interface for parsing new texts with an existing grammar.

| Usage: ``discodop parser [options] <grammar/> [input files]``
| or:    ``discodop parser --simple [options] <rules> <lexicon> [input [output]]``

``grammar/`` is a directory with a model produced by ``discodop runexp``.
When no filename is given, input is read from standard input and the results
are written to standard output. Input should contain one sentence per line
with space-delimited tokens. Output consists of bracketed trees in
selected format. Files must be encoded in UTF-8.

General options:

-x           Input is one token per line, sentences separated by two
             newlines (like bitpar).
-b k         Return the k-best parses instead of just 1.
--prob       Print probabilities as well as parse trees.
--tags       Tokens are of the form ``word/POS``; give both to parser.

--fmt=(export|bracket|discbracket|alpino|conll|mst|wordpos)
             Format of output [default: discbracket].

--numproc=k  Launch k processes, to exploit multiple cores.
--simple     Parse with a single grammar and input file; similar interface
             to bitpar. The files ``rules`` and ``lexicon`` define a binarized
             grammar in bitpar or PLCFRS format.

Options for simple mode:

-s x         Use ``x`` as start symbol instead of default ``TOP``.
--bt=file    Apply backtransform table to recover TSG derivations.
--mpp=k      By default, the output consists of derivations, with the most
             probable derivation (MPD) ranked highest. With a PTSG such as
             DOP, it is possible to aim for the most probable parse (MPP)
             instead, whose probability is the sum of any number of the
             k-best derivations.

--obj=(mpd|mpp|mcc|shortest|sl-dop)
             Objective function to maximize [default: mpd].

-m x         Use x derivations to approximate objective functions;
             mpd and shortest require only 1.
--bitpar     Use bitpar to parse with an unbinarized grammar.


treedraw
--------
Usage: ``discodop treedraw [<treebank>...] [options]``

--fmt=(export|bracket|discbracket|tiger|alpino|dact)
                  Specify corpus format [default: export].

--encoding=enc    Specify a different encoding than the default utf-8.
--functions=x     :'leave'=default: leave syntactic labels as is,
                  :'remove': strip functions off labels,
                  :'add': show both syntactic categories and functions,
                  :'replace': only show grammatical functions.

--morphology=x    :'no': only show POS tags [default],
                  :'add': concatenate morphology tags to POS tags,
                  :'replace': replace POS tags with morphology tags,
                  :'between': add morphological node between POS tag and word.

--abbr            abbreviate labels longer than 5 characters.
--plain           disable ANSI colors.
-n, --numtrees=x  only display the first x trees from the input.

If no treebank is given, input is read from standard input; format is detected.
If more than one treebank is specified, trees will be displayed in parallel.
Pipe the output through ``less -R`` to preserve the colors.

treesearch
----------
Search through treebanks with queries.

Usage: ``discodop treesearch [--engine=(tgrep2|xpath|regex)] [-t|-s|-c] <query> <treebank>...``

Options:

--engine=<x>, -e <x>
                Select query engine; possible options:

                :tgrep2:
                    tgrep2 queries (default); files are bracket corpora
                    (optionally precompiled into tgrep2 format).

                :xpath: arbitrary xpath queries; files are dact XML corpora.
                :regex: search through tokenized sentences with Python regexps
--counts, -c    report counts
--sents, -s     output sentences (default)
--trees, -t     output visualizations of trees
--brackets, -b  output raw trees in the original corpus format
--only-matching, -o
                only output the matching portion
                with ``--sents``, ``--trees``, and ``--brackets``
--line-number, -n
                Prefix each line of output with the sentence number within
                its input file.
--macros=<x>, -m <x>
                file with macros
--numthreads=<x>
                Number of concurrent threads to use.

gen
---
Generate random sentences with a PLCFRS or PCFG.
Reads grammar from a text file in PLCFRS or bitpar format.

| Usage: ``discodop gen [--verbose] <rules> <lexicon>``
| or: ``discodop gen --test``

Grammar is assumed to be in utf-8; may be gzip'ed (.gz extension).
