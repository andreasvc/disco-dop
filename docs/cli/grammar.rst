
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

