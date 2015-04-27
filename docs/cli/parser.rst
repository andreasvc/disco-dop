
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

--verbosity=x
             0 <= x <= 4. Same effect as verbosity in parameter file.


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


