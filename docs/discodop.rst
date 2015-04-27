Command line options
====================

A parser for Probalistic Linear Context-Free Rewriting Systems (LCFRS) and
Probabilistic Context-Free Grammars (PCFG), as well as facilities to extract
and parse with data-oriented parsing (DOP) grammars.

Usage: ``discodop <command> [arguments]``

Command is one of:

:doc:`runexp <cli/runexp>`
    Run experiment: grammar extraction, parsing & evaluation.
:doc:`fragments <cli/fragments>`
    Extract recurring fragments from treebanks.
:doc:`eval <cli/eval>`
    Evaluate discontinuous parse trees; similar to EVALB.
:doc:`treetransforms <cli/treetransforms>`
    Apply tree transformations and convert between formats.
:doc:`treedraw <cli/treedraw>`
    Visualize (discontinuous) trees
:doc:`treesearch <cli/treesearch>`
    Search through treebanks with queries.
:doc:`grammar <cli/grammar>`
    Read off grammars from treebanks.
:doc:`parser <cli/parser>`
    Simple command line parser.
:doc:`gen <cli/gen>`
    Generate sentences from a PLCFRS.
demos:
    Show some demonstrations of formalisms encoded in LCFRS.

for additional instructions issue: ``discodop <command> --help``
or refer to the man page ``discodop-<command>``.

