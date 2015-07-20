
eval
----
Evaluation of (discontinuous) parse trees, following ``EVALB`` as much
as possible.

Usage: ``discodop eval <gold> <parses> [param] [options]``

where `gold`` and ``parses`` are files with parse trees, ``param`` is
an ``EVALB`` parameter file, and options may consist of:

--cutofflen=n    Overrides the sentence length cutoff of the parameter file.
--verbose        Print table with per sentence information.
--debug          Print debug information with per sentence bracketings etc.
--disconly       Only evaluate discontinuous bracketings (affects bracketing
                 scores: precision, recall, f-measure, exact match).

--goldfmt, --parsesfmt=<export|bracket|discbracket|tiger|alpino|dact>
                 Specify corpus format [default: export].

--fmt=<...>      Shorthand for setting both ``--goldfmt`` and ``--parsesfmt``.

--goldenc, --parsesenc=<utf-8|iso-8859-1|...>
                 Specify encoding [default: utf-8].

--la             Enable leaf-ancestor evaluation.
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


If the ``parses`` file contains function tags, these are evaluated with the
non-null metric of Blaheta & Charniak (2000), which scores function tags of
correctly parsed bracketings. Multiple tags on a constituent are scored
separately. We additionally consider function tags on preterminals; this does
not change the evaluation for the Penn treebank as it does not contain function
tags on preterminals. A more stringent metric is to combine phrasal & function
tags with the option ``--functions=add``, which incorporates function tags in
the bracketing scores.

The parameter file should be encoded in ``UTF-8`` and supports the following
options (in addition to those described in the ``README`` of ``EVALB``):

:DELETE_ROOT_PRETERMS:
                 if nonzero, ignore preterminals directly under the root in
                 gold trees for scoring purposes.
:DISC_ONLY:      if nonzero, only consider discontinuous bracketings
                 (affects precision, recall, f-measure, exact match).
:LA:             if nonzero, report leaf-ancestor scores [default: disabled].
:TED:
                 if nonzero, report tree-edit distance scores; disabled by
                 default as these are slow to compute.
:DEBUG:
                 :-1: only print summary table
                 :0: additionally, print category / tag breakdowns (default)
                   (after application of cutoff length).
                 :1: give per-sentence results (``--verbose``)
                 :2: give detailed information for each sentence (``--debug``)
:MAX_ERROR:
                 this value is ignored, no errors are tolerated.
                 the parameter is accepted to support usage of unmodified
                 EVALB parameter files.

