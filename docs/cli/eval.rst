
eval
----
Evaluation of (discontinuous) parse trees, following ``EVALB`` as much
as possible.

Usage: ``discodop eval <gold> <parses> [param] [options]``

where ``gold`` and ``parses`` are files with parse trees, ``param`` is
an ``EVALB`` parameter file.

Options
^^^^^^^
--cutofflen=n    Overrides the sentence length cutoff of the parameter file.
--verbose        Print table with per sentence information.
--debug          Print debug information with per sentence bracketings etc.
--disconly       Only evaluate discontinuous bracketings (affects bracketing
                 scores: precision, recall, f-measure, exact match).

--goldfmt, --parsesfmt=<export|bracket|discbracket|tiger|alpino>
                 Specify corpus format [default: export].

--fmt=<...>      Shorthand for setting both ``--goldfmt`` and ``--parsesfmt``.

--goldenc, --parsesenc=<utf-8|iso-8859-1|...>
                 Specify encoding [default: utf-8].

--la             Enable leaf-ancestor evaluation.
--ted            Enable tree-edit distance evaluation.
                 NB: it is not clear whether this score is applicable to
                 discontinuous trees.
--headrules=x    Specify file with heuristic rules for head assignment of
                 constituents that do not already have a child marked as head.
                 This enables dependency evaluation. NB: this evaluation is
                 affected by the quality of the head markings and heuristics.

--functions=x    'remove'=default: strip functions off labels,
                 'leave': leave syntactic labels as is,
                 'add': evaluate both syntactic categories and functions,
                 'replace': only evaluate grammatical functions.

--morphology=x   'no'=default: only evaluate POS tags,
                 'add': concatenate morphology tags to POS tags,
                 'replace': replace POS tags with morphology tags,
                 'between': add morphological node between POS tag and word.


Function tags
^^^^^^^^^^^^^
If the ``parses`` file contains function tags, these are evaluated with the
non-null metric of Blaheta & Charniak (2000), which scores function tags of
correctly parsed bracketings. Multiple tags on a constituent are scored
separately. We additionally consider function tags on preterminals; this does
not change the evaluation for the Penn treebank as it does not contain function
tags on preterminals. A more stringent metric is to combine phrasal & function
tags with the option ``--functions=add``, which incorporates function tags in
the bracketing scores.

Parameter file
^^^^^^^^^^^^^^
See the :ref:`reference documentation on evaluation parameter files <evalparam-format>`.

The commonly used parameter file for Penn-treebank parsing is ``COLLINS.prm``, distributed as part of ``EVALB``.
The file ``proper.prm`` in the code repository is a version adapted for discontinuous parsing.

Examples
^^^^^^^^
Discontinuous parsing::

    $ discodop eval negra-parses.export negra-dev.export proper.prm

Continuous parsing::

    $ discodop eval wsj-24.mrg parses.mrg COLLINS.prm --fmt=bracket
