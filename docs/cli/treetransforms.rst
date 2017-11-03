
treetransforms
--------------
Treebank binarization and conversion

Usage: ``discodop treetransforms [input [output]] [options]``

where ``input`` and ``output`` are treebanks; standard in/output is used if not given.

Main transformation options
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following transforms are applied in the order given on the command line.

--introducepreterminals
                Add preterminals to terminals without a dedicated preterminal;
                labels will be of the form ``PARENT/terminal``.

--transforms=<NAME1,NAME2...>
                Apply specific treebank transforms; available presets:
                ``negra, wsj, alpino, green2013ftb, km2003wsj,
                km2003simple, fraser2013tiger, lassy, lassy-func``
                For details cf. source of :mod:`discodop.treebanktransforms` module.

--reversetransforms=<NAME1,NAME2,...>
                Undo specified transforms; specify transforms in original order.

--binarize
                Related options: [-h x] [-v x] [--factor=<left|right>] [...]
                Markovized binarization; also see --headrules and other options below.

--optimalbinarize
                Related options: [-h x] [-v x]
                Binarization that minimizes LCFRS fan-out/complexity.

--unbinarize    Restore original n-ary trees.

--splitdisc
                Related options: [--markorigin]
                Split discontinuous nodes into several continuous nodes.

--mergedisc     Reverse the node splitting operation.

Other options
^^^^^^^^^^^^^
--inputfmt=<export|bracket|discbracket|tiger|alpino>
                Input treebank format [default: export].

--outputfmt=<export|bracket|discbracket|conll|mst|tokens|wordpos>
                Output treebank format [default: export].
                Selecting the formats ``conll`` or ``mst`` results in an
                unlabeled dependency conversion and requires the use of
                heuristic head rules (``--headrules``), to ensure that all
                constituents have a child marked as head. A command line
                interface to perform transformations on treebanks such as
                binarization.

--fmt=x         Shortcut to specify both input and output format.

--inputenc, --outputenc, --enc=<utf-8|iso-8859-1|...>
                Treebank encoding [default: utf-8].

--slice=<n:m>   select a range of sentences from input starting with *n*,
                up to but not including *m*; as in Python, *n* or *m* can be left
                out or negative, and the first index is 0.

--renumber      Replace sentence IDs with numbers starting from 1,
                padded with 8 spaces.

--sentid        With 'tokens' or 'wordpos' output format, prefix lines with identifiers of the form ``ID|``.
--maxlen=n      only select sentences with up to *n* tokens.
--punct=x       :'remove': remove any punctuation.
                :'move': re-attach punctuation to nearest constituent
                      to minimize discontinuity.
                :'restore': attach punctuation under root node.
--functions=x   :'leave': (default): leave syntactic labels as is,
                :'remove': strip away hyphen-separated function labels
                :'add': concatenate syntactic categories with functions,
                :'replace': replace syntactic labels w/grammatical functions.
--morphology=x  :'no' (default): use POS tags as preterminals
                :'add': concatenate morphological information to POS tags,
                    e.g., ``DET/sg.def``
                :'replace': use morphological information as preterminal label
                :'between': insert node with morphological information between
                    POS tag and word, e.g., ``(DET (sg.def the))``
--lemmas=x      :'no' (default): do not use lemmas.
                :'add': concatenate lemmas to terminals, e.g., word/lemma
                :'replace': use lemma instead of terminals
                :'between': insert node with lemma between POS tag and word,
                    e.g., ``(NN (man men))``
--ensureroot=x  add root node labeled ``x`` to trees if not already present.
--removeempty   remove empty / ``-NONE-`` terminals.

--factor=<left|right>
                specify left- or right-factored binarization [default: right].

-h n            horizontal markovization. default: infinite (all siblings)
-v n            vertical markovization. default: 1 (immediate parent only)
--headrules=x   turn on head finding; turns on head-outward binarization.
                reads rules from file ``x`` (e.g., "negra.headrules").
--markhead      include label of the head child in all auxiliary labels
                of binarization.
--direction     mark direction when using head-outward binarization.
--labelfun=x    ``x`` is a Python lambda function that takes a node and returns
                a label to be used for markovization purposes. For example,
                to get labels without state splits, pass this function:
                ``'lambda n: n.label.split("^")[0]'``
--leftunary     make initial / final productions of binarized constituents
--rightunary    ... unary productions.
--tailmarker    mark rightmost child (the head if headrules are applied), to
                avoid cyclic rules when ``--leftunary`` and ``--rightunary``
                are used.

Example
^^^^^^^
Binarize a treebank::

      $ discodop treetransforms --binarize --fmt=bracket treebankExample.mrg /tmp/bintrees

