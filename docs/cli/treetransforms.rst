
treetransforms
--------------
Treebank binarization and conversion

Usage: ``discodop treetransforms <action> [input [output]] [options]``

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
                Selecting the formats ``conll`` or ``mst`` results in an
                unlabeled dependency conversion and requires the use of
                heuristic head rules (``--headrules``), to ensure that all
                constituents have a child marked as head. A command line
                interface to perform transformations on treebanks such as
                binarization.

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
--removeempty   remove empty / -NONE- terminals.

--factor=(left|right)
                specify left- or right-factored binarization [default: right].

-h n            horizontal markovization. default: infinite (all siblings)
-v n            vertical markovization. default: 1 (immediate parent only)
--headrules=x   turn on head finding; affects binarization.
                reads rules from file ``x`` (e.g., "negra.headrules").
--markhead      include label of the head child in all auxiliary labels
                of binarization.
--leftunary     make initial / final productions of binarized constituents
--rightunary    ... unary productions.
--tailmarker    mark rightmost child (the head if headrules are applied), to
                avoid cyclic rules when ``--leftunary`` and ``--rightunary``
                are used.
--transforms=x  specify names of tree transformations to apply; for possible
                names, cf. :mod:`discodop.treebanktransforms` module.
--reverse       reverse the transformations given by ``--transform``

