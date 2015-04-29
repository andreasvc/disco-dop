Parser parameters
=================

A parser is defined by a sequence of stages, and a set of
global options:

.. code-block:: python

    stages=[
        stage1,
        stage2,
    ],
    corpusfmt='...',
    traincorpus=dict(...),
    testcorpus=dict(...),
    binarization=dict(...),
    key1=val1,
    key2=val2,

The parameters consist of a Python expression surrounded by an implicit
``'dict('`` and ``')'``. Note that every ``key=value`` is followed by a comma.

Corpora
-------

:corpusfmt: The corpus format; choices:

    :``'export'``: Negra export format
    :``'bracket'``: Penn treebank style bracketed trees.
    :``'discbracket'``: Bracketed parse trees with numeric indices and words of
        sentence specified separately.
    :``'alpino'``: Alpino XML format
    :``'tiger'``: Tiger XML format
:traincorpus: a dictionary with the following keys:

    :path: filename of training corpus; may include wildcards / globbing
        characters ``\*`` and ``?``.
    :encoding: encoding of training corpus (defaults to ``'utf-8'``)
    :maxwords: maximum sentence length to base grammar on
    :numsents: number of sentence to use from training corpus
:testcorpus: a dictionary with the following keys:

    :path: filename of test corpus (may be same as traincorpus, set
        ``skiptrain`` to ``True`` in that case).
    :encoding: encoding of test corpus (defaults to ``'utf-8'``)
    :maxwords: maximum sentence length to parse from test set
    :numsents: number of sentences to parse
    :skiptrain: when training & test corpus are from same file, start reading
        test set after training set sentences
    :skip: number of (additional) sentences to skip before test corpus starts

Binarization
------------
:binarization: a dictionary with the following keys:

    :method: Binarization method; choices:

        :``None``: Treebank is already binarized.
        :``'default'``: basic binarization (recommended).
        :``'optimal'``: binarization which optimizes for lowest fan-out or
            parsing complexity.
        :``'optimalhead'``: like ``optimal``, but only considers head-driven
            binarizations.
    :factor: ``'left'`` or ``'right'``. The direction of binarization when
        using ``default``.
    :headrules: file with rules for finding heads of constituents
    :markhead: whether to prepend head to siblings labels
    :v: vertical markovization context; default 1; 2 means 1 extra level of
        parent annotation.
    :h: horizontal markovization context
    :revh: horizontal markovization context preceding child being generated
    :leftmostunary: whether to start binarization with unary node
    :rightmostunary: whether to end binarization with unary node
    :tailmarker: with headrules, head is last node and can be marked
    :markovthreshold: reduce horizontal markovization threshold of auxiliary
        labels with a frequency lower than this threshold.
    :fanout_marks_before_bin: whether to add fanout markers before binarization
    :labelfun: specify a function from nodes to labels; can be used to change
        how labels appear in markovization, e.g., to strip of annotations.


Stages
------

Through the use of stages it is possible to run multiple parsers on the
same test set, or to exploit coarse-to-fine pruning.

A stage has the form:

.. code-block:: python

    dict(
        key1=val1,
        key2=val2,
        ...
    )

Where the keys and values are:

:name: identifier, used for filenames
:mode: The type of parser to use

    :``'pcfg'``: CKY parser
    :``'plcfrs'``: use the agenda-based PLCFRS parser
    :``'pcfg-bitpar-nbest'``: Use external bitpar parser. Produces n-best list
        (up to n=1000) without producing a parse forest; works with
        non-binarized grammars (experimental).
    :``'pcfg-bitpar-forest'``: Use external bitpar parser (experimental).
    :``'dop-rerank'``: Rerank parse trees from previous stage with DOP
        reduction (experimental).
:prune: specify the name of a previous stage to enable coarse-to-fine pruning.
:split: split disc. nodes ``VP_2[101]`` as ``{ VP*[100], VP*[001] }``
:splitprune: treat ``VP_2[101]`` as ``{VP*[100], VP*[001]}`` for pruning
:markorigin: mark origin of split nodes: ``VP_2 => {VP*1, VP*2}``
:k: pruning parameter:

    :k=0: filter only (only prune items that do not lead to a complete
        derivation)
    :0 < k < 1: posterior threshold for inside-outside probabilities
    :k > 1: no. of coarse pcfg derivations to prune with
:kbest: extract *m*-best derivations from chart
:sample: sample *m* derivations from chart
:m: number of derivations to sample / enumerate.
:binarized: when using ``mode='pcfg-bitpar-nbest'``, this option can be set to
    ``False``, to disable the two auxiliary binarizations needed for
    Double-DOP. This enables bitpar to do the binarization internally, which is
    more efficient.
:dop: enable DOP mode:

    :``None``: Extract treebank grammar
    :``'reduction'``: DOP reduction (Goodman 1996, 2003)
    :``'doubledop'``: Double DOP (Sangti & Zuidema 2011)
    :``'dop1'``: DOP1 (Bod 1992)
:estimator: DOP estimator. Choices:

    :``'rfe'``: relative frequencies.
    :``'ewe'``: equal weights estimate; relative frequencies with correction
        factor to remove bias for larger fragments; useful with DOP reduction.
    :``'bon'``: Bonnema estimator; another correction factor approach.
:objective: Objective function to choose DOP parse tree. Choices:

    :``'mpp'``: Most Probable Parse. Marginalizes over multiple derivations.
    :``'mpd'``: Most Probable Derivation.
    :``'mcc'``:
        Maximum Constituents Parse (Goodman 1996);
        approximation as in Sangati & Zuidema (2011); experimental.
    :``'shortest'``: Most Probable Shortest Derivation;
        i.e., shortest derivation (with minimal number of fragments), where
        ties are broken using probabilities specified by ``estimator``.
    :``'sl-dop'``: Simplicity-Likelihood. Simplest Tree from
        the *n* most Likely trees.
    :``'sl-dop-simple'``: An approximation which does not require parsing the
        sentence twice.
:sldop_n: When using sl-dop or sl-dop-simple,
    number of most likely parse trees to consider.
:maxdepth: with ``'dop1'``, the maximum depth of fragments to extract;
           with ``'doubledop'``, likewise but applying to the
           non-recurring/non-maximal fragments extracted to augment the set of
           recurring fragments.
:maxfrontier: with ``'dop1'``, the maximum number of frontier non-terminals in
              extracted fragments; with ``'doubledop'``, likewise but applying
              to the non-recurring/non-maximal fragments extracted to augment
              the set of recurring fragments.
:collapse: apply a multilevel coarse-to-fine preset. values are of the form
           ``('treebank', level)``; e.g., ``('ptb', 0)`` for the coarsest level
           of the Penn treebank. For the presets,
           see source of :py:data:`discodop.treebanktransforms.MAPPINGS`.
           Include a stage for each of the collapse-levels in ascending
           order (0, 1, and 2 in the current presets), and then add a stage
           where labels are not collapsed.
:packedgraph: use packed graph encoding for DOP reduction
:iterate: for Double-DOP, whether to add fragments of fragments
:complement: for Double-DOP, whether to include fragments which
    form the complement of the maximal recurring fragments extracted
:neverblockre: do not prune nodes with label that match this regex
:estimates: compute, store & use context-summary (outside) estimates
:beam_beta: beam pruning factor, between 0 and 1; 1 to disable.
    if enabled, new constituents must have a larger probability
    than the probability of the best constituent in a cell multiplied by this
    factor; i.e., a smaller value implies less pruning.
    Suggested value: ``1e-4``.
:beam_delta: if beam pruning is enabled, only apply it to spans up to this
    length.


Other options
--------------

:evalparam: EVALB-style parameter file to use for reporting F-scores
:postagging: To disable POS tagging and use the gold POS tags from the
    test set, set this to ``None``.
    Otherwise, pass a dictionary with the keys below; for details,
    see :py:mod:`discodop.lexicon`

    :method: one of:

        :``'unknownword'``: incorporate unknown word model in grammar
        :``'stanford'``: use external Stanford tagger
        :``'treetagger'``: use external tagger ``'treetagger'``
        :``'frog'``: use external tagger 'frog' for Dutch; produces CGN tags,
            use morphology='replace'.
    :model:

        :with 'unknownword', one of:
            :``'4'``: Stanford model 4; language agnostic
            :``'6'``: Stanford model 6, for the English Penn treebank
            :``'base'``: Stanford 'base' model; language agnostic
            :``'ftb'``: Stanford model 2 for French treebank
        :with external taggers: filename of tagger model (not applicable to
            'frog')
    :retag: if ``True``, re-tag the training corpus using the external tagger.
    :unknownthreshold: use probabilities of words that occur this number of
        times or less for unknown words
    :openclassthreshold: add unseen tags for known words when tag rewrites
        at least this number of words. 0 to disable.
    :simplelexsmooth: enable/disable sophisticated smoothing (untested)
:punct: one of ...

    :``None``: leave punctuation as is.
    :``'move'``: move punctuation to appropriate constituents using heuristics.
    :``'moveall'``: same as 'move', but moves all preterminals under root,
        instead of only recognized punctuation.
    :``'prune'``: prune away leading & ending quotes & periods, then move.
    :``'remove'``: eliminate punctuation.
    :``'root'``: attach punctuation directly to root (as in original
        Negra/Tiger treebanks).
:functions: one of ...

    :``None``: leave syntactic labels as is.
    :``'add'``: concatenate grammatical function to syntactic label,
        separated by a hypen: e.g., NP => NP-SBJ
    :``'remove'``: strip away hyphen-separated grammatical function,
        e.g., NP-SBJ => NP
    :``'replace'``: replace syntactic label with grammatical function,
        e.g., NP => SBJ
:morphology: one of ...

    :``None``: use POS tags as preterminals
    :``'add'``: concatenate morphological information to POS tags,
        e.g., DET/sg.def
    :``'replace'``: use morphological information as preterminal label
    :``'between'``: add node with morphological information between
        POS tag and word, e.g., (DET (sg.def the))
:lemmas: one of ...

    :``None``: ignore lemmas
    :``'between'``: insert lemma as node between POS tag and word.
:removeempty: ``True`` or ``False``; whether to remove empty terminals from
    train, test sets.
:ensureroot: Ensure every tree has a root node with this label
:transformations: apply treebank transformations;
    see source of :py:func:`discodop.treebanktransforms.transform`
:relationalrealizational: apply RR-transform;
    see :py:func:`discodop.treebanktransforms.rrtransform`
:verbosity: control the amount of output to console;
    a logfile ``output.log`` is also kept with a fixed log level of 2.

    :0: silent
    :1: summary report
    :2: per sentence results
    :3: dump derivations/parse trees
    :4: dump chart

:numproc: default 1; increase to use multiple CPUs; ``None``: use all CPUs.

