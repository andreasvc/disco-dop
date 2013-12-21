.. _params:

Parameters for specifying a parsing experiment (runexp)
=======================================================

A parsing experiment consists of a sequence of stages, and a set of global options:

.. code-block:: python

    stages=[
        stage1,
        stage2,
    ],
    key1=val1,
    key2=val2,

The parameters consist of a Python expression surrounded by an implicit 'dict(' and ')'.

Stages
------

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

    :'pcfg': CKY parser
    :'plcfrs': use the agenda-based PLCFRS parser
    :'pcfg-posterior': Compute inside-outside probabilities (does not produce
        parse trees, can only be used for pruning of next stage).
    :'pcfg-bitpar-nbest': Use external bitpar parser. Produces n-best list
        (up to n=1000) without producing a parse forest; works with
        non-binarized grammars. (experimental).
    :'pcfg-bitpar-forest': Use external bitpar parser (experimental).
    :'dop-rerank': Rerank parse trees from previous stage with DOP reduction (experimental).
:prune: whether to use previous chart to prune this stage
:split: split disc. nodes ``VP_2[101]`` as ``{ VP*[100], VP*[001] }``
:splitprune: treat ``VP_2[101]`` as ``{VP*[100], VP*[001]}`` for pruning
:markorigin: mark origin of split nodes: ``VP_2 => {VP*1, VP*2}``
:k: pruning parameter:

    :k > 1: no. of coarse pcfg derivations to prune with
    :k=0: filter only (only prune items that do not lead to a complete derivation)
    :0 < k < 1: posterior threshold for inside-outside probabilities
:neverblockre: do not prune nodes with label that match this regex
:getestimates: compute & store context-summary estimates
:useestimates: load & use estimates
:binarized: when using ``mode='pcfg-bitpar-nbest'``, this option can be set to
    ``False``, to disable the two auxiliary binarizations needed for Double-DOP.
:dop: enable DOP mode (DOP reduction / double DOP)
:packedgraph: use packed graph encoding for DOP reduction
:usedoubledop: when True, use Double-DOP instead of DOP reduction
:iterate: for Double-DOP, whether to add fragments of fragments
:complement: for Double-DOP, whether to include fragments which
    form the complement of the maximal recurring fragments extracted
:m: number of derivations to sample / enumerate.
:sample: sample `m` derivations from chart
:kbest: extract `m`-best derivations from chart
:estimator: DOP estimator. Choices:

    :dop1: relative frequencies.
    :ewe: equal weights estimate; relative frequencies with correction factor
        to remove bias for larger fragments; useful with DOP reduction.
    :bon: Bonnema estimator; another correction factor approach.
:objective: Objective function to choose DOP parse tree. Choices:

    :mpp: Most Probable Parse. Marginalizes over multiple derivations.
    :mpd: Most Probable Derivation.
    :shortest: Most Probable Shortest Derivation; i.e., shortest derivation
        (with minimal number of fragments), where ties are broken using
        probabilities specified by ``estimator``.
    :sl-dop: Simplicity-Likelihood. Simplest Tree from the n most Likely trees.
    :sl-dop-simple: An approximation which does not require parsing the sentence twice.
:sldop_n: When using sl-dop or sl-dop-simple,
    number of most likely parse trees to consider.

Global options
--------------

:stages: see above
:corpusfmt: The corpus format; choices:

    :export: Negra export format
    :bracket: Penn treebank style bracketed trees.
    :discbracket: Bracketed parse trees with numeric indices and words of
        sentence specified separately.
    :alpino: Alpino XML format
    :tiger: Tiger XML format
:corpusdir: root directory for corpora
:traincorpus: filename of training corpus.
:trainencoding: encoding of training corpus (defaults to 'utf-8')
:testcorpus: filename of test corpus (may be same as traincorpus, set
    `skiptrain` to True in that case).
:testencoding: encoding of test corpus (defaults to 'utf-8')
:testmaxwords: maximum sentence length to parse from test set
:trainmaxwords: maximum sentence length to base grammar on
:trainnumsents: number of sentence to use from training corpus
:testnumsents: number of sentences to parse
:skiptrain: when training & test corpus are from same file, start reading
    test set after training set sentences
:skip: number of (additional) sentences to skip before test corpus starts
:punct: one of ...

    :None: leave punctuation as is.
    :'move': move punctuation to appropriate constituents using heuristics.
    :'remove': eliminate punctuation.
    :'root': attach punctuation directly to root (as in original Negra/Tiger
        treebanks).
:functions: one of ...

    :None: leave syntactic labels as is.
    :'add': concatenate grammatical function to syntactic label,
        separated by a hypen: e.g., NP => NP-SBJ
    :'remove': strip away hyphen-separated grammatical function,
        e.g., NP-SBJ => NP
    :'replace': replace syntactic label with grammatical function,
        e.g., NP => SBJ
:morphology: one of ...

    :None: use POS tags as preterminals
    :'add': concatenate morphological information to POS tags,
        e.g., DET/sg.def
    :'replace': use morphological information as preterminal label
    :'between': add node with morphological information between
        POS tag and word, e.g., (DET (sg.def the))
:lemmas: one of ...

    :None: ignore lemmas
    :'between': insert lemma as node between POS tag and word.
:transformations: apply treebank transformations; see :func:`discodop.treebanktransforms.transform`
:postagging: POS tagging; see :mod:`discodop.lexicon`
:relationalrealizational: apply RR-transform; see :func:`discodop.treebanktransforms.rrtransform`
:headrules: file with rules for finding heads of constituents
:bintype: Binarization; choices:

    :binarize: basic binarization (recommended).
    :optimal: binarization which optimizes for lowest fan-out or parsing complexity.
    :optimalhead: like ``optimal``, but only considers head-driven binarizations.
:factor: 'left' or 'right'. The direction of binarization when using ``binarize``.
:revmarkov: whether to reverse the horizontal markovization context when using ``binarize``.
:v: vertical markovization context; default 1; 2 means 1 extra level of parent annotation.
:h: horizontal markovization context
:pospa: when v > 1, whether to add parent annotation to POS tags?
:markhead: whether to prepend head to siblings labels
:leftmostunary: whether to start binarization with unary node
:rightmostunary: whether to end binarization with unary node
:tailmarker: with headrules, head is last node and can be marked
:fanout_marks_before_bin: whether to add fanout markers before binarization
:evalparam: EVALB-style parameter file to use for reporting F-scores
:quiet: default False; True: no per sentence results.
:reallyquiet: default False; True: no summary reports either.
:numproc: default 1; increase to use multiple CPUs; None: use all CPUs.

