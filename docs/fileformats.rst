.. _fileformats:

File formats
============

Treebanks
---------
export
^^^^^^
Negra export format (v3).

For example::

    #BOS 0
    is  VB  --  --  500
    John    NP  --  --  0
    rich    JJ  --  --  500
    ?   ?   --  --  0
    #500    VP  --  --  0
    #EOS 0

An optional lemma field is supported. Secondary edges may or may not be
preserved but mostly ignored. The preamble listing the tag sets is ignored and
not reproduced when trees are written in this format.

This format is supported when input is read incrementally from
standard input with the ``treedraw`` and ``treetransforms`` commands.

Cf. http://www.coli.uni-saarland.de/projects/sfb378/negra-corpus/exformat3.ps


.. _bracket-format:

bracket
^^^^^^^
Penn-treebank style bracketed trees, one tree per line.

For example::

    (S (NP John) (VP (VB is) (JJ rich)) (. .))

This format is supported when input is read incrementally from
standard input with the ``treedraw`` and ``treetransforms`` commands.

Tree fragments can be represented by leaving out the children of
any non-terminal, e.g.::

    (VP (VB is) (JJ ))
    (S (NP John) (VP )) (. .))

This format does not support discontinuous constituents.

discbracket
^^^^^^^^^^^
A corpus format for discontinuous trees in bracket notation, where the
leaves are prefixed with indices indicating word order.
Each leaf must have at least one index; the indices form an unbroken range
starting from 0, with each index occurring exactly once.

For example::

    (S (VP (VB 0=is) (JJ 2=rich)) (NP 1=John) (? 3=?))
    (sentence: is John rich ?)

Note that the leaves are not in the same order as in the sentence. The leaves
must be sorted by the indices to restore the original sentence order.
There is one parse tree per line. Compared to Negra's export format, this
format lacks separate fields for morphology, lemmas, and functional edges.
On the other hand, it is close to the internal representation employed here, so
it can be read efficiently.

This format is supported when input is read incrementally from
standard input with the ``treedraw`` and ``treetransforms`` commands.

Tree fragments can be formed as with bracket trees, by leaving out terminals or whole subtrees::

    (S (VP (VB 0=is) (JJ 2=)) (NP 1=) (? 3=?))
    (VP (VB 0=is) (JJ 2=rich))

There is an extra case that should be handled, which is how to represent a
discontinuous frontier non-terminal. This requires expressing how the spans of
the discontinuous node relate to the other spans in the tree::

    (S (VP 0= 2=) (NP 1=) (? 3=?))

While the VP node does not dominate any terminals, if they were to be added,
they would end up before and after the NP node.


Read-only formats
^^^^^^^^^^^^^^^^^
:``tiger``: Tiger XML format.
    Cf. http://www.ims.uni-stuttgart.de/forschung/ressourcen/werkzeuge/TIGERSearch/doc/html/TigerXML.html
:``ftb``: FTB XML format (does not support discontinuous constituents).
    Cf. http://www.llf.cnrs.fr/Gens/Abeille/French-Treebank-fr.php
:``alpino``: Alpino XML format. One file per sentence. These two properties make it slow
    to process, so the recommended approach is to convert to another format
    (on the other hand, not all attributes are retained in the conversion).
    Cf. http://www.let.rug.nl/~vannoord/Lassy/alpino_ds.dtd
:``alpinocompact``: Alpino XML trees stored as Compact Corpus (dictzip).
    Cf. http://www.let.rug.nl/~vannoord/alp/Alpino/TreebankTools.html#_act_managing_compact_corpora

Write-only formats
^^^^^^^^^^^^^^^^^^
:``conll``: Convert to dependency structures in CoNLL-X tabular format.
   Relies on head identification. Labels based on function tags (if any).
   A discontinuous tree may produce a non-projective dependency structure.
   Cf. https://depparse.uvt.nl/DataFormat.html
:``mst``: Dependency format used by MST parser.
   Cf. https://github.com/travisbrown/mstparser#3a-input-data-format
:``tokens``: Tokens separated by spaces, one sentence per line.
:``wordpos``: Similar to ``tokens`` but of the form ``token/POS``.


.. _grammar-formats:

Grammars
--------
PCFG
^^^^
PCFG grammars are stored in bitpar format. From the bitpar manual:

    The grammar file contains one grammar rule per  line.  Each  grammar rule
    starts with its frequency [...] followed by the parent category (symbol on
    the left-hand side) and the child categories (symbols  on  the  right-hand
    side). The symbols are separated by whitespace. [...]

    The lexicon file contains one lexicon entry per line. Each  lexicon  entry
    starts  with  the  word [...] followed a sequence of part-of-speech
    tag + frequency pairs. The POS tag is preceded by a tab character
    and followed by a blank or tab character.

Cf. http://www.cis.uni-muenchen.de/~schmid/tools/BitPar/

PLCFRS
^^^^^^
The PLCFRS format is as follows. Rules are delimited by newlines.
Fields are separated by tabs. The fields are::

    LHS RHS1    [RHS2]  yield-function  weight

The yield function defines how the spans of the RHS nonterminals
are combined to form the spans of the LHS nonterminal. Components of the yield
function are comma-separated, 0 refers to a component of the first RHS
nonterminal, and 1 from the second.
Weights are treated as (pseudo)counts and normalized when the grammar is loaded.
The lexicon is defined in a separate file. Lines start with a single word,
followed by pairs of possible tags and their probabilities::

    WORD    TAG1    PROB1   [TAG2   PROB2 ...]

Example, rules file::

    S  NP  VP  010 1/2
    VP_2   VB  NP  0,1 2/3
    NP NN  0   1/4

lexicon file::

    is  VB  1/3
    John    NN 1/2
    rich    JJ 1/5

backtransform
^^^^^^^^^^^^^
Double-DOP grammars and other PTSGs employ a grammar in which internal nodes
are removed from fragments to obtain a more compact grammar. Fragments are
restored in derivations using a backtransform table with the original fragments
for each grammar rule.

The backtransform file contains one fragment per line, with the lines
corresponding to the lines of the grammar rule file. Frontier non-terminals
are indicated as ``{0}``, ``{1}``, etc.
The fragments which this backtransform is based on is also saved, with a
filename of the form ``.fragments.gz``.
To view the grammar rules together with the corresponding fragments, issue the
following command::

    $ paste <(zcat dop.rules.gz) <(zcat dop.fragments.gz)
    A       X       Y       01      1       (A (X 0) (Y 1)) 1
    A_2     X       Z       0,1     1       (A_2 (X 0) (Z 2))       2
    RIGHT   A_2     Y       010     1       (RIGHT (A_2 0 2) (Y 1)) 2
    S       S}<0>   Z@z     01      2/5     (S (RIGHT (A_2 (X 0) (Z 2)) (Y 1)))     x y z   2
    S       RIGHT   0       2/5     (S (RIGHT 0))   2
    S       WRONG   0       1/5     (S (WRONG 0))   1
    WRONG   A       Z       01      1       (WRONG (A 0) (Z 1))     1
    S}<0>   X@x     Y@y     01      1

alternate weights
^^^^^^^^^^^^^^^^^
DOP grammars can contain multiple probability models. The alternate models are
stored in a NumPy array::

    $ python
    >>> import numpy
    >>> probs = numpy.load('dop.probs.npz')
    >>> probs.keys()
    ['default', 'shortest', 'bon', 'ewe']
    >>> probs['shortest'][:10]
    array([ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5])

In this case, we see the model for shortest derivation parsing, where
every fragment is assigned a uniform weight of 0.5.
These weights are not normalized when the grammar is loaded.

Miscellaneous
-------------
head assignment rules
^^^^^^^^^^^^^^^^^^^^^
This file specifies a set of heuristic rules to pick for every constituent
one of its children as being the head of the constituent, based on
syntactic categories.

The file is case insensitive. Lines starting with ``%`` are treated as comments
and ignored. Each line specifies a rule of the form::

    CAT direction child1 child2...


This rule specifies how a head child is assigned for a constituent labeled as ``CAT``.
The second argument specifies whether the children of the constituent should
be considered starting from the left or from the right (corresponding to whether
a category is head-first head-final):

    :left: (or left-to-right) for each of the possible heads, try all children from left to right
    :right: (or right-to-left) for each of the possible heads, try all children from right to left
    :leftdis: go from left to right and try each possible head.
    :rightdis: go from right to left and try each possible head.
    :like: treat this label as if it were another label; e.g. 'TOP like ROOT'.

There may be multiple rules for a category, for example if they go in opposite
directions. The rules are applied in the order as they appear in the file.

The list of children may be empty; in that case the leftmost (or rightmost, in
the second case) child will be chosen as head.
If the list of possible children is non-empty, the children of the constituents
are iterated over for each possible child, and the first matching child is
picked as the head.

See also: http://www.cs.columbia.edu/~mcollins/papers/heads


.. _evalparam-format:

evaluation parameters
^^^^^^^^^^^^^^^^^^^^^
The format of this file is a superset of the parameters for EVALB,
cf. http://nlp.cs.nyu.edu/evalb/

The parameter file should be encoded in UTF-8 and supports the following
options in addition to those supported by EVALB:

  :DELETE_ROOT_PRETERMS:
                     if nonzero, ignore preterminals directly under the root in
                     gold trees for scoring purposes.

  :DISC_ONLY:
                     if nonzero, only consider discontinuous bracketings
                     (affects precision, recall, f-measure, exact match).

  :LA:               if nonzero, report leaf-ancestor scores [default: disabled].
  :TED:
                     if nonzero, report tree-edit distance scores; disabled by
                     default as these are slow to compute. NB: it is not clear
                     whether this score is applicable to discontinuous trees.

  :DEBUG:
                     :-1: only print summary table
                     :0:
                          additionally, print category / tag breakdowns (default)
                          (after application of cutoff length).

                     :1: give per-sentence results (``'--verbose'``)
                     :2: give detailed information for each sentence (``'--debug'``)

  :MAX_ERROR:
                     this value is ignored, no errors are tolerated.
                     the parameter is accepted to support usage of unmodified
                     EVALB parameter files.

parser parameters
^^^^^^^^^^^^^^^^^
See :doc:`the reference documentation on parser parameter files <../params>`.
