.. _fileformats:

File formats
============

Corpora
-------
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

An optional lemma field is supported. Secondary edges are ignored.
The preamble listing the tag sets is ignored and not reproduced
when trees are written in this format.

This format is supported when input is read incrementally from
standard input with the ``treedraw`` and ``treetransforms`` commands.

Cf. http://www.coli.uni-saarland.de/projects/sfb378/negra-corpus/exformat3.ps

bracket
^^^^^^^
Penn treebank style bracketed trees, one tree per line.

For example::

    (S (NP John) (VP (VB is) (JJ rich)) (. .))

This format is supported when input is read incrementally from
standard input with the ``treedraw`` and ``treetransforms`` commands.

discbracket
^^^^^^^^^^^
A corpus format for discontinuous trees in bracket notation, where the
leaves are indices pointing to words in a separately specified sentence.

For example::

    (S (NP 1) (VP (VB 0) (JJ 2)) (? 3)) is John rich ?

Note that the tree and the sentence are separated by a tab, while the words
in the sentence are separated by spaces. There is one tree and sentence
per line. Compared to Negra's export format, this format lacks morphology,
lemmas and functional edges. On the other hand, it is very close to the
internal representation employed here, so it can be read efficiently.

This format is supported when input is read incrementally from
standard input with the ``treedraw`` and ``treetransforms`` commands.

Read-only formats
^^^^^^^^^^^^^^^^^
:``alpino``:
    Alpino XML format. One file per sentence.
    The hierarchical tree structure is mirrored in the XML
    structure, which makes it possible to query trees in this
    format with XPath (as opposed to TigerXML which maintains
    the tabular structure of the Negra export format).

    Cf. http://www.let.rug.nl/~vannoord/Lassy/alpino_ds.dtd
:``dact``:
    Alpino XML trees in an XML database as used by Dact.
    Cf. http://rug-compling.github.io/dact/
:``tiger``: Tiger XML format.
    Cf. http://www.ims.uni-stuttgart.de/forschung/ressourcen/werkzeuge/TIGERSearch/doc/html/TigerXML.html

write-only formats
^^^^^^^^^^^^^^^^^^
:``connl``, ``mst``: unlabeled dependencies; relies on head identfication.
:``tokens``: one sentence per line, tokens separated by spaces.
:``wordpos``: similar to ``tokens`` but of the form ``token/POS``.

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
nonterminal, and 1 from the second. Weights are expressed as rational
fractions.
The lexicon is defined in a separate file. Lines start with a single word,
followed by pairs of possible tags and their probabilities::

    WORD    TAG1    PROB1   [TAG2   PROB2 ...]

Example::

    rules:   S  NP  VP  010 1/2
             VP_2   VB  NP  0,1 2/3
             NP NN  0   1/4
    lexicon: Haus   NN  3/10    JJ  1/9

backtransform
^^^^^^^^^^^^^
Double-DOP grammars and other PTSGs employ a grammar in which internal nodes
are removed from fragments to obtain a more compact grammar. Fragments are
restored in derivations using a backtransform table with the original fragments
for each grammar rule.

The backtransform file contains one fragment per line, with the lines
corresponding to the lines of the grammar rule file. Frontier non-terminals
are indicated as ``{0}``, ``{1}``, etc. To view the grammar rules together
with the corresponding fragments, issue the following command::

    $ paste <(zcat dop.rules.gz) <(zcat dop.backtransform.gz)
    NP^<NP> NP^<NP> NNS@gains       01      1/267481        (NP^<NP> {0} {1})
    NP^<NP> NP^<NP>}<592850>        SBAR^<NP>       01      3/534962        (NP^<NP> (NP^<NP> {0} (NP|<NNS;NN>^<NP> {1} {2})) {3})
    NP^<NP> NNP@Eugene      NNP     01      3/534962        (NP^<NP> {0} {1})
    NP^<NP> NP^<NP>}<202929>        NN@agency       01      1/267481        (NP^<NP> {0} (NP|<NN;JJ>^<NP> {1} (NP|<NN;NN>^<NP> {2} {3})))

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

