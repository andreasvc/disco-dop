Discontinuous DOP
=================

.. image:: http://staff.science.uva.nl/~acranenb/disco-dop.png
   :align: right
   :alt: contrived discontinuous constituent for expository purposes.

The aim of this project is to parse discontinuous constituents in natural
language using Data-Oriented Parsing (DOP), with a focus on global world
domination. Concretely, we build a DOP model with a Linear Context-Free
Rewriting System (LCFRS) as the symbolic backbone.
The grammar is extracted from a treebank of sentences annotated with
(discontinuous) phrase-structure trees.

Background
----------
This work is partly described in the following publications:

- van Cranenburgh (2012). Efficient parsing with linear context-free rewriting
  systems. Proc. of EACL.
  http://staff.science.uva.nl/~acranenb/eacl2012corrected.pdf
- van Cranenburgh, Scha, Sangati (2011). Discontinuous Data-Oriented Parsing:
  A mildly context-sensitive all-fragments grammar. Proc. of SPMRL.
  http://www.aclweb.org/anthology/W/W11/W11-3805.pdf

Some references to implemented algorithms:

- parser, estimates: Maier & Kallmeyer (2010), Data-driven parsing with
  probabilistic linear context-free rewriting systems.
- data-oriented parsing (DOP):

  * Goodman (2002), Efficient parsing of DOP with PCFG-reductions
  * Sangati & Zuidema (2011), Accurate parsing with compact tree-substitution grammars: Double-DOP

- *k*-best list: Huang & Chiang (2005), Better *k*-best parsing
- optimal binarization: Gildea (2010), Optimal parsing strategies for linear
  context-free rewriting systems

Requirements
------------

- Python 2.7+/3   http://www.python.org (need headers, e.g. python-dev package)
- Cython 0.18+    http://www.cython.org
- GCC             http://gcc.gnu.org/
- Numpy           http://numpy.org/

For example, to install these dependencies and the latest stable release on
`Ubuntu <http://www.ubuntu.com>`_
using `pip <http://http://www.pip-installer.org>`_,
issue the following commands::

    sudo apt-get install build-essential python-dev python-numpy python-pip
    pip install --user discodop

To compile the latest development version on Ubuntu,
run the following sequence of commands::

    sudo apt-get install build-essential python-dev python-numpy python-pip
    pip install cython --user
    git clone --depth 1 git://github.com/andreasvc/disco-dop.git
    cd disco-dop
    python setup.py install --user

(the ``--user`` option means the packages will be installed to your home
directory which does not require root privileges).

To port the code to another compiler such as Visual C, replace the compiler
intrinsics in ``macros.h``, ``bit.pyx``, and ``bit.pxd`` to their equivalents
for the compiler in question. This mainly concerns operations to scan for bits
in integers, for which these compiler intrinsics provide the most efficient
implementation on a given processor.

Usage: parser
-------------
To run a full experiment from treebank to evaluation on a test set,
make a copy of the file ``sample.prm`` and edit its parameters.
These parameters can then be invoked by executing::

    discodop runexp filename.prm

This will create a new directory with the basename of the parameter file, i.e.,
``filename/`` in this case. This directory must not exist yet, to avoid
accidentally overwriting previous results. The directory will contain the
grammar rules and lexicon in a text format, as well as the parsing results and
the gold standard file in Negra's export format.

Corpora are expected to be in Negra's export format, or in the bracketed Penn
treebank format. Access to the
`Negra corpus <http://www.coli.uni-saarland.de/projects/sfb378/negra-corpus/>`_
can be requested for non-commercial purposes, while the
`Tiger corpus <http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/>`_
is freely available for download for research purposes.

Usage: tools
------------
Aside from the parser there are some standalone tools:

:``fragments``: Finds recurring or common fragments in one or more treebanks.
    It can be used with discontinuous as well as Penn-style bracketed treebanks.
:``treetransforms``: A command line interface to perform transformations on
     treebanks such as binarization.
:``grammar``: A command line interface to read off grammars from (binarized)
      treebanks.
:``parser``: A basic command line interface to the parser comparable to bitpar.
    Reads grammars from text files.
:``eval``: Discontinuous evaluation. Reports F-scores and other metrics.
    Accepts ``EVALB`` parameter files::

    ``discodop eval sample/gold.export sample/dop.export proper.prm``
:``demos``: Contains examples of various formalisms encoded in LCFRS grammars.
:``gen``: An experiment in generation with LCFRS.

All of these can be started with the ``discodop`` command.
For example::

    discodop fragments --help

... prints instructions for the fragment extractor.

Usage: web interfaces
---------------------
There are two web based tools in the ``web/`` directory. These require Flask to
be installed.

:``parse.py``: A web interface to the parser. Expects a series of grammars
    in subdirectories of ``web/grammars/``, each containing grammar files
    as produced by running ``discodop runexp``.
:``treesearch.py``: A web interface for searching trough treebanks. Expects
    one or more (non-discontinuous) treebanks with the ``.mrg`` extension in
    the directory ``web/corpus/`` (sample included).
:``treedraw.py``: A web interface for drawing discontinuous trees in various
    formats.

Acknowledgments
---------------

The Tree data structures in ``tree.py`` and the simple binarization algorithm in
``treetransforms.py`` was taken from `NLTK <http://www.nltk.org>`_.
The Zhang-Shasha tree-edit distance algorithm in ``treedist.py`` was taken from
https://github.com/timtadh/zhang-shasha
Elements of the PLCFRS parser and punctuation re-attachment are based on code from
`rparse <http://wolfgang-maier.de/rparse>`_. Various other bits from the
Stanford parser, Berkeley parser, Bubs parser, &c.

