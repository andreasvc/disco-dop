Discontinuous Data-Oriented Parsing: disco-dop
==============================================

.. image:: images/disco-dop.png
   :align: right
   :alt: contrived discontinuous constituent for expository purposes.

The aim of this project is to parse discontinuous constituents in natural
language using Data-Oriented Parsing (DOP), with a focus on global world
domination. The grammar is extracted from a treebank of sentences annotated
with (discontinuous) phrase-structure trees. Concretely, this project provides
a statistical constituency parser with support for discontinuous constituents
and Data-Oriented Parsing. Discontinuous constituents are supported through the
grammar formalism Linear Context-Free Rewriting System (LCFRS), which is a
generalization of Probabilistic Context-Free Grammar (PCFG). Data-Oriented
Parsing allows re-use of arbitrary-sized fragments from previously seen
sentences using Tree-Substitution Grammar (TSG).

.. toctree::
   :maxdepth: 1

   discodop
   params
   api

Overview
========

Parser
------
To run an end-to-end experiment from grammar extraction to evaluation on a test
set, make a copy of the file ``sample.prm`` and edit its parameters. For example:

.. code-block:: python

    stages=[
      dict(name='pcfg', mode='pcfg',
        split=True, markorigin=True),
      dict(name='plcfrs', mode='plcfrs',
        prune=True, splitprune=True, k=1000),
      dict(name='dop', mode='plcfrs',
        prune=True, k=50, m=1000,
        dop=True, usedoubledop=False,
        estimator="dop1", objective = "mpp")
    ],
    corpusdir='.',
    traincorpus='alpinosample.export', trainencoding='utf-8',
    testcorpus='alpinosample.export', testencoding='utf-8',
    testmaxwords=100, trainmaxwords=100,
    trainnumsents=3, testnumsents=3, skiptrain=False,
    postagging=dict(
        method="unknownword", model="4",
        unknownthreshold=1, openclassthreshold=50,
        simplelexsmooth=True,
    ),
    bintype="binarize",
    factor="right",
    h=1, v=1,
    numproc=1,

See the documentation on the available :ref:`parameters <params>`.
These parameters can be invoked by executing::

    discodop runexp filename.prm

This will create a new directory with the base name of the parameter file, i.e.,
``filename/`` in this case. This directory must not exist yet, to avoid
accidentally overwriting previous results. The directory will contain the
grammar rules and lexicon in a text format, as well as the parsing results and
the gold standard file in Negra's export format.

.. image:: images/runexp.png
   :alt: screenshot of runexp showing a parse tree

Note that there is an option to utilize multiple processor cores by launching a
specific number of processes. This greatly speeds up parsing, but note that for
a nontrivial DOP grammar, each process may require anywhere from 4GB to 16GB.

Corpora can be read in Negra's export format, or in the bracketed Penn
treebank format. Access to the
`Negra corpus <http://www.coli.uni-saarland.de/projects/sfb378/negra-corpus/>`_
can be requested for non-commercial purposes, while the
`Tiger corpus <http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/>`_
is freely available for download for research purposes.

Tools
-----
Aside from the parser there are some standalone tools, invoked as ``discodop <cmd>``:

``fragments``
    Finds recurring or common fragments in one or more treebanks.
    It can be used with discontinuous as well as Penn-style bracketed treebanks.
    Example::

    $ discodop fragments wsj-02-21.mrg > wsjfragments.txt

    Specify the option ``--numproc n`` to use multiple processes, as with ``runexp``.

``eval``
    Discontinuous evaluation. Reports F-scores and other metrics.
    Accepts ``EVALB`` parameter files:

    ``$ discodop eval sample/gold.export sample/dop.export proper.prm``

``treetransforms``
    A command line interface to perform transformations on
    treebanks such as binarization.

``grammar``
    A command line interface to read off grammars from (binarized)
    treebanks.

``treedraw``
    Visualize (discontinuous) trees. Command-line interface:

    ``$ discodop treedraw < negra-corpus.export | less -R``

``parser``
    A basic command line interface to the parser comparable to bitpar.
    Reads grammars from text files.

``demos``
    Contains examples of various formalisms encoded in LCFRS grammars.

``gen``
    An experiment in generation with LCFRS.

For instructions, pass the ``--help`` option to a command.

Web interfaces
--------------
There are three web based tools in the ``web/`` directory. These require Flask to
be installed.

``parse.py``
    A web interface to the parser. Expects a series of grammars
    in subdirectories of ``web/grammars/``, each containing grammar files
    as produced by running ``discodop runexp``.
    `Download grammars <http://staff.science.uva.nl/~acranenb/grammars/>`_
    for English, German, and Dutch, as used in the 2013 IWPT paper.

``treesearch.py``
    A web interface for searching through treebanks. Expects
    one or more treebanks with the ``.mrg`` or ``.dact`` extension in the
    directory ``web/corpus/`` (sample included). Depends on
    `tgrep2 <http://tedlab.mit.edu/~dr/Tgrep2/>`_,
    `alpinocorpus <https://github.com/rug-compling/alpinocorpus-python>`_, and
    `style <http://www.gnu.org/software/diction/diction.html>`_.

``treedraw.py``
    A web interface for drawing discontinuous trees in various
    formats.

.. image:: images/treesearch1.png
   :alt: screenshot of treesearch showing counts of pattern

.. image:: images/treesearch2.png
   :alt: screenshot of treesearch showing bar plot of pattern
