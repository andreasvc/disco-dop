=================
Discontinuous DOP
=================

.. image:: http://staff.science.uva.nl/~acranenb/disco-dop.png
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

.. contents:: Contents of this README:
   :local:

Features
========
General statistical parsing:

- grammar formalisms: PCFG, PLCFRS
- extract treebank grammar: trees decomposed into productions, relative
  frequencies as probabilities
- exact *k*-best list of derivations
- coarse-to-fine pruning: posterior pruning (PCFG only),
  *k*-best coarse-to-fine

DOP specific (parsing with tree fragments):

- implementations: Goodman's DOP reduction, Double-DOP.
- estimators: relative frequency estimate (RFE), equal weights estimate (EWE).
- objective functions: most probable parse (MPP),
  most probable derivation (MPD), most probable shortest derivation (MPSD),
  most likely tree with shortest derivation (SL-DOP).
- marginalization: n-best derivations, sampled derivations.

Installation
============

Requirements:

- Python 2.7+/3   http://www.python.org (need headers, e.g. python-dev package)
- Cython 0.18+    http://www.cython.org
- GCC             http://gcc.gnu.org/
- Numpy 1.5+      http://numpy.org/

For example, to install these dependencies and the latest stable release on
an `Ubuntu <http://www.ubuntu.com>`_ system
using `pip <http://http://www.pip-installer.org>`_,
issue the following commands::

    sudo apt-get install build-essential python-dev python-numpy python-pip
    pip install --user Cython
    pip install --user disco-dop

To compile the latest development version on Ubuntu,
run the following sequence of commands::

    sudo apt-get install build-essential python-dev python-numpy python-pip git
    pip install cython --user
    git clone --depth 1 git://github.com/andreasvc/disco-dop.git
    cd disco-dop
    python setup.py install --user

(the ``--user`` option means the packages will be installed to your home
directory which does not require root privileges).

If you do not run Linux, it is possible to run the code inside a virtual machine.
To do that, install `Virtualbox <https://www.virtualbox.org/wiki/Downloads>`_
and `Vagrant <http://docs.vagrantup.com/v2/installation/>`_,
and copy ``Vagrantfile`` from this repository to a new directory. Open a
command prompt (terminal) in this directory, and run the command
``vagrant up``. The virtual machine will boot and run a script to install the
above prerequisites automatically. The command ``vagrant ssh`` can then be used
to log in to the virtual machine (use ``vagrant halt`` to stop the virtual
machine).

Compilation requires the GCC compiler. To port the code to another compiler such
as Visual C, replace the compiler intrinsics in ``macros.h``, ``bit.pyx``, and
``bit.pxd`` with their equivalents for the compiler in question. This mainly
concerns operations to scan for bits in integers, for which these compiler
intrinsics provide the most efficient implementation on a given processor.

Usage
=====

Parser
------
To run an end-to-end experiment from grammar extraction to evaluation on a test
set, make a copy of the file ``sample.prm`` and edit its parameters.
These parameters can then be invoked by executing::

    discodop runexp filename.prm

This will create a new directory with the base name of the parameter file, i.e.,
``filename/`` in this case. This directory must not exist yet, to avoid
accidentally overwriting previous results. The directory will contain the
grammar rules and lexicon in a text format, as well as the parsing results and
the gold standard file in Negra's export format.

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

        discodop fragments wsj-02-21.mrg > wsjfragments.txt

    Specify the option ``--numproc n`` to use multiple processes, as with ``runexp``.

``eval``
    Discontinuous evaluation. Reports F-scores and other metrics.
    Accepts ``EVALB`` parameter files::

        discodop eval sample/gold.export sample/dop.export proper.prm

``treetransforms``
    A command line interface to perform transformations on
    treebanks such as binarization.

``grammar``
    A command line interface to read off grammars from (binarized)
    treebanks.

``treedraw``
    Visualize (discontinuous) trees. Command-line interface::

        discodop treedraw < negra-corpus.export | less -R

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

``treesearch.py``
    A web interface for searching trough treebanks. Expects
    one or more (non-discontinuous) treebanks with the ``.mrg`` extension in
    the directory ``web/corpus/`` (sample included). Depends on
    `tgrep2 <http://tedlab.mit.edu/~dr/Tgrep2/>`_ and
    `style <http://www.gnu.org/software/diction/diction.html>`_.

``treedraw.py``
    A web interface for drawing discontinuous trees in various
    formats.

See https://github.com/andreasvc/disco-dop/wiki for screenshots.

Documentation
=============
The API documentation can be perused at http://staff.science.uva.nl/~acranenb/discodop/

To generate a local copy install `Sphinx <http://sphinx-doc.org/>`_
and issue ``make html`` in the ``docs/`` directory; the result will be in
``_build/html``.

Acknowledgments
===============

The Tree data structures in ``tree.py`` and the simple binarization algorithm in
``treetransforms.py`` was taken from `NLTK <http://www.nltk.org>`_.
The Zhang-Shasha tree-edit distance algorithm in ``treedist.py`` was taken from
https://github.com/timtadh/zhang-shasha
Elements of the PLCFRS parser and punctuation re-attachment are based on code from
`rparse <http://wolfgang-maier.de/rparse>`_. Various other bits from the
Stanford parser, Berkeley parser, Bubs parser, &c.

References
==========
This work is partly described in the following publications:

- van Cranenburgh (2012). Efficient parsing with linear context-free rewriting
  systems. Proc. of EACL.
  http://staff.science.uva.nl/~acranenb/eacl2012corrected.pdf
- van Cranenburgh, Scha, Sangati (2011). Discontinuous Data-Oriented Parsing:
  A mildly context-sensitive all-fragments grammar. Proc. of SPMRL.
  http://www.aclweb.org/anthology/W/W11/W11-3805.pdf

