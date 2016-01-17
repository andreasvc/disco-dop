=================
Discontinuous DOP
=================

.. image:: docs/images/disco-dop.png
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
- coarse-to-fine pruning: posterior threshold,
  *k*-best coarse-to-fine

DOP specific (parsing with tree fragments):

- implementations: Goodman's DOP reduction, Double-DOP, DOP1.
- estimators: relative frequency estimate (RFE), equal weights estimate (EWE).
- objective functions: most probable parse (MPP),
  most probable derivation (MPD), most probable shortest derivation (MPSD),
  most likely tree with shortest derivation (SL-DOP).
- marginalization: n-best derivations, sampled derivations.

.. image:: docs/images/runexp.png
   :alt: screenshot of parse tree produced by parser

Installation
============

Requirements:

- Python 3.3+     http://www.python.org (headers required, e.g. python3-dev package)
- Cython 0.21+    http://www.cython.org
- Numpy 1.5+      http://numpy.org/

Python 2.7 is supported, but Python 3 is recommended.

Debian, Ubuntu based systems
----------------------------
To compile the latest development version on an `Ubuntu <http://www.ubuntu.com>`_ system,
run the following sequence of commands::

    sudo apt-get install build-essential python3-dev python3-numpy python3-pip git
    git clone --depth 1 git://github.com/andreasvc/disco-dop.git
    cd disco-dop
    pip3 install --user -r requirements.txt
    make install

The ``--user`` option means the packages will be installed to your home
directory which does not require root privileges. Make sure that
``~/.local/bin`` directory is in your PATH, or add it as follows::

    echo export PATH=$HOME/.local/bin:$PATH >> ~/.bashrc

Other Linux systems
-------------------
This assumes no root access, but assumes that ``gcc`` is installed.

Set environment variables so that software can be installed to the home directory
(replace with equivalent for your shell if you do not use bash)::

    mkdir -p ~/.local
    echo export PATH=$HOME/.local/bin:$PATH >> ~/.bashrc
    echo export LD_LIBRARY_PATH=$HOME/.local/lib:/usr/lib64:/usr/lib >>~/.bashrc
    echo export PYTHONIOENCODING="utf-8" >>~/.bashrc

After this, re-login or restart the shell to activate these settings.
Install Python 3.4 from source, if not installed already.
Python may require some libraries such as ``zlib`` and ``readline``;
installation steps are similar to the ones below::

    wget http://www.python.org/ftp/python/3.5.1/Python-3.5.1.tgz
    tar -xzf Python-*.tgz
    cd Python-*
    ./configure --prefix=$HOME/.local --enable-shared
    make install && cd ..

Check by running ``python3`` that version 3.5.1 was installed successfully and
is the default.

Install the latest development version of discodop::

    wget https://github.com/andreasvc/disco-dop/archive/master.zip
    unzip disco-dop-master.zip
    cd disco-dop-master
    pip install --user -r requirements.txt
    make install

Mac OS X
--------
- Install `Xcode <https://developer.apple.com/>`_ and `Homebrew <http://brew.sh>`_
- Install dependencies using Homebrew::

    brew install gcc python3 git
    git clone --depth 1 git://github.com/andreasvc/disco-dop.git
    cd disco-dop
    sudo pip3 install -r requirements.txt
    env CC=gcc sudo python setup.py install
    sudo make

Other systems
-------------
If you do not run Linux, it is possible to run the code inside a virtual machine.
To do that, install `Virtualbox <https://www.virtualbox.org/wiki/Downloads>`_
and download the virtual machine imagine with disco-dop pre-installed:
http://illc-lil0.science.uva.nl/VMs/discodop-vboximage.zip


Documentation
=============
A manual page for the ``discodop`` command is installed as part of the
installation: ``man discodop``. Further documentation can be found at
http://discodop.readthedocs.org
To generate a local copy see the ``docs/README`` file.

Grammars
========
Cf. https://staff.fnwi.uva.nl/a.w.vancranenburgh/grammars/

The English, German, and Dutch grammars are described in a forthcoming paper
(van Cranenburgh et al., to appear); the French grammar appears in `Sangati &
van Cranenburgh (2015) <http://aclweb.org/anthology/W15-0902>`_.
For comparison, there is also an English grammar without discontinuous
constituents (``ptb-nodisc``).

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

- van Cranenburgh, Bod (2013). Discontinuous Parsing with an Efficient and Accurate
  DOP Model. Proc. of IWPT.
  http://www.illc.uva.nl/LaCo/CLS/papers/iwpt2013parser_final.pdf
- van Cranenburgh (2012). Efficient parsing with linear context-free rewriting
  systems. Proc. of EACL.
  http://andreasvc.github.io/eacl2012corrected.pdf
- van Cranenburgh, Scha, Sangati (2011). Discontinuous Data-Oriented Parsing:
  A mildly context-sensitive all-fragments grammar. Proc. of SPMRL.
  http://www.aclweb.org/anthology/W/W11/W11-3805.pdf

