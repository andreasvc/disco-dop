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
- coarse-to-fine pruning: posterior pruning (PCFG only),
  *k*-best coarse-to-fine

DOP specific (parsing with tree fragments):

- implementations: Goodman's DOP reduction, Double-DOP.
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

- Python 2.7+/3   http://www.python.org (need headers, e.g. python-dev package)
- Cython 0.20+    http://www.cython.org
- GCC             http://gcc.gnu.org/
- Numpy 1.5+      http://numpy.org/

Debian, Ubuntu based systems
----------------------------
For example, to install these dependencies and the latest stable release on
an `Ubuntu <http://www.ubuntu.com>`_ system
using `pip <http://www.pip-installer.org>`_,
issue the following commands::

    sudo apt-get install build-essential python-dev python-numpy python-pip
    wget https://github.com/andreasvc/disco-dop/raw/master/requirements.txt
    pip install --user -r requirements.txt
    pip install --user disco-dop

To compile the latest development version on Ubuntu,
run the following sequence of commands::

    sudo apt-get install build-essential python-dev python-numpy python-pip git
    git clone --depth 1 git://github.com/andreasvc/disco-dop.git
    cd disco-dop
    pip install --user -r requirements.txt
    python setup.py install --user

(the ``--user`` option means the packages will be installed to your home
directory which does not require root privileges).

Other Linux systems
-------------------
This assumes no root access, but assumes that ``gcc`` is installed.

Add local installation directory to PATH
(replace with equivalent for your shell if you do not use bash)::

    mkdir -p ~/.local
    echo export PATH=$HOME/.local/bin:$PATH >> ~/.bashrc
    export LD_LIBRARY_PATH=$HOME/.local/lib:/usr/lib64:/usr/lib >>~/.bashrc

Install Python 2.7 from source, if not installed already.
Python may require some libraries such as ``zlib`` and ``readline``;
installation steps are similar to the ones below.

::

    wget http://www.python.org/ftp/python/2.7.6/Python-2.7.6.tgz
    tar -xzf Python-2.7.6.tgz
    cd Python-2.7.6
    ./configure --prefix=$HOME/.local --enable-shared
    make && make install

Check by running ``python`` that version 2.7.6 was installed successfully and
is the default.

Install pip::

    wget https://raw.github.com/pypa/pip/master/contrib/get-pip.py
    python get-pip.py --user

Install the latest development version of discodop::

    wget https://github.com/andreasvc/disco-dop/archive/master.zip
    unzip disco-dop-master.zip
    cd disco-dop-master
    pip install --user -r requirements.txt
    python setup.py install --user

Other systems
-------------
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


Documentation
=============
The documentation can be found at http://staff.science.uva.nl/~acranenb/discodop/

To generate a local copy install `Sphinx <http://sphinx-doc.org/>`_
and issue ``python genrst.py && make html`` in the ``docs/`` directory; the
result will be in ``_build/html``.

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

- van Cranenburgh, Bod (2013). Discontinuous Parsing with an Efficient and Accurate DOP Model.
  Proc. of IWPT.
  http://acl.cs.qc.edu/iwpt2013/proceedings/Splits/9_pdfsam_IWPTproceedings.pdf
- van Cranenburgh (2012). Efficient parsing with linear context-free rewriting
  systems. Proc. of EACL.
  http://staff.science.uva.nl/~acranenb/eacl2012corrected.pdf
- van Cranenburgh, Scha, Sangati (2011). Discontinuous Data-Oriented Parsing:
  A mildly context-sensitive all-fragments grammar. Proc. of SPMRL.
  http://www.aclweb.org/anthology/W/W11/W11-3805.pdf

