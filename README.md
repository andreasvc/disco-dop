Discontinuous DOP
=================

The aim of this project is to parse discontinuous constituents with
Data-Oriented Parsing (DOP), with a focus on global world domination.
Concretely, we build a DOP model with a Linear Context-Free Rewriting
System (LCFRS) as the symbolic backbone.

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
- k-best list: Huang & Chiang (2005), Better k-best parsing
- optimal binarization: Gildea (2010), Optimal parsing strategies for linear
  context-free rewriting systems

Requirements:
-------------
- Python 2.6+   http://www.python.org (need headers, e.g. python-dev package)
- Cython        http://www.cython.org
- Cython array	[array.pxd](https://github.com/andreasvc/cython/raw/master/Cython/Includes/cpython/array.pxd) [arrayarray.h](https://github.com/andreasvc/cython/raw/master/Cython/Includes/cpython/arrayarray.h) (to be included in Cython 0.17)
- GCC           http://gcc.gnu.org/
- NLTK          http://www.nltk.org
- Numpy         http://numpy.scipy.org/

For example, to install these dependencies and compile the code on Ubuntu
(tested on 12.04), run the following sequence of commands:

	sudo apt-get install python-dev python-nltk python-numpy cython build-essential
	git clone --depth 1 git://github.com/andreasvc/disco-dop.git
	cd disco-dop
	wget https://github.com/andreasvc/cython/raw/master/Cython/Includes/cpython/array.pxd
	wget https://github.com/andreasvc/cython/raw/master/Cython/Includes/cpython/arrayarray.h 
	make

Alternatively Cython, NLTK, and Numpy can be installed with
`pip install cython nltk numpy`,
which does not require root rights and may be more up-to-date.
NB: compilation will by default use `CFLAGS` used to compile Python.

To port the code to another compiler such as Visual C, replace the compiler
intrinsics in `macros.h`, `bit.pyx`, and `bit.pxd` to their equivalents in the
compiler in question. This mainly concerns operations to scan for bits in
integers, for which these compiler intrinsics provide the most efficient
implementation on a given processor.

Usage: parser
-------------
To run an experiment, copy the file sample.prm and edit its parameters. These
parameters can then be invoked by executing:

	python runexp.py filename.prm

This will create a new directory with the basename of the parameter file, i.e.,
filename/ in this case. This directory must not exist yet, to avoid overwriting
previous results. The directory will contain the grammar rules and lexicon in a
text format, as well as the parsing results and the gold standard file in
Negra's export format. 

Corpora are expected to be in Negra's export format. Access to the [Negra
corpus](http://www.coli.uni-saarland.de/projects/sfb378/negra-corpus/) itself
can be requested for non-commercial purposes, while the [Tiger
corpus](http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/) is freely
available for download for research purposes.

Alternatively, there is a simpler parser in the `shedskin/` directory. This
LCFRS parser only produces the Viterbi parse. The grammar is supplied in a file
following a simple text format. The `plcfrs.py` script can be translated to C++
by the [Shed Skin](http://code.google.com/p/shedskin/) compiler, after which
the resulting code can be compiled with `make`:

    sudo apt-get install shedskin
    cd disco-dop/shedskin
    shedskin -b -l -w plcfrs.py
    make

Usage: tools
------------
Aside from the parser there are some standalone tools.

- `eval.py`:             while `runexp.py` already shows F-scores, more detailed
                         evaluation can be done with `eval.py`, which accepts
                         `EVALB` style parameter files: `python eval.py
                         sample/results.gold sample/results.dop proper.prm`
- `treetransforms.py`:   a command line interface to perform transformations on
                         treebanks 
- `grammar.py`:          a command line interface to read off grammars
                         from (binarized) treebanks
- `fragments.py`:        finds recurring or common fragments in one or more
                         treebanks. It can be used with discontinuous as well as
                         Penn-style bracketed treebanks.
- `demos.py`:            contains examples of various formalisms encoded in
                         LCFRS grammars.
- `gen.py`:              an experiment in LCFRS generation.

These programs can be started without arguments for instructions.
