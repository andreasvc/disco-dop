Discontinuous DOP
=================

The aim of this project is to parse discontinuous constituents with
Data-Oriented Parsing (DOP), with a focus on global world domination.
Concretely, we build a DOP model with a Linear Context-Free Rewriting
System (LCFRS) as the symbolic backbone.

- parser, estimates: Maier & Kallmeyer (2010), Data-driven parsing with
  probabilistic linear context-free rewriting systems.
- data-oriented parsing:

  * Goodman (2002), Efficient parsing of DOP with PCFG-reductions
  * Sangati & Zuidema (2011), Accurate parsing with compact tree-substitution grammars: Double-DOP

- k-best list: Huang & Chiang (2005), Better k-best parsing
- optimal binarization: Gildea (2010), Optimal parsing strategies for linear
  context-free rewriting systems


Requirements:
-------------
- python 2.6+   http://www.python.org (need headers, e.g. python-dev package)
- NLTK          http://www.nltk.org
- cython        http://www.cython.org
- cython array	[array.pxd](https://github.com/andreasvc/cython/raw/c7aa65ab69a396438e2bf06279a06f8714dc3f29/Cython/Includes/cpython/array.pxd) [arrayarray.h](https://github.com/andreasvc/cython/raw/c7aa65ab69a396438e2bf06279a06f8714dc3f29/Cython/Includes/cpython/arrayarray.h)
- numpy         http://numpy.scipy.org/
- GCC

For example, to install these dependencies and compile the code on Ubuntu (tested on 12.04), run the following sequence of commands:

	apt-get install python-dev python-nltk cython python-numpy build-essential
	git clone --depth 1 git://github.com/andreasvc/disco-dop.git
	cd disco-dop
	wget https://github.com/andreasvc/cython/raw/c7aa65ab69a396438e2bf06279a06f8714dc3f29/Cython/Includes/cpython/array.pxd
	wget https://github.com/andreasvc/cython/raw/c7aa65ab69a396438e2bf06279a06f8714dc3f29/Cython/Includes/cpython/arrayarray.h
	make

NB: compilation will by default use `CFLAGS` used to compile Python.

The python files can be run without arguments as a demonstration.

To run an experiment, copy the file sample.prm and edit its parameters.  These
parameters can then be invoked by executing "python runexp.py filename.prm".
This will create a new directory with the basename of the parameter file, i.e.,
filename/ in this case.

This directory will contain the grammar rules and lexicon in a text format, as
well as the parsing results and the gold standard file in Negra's export
format.

Corpora are expected to be in Negra's export format. Access to the [Negra
corpus](http://www.coli.uni-saarland.de/projects/sfb378/negra-corpus/) itself
can be requested for non-commercial purposes, while the [Tiger
corpus](http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/) is freely
available for download for research purposes.

Alternatively, there is a simpler parser in the "shedskin" directory. This
LCFRS parser only produces the Viterbi parse. The grammar is supplied in a file
following a simple text format. The "plcfrs.py" script can be translated to C++
by the [Shed Skin](http://code.google.com/p/shedskin/) compiler, after which
the resulting code can be compiled with `make`.


