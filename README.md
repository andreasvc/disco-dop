<img align="right" alt="Discontinuous Data-Oriented Parsing" style="float:right" src="http://staff.science.uva.nl/~acranenb/disco-dop.png" title="contrived discontinuous constituent for expository purposes." />

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
  * Sangati & Zuidema (2011), Accurate parsing with compact tree-substitution
    grammars: Double-DOP
- k-best list: Huang & Chiang (2005), Better k-best parsing
- optimal binarization: Gildea (2010), Optimal parsing strategies for linear
  context-free rewriting systems

Requirements
------------
- Python 2.7+     http://www.python.org (need headers, e.g. python-dev package)
- Cython 0.17.2+  http://www.cython.org
- GCC             http://gcc.gnu.org/
- Numpy           http://numpy.scipy.org/

For example, to install these dependencies and compile the code on Ubuntu
(tested on 12.04), run the following sequence of commands:

    sudo apt-get install python-dev python-numpy build-essential
    sudo pip install cython
    git clone --depth 1 git://github.com/andreasvc/disco-dop.git
    cd disco-dop
    make

Alternatively Cython and Numpy can be installed with
`pip install --user cython numpy`,
which does not require root rights and may be more up-to-date than the
packages in your distribution.

To port the code to another compiler such as Visual C, replace the compiler
intrinsics in `macros.h`, `bit.pyx`, and `bit.pxd` to their equivalents in the
compiler in question. This mainly concerns operations to scan for bits in
integers, for which these compiler intrinsics provide the most efficient
implementation on a given processor.

Usage: parser
-------------
To run a full experiment from treebank to evaluation on a test set,
copy the file sample.prm and edit its parameters.
These parameters can then be invoked by executing:

    python runexp.py filename.prm

This will create a new directory with the basename of the parameter file, i.e.,
`filename/` in this case. This directory must not exist yet, to avoid
accidentally overwriting previous results. The directory will contain the
grammar rules and lexicon in a text format, as well as the parsing results and
the gold standard file in Negra's export format.

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
Aside from the parser there are some standalone tools:

- `treetransforms.py`:   A command line interface to perform transformations on
                         treebanks such as binarization.
- `grammar.py`:          A command line interface to read off grammars
                         from (binarized) treebanks.
- `parser.py`:           A basic command line interface to the parser
                         comparable to bitpar. Reads grammars from text files.
- `eval.py`:             Discontinuous evaluation. Reports F-scores and other
                         metrics. Accepts `EVALB` parameter files:
                         `python eval.py negra-corpus.export sample/plcfrs.export proper.prm`
- `fragments.py`:        Finds recurring or common fragments in one or more
                         treebanks. It can be used with discontinuous as well as
                         Penn-style bracketed treebanks.
- `demos.py`:            Contains examples of various formalisms encoded in
                         LCFRS grammars.
- `gen.py`:              An experiment in generation with LCFRS.

These programs can be started without arguments for instructions.

Acknowledgments
---------------

The Tree datastructures in `tree.py` and the simple binarization algorithm in
`treetransforms.py` was taken from [NLTK](http://www.nltk.org).
The Zhang-Shasha tree-edit distance algorithm in `treedist.py` was taken from
[https://github.com/timtadh/zhang-shasha](https://github.com/timtadh/zhang-shasha)
Elements of the PLCFRS parser and punctuation re-attachment are based on code from
`rparse`. Various other bits from the Stanford parser, Berkeley parser, Bubs parser, &c.
