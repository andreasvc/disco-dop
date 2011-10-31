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
- python 2.6+   http://www.python.org
- NLTK          http://www.nltk.org
- cython        http://www.cython.org (optional)

  * requires array support http://trac.cython.org/cython_trac/ticket/314
  * compile with: "make". NB: compilation uses CFLAGS used to compile Python as defaults.

- numpy         http://numpy.scipy.org/ (only for estimates)

The python files can be run without arguments as a demonstration.

To run an experiment, copy the file sample.prm and edit its parameters.  These
parameters can then be invoked by executing "python runexp.py filename.prm".
This will create a new directory with the name specified in the parameter file,
e.g., negra25/

This directory will contain the grammar rules and lexicon in a text format, as
well as the parsing results and the gold standard file in Negra's export
format.

Corpora are expected to be in Negra's export format. Access to the Negra corpus
itself can be requested for non-commercial purposes, while the Tiger corpus is
freely available.

