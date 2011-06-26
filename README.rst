Discontinuous DOP
=================

The aim of this project is to parse discontinuous constituents with
Data-Oriented Parsing (DOP), with a focus on global world domination.
Concretely, we build a DOP model with a Simple Range Concatenation grammar as
the symbolic backbone.

- parser, estimates: Maier & Kallmeyer (2010), Data-driven parsing with
  probabilistic linear context-free rewriting systems.
- data-oriented parsing: Goodman (2002), Efficient parsing of DOP with
  PCFG-reductions
  Sangati & Zuidema (to appear), Double DOP
- k-best list: Huang & Chiang (2005), Better k-best parsing
- optimal binarization: Gildea (2010), Optimal parsing strategies for linear
  context-free rewriting systems


Requirements: 
-------------
- python 2.6+   http://www.python.org
- NLTK          http://www.nltk.org
- cython        http://www.cython.org (optional)
  requires array support http://trac.cython.org/cython_trac/ticket/314
  compile with: "make". NB: compilation uses CFLAGS used to compile
  Python as defaults.
- numpy         http://numpy.scipy.org/ (only for estimates)

plcfrs.py, rcgrules.py, kbest.py and runexp.py can be run without arguments as
a demonstration.

