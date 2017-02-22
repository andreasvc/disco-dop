
runexp
------
Run an experiment given a parameter file. Does grammar extraction, parsing, and evaluation.

Usage: ``discodop runexp <parameter file> [--rerun]``

If a parameter file is given, an experiment is run.
Given the parameter file ``sample.prm``, a new directory will be created with
the base name of the parameter file, i.e., ``sample/`` in this case. This
directory must not exist yet, to avoid accidentally overwriting previous
results. To this directory the grammar rules and lexicon will be written in a
text format, as well as the parsing results and the gold standard parse trees
in the same format.

To repeat an experiment with an existing grammar, pass the option ``--rerun``.
The directory with the name of the parameter file without extension must exist
in the current path; its results will be overwritten.

Parameter file and example invocation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
See the :doc:`reference documentation on parameter files <../params>`.
A minimal parameter file::

    stages=[
      dict(
        name='pcfg',  # an identifier, used as filename when writing results
        mode='pcfg',  # use the PCFG CKY parser
      ),
    ],

    evalparam='proper.prm',  # EVALB-style parameter file
    # train / test sets
    corpusfmt='bracket',  # choices: export, bracket, discbracket, alpino, tiger
    traincorpus=dict(
        path='ptb-02-21.mrg',
        maxwords=100,  # max number of words for sentences in train corpus
    ),
    testcorpus=dict(
        path='ptb-24.mrg',
        maxwords=100,  # max number of words for sentences in test corpus
    ),

See ``sample.prm`` in the code repository for a more extensive example. The
file ``proper.prm`` can also be found there, which is a version of the
``COLLINS.prm`` file typically used with ``EVALB``, adapted for discontinuous
parsing. Ensure that all referenced files are in the current directory or
specified with a path, and run as::

    $ discodop runexp sample.prm

Parsing statistics
^^^^^^^^^^^^^^^^^^
After running ``discodop runexp``, a number of additional files are produced
with parsing statistics:

:``output.log``: a log file with all messages displayed during parsing. This
                 file contains ANSI codes for colors, so view it with ``less -R`` in a terminal,
                 or remove them: ``sed "s,\x1B\[[0-9;]*[a-zA-Z],,g" output.log | less``.
:``pcdist.txt``: shows the distribution of parsing complexity (cf.
                 `Gildea, NAACL 2010 <http://aclweb.org/anthology/N10-1118>`_
                 for the definition) among the grammar rules.
:``stats.tsv``:
                is a tab-separated file with additional information. For each
                tuple of ``sentid, len, stage``, the following columns are
                given:

    :elapsedtime: CPU time to complete a given stage (not including any
                  required preceding stages)
    :logprob: the log probability of the selected parse tree.
    :frags: the number of fragments in the best derivation of the selected
            parse tree. If this grammar did not use tree fragments, this number will
            equal the number of non-terminal nodes in the tree.
    :numitems: the total number of items ``(label, span)`` in the chart of this
               stage.
    :golditems: if this stage is pruned by the previous stage, the number of
                items from the gold tree that remain after pruning. if this
                stage is not pruned, the number of gold items in the chart.
    :totalgolditems: the number of items in the gold tree, for the binarized
                     tree; the discontinuities in the tree are splitted to make
                     the number of items between discontinuous trees and
                     splitted trees comparable.

