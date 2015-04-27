
runexp
------
Usage: ``discodop runexp <parameter file> [--rerun]``

If a parameter file is given, an experiment is run. See the file sample.prm for
an example parameter file. To repeat an experiment with an existing grammar,
pass the option ``--rerun``. The directory with the name of the parameter file
without extension must exist in the current path; its results will be
overwritten.

Example invocation::

    discodop runexp filename.prm

This will create a new directory with the base name of the parameter file, i.e.,
``filename/`` in this case. This directory must not exist yet, to avoid
accidentally overwriting previous results. The directory will contain the
grammar rules and lexicon in a text format, as well as the parsing results and
the gold standard file in Negra's export format.

After running ``discodop runexp``, a number of files are produced with parsing statistics:

:``output.log``: a log file with all messages displayed during parsing. This
                 file contains ANSI codes for colors, so view it with ``less -R`` in a terminal.
:``pcdist.txt``: shows the distribution of parsing complexity (cf.
                 `Gildea, NAACL 2010 <http://aclweb.org/anthology/N10-1118>`_
                 for the definition) among the grammar rules.
:``stats.tsv``: is a tab-separated file with additional information. For each
                tuple of ``sentid, len, stage``, the following columns are
                given:

    :elapsedtime: CPU time to complete a given stage (not including any
                  required preceding stages)
    :logprob: the log probability of the selected parse.
    :frags: the number of fragments in the best derivation of the selected
            parse. If this grammar did not use tree fragments, this number will
            equal the number of nodes in the tree.
    :numitems: the total number of items ``(label, span)`` in the chart of this
               stage.
    :golditems: if this stage is pruned by the previous stage, the number of
                items from the gold tree that remain after pruning. if this
                stage is not pruned, the number of gold items in the chart.
    :totalgolditems: the number of items in the gold tree, for the binarized
                     tree; the discontinuities in the tree are splitted to make
                     the number of items between discontinuous trees and
                     splitted trees comparable.

