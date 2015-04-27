
fragments
---------
| Usage: ``discodop fragments <treebank1> [treebank2] [options]``
| or: ``discodop fragments --batch=<dir> <treebank1> <treebank2>... [options]``

If only one treebank is given, fragments occurring at least twice are sought.
If two treebanks are given, finds common fragments between first & second.
Input is in Penn treebank format (S-expressions), one tree per line.
Output contains lines of the form "tree<TAB>frequency".
Frequencies refer to the first treebank by default.
Output is sent to stdout; to save the results, redirect to a file.

Options:

--fmt=(export|bracket|discbracket|tiger|alpino|dact)
              when format is not ``bracket``, work with discontinuous trees;
              output is in ``discbracket`` format:
              tree<TAB>sentence<TAB>frequency
              where ``tree`` has indices as leaves, referring to elements of
              ``sentence``, a space separated list of words.

-o file       Write output to ``file`` instead of stdout.
--indices     report sets of 0-based indices instead of frequencies.

--cover=n[,m]
              include all non-maximal/non-recurring fragments up to depth ``n``
              of first treebank; optionally, limit number of substitution
              sites to ``m``.

--complete    find complete matches of fragments from treebank1 (needle) in
              treebank2 (haystack); frequencies are from haystack.
--batch=dir   enable batch mode; any number of treebanks > 1 can be given;
              first treebank (A) will be compared to all others (B).
              Results are written to filenames of the form dir/A_B.
              Counts/indices are from B.
--numproc=n   use n independent processes, to enable multi-core usage
              (default: 1); use 0 to detect the number of CPUs.
--numtrees=n  only read first n trees from first treebank
--encoding=x  use x as treebank encoding, e.g. utf-8, iso-8859-1, etc.
--nofreq      do not report frequencies.
--approx      report counts of occurrence as maximal fragment (lower bound)
--relfreq     report relative frequencies wrt. root node of fragments.
--debin       debinarize fragments.
--twoterms    only consider fragments with at least two lexical terminals.
--adjacent    only consider pairs of adjacent fragments (n, n + 1).
--alt         alternative output format: (NP (DT "a") NN)
              default: (NP (DT a) (NN ))
--debug       extra debug information, ignored when numproc > 1.
--quiet       disable all messages.


