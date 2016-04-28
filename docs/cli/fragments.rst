
fragments
---------
Extract recurring tree fragments from constituency treebanks.

| Usage: ``discodop fragments <treebank1> [treebank2] [options]``
| or: ``discodop fragments --batch=<dir> <treebank1> <treebank2>... [options]``

If only one treebank is given, extract fragments in common between its pairs of
trees. If two treebanks are given, extract fragments in common between the
trees of the first & second treebank.
Input is in Penn treebank format (S-expressions), one tree per line.
Output contains lines of the form "tree<TAB>frequency".
Frequencies refer to the first treebank by default.
Output is sent to stdout; to save the results, redirect to a file.

Options:
^^^^^^^^
--fmt=<export|bracket|discbracket|tiger|alpino|dact>
              when format is not ``bracket``, work with discontinuous trees;
              output is in ``discbracket`` format:
              tree<TAB>sentence<TAB>frequency
              where ``tree`` has indices as leaves, referring to elements of
              ``sentence``, a space separated list of words.

--numtrees=n  only read first n trees from first treebank
--encoding=x  specify treebank encoding, e.g. utf-8 [default], iso-8859-1, etc.
-o file       Write output to ``file`` instead of stdout.
--complete    ``treebank1`` is a list of fragments (needle), result is the
              indices / counts of these fragments in ``treebank2`` (haystack).
--batch=dir   enable batch mode; any number of treebanks ``> 1`` can be given;
              first treebank (A) will be compared to each (B) of the rest.
              Results are written to filenames of the form ``dir/A_B``.
              Counts/indices are from B.
--indices     report sets of 0-based indices where fragments occur instead of
              frequencies.

--relfreq     report relative frequencies wrt. root node of fragments of the form ``n/m``.
--approx      report counts of occurrence as maximal fragment (lower bound)
--nofreq      do not report frequencies.
--cover=<n[,m]>
              include all non-maximal/non-recurring fragments up to depth ``n``
              of first treebank; optionally, limit number of substitution
              sites to ``m`` (default is unlimited).

--twoterms=x  only extract fragments with at least two lexical terminals,
              one of which has a POS tag which matches the given regex.
              For example, to match POS tags of content words in the
              Penn treebank: ``^(?:NN(?:[PS]|PS)?|(?:JJ|RB)[RS]?|VB[DGNPZ])$``
--adjacent    only consider pairs of adjacent trees (i.e., sent no. ``n, n + 1``).
--debin       debinarize fragments.
--alt         alternative output format: ``(NP (DT "a") NN)``
              default: ``(NP (DT a) (NN ))``
--numproc=n   use ``n`` independent processes, to enable multi-core usage
              (default: 1); use 0 to detect the number of CPUs.
--debug       extra debug information, ignored when ``numproc > 1``.
--quiet       disable all messages.

