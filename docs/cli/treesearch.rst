
treesearch
----------
Search through treebanks with queries.

Usage: ``discodop treesearch [--engine=X] [-t|-s|-c] <query> <treebank>...``

Options:

-e X, --engine=X
                Select query engine; possible options:

                :frag:
                    tree fragment queries (default); queries and files are in
                    bracket, discbracket, or export format.

                :regex: search through tokenized sentences with Python regexps.
                :tgrep2:
                    tgrep2 queries; files are bracket corpora
                    (optionally precompiled into tgrep2 format).

                :xpath: arbitrary xpath queries; files are dact XML corpora.
-c, --counts    Report counts; multiple queries can be given.
-s, --sents     Output sentences (default); multiple queries can be given.
-t, --trees     Output visualizations of trees.
-b, --brackets  Output raw trees in the original corpus format.
--indices       Report a sentence numebers of matches for each corpus
--breakdown     Report counts of types that match query.
--csv           Report counts in CSV format instead of the default flat format.
-f, --file      Read queries (one per line) from filename given as first argument.
--slice=<N:M>
                Only search in sentences N to M of each file; either N or
                M may be left out; slice indexing is 1-based and inclusive.
-m N, --max-count=N
                Stop after finding N matches; 0 for no limit.
-n, --line-number
                Prefix each line of output with the sentence number within
                its input file.
-o, --only-matching
                Only output the matching portion
                with ``--sents``, ``--trees``, and ``--brackets``.
-i, --ignore-case
                Ignore case in regex queries.
-M X, --macros=X
                A file with macros.
--numproc=N
                Use N independent processes, to enable multi-core usage
                (default: use all detected cores).

Tree fragments
^^^^^^^^^^^^^^
Search for literal matches of tree fragments, i.e., one or more connected grammar productions.
The treebanks being searched will be binarized if they are not already binarized;
treebanks may be discontinuous.

regular bracket trees::

(S (NP Mary) (VP (VB is) (JJ rich)) (. .))
(S (NP ) (VP (VB is) (JJ )) (. .))

discontinuous trees::

(S (VP (VB 0=is) (JJ 2=)) (NP 1=) (? 3=?))
(VP (VB 0=is) (JJ 2=rich))

More information: :ref:`file format documentation <bracket-format>`

Regular expressions
^^^^^^^^^^^^^^^^^^^
In contrast with the other engines, regular expressions treat the input as
plain text files, one sentence per line. The options --trees and --brackets are
not applicable. The syntax is Python's ``re`` syntax.

Regular expressions can contain both special and ordinary characters.
Most ordinary characters, like "A", "a", or "0", are the simplest
regular expressions; they simply match themselves.

The special characters are::

    "."      Matches any character except a newline.
    "^"      Matches the start of the string.
    "$"      Matches the end of the string or just before the newline at
             the end of the string.
    "*"      Matches 0 or more (greedy) repetitions of the preceding RE.
             Greedy means that it will match as many repetitions as possible.
    "+"      Matches 1 or more (greedy) repetitions of the preceding RE.
    "?"      Matches 0 or 1 (greedy) of the preceding RE.
    *?,+?,?? Non-greedy versions of the previous three special characters.
    {m,n}    Matches from m to n repetitions of the preceding RE.
    {m,n}?   Non-greedy version of the above.
    "\\"     Either escapes special characters or signals a special sequence.
    []       Indicates a set of characters.
             A "^" as the first character indicates a complementing set.
    "|"      A|B, creates an RE that will match either A or B.
    (...)    Matches the RE inside the parentheses.
             The contents can be retrieved or matched later in the string.
    (?:...)  Non-grouping version of regular parentheses.
    (?i)     Perform case-insensitive matching.

The special sequences consist of "\\" and a character from the list
below.  If the ordinary character is not on the list, then the
resulting RE will match the second character::

    \A       Matches only at the start of the string.
    \Z       Matches only at the end of the string.
    \b       Matches the empty string, but only at the start or end of a word.
    \B       Matches the empty string, but not at the start or end of a word.
    \d       Matches any decimal digit.
    \D       Matches any non-digit character.
    \s       Matches any whitespace character.
    \S       Matches any non-whitespace character.
    \w       Matches any alphanumeric character.
    \W       Matches the complement of \w.
    \\       Matches a literal backslash.

More information: https://docs.python.org/3/library/re.html#regular-expression-syntax

TGrep2 syntax overview
^^^^^^^^^^^^^^^^^^^^^^
Only treebanks in bracket format ary supported, but trees can be n-ary.
Note that the tgrep2 command needs to be installed.

TGrep2 operators::

  A < B       A is the parent of (immediately dominates) B.
  A > B       A is the child of B.
  A <N B      B is the Nth child of A (the first child is <1).
  A >N B      A is the Nth child of B (the first child is >1).
  A <, B      Synonymous with A <1 B.
  A >, B      Synonymous with A >1 B.
  A <-N B     B is the Nth-to-last child of A (the last child is <-1).
  A >-N B     A is the Nth-to-last child of B (the last child is >-1).
  A <- B      B is the last child of A (synonymous with A <-1 B).
  A >- B      A is the last child of B (synonymous with A >-1 B).
  A <` B      B is the last child of A (also synonymous with A <-1 B).
  A >` B      A is the last child of B (also synonymous with A >-1 B).
  A <: B      B is the only child of A.
  A >: B      A is the only child of B.
  A << B      A dominates B (A is an ancestor of B).
  A >> B      A is dominated by B (A is a descendant of B).
  A <<, B     B is a left-most descendant of A.
  A >>, B     A is a left-most descendant of B.
  A <<` B     B is a right-most descendant of A.
  A >>` B     A is a right-most descendant of B.
  A <<: B     There is a single path of descent from A and B is on it.
  A >>: B     There is a single path of descent from B and A is on it.
  A . B       A immediately precedes B.
  A , B       A immediately follows B.
  A .. B      A precedes B.
  A ,, B      A follows B.
  A $ B       A is a sister of B (and A != B).
  A $. B      A is a sister of and immediately precedes B.
  A $, B      A is a sister of and immediately follows B.
  A $.. B     A is a sister of and precedes B.
  A $,, B     A is a sister of and follows B.
  A = B       A is also matched by B.

More information: http://tedlab.mit.edu/~dr/Tgrep2/

XPath syntax examples
^^^^^^^^^^^^^^^^^^^^^
Search through treebanks in XML format with XPath; treebanks must be in
``dact`` format. Note: XPath support depends on the ``alpinocorpus`` library.

Find a particular word::

//node[@word='loopt']

This is case-sensitive.
If you want to find all inflectional variants of the verb ``lopen``, do::

//node[@lemma='lopen']

To find main clauses::

//node[@cat="smain"]

Finite subordinate clauses::

//node[@cat="cp" and node[@rel="body" and @cat="ssub"]]

This locates ``cp`` nodes with an ``ssub`` child that has ``body`` as function
tag (relation).

General XPath overview: https://en.wikipedia.org/wiki/XPath
Using XPath on Alpino treebanks: http://rug-compling.github.io/dact/cookbook/

Examples
^^^^^^^^
Show trees that can contain a NP modified by a PP::

    $ discodop treesearch --trees -e frag '(NP (NP ) (PP ))' wsj-02-21.mrg

Same query, but only show matching terminals::

    $ discodop treesearch --only-matching --sents -e frag '(NP (NP ) (PP ))' ~/data/wsj-02-21.mrg

Perform a large number of regex queries from a file, and store counts in a CSV file::

    $ discodop treesearch --csv --counts -e regex --file queries.txt corpus.txt > results.csv

