
treesearch
----------
Search through treebanks with queries.

Usage: ``discodop treesearch [--engine=X] [-t|-s|-c] <query> <treebank>...``

Options:

-e X, --engine=X
                Select query engine; possible options:

                :tgrep2:
                    tgrep2 queries (default); files are bracket corpora
                    (optionally precompiled into tgrep2 format).

                :frag:
                    tree fragment queries; queries and files are in
                    bracket, discbracket, or export format.

                :xpath: arbitrary xpath queries; files are dact XML corpora.
                :regex: search through tokenized sentences with Python regexps.
-c, --counts    Report counts; if more than one query is given, output is CSV.
-s, --sents     Output sentences (default).
-t, --trees     Output visualizations of trees.
-b, --brackets  Output raw trees in the original corpus format.
--indices       Report a sentence numebers of matches for each corpus
--breakdown     Report counts of types that match query; output is CSV.
-f, --file      Read queries from filename given as first argument.
--slice=<N:M>
                Only search in sentences N to M of each file; either N or
                M may be left out; slice indexing is 1-based and inclusive.
-m N, --max-count=N
                Stop after finding N matches.
-n, --line-number
                Prefix each line of output with the sentence number within
                its input file.
-o, --only-matching
                Only output the matching portion
                with ``--sents``, ``--trees``, and ``--brackets``.
-M X, --macros=X
                A file with macros.
--numproc=N
                Use N independent processes, to enable multi-core usage
                (default: use all detected cores).

TGrep2 syntax overview
^^^^^^^^^^^^^^^^^^^^^^
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

Tree fragments
^^^^^^^^^^^^^^

regular bracket trees::

(S (NP Mary) (VP (VB is) (JJ rich)) (. .))
(S (NP ) (VP (VB is) (JJ )) (. .))

discontinuous trees::

(S (VP (VB 0=is) (JJ 2=)) (NP 1=) (? 3=?))
(VP (VB 0=is) (JJ 2=rich))

More information: :ref:`file format documentation <bracket-format>`

Regular expressions
^^^^^^^^^^^^^^^^^^^

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

More information: http://docs.python.org/3/library/re.html#regular-expression-syntax
