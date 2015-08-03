
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

