
treesearch
----------
Search through treebanks with queries.

Usage: ``discodop treesearch [--engine=(tgrep2|xpath|regex)] [-t|-s|-c] <query> <treebank>...``

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
-c, --counts    Report counts.
-s, --sents     Output sentences (default).
-t, --trees     Output visualizations of trees.
-b, --brackets  Output raw trees in the original corpus format.
-f, --file      Read query from filename given as first argument.
-m N, --max-count=N
                Stop after finding n matches.
--slice=<N:M>
                Only search in sentences N to M of each file; either N or
                M may be left out; slice indexing is 1-based and inclusive.
-o, --only-matching
                Only output the matching portion
                with ``--sents``, ``--trees``, and ``--brackets``.
-n, --line-number
                Prefix each line of output with the sentence number within
                its input file.
-M X, --macros=X
                A file with macros.
--numthreads=X
                Number of concurrent threads to use.

