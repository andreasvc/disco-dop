
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
                :regex: search through tokenized sentences with Python regexps
-c, --counts    report counts
-s, --sents     output sentences (default)
-t, --trees     output visualizations of trees
-b, --brackets  output raw trees in the original corpus format
-f, --file      read query from filename given as first argument
-o, --only-matching
                only output the matching portion
                with ``--sents``, ``--trees``, and ``--brackets``
-m N, --max-count=N
                with ``--counts``: only consider first n sentences;
                otherwise: stop after n matches.
-n, --line-number
                Prefix each line of output with the sentence number within
                its input file.
-M X, --macros=X
                file with macros
--numthreads=X
                Number of concurrent threads to use.

