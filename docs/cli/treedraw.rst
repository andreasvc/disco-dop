
treedraw
--------
Usage: ``discodop treedraw [<treebank>...] [options]``

--fmt=(export|bracket|discbracket|tiger|alpino|dact)
                  Specify corpus format [default: export].

--encoding=enc    Specify a different encoding than the default utf-8.
--functions=x     :'leave'=default: leave syntactic labels as is,
                  :'remove': strip functions off labels,
                  :'add': show both syntactic categories and functions,
                  :'replace': only show grammatical functions.

--morphology=x    :'no': only show POS tags [default],
                  :'add': concatenate morphology tags to POS tags,
                  :'replace': replace POS tags with morphology tags,
                  :'between': add morphological node between POS tag and word.

--abbr            abbreviate labels longer than 5 characters.
--plain           disable ANSI colors.
-n, --numtrees=x  only display the first x trees from the input.

If no treebank is given, input is read from standard input; format is detected.
If more than one treebank is specified, trees will be displayed in parallel.
Pipe the output through ``less -RS`` to preserve the colors.

