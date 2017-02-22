
treedraw
--------
Visualize parse trees with ASCII art or LaTeX/SVG.

Usage: ``discodop treedraw [<treebank>...] [options]``

--fmt=<export|bracket|discbracket|tiger|alpino|dact>
                  Specify input format [default: export].

--encoding=enc    Specify input encoding [default: utf-8].
--functions=x     :'leave': leave syntactic labels as is [default],
                  :'remove': strip functions off labels,
                  :'add': show both syntactic categories and functions,
                  :'replace': only show grammatical functions.

--morphology=x    :'no': only show POS tags [default],
                  :'add': concatenate morphology tags to POS tags,
                  :'replace': replace POS tags with morphology tags,
                  :'between': add morphological node between POS tag and word.

--abbr            abbreviate labels longer than 5 characters.
--plain           disable ANSI/HTML colors.
--frontier        only show terminal and non-terminal leaves.
--output=x        :'text': (default) output in ASCII/ANSI art format.
                  :'html': similar to 'text', but wrap output in HTML.
                  :'svg': SVG wrappend in HTML.
                  :'tikznode': generate LaTeX code using TiKZ.
                  :'tikzmatrix': generate LaTeX code using TiKZ.
                  :'tikzqtree': generate LaTeX code using TiKZ-qtree, only applicable to continuous trees.

-n, --numtrees=x  only display the first x trees from the input.

If no treebank is given, input is read from standard input; format is detected;
has the advantage of reading the input incrementally.
If more than one treebank is specified, trees will be displayed in parallel.
The ANSI colors can be viewed in a terminal with ``less -RS``.

Examples
^^^^^^^^
View trees in a treebank::

    $ discodop treedraw < wsj-02-21.mrg | less -RS

Apply a transformation and view the result::

    $ discodop treetransforms --binarize --fmt=bracket wsj-02-21.mrg | discodop treedraw | less -RS

