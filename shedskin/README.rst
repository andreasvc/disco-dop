This is a parser for Probabilistic Linear Context-Free Rewriting Systems (PLCFRS).
It only produces the Viterbi parse. The grammar is supplied in a
file following a simple text format. The ``plcfrs.py`` script can be translated
to C++ by the `Shed Skin <http://code.google.com/p/shedskin/>`_ compiler, after
which the resulting code can be compiled with ``make``::

    sudo apt-get install shedskin
    cd disco-dop/shedskin
    shedskin -b -l -w plcfrs.py
    make

