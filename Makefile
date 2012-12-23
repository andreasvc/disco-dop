all:
	python setup.py build_ext --inplace

.PHONY: clean test debug testdebug html lint

clean:
	rm -f *.c *.so

test: all sample2.export
	python -c 'import runexp; runexp.test()'

sample2.export:
	wget http://www.ims.uni-stuttgart.de/projekte/TIGER/TIGERCorpus/annotation/sample2.export

debug:
	python-dbg setup.py build_ext --inplace --debug --pyrex-gdb

testdebug: debug valgrind-python.supp
	valgrind --tool=memcheck --leak-check=full --num-callers=30 --suppressions=valgrind-python.supp python-dbg testall.py

valgrind-python.supp:
	wget http://codespeak.net/svn/lxml/trunk/valgrind-python.supp

html:
	cython --annotate --timestamps --verbose -X \
cdivision=True,\
nonecheck=False,\
wraparound=False,\
boundscheck=False \
*.pyx

lint:
	pylint eval.py fragments.py grammar.py runexp.py treebank.py treetransforms.py \
		demos.py gen.py setup.py parser.py tree.py treedist.py tikzdtree.py
