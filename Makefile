all:
	python setup.py build_ext --inplace

.PHONY: clean test debug testdebug html lint

clean:
	rm -f *.c *.so

test: all sample2.export
	rm -rf sample/
	python -tt runexp.py --test

test3: sample2.export
	rm -rf sample/
	python3 setup.py build_ext --inplace
	PYTHONIOENCODING=utf-8 python3 -bb -tt runexp.py --test

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

# R=refactor, C0103 == Invalid name
lint:
	pylint --indent-string='\t' --disable=R,C0103 --good-names=a,b,c,i,j,k,ex,Run,_ \
		demos.py eval.py fragments.py gen.py grammar.py lexicon.py runexp.py \
		setup.py parser.py tree.py treebank.py treedist.py treedraw.py treetransforms.py
