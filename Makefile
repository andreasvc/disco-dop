all:
	python setup.py build_ext --inplace

.PHONY: clean test debug testdebug lint

clean:
	rm -f discodop/*.c discodop/*.so

test: all sample2.export
	rm -rf sample/
	python -tt -3 tests.py

test3: sample2.export
	rm -rf sample/
	python3 setup.py build_ext --inplace
	PYTHONIOENCODING=utf-8 python3 -bb -tt tests.py

sample2.export:
	wget http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/annotation/sample2.export

debug:
	python-dbg setup.py build_ext --inplace --debug --pyrex-gdb

testdebug: debug valgrind-python.supp
	valgrind --tool=memcheck --leak-check=full --num-callers=30 --suppressions=valgrind-python.supp python-dbg -tt -3 tests.py

valgrind-python.supp:
	wget http://codespeak.net/svn/lxml/trunk/valgrind-python.supp

# R=refactor, C0103 == Invalid name
lint:
	cd discodop/ ; \
	pylint \
		--indent-string='\t' \
		--disable=R,C0103 \
		demos.py eval.py fragments.py gen.py grammar.py lexicon.py runexp.py \
		parser.py tree.py treebank.py treedist.py treedraw.py treetransforms.py
