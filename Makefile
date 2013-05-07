all:
	python setup.py build_ext --inplace

.PHONY: clean test debug testdebug lint

clean:
	rm -f discodop/*.c discodop/*.so

test: all sample2.export
	rm -rf discodop/sample/
	cd discodop/ ; python -tt -3 -m discodop.runexp --test

test3: sample2.export
	rm -rf discodop/sample/
	python3 setup.py build_ext --inplace
	cd discodop/ ; PYTHONIOENCODING=utf-8 python3 -bb -tt -m discodop.runexp --test

sample2.export:
	cd discodop/ ; wget http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/annotation/sample2.export

debug:
	cd discodop/ ; python-dbg setup.py build_ext --inplace --debug --pyrex-gdb

testdebug: debug valgrind-python.supp
	cd discodop/ ; valgrind --tool=memcheck --leak-check=full --num-callers=30 --suppressions=valgrind-python.supp python-dbg -tt -3 -m discodop.runexp --test

valgrind-python.supp:
	cd discodop/ ; wget http://codespeak.net/svn/lxml/trunk/valgrind-python.supp

# R=refactor, C0103 == Invalid name
lint:
	cd discodop/ ; pylint \
		--indent-string='\t' \
		--disable=R,C0103 \
		demos.py eval.py fragments.py gen.py grammar.py lexicon.py runexp.py \
		parser.py tree.py treebank.py treedist.py treedraw.py treetransforms.py
