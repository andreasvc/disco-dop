all:
	python setup.py build_ext --inplace
	python setup.py install --user

.PHONY: clean test debug testdebug lint

clean:
	rm -rf build/
	cd discodop; rm -rf *.c *.so *.html *.pyc __pycache__

test: all sample2.export
	rm -rf sample/
	python -tt -3 tests.py

test3: sample2.export
	rm -rf sample/
	python3 setup.py build_ext --inplace
	PYTHONIOENCODING=utf-8 python3 -bb -tt tests.py

sample2.export:
	# kludge to restore original encoding & strip spurious HTML sent by server
	curl http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/annotation/sample2.export \
	| iconv --from-code=utf8 --to-code=iso8859-1 \
	| sed -e 's/\(<[^>]*>\)\+//' > sample2.export

debug:
	python-dbg setup.py build_ext --inplace --debug --pyrex-gdb

testdebug: debug valgrind-python.supp
	valgrind --tool=memcheck --leak-check=full --num-callers=30 --suppressions=valgrind-python.supp python-dbg -tt -3 tests.py

valgrind-python.supp:
	wget http://codespeak.net/svn/lxml/trunk/valgrind-python.supp

# pylint: R=refactor, C0103 == Invalid name
lint:
	pep8 --ignore=E1,W1 \
		discodop/*.py web/*.py && \
	pep8 --ignore=E1,W1,F,E901,E225,E227,E211 \
		discodop/*.pyx && \
	pylint \
		--indent-string='\t' \
		--disable=R,C0103 \
		discodop/*.py web/*.py
