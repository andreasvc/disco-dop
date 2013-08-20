.PHONY: clean test debug testdebug lint

all:
	python setup.py install --user

clean:
	python setup.py clean
	rm -rf build/
	cd discodop; rm -rf *.c *.so *.html *.pyc __pycache__

test: all
	rm -rf sample/
	python -tt -3 tests.py && \
	cd tests/ && \
	sh run.sh

test3:
	rm -rf sample/
	python3 setup.py build_ext --inplace
	PYTHONIOENCODING=utf-8 python3 -bb -tt tests.py

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
	cd web; pylint \
		--indent-string='\t' \
		--disable=R,C0103 \
		~/.local/lib/python2.7/site-packages/discodop/*py *.py
