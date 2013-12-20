.PHONY: clean test testdebug lint

all: man
	python setup.py install --user

clean:
	rm -rf build/
	find discodop -name '*.c' -delete
	find discodop -name '*.so' -delete
	find discodop -name '*.pyc' -delete
	find discodop -name '*.html' -delete
	rm -rf discodop/__pycache__
	cd docs && make clean

man:
	mkdir -p ~/.local/man/man1
	cd docs && make man && cp _build/man/discodop.1 ~/.local/man/man1/

test: all inplace
	py.test --doctest-modules discodop/ tests/unittests.py && \
	python -tt -3 tests.py && \
	cd tests/ && \
	sh run.sh

test3:
	rm -rf sample/
	python3 setup.py install --user
	PYTHONIOENCODING=utf-8 python3 -bb -tt tests.py

debug:
	python-dbg setup.py build_ext --inplace --debug --pyrex-gdb

testdebug: debug valgrind-python.supp
	valgrind --tool=memcheck --leak-check=full --num-callers=30 \
		--suppressions=valgrind-python.supp python-dbg -tt -3 tests.py

valgrind-python.supp:
	wget http://svn.python.org/projects/python/trunk/Misc/valgrind-python.supp

inplace:
	python setup.py build_ext --inplace

# pylint: R=refactor, C0103 == Invalid name
lint: inplace
	pep8 --ignore=E1,W1 \
		discodop/*.py web/*.py && \
	pep8 --ignore=E1,W1,F,E901,E225,E227,E211 \
		discodop/*.pyx discodop/*.pxi && \
	cd web; pylint --indent-string='\t' --disable=R,C0103 ../discodop/*.py *.py
