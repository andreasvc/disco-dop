.PHONY: clean install test testdebug lint docs discodop

all: discodop

clean:
	rm -rf build/
	find discodop -name '*.c' -delete
	find discodop -name '*.so' -delete
	find discodop -name '*.pyc' -delete
	find discodop -name '*.html' -delete
	rm -rf discodop/__pycache__
	cd docs && make clean

discodop:
	python setup.py install --user

docs:
	mkdir -p ~/.local/man/man1
	cd docs && make man && cp _build/man/discodop.1 ~/.local/man/man1/
	cd docs && make html

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

install: discodop docs

# pylint: R=refactor, C0103 == Invalid name
lint: inplace
	#Any files with more than 999 lines?
	cd discodop; wc -l *.py *.pyx *.pxi *.pxd | egrep '[0-9]{4,}'
	#Docstrings without single line summaries?
	cd discodop; egrep -n '""".*[^.\"\\)]$$' *.pxd *.pyx *.py || echo 'none!'
	pep8 --ignore=E1,W1 \
		discodop/*.py web/*.py tests/*.py && \
	pep8 --ignore=E1,W1,F,E901,E225,E227,E211 \
		discodop/*.pyx discodop/*.pxi && \
	pylint --indent-string='\t' --disable=R,C0103 discodop/*.py web/*.py tests/*.py
