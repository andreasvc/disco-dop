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
	py.test --doctest-modules discodop/ tests/unittests.py \
	&& python -tt -3 tests.py \
	&& cd tests/ \
	&& sh run.sh

test3: clean
	python3 setup.py install --user
	cp build/lib.*/discodop/*.so discodop/
	PYTHONIOENCODING=utf-8 python3 -bb -tt tests.py

inplace: discodop
	# python setup.py build_ext --inplace
	cp build/lib.*/discodop/*.so discodop/

install: discodop docs

debug:
	# NB: debug build requires all external modules to be compiled
	# with debug symbols as well (e.g., install python-numpy-dbg package)
	python-dbg setup.py build_ext --inplace --debug --pyrex-gdb

debugvalgrind: debug inplace
	valgrind --tool=memcheck --leak-check=full --num-callers=30 \
		--suppressions=valgrind-python.supp --show-leak-kinds=definite \
		python-dbg tests.py

valgrind: inplace
	valgrind --tool=memcheck --leak-check=full --num-callers=30 \
		--suppressions=valgrind-python.supp --show-leak-kinds=definite \
		python tests.py

# pylint: R=refactor, C0103 == Invalid name
lint: inplace
	# Any files with more than 999 lines?
	cd discodop; wc -l *.py *.pyx *.pxi *.pxd | egrep '[0-9]{4,}'
	# Docstrings without single line summaries?
	cd discodop; egrep -n '""".*[^.\"\\)]$$' *.pxd *.pyx *.py || echo 'none!'
	pep8 --ignore=E1,W1 \
		discodop/*.py web/*.py tests/*.py \
	&& pep8 --ignore=E1,W1,F,E901,E225,E227,E211 \
		discodop/*.pyx discodop/*.pxi \
	&& pylint --indent-string='\t' \
		--disable=R,bad-continuation,invalid-name,star-args \
		discodop/*.py web/*.py tests/*.py
