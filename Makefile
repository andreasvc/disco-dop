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
	python3 setup.py build --with-cython
	cp build/lib.*/discodop/*.so discodop/

py3install: discodop
	python3 setup.py install --user --with-cython

py2:
	python2 setup.py install --user --with-cython
	cp build/lib.*/discodop/*.so discodop/

py2install: py2
	python2 setup.py install --user --with-cython

docs:
	mkdir -p ~/.local/man/man1
	cd docs && make man && cp _build/man/* ~/.local/man/man1/
	cd docs && make html

install: py3install docs

debug:
	# NB: debug build requires all external modules to be compiled
	# with debug symbols as well (e.g., install python-numpy-dbg package)
	python-dbg setup.py build_ext --inplace --debug --pyrex-gdb

debugvalgrind: debug discodop
	valgrind --tool=memcheck --leak-check=full --num-callers=30 \
		--suppressions=valgrind-python.supp --show-leak-kinds=definite \
		python-dbg tests.py

valgrind: discodop
	valgrind --tool=memcheck --leak-check=full --num-callers=30 \
		--suppressions=valgrind-python.supp --show-leak-kinds=definite \
		python tests.py

test: py3install
	python3 `which py.test` --doctest-modules discodop/ tests/unittests.py \
	&& python3 -bb -tt tests.py \
	&& cd tests/ && sh run.sh

test2: py2install
	python2 setup.py install --user --with-cython
	cp build/lib.*/discodop/*.so discodop/
	PYTHONIOENCODING=utf-8 PYTHONHASHSEED=42 python2 -bb -tt -3 tests.py

# pylint: R=refactor, C0103 == Invalid name
lint: discodop
	# Any files with more than 999 lines?
	cd discodop; wc -l *.py *.pyx *.pxi *.pxd | egrep '[0-9]{4,}' | sort -n
	# Docstrings without single line summaries?
	cd discodop; egrep -n '""".*[^.\"\\)]$$' *.pxd *.pyx *.py || echo 'none!'
	pep8 --ignore=E1,W1 \
		discodop/*.py web/*.py tests/*.py \
	&& pep8 --ignore=E1,W1,F,E901,E225,E227,E211 \
		discodop/*.pyx discodop/*.pxi discodop/*.pxd \
	&& python3 `which pylint` --indent-string='\t' --max-line-length=80 \
		--disable=I,R,bad-continuation,invalid-name,star-args,wrong-import-position,wrong-import-order,ungrouped-imports \
		--enable=cyclic-import \
		--extension-pkg-whitelist=discodop,faulthandler,alpinocorpus,numpy,roaringbitmap \
		discodop/*.py web/*.py tests/*.py

lint2: py2
	python2 `which pylint` --indent-string='\t' --max-line-length=80 \
		--disable=I,R,bad-continuation,invalid-name,star-args,wrong-import-position,wrong-import-order,ungrouped-imports \
		--enable=cyclic-import \
		--extension-pkg-whitelist=discodop,faulthandler,alpinocorpus,numpy,roaringbitmap \
		discodop/*.py web/*.py tests/*.py
