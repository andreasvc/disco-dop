.PHONY: clean install test testdebug lint docs discodop py2

all: discodop

clean:
	rm -rf build/
	find discodop -name '*.c' -delete
	find discodop -name '*.cpp' -delete
	find discodop -name '*.so' -delete
	find discodop -name '*.pyc' -delete
	find discodop -name '*.html' -delete
	rm -rf */__pycache__
	cd docs && make clean

discodop:
	python3 setup.py install --user
	cp build/lib.*/discodop/*.so discodop/

docs:
	mkdir -p ~/.local/man/man1
	cd docs && make man && cp _build/man/* ~/.local/man/man1/
	cd docs && make html

install: discodop docs

debug:
	# NB: debug build requires all external modules to be compiled
	# with debug symbols as well (e.g., install python-numpy-dbg package)
	python3-dbg setup.py install --user --debug
	cp build/lib.*/discodop/*.so discodop/

debug35:
	python3.5-dbg setup.py install --user --debug
	cp build/lib.*/discodop/*.so discodop/

debugvalgrind: debug35
	valgrind --tool=memcheck --leak-check=full --num-callers=30 \
		--suppressions=valgrind-python.supp --show-leak-kinds=definite \
		python3.5-dbg `which py.test` --doctest-modules discodop/ tests/unittests.py
	valgrind --tool=memcheck --leak-check=full --num-callers=30 \
		--suppressions=valgrind-python.supp --show-leak-kinds=definite \
		python3.5-dbg -bb -tt tests.py

valgrind: discodop
	valgrind --tool=memcheck --leak-check=full --num-callers=30 \
		--suppressions=valgrind-python.supp --show-leak-kinds=definite \
		python tests.py

test: discodop
	python3 `which py.test` --doctest-modules discodop/ tests/unittests.py \
	&& python3 -bb -tt tests.py \
	&& cd tests/ && sh run.sh

# pylint: R=refactor, C0103 == Invalid name
# pep8: W503 line break before binary operator; E402 module level import not at top of file
lint: discodop
	# Any files with more than 999 lines?
	cd discodop; wc -l *.py *.pyx *.pxi *.pxd | egrep '[0-9]{4,}' | sort -n
	# Docstrings without single line summaries?
	cd discodop; egrep -n '""".*[^].\"\\)]$$' *.pxd *.pyx *.py || echo 'none!'
	pep8 --ignore=E1,W1,E402,W503 \
		discodop/*.py web/*.py tests/*.py \
	&& pep8 --ignore=F,W1,W503,E1,E211,E225,E226,E227,E402,E901 \
		discodop/*.pyx discodop/*.pxi discodop/*.pxd \
	&& python3 `which pylint` --indent-string='\t' --max-line-length=80 \
		--disable=I,R,bad-continuation,invalid-name,star-args,wrong-import-position,wrong-import-order,ungrouped-imports,superfluous-parens \
		--enable=cyclic-import \
		--extension-pkg-whitelist=discodop,faulthandler,alpinocorpus,numpy,roaringbitmap \
		discodop/*.py web/*.py tests/*.py
