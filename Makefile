.PHONY: all clean install test testdebug lint docs discodop debug debugvalgrind valgrind

PYTHON3 ?= python3
PYTHON3_DEBUG ?= python3-dbg
PIP_INSTALL_FLAGS ?= --user

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
	CYTHON_DEBUG=0 $(PYTHON3) -m pip install $(PIP_INSTALL_FLAGS) --no-build-isolation --editable .

docs:
	mkdir -p ~/.local/man/man1
	cd docs && make man && cp _build/man/* ~/.local/man/man1/
	cd docs && make html

install: discodop docs

test: discodop
	$(PYTHON3) -bb -tt -m pytest --doctest-modules discodop/ tests/

debug:
	# NB: debug build requires all external modules to be compiled
	# with debug symbols as well (e.g., install python-numpy-dbg package)
	CYTHON_DEBUG=1 $(PYTHON3_DEBUG) -m pip install $(PIP_INSTALL_FLAGS) --no-build-isolation --editable .

debugvalgrind: debug
	valgrind --tool=memcheck --leak-check=full --num-callers=30 \
		--suppressions=valgrind-python.supp --show-leak-kinds=definite \
		$(PYTHON3_DEBUG) -bb -tt -m pytest --doctest-modules discodop/ tests/

valgrind: discodop
	valgrind --tool=memcheck --leak-check=full --num-callers=30 \
		--suppressions=valgrind-python.supp --show-leak-kinds=definite \
		$(PYTHON3) -m pytest tests/test_modules.py

# pylint: R=refactor, I=informational, e.g. "I0011: Locally disabling %s"
# pycodestyle: W503 line break before binary operator; E402 module level import not at top of file
lint: discodop
	# Any files with more than 999 lines?
	cd discodop; wc -l *.py *.pyx *.pxi *.pxd | egrep '[0-9]{4,}' | sort -n
	# Docstrings without single line summaries?
	cd discodop; egrep -n '""".*[^].\"\\)]$$' *.pxd *.pyx *.py || echo 'none!'
	pycodestyle --ignore=E1,W1,E402,E501,W503 \
		discodop/*.py web/*.py tests/*.py \
	&& pycodestyle --ignore=F,W1,W503,E1,E211,E225,E226,E227,E402,E901 \
		discodop/*.pyx discodop/*.pxi discodop/*.pxd \
	&& $(PYTHON3) -m pylint --indent-string='\t' --max-line-length=80 \
		--disable=I,R,bad-continuation,invalid-name,star-args,wrong-import-position,wrong-import-order,ungrouped-imports,import-outside-toplevel,superfluous-parens \
		--enable=cyclic-import \
		--extension-pkg-whitelist=discodop,faulthandler,roaringbitmap \
		discodop/*.py web/*.py tests/*.py
