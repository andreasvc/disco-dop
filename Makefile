all:
	python setup.py build_ext --inplace
clean:
	rm -f *.c *.so
