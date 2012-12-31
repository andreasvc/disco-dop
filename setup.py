""" Generic setup.py for Cython code. """
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy

# some of these directives increase performance,
# but at the cost of failing in mysterious ways.
cython_directives = dict(
	profile=True,
	cdivision=True,
	nonecheck=False,
	wraparound=False,
	boundscheck=False,
	embedsignature=True,
	language_level=3
)

#Options.fast_fail = True
Options.extra_compile_args = ["-O3"]
Options.extra_link_args = ["-O3"]  #["-g"],
if __name__ == '__main__':
	setup(
		name = 'disco-dop',
		include_dirs = [numpy.get_include()], #cythonutils],
		ext_modules = cythonize(
			[Extension('*', ['*.pyx'],
				extra_compile_args=["-O3"],
				extra_link_args=["-O3"], #["-g"],
			)],
			nthreads=4,
			annotate=True,
			cython_directives=cython_directives,
			)
	)
