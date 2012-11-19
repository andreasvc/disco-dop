from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import os, cython, numpy

# some of these directives increase performance,
# but at the cost of failing in mysterious ways.
cython_directives = dict(
	profile=False,
	cdivision=True,
	nonecheck=False,
	wraparound=False,
	boundscheck=False,
	embedsignature=True,
)

# this directory includes 'arrayarray.h', which we include manually
# to get around a problem where it's not loaded early enough.
cythonutils = os.path.join(os.path.split(
	cython.__file__)[0], 'Cython', 'Utility')

Options.extra_compile_args=["-O3"],
Options.extra_link_args=["-O3"], #["-g"],
setup(
	name = 'disco-dop',
	include_dirs = [numpy.get_include()], #cythonutils],
	ext_modules = cythonize(
		[Extension('*', ['*.pyx'],
			extra_compile_args=["-O3"],
			extra_link_args=["-O3"], #["-g"],
		)],
		nthreads=4,
		cython_directives=cython_directives,
		)
)
