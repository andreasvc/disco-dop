from distutils.core import setup
from Cython.Build import cythonize
import os, cython, numpy

# some of these directives increase performance,
# but at the cost of failing in mysterious ways.
directives = {
	"profile" : False,
	"nonecheck" : False,
	"cdivision" : True,
	"wraparound" : False,
	"boundscheck" : False,
	"embedsignature" : True,
	#"extra_compile_args" : ["-O3"],
	#extra_link_args=["-g"]
	}

# this directory includes 'arrayarray.h', which we include manually
# to get around a problem where it's not loaded early enough.
cythonutils = os.path.join(os.path.split(
	cython.__file__)[0], 'Cython', 'Utility')

setup(
	name = 'disco-dop',
	include_dirs = [numpy.get_include(), cythonutils],
	ext_modules = cythonize('*.pyx',
		pyrex_directives=directives,
		directives=directives,
		**directives)
)
