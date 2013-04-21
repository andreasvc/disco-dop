""" Generic setup.py for Cython code. """
from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
#from Cython.Compiler import Options
import numpy

# some of these directives increase performance,
# but at the cost of failing in mysterious ways.
directives = dict(
		profile=False,
		cdivision=True,
		nonecheck=False,
		wraparound=False,
		boundscheck=False,
		embedsignature=True,
)

#Options.fast_fail = True
#Options.extra_compile_args = ["-O3"]
#Options.extra_link_args = ["-O3"]  #["-g"],
if __name__ == '__main__':
	setup(name = 'disco-dop',
			include_dirs = [numpy.get_include()],
			cmdclass = dict(build_ext=build_ext),
			ext_modules = cythonize('*.pyx',
					nthreads=4,
					annotate=True,
					#language_level=3, # FIXME make this work ...
					compiler_directives=directives,
			)
	)
