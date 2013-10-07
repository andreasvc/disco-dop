""" Generic setup.py for Cython code. """
from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from Cython.Compiler import Options
import numpy

metadata = dict(name='disco-dop',
		version='0.4pre1',
		description='Discontinuous Data-Oriented Parsing',
		long_description=open('README.rst').read(),
		author='Andreas van Cranenburgh',
		author_email='A.W.vanCranenburgh@uva.nl',
		url='https://github.com/andreasvc/disco-dop/',
		classifiers=[
				'Development Status :: 4 - Beta',
				'Environment :: Console',
				'Environment :: Web Environment',
				'Intended Audience :: Science/Research',
				'License :: OSI Approved :: GNU General Public License (GPL)',
				'Operating System :: POSIX',
				'Programming Language :: Python',
				'Programming Language :: Python :: 3',
				'Programming Language :: Cython',
				'Topic :: Text Processing :: Linguistic'],
		requires=[
				'cython (>=0.18)',
				'numpy',
				'lxml',
		],
		packages=['discodop'],
		scripts=['bin/discodop'],
)

# some of these directives increase performance,
# but at the cost of failing in mysterious ways.
directives = {
		'profile': False,
		'cdivision': True,
		'nonecheck': False,
		'wraparound': False,
		'boundscheck': False,
		'embedsignature': True,
		'warn.unreachable': True,
		'warn.unused': True,
		'warn.maybe_uninitialized': True,
		'warn.undeclared': False,
		'warn.unused_arg': False,
		'warn.unused_result': False,
		}

Options.fast_fail = True
#Options.extra_compile_args = ["-O3"]
#Options.extra_link_args = ["-O3"]  #["-g"],
if __name__ == '__main__':
	setup(
			include_dirs = [numpy.get_include()],
			cmdclass = dict(build_ext=build_ext),
			ext_modules = cythonize(
					'discodop/*.pyx',
					#nthreads=4,
					annotate=True,
					compiler_directives=directives,
					#language_level=3, # FIXME make this work ...
			),
			#test_suite = 'tests'
			**metadata)
