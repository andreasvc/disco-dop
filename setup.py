"""Generic setup.py for Cython code."""
import os
import sys
from distutils.core import setup
from distutils.extension import Extension
try:
	from Cython.Build import cythonize
	from Cython.Distutils import build_ext
	havecython = True
except ImportError as err:
	print(err)
	havecython = False
from discodop import __version__

DEBUG = False

requires = [
		'cython (>=0.21)',
		'numpy (>=1.6.1)',
		'roaringbitmap',
		'cyordereddict',
		'pytest',
		'sphinx',
		]
if sys.version_info[0] == 2:
	requires.extend([
		'futures',
		'faulthandler',
		])
metadata = dict(name='disco-dop',
		version=__version__,
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
				'Programming Language :: Python :: 2.7',
				'Programming Language :: Python :: 3.3',
				'Programming Language :: Cython',
				'Topic :: Text Processing :: Linguistic',
		],
		requires=requires,
		packages=['discodop'],
		scripts=['bin/discodop'],
)

# some of these directives increase performance,
# but at the cost of failing in mysterious ways.
directives = {
		'profile': False,
		'cdivision': True,
		'fast_fail': True,
		'nonecheck': False,
		'wraparound': False,
		'boundscheck': False,
		'embedsignature': True,
		'warn.unused': True,
		'warn.unreachable': True,
		'warn.maybe_uninitialized': True,
		'warn.undeclared': False,
		'warn.unused_arg': False,
		'warn.unused_result': False,
		}

if __name__ == '__main__':
	if havecython:
		os.environ['GCC_COLORS'] = 'auto'
		if DEBUG:
			directives.update(wraparound=True, boundscheck=True)
			extensions = [Extension(
					'*',
					['discodop/*.pyx'],
					extra_compile_args=['-g', '-O0'],
					extra_link_args=['-g'],
					)]
		else:
			extensions = [Extension(
					'*',
					['discodop/*.pyx'],
					extra_compile_args=['-O3', '-march=native', '-DNDEBUG'],
					extra_link_args=['-DNDEBUG'],
					# include_dirs=[...],
					# libraries=[...],
					# library_dirs=[...],
					)]
		setup(
				cmdclass=dict(build_ext=build_ext),
				ext_modules=cythonize(
						extensions,
						annotate=True,
						compiler_directives=directives,
						language_level=3,
						# nthreads=4,
				),
				# test_suite = 'tests'
				**metadata)
	else:
		setup(**metadata)
		print('\nWarning: Cython not found.\n')
