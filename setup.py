"""Generic setup.py for Cython code."""
import os
import sys
import glob
try:
	from setuptools import Extension, setup
	SETUPTOOLS = True
except ImportError:
	from distutils.core import Extension, setup
	SETUPTOOLS = False

from discodop import __version__

# In releases, include C sources but not Cython sources; otherwise, use cython
# to figure out which files may need to be re-cythonized.
USE_CYTHON = os.path.exists('discodop/containers.pyx')
if USE_CYTHON:
	try:
		from Cython.Build import cythonize
		from Cython.Distutils import build_ext
		from Cython.Compiler import Options
	except ImportError as err:
		raise RuntimeError('could not import Cython.')
	cmdclass = dict(build_ext=build_ext)
else:
	cmdclass = dict()

DEBUG = '--debug' in sys.argv
if DEBUG:
	sys.argv.remove('--debug')

with open('README.rst') as inp:
	README = inp.read()

REQUIRES = [
		'numpy',  # '>=1.6.1',
		'roaringbitmap',  # '>=0.4',
		]
if USE_CYTHON:
	REQUIRES.append('cython')  # '>=0.21'
METADATA = dict(name='disco-dop',
		version=__version__,
		description='Discontinuous Data-Oriented Parsing',
		long_description=README,
		author='Andreas van Cranenburgh',
		author_email='A.W.vanCranenburgh@uva.nl',
		url='http://discodop.readthedocs.io',
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
		requires=REQUIRES,
		packages=['discodop'],
	)
if SETUPTOOLS:
	METADATA['install_requires'] = REQUIRES
	METADATA['entry_points'] = {
			'console_scripts': ['discodop = discodop.cli:main']}
else:
	METADATA['scripts'] = ['bin/discodop']

# some of these directives increase performance,
# but at the cost of failing in mysterious ways.
directives = {
		'profile': False,
		'cdivision': True,
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
Options.fast_fail = True
if 'auto_pickle' in Options.get_directive_defaults():
	directives.update(auto_pickle=False)

if __name__ == '__main__':
	if sys.version_info[:2] < (3, 3):
		raise RuntimeError('Python version 3.3+ required.')
	os.environ['GCC_COLORS'] = 'auto'
	extra_compile_args = ['-Isparsepp/',
			'-Wno-strict-prototypes', '-Wno-unused-function',
			'-Wno-unreachable-code', '-Wno-sign-compare',
			'-D__STDC_LIMIT_MACROS']  # http://stackoverflow.com/a/3233069
	if DEBUG:
		directives.update(wraparound=True, boundscheck=True)
		extra_compile_args += ['-g', '-O0',
				# '-fsanitize=address', '-fsanitize=undefined',
				'-fno-omit-frame-pointer']
		extra_link_args = ['-g']
	else:
		extra_compile_args += ['-O3', '-march=native', '-DNDEBUG']
		extra_link_args = ['-DNDEBUG']
	if USE_CYTHON:
		ext_modules = cythonize(
				[Extension(
					'*',
					['discodop/*.pyx'],
					extra_compile_args=extra_compile_args,
					extra_link_args=extra_link_args,
					language='c++',
					# include_dirs=[...],
					# libraries=[...],
					# library_dirs=[...],
					)],
				annotate=True,
				compiler_directives=directives,
				language_level=3,
				# nthreads=4,
				)
	else:
		ext_modules = [Extension(
				os.path.splitext(filename)[0].replace('/', '.'),
				sources=[filename],
				extra_compile_args=extra_compile_args,
				extra_link_args=extra_link_args,
				)
				for filename in glob.glob('discodop/*.c')]
	setup(
			cmdclass=cmdclass,
			ext_modules=ext_modules,
			**METADATA)
