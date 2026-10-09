"""Generic setup.py for Cython code."""
import glob
import os
import platform
import sys

from setuptools import Extension, setup
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext

DEBUG = os.environ.get("CYTHON_DEBUG", "0") == "1"
BUILD_BASE = os.path.join('build', 'debug' if DEBUG else 'normal')


class ModeBuild(build):
	"""Separate normal and debug outputs, including Python ABI directories."""

	def initialize_options(self):
		super().initialize_options()
		self.build_base = BUILD_BASE


class IncrementalBuildExt(build_ext):
	"""Keep editable extension builds in the project's build directory."""

	def finalize_options(self):
		if self.editable_mode:
			# Editable wheels normally use temporary build directories, which
			# discard the extensions and objects needed for incremental builds.
			cached_build = ModeBuild(self.distribution)
			cached_build.ensure_finalized()
			self.build_lib = cached_build.build_lib
			self.build_temp = cached_build.build_temp
		super().finalize_options()

	def run(self):
		super().run()
		# Cached Cython debugger metadata may name a previous environment.
		interpreter = os.path.join(BUILD_BASE, 'cython_debug', 'interpreter')
		if DEBUG and os.path.isfile(interpreter):
			with open(interpreter, 'w') as out:
				out.write(sys.executable)

	def copy_extensions_to_source(self):
		# Switching modes can select a cached binary older than the local one.
		# Force only the copy, so compilation still remains incremental.
		force = self.force
		self.force = True
		try:
			super().copy_extensions_to_source()
		finally:
			self.force = force


def macos_version():
	version = platform.mac_ver()[0]
	return tuple(map(int, version.split(".")))


# In releases, include C sources but not Cython sources; otherwise, use cython
# to figure out which files may need to be re-cythonized.
USE_CYTHON = os.path.exists('discodop/containers.pyx')
if USE_CYTHON:
	try:
		from Cython.Build import cythonize
		from Cython.Compiler import Options
	except ImportError as err:
		raise RuntimeError(f'could not import Cython: {err}')
cmdclass = {'build': ModeBuild, 'build_ext': IncrementalBuildExt}

# some of these directives increase performance,
# but at the cost of failing in mysterious ways.
directives = {
		'profile': False,
		'cdivision': True,
		'nonecheck': False,
		'wraparound': DEBUG,
		'boundscheck': DEBUG,
		'embedsignature': True,
		'warn.unused': True,
		'warn.unreachable': True,
		'warn.maybe_uninitialized': True,
		'warn.undeclared': False,
		'warn.unused_arg': False,
		'warn.unused_result': False,
		}
if USE_CYTHON:
	Options.fast_fail = True
	if 'auto_pickle' in Options.get_directive_defaults():
		directives.update(auto_pickle=False)

if __name__ == '__main__':
	if sys.version_info[:2] < (3, 3):  # noqa
		raise RuntimeError('Python version 3.3+ required.')
	os.environ['GCC_COLORS'] = 'auto'
	# Cython copies direct header dependencies beside generated sources, but
	# their transitive includes still need the original header directories.
	include_dirs = ['discodop', 'cpp-btree', 'gheap', 'sparsepp/sparsepp']
	extra_compile_args = ['-Isparsepp/',
			'-Wno-strict-prototypes', '-Wno-unused-function',
			'-Wno-unreachable-code', '-Wno-sign-compare',
			'-D__STDC_LIMIT_MACROS']  # http://stackoverflow.com/a/3233069
	if sys.platform == 'darwin' and macos_version() >= (10, 15):
		# https://github.com/andreasvc/disco-dop/issues/68
		extra_compile_args += ['-DCPP_BTREE_CXX11', '-DSPP_CXX11']

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
					include_dirs=include_dirs,
					language='c++',
					# include_dirs=[...],
					# libraries=[...],
					# library_dirs=[...],
					)],
				annotate=True,
				build_dir=os.path.join(BUILD_BASE, 'cython'),
				output_dir=BUILD_BASE,
				compiler_directives=directives,
				gdb_debug=DEBUG,
				language_level=3,
				# nthreads=4,
				)
	else:
		ext_modules = [Extension(
				os.path.splitext(filename)[0].replace('/', '.'),
				sources=[filename],
				extra_compile_args=extra_compile_args,
				extra_link_args=extra_link_args,
				include_dirs=include_dirs,
				)
				for filename in glob.glob('discodop/*.c')]
	setup(cmdclass=cmdclass, ext_modules=ext_modules)
