"""Regression tests for incremental editable extension builds."""
import os
import runpy
import subprocess
import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Distribution, Extension


def build_config():
	return runpy.run_path(str(Path(__file__).resolve().parents[1] / 'setup.py'))


def test_editable_build_reuses_extensions(tmp_path, monkeypatch):
	build_ext = build_config()['IncrementalBuildExt']
	monkeypatch.chdir(tmp_path)
	source = tmp_path / 'example.c'
	source.write_text('''#include <Python.h>
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "example", NULL, -1, NULL
};
PyMODINIT_FUNC PyInit_example(void) { return PyModule_Create(&module); }
''')

	def compile_extension():
		dist = Distribution({'ext_modules': [Extension('example', ['example.c'])]})
		cmd = build_ext(dist)
		cmd.editable_mode = True
		# Emulate setuptools' editable wheel configuration.
		cmd.build_lib = str(tmp_path / 'wheel-lib')
		cmd.build_temp = str(tmp_path / 'wheel-temp')
		cmd.ensure_finalized()
		assert cmd.inplace
		assert cmd.build_lib.startswith('build/')
		assert cmd.build_temp.startswith('build/')
		cmd.run()
		return Path(cmd.build_lib) / cmd.get_ext_filename('example')

	output = compile_extension()
	mtime = output.stat().st_mtime_ns
	local_output = tmp_path / output.name
	assert local_output.read_bytes() == output.read_bytes()
	compile_extension()
	assert output.stat().st_mtime_ns == mtime
	# Restore a stale local binary, then change the extension's source.
	local_output.write_bytes(b'stale extension')
	os.utime(source, ns=(mtime + 2_000_000_000, mtime + 2_000_000_000))
	compile_extension()
	assert output.stat().st_mtime_ns != mtime
	assert local_output.read_bytes() == output.read_bytes()


def test_regular_build_keeps_frontend_directories(tmp_path):
	build_ext = build_config()['IncrementalBuildExt']
	cmd = build_ext(Distribution())
	cmd.build_lib = str(tmp_path / 'lib')
	cmd.build_temp = str(tmp_path / 'temp')
	cmd.ensure_finalized()
	assert cmd.build_lib == str(tmp_path / 'lib')
	assert cmd.build_temp == str(tmp_path / 'temp')
	assert not cmd.inplace


def test_regular_build_separates_modes(monkeypatch):
	for debug, mode in ((False, 'normal'), (True, 'debug')):
		monkeypatch.setenv('CYTHON_DEBUG', str(int(debug)))
		config = build_config()
		dist = Distribution({'ext_modules': [Extension('example', ['example.c'])],
				'cmdclass': config['cmdclass']})
		cmd = dist.get_command_obj('build_ext')
		cmd.ensure_finalized()
		assert cmd.build_lib.startswith(f'build/{mode}/lib.')
		assert cmd.build_temp.startswith(f'build/{mode}/temp.')
		assert sys.implementation.cache_tag in cmd.build_temp


def test_generated_source_build_without_cython(monkeypatch):
	exists = os.path.exists
	monkeypatch.setattr(os.path, 'exists',
			lambda path: False if path == 'discodop/containers.pyx' else exists(path))
	config = build_config()
	assert not config['USE_CYTHON']
	assert config['cmdclass']['build_ext'] is config['IncrementalBuildExt']
	cmd = config['IncrementalBuildExt'](Distribution())
	cmd.editable_mode = True
	cmd.ensure_finalized()
	cached_build = config['ModeBuild'](cmd.distribution)
	cached_build.ensure_finalized()
	assert cmd.build_lib == cached_build.build_lib
	assert cmd.build_temp == cached_build.build_temp


def test_switch_build_modes_without_recompiling(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	Path('vendor').mkdir()
	Path('vendor/inner.h').write_text('static int header_value(void) { return 42; }\n')
	Path('vendor/wrapper.h').write_text('#include "inner.h"\n')
	Path('example.pyx').write_text('''cdef extern from *:
    """
    #ifdef MODE_DEBUG
    static int build_mode(void) { return 1; }
    #else
    static int build_mode(void) { return 0; }
    #endif
    """
    int build_mode()

cdef extern from "vendor/wrapper.h":
    int header_value()

def mode():
    return build_mode()

def first(list values):
    return values[0]

def header():
    return header_value()
''')

	def compile_mode(debug):
		monkeypatch.setenv('CYTHON_DEBUG', str(int(debug)))
		config = build_config()
		extensions = cythonize(
				[Extension('example', ['example.pyx'],
					include_dirs=['.', 'vendor'],
					extra_compile_args=['-DMODE_DEBUG'] if debug else [])],
				build_dir=os.path.join(config['BUILD_BASE'], 'cython'),
				output_dir=config['BUILD_BASE'],
				compiler_directives=config['directives'],
				gdb_debug=debug, language_level=3)
		dist = Distribution({'ext_modules': extensions, 'cmdclass': config['cmdclass']})
		cmd = dist.get_command_obj('build_ext')
		cmd.editable_mode = True
		cmd.ensure_finalized()
		interpreter = Path(config['BUILD_BASE'], 'cython_debug', 'interpreter')
		if debug:
			interpreter.write_text('/previous/environment/python')
		cmd.run()
		if debug:
			assert interpreter.read_text() == sys.executable
		assert not cmd.force
		result = subprocess.check_output(
				[sys.executable, '-c',
					'import example; assert example.header() == 42; print(example.mode())'],
				text=True)
		assert int(result) == int(debug)
		generated = Path(extensions[0].sources[0])
		binary = Path(cmd.build_lib) / cmd.get_ext_filename('example')
		return generated, binary

	normal_source, normal_binary = compile_mode(False)
	normal_mtimes = (normal_source.stat().st_mtime_ns,
			normal_binary.stat().st_mtime_ns)
	debug_source, debug_binary = compile_mode(True)
	debug_mtimes = (debug_source.stat().st_mtime_ns,
			debug_binary.stat().st_mtime_ns)
	assert normal_source != debug_source
	assert normal_binary != debug_binary
	assert normal_source.read_bytes() != debug_source.read_bytes()
	assert Path('build/debug/cython_debug/cython_debug_info_example').is_file()
	# A newer local debug binary must still be replaced by the older normal one.
	local_binary = Path(debug_binary.name)
	os.utime(local_binary, ns=(debug_mtimes[1] + 2_000_000_000,) * 2)
	compile_mode(False)
	assert (normal_source.stat().st_mtime_ns,
			normal_binary.stat().st_mtime_ns) == normal_mtimes
	compile_mode(True)
	assert (debug_source.stat().st_mtime_ns,
			debug_binary.stat().st_mtime_ns) == debug_mtimes
