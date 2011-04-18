from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
	Extension("plcfrs_cython", 
			["plcfrs_cython.pyx"],
			include_dirs=['.'],
			library_dirs=['.']),
	Extension("estimates",
		["estimates.py", "estimates.pxd"]),
	Extension("plcfrs",
		["plcfrs.py", "plcfrs.pxd"])
	]

setup(
	name = 'plcfrs',
	cmdclass = {'build_ext': build_ext},
	ext_modules = ext_modules
)
