from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
	Extension("plcfrs_cython",
			["plcfrs_cython.pyx", "plcfrs_cython.pxd"],
			#extra_compile_args=["-g"],
			#extra_link_args=["-g"],
			include_dirs=['.'],
			library_dirs=['.']
		),
	Extension("estimates",
			["estimates.py", "estimates.pxd"],
		),
	Extension("kbest",
		["kbest.py", "kbest.pxd"]
		),
	Extension("plcfrs",
			["plcfrs.py", "plcfrs.pxd"],
		),
	Extension("cpq",
			["cpq.pyx", "cpq.pxd"],
		),
	Extension("containers",
			["containers.pyx", "containers.pxd"],
		)
	]

setup(
	name = 'plcfrs',
	cmdclass = {'build_ext': build_ext},
	ext_modules = ext_modules
)
