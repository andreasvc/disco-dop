from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

directives = {
	"boundscheck" : False,
	"profile" : False,
	"nonecheck" : False,
	"wraparound" : False,
	"embedsignature" : True
	}

ext_modules = [
	Extension("plcfrs_cython",
			["plcfrs_cython.pyx", "plcfrs_cython.pxd"],
			#extra_compile_args=["-g"],
			#extra_link_args=["-g"],
			include_dirs=['.']
		),
	Extension("cpq",
			["cpq.pyx", "cpq.pxd"]
		),
	Extension("containers",
			["containers.pyx", "containers.pxd"]
		),
	Extension("kbest",
			["kbest.py", "kbest.pxd"]
		),
	Extension("estimates",
			["estimates.py", "estimates.pxd"]
		),
	Extension("plcfrs",
			["plcfrs.py", "plcfrs.pxd"]
		),
	Extension("fragmentseeker",
			["fragmentseeker.py", "fragmentseeker.pxd"]
		)
	]

for e in ext_modules:
	e.pyrex_directives = directives

setup(
	name = 'plcfrs',
	cmdclass = {'build_ext': build_ext},
	ext_modules = ext_modules
)
