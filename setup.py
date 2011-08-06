from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

# some of these directives increase performance, but at the cost of failing
# in mysterious ways.
directives = {
	"boundscheck" : False,
	"profile" : False,
	"nonecheck" : False,
	"wraparound" : False,
	"embedsignature" : True
	}

ext_modules = [
	Extension("plcfrs",
			["plcfrs.pyx", "plcfrs.pxd"]
		),
	Extension("agenda",
			["agenda.pyx", "agenda.pxd"]
		),
	Extension("disambiguation",
			["disambiguation.py", "disambiguation.pxd"]
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
	Extension("oldplcfrs",
			["oldplcfrs.py", "oldplcfrs.pxd"]
		),
	Extension("fragmentseeker",
			["fragmentseeker.py", "fragmentseeker.pxd"]
		),
	Extension("coarsetofine",
			["coarsetofine.py", "coarsetofine.pxd"]
		)
	]

for e in ext_modules:
	e.pyrex_directives = directives

setup(
	name = 'disco-dop',
	cmdclass = {'build_ext': build_ext},
	include_dirs = [np.get_include(), '.'],
	ext_modules = ext_modules
)
