from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

# some of these directives increase performance,
# but at the cost of failing in mysterious ways.
directives = {
	"profile" : False,
	"boundscheck" : False,
	"nonecheck" : False,
	"wraparound" : False,
	"embedsignature" : True
	}

ext_modules = [
	Extension("bit",				["bit.pyx"]),
	Extension("kbest",				["kbest.py", "kbest.pxd"]),
	Extension("agenda",				["agenda.pyx", "agenda.pxd"]),
	Extension("plcfrs",				["plcfrs.pyx",  "plcfrs.pxd"]),
	Extension("estimates",			["estimates.py", "estimates.pxd"]),
	Extension("oldplcfrs",			["oldplcfrs.py",  "oldplcfrs.pxd"]),
	Extension("containers",			["containers.pyx", "containers.pxd"]),
	Extension("coarsetofine",		["coarsetofine.py", "coarsetofine.pxd"]),
	Extension("disambiguation",		["disambiguation.py", "disambiguation.pxd"]),
	Extension("_fragmentseeker",	["_fragmentseeker.pyx", "_fragmentseeker.pxd"],
	)]
	#extra_compile_args=["-g", "-O0"], extra_link_args=["-g"],)]

for e in ext_modules:
	e.pyrex_directives = directives

setup(
	name = 'disco-dop',
	cmdclass = {'build_ext': build_ext},
	include_dirs = [np.get_include(), '.'],
	ext_modules = ext_modules
)
