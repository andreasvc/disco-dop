from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

# some of these directives increase performance,
# but at the cost of failing in mysterious ways.
directives = {
	"profile" : False,
	"nonecheck" : False,
	"cdivision" : True,
	"wraparound" : False,
	"boundscheck" : False,
	"embedsignature" : True }

ext_modules = [
	Extension("kbest",           ["kbest.pyx"]),
	Extension("parser",          ["parser.pyx"]),
	Extension("estimates",       ["estimates.pyx"]),
	Extension("_fragments",      ["_fragments.pyx"]),
	Extension("coarsetofine",    ["coarsetofine.pyx"]),
	Extension("disambiguation",  ["disambiguation.pyx"]),
	Extension("bit",             ["bit.pyx", "bit.pxd"]),
	Extension("agenda",          ["agenda.pyx", "agenda.pxd"]),
	Extension("containers",      ["containers.pyx", "containers.pxd"]),
	]

for e in ext_modules:
	e.pyrex_directives = directives

setup(
	name = 'disco-dop',
	cmdclass = {'build_ext': build_ext},
	include_dirs = [numpy.get_include(), '.'],
	ext_modules = ext_modules
)
