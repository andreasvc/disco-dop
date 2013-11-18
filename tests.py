""" Run doctests and other tests from all modules. """
import os
import sys
from doctest import testmod, NORMALIZE_WHITESPACE, REPORT_NDIFF
from operator import itemgetter
# do not import from current directory, since our compiled modules will only be
# installed system-wide or in the user's equivalent.
sys.path.pop(0)
MODULES = """bit coarsetofine demos disambiguation estimates eval fragments
		_fragments gen grammar lexicon kbest plcfrs pcfg tree treedist treedraw
		treebank treebanktransforms treetransforms runexp""".split()
MODULES = [__import__('discodop.%s' % mod, globals(), locals(), [mod])
		for mod in MODULES]

results = {}
for mod in MODULES:
	modname = str(getattr(mod, '__file__', mod))
	if not modname.endswith('.so'):
		continue
	print('running doctests of %s' % modname)
	results[modname] = fail, attempted = testmod(mod, verbose=False,
			optionflags=NORMALIZE_WHITESPACE | REPORT_NDIFF)
	assert fail == 0, modname
for mod in MODULES:
	mod.test()
for modname, (fail, attempted) in sorted(results.items(), key=itemgetter(1)):
	if attempted:
		print('%s: %d doctests succeeded!' % (modname, attempted))
