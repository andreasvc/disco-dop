""" Run doctests and other tests from all modules. """
import os
import sys
from doctest import testmod, NORMALIZE_WHITESPACE, REPORT_NDIFF
from operator import itemgetter

MODULES = """bit coarsetofine demos disambiguation estimates eval fragments
		_fragments functiontags gen grammar heads lexicon kbest plcfrs pcfg
		punctuation tree treedist treedraw treebank treebanktransforms
		treetransforms runexp""".split()
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
	if hasattr(mod, 'test'):
		mod.test()
for modname, (fail, attempted) in sorted(results.items(), key=itemgetter(1)):
	if attempted:
		print('%s: %d doctests succeeded!' % (modname, attempted))
