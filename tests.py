""" Run doctests and other tests from all modules. """
from doctest import testmod, NORMALIZE_WHITESPACE, REPORT_NDIFF
from operator import itemgetter
import sys
from discodop import runexp
modules = """agenda bit coarsetofine demos disambiguation estimates eval
		fragments _fragments gen grammar lexicon kbest plcfrs pcfg tree
		treebank treedist treedraw treetransforms""".split()
modules = [__import__('discodop.%s' % mod, globals(), locals(), [mod])
		for mod in modules]
results = {}
for mod in modules:
	print('running doctests of %s' % mod.__file__)
	results[mod.__file__] = fail, attempted = testmod(mod, verbose=False,
		optionflags=NORMALIZE_WHITESPACE | REPORT_NDIFF)
	assert fail == 0, mod.__file__
if any(not attempted for fail, attempted in results.values()):
	print('no doctests: %s' % ' '.join(mod for mod, (_, attempted)
			in results.items() if not attempted))
for mod in modules:
	if hasattr(mod, 'test'):
		mod.test()
	else:
		mod.main()
runexp.main(argv=['runexp.py', 'sample.prm'])
for mod, (fail, attempted) in sorted(results.items(),
		key=itemgetter(1)):
	if attempted:
		print('%s: %d doctests succeeded!' % (mod, attempted))
