""" Run doctests and other tests from all modules. """
import sys
# do not import from current directory, since our compiled modules will only be
# installed system-wide or in the user's equivalent.
sys.path.pop(0)
from doctest import testmod, NORMALIZE_WHITESPACE, REPORT_NDIFF
from operator import itemgetter
from discodop import runexp
modules = """bit coarsetofine demos disambiguation estimates eval
		fragments _fragments gen grammar lexicon kbest plcfrs pcfg tree
		treedist treedraw treebank treebanktransforms treetransforms""".split()
modules = [__import__('discodop.%s' % mod, globals(), locals(), [mod])
		for mod in modules]
results = {}
for mod in modules:
	modname = str(getattr(mod, '__file__', mod))
	print('running doctests of %s' % modname)
	results[modname] = fail, attempted = testmod(mod, verbose=False,
			optionflags=NORMALIZE_WHITESPACE | REPORT_NDIFF)
	assert fail == 0, modname
if any(not attempted for fail, attempted in results.values()):
	print('no doctests: %s' % ' '.join(mod for mod, (_, attempted)
			in results.items() if not attempted))
for mod in modules:
	mod.test()
runexp.main(argv=['runexp.py', 'sample.prm'])
for mod, (fail, attempted) in sorted(results.items(),
		key=itemgetter(1)):
	if attempted:
		print('%s: %d doctests succeeded!' % (mod, attempted))
