""" Run doctests and other tests from all modules. """
from doctest import testmod, NORMALIZE_WHITESPACE, REPORT_NDIFF
from operator import itemgetter
import sys
import discodop.runexp
from discodop import bit, demos, kbest, grammar, treebank, estimates, pcfg, \
		fragments, _fragments, plcfrs, agenda, coarsetofine, eval, gen, \
		disambiguation, lexicon, tree, treetransforms, treedist, treedraw
modules = (agenda, bit, coarsetofine, demos, disambiguation, estimates,
		eval, fragments, _fragments, gen, grammar, lexicon, kbest, plcfrs,
		pcfg, tree, treebank, treedist, treedraw, treetransforms)
results = {}
for mod in modules:
	print('running doctests of %s' % mod.__file__)
	results[mod] = fail, attempted = testmod(mod, verbose=False,
		optionflags=NORMALIZE_WHITESPACE | REPORT_NDIFF)
	assert fail == 0, mod.__file__
if any(not attempted for fail, attempted in results.values()):
	print('no doctests: %s' % ' '.join(mod.__file__
			for mod, (_, attempted) in results.items() if not attempted))
for mod in modules:
	if hasattr(mod, 'test'):
		mod.test()
	else:
		mod.main()
discodop.runexp.main(argv="runexp.py sample.prm".split())
for mod, (fail, attempted) in sorted(results.items(),
		key=itemgetter(1)):
	if attempted:
		print('%s: %d doctests succeeded!' % (mod.__file__, attempted))
