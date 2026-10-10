"""Discover module self-tests and doctests in compiled extensions."""
import doctest
from importlib import import_module
from importlib.machinery import EXTENSION_SUFFIXES
from pkgutil import walk_packages

import pytest

import discodop

MODULES = [import_module(info.name) for info in walk_packages(
		discodop.__path__, discodop.__name__ + '.')]
SELFTEST_MODULES = [module for module in MODULES
		if callable(getattr(module, 'test', None))]
CYTHON_DOCTEST_MODULES = [module for module in MODULES
		if module.__file__.endswith(tuple(EXTENSION_SUFFIXES))
		and any(test.examples for test in doctest.DocTestFinder().find(module))]


@pytest.mark.parametrize('module', CYTHON_DOCTEST_MODULES,
		ids=lambda module: module.__name__)
def test_cython_doctests(module):
	result = doctest.testmod(module,
			optionflags=doctest.NORMALIZE_WHITESPACE | doctest.REPORT_NDIFF)
	assert result.failed == 0


@pytest.mark.parametrize('module', SELFTEST_MODULES,
		ids=lambda module: module.__name__)
def test_module_selftest(module):
	module.test()
