"""Run small parsing experiments and check their evaluation scores."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('experiment', ['pcfg', '2dop', 'dopred', 'longsent'])
def test_experiment(experiment, tmp_path):
	testsdir = Path(__file__).resolve().parent
	paramfile = tmp_path / f'{experiment}.prm'
	shutil.copyfile(testsdir / paramfile.name, paramfile)
	result = subprocess.run(
			[sys.executable, '-m', 'discodop.cli', 'runexp', str(paramfile)],
			cwd=testsdir, capture_output=True, text=True, check=False)
	assert result.returncode == 0, result.stdout + result.stderr
	log = (tmp_path / experiment / 'output.log').read_text()
	assert 'Error' not in log, log
	# Coarse PCFG and posterior-pruned stages need not reach perfect scores.
	scores = [line for line in log.splitlines()
			if 'ex ' in line and not re.search(r'pcfg |post ', line)]
	assert scores, log
	assert all(re.search(r'cov 100\.0.*ex 100\.0', line) for line in scores), (
			'\n'.join(scores))
