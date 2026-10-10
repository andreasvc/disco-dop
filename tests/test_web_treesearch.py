"""Tree-search links must preserve parameters with standard query separators."""
import importlib.util
import pickle
import re
import sys
from html import unescape
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from discodop import fragments, treesearch
from discodop.tree import Tree


@pytest.fixture
def webapp(tmp_path, monkeypatch):
	pytest.importorskip('flask')
	# Supply a cached corpus so importing the app needs no external search tools.
	corpusdir = tmp_path / 'corpus'
	corpusdir.mkdir()
	filename = corpusdir / 'sample.tok'
	filename.write_text('cat\n')
	info = SimpleNamespace(len=100, numwords=100, numnodes=100)
	with (corpusdir / 'treesearchcorpus.pickle').open('wb') as out:
		pickle.dump((['sample'], {'regex': [info]}, None), out)
	monkeypatch.chdir(tmp_path)
	monkeypatch.setattr(treesearch, 'RegexSearcher',
			lambda files, **kwargs: SimpleNamespace(files=files))
	monkeypatch.setattr(fragments, 'PARAMS', fragments.PARAMS.copy())
	path = Path(__file__).resolve().parents[1] / 'web' / 'treesearch.py'
	spec = importlib.util.spec_from_file_location('treesearch_web_test', path)
	module = importlib.util.module_from_spec(spec)
	monkeypatch.setitem(sys.modules, spec.name, module)
	spec.loader.exec_module(module)
	tree = Tree('(ROOT (NN 0))')

	def extracts(file, indices, *args, sents=False, **kwargs):
		return ['cat' if sents else (tree, ['cat']) for _ in indices]

	def sentences(*args, brackets=False, **kwargs):
		text = '(ROOT (NN 0))\tcat' if brackets else 'cat'
		return [(str(filename), 41, text, 'cat' if brackets else set(), set())]

	corpus = SimpleNamespace(files=[str(filename)], extract=extracts,
			sents=sentences, trees=lambda *args, **kwargs:
			[(str(filename), 41, tree, ['cat'], [])])
	module.CORPORA = {'frag': corpus}
	module.CORPUSINFO = {'frag': [info]}
	return module


def links(html):
	return [unescape(url) for url in re.findall(r'href="([^"]+)"', html)
			if '?' in url]


def params(url):
	assert ';' not in urlsplit(url).query, url
	return parse_qs(urlsplit(url).query, keep_blank_values=True)


@pytest.mark.parametrize('view', ['trees', 'sents', 'brackets'])
def test_result_links(webapp, view):
	query = 'NP & NN; café'
	form = {'query': query, 'engine': 'frag', 'nofunc': '', 'nomorph': ''}
	urls = links(''.join(getattr(webapp, view)(form)))
	export = next(url for url in urls if 'linenos=' in url)
	assert params(export)['linenos'] == ['1']
	assert params(export)['query'] == [query]
	browse = next(url for url in urls if url.startswith('browse?'))
	assert params(browse) == {'text': ['0'], 'sent': ['41'], 'nofunc': [''], 'nomorph': ['']}
	context = next(url for url in urls if url.startswith('browsesents?'))
	assert params(context)['query'] == [query]
	assert params(context)['engine'] == ['frag']


@pytest.mark.parametrize('route', ['browse', 'browsesents'])
def test_browse_links(webapp, route):
	client = webapp.APP.test_client()
	listing = client.get('/' + route).get_data(as_text=True)
	assert params(links(listing)[0]) == {'text': ['0'], 'sent': ['1'], 'nomorph': ['']}
	query = 'NP & NN; café'
	response = client.get('/' + route, query_string=urlencode({
			'text': 0, 'sent': 41, 'engine': 'frag', 'query': query}))
	assert response.status_code == 200
	urls = links(response.get_data(as_text=True))
	assert len(urls) == 2
	expected = {22, 62} if route == 'browse' else {21, 61}
	assert {int(params(url)['sent'][0]) for url in urls} == expected
	if route == 'browsesents':
		assert all(params(url)['query'] == [query] for url in urls)
		assert all(params(url)['engine'] == ['frag'] for url in urls)


def test_fragment_draw_link(webapp, monkeypatch):
	monkeypatch.setattr(fragments, 'regular', lambda *args:
			([('(ROOT (NN 0))', ('cat;&café',))], [2]))
	urls = links(''.join(webapp.fragmentsinresults({'query': 'NN', 'engine': 'frag'})))
	draw = next(url for url in urls if url.startswith('draw?'))
	assert params(draw) == {'tree': ['(ROOT (NN 0))'], 'sent': ['cat;&café']}
	response = webapp.APP.test_client().get('/' + draw)
	assert response.status_code == 200
	assert 'cat;&amp;café' in response.get_data(as_text=True)


def test_complexity_template_links(webapp):
	with webapp.APP.test_request_context():
		html = webapp.APP.jinja_env.get_template('browsemain.html').render(
				texts=['sample'])
	assert params(links(html)[0]) == {'text': ['0'], 'sent': ['1'],
			'feat': ['average dependency length']}
