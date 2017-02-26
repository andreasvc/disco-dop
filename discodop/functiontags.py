"""Function tags classifier."""
from __future__ import division, print_function, absolute_import, \
		unicode_literals
from .tree import Tree
from .treebanktransforms import base, functions, FUNC
from .heads import getheadpos
from .util import ishead


FIELDS = tuple(range(8))
WORD, LEMMA, TAG, MORPH, FUNC, PARENT, SECEDGETAG, SECEDGEPARENT = FIELDS


def trainfunctionclassifier(trees, sents, numproc):
	"""Train a classifier to predict functions tags in trees."""
	from sklearn import linear_model, multiclass, pipeline
	from sklearn import preprocessing, feature_extraction
	from sklearn.grid_search import GridSearchCV
	from sklearn.metrics import make_scorer, jaccard_similarity_score
	vectorizer = pipeline.Pipeline([
			feature_extraction.DictVectorizer(sparse=True),
			preprocessing.StandardScaler(copy=False)])
	# PTB has no function tags on pretermintals, Negra etc. do.
	posfunc = any(functions(node) for tree in trees
			for node in tree.subtrees()
			if node and isinstance(node[0], int))
	target = [functions(node) for tree in trees
			for node in tree.subtrees()
			if tree is not node and node
				and (posfunc or isinstance(node[0], Tree))]
	# PTB may have multiple tags (or 0) per node.
	# Negra etc. have exactly 1 tag for every node.
	multi = any(len(a) > 1 for a in target)
	if multi:
		encoder = preprocessing.MultiLabelBinarizer()
	else:
		encoder = preprocessing.LabelEncoder()
		target = [a[0] if a else '--' for a in target]
	# binarize features (output is a sparse array)
	trainfeats = vectorizer.fit_transform(functionfeatures(node, sent)
			for tree, sent in zip(trees, sents)
				for node in tree.subtrees()
				if tree is not node
				and node and (posfunc or isinstance(node[0], Tree)))
	trainfuncs = encoder.fit_transform(target)
	classifier = linear_model.SGDClassifier(
			loss='hinge',
			penalty='elasticnet',
			n_iter=int(10 ** 6 / len(trees)))
	alphas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
	if multi:
		classifier = multiclass.OneVsRestClassifier(
				classifier, n_jobs=numproc or -1)
		param_grid = dict(
				estimator__alpha=alphas)
	else:
		param_grid = dict(alpha=alphas)
	classifier = GridSearchCV(estimator=classifier, param_grid=param_grid,
			scoring=make_scorer(jaccard_similarity_score))
	# train classifier
	classifier.fit(trainfeats, trainfuncs)
	msg = ('trained classifier; grid search results:\n%s\n'
			'multi=%r, posfunc=%r; best score on training set: %g %%\n'
			'parameters: %r\nfunction tags: %s' % (
			'\n'.join(str(a) for a in classifier.grid_scores_),
			multi, posfunc, 100.0 * classifier.best_score_,
			classifier.best_estimator_,
			' '.join(str(a) for a in encoder.classes_)))
	return (classifier, vectorizer, encoder, posfunc, multi), msg


def applyfunctionclassifier(funcclassifier, tree, sent):
	"""Add predicted function tags to tree using classifier."""
	classifier, vectorizer, encoder, posfunc, multi = funcclassifier
	# get features and use classifier
	funclabels = encoder.inverse_transform(classifier.predict(
			vectorizer.transform(functionfeatures(node, sent)
			for node in tree.subtrees(lambda n: n is not tree and n
				and (posfunc or isinstance(n[0], Tree))))))
	# store labels in tree
	for node, func in zip(tree.subtrees(lambda n: n is not tree and n
			and (posfunc or isinstance(n[0], Tree))), funclabels):
		if not getattr(node, 'source', None):
			node.source = ['--'] * 6
		elif isinstance(node.source, tuple):
			node.source = list(node.source)
		if not func:
			node.source[FUNC] = '--'
		elif multi:
			node.source[FUNC] = '-'.join(func)
		else:
			node.source[FUNC] = func


def functionfeatures(node, sent):
	"""Return a list of features for node to predict its function tag.

	The node must be a ParentedTree, with head information.

	The features are based on Blaheta & Charniak (2000),
	Assigning Function Tags to Parsed Text.
	http://aclweb.org/anthology/A00-2031"""
	headsib = headsibpos = None
	for sib in node.parent:
		if ishead(sib):
			headsib = sib
			headsibpos = getheadpos(headsib)
			break
	result = {
			# 4. head sister const label
			'hsc': headsib.label if headsib else '',
			# 5. head sister head word POS
			'hsp': headsibpos.label if headsibpos else '',
			# 6. head sister head word
			'hsf': sent[headsibpos[0]] if headsibpos else '',
			# 10. parent label
			'moc': node.parent.label,
			# 11. grandparent label
			'grc': node.parent.parent.label
					if node.parent.parent else '',
			# 12. Offset of this node to head sister
			'ohs': (node.parent_index - headsib.parent_index)
					if headsib is not None else -1,
			}
	result.update(basefeatures(node, sent))
	# add similar features for neighbors
	if node.parent_index > 0:
		result.update(basefeatures(
				node.parent[node.parent_index - 1], sent, prefix='p'))
	if node.parent_index + 1 < len(node.parent):
		result.update(basefeatures(
				node.parent[node.parent_index + 1], sent, prefix='n'))
	return result


def basefeatures(node, sent, prefix=''):
	"""A set features describing this particular node."""
	headpos = getheadpos(node)
	if base(node, 'PP'):
		# NB: we skip the preposition here; need way to identify it.
		altheadpos = getheadpos(node[1:])
	else:
		altheadpos = None
	return {
			# 1. syntactic category
			prefix + 'cat': node.label,
			# 2. head POS
			prefix + 'hwp': headpos.label if headpos else '',
			# 3. head word
			prefix + 'hwf': sent[headpos[0]] if headpos else '',
			# 7. alt (for PPs, non-prep. node) head POS
			prefix + 'ahc': altheadpos.label if altheadpos else '',
			# 8. alt head word
			prefix + 'ahf': sent[altheadpos[0]] if altheadpos else '',
			# 9 yield length
			prefix + 'yis': len(node.leaves()),
			}


__all__ = ['trainfunctionclassifier', 'applyfunctionclassifier',
		'functionfeatures']
