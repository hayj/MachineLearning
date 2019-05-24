"""

/home/hayj/.local/share/virtualenvs/st-venv/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.

means a categorie is never predicted so we cannot calculate a F1 score


/home/hayj/.local/share/virtualenvs/st-venv/lib/python3.6/site-packages/sklearn/dummy.py:227: FutureWarning: arrays to stack must be passed as a "sequence" type such as list or tuple. Support for non-sequence iterables such as generators is deprecated as of NumPy 1.16 and will raise an error in the future.

this warning appear only for the stratified strategy

"""


# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)


from sklearn.dummy import DummyClassifier
from systemtools.system import *
from systemtools.logger import *
from systemtools.basics import *
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from machinelearning.seeder import *
from machinelearning.eval import *


def bestDummyScore(y, scoring=None, *args, logger=None, verbose=True, **kwargs):
	if scoring is None:
		scoring = "accuracy"
		logWarning("We will use " + str(scoring) + " as the default scoring funct", logger, verbose=verbose)
	bestScore = None
	results = dummyScores(y, *args, scoring=scoring, logger=logger, verbose=verbose, **kwargs)
	assert len(results) > 0
	foundMetric = None
	if scoring is None:
		foundMetric = None
	else:
		if isinstance(scoring, list):
			metric = scoring[0]
		else:
			metric = scoring
		for strategy, infos in results.items():
			for key, value in infos.items():
				if isinstance(value, dict) and "score" in value and metric in key:
					foundMetric = key
			break
		assert foundMetric is not None
	for strategy, infos in results.items():
		for key, value in infos.items():
			if foundMetric is None or key == foundMetric:
				if bestScore is None or value["score"] > bestScore:
					bestScore = value["score"]
	assert bestScore is not None
	return bestScore

def dummyScores\
(
	y,
	*args,
	scoring=None,
	isRegression=None,
	DummyModel=None,
	strategies=None,
	evalFunct=crossValidate,
	verbose=True,
	logger=None,
	**kwargs,
):
	"""
		This function return a dict containing results given by evalFunct for each strategy
		The default strategies for classiication is ['uniform', 'most_frequent', 'stratified']
		The default DummyModel for classiication is sklearn.dummy.DummyClassifier
	"""
	if scoring is None:
		scoring = "accuracy"
		logWarning("We will use " + str(scoring) + " as the default scoring funct", logger, verbose=verbose)
	if isRegression is None:
		isRegression = isinstance(y[0], float)
	regressionErrorMessage = "Regression not yet implemented"
	if DummyModel is None:
		if isRegression:
			raise Exeption(regressionErrorMessage)
		else:
			DummyModel = DummyClassifier
	if strategies is None:
		if isRegression:
			raise Exeption(regressionErrorMessage)
		else:
			strategies = ['uniform', 'most_frequent', 'stratified']
	if not isinstance(strategies, list):
		strategies = [strategies]
	X = [[0]] * len(y)
	result = dict()
	for strategy in strategies:
		clf = DummyModel(strategy=strategy)
		currentResult = evalFunct(clf, X, y, *args, logger=logger, verbose=verbose, scoring=scoring, **kwargs)
		result[strategy] = currentResult
	# if verbose:
	# 	log(lts(result), logger, verbose=verbose)
	return result


if __name__ == '__main__':
	for i in range(1):
		seed(i)
		labels = np.array([1, 1, 1, 1, 1, 1, 2, 3] * 30)
		labels = list([1, 1, 1, 1, 1, 1, 2, 3] * 30)
		labels = list([1, 5, 6, 2, 3, 4] * 30)
		labels = np.array(["a", "b", "c"] * 30)
		# print(labels)
		# printLTS(dummyScores(labels))
		print(bestDummyScore(labels))
