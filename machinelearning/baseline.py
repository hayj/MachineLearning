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


def bestDummyScore(*args, **kwargs):
	bestScore = -10000
	for key, value in dummyScores(*args, **kwargs).items():
		if value["score"] > bestScore:
			bestScore = value["score"]
	return bestScore

def dummyScores\
(
	y,
	*args,
	isRegression=None,
	DummyModel=None,
	strategies=None,
	evalFunct=crossValidate,
	verbose=False,
	logger=None,
	**kwargs,
):
	"""
		This function return a dict containing results given by evalFunct for each strategy
		The default strategies for classiication is ['uniform', 'most_frequent', 'stratified']
		The default DummyModel for classiication is sklearn.dummy.DummyClassifier
	"""
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
		currentResult = evalFunct(clf, X, y, *args, **kwargs)
		result[strategy] = currentResult
	# if verbose:
	# 	log(lts(result), logger, verbose=verbose)
	return result


if __name__ == '__main__':
	seed()
	labels = np.array(["a", "a", "a", "a", "a", "a", "b", "b", "b", "c"] * 30)
	labels = np.array([1, 1, 1, 1, 1, 1, 2, 3] * 30)
	labels = list([1, 1, 1, 1, 1, 1, 2, 3] * 30)
	labels = list([1, 1, 1, 2, 3, 4] * 30)
	# print(labels)
	printLTS(dummyScores(labels))
	# print(bestDummyScore(labels))
