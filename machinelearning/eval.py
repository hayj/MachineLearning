
"""
The difference betwen `cross_validate` and `cross_val_score` is that `cross_validate` can take multiple scoring methods and return fit/score times :

 * <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate>
 * <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html>

Here scoring methods:

 * <https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values>

Metrics (`'macro'` means in the multi-class and multi-label case, we take the average of the F1 score of each class):

 * `'accuracy``: "(TP+TN)/(TP+TN+FP+FN)" ou "Bien classé / Nombre d'exemples"
 * `'precision_macro``: "true positive / (true positive + false positive)", equivalent to "true positive / total predicted positive" ou "bien classé en 1 (pertinent) / ceux classé en 1 (pertinent)" (<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score>).
 * `'recall_macro``: "true positive / (true positive + false negative)" ou "true positive / (total actual positive)" ou "bien classé en 1 (pertinent) / nombre de 1 (pertinent)" (<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score>).
 * `'f1_macro'`: `2 * (precision * recall) / (precision + recall)` (<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score>).

More explanations :

 * <https://simonhessner.de/why-are-precision-recall-and-f1-score-equal-when-using-micro-averaging-in-a-multi-class-problem/>
 * <https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult>
 * <https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9>
 
"""

from systemtools.basics import *
from systemtools.logger import *
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold, StratifiedKFold
from matplotlib import pyplot as plt # WARNING see below


def crossValidate\
(
	model,
	X,
	y,
	*args,
	scoring=None, # If None, the estimator’s score method is used, you can use ['accuracy', 'f1_macro'] # 'precision_macro', 'recall_macro'
	isRegression=None,
	addTime=False,
	addMean=True,
	addConfidence=True,
	addExtraConfidence=False,
	removeLists=True,
	n_jobs=-1,
	decimals=5,
	return_train_score=False,

	cv=5,
	shuffle=True,
	random_state=0,

	verbose=True,
	logger=None,
	**kwargs,
):
	"""
		This function return a dict containing cross validation scores, times, 95% confidence etc.
		The default strategies for classiication is ['uniform', 'most_frequent', 'stratified']
		The default DummyModel for classiication is sklearn.dummy.DummyClassifier
		For scoring, if None, the estimator’s score method is used, you can use ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'] etc.
	"""

	if not addMean and removeLists:
		raise Exeption("You cannot have no score")
	if isRegression is None:
		isRegression = isinstance(y[0], float)
	if scoring is None:
		log("The scoring methods will be: " + str(model.score), logger, verbose=verbose)
	else:
		log("The scoring methods will be: " + str(scoring), logger, verbose=verbose)
	# if scoring is None:
	# 	if isRegression:
	# 		scoring = None
	# 	else:
	# 		scoring = ['accuracy', 'f1_macro'] # 'precision_macro', 'recall_macro'
	# elif not isinstance(scoring, list):
	# 	scoring = [scoring]
	if isinstance(cv, int):
		kfold = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
	else:
		kfold = cv
	if isinstance(scoring, str):
		scoring = [scoring]
	result = cross_validate(model, X, y, *args, cv=kfold, scoring=scoring, n_jobs=n_jobs, return_train_score=return_train_score, **kwargs)
	scoreKeys = [("test_score", "score")]
	if scoring is not None:
		for current in scoring:
			scoreKeys.append(("test_" + current, current))
	for scoreKey, token in scoreKeys:
		if scoreKey in result:
			currentResult = dict()
			if not removeLists:
				currentResult["scores"] = result[scoreKey]
			if addMean:
				currentResult["score"] = result[scoreKey].mean()
			if addConfidence:
				currentResult["confidence"] = result[scoreKey].std() * 2
			if decimals is not None and isinstance(decimals, int) and decimals >= 1:
				currentResult["score"] = np.around(currentResult["score"], decimals=decimals)
				if "confidence" in currentResult:
					currentResult["confidence"] = np.around(currentResult["confidence"], decimals=decimals)
			result[token] = currentResult
			del result[scoreKey]
	result['fit_times'] = result['fit_time']
	result['score_times'] = result['score_time']
	if "train_score" in result:
		result['train_scores'] = result['train_score']
	if addMean:
		result['fit_time'] = result['fit_times'].mean()
		result['score_time'] = result['score_times'].mean()
		if "train_scores" in result:
			result['train_score'] = result['train_scores'].mean()
	if addExtraConfidence:
		result['fit_time_confidence'] = result['fit_times'].std() * 2
		result['score_time_confidence'] = result['score_times'].std() * 2
	if removeLists:
		keysToRemove = []
		for key, value in result.items():
			if isinstance(value, type(np.array([0]))):
				keysToRemove.append(key)
		for key in keysToRemove:
			del result[key]
	if not addTime:
		keysToRemove = []
		for key, value in result.items():
			if "time" in key:
				keysToRemove.append(key)
		for key in keysToRemove:
			del result[key]
	if decimals is not None and isinstance(decimals, int) and decimals >= 1:
		for key in result.keys():
			if isinstance(result[key], float):
				result[key] = np.around(result[key], decimals=decimals)
	return result




def scikitLearnFit\
(
    clf,
    xTrain, yTrain,
    xVal, yVal,
    patience=None, maxIter=1000,
    scoreFunct=None,
    figPath=None,
    doFigShow=False,
    doFigSave=True,
    figShowInterval=None,
    logger=None,
    verbose=True,
    doClearOutput=True,
    stopFile=None,
):
    """
        https://superuser.com/questions/557622/how-can-i-view-pictures-via-ssh
    """
    def __acc(clf, xVal, yVal):
        rightPredictedCount = 0
        for i, pred in enumerate(clf.predict(xVal)):
            label = yVal[i]
            if label == pred:
                rightPredictedCount += 1
        score = rightPredictedCount / len(yVal)
        return score
    if stopFile is None:
        stopFile = tmpDir("stop-files")
        stopFileName = "stop-" + getRandomStr()
        stopFile += "/" + stopFileName
        stopFileMessage = "Use this command to stop training:\n\ttouch ~/tmp/stop-files/" + stopFileName
    else:
        stopFileMessage = "Use this command to stop training:\n\ttouch " + stopFile
    if doClearOutput:
        from IPython.display import clear_output
        if figShowInterval is None:
            figShowInterval = 1
    if patience is None:
        patience = int(0.1 * maxIter)
    if figShowInterval is None:
        figShowInterval = int(0.01 * maxIter)
    if scoreFunct is None:
        scoreFunct = __acc
    if figPath is None:
        figPath = tmpDir("figures") + "/main.png"
    classes = np.unique(np.array(list(yVal) + list(yTrain)))
    epochs = []
    scores = []
    plt.figure()
    lastBestScore = -100000000.0
    epochsSinceLastBestScore = 0
    for i in range(maxIter):
        clf.partial_fit(xTrain, yTrain, classes=classes)
        score = scoreFunct(clf, xVal, yVal)
        scores.append(score)
        epochs.append(i)
        plt.clf()
        if doClearOutput:
            clear_output(wait=True)
        log(stopFileMessage, logger=logger, verbose=verbose)
        plt.plot(epochs, scores)
        legend = ['Test']
        plt.ylabel('Score')
        plt.xlabel('Epoch')
        plt.legend(legend, loc='upper left')
        if doFigShow and i % figShowInterval == 0 and i >= 1:
            plt.show()
        plt.savefig(figPath, format='png')
        if score > lastBestScore:
            lastBestScore = score
            epochsSinceLastBestScore = 0
        else:
            epochsSinceLastBestScore += 1
        if epochsSinceLastBestScore > patience:
            log("We early stop at epoch " + str(i) + ".", logger=logger, verbose=verbose)
            break
        if isFile(stopFile):
            break
    return scores


def test2():
	from newssource.asattribution.asamin import getTFIDFSGDClassifier
	from sklearn.naive_bayes import GaussianNB
	from sklearn.linear_model import SGDClassifier
	clf = getTFIDFSGDClassifier()
	# clf = SGDClassifier()
	# clf = GaussianNB()
	print(crossValidate(clf, np.array([[0]] * 50 + [[1]] * 50), np.array([0] * 50 + [1] * 50), cv=5))


def test1():
	from sklearn import datasets, linear_model
	from sklearn.metrics.scorer import make_scorer
	from sklearn.metrics import confusion_matrix
	from sklearn.svm import LinearSVC
	diabetes = datasets.load_diabetes()
	X = diabetes.data[:150]
	y = diabetes.target[:150]
	lasso = linear_model.Lasso()
	printLTS(crossValidate(lasso, X, y, cv=3))

if __name__ == '__main__':
	test2()

