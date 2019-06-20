from keras.utils import multi_gpu_model
from keras import backend
from systemtools.logger import *
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.callbacks import Callback, History
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.utils import multi_gpu_model
from machinelearning.metrics import *
from systemtools.logger import *
from systemtools.duration import *
from systemtools.basics import *
from systemtools.file import *
from systemtools.location import *
from systemtools.system import *
from systemtools.logger import *
from datatools.jsonutils import *
from machinelearning.utils import *
from machinelearning.iterator import *

def isCompiled(model):
	try:
		model.optimizer.lr
		return True
	except:
		return False

def toMultiGPU(model, logger=None, verbose=True):
	try:
		gpuCount = len(backend.tensorflow_backend._get_available_gpus())
		model = multi_gpu_model(model, gpus=gpuCount)
	except Exception as e:
		logException(e, logger, verbose=verbose)
	return model


def iteratorToArray(it, steps=None):
	if it is None:
		return None
	newVal = None
	if isinstance(it, InfiniteBatcher):
		batchs = []
		for i in range(steps):
			current = next(it)
			batchs.append(current)
		if isListOrArray(batchs[0][0]):
			newVal = np.vstack(batchs)
		else:
			newVal = np.array(flattenLists(batchs))
	elif isinstance(it, list):
		newVal = np.array(it)
	elif isinstance(it, np.ndarray):
		newVal = it
	else:
		newVal = []
		for current in it:
			newVal.append(current)
		newVal = np.array(newVal)
	return newVal

AUTO_MODES = \
{
	'val_loss': 'min',
	'val_acc': 'max',
	'val_top_k_categorical_accuracy': 'max',
}

class KerasCallback(Callback): # https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L614
	def __init__\
	(
		self,
		originalModel=None,
		xVal=None,
		yVal=None,
		steps=None,
		logger=None,
		verbose=True,
		metricsFreq=1,
		metrics=None,
		metricsFirstEpoch=0,
		logHistoryOnEpochEnd=False,
		plotFiguresOnEpochEnd=True,
		graphDir=None,
		eraseGraphDir=True,
		modelsDir=None,
		doNotif=False,
		doPltShow=False,
		saveMetrics={"val_loss": "min", "val_acc": "max",},
		stopFile=None,
		historyFile=None,
		earlyStopMonitor=None,
	):
		"""
			xVal and yVal can be an InfiniteBatcher instance (see machinelearing.iterator.InfiniteBatcher) or any iterable (np arrays, list, generator, machinelearing.iterator.ConsistentIterator...)

			Example of earlyStopMonitor:
			{
				'val_loss': {'patience': 50, 'min_delta': 0, 'mode': 'min'},
				'val_acc': {'patience': 50, 'min_delta': 0.05, 'mode': 'max'},
				'val_top_k_categorical_accuracy': {'patience': 50, 'min_delta': 0, 'mode': 'auto'},
			}
			You have to give at least a patience for each 
		"""
		self.logger = logger
		self.verbose = verbose
		self.earlyStopMonitor = earlyStopMonitor
		normalizeEarlyStopMonitor(self.earlyStopMonitor,
				logger=self.logger, verbose=self.verbose)
		assert not isinstance(xVal, InfiniteBatcher) or steps is not None
		self.historyFile = historyFile
		self.stopFile = stopFile
		self.originalModel = originalModel
		self.xVal = xVal
		self.yVal = yVal
		self.steps = steps
		self.xVal = iteratorToArray(self.xVal, steps=self.steps)
		self.yVal = iteratorToArray(self.yVal, steps=self.steps)
		self.metricsFreq = metricsFreq
		self.metrics = metrics
		if self.metrics is None:
			self.metrics = []
		self.metricsFirstEpoch = metricsFirstEpoch
		self.logHistoryOnEpochEnd = logHistoryOnEpochEnd
		self.plotFiguresOnEpochEnd = plotFiguresOnEpochEnd
		self.graphDir = graphDir
		self.eraseGraphDir = eraseGraphDir
		self.modelsDir = modelsDir
		self.saveModelOnEpochEnd = self.modelsDir is not None
		self.doNotif = doNotif
		self.saveMetrics = saveMetrics
		self.doPltShow = doPltShow
		if self.saveModelOnEpochEnd and (self.modelsDir is None or self.originalModel is None):
			self.saveModelOnEpochEnd = False
			logError("Please provide a modelsDir and an originalModel", self)
		if self.graphDir is None:
			self.plotFiguresOnEpochEnd = False
			logError("Please provide a graph directory", self)
		self.epochs = dict()
		self.history = dict()
		self.tt = TicToc(logger=self.logger, verbose=self.verbose)
	
	def on_train_begin(self, logs=None):
		self.tt.tic(display=False)

	def getEvalArrays(self):
		y_true = iteratorToArray(self.yVal, steps=self.steps)
		x = iteratorToArray(self.xVal, steps=self.steps)
		y_pred = self.originalModel.predict(x)
		return (y_true, y_pred)
	
	def recordLogs(self, epoch, logs):
		for key, score in logs.items():
			if key not in self.epochs:
				self.epochs[key] = []
				self.history[key] = []
			self.epochs[key].append(epoch)
			self.history[key].append(score)
	
	def recordMetrics(self, epoch, logs):
		try:
			localTT = TicToc(logger=self.logger, verbose=self.verbose)
			localTT.tic(display=False)
			(y_true, y_pred) = self.getEvalArrays()
			for key, funct in self.metrics.items():
				score = funct(y_true, y_pred)
				if key not in self.epochs:
					self.epochs[key] = []
					self.history[key] = []
				self.epochs[key].append(epoch)
				self.history[key].append(score)
			localTT.toc("Custom metrics done")
		except Exception as e:
			logException(e, self)
	
	def logHistory(self):
		log("History:\n" + lts(self.history), self) # TODO truncate floats
		log("Last scores:\n" + lts(self.getLastScores()), self)
			
	def plotFigures(self):
		try:
			# We handle the dir:
			if self.eraseGraphDir:
				graphDir = self.graphDir
				remove(graphDir, minSlashCount=4)
			else:
				graphDir = self.graphDir + "/" + getDateSecond()
			mkdir(graphDir)
			# We find all keys:
			keys = []
			for key in self.epochs.keys():
				if key.startswith("val"):
					keys.append(key)
			import matplotlib.pyplot as plt
			# We plot all:
			for key in keys:
				try:
					if len(self.history[key]) > 1:
						trainKey = None
						if key[4:] in self.history:
							trainKey = key[4:]
						if key in self.history:
							plt.plot(self.epochs[key], self.history[key])
							legend = ['Test']
						if trainKey in self.history:
							plt.plot(self.epochs[trainKey], self.history[trainKey])
							legend = ['Test', 'Train']
						plt.title(key)
						plt.ylabel('Score')
						plt.xlabel('Epoch')
						plt.legend(legend, loc='upper left')
						plt.savefig(graphDir + "/" + key + ".png", format='png')
						plt.figure()
				except Exception as e:
					logException(e, self)
			try:
				if self.doPltShow:
					plt.show()
				else:
					plt.close()
			except Exception as e:
				logException(e, self)
		except Exception as e:
			logException(e, self)
	
	def isBetterScore(self, key, score, previousScores):
		if not isinstance(previousScores, list):
			previousScores = [previousScores]
		comparisonType = self.saveMetrics[key]
		if comparisonType == "max":
			return score >= max(previousScores)
		elif comparisonType == "min":
			return score <= min(previousScores)
	
	def getLastScores(self):
		lastScores = dict()
		for key, scores in self.history.items():
			lastScores[key] = scores[-1]
		return lastScores
	
	def saveModel(self, epoch):
		epochToken = digitalizeIntegers(str(epoch), 4)
		foundBetter = False
		lastScores = self.getLastScores()
		for key, scores in self.history.items():
			if len(scores) > 0 and key in self.saveMetrics:
				currentLastScore = lastScores[key]
				if self.isBetterScore(key, currentLastScore, scores):
					foundBetter = True
					break
		if foundBetter:
			epochDir = self.modelsDir + "/epoch" + epochToken
			mkdir(epochDir)
			self.originalModel.save(epochDir + "/model.h5")
			log("We saved the current model in " + epochDir, self)
			toJsonFile(lastScores, epochDir + "/scores.json")
			# Now we remove old models that have all metrics lower than the current:
			for currentDir in sortedGlob(self.modelsDir + "/epoch*"):
				if "/epoch" + epochToken not in currentDir:
					currentScores = fromJsonFile(currentDir + "/scores.json")
					foundBetter = False
					for currentKey, currentScore in currentScores.items():
						if currentKey in self.saveMetrics\
						and self.isBetterScore(currentKey, currentScore, lastScores[currentKey]):
							foundBetter = True
							break
					if not foundBetter:
						log("We remove " + currentDir + " because all scores are lower", self)
						remove(currentDir, minSlashCount=4)				
	
	def on_epoch_end(self, epoch, logs=dict()):
		self.tt.tic("Epoch " + str(epoch) + " done")
		self.recordLogs(epoch, logs)
		if epoch >= self.metricsFirstEpoch and (epoch-self.metricsFirstEpoch) % self.metricsFreq == 0 and len(self.metrics) > 0: # WARNING epoch start at 0
			self.recordMetrics(epoch, logs)
		if self.logHistoryOnEpochEnd:
			self.logHistory()
		if self.plotFiguresOnEpochEnd:
			self.plotFigures()
		if self.saveModelOnEpochEnd:
			self.saveModel(epoch)
		if self.doNotif:
			try:
				notif("Last scores on " + getHostname(), lts(self.getLastScores()))
			except Exception as e:
				logException(e, self)
		if self.historyFile is not None:
			toJsonFile({"epochs": self.epochs, "history": self.history}, self.historyFile)
		if self.stopFile is not None and isFile(self.stopFile):
			self.model.stop_training = True
			log("We stop training because we found " + self.stopFile, self)
		if self.earlyStopMonitor is not None and len(self.earlyStopMonitor) > 0:
			esm = normalizeEarlyStopMonitor(self.earlyStopMonitor,
				logger=self.logger, verbose=self.verbose)
			if hasToEarlyStop(self.history, esm,
				logger=self.logger, verbose=self.verbose):
				self.model.stop_training = True
				log("We early stop.", self)
		self.tt.tic("on_epoch_end done")
		self.tt.toc()


def normalizeEarlyStopMonitor(earlyStopMonitor, logger=None, verbose=True):
	esm = dict()
	for monitor in earlyStopMonitor.keys():
		current = earlyStopMonitor[monitor]
		if not dictContains(current, "patience"):
			raise Exception(monitor + " has no patience")
		if not dictContains(current, "mode") or current["mode"] == 'auto':
			if monitor in AUTO_MODES:
				current["mode"] = AUTO_MODES[monitor]
			else:
				raise Exception("Please provide a mode for " + monitor)
		if not dictContains(current, "min_delta"):
			current["min_delta"] = 0.0
		if current["min_delta"] < 0.0:
			raise Exception("min_delta of " + monitor + " must be greater or equal that 0.0")
		esm[monitor] = current
		if monitor in AUTO_MODES and current["mode"] != AUTO_MODES[monitor]:
			logError(monitor + " mode inconsistent... Found " + AUTO_MODES[monitor] + " but got " + current["mode"], logger=logger, verbose=verbose)
	return esm


def hasToEarlyStop(histories, esm, logger=None, verbose=True):
	doStop = True
	for monitor, values in esm.items():
		if monitor in histories.keys():
			history = histories[monitor]
			if len(history) > values["patience"]:
				notEnhancedCount = 0
				index = len(history)
				historyIteration = history[-values["patience"]-1:]
				for targetScore in reversed(historyIteration):
					index -= 1
					historyPart = history[:index] # + history[index+1:]
					if len(historyPart) == 0:
						break
					minScore = min(historyPart)
					maxScore = max(historyPart)
					if values['mode'] == 'min':
						if minScore - targetScore >= values['min_delta']:
							break # We got an enhancment
					else:
						if targetScore - maxScore >= values['min_delta']:
							break # We got an enhancment
					notEnhancedCount += 1
				if notEnhancedCount <= values["patience"]:
					doStop = False
					break
			else:
				doStop = False
				break
		else:
			logError("We didn't found " + monitor + " in the history which contains " + str(list(histories.keys())), logger=logger, verbose=verbose)
			doStop = False
			break
	return doStop



def buildRNN\
(
    docLength=None,
    vocSize=None,
    embeddingMatrix=None,
    nbClasses=None,
    isEmbeddingsTrainable=False,
    denseUnits=100,
    nbDenses=0,
    denseActivation='tanh', # tanh, sigmoid, relu
    rnnUnits=100,
    firstDropout=0.2,
    recurrentDropout=0.2,
    denseDropout=0.2,
    useRNNDropout=True,
    isBidirectional=False,
    isCuDNN=False,
    rnnType='LSTM', # GRU, LSTM
    addConv1D=False,
    conv1dActivation='relu',
    filters=32,
    kernelSize=3,
    poolSize=3,
    logger=None,
    verbose=False,
):
    """
        This function return a RNN Keras model (GRU or LSTM) which can, optionaly, be CuDNN optimized, bidirectional)
        After getting the model you can convert the model to a multi gpu model, compile the model, print the summary and plot the architecture
    """
    # We check values:
    assert denseActivation in [None, 'tanh', 'sigmoid', 'relu']
    assert rnnType in ['GRU', 'LSTM']
    assert docLength is not None
    assert vocSize is not None
    assert embeddingMatrix is not None
    assert nbClasses is not None
    assert denseUnits is not None
    # We find some informations:
    embeddingsDimension = embeddingMatrix.shape[1]
    assert embeddingsDimension <= 1500
    # We define the model:
    model = Sequential()
    # We add en embedding layer:
    model.add(Embedding(vocSize, embeddingsDimension, input_length=docLength, weights=[embeddingMatrix], trainable=isEmbeddingsTrainable))
    # Optionnaly we add a Conv1D layer:
    if addConv1D:
        model.add(Conv1D(filters=filters, kernel_size=kernelSize, padding='same', activation=conv1dActivation))
        model.add(MaxPooling1D(pool_size=poolSize))
    # We find the RNN to add:
    if isCuDNN:
        if rnnType == 'GRU':
            TheLayerClass = CuDNNGRU
        else:
            TheLayerClass = CuDNNLSTM
        useRNNDropout = False
    else:
        if rnnType == 'GRU':
            TheLayerClass = GRU
        else:
            TheLayerClass = LSTM
    # We add a dropout:
    if firstDropout is not None and firstDropout > 0.0 and not useRNNDropout:
        model.add(Dropout(firstDropout))
    if useRNNDropout:
        if isBidirectional:
            model.add(Bidirectional(TheLayerClass(rnnUnits, dropout=firstDropout, recurrent_dropout=recurrentDropout)))
        else:
            model.add(TheLayerClass(rnnUnits, dropout=firstDropout, recurrent_dropout=recurrentDropout))
    else:
        if isBidirectional:
            model.add(Bidirectional(TheLayerClass(rnnUnits)))
        else:
            model.add(TheLayerClass(rnnUnits))
    if recurrentDropout is not None and recurrentDropout > 0.0 and not useRNNDropout:
        model.add(Dropout(recurrentDropout))
    # We add dense layers:
    for i in range(nbDenses):
        model.add(Dense(denseUnits, activation=denseActivation))
        if denseDropout is not None and denseDropout > 0.0:
            model.add(Dropout(denseDropout))
    # We add the last layer:
    model.add(Dense(nbClasses, activation='softmax'))
    return model