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
try:
	import matplotlib.pyplot as plt
except Exception as e:
	print(e)


def toMultiGPU(model, logger=None, verbose=True):
	try:
		gpuCount = len(backend.tensorflow_backend._get_available_gpus())
		model = multi_gpu_model(model, gpus=gpuCount)
	except Exception as e:
		logException(e, logger, verbose=verbose)
	return model


class KerasCallback(Callback): # https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L614
    def __init__\
    (
        self,
        originalModel=None,
        tokensInfiniteBatcher=None,
        labelInfiniteBatcher=None,
        steps=None,
        logger=None,
        verbose=True,
        metricsFreq=1,
        metrics=[],
        logHistoryOnEpochEnd=True,
        plotFiguresOnEpochEnd=True,
        graphDir=None,
        eraseGraphDir=True,
        saveModelOnEpochEnd=True,
        modelsDir=None,
        doNotif=True,
        doPltShow=False,
        saveMetrics={"val_loss": "min", "val_acc": "max",},
    ):
        self.originalModel = originalModel
        self.tokensInfiniteBatcher = tokensInfiniteBatcher
        self.labelInfiniteBatcher = labelInfiniteBatcher
        self.steps = steps
        self.logger = logger
        self.verbose = verbose
        self.metricsFreq = metricsFreq
        self.metrics = metrics
        self.logHistoryOnEpochEnd = logHistoryOnEpochEnd
        self.plotFiguresOnEpochEnd = plotFiguresOnEpochEnd
        self.graphDir = graphDir
        self.eraseGraphDir = eraseGraphDir
        self.saveModelOnEpochEnd = saveModelOnEpochEnd
        self.modelsDir = modelsDir
        self.doNotif = doNotif
        self.saveMetrics = saveMetrics
        self.doPltShow = doPltShow
        if self.modelsDir is None or self.originalModel is None:
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
        batchs = []
        for i in range(self.steps):
            current = next(self.labelInfiniteBatcher)
            batchs.append(current)
        y_true = np.vstack(batchs)
        y_pred = self.model.predict_generator(self.tokensInfiniteBatcher, steps=self.steps)
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
                        plt.savefig(graphDir + "/" + key + ".png")
                        plt.figure()
                except Exception as e:
                    logException(e, self)
            if self.doPltShow:
                try:
                    plt.show()
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
        if epoch % self.metricsFreq == 0 and len(self.metrics) > 0: # WARNING epoch start at 0
            self.recordMetrics(epoch, logs)
        if self.logHistoryOnEpochEnd:
            self.logHistory()
        if self.plotFiguresOnEpochEnd:
            self.plotFigures()
        if self.saveModelOnEpochEnd:
            self.saveModel(epoch)
        if self.doNotif:
            try:
                notif("Last scores", lts(self.getLastScores()))
            except Exception as e:
                logException(e, self)
        self.tt.tic("on_epoch_end done")
        self.tt.toc()