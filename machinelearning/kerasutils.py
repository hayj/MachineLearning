from keras.utils import multi_gpu_model
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
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
import tensorflow


from sklearn.model_selection import KFold, StratifiedKFold
import gc


from keras.layers import LSTM, GRU, Dense, CuDNNLSTM, CuDNNGRU, Bidirectional
from keras.layers import BatchNormalization, Activation, SpatialDropout1D, InputSpec
from keras.layers import MaxPooling1D, TimeDistributed, Flatten, concatenate, Conv1D
from keras.utils import multi_gpu_model, plot_model
from keras.layers import concatenate, Input, Dropout
from keras.models import Model, load_model, Sequential
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback, History, ModelCheckpoint, EarlyStopping
from keras import optimizers
from keras import callbacks
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras import backend as K




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


AUTO_MODES = \
{
    'val_loss': 'min',
    'val_acc': 'max',
    'val_top_k_categorical_accuracy': 'max',
    'val_sparse_categorical_accuracy': 'max',
    'val_sparse_top_k_categorical_accuracy': 'max',
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
        initialEpoch=None,
        batchesAmount=None,
        batchesPassed=0,

        saveFunct=None,
        saveFunctKwargs=None,
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

            Set batchesPassedFile to save the current batches count until batchesAmount
            Set batchesPassed to start the count from this value
            Usefull to jump some batches in yur batch generator
            set batchesPassedSerializeEach > 1 to do not serialize at each batch
        """
        self.logger = logger
        self.verbose = verbose
        self.earlyStopMonitor = earlyStopMonitor
        normalizeEarlyStopMonitor(self.earlyStopMonitor,
                logger=self.logger, verbose=self.verbose)
        assert not isinstance(xVal, InfiniteBatcher) or steps is not None
        self.saveFunct = saveFunct
        self.saveFunctKwargs = saveFunctKwargs
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
            # self.plotFiguresOnEpochEnd = False
            self.graphDir =  tmpDir("kerasutils-graphs") + "/" + getDateSec()
            logError("Please provide a graph directory", self)
        self.initialEpoch = initialEpoch
        if self.historyFile is not None and isFile(self.historyFile):
            historyFileContent = fromJsonFile(self.historyFile)
            self.epochs = historyFileContent["epochs"]
            self.history = historyFileContent["history"]
            logWarning("We loaded previous epochs and history", self)
            if self.initialEpoch is not None:
                previousLength = len(dictFirstValue(self.epochs))
                newLength = len(dictFirstValue(self.epochs)[:self.initialEpoch])
                if previousLength != newLength:
                    logWarning("We reduced the history from a length of " + str(previousLength) + " to a length of " + str(newLength), self)
                    for key in self.epochs.keys():
                        self.epochs[key] = self.epochs[key][:self.initialEpoch]
                    for key in self.history.keys():
                        self.history[key] = self.history[key][:self.initialEpoch]
        else:
            self.epochs = dict()
            self.history = dict()
        self.tt = TicToc(logger=self.logger, verbose=self.verbose)
        self.alreadyFigured = False
        # BATCH STATE:
        self.batchesAmount = batchesAmount
        self.batchesPassed = batchesPassed

    def saveBatchesPassed(self):
        if self.batchesAmount is not None:
            # Here we set batchesPassed as the number of batches done (already passed):
            self.batchesPassed += 1
            # Then we reset to 0 if all batches passed:
            if self.batchesPassed == self.batchesAmount:
                self.batchesPassed = 0
                log("We passed all batches in the dataset", self)

    def on_batch_end(self, batch, logs={}):
        self.saveBatchesPassed()

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
            # We plot all:
            for key in keys:
                try:
                    if len(self.history[key]) > 1:
                        if not self.alreadyFigured:
                            plt.figure()
                            self.alreadyFigured = True
                        plt.clf()
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
            # We save the model using a custom function (for exemple machinelearning.kerasmodels.saveModel):
            if self.saveFunct is not None:
                currentKwargs = self.saveFunctKwargs
                if currentKwargs is None:
                    currentKwargs = dict()
                if "verbose" not in currentKwargs:
                	currentKwargs["verbose"] = False
                self.saveFunct(self.originalModel, epochDir,
                    **currentKwargs, logger=self.logger)
            # Or we just save the model using the keras method:
            else:
                self.originalModel.save(epochDir + "/model.h5")
            log("We saved the current model in " + epochDir, self)
            toJsonFile(lastScores, epochDir + "/scores.json")
            log("We saved scores in " + epochDir + "/scores.json", self)
            # And we serialize batchesPassed:
            if self.batchesAmount is not None:
                strToFile(str(self.batchesPassed), epochDir + "/batchesPassed.txt")
                log("We saved batched passed (integer) in " + epochDir + "/batchesPassed.txt", self)
            # Now we remove old models that have all metrics lower than the current:
            if self.saveModelOnEpochEnd:
                for currentDir in sortedGlob(self.modelsDir + "/epoch*"):
                    if "/epoch" + epochToken not in currentDir:
                        currentScores = fromJsonFile(currentDir + "/scores.json")
                        try:
                            if currentScores is not None:
                                foundBetter = False
                                for currentKey, currentScore in currentScores.items():
                                    if currentKey in self.saveMetrics\
                                    and self.isBetterScore(currentKey, currentScore, lastScores[currentKey]):
                                        foundBetter = True
                                        break
                                if not foundBetter:
                                    log("We remove " + currentDir + " because all scores are lower", self)
                                    remove(currentDir, minSlashCount=4)
                        except Exception as e:
                            logException(e, self)
                            remove(currentDir, minSlashCount=4, doRaise=False)

    
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
    if earlyStopMonitor is not None:
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


def estimateOptimalEpochs(model, x, y, patience=None,
                          batchSize=128, patienceRatioToKeep=0.3,
                          validationSplit=0.2,
                          logger=None, verbose=True, maxEpochs=10000,
                          kerasVerbose=False):
    if patience is None:
        patience = 10
        logWarning("We will use the default patience " + str(patience), logger, verbose=verbose)
    mainCallback = KerasCallback\
    (
        logger=logger,
        verbose=False,
        earlyStopMonitor=\
        {
            'val_loss': {'patience': patience},
            'val_acc': {'patience': patience},
        },
    )
    history = model.fit\
    (
        x, y,
        validation_split=validationSplit,
        epochs=maxEpochs,
        batch_size=batchSize,
        verbose=1 if kerasVerbose else 0,
        callbacks=[mainCallback],
    )
    epochs = len(history.history[list(history.history.keys())[0]])
    log("Total epochs: " + str(epochs), logger=logger, verbose=verbose)
    epochs = (epochs - patience) + math.ceil(patienceRatioToKeep * patience)
    log("Estimated optimal epochs: " + str(epochs), logger=logger, verbose=verbose)
    return epochs, history

def kerasCrossValidate(x, y, modelBuilder, labelEncoder=None, modelBuilderKwargs=None,
                   patience=10, cv=10, batchSize=128, estimateOptimalEpochsKwargs=None,
                   shuffle=True, randomState=0, logger=None, verbose=True, doPltShow=False,
                   kerasVerbose=False,
                   graphsDir=None,
                   defaultOptimalEpochs=None,
                   # tfBackend=None, clearTF=None,
                   ):
    """
        modelBuilder is a function which must return a keras model.
        modelBuilderKwargs are its parameters.
    """
    if patience <= 1:
        logWarning("The patience is set to " + str(patience) + ", you should set it greater...", logger, verbose=verbose)
    if modelBuilderKwargs is None:
        modelBuilderKwargs = {}
    if estimateOptimalEpochsKwargs is None:
        estimateOptimalEpochsKwargs = {}
    # We propagate estimateOptimalEpochsKwargs:
    assert "patience" not in estimateOptimalEpochsKwargs
    estimateOptimalEpochsKwargs["patience"] = patience
    assert "batchSize" not in estimateOptimalEpochsKwargs
    estimateOptimalEpochsKwargs["batchSize"] = batchSize
    # We split data:
    kfold = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=randomState)
    i = 0
    cvScores = []
    pbar = ProgressBar(cv, logger=logger, verbose=verbose)
    if doPltShow or graphsDir is not None:
        plt.figure()
        plt.clf()
    for trainIds, testIds in kfold.split(x, y):
        # We get xTrain, yTrain, xTest and yTest:
        log("Starting kfold " + str(i) + "...", logger, verbose=verbose)
        xTrain = []
        yTrain = []
        for id in trainIds:
            xTrain.append(x[id])
            label = y[id]
            if labelEncoder is not None:
                label = labelEncoder[label]
            yTrain.append(label)
        xTrain = np.array(xTrain)
        yTrain = np.array(yTrain)
        xTest = []
        yTest = []
        for id in testIds:
            xTest.append(x[id])
            label = y[id]
            if labelEncoder is not None:
                label = labelEncoder[label]
            yTest.append(label)
        xTest = np.array(xTest)
        yTest = np.array(yTest)
        # We clean the tf session:
        # if clearTF is not None:
        #     clearTF()
        # if tfBackend is not None:
        #     log("Cleaning tf session...", logger, verbose=verbose)
        #     tfBackend.clear_session()
        optimalEpochs = None
        if defaultOptimalEpochs is not None:
            optimalEpochs = defaultOptimalEpochs
        if optimalEpochs is None:
            # logError("TODO corriger ici, faire 2 variable, une pour les params, une pour la boucle", logger)
            # We get the model:
            model = modelBuilder(**modelBuilderKwargs)
            # We estimate the best amount of epochs:
            optimalEpochs, history = estimateOptimalEpochs\
            (
                model,
                xTrain,
                yTrain,
                logger=logger,
                verbose=False,
                kerasVerbose=kerasVerbose,
                **estimateOptimalEpochsKwargs,
            )
            if doPltShow or graphsDir is not None:
                plt.clf()
                plt.plot(history.history['val_acc'])
                plt.plot(history.history['acc'])
                plt.title('Model accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Test', 'Train'], loc='upper left')
            if graphsDir is not None:
                plt.savefig(graphsDir + "/" + getDateSec() + ".png", format='png')
            if doPltShow:
                plt.show()
            log("Optimal epochs is: " + str(optimalEpochs), logger, verbose=verbose)
        # We clean the tf session:
        # if clearTF is not None:
        #     clearTF()
        # if tfBackend is not None:
        #     log("Cleaning tf session...", logger, verbose=verbose)
        #     tfBackend.clear_session()
        # We get a new model and train it on all data:
        model = modelBuilder(**modelBuilderKwargs)
        history = model.fit\
        (
            xTrain, yTrain,
            epochs=optimalEpochs,
            batch_size=batchSize,
            verbose=1 if kerasVerbose else 0,
        )
        if doPltShow or graphsDir is not None:
            plt.clf()
            plt.plot(history.history['acc'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train'], loc='upper left')
        if graphsDir is not None:
            plt.savefig(graphsDir + "/" + getDateSec() + ".png", format='png')
        if doPltShow:
            plt.show()
        # We evaluate the model on the test set:
        scores = model.evaluate(xTest, yTest, verbose=0)
        # We get the acc:
        accIndex = None
        currentIndex = 0
        for key in model.metrics_names:
            if key == "acc":
                accIndex = currentIndex
                break
            currentIndex += 1
        # We clean the tf session:
        # if clearTF is not None:
        #     clearTF()
        # if tfBackend is not None:
        #     log("Cleaning tf session...", logger, verbose=verbose)
        #     tfBackend.clear_session()
        # We take the acc score:
        score = scores[accIndex]
        # We add the current fold score:
        cvScores.append(score)
        # We print end of the fold:
        log("Fold " + str(i) + " done with score: " + str(score), logger, verbose=verbose)
        # We inc the fold:
        i += 1
        pbar.tic("Cross-validation")
    if doPltShow or graphsDir is not None:
        plt.close()
    cvScores = np.array(cvScores)
    currentMean = cvScores.mean()
    currentConfidence = cvScores.std() * 2
    currentResult = {"accuracy": {"score": currentMean, "confidence": currentConfidence}}
    return currentResult




# Reset Keras Session
def resetKeras():
    # https://github.com/keras-team/keras/issues/12625
    sess = K.tensorflow_backend.get_session()
    K.tensorflow_backend.clear_session()
    sess.close()
    sess = K.tensorflow_backend.get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    gc.collect() # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    
    # config.gpu_options.per_process_gpu_memory_fraction = 1 # deleted
    # config.gpu_options.visible_device_list = "0" # deleted
    
    K.tensorflow_backend.set_session(tensorflow.Session(config=config))