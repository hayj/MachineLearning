

isNotebook = '__file__' not in locals()
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from newssource.asattribution.asamin import *
from newssource.asa.asapreproc import *
from databasetools.mongo import mongoStorable
from systemtools.basics import *
from systemtools.file import *
logger = None
# Parameters:
vector_size = 300
window = 4
min_count = 10
epochs = 15
negative = 15
storeEntire = False
filesRatio = None
tokensCol = "textSentences" # 3gramsFiltered, 2gramsFiltered, 1gramsFiltered, textSentences
if lri():
    # datasetPath = dataDir() + "/Asa/asaminbis/asamin-train-whiteasarelevance0.8-2019.05.26-17.18"
    # datasetPath = dataDir() + "/Asa/asaminbis/asamin-train-2019.05.22-19.47"
    datasetPath = rtmpDir() + "/asamin-train-textSentences-whiteasarelevance0.8-2019.05.27-10.17"
    # datasetPath = rtmpDir() + "/asamin-train-3gramsFiltered-whiteasarelevance0.8-2019.05.27-10.16"
    outputPath = nosaveDir() + "/d2v"
elif octods():
    datasetPath = dataDir() + "/Asa/asaminbis/asamin-train-2019.05.22-19.47"
    outputPath = homeDir() + "/d2v"
elif hjlat():
    datasetPath = "/home/hayj/tmp/asaminbis-for-d2v"
    outputPath = tmpDir("d2v")
else:
    logError("Host unknown...", logger)
    exit()
outputPath = outputPath + "/d2vmodel-" + tokensCol
outputPath += "-datasetsize" + str(truncateFloat(getSize(datasetPath, unit='g'), 1)) + "g"
outputPath += "-vectorsize" + str(vector_size)
outputPath += "-window" + str(window)
outputPath += "-negative" + str(negative)    
if isNotebook:
    outputPath += "-notebook"
mkdir(outputPath)
logger = Logger(outputPath + "/d2v-train-" + getHostname() + ".log")
tt = TicToc(logger=logger)
tt.tic()
# We make the config object:
config = \
{
    "d2vParams": \
    {
        "vector_size": vector_size,
        "window": window,
        "min_count": min_count,
        "epochs": epochs,
        "negative": negative,
    },
    "tokensCol": tokensCol,
    "filesRatio": filesRatio,
    "datasetPath": datasetPath,
    "timestamp": time.time(),
    "hostname": getHostname(),
    "outputPath": outputPath,
    "storeEntire": storeEntire,
    "punct": {',', '...', '(', ';', ':', "'", ')', '?', '!', '-', '"', '.'},
}
log(lts(config), logger)
toJsonFile(mongoStorable(config), outputPath + "/config.json")
# We init AsaPreproc:
# documents = Gen2Iter(asaGenerator, tokensCol, datasetPath=datasetPath, filesRatio=filesRatio, logger=logger)
trainFilesPath = sortedGlob(datasetPath + "/*.bz2")
if filesRatio is not None:
    trainFilesPath = trainFilesPath[:int(len(trainFilesPath) * filesRatio)]
validationFilesPath = trainFilesPath
asap = AsaPreproc\
(
    trainFilesPath,
    validationFilesPath,
    config["tokensCol"],
    True,
    True,
    True,
    config["punct"],
    3,
    1,
    None,
    1,
    600,
    True,
    trainSamplesCount=None,
    validationSamplesCount=None,
    vocIndex=None,
    encodedAds=None,
    logger=logger,
    verbose=True,
)
datasetPathSize = getSize(datasetPath, unit='g')
documents = asap.getDoc2VecTrainCI()
model = Doc2Vec(documents, workers=cpuCount(), **config["d2vParams"])



