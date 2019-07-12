from systemtools.duration import *
from systemtools.basics import *
from systemtools.file import *
from systemtools.location import *
from systemtools.printer import *
from systemtools.system import *
from systemtools.logger import *
from nlptools.utils import *
from nlptools.tokenizer import *
from datatools.jsonutils import *
from datatools.url import *
from datastructuretools.trie import *
from datastructuretools.processing import *
from nlptools.pipeline1 import newscleaner
from nlptools.pipeline1 import pipeline
from nlptools import tokenizer
from nlptools.embedding import *
from nlptools.preprocessing import *
from machinelearning.iterator import *
from machinelearning.utils import *
import random
from keras.preprocessing.sequence import pad_sequences


def encodeTokens(tokens, *args, vocIndex=None, docLength=None, **kwargs):
	newTokens = []
	for word in tokens:
		if word in vocIndex:
			newTokens.append(vocIndex[word])
	tokens = newTokens
	tokens = pad_sequences([tokens], maxlen=docLength,
					   padding='pre', truncating='post')[0]
	return tokens

def lowerTokens(tokens):
	if tokens is None or not isinstance(tokens, list):
		raise Exception("lowerTokens: tokens must be a list of strings or a list of lists of strings... Got " + str(tokens))
	if len(tokens) == 0:
		return tokens
	if isinstance(tokens[0], str):
		for i in range(len(tokens)):
			tokens[i] =tokens[i].lower()
	else:
		for i in range(len(tokens)):
			tokens[i] = lowerTokens(tokens[i])
	return tokens

def filterPunctAndNonWord(tokens, punct=None, letterRegex=None, logger=None, verbose=True):
	if letterRegex is None:
		letterRegex = re.compile("[a-zA-Z]")
	if punct is None:
		punct = {',', ')', '...', "'", ';', '-', '!', ':', '?', '"', '.', '('}
	if tokens is None or not isinstance(tokens, list):
		raise Exception("filterPunctAndNonWord: tokens must be a list of strings or a list of lists of strings... Got " + str(tokens))
	if len(tokens) == 0:
		return tokens
	if isinstance(tokens[0], str):
		newTokens = []
		for i in range(len(tokens)):
			if tokens[i] in punct or letterRegex.search(tokens[i]):
				newTokens.append(tokens[i])
		tokens = newTokens
	else:
		for i in range(len(tokens)):
			tokens[i] = filterPunctAndNonWord(tokens[i], punct, letterRegex)
	return tokens

def itemGen(container, doLower, doFlattenSentences, *args, **kwargs):
	for file in container:
		content = fileToStr(file)
		content = preprocess\
		(
			content,
			removeHtml=True,
			doRemoveUrls=True,
			doTokenizingHelp=True,
			doReduceCharSequences=True,
			doQuoteNormalization=True,
			doReduceBlank=True,
			keepNewLines=False,
			stripAccents=True,
		)
		content = sentenceTokenize(content)
		content = filterPunctAndNonWord(content)
		if doFlattenSentences:
			content = flattenLists(content)
		if doLower:
			content = lowerTokens(content)
		label = 0 if "/neg/" in file else 1
		yield (content, label)

class IMDB:
	def __init__\
	(
		self,
		path=None,
		wordEmbeddings=None,
		minVocDF=1,
		useMasking=True,
		fileRatio=None,
		doLower=None,
		docLength=600,
		seed=1,
		toNumpyArray=True,
		logger=None,
		verbose=True,
	):
		self.logger = logger
		self.verbose = verbose
		self.path = path
		self.fileRatio = fileRatio
		self.wordEmbeddings = wordEmbeddings
		self.minVocDF = minVocDF
		self.useMasking = useMasking
		self.doLower = doLower
		self.docLength = docLength
		self.seed = seed
		self.toNumpyArray = toNumpyArray
		if self.wordEmbeddings is None:
			logWarning("You didn't give word embedding so we will take a few embedding for test...", self)
			self.wordEmbeddings = Embeddings("test", logger=self.logger, verbose=self.verbose).getVectors()
		self.getPath()
		self.sd = SerializableDict("imdb-dataset-path", funct=sortedGlob, cacheCheckRatio=0.0, serializeEachNAction=1, logger=self.logger, verbose=self.verbose)
		self.tt = TicToc(logger=self.logger)
		self.trainPosFiles = self.sd[self.path + "/train/pos/*.txt"]
		self.trainNegFiles = self.sd[self.path + "/train/neg/*.txt"]
		self.testPosFiles = self.sd[self.path + "/test/pos/*.txt"]
		self.testNegFiles = self.sd[self.path + "/test/neg/*.txt"]
		if self.fileRatio is not None:
			self.trainPosFiles = self.trainPosFiles[:int(self.fileRatio * len(self.trainPosFiles))]
			self.trainNegFiles = self.trainNegFiles[:int(self.fileRatio * len(self.trainNegFiles))]
			self.testPosFiles = self.testPosFiles[:int(self.fileRatio * len(self.testPosFiles))]
			self.testNegFiles = self.testNegFiles[:int(self.fileRatio * len(self.testNegFiles))]
		self.trainTokens = None
		self.trainEncodedTokens = None
		self.trainLabels = None
		self.testTokens = None
		self.testEncodedTokens = None
		self.testLabels = None
		self.vocIndex = None
		self.embeddingMatrix = None
		self.__load()

	def getDocLength(self):
		return self.docLength

	def __load(self, *args, **kwargs):
		self.loadTrain(*args, **kwargs)
		self.loadTest(*args, **kwargs)
		self.getTrainEncodedTokens(*args, **kwargs)
		self.getTestEncodedTokens(*args, **kwargs)
		self.getEmbeddingMatrix()

	def getVocIndex(self):
		if self.vocIndex is None:
			log("We generate vocIndex", self)
			# We get the words document frequencies:
			vocDF = dict()
			for tokens in self.getTrainTokens() + self.getTestTokens():
				alreadySeenInThisDoc = set()
				for token in tokens:
					if token not in alreadySeenInThisDoc:
						if token not in vocDF:
							vocDF[token] = 0
						vocDF[token] += 1
						alreadySeenInThisDoc.add(token)
			# We get the voc according to minVocDF:
			voc = set()
			for word, count in vocDF.items():
				if count >= self.minVocDF:
					voc.add(word)
			# We print informations:
			log("Entire vocabulary length: " + str(len(vocDF)), self)
			log("Vocabulary length with minVocDF=" + str(self.minVocDF) + ": " + str(len(voc)), self)
			# We add the voc of wordEmbeddings:
			for word in self.wordEmbeddings.keys():
				voc.add(word)
			# We print informations:
			log("Vocabulary length with wordEmbeddings: " + str(len(voc)), self)
			# We convert the voc to a dict of indexes:
			self.vocIndex = dict()
			# First we add a MASK:
			if self.useMasking:
				self.vocIndex[MASK_TOKEN] = 0
				i = 1
			else:
				i = 0
			for word in sorted(list(voc)):
				self.vocIndex[word] = i
				i += 1
		return self.vocIndex

	def getEmbeddingsDimension(self):
		return len(self.wordEmbeddings["the"])

	def getEmbeddingMatrix(self):
		if self.embeddingMatrix is None:
			log("We generate the embedding matrix", self)
			unknownWordCount = 0
			embeddingMatrix = np.zeros((len(self.getVocIndex()), self.getEmbeddingsDimension()))
			unknownWords = set()
			for word, i in self.getVocIndex().items():
				embeddingVector = self.wordEmbeddings.get(word, None)
				if embeddingVector is None:
					embeddingMatrix[i] = np.random.normal(scale=0.6, size=(self.getEmbeddingsDimension(),))
					unknownWordCount += 1
					unknownWords.add(word)
				else:
					embeddingMatrix[i] = embeddingVector
			# We check the % unknown words:
			log(str(truncateFloat(unknownWordCount / len(self.getVocIndex()) * 100, 2)) + "% of words are not in word embeddings.", self)
			self.embeddingMatrix = embeddingMatrix
		return self.embeddingMatrix


	def getTrainTokens(self, *args, addMasks=False, **kwargs):
		self.loadTrain(*args, **kwargs)
		if addMasks:
			return padSequences(self.trainTokens, maxlen=self.docLength,
					   padding='pre', truncating='post', value=MASK_TOKEN)
		else:
			return self.trainTokens
	def getTrainEncodedTokens(self, *args, **kwargs):
		if self.trainEncodedTokens is None:
			log("We generate trainEncodedTokens", self)
			self.loadTrain(*args, **kwargs)
			self.trainEncodedTokens = []
			for tokens in self.getTrainTokens():
				self.trainEncodedTokens.append(encodeTokens(tokens, vocIndex=self.getVocIndex(), docLength=self.docLength))
			if self.toNumpyArray:
				self.trainEncodedTokens = np.array(self.trainEncodedTokens)
		return self.trainEncodedTokens
	def getTrainLabels(self, *args, **kwargs):
		self.loadTrain(*args, **kwargs)
		return self.trainLabels

	def getTestTokens(self, *args, addMasks=False, **kwargs):
		self.loadTest(*args, **kwargs)
		if addMasks:
			return padSequences(self.testTokens, maxlen=self.docLength,
					   padding='pre', truncating='post', value=MASK_TOKEN)
		else:
			return self.testTokens
	def getTestEncodedTokens(self, *args, **kwargs):
		if self.testEncodedTokens is None:
			log("We generate testEncodedTokens", self)
			self.loadTest(*args, **kwargs)
			self.testEncodedTokens = []
			for tokens in self.getTestTokens():
				self.testEncodedTokens.append(encodeTokens(tokens, vocIndex=self.getVocIndex(), docLength=self.docLength))
			if self.toNumpyArray:
				self.testEncodedTokens = np.array(self.testEncodedTokens)
		return self.testEncodedTokens
	def getTestLabels(self, *args, **kwargs):
		self.loadTest(*args, **kwargs)
		return self.testLabels

	def loadTrain(self, *args, **kwargs):
		if self.trainTokens is None:
			log("We generate the train set", self)
			(self.trainTokens, self.trainLabels) = self.__getData(self.trainPosFiles + self.trainNegFiles, *args, **kwargs)
			if self.toNumpyArray:
				# self.trainTokens = np.array(self.trainTokens)
				self.trainLabels = np.array(self.trainLabels)

	def loadTest(self, *args, **kwargs):
		if self.testTokens is None:
			log("We generate the test set", self)
			(self.testTokens, self.testLabels) = self.__getData(self.testPosFiles + self.testNegFiles, *args, **kwargs)
			if self.toNumpyArray:
				# self.testTokens = np.array(self.testTokens)
				self.testLabels = np.array(self.testLabels)

	def getData(self, *args, **kwargs):
		if self.toNumpyArray:
			return \
			(
				np.concatenate(self.trainTokens, self.testTokens),
				np.concatenate(self.trainEncodedTokens, self.testEncodedTokens),
				np.concatenate(self.trainLabels, self.testLabels),
			)
		else:
			return \
			(
				self.trainTokens + self.testTokens,
				self.trainEncodedTokens + self.testEncodedTokens,
				self.trainLabels + self.testLabels,
			)

	def __getSentences(self, files, *args, addMasks=False, doLower=False, doFlattenSentences=False, **kwargs):
		result = self.__getData(files, *args, doLower=doLower, doFlattenSentences=doFlattenSentences, **kwargs)[0]
		if addMasks:
			for i in range(len(result)):
				current = result[i]
				nbTokens = len(flattenLists(current))
				# print(nbTokens)
				nbMasks = self.docLength - nbTokens
				if nbMasks > 0:
					# print(nbMasks)
					currentMasks = [MASK_TOKEN] * nbMasks
					# print(currentMasks)
					# print("len(current): " + str(len(current)))
					if "padding" in kwargs and kwargs["padding"] == "post":
						current.append(currentMasks)
					else:
						current.insert(0, currentMasks)
					# print("len(current): " + str(len(current)))
					result[i] = current
		return result
	def getTrainSentences(self, *args, **kwargs):
		files = self.trainPosFiles + self.trainNegFiles
		return self.__getSentences(files, *args, **kwargs)
	def getTestSentences(self, *args, **kwargs):
		files = self.testPosFiles + self.testNegFiles
		return self.__getSentences(files, *args, **kwargs)
	def getSentences(self, *args, **kwargs):
		return self.getTrainSentences(*args, **kwargs) + self.getTestSentences(*args, **kwargs)


	def __getData(self, files, doLower=None, doFlattenSentences=None):
		if doLower is None:
			doLower = self.doLower
		if doFlattenSentences is None:
			doFlattenSentences = True
		rd = random.Random(self.seed)
		rd.shuffle(files)
		ci = ConsistentIterator(chunks(files, 100), itemGen, itemGeneratorArgs=(doLower, doFlattenSentences))
		tokenss = []
		labels = []
		for tokens, label in ci:
			tokenss.append(tokens)
			labels.append(label)
		return (tokenss, labels)

	def getPath(self):
		if self.path is None:
			self.path = tmpDir() + "/imdb-dataset"
		if isDir(self.path) and not isDir(self.path + "/test"):
			raise Exception("Please delete " + self.path)
		if not isDir(self.path):
			log("Starting to download imdb dataset...", self)
			filePath = download("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")
			extractedDirPath = extract(filePath)
			remove(filePath)
			(path, _, _, dirName) = decomposePath(extractedDirPath)
			newDirPath = path + "/" + "imdb-dataset"
			rename(extractedDirPath, newDirPath)
			self.path = move(newDirPath, tmpDir())

def test1():
	imdb = IMDB(docLength=100, doLower=True, fileRatio=0.05)
	sentences = imdb.getTestSentences()
	for i in range(10):
		print(detokenize(sentences[i]))

def test2():
	imdb = IMDB(docLength=300, doLower=True, fileRatio=0.05)
	for addMasks in [False, True]:
		sentences = imdb.getTestSentences(addMasks=addMasks)
		for i in range(2):
			print(detokenize(sentences[i]))
			print()
		print()
		print()

if __name__ == '__main__':
	test2()