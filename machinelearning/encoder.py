from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from systemtools.location import *
from systemtools.printer import *
from systemtools.basics import *
import numpy as np
import copy
import random
from nlptools.embedding import *
from machinelearning.iterator import *
from machinelearning import config as mlConf


def encodeSample\
(
	sample,
	vocIndex=None,
	docLength=None,
	encode=True,
	pad=True,
	padding='pre',
	truncating='post',
	encoding="index", # embedding # TODO
	encodeLabel=True, # TODO
	wordEmbeddings=None, # TODO
	labelEncoder=None, # TODO
	logger=None, verbose=True,
):
	"""
		This function encode an input (tokens, label)
	"""
	(tokens, label) = sample
	mask = None
	if encode:
		newTokens = []
		if encoding == "index":
			encoder = vocIndex
			mask = 0
			oovElement = 1
		else:
			encoder = wordEmbeddings
			mask = mlConf.MASK_EMBEDDING_FUNCTION(len(wordEmbeddings["the"]))
			oovElement = mlConf.OOV_EMBEDDING_FUNCTION(len(wordEmbeddings["the"]))
		assert encoder is not None
		for word in tokens:
			if word in encoder:
				newTokens.append(encoder[word])
			else:
				newTokens.append(oovElement)
		tokens = np.array(newTokens)
	else:
		mask = mlConf.MASK_TOKEN
	if pad:
		wasArray = isinstance(tokens, np.ndarray)
		tokens = padSequence(list(tokens), docLength, padding=padding, truncating=truncating, value=mask, removeEmptySentences=True)
		if wasArray:
			tokens = np.array(tokens)
	if encodeLabel and labelEncoder is not None:
		label = labelEncoder[label]
	return (tokens, label)


def encodeMulticlassLabels(labels, encoding='index', logger=None, verbose=True):
	"""
		:arg: encoding: index or onehot
		:example:
		>>> encodeMulticlassLabels(['e', 'b', 'e', 'o', 'e', 'b'], encoding='onehot')
		array([[0., 1., 0.],
			   [1., 0., 0.],
			   [0., 1., 0.],
			   [0., 0., 1.],
			   [0., 1., 0.],
			   [1., 0., 0.]], dtype=float32)
	"""
	if encoding == 'onehot':
		# Encode class values as integers:
		encoder = LabelEncoder()
		encoder.fit(labels)
		encodedY = encoder.transform(labels)
		# Convert integers to dummy variables (i.e. one hot encoded):
		return (encoder.classes_, np_utils.to_categorical(encodedY))
	elif encoding == 'index':
		return np.unique(labels, return_inverse=True)
	else:
		raise Exception("Please choose a right encoding")


class TextEncoder:
	def __init__\
	(
		self,

		files,
		samplesGenerator,
		samplesGeneratorArgs=(),
		samplesGeneratorKwargs=dict(),
		filesRatio=None,

		split=[1.0], # [1.0], [0.8, 0.1, 0.1], [0.1] * 10
		persist=None,

		labelEncoder=None,
		computeLabelEncoder=True,
		labelEncoding='onehot', # onehot, index

		padding='pre',
		truncating='post',

		encoding="index", # index, embedding

		wordEmbeddings=None,

		vocIndex=None,
		minVocDF=3,

		batchSize=128,

		samplesCounts=None,

		docLength=1200,

		seed=0,

		logger=None,
		verbose=True,
	):
		# We retain params:
		self.logger = logger
		self.verbose = verbose
		self.files = files
		self.samplesGenerator = samplesGenerator
		self.samplesGeneratorArgs = samplesGeneratorArgs
		self.samplesGeneratorKwargs = samplesGeneratorKwargs
		self.filesRatio = filesRatio
		self.split = split
		self.persist = persist
		self.labelEncoder = labelEncoder
		self.computeLabelEncoder = computeLabelEncoder
		self.labelEncoding = labelEncoding
		self.padding = padding
		self.truncating = truncating
		self.encoding = encoding
		self.wordEmbeddings = wordEmbeddings
		self.vocIndex = vocIndex
		self.minVocDF = minVocDF
		self.batchSize = batchSize
		self.samplesCounts = samplesCounts
		self.docLength = docLength
		# We build word embeddings:
		if self.wordEmbeddings is None:
			self.wordEmbeddings = Embeddings("test").getVectors()
		self.wordEmbeddings = copy.copy(self.wordEmbeddings)
		# We init a random object for split and other things:
		self.rnd = random.Random(seed)
		# We split files:
		try:
			self.rnd.shuffle(self.files)
			if self.filesRatio is not None:
				self.files = self.files[:int(len(self.files) * self.filesRatio)]
			self.parts = []
			if self.split is None:
				self.split = [1.0]
			if len(self.files) < len(self.split):
				self.files = files[:len(self.split)]
			assert len(set(self.files)) == len(self.files)
			self.parts = ratioSplit(self.files, self.split)
			assert len(set(flattenLists(self.parts))) == len(set(self.files))
		except Exception as e:
			logException(e, self)
			message = "Cannot split files of length " + str(len(self.files))
			message += " with the split schema " + str(self.split)
			if self.filesRatio is not None:
				message += " with files ratio to keep of " + str(self.filesRatio) + " which reduced files from " + str(len(files)) + " to " + str(len(self.files)) + " items"
			raise Exception(message)
		# We handle the persistence:
		self.cache = None
		if self.persist is None:
			self.persist = [False] * len(self.parts)
		self.__cache()
		# We build all:
		self.__build()
		self.embeddingMatrix = None
		# We show parts:
		self.show()

	def show(self):
		message = "Dataset parts:\n"
		for i in range(len(self.parts)):
			partLen = len(self.parts[i])
			partRatio = self.split[i]
			persist = self.persist[i]
			samples = self.getSamplesCount(i)
			message += " [" + str(i) + "] "
			message += str(partLen) + " files"
			message += ", " + str(partRatio) + " ratio"
			message += ", " + str(samples) + " samples"
			if persist:
				message += " (cached)"
			message += "\n"
		message = message[:-1]
		log(message, self)

	def __cache(self):
		if self.cache is None:
			self.cache = [None] * len(self.parts)
			for partIndex in range(len(self.parts)):
				persist = self.persist[partIndex]
				part = self.parts[partIndex]
				currentCache = []
				if persist:
					log("Starting to cache the part " + str(partIndex) + "...", self)
					currentCache = [row for row in self.getPart\
					(
						partIndex,
						pad=False, encodeLabel=False, encode=False,
					)]
					self.cache[partIndex] = currentCache
					log("Part " + str(partIndex) + " cached.", self)



	def __build(self):
		if (self.computeLabelEncoder and self.labelEncoder is None) \
		or self.samplesCounts is None \
		or self.vocIndex is None:
			log("Starting to build voc index and samples counts...", self)
			# We count samples, get labels and collect vocabulary:
			embUnknownTokens = 0
			tokensCount = 0
			labels = set()
			vocDF = dict()
			self.samplesCounts = [0] * len(self.parts)
			for partIndex in range(len(self.parts)):
				part = self.parts[partIndex]
				data = self.getPart(partIndex, pad=False, encodeLabel=False, encode=False)
				for tokens, label in data:
					tokens = set(tokens)
					for token in tokens:
						tokensCount += 1
						if token not in self.wordEmbeddings:
							embUnknownTokens += 1
						if token not in vocDF:
							vocDF[token] = 0
						vocDF[token] += 1
					labels.add(label)
					self.samplesCounts[partIndex] += 1
			# Print some infos:
			# log("Samples counts are " + str(self.samplesCounts) + " (total " + str(sum(self.samplesCounts)) + ")", self)
			log(str(truncateFloat(embUnknownTokens / tokensCount * 100, 2)) + "% of tokens are not in word embeddings", self)
			log("Vocabulary size without minVocDF filtering: " + str(len(vocDF)), self)
			# We count the number of voc element which is not in wordEmbeddings:
			c = 0
			for w in vocDF.keys():
				if w not in self.wordEmbeddings:
					c += 1
			log(str(truncateFloat(c / len(vocDF) * 100, 2)) + "% of vocabulary are not in word embeddings", self)
			# We get the voc according to minVocDF:
			voc = set()
			for word, count in vocDF.items():
				if count >= self.minVocDF:
					voc.add(word)
			# We print informations:
			log("Vocabulary size with a minVocDF of " + str(self.minVocDF) + ": " + str(len(voc)), self)
			# We count the number of voc element which is not in wordEmbeddings:
			c = 0
			for w in voc:
				if w not in self.wordEmbeddings:
					c += 1
			log("With a minVocDF of " + str(self.minVocDF) + ", " + str(truncateFloat(c / len(voc) * 100, 2)) + "% of vocabulary are not in word embeddings", self)
			# We add the voc of wordEmbeddings:
			for word in self.wordEmbeddings.keys():
				voc.add(word)
			# We print informations:
			log("Vocabulary size with wordEmbeddings: " + str(len(voc)), self)
			# We convert the voc to a dict of indexes:
			self.vocIndex = dict()
			# First we add a MASK:
			self.vocIndex[mlConf.MASK_TOKEN] = 0
			self.vocIndex[mlConf.OOV_TOKEN] = 1
			i = 2
			for word in sorted(list(voc)):
				self.vocIndex[word] = i
				i += 1
			# We add generate embedding for unknown words:
			for w in self.vocIndex.keys():
				if w not in self.wordEmbeddings:
					self.wordEmbeddings[w] = np.random.normal(scale=0.6, size=(self.getEmbeddingsDimension(),))
			# And finally we build the label encoder:
			if self.computeLabelEncoder and self.labelEncoder is None:
				labels = list(labels)
				try:
					labels = sorted(labels)
				except: pass
				encodedLabels = encodeMulticlassLabels(labels, encoding=self.labelEncoding, logger=self.logger, verbose=self.verbose)
				self.labelEncoder = dict()
				assert len(encodedLabels) == len(labels)
				for i in range(len(encodedLabels)):
					self.labelEncoder[labels[i]] = encodedLabels[i]

	def encodeLabel(self, label):
		return self.labelEncoder[label]
	def decodeLabel(self, enc):
		for label, currentEnc in self.labelEncoder.items():
			if np.array_equal(currentEnc, enc):
				return label
		return None
	def getVocIndex(self):
		return self.vocIndex
	def getWordEmbeddings(self):
		return self.wordEmbeddings
	def getDocLength(self):
		return self.docLength
	def getSamplesCount(self, index):
		return self.samplesCounts[index]
	def getEmbeddingsDimension(self):
		return len(self.wordEmbeddings["the"])
	def getEmbeddingMatrix(self):
		if self.embeddingMatrix is None:
			log("We generate the embedding matrix...", self)
			embeddingMatrix = np.zeros((len(self.getVocIndex()), self.getEmbeddingsDimension()))
			for word, i in self.getVocIndex().items():
				embeddingMatrix[i] = self.wordEmbeddings[word]
			# We check the % unknown words:
			self.embeddingMatrix = embeddingMatrix
		return self.embeddingMatrix


	def getPart(self, index, *args, **encodeSampleKwargs):
		files = self.parts[index]
		if "logger" in encodeSampleKwargs:
			del encodeSampleKwargs["logger"]
		if "verbose" in encodeSampleKwargs:
			del encodeSampleKwargs["verbose"]
		if "encoding" not in encodeSampleKwargs:
			encodeSampleKwargs["encoding"] = self.encoding
		if "padding" not in encodeSampleKwargs:
			encodeSampleKwargs["padding"] = self.padding
		if "truncating" not in encodeSampleKwargs:
			encodeSampleKwargs["truncating"] = self.truncating
		if "docLength" not in encodeSampleKwargs:
			encodeSampleKwargs["docLength"] = self.docLength
		encodeSampleKwargs["vocIndex"] = self.vocIndex
		encodeSampleKwargs["wordEmbeddings"] = self.wordEmbeddings
		encodeSampleKwargs["labelEncoder"] = self.labelEncoder
		if self.cache[index] is not None:
			def __gen(data, *args, **kwargs):
				for row in data:
					row = encodeSample(row, *args, **kwargs)
					yield row
			data = self.cache[index]
			return AgainAndAgain\
			(
				__gen,
				data,
				*args,
				logger=self.logger, verbose=self.verbose,
				**encodeSampleKwargs,
			)
		else:
			return AgainAndAgain\
			(
				ConsistentIterator,
				files,
				self.samplesGenerator,
				itemGeneratorArgs=self.samplesGeneratorArgs,
				itemGeneratorKwargs=self.samplesGeneratorKwargs,
				subProcessParseFunct=encodeSample,
				subProcessParseFunctKwargs=encodeSampleKwargs,
				logger=self.logger,
				verbose=self.verbose,
			)

	def getInfiniteBatcher(self, index, *args, skip=0, shuffle=0, **kwargs):
		ci = self.getPart(index, *args, **kwargs)
		return InfiniteBatcher(ci, batchSize=self.batchSize, 
							shuffle=shuffle,
							skip=skip,
							logger=self.logger, verbose=self.verbose)

	def getBatchesCount(self, index):
		c = self.getSamplesCount(index)
		return math.ceil(c / self.batchSize)






if __name__ == '__main__':
	files = sortedGlob(tmpDir("mbti-datasets") + "/*/*.bz2")[:5]
	te = TextEncoder(files, split=[0.8, 0.1, 0.1])