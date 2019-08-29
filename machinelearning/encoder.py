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
	doLower=False,
	vocIndex=None,
	docLength=None,
	encode=True,
	pad=True,
	truncate=True,
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
		pad will truncate and pad sequences
		# TODO encoding of sequence given a function... (for ELMO models for example)
	"""
	(tokens, label) = sample
	if doLower and tokens is not None and len(tokens) > 0 and isinstance(tokens[0], str):
		newTokens = []
		for token in tokens:
			newTokens.append(token.lower())
		tokens = newTokens
	mask = None
	encoder = None
	if encoding == "index":
		encoder = vocIndex
	elif encoding == "embedding":
		encoder = wordEmbeddings
	if encode:
		if encoder is None:
			raise Exception("Unknown token encoding")
		mask = encoder[mlConf.MASK_TOKEN]
		oovElement = encoder[mlConf.OOV_TOKEN]
		newTokens = []
		for word in tokens:
			if word in encoder:
				newTokens.append(encoder[word])
			else:
				newTokens.append(oovElement)
		tokens = np.array(newTokens)
	else:
		mask = mlConf.MASK_TOKEN
		oovElement = mlConf.OOV_TOKEN
		newTokens = []
		for word in tokens:
			if vocIndex is None or word in vocIndex:
				newTokens.append(word)
			else:
				newTokens.append(oovElement)
		tokens = newTokens
	if pad and docLength is not None:
		wasArray = isinstance(tokens, np.ndarray)
		tokens = padSequence(list(tokens), docLength, padding=padding, truncating=truncating, value=mask, removeEmptySentences=True)
		if wasArray:
			tokens = np.array(tokens)
		assert len(tokens) > 0 and len(tokens) <= docLength
	elif truncate:
		if truncating == "post":
			tokens = tokens[:docLength]
		else:
			tokens = tokens[:-docLength]
		assert len(tokens) > 0 and len(tokens) <= docLength
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
	"""
		This class encode text for machine learning libs. You must give a list of files and a generator function which have to yield (tokens, label). tokens must be non-lowered.
	"""
	def __init__\
	(
		self,

		files,
		samplesGenerator,
		samplesGeneratorArgs=(),
		samplesGeneratorKwargs=dict(),
		filesRatio=None,

		doLower=None, # Automatically set according to word embeddings

		split=[1.0], # [1.0], [0.8, 0.1, 0.1], [0.1] * 10
		persist=None,

		computeLabelEncoder=True,
		labelEncoding='onehot', # onehot, index

		padding='pre',
		truncating='post',

		encoding="index", # index, embedding

		prebuilt=None,

		samplesCounts=None,
		vocIndex=None,
		labelEncoder=None,
		wordEmbeddings=None,

		minVocDF=2, # min voc document frequency
		minVocLF=2, # min voc label frequency

		batchSize=128,

		docLength=1200,

		seed=0,

		logger=None,
		verbose=True,

		alwaysTruncateRawData=False,
	):
		"""
			files can be a list of list of files.
			O a list of files and you can set split as [0.8, 0.1, 0.1] for example to have train, val, test.
			persist allow you to cache parts of your dataset, e.g. [False, True, True]
		"""
		# We store the logger:
		self.logger = logger
		self.verbose = verbose
		# If check for prebuilt data:
		if prebuilt is not None:
			try:
				if isinstance(prebuilt, str):
					assert isFile(prebuilt)
					prebuilt = deserialize(prebuilt)
				if prebuilt["samplesCounts"] is not None:
					samplesCounts = prebuilt["samplesCounts"]
				if prebuilt["vocIndex"] is not None:
					vocIndex = prebuilt["vocIndex"]
				if prebuilt["labelEncoder"] is not None:
					labelEncoder = prebuilt["labelEncoder"]
				if prebuilt["wordEmbeddings"] is not None:
					wordEmbeddings = prebuilt["wordEmbeddings"]
			except Exception as e:
				logException(e, self, message="Cannot parse prebuilt data")
		# We retain params:
		self.files = files
		self.samplesGenerator = samplesGenerator
		self.samplesGeneratorArgs = samplesGeneratorArgs
		self.samplesGeneratorKwargs = samplesGeneratorKwargs
		self.filesRatio = filesRatio
		self.doLower = doLower
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
		self.minVocLF = minVocLF
		self.batchSize = batchSize
		self.samplesCounts = samplesCounts
		self.docLength = docLength
		self.bookedWords = None
		self.alwaysTruncateRawData = alwaysTruncateRawData
		# We build word embeddings:
		if self.wordEmbeddings is None:
			self.wordEmbeddings = Embeddings("test").getVectors()
		self.wordEmbeddings = copy.copy(self.wordEmbeddings)
		# We set doLower according to word embeddings:
		foundUpper = False
		for w in self.wordEmbeddings.keys():
			if w != w.lower():
				foundUpper = True
				break
		if self.doLower is None:
			self.doLower = not foundUpper
		else:
			assert self.doLower == (not foundUpper)
		# We init a random object for split and other things:
		self.rnd = random.Random(seed)
		# We check if files are already splitted:
		if isinstance(self.files[0], list):
			self.split = []
			newFiles = []
			total = len(flattenLists(self.files))
			for i in range(len(self.files)):
				current = self.files[i]
				self.split.append(truncateFloat(len(current) / total, 2))
				if self.filesRatio is not None:
					current = current[:int(len(current) * self.filesRatio)]
				newFiles.append(current)
			self.parts = newFiles
			self.files = flattenLists(self.parts)
		# Else we split files:
		else:
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


	def getPrebuilt(self):
		prebuilt = dict()
		prebuilt["samplesCounts"] = self.samplesCounts
		prebuilt["vocIndex"] = self.vocIndex
		prebuilt["labelEncoder"] = self.labelEncoder
		prebuilt["wordEmbeddings"] = self.wordEmbeddings
		return prebuilt

	def serializePrebuilt(self, path):
		if path[-7:] != ".pickle":
			raise Exception("Please provide a .pickle file path")
		prebuilt = self.getPrebuilt()
		serialize(prebuilt, path)

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
					currentCache = [row for row in self.getRawPart(partIndex)]
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
			vocLF = dict()
			self.samplesCounts = [0] * len(self.parts)
			# count = 0
			for partIndex in range(len(self.parts)):
				data = self.getPart(partIndex, pad=False, encodeLabel=False, encode=False, doLower=self.doLower)
				# for row in data:
				# 		if "katy" in row[0] and "2414493" in row[1]:
				# 			print(row)
				# 			count += 1
				# print(len(list(data)))
				for tokens, label in data:
					# if "katy" in tokens and "@33" not in label:
					# 	print(tokens)
					# 	print(label)
					# 	print()
					if isinstance(tokens[0], list):
						tokens = flattenLists(tokens)
					# if len(tokens) > 1200:
					# 	print(len(tokens))
					tokens = set(tokens)
					for token in tokens:
						tokensCount += 1
						if token not in self.wordEmbeddings:
							embUnknownTokens += 1
						if token not in vocDF:
							vocDF[token] = 0
						vocDF[token] += 1
						if token not in vocLF:
							vocLF[token] = set()
						vocLF[token].add(label)
					labels.add(label)
					self.samplesCounts[partIndex] += 1

			# print("cccccccc" + str(count))
			# TEST
			# print(vocLF["katy"])
			# print(vocDF["katy"])
			# Print some infos:
			# log("Samples counts are " + str(self.samplesCounts) + " (total " + str(sum(self.samplesCounts)) + ")", self)
			log(str(truncateFloat(embUnknownTokens / tokensCount * 100, 2)) + "% of tokens are not in word embeddings", self)
			log("Vocabulary size without minVocDF and minVocLF filtering: " + str(len(vocDF)), self)
			# We count the number of voc element which is not in wordEmbeddings:
			c = 0
			for w in vocDF.keys():
				if w not in self.wordEmbeddings:
					c += 1
			log(str(truncateFloat(c / len(vocDF) * 100, 2)) + "% of vocabulary are not in word embeddings", self)
			# We get the voc according to minVocDF:
			voc = set()
			for word, count in vocDF.items():
				if count >= self.minVocDF and len(vocLF[word]) >= self.minVocLF:
					voc.add(word)
				# if word == "katy":
				# 	print(count)
				# 	print(self.minVocDF)
				# 	print(len(vocLF[word]))
				# 	print(self.minVocLF)
			# We print informations:
			log("Vocabulary size with a minVocDF of " + str(self.minVocDF) + " and minVocLF of " + str(self.minVocLF) + ": " + str(len(voc)), self)
			# We count the number of voc element which is not in wordEmbeddings:
			c = 0
			for w in voc:
				if w not in self.wordEmbeddings:
					c += 1
			log("With a minVocDF of " + str(self.minVocDF) + " and minVocLF of " + str(self.minVocLF) + ", " + str(truncateFloat(c / len(voc) * 100, 2)) + "% of vocabulary are not in word embeddings", self)
			# We add the voc of wordEmbeddings DEPRECATED:
			# for word in self.wordEmbeddings.keys():
			# 	voc.add(word)
			# We remove OOV from wordEmbeddings:
			newWordEmbeddings = dict()
			for w, v in self.wordEmbeddings.items():
				if w in voc:
					newWordEmbeddings[w] = v
			self.wordEmbeddings = newWordEmbeddings
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
			# For word embeddings, we add the mask and the out of vocabulary element:
			self.wordEmbeddings[mlConf.MASK_TOKEN] = mlConf.MASK_EMBEDDING_FUNCTION(len(self.wordEmbeddings["the"]))
			self.wordEmbeddings[mlConf.OOV_TOKEN] = mlConf.OOV_EMBEDDING_FUNCTION(len(self.wordEmbeddings["the"]))
			# We add generate embedding for unknown words:
			for w in self.vocIndex.keys():
				if w not in self.wordEmbeddings:
					self.wordEmbeddings[w] = np.random.normal(scale=0.6, size=(self.getEmbeddingsDimension(),))
			# And finally we build the label encoder:
			if self.computeLabelEncoder and self.labelEncoder is None:
				try:
					labels = sorted(list(set(labels)))
				except: pass
				(labels, encodedLabels) = encodeMulticlassLabels(labels, encoding=self.labelEncoding, logger=self.logger, verbose=self.verbose)
				self.labelEncoder = dict()
				assert len(encodedLabels) == len(labels)
				for i in range(len(encodedLabels)):
					self.labelEncoder[labels[i]] = encodedLabels[i]
			assert len(self.wordEmbeddings) == len(self.vocIndex)


	def __getitem__(self, index):
		return self.getPart(index)
	def __len__(self):
		return len(self.parts)

	def getLabelEncoder(self):
		return self.labelEncoder
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
	def getSamplesCounts(self):
		return self.samplesCounts
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


	def getRawParts(self, *args, **kwargs):
		for i in range(len(self.parts)):
			for current in self.getRawPart(i, *args, **kwargs):
				yield current

	def getRawPart(self, *args, **kwargs):
		if "doLower" not in kwargs:
			kwargs["doLower"] = False
		if "pad" not in kwargs:
			kwargs["pad"] = False
		if "truncate" not in kwargs:
			kwargs["truncate"] = self.alwaysTruncateRawData
		if "encode" in kwargs and kwargs["encode"]:
			raise Exception("You cannot call getRawPart with encode as True, please use getPart instead.")
		else:
			kwargs["encode"] = False
		if "encodeLabel" not in kwargs:
			kwargs["encodeLabel"] = False
		if "vocIndex" not in kwargs:
			kwargs["vocIndex"] = None
		kwargs["doLowerWarning"] = False
		return self.getPart(*args, **kwargs)

	def getParts(self, *args, **kwargs):
		for i in range(len(self.parts)):
			for current in self.getPart(i, *args, **kwargs):
				yield current

	def getPart(self, index=None, doLowerWarning=True, *args, **encodeSampleKwargs):
		if index is None:
			assert len(self.parts) == 1
			index = 0
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
		if "doLower" not in encodeSampleKwargs:
			encodeSampleKwargs["doLower"] = self.doLower
		if "vocIndex" not in encodeSampleKwargs:
			encodeSampleKwargs["vocIndex"] = self.vocIndex
		encodeSampleKwargs["wordEmbeddings"] = self.wordEmbeddings
		encodeSampleKwargs["labelEncoder"] = self.labelEncoder
		if doLowerWarning and "doLower" in encodeSampleKwargs and encodeSampleKwargs["doLower"] == False and self.doLower == True:
			logWarning("Some words will be out of vocabulary because allowed voc element are only lowered tokens and you forced doLower as False", self)
		pad = True # default
		truncate = True # default
		if "pad" in encodeSampleKwargs:
			pad = encodeSampleKwargs["pad"]
		if "truncate" in encodeSampleKwargs:
			truncate = encodeSampleKwargs["truncate"]
		if pad and not truncate:
			logWarning("You set truncate as False even pad was set as True. To pad tokens and to do not truncate too long tokens is not implemented.", self)
			truncate = True
			encodeSampleKwargs["truncate"] = True
		if (pad or truncate) and encodeSampleKwargs["docLength"] is None:
			raise Exception("Please provide a doc length to truncate tokens.")
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

	def getInfiniteBatcher(self, index=None, *args, skip=0, shuffle=0, **kwargs):
		if index is None:
			assert len(self.parts) == 1
			index = 0
		ci = self.getPart(index, *args, **kwargs)
		return InfiniteBatcher(ci, batchSize=self.batchSize, 
							shuffle=shuffle,
							skip=skip,
							logger=self.logger, verbose=self.verbose)

	def getBatchesCount(self, index=None):
		if index is None:
			assert len(self.parts) == 1
			index = 0
		c = self.getSamplesCount(index)
		return math.ceil(c / self.batchSize)


if __name__ == '__main__':
	files = sortedGlob(tmpDir("mbti-datasets") + "/*/*.bz2")[:5]
	te = TextEncoder(files, split=[0.8, 0.1, 0.1])











	# def depr_maskBookedWords(self):
	# 	assert False not in self.persist
	# 	self.__buildBookedWords()
	# 	for cache in self.cache:
	# 		for row in cache:
	# 			for i in range(len(row[0])):
	# 				if row[0][i] in self.bookedWords:
	# 					row[0][i] = mlConf.MASK_TOKEN

	# def depr_buildBookedWords(self):
	# 	"""
	# 		See bookedWordsReport function
	# 	"""
	# 	if self.bookedWords is None:
	# 		# logWarning("buildBookedWords will load all in memory", self)
	# 		docs = [e[0] for e in self.getRawParts()]
	# 		ads = [e[1] for e in self.getRawParts()]
	# 		indexLabels = ads
	# 		# Here we make the inverted index word -> doc id:
	# 		invertedIndex = dict()
	# 		for i, doc in enumerate(docs):
	# 			for word in doc:
	# 				if word not in invertedIndex:
	# 					invertedIndex[word] = set()
	# 				invertedIndex[word].add(i)
	# 		# bp(invertedIndex)
	# 		# Here we separate words that appear in only one doc and those appearing in multiple docs:
	# 		multiDocWords = set()
	# 		oneDocWords = set()
	# 		for word, docsId in invertedIndex.items():
	# 			if len(docsId) > 1:
	# 				multiDocWords.add(word)
	# 			else:
	# 				oneDocWords.add(word)
	# 		# bp(multiDocWords, 4)
	# 		# print(len(multiDocWords))
	# 		# bp(oneDocWords, 4)
	# 		# print(len(oneDocWords))
	# 		# Here we collect all words per author so that the word has min df > 1:
	# 		authorsVocab = dict()
	# 		for i in range(len(docs)):
	# 			author = indexLabels[i]
	# 			if author not in authorsVocab:
	# 				authorsVocab[author] = set()
	# 			doc = set()
	# 			for word in docs[i]:
	# 				if word not in oneDocWords:
	# 					doc.add(word)
	# 			authorsVocab[author] = authorsVocab[author].union(doc)
	# 		# bp(authorsVocab, 3)
	# 		# Here we retain only words that only one author has:
	# 		bookedVocab = dict()
	# 		for author, voc in authorsVocab.items():
	# 			newVoc = set()
	# 			for word in voc:
	# 				foundInAnOtherAuthor = False
	# 				for author2, voc2 in authorsVocab.items():
	# 					if author != author2:
	# 						if word in voc2:
	# 							foundInAnOtherAuthor = True
	# 							break
	# 				if not foundInAnOtherAuthor:
	# 					newVoc.add(word)
	# 			bookedVocab[author] = newVoc
	# 		# And finally we collect the vocab:
	# 		bookedWords = set()
	# 		for ad, voc in bookedVocab.items():
	# 			bookedWords = bookedWords.union(voc)
	# 		self.bookedWords = bookedWords
	# 	return self.bookedWords