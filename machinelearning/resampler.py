from systemtools.basics import *
from systemtools.system import *
from systemtools.printer import *
from systemtools.duration import *




def generateMultiLabelDataset(nbLabels=4, nbSamples=10000, balance=[0.6, 0.4, 0.8, 0.2]):
	assert len(balance) == nbLabels
	dataset = []
	if balance is None:
		balance = [0.65] * nbLabels
	for i in range(nbSamples):
		current = [None] * nbLabels
		for u in range(len(current)):
			current[u] = 1 if getRandomFloat() > balance[u] else 0
		dataset.append(current)
	return dataset


def randomCandidat(samples, k):
	candidat = [False] * (len(samples) - k) + [True] * k
	random.shuffle(candidat)
	assert len(candidat) == len(samples)
	return candidat


def fitness(samples, candidat):
	newSamples = []
	for i in range(len(samples)):
		if candidat[i]:
			newSamples.append(samples[i])
	ratios = labelsRatio(newSamples)
	score = ratiosFitness(ratios)
	return score

def ratiosFitness(ratios):
	"""
		Return a balance score between 0 and 1
		1 means the dataset is well balanced, for example [0.5, 0.5, 0.5, 0.5]
	"""
	score = 0
	for u in range(len(ratios)):
		current = ratios[u]
		current = abs(0.5 - current)
		current = current / 0.5
		score += current
	score = score / len(ratios)
	score = 1.0 - score
	return score


def labelsRatio(samples, activation=None):
	if activation is not None:
		assert len(activation) == len(samples)
		newSamples = []
		for i in range(len(activation)):
			if activation[i]:
				newSamples.append(samples[i])
		samples = newSamples
	ratios = [0] * len(samples[0])
	for sample in samples:
		for u in range(len(sample)):
			if sample[u] == 0:
				ratios[u] += 1
	for u in range(len(ratios)):
		ratios[u] = ratios[u] / len(samples)
	return ratios

def mutate(candidat, ratio):
	mutationsCount = 0
	candidat = copy.copy(candidat)
	falseIndexes = []
	trueIndexes = []
	for i in range(len(candidat)):
		if candidat[i]:
			trueIndexes.append(i)
		else:
			falseIndexes.append(i)
	nbMutation = int(len(candidat) * ratio)
	assert nbMutation > 0
	assert nbMutation < len(candidat)
	for mutIt in range(nbMutation):
		try:
			falseIndex = random.choice(falseIndexes)
			trueIndex = random.choice(trueIndexes)
		except:
			break
		candidat[falseIndex] = True
		candidat[trueIndex] = False
		mutationsCount += 1
		falseIndexes.remove(falseIndex)
		trueIndexes.remove(trueIndex)
	assert mutationsCount > 0
	return candidat


#We end with a balance of: [0.5000147253718157, 0.5199823295538213, 0.5000147253718157, 0.5000147253718157]
#We got only 33955 samples on 76000


#CWe end with a balance of: [0.5001137656427759, 0.5196814562002275, 0.5001137656427759, 0.5001137656427759]
#We got only 4395 samples on 10000
def genetic(samples, nbBags=1000, bagSizeRatio=0.05, kRatio=0.3):
	k = int(kRatio * len(samples))
	bagSize = int(len(samples) * bagSizeRatio)
	indexes = list(range(len(samples)))
	animal = [False] * len(samples)
	score = -1
	currentK = 0
	while currentK < k:
		bags = []
		for i in range(nbBags):
			bags.append(list(random.sample(indexes, bagSize)))
		bestScore = -1
		bestBag = None
		bestCandidat = None
		foundOne = False
		for bag in bags:
			candidat = copy.copy(animal)
			for index in bag:
				candidat[index] = True
			score = fitness(samples, candidat)
			if score > bestScore:
				foundOne = True
				bestBag = bag
				bestScore = score
				bestCandidat = candidat
		if not foundOne:
			break
		animal = bestCandidat
		score = bestScore
		bestBag = set(bestBag)
		newIndexes = []
		for i in range(len(indexes)):
			if indexes[i] not in bestBag:
				newIndexes.append(indexes[i])
		indexes = newIndexes
		currentK = animal.count(True)
		print(currentK)
		print(bestScore)
	return animal, score
	bp(candidats)


def genetic2(samples, kRatio=0.3, nbAnimals=100, nbIterations=500, logger=None, verbose=True, keepRatio=0.2, nbAnimalsSplit=50, mutationRatio=0.2, animals=None):
	if animals is None:
		animals = []
		k = int(kRatio * len(samples))
		for i in range(nbAnimals):
			animals.append(randomCandidat(samples, k))
	pbar = ProgressBar(nbIterations, logger=logger, verbose=verbose, printRatio=0.0001)
	for it in range(nbIterations):
		# Selection:
		scores = []
		for i in range(len(animals)):
			scores.append((animals[i], fitness(samples, animals[i])))
		animals = sortBy(scores, desc=True)
		# bp(animals)
		animals = animals[:int(len(animals) * keepRatio)]
		# Print:
		pbar.tic("Score of the best animal: " + str(truncateFloat(animals[0][1], 2)))
		# We keep only animals (not scores):
		animals = [animal for animal, score in animals]
		# We cannot do cross-over:
		# # Crossover:
		# splitedAnimals = []
		# for i in range(len(animals)):
		# 	splitedAnimals.append(split(animals[i], nbAnimalsSplit))
		# animals = []
		# for i in range(nbAnimals):
		# 	childAnimal = []
		# 	for u in range(nbAnimalsSplit):
		# 		index = getRandomInt(0, len(splitedAnimals) - 1)
		# 		childAnimal += splitedAnimals[index][u]
		# 	# print(len(childAnimal))
		# 	# bp(childAnimal)
		# 	# print(len(samples))
		# 	# bp(samples)
		# 	assert len(childAnimal) == len(samples)
		# 	animals.append(childAnimal)
		# assert len(animals) == nbAnimals
		# Mutation:
		newAnimals = []
		i = 0
		while len(newAnimals) < nbAnimals:
			newAnimals.append(mutate(animals[i], mutationRatio))
			i += 1
			if i == len(animals):
				i = 0
		animals = newAnimals
	# We get the best one:
	bestAnimal = (None, -1)
	for i in range(len(animals)):
		score = fitness(samples, animals[i])
		if score > bestAnimal[1]:
			bestAnimal = (animals[i], score)
	# We return all:
	return (bestAnimal[0], bestAnimal[1], animals)


def fitnessTest():
	print(fitness([0.5, 0.54, 0.53, 0.42]))
	print(fitness([0.8, 0.9, 0.2, 0.3]))
	print(fitness([0.98, 0.02, 0.22, 0.03]))
	print(fitness([0.5, 0.5, 0.5, 0.5]))
	print(fitness([0.0, 1.0, 1.0, 0.0]))

def mutateTest():
	print(mutate([True, False, True, False, True, False] * 10, 0.5))

def mbtiTest():
	# pew in st-venv python ~/Workspace/Python/Utils/MachineLearning/machinelearning/resampler.py
	print("Starting...")
	random.seed(0)


	# mbtiBalance = [0.66, 0.78, 0.6, 0.45]
	# dataset = generateMultiLabelDataset(nbLabels=4, nbSamples=86000, balance=mbtiBalance)
	mbtiBalance = [0.66, 0.78, 0.6, 0.45]
	dataset = generateMultiLabelDataset(nbLabels=4, nbSamples=10000, balance=mbtiBalance)

	# mbtiBalance = [0.66, 0.78, 0.6, 0.45]
	# dataset = generateMultiLabelDataset(nbLabels=4, nbSamples=10000, balance=mbtiBalance)


	# mbtiBalance = [0.66, 0.78, 0.6, 0.45]
	# dataset = generateMultiLabelDataset(nbLabels=4, nbSamples=1000, balance=mbtiBalance)

	(bestAnimal, score) = glouton(dataset)
	bp(bestAnimal)
	bp(score)







def getSubKeys(sample, doFlatten=True):
	nbLabels = len(sample)
	mainKey = sample
	if not isinstance(mainKey, str):
		mainKey = ""
		for e in sample:
			mainKey += str(e)
	keys = []
	keys.append([mainKey])
	for u in range(1, nbLabels + 1):
		current = list(fillByX(mainKey, u))
		random.shuffle(current)
		keys.append(current)
	if doFlatten:
		return magicFlatten(keys)
	else:
		return keys

def fillByX(key, nbX):
	voc = list(range(len(key)))
	indexess = combine(voc, nbX, strConcat=False)
	indexess = [e for e in indexess if len(set(e)) == len(e)]
	keys = []
	for indexes in indexess:
		current = "" + key
		for index in indexes:
			current = current[:index] + 'x' + current[index + 1:]
		keys.append(current)
	return set(keys)

def addSample(index, samples, mapper):
	sample = samples[index]
	subKeys = getSubKeys(sample, doFlatten=True)
	for subKey in subKeys:
		mapper[subKey].append(index)

def removeSample(index, samples, mapper):
	sample = samples[index]
	subKeys = getSubKeys(sample, doFlatten=True)
	for subKey in subKeys:
		mapper[subKey].remove(index)

def getRandomMostSimilar(element, balance, mapper):
	subKeys = getSubKeys(element, doFlatten=True)
	subKeysScores = []
	for subKey in subKeys:
		score = 0
		for i in range(len(subKey)):
			if subKey[i] == 'x':
				pass
			else:
				score += abs(0.5 - balance[i])
		subKeysScores.append((subKey, score))
	subKeysScores = sortBy(subKeysScores, index=1, desc=True)
	# print(subKeysScores)
	orderedSubKeys = []
	for subKey, score in subKeysScores:
		if subKey in mapper and len(mapper[subKey]) > 0:
			orderedSubKeys.append(subKey)
	# print(orderedSubKeys)
	mostSimilar = None
	for subKey in orderedSubKeys:
		candidats = mapper[subKey]
		if len(candidats) > 0:
			mostSimilar = random.choice(candidats)
			break
	return mostSimilar

def glouton(samples, kRatio=1.0, minScore=0.99):
	ks = []
	scores = []
	nbLabels = len(samples[0])
	mapperKeys = combine(['0', '1', 'x'], nbLabels, strConcat=True)
	mapperTrue = dict()
	for key in mapperKeys:
		mapperTrue[key] = list()
	mapperFalse = copy.deepcopy(mapperTrue)
	pbar = ProgressBar(len(samples), printRatio=0.1, message="Generating the base structure")
	for i, sample in enumerate(samples):
		subKeys = getSubKeys(sample, doFlatten=True)
		for key in subKeys:
			mapperFalse[key].append(i)
		pbar.tic()
	k = int(kRatio * len(samples))
	# First we generate animal so that it takes the min of each:
	counts = dict()
	countsKeys = combine(['0', '1'], nbLabels, strConcat=True)
	for key in countsKeys:
		counts[key] = 0
	for sample in samples:
		sample = "".join([str(e) for e in sample])
		counts[sample] += 1
	theMin = None
	for key, value in counts.items():
		if theMin is None or value < theMin:
			theMin = value
	animal = [False] * len(samples)
	if theMin == 0:
		animal[0] = True
		removeSample(0, samples, mapperFalse)
		addSample(0, samples, mapperTrue)
	else:
		toAdd = []
		for key in countsKeys:
			remainingTodAdd = theMin
			for i, sample in enumerate(samples):
				if "".join([str(e) for e in sample]) == key:
					toAdd.append(i)
					remainingTodAdd -= 1
				if remainingTodAdd == 0:
					break
		# for index in toAdd:
		# 	print(samples[index])
		for index in toAdd:
			animal[index] = True
			removeSample(index, samples, mapperFalse)
			addSample(index, samples, mapperTrue)
	print("Real balance: " + str(labelsRatio(samples)))
	print("We start with a balance of: " + str(labelsRatio(samples, activation=animal)))
	print("We start with " + str(animal.count(True)) + " samples on " + str(len(samples)))
	# restants = []
	# for i in range(len(animal)):
	# 	if animal[i] == False:
	# 		restants.append(samples[i])
	# bp(restants)
	# print(labelsRatio(restants))
	# exit()
	bpar = ProgressBar(k - animal.count(True), printRatio=0.1, message="Searching for the best combinason")
	while True:
		balance = labelsRatio(samples, activation=animal)
		# print("balance: " + str(balance))
		ideal = [math.ceil(e - 0.5) for e in balance]
		worst = [1 - e for e in ideal]
		# ideal, worst = worst, ideal
		idealMostSimilar = getRandomMostSimilar(ideal, balance, mapperFalse)

		balanceTmp = labelsRatio(samples, activation=animal)
		for i in range(len(balanceTmp)):
			balanceTmp[i] = truncateFloat(balanceTmp[i], 2)
		# print("balance: " + str(balanceTmp))
		# print("ideal: " + str(ideal))
		# print("adding: " + str(samples[idealMostSimilar]))
		# print()
		if idealMostSimilar is None:
			break
		else:
			animal[idealMostSimilar] = True
			removeSample(idealMostSimilar, samples, mapperFalse)
			addSample(idealMostSimilar, samples, mapperTrue)
		# if getRandomFloat() > 0.6:
		# 	worstMostSimilar = getRandomMostSimilar(worst, balance, mapperTrue)
		# 	animal[worstMostSimilar] = False
		# 	removeSample(worstMostSimilar, samples, mapperTrue)
		# 	addSample(worstMostSimilar, samples, mapperFalse)


		
		currentK = animal.count(True)
		if currentK == k:
			break
		score = fitness(samples, animal)
		ks.append(currentK)
		scores.append("score: " + str(score))
		# print("currentK:" + str(currentK))
		# print("added element: " + str(samples[idealMostSimilar]))
		# print("score: " + str(score))
		# print()
		bpar.tic(str(score))
		if score < minScore:
			break
	print("We end with a balance of: " + str(labelsRatio(samples, activation=animal)))
	print("We got only " + str(animal.count(True)) + " samples on " + str(len(samples)))
	# import matplotlib.pyplot as plt
	# plt.plot(ks, scores)
	# # plt.axis([0, 6, 0, 20])
	# plt.show()
	# exit()

	# bp(mapper, 4)
	return animal
	# result = []
	# for i in range(len(animal)):
	# 	if animal[i]:
	# 		result.append(samples[i])
	# return result

if __name__ == '__main__':
	# print(combine([True, True, True, True, False, False]))
	mbtiTest()
	# fitnessTest()
	# mutateTest()