
import numpy as np
from systemtools.basics import *
from systemtools.logger import *
from systemtools.printer import *
from machinelearning import config as mlConf
import scipy


def assertTrainTestShape(xTrain, yTrain, xVal, yVal):
    assert xTrain.shape[0] == yTrain.shape[0]
    assert xVal.shape[0] == yVal.shape[0]
    assert xTrain.shape[1] == xVal.shape[1]


def splitTrainTest(x, ratio=0.8, splitIndex=None):
    if splitIndex is None:
        if isinstance(x, list):
            splitIndex = int(len(x) * 0.8)
        else:
            splitIndex = int(x.shape[0] * 0.8)
    if isinstance(x, scipy.sparse.coo.coo_matrix):
        x = scipy.sparse.csr.csr_matrix(x)
    if isinstance(x, scipy.sparse.csr.csr_matrix):
        x = list(x)
        xTrain = x[:splitIndex]
        xTrain = scipy.sparse.vstack(xTrain)
        xVal = x[splitIndex:]
        xVal = scipy.sparse.vstack(xVal)
    else:
        xTrain = x[:splitIndex]
        xVal = x[splitIndex:]
    return xTrain, xVal

def concatSparseMatrices(*args, logger=None, verbose=False):
    assert len(args) > 0
    result = None
    log("+" * 3 + " " + str(len(args)) + " matrices to concat: " + "+" * 3, logger=logger, verbose=verbose)
    for current in args:
        if not isinstance(current, scipy.sparse.csr.csr_matrix):
            current = scipy.sparse.csr_matrix(current, dtype=np.float64)
            log("\t" + str(current.shape) + " (not a csr_matrix)", logger=logger, verbose=verbose)
        else:
            log("\t" + str(current.shape), logger=logger, verbose=verbose)
        if result is None:
            result = current
        else:
            result = scipy.sparse.hstack((result, current))
    log("\t" + "Result shape: " + str(result.shape), logger=logger, verbose=verbose)
    return result


def isListOrArray(*args, **kwargs):
	return isArrayOrList(*args, **kwargs)
def isArrayOrList(a):
	return isinstance(a, list) or isinstance(a, np.ndarray)


def stackArrays(arrays):
    assert len(arrays) > 0
    assert isinstance(arrays, list)
    mtx = None
    for currentArray in arrays:
        if isinstance(currentArray, list):
            currentArray = np.array(currentArray)
        if mtx is None:
            mtx = currentArray
        else:
            mtx = np.vstack((mtx, currentArray))
    return mtx

def softmax2onehot(arr):
    # return np.array([0.0 for e in range(index)] + [1.0] + [0.0 for e in range(index + 1, len(arr))])
    index = np.argmax(arr)
    result = np.zeros(len(arr))
    result[index] = 1.0
    return result

def padSequences(ls, *args, **kwargs):
    return [padSequence(l, *args, **kwargs) for l in ls]

def padSequence(ls, maxlen, padding='pre', truncating='post', value=mlConf.MASK_TOKEN, removeEmptySentences=True):
    """
        This function pad sequence of tokens (string) as the keras pad_sequences function.
        It can also pad senquences of sentences.
    """
    assert removeEmptySentences
    kwargs = locals()
    assert maxlen is not None and isinstance(maxlen, int) and maxlen > 0
    if ls is None:
        return ls
    # if len(ls) > 0 and isinstance(ls[0], list):
    #     newL = []
    #     del kwargs["ls"]
    #     for current in ls:
    #         newL.append(padSequence(current, **kwargs))
    #     return newL
    # else:
    # In case we got a list of sentences:
    if len(ls) > 0 and isinstance(ls[0], list):
        if removeEmptySentences:
            ls = [s for s in ls if len(s) > 0]
        nbTokens = len(flattenLists(ls))
        if nbTokens > maxlen:
            if truncating == 'post':
                c = 0
                result = []
                i = 0
                while c < maxlen:
                    result.append(ls[i])
                    c += len(ls[i])
                    i += 1
                amountToDelete = c - maxlen
                if amountToDelete > 0:
                    result[-1] = result[-1][:-amountToDelete]
                ls = result
            else:
                c = 0
                result = []
                i = len(ls) - 1
                while c < maxlen:
                    result.insert(0, ls[i])
                    c += len(ls[i])
                    i -= 1
                amountToDelete = c - maxlen
                if amountToDelete > 0:
                    result[0] = result[0][amountToDelete:]
                ls = result
        elif nbTokens < maxlen:
            amountToAdd = maxlen - nbTokens
            if padding == 'pre':
                ls.insert(0, [value] * amountToAdd)
            else:
                ls.append([value] * amountToAdd)
        return ls
    # Else in case we got a sentence (a list of words):
    else:
        if len(ls) > maxlen:
            if truncating == 'pre':
                ls = ls[-maxlen:]
            else:
                ls = ls[:maxlen]
        elif len(ls) < maxlen:
            if padding == 'pre':
                ls = [value] * (maxlen - len(ls)) + ls
            else:
                ls = ls + [value] * (maxlen - len(ls))
        return ls


def bookedWordsReport(docs, authors, logger=None, verbose=True):
    """
        This function allow you to evaluate how well a TFIDF baseline
        will be better than any other predictors like DNN etc.
        For exemple, if 40% of documents has a words that only one author use,
        the TFIDF baseline will allways be better than any other predictor
        for the authorship attribution task.
        
        This function search words that only one author use and that its
        document frequency is greater that 1 (because you cannot use a clue
        that appear in only one document...)
        
        You must give a list of docs (list of list of words) and a list of
        authors (same length)
    """
    indexLabels = authors
    # Here we make the inverted index word -> doc id:
    invertedIndex = dict()
    for i, doc in enumerate(docs):
        for word in doc:
            if word not in invertedIndex:
                invertedIndex[word] = set()
            invertedIndex[word].add(i)
    # bp(invertedIndex)
    # Here we separate words that appear in only one doc and those appearing in multiple docs:
    multiDocWords = set()
    oneDocWords = set()
    for word, docsId in invertedIndex.items():
        if len(docsId) > 1:
            multiDocWords.add(word)
        else:
            oneDocWords.add(word)
    # bp(multiDocWords, 4)
    # print(len(multiDocWords))
    # bp(oneDocWords, 4)
    # print(len(oneDocWords))
    # Here we collect all words per author so that the word has min df > 1:
    authorsVocab = dict()
    for i in range(len(docs)):
        author = indexLabels[i]
        if author not in authorsVocab:
            authorsVocab[author] = set()
        doc = set()
        for word in docs[i]:
            if word not in oneDocWords:
                doc.add(word)
        authorsVocab[author] = authorsVocab[author].union(doc)
    # bp(authorsVocab, 3)
    # Here we retain only words that only one author has:
    bookedVocab = dict()
    for author, voc in authorsVocab.items():
        newVoc = set()
        for word in voc:
            foundInAnOtherAuthor = False
            for author2, voc2 in authorsVocab.items():
                if author != author2:
                    if word in voc2:
                        foundInAnOtherAuthor = True
                        break
            if not foundInAnOtherAuthor:
                newVoc.add(word)
        bookedVocab[author] = newVoc
    # bp(bookedVocab, 3)
    # Here for each of these words, we count the DF:
    bookedWordsCount = dict()
    for author, voc in bookedVocab.items():
        bookedWordsCount[author] = dict()
        for word in voc:
            bookedWordsCount[author][word] = len(invertedIndex[word])
            for docId in invertedIndex[word]:
                assert indexLabels[docId] == author
    bp(bookedWordsCount, 3)
    # Let's start for statistics:
    significantDocsCount = 0
    significantWordsPerDoc = 0
    for i, doc in enumerate(docs):
        author = indexLabels[i]
        doc = set(doc)
        found = False
        for word in doc:
            if word in bookedWordsCount[author]:
                significantWordsPerDoc += 1
                found = True
        if found:
            significantDocsCount += 1
    log("\n", logger=logger, verbose=verbose)
    log(str(truncateFloat(significantDocsCount / len(docs) * 100, 2)) + \
        "% of docs have words that strongly indicate the author...",
        logger=logger, verbose=verbose)
    log("~" + str(truncateFloat(significantWordsPerDoc / len(docs), 2)) + \
        " words are significant for the author in docs",
        logger=logger, verbose=verbose)
    if significantDocsCount > 0:
        log("~" + str(truncateFloat(significantWordsPerDoc / significantDocsCount, 2)) + \
        " words are significant for the author for docs that has at least one significant word",
        logger=logger, verbose=verbose)