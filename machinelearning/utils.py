
import numpy as np
from systemtools.basics import *
from machinelearning import config as mlConf


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
