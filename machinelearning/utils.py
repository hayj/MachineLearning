
import numpy as np

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