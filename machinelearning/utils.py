
import numpy as np

def isListOrArray(*args, **kwargs):
	return isArrayOrList(*args, **kwargs)
def isArrayOrList(a):
	return isinstance(a, list) or isinstance(a, np.ndarray)