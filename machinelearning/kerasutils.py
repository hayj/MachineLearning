from keras.utils import multi_gpu_model
from keras import backend
from systemtools.logger import *

def toMultiGPU(model, logger=None, verbose=True):
	try:
		gpuCount = len(backend.tensorflow_backend._get_available_gpus())
		model = multi_gpu_model(model, gpus=gpuCount)
	except Exception as e:
		logException(e, logger, verbose=verbose)
	return model