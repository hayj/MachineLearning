import random
try:
	import tensorflow as tf
except: pass
import os
import numpy as np

def seed(seed_value=0):
	# From https://stackoverflow.com/a/52897216/3406616
	# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
	os.environ['PYTHONHASHSEED'] = str(seed_value)
	# 2. Set `python` built-in pseudo-random generator at a fixed value
	random.seed(seed_value)
	# 3. Set `numpy` pseudo-random generator at a fixed value
	np.random.seed(seed_value)
	# 4. Set `tensorflow` pseudo-random generator at a fixed value
	try:
		tf.set_random_seed(seed_value)
	except: pass

