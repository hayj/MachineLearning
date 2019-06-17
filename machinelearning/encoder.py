from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np


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