# coding: utf-8

from systemtools.basics import *
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

NormalizedLawBeta = Enum("NormalizedLawBeta", "LOG EXP")

def plotAll(allX, allY):
    plt.plot(allX, allY)
    plt.show()

def plotNormalizedLaw(*args, **kwargs):
    allX = np.arange(0.0, 1.0, 0.0001)
    allY = normalizedLaw(allX, *args, **kwargs)
    plotAll(allX, allY)
    
def normalizedLaw(x, *args, **kwargs):
    if isIterable(x):
        return [normalizedLawX(currentX, *args, **kwargs) for currentX in x]
    else:
        return normalizedLawX(x, *args, **kwargs)

def normalizedLawX(x, alpha=0.5, beta=NormalizedLawBeta.LOG, inverse=True):
    # Vars:
    a = alpha
    eps = 1e-6
    
    # Particular cases:
    if a < eps:
        if x < eps:
            return 1.0
        else:
            return eps
    elif a > (1.0 - eps):
        return 1.0
    
    # Functions:
    if beta == NormalizedLawBeta.LOG:
        if a < 0.5:
            y = -(x ** (2 * a)) + 1
        else:
            y = -(x ** (1 / (2 * (abs(a - 1) + 0.5) - 1))) + 1
    else:
        if a < 0.5:
            y = (1 - x) ** (1 / (2 * a))
        else:
            y = (1 - x) ** (2 * (abs(a - 1) + 0.5) - 1)
    if inverse:
        return y
    else:
        return 1 - y


def test1():
    plotNormalizedLaw(alpha=0.5, distanceType=DistanceType.LOG, inverse=True)
    plotNormalizedLaw(alpha=0.5, distanceType=DistanceType.EXP, inverse=False)
    plotNormalizedLaw(alpha=0.2, distanceType=DistanceType.LOG, inverse=False)
    plotNormalizedLaw(alpha=0.8, distanceType=DistanceType.LOG, inverse=False)
    plotNormalizedLaw(alpha=0.1, distanceType=DistanceType.LOG, inverse=False)
    plotNormalizedLaw(alpha=0.9, distanceType=DistanceType.LOG, inverse=False)
    plotNormalizedLaw(alpha=1, distanceType=DistanceType.LOG, inverse=False)
    plotNormalizedLaw(alpha=0, distanceType=DistanceType.LOG, inverse=False)

if __name__ == '__main__':
    test1()













