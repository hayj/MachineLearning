# coding: utf-8

import collections
from threading import Thread, Lock
import time
from systemtools.logger import log, logInfo, logWarning, logError, Logger
from systemtools.system import *
from systemtools.basics import *
import random


def getRandomParams(paramsDomain):
    if isinstance(paramsDomain, list):
        return random.choice(paramsDomain)
    else:
        randomParams = dict()
        for key, value in paramsDomain.items():
            randomParams[key] = getRandomParams(value)
        return randomParams

class Bandit:
    def __init__\
                (
                    self,
                    verbose=True,
                    logger=None,
                    generateParams=None, # Funct
                    paramsDomain=None,
                    exploreRate=0.1, # 0.1 is a good choice
                    paramHistoryCount=30, # 30
                    startExploitRate=0.2, # 0.5 is a good choice
                    paramsChecker=None, # 0.5 is a good choice
                ):
        self.paramsChecker = paramsChecker
        self.paramHistoryCount = paramHistoryCount
        self.startExploitRate = startExploitRate
        self.verbose = verbose
        self.exploreRate = exploreRate
        self.logger = logger
        self.generateParams = generateParams
        self.previousParams = getFixedLengthQueue(paramHistoryCount)
        self.previousParam = None
        self.paramsDomain = paramsDomain
    
    def getParams(self):
        # If we have a function from the creator:
        if self.generateParams is not None:
            newParams = self.generateParams()
            return newParams
        # Else we choose as random:
        elif self.paramsDomain is not None:
            newParams = getRandomParams(self.paramsDomain)
            if self.paramsChecker is not None:
                paramsOk = self.paramsChecker(newParams)
                if paramsOk:
                    return newParams
                else:
#                     logInfo("These params failed:", self)
#                     logInfo(listToStr(newParams), self)
                    return self.getParams()
            else:
                return newParams
                

    def nextParams(self, score=None):
        # If we start, we just generate the first param:
        if score is None or self.previousParam is None:
            newParams = self.getParams()
        else:
            # We add an entry to the history:
            newHistoryEntry = None
            if self.previousParam is not None:
                newHistoryEntry = (self.previousParam, score)
            self.previousParams.append(newHistoryEntry)
            
            logInfo("Previous params score: " + str(score), self)
            
            # Then we choose explore or exploit:
            isExplore = getRandomFloat() <= self.exploreRate
            newParams = None
            firstExploresDoneCount = len(self.previousParams) - countNone(self.previousParams)
            minimumExploreAtFirstTime = self.startExploitRate * len(self.previousParams)
            if isExplore or minimumExploreAtFirstTime > firstExploresDoneCount:
                logInfo("Now we explore...", self)
                newParams = self.getParams()
            else:
                logInfo("Now we exploit...", self)
                newParams = maxTupleList(self.previousParams, 1, getAll=False)
        
        # Then we retain newParams as the previous:
        self.previousParam = newParams
        
        # And we return the current params choice:
        logInfo("New params:\n" + listToStr(newParams), self)
        return newParams
        
        

# class Browser:
#     def __init__(self, callback=None):
#         pass
#     def html(self, url):
#         currentHtml = "<html>" + url + "</html>"
#         time.sleep(0.1)
#         if self.callback is not None:
#             self.callback(html=currentHtml, browser=self)
#         return currentHtml
# 
# 
# 
# class MultipleBrowser:
#     def __init__(self):
#         self.bandit = Bandit()
#         self.browsers = []
#         for i in range(10):
#             self.browsers.append(Browser(self.browserCallback))
#     def start(self, urls):
#         for b in self.browsers:
#             t = Thread(target=b.html, args=(urls.pop(),))
#                 
#     def browserCallback(self, browser=None, html=None):
#         pass

class MultipleBrowserTest:
    def __init__(self):
        paramsDomain = \
        {
            "a": [10, 20, 50, 100],
            "b": [0.1, 0.3, 0.5, 0.8, 1.0],
        }
        self.bandit = Bandit(paramsDomain=paramsDomain)
        params = self.bandit.nextParams()
        for i in range(300):
            score = self.makeScore(params)
            params = self.bandit.nextParams(score)
#             print(listToStr(params))
#             print(self.bandit.previousParams)
#             print()

    def makeScore(self, params):
        score = 0.0
        if params["a"] == 50:
            score += 1
        elif params["a"] == 100:
            score += 0.8
        else:
            score += 0.2
        if params["b"] > 0.5:
            score += 0.5
        else:
            score += 0.1   
        return score



if __name__ == '__main__':
#     mb = MultipleBrowserTest()
    printLTS(getRandomParams({
                        "proxyInstanciationRate":
                        {
                            "alpha": [0.99],
                            "beta": [1]
                        },
                        "browserCount": [3],
                        "parallelRequests": [20, 50, 100, 150],
                    }))
    
    
#     mb.start(range(1200, 1800))
    
    


# TODO faire deux variables simples, une entre 0 et 1 optimal à 0.8 et une autre entre 20 et 100 optimal à 50







