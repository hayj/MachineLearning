# coding: utf-8

from systemtools.basics import *
from systemtools.duration import *
# from sts.mongohandler import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

class TopOutTypeEnum():
    (
        STRING,
        STRING_TOP,
        DATA_FRAME,
        LIST_DICT,
        LIST_STRING
    ) = list(range(5))

class Graph(object):
    def __init__(self, data):        
        # Expand the cursor and construct the DataFrame
        if isinstance(data, pd.DataFrame):
            self.df =  data
        else:
            self.df =  pd.DataFrame(list(data))
                    
        self.resetSelection()
    
    def resetSelection(self):
        self.selection = self.df.copy(deep=True)
    
    def getFieldSet(self, by, sort=False):
        if sort:
            self.sortSelection()
        fieldSet = []
        for index, current in self.selection.iterrows():
            if current[by] not in fieldSet:
                fieldSet.append(current[by])
        return fieldSet
    
    def infos(self):
        result = ""
        result += str(self.df.dtypes)
        # result += str(self.df.tail(3))
        result += str(self.df.index)
        result += str(self.df.describe())
        return result
    
    @staticmethod
    def sort(selection, by="score", ascending=False):
        return selection.sort_values(by=by, ascending=ascending)
        
    def top(self, max=20, outType=TopOutTypeEnum.STRING_TOP, realIndex=False):
        sortedValues = Graph.sort(self.selection, by="score")
        sortedValuesHead = sortedValues.head(max)
        top = sortedValuesHead
        return self.convertTop(top, outType=outType, realIndex=realIndex)
    
    def convertTop(self, top, outType, realIndex=False):
        if outType == TopOutTypeEnum.DATA_FRAME:
            return top
        else:
            sortedDf = None
            result = None
            started = False
            u = 1
            for index, row in top.iterrows():
                if outType == TopOutTypeEnum.STRING_TOP:
                    currentRank = u
                    if realIndex:
                        if sortedDf is None:
                            sortedDf = Graph.sort(self.df, by="score")
                        i = 1
                        for currentIndex, currentSortedDf in sortedDf.iterrows():
                            if currentIndex == index:
                                currentRank = i
                            i += 1
                    if started is False: result = ""
                    result += str(currentRank) + ") " + str(row["score"]) + " --> " + configToFileId(postToConfig(row)) + "\n"
                elif outType == TopOutTypeEnum.STRING:
                    if started is False: result = ""
                    result += '"' + configToFileId(postToConfig(row)) + '",\n'
                elif outType == TopOutTypeEnum.LIST_DICT:
                    if started is False: result = []
                    currentDict = dict()
                    for key, column in row.items():
                        currentDict[key] = column
                    result.append(currentDict)
                elif outType == TopOutTypeEnum.LIST_STRING:
                    if started is False: result = []
                    result.append(configToFileId(postToConfig(row)))
                started = True
                u += 1
            return result
    
    def topBy(self, by, additionalFiled=["count"]):
        top = []
        self.sortSelection()
        byList = []
        for index, current in self.selection.iterrows():
            if current[by] not in byList:
                byList.append(current[by])
                top.append({"score": current["score"], additionalFiled[0]: current[additionalFiled[0]], by: current[by]})
        return top
    
    def meanBy(self, by="score"):
        sum = 0.0
        count = 0
        for index, current in self.selection.iterrows():
            sum += current[by]
            count += 1
        mean = sum / float(count)
        return mean
    
    @staticmethod
    def notEqualsParamToStr(nep):
        if nep is None or len(nep) == 0:
            return str(nep)
        else:
            if isinstance(nep[0], dict):
                newList = []
                for current in nep:
                    newList.append(collections.OrderedDict(sorted(current.items())))
                return str(newList)
            else:
                return str(nep)
    
    def topHierarchNotEqualsOn(self, params, outType=TopOutTypeEnum.STRING_TOP, realIndex=False, max=None):
        def contains(element1, elementList):
            for element2 in elementList.iterrows():
                (index, element2) = element2
                i = 0
                isEquals = True
                for key1, column1 in element1.items():
                    column2 = element2[key1]
                    if column1 != column2:
                        isEquals = False
                    i += 1
                if isEquals:
                    return True
            return False
        
        def getValues(dictOrTuple):
            if isinstance(dictOrTuple, tuple):
                fieldList, currentMax = dictOrTuple
                dictOrTuple = {"fieldList": fieldList, "max": currentMax}
            fieldList = dictOrTuple.get("fieldList", None)
            currentMax = dictOrTuple.get("max", 1)
            minDiff = dictOrTuple.get("minDiff", None)
            jump = dictOrTuple.get("jump", False)
            
            return (fieldList, currentMax, minDiff, jump)

        top = None
        if max is None:
            max = 0
            for values in params:
                (fieldList, currentMax, minDiff, jump) = getValues(values)
                max += currentMax
        
        for values in params:
            (fieldList, currentMax, minDiff, jump) = getValues(values)
            if top is None:
                top = self.topNotEqualsOn(fieldList, max=currentMax, outType=TopOutTypeEnum.DATA_FRAME, minDiff=minDiff)
            else:
                if jump:
                    requestMax = currentMax + len(top)
                    currentTop = self.topNotEqualsOn(fieldList, max=requestMax, outType=TopOutTypeEnum.DATA_FRAME, minDiff=minDiff)
                    currentTopList = currentTop.iterrows()
                    addedCount = 0
                    for index, current in currentTopList:
                        if not contains(current, top) and addedCount <= currentMax:
                            top = top.append(current)
                            addedCount += 1
                else:
                    currentTop = self.topNotEqualsOn(fieldList, max=currentMax, outType=TopOutTypeEnum.DATA_FRAME, minDiff=minDiff)
                    currentTopList = currentTop.iterrows()
                    for index, current in currentTopList:
                        if not contains(current, top):
                            top = top.append(current)

        
        top = top[:max]
        
        return self.convertTop(top, outType=outType, realIndex=realIndex)
  

    def topNotEqualsOn(self, fieldList, max=20, outType=TopOutTypeEnum.STRING_TOP, realIndex=False, minDiff=None):
        def contains(valueList, valueMatrix, minDiff=None):
            if valueMatrix is None or len(valueMatrix) == 0:
                return False
            for currentValues in valueMatrix:
                containsValueList = True
                i = 0
                for currentValue in currentValues:
                    currentMinDiff = None
                    if minDiff is not None:
                        currentMinDiff = minDiff[i]
                    if currentMinDiff is None:
                        if currentValue != valueList[i]:
                            containsValueList = False
                            break
                    else:
                        if abs(currentValue - valueList[i]) > currentMinDiff:
                            containsValueList = False
                            break
                    i += 1
                if containsValueList is True:
                    return True
                    
            return False
        
        
        if fieldList is None or not isinstance(fieldList, list) or len(fieldList) == 0:
            return self.top(max=max)
        else:
            sortedValues = Graph.sort(self.selection, by="score")
            result = []
            valueMatrix = []
            rank = 1
            for index, currentRow in sortedValues.iterrows():
                valueList = []
                for field in fieldList:
                    valueList.append(currentRow[field])
                if not contains(valueList, valueMatrix, minDiff=minDiff):
                    valueMatrix.append(valueList)
                    result.append(currentRow)
                rank += 1
                
                if len(result) == max:
                    break
            
            
            return self.convertTop(pd.DataFrame(result), outType=outType, realIndex=realIndex)

    def sortSelection(self, by="score"):
        self.selection = Graph.sort(self.selection, by=by)
        

    def select(self, queries):
        for key, value in list(queries.items()):
            if value is not None:
                if isinstance(value, tuple) and len(value) == 2:
                    self.selectInterval(key, value[0], value[1])
                elif isinstance(value, list) and len(value) >= 2:
                    self.selectEqualsOr(key, value)
                else:
                    self.selectEquals(key, value)
    
    def selectInterval(self, field, min, max):
        df = self.selection
        selection1 = df[df[field] >= min]
        selection2 = selection1[selection1[field] <= max]
        self.selection = selection2
                
    def selectTopScore(self, field, values):
        if not isinstance(values, list):
            values = [values]
        
        originalSelection = self.getSelectionCopy()
        resultSelection = None
        for value in values:
            self.setSelection(originalSelection)
            self.selectEquals(field, value)
            sortedValues = Graph.sort(self.selection, by="score")
            currentSelection = sortedValues.head(1)
            if resultSelection is None:
                resultSelection = currentSelection
            else:
                resultSelection = pd.concat([resultSelection, currentSelection])
        self.selection = resultSelection
    
        
    def getFirstSelection(self):
        for index, current in self.selection.iterrows():
            return current
                
    def selectNotEquals(self, field, value):
        self.selection = self.selection[self.selection[field] != value]
        
    def selectEquals(self, field, value):
        self.selection = self.selection[self.selection[field] == value]
        
    def deleteEqualsAnd(self, fields, values):
        originalSelection = self.getSelectionCopy()
        
        newData = []
        for index, current in originalSelection.iterrows():
            i = 0
            isEquals = True
            for field in fields:
                value = values[i]
                if current[field] != value:
                    isEquals = False
                    break
                i += 1
            if not isEquals:
                newData.append(current)
        self.selection = pd.DataFrame(newData)
        
    def selectEqualsOr(self, field, values):
        originalSelection = self.getSelectionCopy()
        
        resultSelection = None
        for value in values:
            self.setSelection(originalSelection)
            currentSelection = self.selection[self.selection[field] == value]
            if resultSelection is None:
                resultSelection = currentSelection
            else:
                resultSelection = pd.concat([resultSelection, currentSelection])
        self.selection = resultSelection
            
        
    def selectLowerEquals(self, field, value):
        self.selection = self.selection[self.selection[field] <= value]
        
    def selectHigherEquals(self, field, value):
        self.selection = self.selection[self.selection[field] >= value]
        
    def selectLower(self, field, value):
        self.selection = self.selection[self.selection[field] < value]
        
    def selectHigher(self, field, value):
        self.selection = self.selection[self.selection[field] > value]
            
    def hist(self, by, bins=None, yLogScale=True, height=6):
        freq = self.selection[by].value_counts(bins=bins)
        freqLen = len(freq)
        width = None
        if freqLen > 35:
            width = 20
        else:
            width = (float(20-6)/float(35-0)) * freqLen + 6
        fig = plt.figure(figsize=(width, height))
        freq = freq.sort_index()
        width = 1.0
        binsPrint = ""
        if bins is not None:
            binsPrint = " (" + str(bins) + " bins)"
        plt.title(by + ' frequency' + binsPrint)
        yLogScalePrint = ""
        if yLogScale:
            yLogScalePrint = " (log scale)"
        plt.ylabel('Frequency' + yLogScalePrint)
        plt.xlabel('Values')
        ax = plt.axes()
        if yLogScale:
            plt.yscale('log', nonposy='clip')
        ax.set_xticks(np.arange(len(freq)) + (width / 2))
        ax.set_xticks(freq)
        fig = freq.plot(kind='bar')
        # plt.show()
        return fig
    
    def test(self):
        self.selection.cumsum()
        
        plt.figure(); self.selection.plot(); plt.legend(loc='best')
        plt.show()
    
    def selectionSize(self):
        return len(self.selection)

    def getSelection(self):
        return self.selection
    
    def getSelectionCopy(self):
        return self.selection.copy(deep=True)
    
    def setSelection(self, selection):
        self.selection = selection
    
    @staticmethod
    def getPointSize(size):
        if size < 10 :
            return 60
        elif size < 30:
            return 50
        elif size < 100:
            return 40
        elif size < 500:
            return 30
        elif size < 3000:
            return 20
        else:
            return 10
        
    # TODO Print first top with big points and the curve on these tops
    def scatterPlot3D(self, x, y, z=None, plotCurve=None, curvesQueries=None, zBins=None, xLogScale=False, mean=None,
                      yLogScale=False, title=None, plotSize=10, pointSize=None, verbose=False, degree=None, legend=None, paletteType="multicolor", palette=None,
                      windowRatio=0.3, alphaCurve=0.13, distanceType=1, curveType=None,
                      xLim=None, yLim=None, onlyTopScore=False, curveLegend=True):
        if onlyTopScore:
            xGroups = []
            xElements = self.selection.groupby(by=x)
            for el in xElements:
                xGroups.append(el[0])
            self.selectTopScore(x, xGroups)
        
        # Init some vars :
        if pointSize is None:
            pointSize = Graph.getPointSize(self.selectionSize())
        
        if curveType == None:
            if degree is None:
                curveType = Graph.CurveTypeEnum.MEAN
            else:
                curveType = Graph.CurveTypeEnum.POLY
        
        if z is None:
            zBins = 0
        if zBins is None:
            try:
                zBins = len(self.selection.groupby(by=z))
                if zBins > 500:
                    zBins = 500
            except TypeError:
                if curvesQueries is not None:
                    zBins = len(curvesQueries)
            
                
        if plotCurve is None:
            if curvesQueries is not None:
                plotCurve = True
            elif y == "score":
                plotCurve = True
            else:
                plotCurve = False
        
        if legend is None: 
            if z is not None and plotCurve and zBins < 10 and zBins > 1 and curvesQueries is not None:
                legend = True
            else:
                legend = False
        
        # Color :
        if palette is None: 
            if z is None:
                palette = None
            elif z == "score":
                palette = sns.diverging_palette(0, 122, l=40, sep=1, n=zBins, center="light")
            elif legend:
                # palette = sns.light_palette((210, 90, 60), n_colors=zBins, input="husl")
                # palette = sns.light_palette((210, 90, 60), n_colors=zBins, input="husl")
                # palette = sns.color_palette("GnBu_d", n_colors=zBins)
                if paletteType == "degraded":
                    palette = sns.color_palette("GnBu_d", n_colors=zBins)
                else:
                    palette = sns.color_palette(n_colors=zBins)
            else:
                # palette = sns.diverging_palette(35, 215, l=40, sep=1, n=zBins, center="light") # TODO
                palette = None
            
        # TODO delete :
#         flatui = ["#969696", "#007ACC", "#009B95"]
        flatui = ["#007ACC", "#969696"]
#         flatui = ["#007ACC"]
        palette = sns.color_palette(flatui)
            
        if mean is None:
            if plotCurve:
                if curveType == Graph.CurveTypeEnum.MEAN:
                    mean = True
                else:
                    mean = False
        
        # Print some infos :
        if verbose:
            zMin = self.selection[z].min()
            zMax = self.selection[z].max()
            print("selectionSize = " + str(self.selectionSize()))
            print("zMin = " + str(zMin))
            print("zMax = " + str(zMax))
            print("zBins = " + str(zBins))
            print("len(self.selection.groupby(by=z)) = " + str(len(self.selection.groupby(by=z))))
            print("len(self.df.groupby(by=z)) = " + str(len(self.df.groupby(by=z))))
        
        [
            (["size", "window"], [100, 2], 20),
            (["removeStopWords", "removePunct", "lemma"], None, 6),
            (["min_count"], [10], 3),
        ]
        
        # Plot points :
        if palette is None:
            hue = None
        else:
            hue = z
        fig = sns.lmplot(x, y, size=plotSize, hue=hue, data=self.selection, fit_reg=False, legend=legend, palette=palette, scatter_kws={'alpha': 1.0, "s": pointSize})
        
        # Set the scale :
        self.setScaleAndTitle(fig, x, y, z, title, xLogScale, yLogScale, xLim, yLim, onlyTopScore)
        
        # Plot curves :
        if plotCurve:
            if curvesQueries is None or len(curvesQueries) == 0:
                if curveType == Graph.CurveTypeEnum.POLY:
                    Graph.plotPolyCurve(x, y, self.getSelection(), xLogScale=xLogScale, degree=degree, mean=mean)
                else:
                    Graph.plotMeanCurve(x, y, self.getSelection(), windowRatio, alphaCurve, distanceType, mean=mean, xLogScale=xLogScale)
            else:
                originalSelection = self.getSelectionCopy()
                # colors = ["black", "green", "red", "purple", "yellow", "pink", "brown", "blue", "white"]
                colors = sns.color_palette(n_colors=zBins)
                colorIndex = 0
                curvePlotList = []
                curveLabelList = []
                for query in curvesQueries:
                    if len(colors) > 0:
                        color = colors[0]
                    else:
                        color  = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)
                        
                    # TODO delete :
#                     colors = ["#969696", "#007ACC", "#009B95"]
                    colors = ["#969696", "#007ACC"]
#                     colors = ["#007ACC"]
                    
                    if hue is not None:
                        color = colors[colorIndex]
                    self.setSelection(originalSelection)
                    self.select({z: query})
                    if self.selectionSize() > 1:
#                         for index, value in self.selection.iterrows():
#                             print value
#                         print query
                        if curveType == Graph.CurveTypeEnum.POLY:
                            curvePlot = Graph.plotPolyCurve(x, y, self.getSelection(), xLogScale=xLogScale, degree=degree, alpha=0.5, color=color, mean=mean)
                        else:
                            curvePlot = Graph.plotMeanCurve(x, y, self.getSelection(), windowRatio, alphaCurve, distanceType, mean=mean, xLogScale=xLogScale, color=color, alphaColor=0.5)
                        curvePlotList.append(curvePlot)
                        curveLabelList.append(str(z) + " = " + str(query))
                        if isinstance(query, tuple):
                            colorIndex += 2
                        else:
                            colorIndex += 1
                self.setSelection(originalSelection)
                if curveLegend:
                    plt.legend(curvePlotList, curveLabelList)
        
        return fig

    def setScaleAndTitle(self, fig, x, y, z, title, xLogScale, yLogScale, xLim, yLim, onlyTopScore):
        if z is None:
            z = ""
        else:
            z = " " + z + "s"
        
        if onlyTopScore:
            y = "best " + y
        
        if title is None:
            plt.title(x + "/" + y + z)
        else:
            plt.title(title)
                
        ax = fig.axes[0][0]
        
        xLogScoreStr = ""
        yLogScoreStr = ""
        if xLogScale:
            xLogScoreStr = " (log scale)"
            ax.set_xscale('log')
        if yLogScale:
            yLogScoreStr = " (log scale)"
            ax.set_yscale('log')
        
        plt.xlabel(x + xLogScoreStr)
        plt.ylabel(y + yLogScoreStr)
            
        
        if xLim is not None:
            ax.set_xlim(xLim)
        if yLim is not None:
            ax.set_ylim(yLim)
    
    @staticmethod
    def initCurve(xId, yId, selection, xLogScale, mean, base):
        # Sort the selection :
        # selection = selection.copy(deep=True)
        selection = Graph.sort(selection, by=yId, ascending=True)
        
        # Create x and y :
        x = []
        y = []
        for _, row in selection.iterrows():
            x.append(row[xId])
            y.append(row[yId])
        
        # We create the mean on y for all different x:
        if mean:
            xMean = []
            yMean = []
            previous = None
            yCount = []
            yIndex = None
            for i in range(len(x)):
                currentX = x[i]
                currentY = y[i]
                if previous is None:
                    previous = currentX
                    yCount.append(1)
                    yMean.append(currentY)
                    yIndex = 0
                    xMean.append(currentX)
                else:
                    if currentX == previous:
                        yCount[yIndex] += 1
                        yMean[yIndex] += currentY
                    else:
                        yIndex += 1
                        previous = currentX
                        yCount.append(1)
                        yMean.append(currentY)
                        xMean.append(currentX)
            for i in range(len(yMean)):
                yMean[i] = float(yMean[i]) / float(yCount[i])
            x = xMean
            y = yMean
        
        # if the x is on log scale, we convert it :
        if xLogScale:
            newX = []
            for current in x:
                newX.append(math.log(current, base))
            x = newX
        
        # We get mins and max :
        xMin = min(x)
        xMax = max(x)
        
        return (x, y, xMin, xMax)

    class CurveTypeEnum():
        (
            POLY,
            MEAN
        ) = list(range(2))
    
    @staticmethod
    def plotMeanCurve(xId, yId, selection, windowRatio, alphaCurve, distanceType, xLogScale=False, mean=True, color="black", alphaColor=0.2, base=10, pointCount=1000):
        (x, y, xMin, xMax) = Graph.initCurve(xId, yId, selection, xLogScale, mean, base)
        
        # Create all x values of the curve:
        xp = np.linspace(xMin, xMax, pointCount)
        
        window = windowRatio * float(abs(xMax - xMin))
        
        yp = []
        for currentXp in xp:
            xList = []
            yList = []
            coefList = []
            # We take all x in the window:
            weGotOnePoint = False
            for i in range(len(x)):
                currentX = x[i]
                if currentX > (currentXp - window/2.0) and currentX < (currentXp + window/2.0):
                    # We calculate the distance:
                    xList.append(currentX)
                    yList.append(y[i])
                    realDistance = abs(currentX - currentXp)
                    normalizedRealDistance = realDistance / (window/2.0)
                    # We calculate the coef:
                    coef = hjNormalizedLaw(normalizedRealDistance, distanceType=distanceType, alpha=alphaCurve)
                    coefList.append(coef)
                    weGotOnePoint = True
            # Then we normalize all coefs:
            coefSum = sum(coefList)
            for i in range(len(coefList)):
                coefList[i] = coefList[i] / coefSum
            # And we take the weighted sum :
            currentYp = 0
            for i in range(len(coefList)):
                currentYp += coefList[i] * yList[i]
            # Finally we add the y coordinate of the current point of the curve:
            if not weGotOnePoint:
                yp.append(yp[-1])
            else:
                yp.append(currentYp)
            
        return Graph.finishCurve(xp, yp, xLogScale, color, alphaColor, base)
    
    @staticmethod
    def finishCurve(xp, yp, xLogScale, color, alpha, base):
        # Now the real x of the curve must be on the log scale :
        if xLogScale:
            xpNew = []
            for current in xp:
                xpNew.append(base ** current)
            xp = xpNew
    
        curvePlot, = plt.plot(xp, yp, color=color, alpha=alpha)
        
        return curvePlot

    @staticmethod
    def plotPolyCurve(xId, yId, selection, xLogScale=False, mean=False, color="black", alpha=0.2, base=10, pointCount=1000, degree=3):

        (x, y, xMin, xMax) = Graph.initCurve(xId, yId, selection, xLogScale, mean, base)
        
        # Then we calculate a polynomial curve on a normal scale:
        z = np.polyfit(x, y, degree)
        p = np.poly1d(z)
        xp = np.linspace(xMin, xMax, pointCount)
        
        # The y of the curve must be calculated on the xp with normal scale :
        yp = p(xp)
        
        return Graph.finishCurve(xp, yp, xLogScale, color, alpha, base)
        

    
def test2():
    
    x = np.array([0.0, 1.0, 1.5, 2.5, 5.0,  9.0,  13.0])
    y = np.array([0.0, 1.0, 3.0, 5.0, 7.0, 8.0, 8.0])
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    
    import matplotlib.pyplot as plt
    xp = np.linspace(0, 13.0, 200)
    _ = plt.plot(x, y, '.', xp, p(xp))
    plt.show()
    
def test1(g):
    g.select({ "dataPart": 0.5 })
    fig = g.scatterPlot3D("window", "size", "score", yLogScale=True, xLogScale=False)
    
    plt.plot([10, 3], [1000, 4], 'k-', lw=2)
    plt.plot([3, 10, 11], [4, 1000, 10000], 'k-', lw=2)
    plt.show(fig)


def test4(g):
    g.resetSelection()
    queries = { "dataPart": 0.0, "window": 3, "size": None, "min_count": 0,
                "sample": None, "negative": None, "iter": 10, "alpha": (0.02, 0.2),
                "removeStopWords": False, "removePunct": True, "toLowerCase": True, "lemma": True }
    g.select(queries)
    
    fig = g.scatterPlot3DRegression("size", "score", "sample", curvesQueries=[0.0, 1e-5], yLogScale=False, xLogScale=True, pointSize=20)
    
    plt.show(fig)

def test5(g):
    sizeCurves = [100, (1000, 6000)]
    windowCurves = [1, 2, 3, (4, 8)]
    dataPartCurves = [(0.0, 0.25), (0.5, 0.75), 1.0]
    g.resetSelection()
    queries = { "dataPart": None, "window": (1, 6), "size": (50, 6000), "min_count": 0,
                "sample": 0.0, "negative": 0.0, "iter": None, "alpha": 0.1,
                "removeStopWords": False, "removePunct": True, "toLowerCase": True, "lemma": True }
    g.select(queries)
    plt.show(g.scatterPlot3D("iter", "score", "size", degree=7, curvesQueries=sizeCurves, legend=True))
    plt.show(g.scatterPlot3D("iter", "score", "window", degree=7, curvesQueries=windowCurves, legend=True))
    plt.show(g.scatterPlot3D("iter", "score", "dataPart", degree=7, curvesQueries=dataPartCurves, legend=True))
    
    
def test6(g):
    g.resetSelection()
    queries = { "dataPart": None, "window": None, "size": None, "min_count": 0,
                "sample": 0.0, "negative": 0, "iter": 22, "alpha": None,
                "removeStopWords": False, "removePunct": True, "toLowerCase": True, "lemma": True }
    g.select(queries)
    print(g.selectionSize())
    plt.show(g.scatterPlot3D("alpha", "window", "score", zBins=10))
    
    
    
def test7(g):
    # Il faut vérifier qu'on a bien essayer toutes les dataPart pour les iter
    g.resetSelection()
    queries = { "dataPart": None, "window": None, "size": None, "min_count": 0,
                "sample": 0.0, "negative": 0, "iter": None, "alpha": 0.1,
                "removeStopWords": False, "removePunct": True, "toLowerCase": True, "lemma": True }
    g.select(queries)
    print(g.selectionSize())
    plt.show(g.scatterPlot3D("dataPart", "iter", "score", zBins=1))
    
    
    
def test8(g):
    g.resetSelection()
    queries = { "dataPart": None, "window": None, "size": None, "min_count": 0,
                "sample": 0.0, "negative": 0, "iter": 22, "alpha": (0.0, 0.26),
                "removeStopWords": False, "removePunct": True, "toLowerCase": True, "lemma": True }
    g.select(queries)
    plt.show(g.scatterPlot3D("alpha", "score", "size", degree=8,
                curvesQueries=[100, (500, 4400)], legend=True))
    
    
    
    
    
def test9(g):
    g.resetSelection()
    queries = { "dataPart": None, "window": None, "size": None, "min_count": None,
                "sample": None, "negative": None, "iter": None, "alpha": None,
                "removeStopWords": None, "removePunct": None, "toLowerCase": None, "lemma": None }
    g.select(queries)
    g.selectEqualsOr("size", [100, 3000])
    g.selectEqualsOr("window", [2, 3, 8])
    g.hist("sample", bins=20)
    
    
    
    
    
def test10(g):
    g.resetSelection()
    queries = { "dataPart": 0.75, "window": None, "size": None, "min_count": 0,
                "sample": None, "negative": 0, "iter": 22, "alpha": 0.1,
                "removeStopWords": False, "removePunct": True, "toLowerCase": True, "lemma": True,
                "score": (0.63, 1.0)}
    g.select(queries)
    g.selectEqualsOr("size", [100, 3000])
    g.selectEqualsOr("window", [2, 3, 8])
    plt.show(g.scatterPlot3D("sample", "score", "window", windowRatio=0.5, alphaCurve=0.7, legend=True, xLim=[-0.00005, 0.0004]))
    
    
    
    
    
def test11(g):
    g.resetSelection()
    queries = { "dataPart": 0.75, "window": None, "size": None, "min_count": 0,
                "sample": None, "negative": 0, "iter": 22, "alpha": 0.1,
                "removeStopWords": False, "removePunct": True, "toLowerCase": True, "lemma": True,
                "score": (0.63, 1.0)}
    g.select(queries)
    g.selectEqualsOr("size", [100, 3000])
    g.selectEqualsOr("window", [2, 3, 8])
    plt.show(g.scatterPlot3D("sample", "score", "size", windowRatio=0.5, alphaCurve=0.7, xLim=[-0.00005, 0.0004], degree=7, curvesQueries=[100, (101, 60000)]))
    
    
    
def testOnlyTop(g):
    g.resetSelection()
    queries = { "dataPart": 0.75, "window": None, "size": None, "min_count": 0,
                "sample": None, "negative": 0, "iter": 22, "alpha": 0.1,
                "removeStopWords": False, "removePunct": True, "toLowerCase": True, "lemma": True,
                "score": (0.63, 1.0)}
    g.select(queries)
    g.selectEqualsOr("size", [100, 3000])
    g.selectEqualsOr("window", [2, 3, 8])
    plt.show(g.scatterPlot3D("size", "sample", "window", plotCurve=True, degree=7, legend=True, onlyTopScore=True))
    
    
def testTopNotEqualsOn(g):
    g.resetSelection()
    queries = { "dataPart": None, "window": None, "size": None, "min_count": None,
                "sample": None, "negative": None, "iter": None, "alpha": None,
                "removeStopWords": None, "removePunct": None, "toLowerCase": None, "lemma": None}
    g.select(queries)
    print(g.topNotEqualsOn(["size", "window"], multiD2vFormat=False))
    
def testHierTopNotEqualsOn(g):
    multiD2vHierarchicalTop = [{ "fieldList": ["window"], "minDiff": 2, "max": 8, "jump": False }]
    print(g.topHierarchNotEqualsOn(multiD2vHierarchicalTop))

def testHierTopNotEqualsOn2():
    data = MongoClient()["multid2vscore"]["pc-lf-saf-n2016-1"].find({})
    g = Graph(data)
    g.resetSelection()
    queries = { "randomMultiD2vModel": None, "count": None,
                "dbNameSource": None, "collectionNameSource": None, 
                "fileIdSet": None, "score": None,
                "notEqualsParams": None }
    g.select(queries)
    curvesQueries = \
    [
        [(["size"], 10), (["window"], 20)],
        [(["window"], 10), (["size"], 20)],
        [(["window", "size"], 10), (["dataPart"], 20)],
        [(["window", "size"], 25)],
        [(["sample"], 3), (["iter"], 8), (["alpha"], 12), (["negative"], 16)],
        [(["min_count"], 20), (["dataPart"], 4), (["window", "size"], 15)],
        [(["window"], 8), (["size"], 10), (["dataPart"], 10)],
    ]
    # g.selectEquals("notEqualsParams", t)
    plt.show(g.scatterPlot3D("count", "score", "notEqualsParams", degree=4, curvesQueries=curvesQueries))

def testHierTopNotEqualsOn3():
    data = MongoClient()["multid2vscore"]["pc-lf-saf-n2016-1"].find({})
    g = Graph(data)
    g.resetSelection()
    queries = { "randomMultiD2vModel": None, "count": None,
                "dbNameSource": None, "collectionNameSource": None, 
                "fileIdSet": None, "score": None,
                "notEqualsParams": None }
    g.select(queries)
    
    for index, current in g.getSelection().iterrows():
        print(current["notEqualsParams"])
    
if __name__ == '__main__':
    scores = MongoD2VScore(dbId="d2vscore-archive3", scoreTypeAndFeatures="pc-lf-cv2016")
    print(listToStr(scores.toDataFrame()[0]))
    exit()
    g = Graph()
#     test1(g)
#     test2()
#     test3(g)
#     test4(g)
#     test5(g)
#     test6(g)
#     test7(g)
#     test8(g)
#     test9(g)
#     test10(g)
#     test11(g)
#     testOnlyTop(g)
#     testHierTopNotEqualsOn(g)
#     testHierTopNotEqualsOn2()
#     testHierTopNotEqualsOn3()

    g.increaseData("dataPart", 0.2)

#     g = Graph(MongoClient()["multid2vscore"]["pc-lf-saf-cv2016-1"].find({}))
#     g.resetSelection()
#     queries = { "randomMultiD2vModel": False, "count": None,
#                 "dbNameSource": None, "collectionNameSource": None, 
#                 "fileIdSet": None, "score": None,
#                 "notEqualsParams": None }
#     g.select(queries)
#     top = g.top(max=1, outType=TopOutTypeEnum.LIST_DICT)[0]
#     print listToStr(top["notEqualsParams"])

