# coding: utf-8

from fr.hayj.machinelearning.stat import *;
from fr.hayj.util.system import *;

import unittest

# The level allow the unit test execution to choose only the top level test 
unittestLevel = 12

 
if unittestLevel <= 1:
    class Test1(unittest.TestCase):
        def testTopNotEqualsOn(self):
            data1 = \
            [
                {"score": 0.2, "window": 2, "size": 100, "alpha": 0.1},
                {"score": 0.3, "window": 3, "size": 100, "alpha": 0.1},
                {"score": 0.4, "window": 2, "size": 3000, "alpha": 0.1},
                {"score": 0.5, "window": 2, "size": 3000, "alpha": 0.1},
                {"score": 0.3, "window": 2, "size": 100, "alpha": 0.2},
                {"score": 0.3, "window": 2, "size": 100, "alpha": 0.1},
                {"score": 0.4, "window": 2, "size": 100, "alpha": 0.1},
                {"score": 0.4, "window": 2, "size": 100, "alpha": 0.3},
            ]
            
            g = Graph(data1)
            
            self.assertTrue(len(g.top(outType=TopOutTypeEnum.DATA_FRAME)) == 8)
            self.assertTrue(len(g.topNotEqualsOn(["size", "window"], outType=TopOutTypeEnum.DATA_FRAME)) == 3)
            self.assertTrue(len(g.topNotEqualsOn(["size"], outType=TopOutTypeEnum.DATA_FRAME)) == 2)
            self.assertTrue(len(g.topNotEqualsOn(["size", "window", "alpha"], outType=TopOutTypeEnum.DATA_FRAME)) == 5)
            self.assertTrue(len(g.topNotEqualsOn(["size", "window", "alpha"], max=1, outType=TopOutTypeEnum.DATA_FRAME)) == 1)
            self.assertTrue(len(g.topNotEqualsOn(["size", "window", "alpha"], max=3, outType=TopOutTypeEnum.DATA_FRAME)) == 3)
            self.assertTrue(len(g.topNotEqualsOn(["size", "window", "alpha"], max=5, outType=TopOutTypeEnum.DATA_FRAME)) == 5)
            self.assertTrue(len(g.topNotEqualsOn(["size", "window", "alpha"], max=6, outType=TopOutTypeEnum.DATA_FRAME)) == 5)
            
if unittestLevel <= 2:
    class Test2(unittest.TestCase):            
        def testTopHierarchNotEqualsOn(self):
            data1 = \
            [
                {"score": 0.2, "window": 2, "size": 100, "alpha": 0.1},
                {"score": 0.3, "window": 3, "size": 100, "alpha": 0.1},
                {"score": 0.4, "window": 2, "size": 3000, "alpha": 0.1},
                {"score": 0.5, "window": 2, "size": 3000, "alpha": 0.1},
                {"score": 0.3, "window": 2, "size": 100, "alpha": 0.2},
                {"score": 0.3, "window": 2, "size": 100, "alpha": 0.1},
                {"score": 0.45, "window": 2, "size": 100, "alpha": 0.1},
                {"score": 0.4, "window": 2, "size": 100, "alpha": 0.3},
            ]
             
            g = Graph(data1)
             
            top = g.topHierarchNotEqualsOn([(["size"], 2), (["window"], 2), (["alpha"], 2)], outType=TopOutTypeEnum.DATA_FRAME)
            
            
            tempTop = []
            for index, current in top.iterrows():
                tempTop.append(current)
            top = tempTop
            
            self.assertTrue(top[0]["size"] == 3000)
            self.assertTrue(top[0]["window"] == 2)
            self.assertTrue(top[0]["alpha"] == 0.1)
             
            self.assertTrue(top[1]["size"] == 100)
            self.assertTrue(top[1]["window"] == 2)
            self.assertTrue(top[1]["alpha"] == 0.1)
             
            self.assertTrue(top[2]["size"] == 100)
            self.assertTrue(top[2]["window"] == 3)
            self.assertTrue(top[2]["alpha"] == 0.1)
             
            self.assertTrue(top[3]["size"] == 100)
            self.assertTrue(top[3]["window"] == 2)
            self.assertTrue(top[3]["alpha"] == 0.3)
             
             
            self.assertTrue(len(top) == 4)
        
        def testWithMongo(self):
            scores = MongoD2VScore(dbId="d2vscore-archive6", scoreTypeAndFeatures="pc-lf-cv2016")
            g = Graph(scores.toDataFrame())
               
            
            top = g.topHierarchNotEqualsOn([(["size", "window"], 2), (["sample"], 2)], outType=TopOutTypeEnum.DATA_FRAME)
            
            
            tempTop = []
            for index, current in top.iterrows():
                tempTop.append(current)
            top = tempTop
            
            self.assertTrue(top[0]["size"] == 100)
            self.assertTrue(top[0]["window"] == 2)
            self.assertTrue(top[0]["sample"] == 0.000101)
               
            self.assertTrue(top[1]["size"] == 100)
            self.assertTrue(top[1]["window"] == 3)
            self.assertTrue(top[1]["sample"] == 9.4e-05)
               
            self.assertTrue(top[2]["size"] == 100)
            self.assertTrue(top[2]["window"] == 2)
            self.assertTrue(top[2]["sample"] == 8.1e-05)
               
            self.assertTrue(len(top) == 3)
  
  
            top = g.topHierarchNotEqualsOn([(["size", "window"], 2), (["alpha"], 2)], outType=TopOutTypeEnum.STRING_TOP)
            self.assertTrue('-->' in top)
            top = top.strip().split("\n")
            self.assertTrue(len(top) == 3)
            
            top = g.topHierarchNotEqualsOn([(["size", "window"], 2), (["alpha"], 2)], outType=TopOutTypeEnum.STRING)
            self.assertTrue('-->' not in top)
            top = top.strip().split("\n")
            self.assertTrue(len(top) == 3)

if unittestLevel <= 3:
    class Test3(unittest.TestCase):            
        def testOut(self):
            scores = MongoD2VScore(dbId="d2vscore-archive6", scoreTypeAndFeatures="pc-lf-cv2016")
            g = Graph(scores.toDataFrame())
               
            top = g.topHierarchNotEqualsOn([(["size", "window"], 2), (["sample"], 2)], outType=TopOutTypeEnum.DATA_FRAME)
            self.assertTrue(isinstance(top, pd.DataFrame))
            top = g.topHierarchNotEqualsOn([(["size", "window"], 2), (["sample"], 2)], outType=TopOutTypeEnum.STRING)
            self.assertTrue(isinstance(top, str))
            top = g.topHierarchNotEqualsOn([(["size", "window"], 2), (["sample"], 2)], outType=TopOutTypeEnum.LIST_STRING)
            self.assertTrue(isinstance(top, list))
            self.assertTrue(isinstance(top[0], str))
            top = g.topHierarchNotEqualsOn([(["size", "window"], 2), (["sample"], 2)], outType=TopOutTypeEnum.STRING_TOP)
            self.assertTrue(isinstance(top, str))
            top = g.topHierarchNotEqualsOn([(["size", "window"], 2), (["sample"], 2)], outType=TopOutTypeEnum.LIST_DICT)
            self.assertTrue(isinstance(top, list))
            self.assertTrue(isinstance(top[0], dict))
            
            top = g.top(outType=TopOutTypeEnum.DATA_FRAME)
            self.assertTrue(isinstance(top, pd.DataFrame))
            top = g.top(outType=TopOutTypeEnum.STRING)
            self.assertTrue(isinstance(top, str))
            top = g.top(outType=TopOutTypeEnum.LIST_STRING)
            self.assertTrue(isinstance(top, list))
            self.assertTrue(isinstance(top[0], str))
            top = g.top(outType=TopOutTypeEnum.STRING_TOP)
            self.assertTrue(isinstance(top, str))
            top = g.top(outType=TopOutTypeEnum.LIST_DICT)
            self.assertTrue(isinstance(top, list))
            self.assertTrue(isinstance(top[0], dict))
            
if unittestLevel <= 4:
    class Test4(unittest.TestCase):            
        def testDelete(self):
            data1 = \
            [
                {"score": 0.2, "window": 2, "size": 100, "alpha": 0.1},
                {"score": 0.3, "window": 3, "size": 100, "alpha": 0.1},
                {"score": 0.4, "window": 2, "size": 3000, "alpha": 0.1},
                {"score": 0.5, "window": 2, "size": 3000, "alpha": 0.1},
                {"score": 0.3, "window": 2, "size": 100, "alpha": 0.2},
                {"score": 0.3, "window": 2, "size": 100, "alpha": 0.1},
                {"score": 0.4, "window": 2, "size": 100, "alpha": 0.1},
                {"score": 0.4, "window": 2, "size": 100, "alpha": 0.3},
            ]
            
            g = Graph(data1)
            
            g.deleteEqualsAnd(["window", "size", "alpha"], [2, 100, 0.1])
            
            self.assertTrue(g.selectionSize() == 5)

            
            
            

if unittestLevel <= 6:
    class Test6(unittest.TestCase):            
        def testMinDiff(self):
            data1 = \
            [
                {"score": 0.2, "window": 2, "size": 100, "alpha": 0.1},
                {"score": 0.3, "window": 3, "size": 100, "alpha": 0.1},
                {"score": 0.4, "window": 2, "size": 3000, "alpha": 0.1},
                {"score": 0.5, "window": 2, "size": 3000, "alpha": 0.1},
                {"score": 0.3, "window": 2, "size": 100, "alpha": 0.2},
                {"score": 0.3, "window": 2, "size": 100, "alpha": 0.1},
                {"score": 0.45, "window": 2, "size": 100, "alpha": 0.1},
                {"score": 0.4, "window": 2, "size": 100, "alpha": 0.3},
                {"score": 0.1, "window": 2, "size": 9000, "alpha": 0.3},
            ]
               
            g = Graph(data1)
               
            top = g.topHierarchNotEqualsOn([{ "fieldList": ["size"], "max": 2, "minDiff": [5000], "jump": False }], outType=TopOutTypeEnum.DATA_FRAME)
              
            tempTop = []
            for index, current in top.iterrows():
                tempTop.append(current)
            top = tempTop
              
            self.assertTrue(top[0]["size"] == 3000)
            self.assertTrue(top[0]["window"] == 2)
            self.assertTrue(top[0]["alpha"] == 0.1)
              
            self.assertTrue(top[1]["size"] == 9000)
            self.assertTrue(top[1]["window"] == 2)
            self.assertTrue(top[1]["alpha"] == 0.3)
              
              
            self.assertTrue(len(top) == 2)
              
            top = g.topHierarchNotEqualsOn([{ "fieldList": ["size"], "max": 10, "minDiff": [5000], "jump": False }], outType=TopOutTypeEnum.DATA_FRAME)
              
            self.assertTrue(len(top) == 2)
            
            
        def testJump(self):
            data1 = \
            [
                {"score": 0.5, "window": 2, "size": 3000, "alpha": 0.1},
                {"score": 0.4, "window": 1, "size": 100, "alpha": 0.1},
                {"score": 0.3, "window": 2, "size": 100, "alpha": 0.1},
                {"score": 0.35, "window": 1, "size": 3000, "alpha": 0.3},
                {"score": 0.2, "window": 3, "size": 100, "alpha": 0.3},
                {"score": 0.1, "window": 4, "size": 100, "alpha": 0.3},
                {"score": 0.05, "window": 6, "size": 100, "alpha": 0.3},
            ]
             
            g = Graph(data1)
             
            top = g.topHierarchNotEqualsOn([{ "fieldList": ["size"], "max": 2, "jump": False }, { "fieldList": ["window"], "max": 2, "jump": True }], outType=TopOutTypeEnum.DATA_FRAME)
            
            tempTop = []
            for index, current in top.iterrows():
                tempTop.append(current)
            top = tempTop
            
            self.assertTrue(len(top) == 4)
             
            top = g.topHierarchNotEqualsOn([{ "fieldList": ["size"], "max": 2, "jump": False }, { "fieldList": ["window"], "max": 2, "jump": False }], outType=TopOutTypeEnum.DATA_FRAME)
            
            tempTop = []
            for index, current in top.iterrows():
                tempTop.append(current)
            top = tempTop
            
            self.assertTrue(len(top) == 2)


if unittestLevel <= 7:
    class Test7(unittest.TestCase):  
        def testBoth(self):
            scores = MongoD2VScore(dbId="d2vscore-archive8", scoreTypeAndFeatures="pc-lf-cv2016")
            g = Graph(scores.toDataFrame())
              
            top = g.topHierarchNotEqualsOn([{ "fieldList": ["size", "window"], "max": 3, "jump": False }, { "fieldList": ["size"], "max": 2, "jump": False }], outType=TopOutTypeEnum.DATA_FRAME)
              
            tempTop = []
            for index, current in top.iterrows():
                tempTop.append(current)
            top = tempTop
              
            self.assertTrue(len(top) == 3)
              
            top = g.topHierarchNotEqualsOn([{ "fieldList": ["size", "window"], "max": 3, "jump": False }, { "fieldList": ["size"], "max": 2, "jump": True }], outType=TopOutTypeEnum.DATA_FRAME)
              
            tempTop = []
            for index, current in top.iterrows():
                tempTop.append(current)
            top = tempTop
              
            self.assertTrue(len(top) == 5)
              
            top = g.topHierarchNotEqualsOn([{ "fieldList": ["size", "window"], "max": 3, "jump": False }, { "fieldList": ["size"], "max": 2, "minDiff": [2600] }], outType=TopOutTypeEnum.DATA_FRAME)
              
            tempTop = []
            for index, current in top.iterrows():
                tempTop.append(current)
            top = tempTop
              
            self.assertTrue(len(top) == 4)
            self.assertTrue(top[3]["size"] > 2700)
              
            top = g.topHierarchNotEqualsOn([{ "fieldList": ["size", "window"], "max": 3, "jump": False }, { "fieldList": ["size"], "max": 9, "minDiff": [500], "jump": True }], outType=TopOutTypeEnum.DATA_FRAME)            
            
            tempTop = []
            for index, current in top.iterrows():
                tempTop.append(current)
            top = tempTop
              
            self.assertTrue(len(top) == 12)
            self.assertTrue(top[11]["size"] > 6000)


if unittestLevel <= 8:
    class Test8(unittest.TestCase):  
        def testBoth2(self):
            scores = MongoD2VScore(dbId="d2vscore-archive9", scoreTypeAndFeatures="pc-lf-cv2016")
            g = Graph(scores.toDataFrame())
            
            multiD2vHierarchicalTop = \
            [
                { "fieldList": ["size", "window"], "max": 2, "minDiff": None, "jump": False },
                { "fieldList": ["window"], "max": 1, "minDiff": [2], "jump": True },
            ]
            
            top = g.topHierarchNotEqualsOn(multiD2vHierarchicalTop, outType=TopOutTypeEnum.DATA_FRAME)            
            
            tempTop = []
            for index, current in top.iterrows():
                tempTop.append(current)
            top = tempTop
            
            self.assertTrue(len(top) == 3)
            
if unittestLevel <= 9:
    class Test9(unittest.TestCase):            
        def test1(self):
#             client = MongoClient()
#             db = client["multid2vscore"]
#             collection = db["pc-lf-saf-n2016-1"]
#             all = collection.find({})
            
            data = MongoClient()["multid2vscore"]["pc-lf-saf-n2016-1"].find({})
            
            g = Graph(data)
            
            for current in g.top(outType=TopOutTypeEnum.LIST_DICT):
                self.assertTrue(isinstance(current, dict))

if unittestLevel <= 10:
    class Test10(unittest.TestCase):            
        def test1(self):
            data = MongoClient()["multid2vscore"]["pc-lf-saf-n2016-1"].find({})
            
            g = Graph(data)
            
            
            mean = g.meanBy()
            
            print(mean)
            
            self.assertTrue(mean > 0.68 and mean < 0.8)

            
            
if unittestLevel <= 11:
    class Test11(unittest.TestCase):            
        def test1(self):
            t1 = \
            [
                { "fieldList": ["size", "window"], "max": 2, "minDiff": None, "jump": False },
                { "fieldList": ["window"], "max": 1, "minDiff": [2], "jump": True },
            ]
            
            t2 = \
            [
                { "fieldList": ["size", "window"], "minDiff": None, "jump": False, "max": 2 },
                { "jump": True, "fieldList": ["window"], "max": 1, "minDiff": [2] },
            ]
            
            t3 = \
            [
                { "fieldList": ["size", "window"], "minDiff": None, "jump": False, "max": 2 },
                { "jump": True, "fieldList": ["window"], "max": 1, "minDiff": [1] },
            ]

            self.assertTrue(t1 == t2)
            self.assertTrue(t1 != t3)
            
            
            print(Graph.notEqualsParamToStr(t3))
            print(Graph.notEqualsParamToStr([(["size", "window"], 2), (["sample"], 2)]))
            print(Graph.notEqualsParamToStr([]))
            
            
            
            
if unittestLevel <= 12:
    class Test12(unittest.TestCase):            
        def test1(self):
            g = Graph(MongoClient()["multid2vscore"]["pc-lf-saf-cv2016-3"].find({}))
            g.resetSelection()
            queries = { "type": None, "count": None,
                        "dbNameSource": None, "collectionNameSource": None, 
                        "fileIdSet": None, "score": None,
                        "notEqualsParams": None }
            g.select(queries)
            
            self.assertTrue(len(g.getFiledSet("notEqualsParamsStr")) > 1)
        
        
        def test2(self):
            g = Graph(MongoClient()["multid2vscore"]["pc-lf-saf-cv2016-3"].find({}))
            g.resetSelection()
            queries = { "type": "topdelta", "count": None,
                        "dbNameSource": None, "collectionNameSource": None, 
                        "fileIdSet": None, "score": (0.75, 1.0),
                        "notEqualsParams": None }
            g.select(queries)
            g.scatterPlot3D("count", "score", "topdeltaParametersStr", legend=False, curveLegend=False, curvesQueries=g.getFiledSet("topdeltaParametersStr"))
            