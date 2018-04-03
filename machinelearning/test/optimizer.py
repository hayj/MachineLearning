# coding: utf-8

from fr.hayj.machinelearning.optimizer import *;

import unittest
import numpy as np
import fr.hayj.util.text as text
from fr.hayj.util.text import listToStr
from fr.hayj.sts.d2vloader import *


# The level allow the unit test execution to choose only the top level test 
unittestLevel = 7;

if unittestLevel <= 1: 
    class TestCombinason(unittest.TestCase):
        def setUp(self):
            pass
        
        def testDepth0(self):
            optimizer = HierarchicalSearch(parameters1, self.funct1)
            allComb = optimizer.getCombinasons()
            self.assertTrue(len(allComb) == 12)
         
        def testDepth1(self):
            optimizer = HierarchicalSearch(parameters2, self.funct1)
            allComb = optimizer.getCombinasons()
            self.assertTrue(len(allComb) == 3)
          
        def testComplex(self):
            optimizer = HierarchicalSearch(parameters3, self.funct1)
            allComb = optimizer.getCombinasons()
            self.assertTrue(len(allComb) == 40)
  
        def testDisabled(self):
            optimizer = HierarchicalSearch(parameters4, self.funct1)
            allComb = optimizer.getCombinasons()
            self.assertTrue(len(allComb) == 8)
 
        def testComplexCross(self):
            optimizer = HierarchicalSearch(parameters5, self.funct1)
            allComb = optimizer.getCombinasons()
            self.assertTrue(len(allComb) == 6)
 
        def testComplexCross_1(self):
            optimizer = HierarchicalSearch(parameters5_1, self.funct1)
            allComb = optimizer.getCombinasons()
            self.assertTrue(len(allComb) == 72)
 
        def testComplexCross_2(self):
            optimizer = HierarchicalSearch(parameters6, self.funct1)
            allComb = optimizer.getCombinasons()
            self.assertTrue(len(allComb) == 60)
        
        def funct1(self, d):
            return len(list(d.items()))

if unittestLevel <= 2: 
    class TestFunct(unittest.TestCase):
        def setUp(self):
            pass
    
        def testNameConflict(self):
            optimizer = HierarchicalSearch(parameters10, self.funct1)
            allComb = optimizer.getCombinasons()
            self.assertTrue(len(allComb) == 16)
    
        def testNameConflict2(self):
            optimizer = HierarchicalSearch(parameters11, self.funct1)
            allComb = optimizer.getCombinasons()
            self.assertTrue(len(allComb) == 4)
         
        def testFunct1(self):
            optimizer = HierarchicalSearch(parameters20, self.funct1)
            (best, results) = optimizer.optimize()
            self.assertTrue(len(optimizer.getCombinasons()) == len(results))
#             print "Combinaisons length: " + str(len(optimizer.getCombinasons()))
#             print best[0][1]
#             print text.listToStr(best[0][0])
            self.assertTrue(best[0][1] == 9)
            self.assertTrue(len(best) < len(results))
            self.assertTrue(len(best) > 4)

        def testOverFlow(self):
            for param in parameters20:
                if param['name'] == 'regresser':
                    del param['force']
            optimizer = HierarchicalSearch(parameters20, self.funct1)
            getException = False
            try:
                (best, results) = optimizer.optimize()
            except OverflowError:
                getException = True
            self.assertTrue(getException)
            for param in parameters20:
                if param['name'] == 'regresser':
                    param['force'] = ['Ridge']

        def funct1(self, d):
            return len(list(d.items()))

if unittestLevel <= 3: 
    class TestConstraint(unittest.TestCase):
        def setUp(self):
            pass
    
        def testConstraintConflict(self):
            optimizer = HierarchicalSearch(parameters30, self.funct1)
            allComb = optimizer.getCombinasons()
            self.assertTrue(len(allComb) == 1)
            self.assertTrue(len(list(allComb[0].items())) == 3)
            self.assertTrue("Word2VecFeature.w2vNSimilarity" in allComb[0])
            
    
        def testDeleteCount(self):
            optimizer = HierarchicalSearch(parameters31, self.funct1)
            allComb = optimizer.getCombinasons()
            self.assertTrue(len(allComb) == 1)
            

        def funct1(self, d):
            return len(list(d.items()))

if unittestLevel <= 4: 
    class TestClean(unittest.TestCase):
        def setUp(self):
            pass
    
        def test1(self):
            optimizer = HierarchicalSearch(parameters3, self.funct1)
            allComb = optimizer.getCombinasons()
            self.assertTrue(len(allComb) == 40)
            
            optimizer = HierarchicalSearch(parameters3, self.funct1, validators=[self.validator])
            allComb = optimizer.getCombinasons()
            self.assertTrue(len(allComb) == 8)

            
        def funct1(self, d):
            return len(list(d.items()))
            
        def validator(self, config):
            if config['test2'] == 2 or config['test2'] == 4 or config['test2'] == 1:
                return False
            if 'test2.subtest2_2' in config and (config['test2.subtest2_2'] == 1 or config['test2.subtest2_2'] == 2 or config['test2.subtest2_2'] == 3 or config['test2.subtest2_2'] == 4):
                return False
            return True
        
        def test2(self):
            optimizer = HierarchicalSearch(parameters40, self.funct1)
            allComb = optimizer.getCombinasons()
            self.assertTrue(len(allComb) > 100)
            
            optimizer = HierarchicalSearch(parameters40, self.funct1, validators=[d2vValidator])
            allComb = optimizer.getCombinasons()
            
            self.assertTrue(len(allComb) == 7) # TODO retester ça
            
            

if unittestLevel <= 5: 
    class TestD2VData(unittest.TestCase):
        def funct1(self, d):
            return len(list(d.items()))
            
        def validator(self, config):
            if config['test2'] == 2 or config['test2'] == 4 or config['test2'] == 1:
                return False
            if 'test2.subtest2_2' in config and (config['test2.subtest2_2'] == 1 or config['test2.subtest2_2'] == 2 or config['test2.subtest2_2'] == 3 or config['test2.subtest2_2'] == 4):
                return False
            return True
        
        def test2(self):
            optimizer = HierarchicalSearch(parameters50, self.funct1)
            allComb = optimizer.getCombinasons()
            print(len(allComb))
            self.assertTrue(len(allComb) == 3)
            
            
            

if unittestLevel <= 6: 
    class TestBrown(unittest.TestCase):
        def funct1(self, d):
            return len(list(d.items()))
        
        def test2(self):
            
            regex = "*model*brown_*part0.001*";
            for path in sortedGlob(getWorkingDirectory() + "/" + regex):
                os.remove(path);
            
            optimizer = HierarchicalSearch(parameters60, self.funct1)
            allComb = optimizer.getCombinasons()
            print(len(allComb))
            self.assertTrue(len(allComb) > 20)
            
            fileBrownNumber = len(sortedGlob(getWorkingDirectory() + "/" + "*brown*"))
            for i in range(10):
                brownModel = getBrownDoc2VecModel(allComb[i], r".*201[1-2].*test*", None)
                newFileBrownNumber = len(sortedGlob(getWorkingDirectory() + "/" + "*brown*"))
                print(newFileBrownNumber)
                self.assertTrue(newFileBrownNumber == fileBrownNumber + 1)
                fileBrownNumber = newFileBrownNumber
                
            
        

if unittestLevel <= 7: 
    class TestTop(unittest.TestCase):
        def test1(self):
            d = dict()
            n = 3
            results = [(d, 1.0), (d, 2.0), (d, 0.5), (d, 20.0), (d, 10.0), (d, 30.0), (d, 1.2), (d, 1.1), (d, 1.3)]
            optimizer = HierarchicalSearch(None, None)
            top = optimizer.getTop(results, n)
            self.assertTrue(len(top) == n)
            self.assertTrue(top[0][1] == 30.0)
            self.assertTrue(top[1][1] == 20.0)
            self.assertTrue(top[2][1] == 10.0)
                
            
        

parameters1 = \
[
    {
        'name': 'test1',
        'domain': [0, 1],
        'sorted': True,
        'disabled': False
    },
    {
        'name': 'test2',
        'domain': [100, 101, 102],
        'sorted': True,
        'disabled': False
    },
    {
        'name': 'test3',
        'domain': [10, 11],
        'sorted': True,
        'disabled': False
    }
]

parameters2 = \
[
    {
        'name': 'test2',
        'domain': [1, 2],
        'sorted': True,
        'disabled': False,
        'subparams':
        [
            {
                'constraints': [[2]],
                'name': 'subtest2_2',
                'domain': [25, 26],
                'sorted': True,
                'disabled': False
            }
        ]
    }
]

allParamCombinasons1 = \
[
    [
        {"test1": 1, "test2": 2},
        {"test1": 1, "test2": 3}
    ],
    [
        {"test3": 6, "test4": 7}
    ],
    [
        {"test5": 100, "test6": 101},
        {"test5": 100, "test6": 102}
    ]
]

parameters3 = \
[
    {
        'name': 'test1',
        'domain': [1, 2],
        'sorted': True,
        'disabled': False
    },
    {
        'name': 'test2',
        'domain': [1, 2, 3, 4],
        'sorted': True,
        'disabled': False,
        'subparams':
        [
            {
                'constraints': [[1, 3]],
                'name': 'subtest2_1',
                'domain': [0, 1],
                'sorted': True,
                'disabled': False
            },
            {
                'constraints': [[2]],
                'name': 'subtest2_2',
                'domain': [0, 1, 2, 3, 4],
                'sorted': True,
                'disabled': False
            }
        ]
    },
    {
        'name': 'test3',
        'domain': [10, 11],
        'sorted': True,
        'disabled': False
    }
]


parameters4 = \
[
    {
        'name': 'test1',
        'domain': [1, 2],
        'sorted': True,
        'disabled': True
    },
    {
        'name': 'test2',
        'domain': [1, 2, 3, 4],
        'sorted': True,
        'disabled': False,
        'subparams':
        [
            {
                'constraints': [[1, 3]],
                'name': 'subtest2_1',
                'domain': [0, 1],
                'sorted': True,
                'disabled': True
            },
            {
                'constraints': [[2]],
                'name': 'subtest2_2',
                'domain': [0, 1, 2, 3, 4],
                'sorted': True,
                'disabled': False,
                'force': [555]
            }
        ]
    },
    {
        'name': 'test3',
        'domain': [10, 11],
        'sorted': True,
        'disabled': False
    }
]

parameters5 = \
[
    {
        'name': 'test2',
        'domain': [1, 2, 3, 4],
        'sorted': True,
        'disabled': False,
        'subparams':
        [
            {
                'constraints': [[1, 3]],
                'name': 'yooo',
                'domain': ['a'],
                'sorted': True,
                'disabled': False
            },
            {
                'constraints': [[1, 2]],
                'name': 'yiii',
                'domain': ['b', 'c'],
                'sorted': True,
                'disabled': False
            }
        ]
    }
]

parameters5_1 = \
[
    {
        'name': 'test1',
        'domain': [1, 2],
        'sorted': True,
        'disabled': False
    },
    {
        'name': 'test2',
        'domain': [1, 2, 3, 4],
        'sorted': True,
        'disabled': False,
        'subparams':
        [
            {
                'constraints': [[1, 3]],
                'name': 'yooo',
                'domain': [0, 1],
                'sorted': True,
                'disabled': False
            },
            {
                'constraints': [[1, 2]],
                'name': 'yiii',
                'domain': [0, 1, 2, 3, 4],
                'sorted': True,
                'disabled': False
            }
        ]
    },
    {
        'name': 'test3',
        'domain': [10, 11],
        'sorted': True,
        'disabled': False
    }
]

parameters6 = \
[
    {
        'name': 'test1',
        'domain': [1, 2],
        'sorted': True,
        'disabled': True
    },
    {
        'name': 'test2',
        'domain': [1, 2, 3, 4],
        'sorted': True,
        'disabled': False,
        'subparams':
        [
            {
                'constraints': [[1, 3]],
                'name': 'yooo',
                'domain': [0, 1],
                'sorted': True,
                'disabled': False,
                'subparams':
                [
                    {
                        'constraints': [[0, 1]],
                        'name': 'yooo_bottom',
                        'domain': [0, 1],
                    }
                ]
            },
            {
                'constraints': [[1, 2]],
                'name': 'yiii',
                'domain': [0, 1, 2, 3, 4],
                'sorted': True,
                'disabled': False
            }
        ]
    },
    {
        'name': 'test3',
        'domain': [10, 11],
        'sorted': True,
        'disabled': False
    }
]

parameters10 = \
[
    {
        'name': 'alpha',
        'domain': [0, 1],
        'sorted': True,
    },
    {
        'name': 'test',
        'domain': [10, 11],
        'subparams':
        [
            {
                'name': 'alpha',
                'domain': [100, 101],
                'subparams':
                [
                    {
                        'name': 'alpha',
                        'domain': [1000, 1001],
                    }
                ]
            }
        ]
    }
]

parameters11 = \
[
    {
        'name': 'test',
        'domain': [10, 11, 12],
        'subparams':
        [
            {
                'constraints': [[10]],
                'name': 'alpha',
                'domain': [100, 101],
                'subparams':
                [
                    {
                        'name': 'alpha',
                        'domain': ['a']
                    }
                ]
            },
            {
                'constraints': [[11]],
                'name': 'alpha',
                'domain': [1000]
            }
        ]
    }
]


parameters20 = \
[
    {
        'name': 'remove_stopwords',
        'domain': [True, False],
        'sorted': False,
        'disabled': False,
        'subparams':
        [
            {
                'constraints': [[True]],
                'name': 'lists',
                'domain': [1, 2, 3, 4, 5],
                'sorted': False,
                'disabled': False
            }
        ]
    },
    {
        'name': 'regresser',
        'domain': ['Ridge', 'Lasso'],
        'sorted': False,
        'disabled': False,
        'force': ['Ridge'],
        'subparams':
        [
            {
                'constraints': [['Ridge']],
                'name': 'alpha',
                'domain': [0.0, 0.3, 0.6, 0.9, 1.0],
                'sorted': True,
                'disabled': False
            },
            {
                'constraints': [['Lasso']],
                'name': 'alpha',
                'domain': np.arange(0.01, 3, 0.01),
                'sorted': True,
                'disabled': False
            },
            {
                'constraints': [['Lasso']],
                'name': 'normalise',
                'domain': [True, False],
                'sorted': False,
                'disabled': False
            }
        ]
    },
    {
        'name': 'doc2vec',
        'domain': [True, False],
        'sorted': False,
        'disabled': False,
        'subparams':
        [
            {
                'constraints': [[True]],
                'name': 'min_count',
                'domain': list(range(20)),
                'sorted': True,
                'disabled': False
            },
            {
                'constraints': [[True]],
                'name': 'window',
                'domain': list(range(2, 15)),
                'sorted': True,
                'disabled': False
            },
            {
                'constraints': [[True]],
                'name': 'size',
                'domain': [20, 100, 500, 1000],
                'sorted': True,
                'disabled': False
            },
            {
                'constraints': [[True]],
                'name': 'data',
                'domain': ['ukwac', 'enwiki'],
                'sorted': False,
                'disabled': False
            }
        ]
    }
]


parameters30 = \
[
    {
        'name': 'Word2VecFeature',
        'domain': [True, False],
        'force': [True],
        'subparams':
        [
            {
                'constraints': [[True]],
                'name': 'w2vNSimilarity',
                'domain': [True, False],
                'force': [True],
                'subparams':
                [
                    {
                        'constraints': [[True]],
                        'name': 'defaultSimilarity',
                        'domain': [0.0, 0.2, 0.5, 0.8, 1.0],
                        'force': [0.8]
                    }
                ]
            }
        ]
    }
]

parameters31 = \
[
    {
        'name': 'alpha',
        'domain': [0, 1],
        'sorted': True,
        'disabled': True
    },
    {
        'name': 'Word2VecFeature',
        'domain': [True, False],
        'force': [True],
        'subparams':
        [
            {
                'constraints': [[True]],
                'name': 'w2vNSimilarity',
                'domain': [True, False],
                'force': [True],
                'subparams':
                [
                    {
                        'constraints': [[True]],
                        'name': 'defaultSimilarity',
                        'domain': [0.0, 0.2, 0.5, 0.8, 1.0],
                        'force': [0.8]
                    }
                ]
            }
        ]
    },
    {
        'name': 'beta',
        'domain': [0, 1],
        'sorted': True,
        'disabled': True
    },
    {
        'name': 'omega',
        'domain': [0, 1],
        'sorted': True,
        'disabled': True
    },
    {
        'name': 'alpha2',
        'domain': [0, 1],
        'sorted': True,
        'disabled': True
    },
]


parameters40 = \
[
    {
        'name': 'score',
        'domain': ['MeanLeastSquares', 'MeanDifference', 'PearsonCorrelation'],
        'force': ['PearsonCorrelation']
    },
    {
        'name': 'regresser',
        'domain': ['Ridge', 'Lasso', 'ElasticNet', 'Linear'],
        'force': ['Ridge'],
        'subparams':
        [
            {
                'constraints': [['Ridge']],
                'name': 'alpha',
                'domain': np.arange(0.1, 3.0, 0.1),
                'force': [1.0]
            }
        ]
    },
    {
        'name': 'data',
        'domain': ['Normal2015', 'Normal2016', 'CrossValidation2015', 'CrossValidation2016', 'CrossValidation2017'],
        'disabled': False,
        'force': ['CrossValidation2017'],
        'subparams':
        [
            {
                'constraints': [['CrossValidation2015', 'CrossValidation2016']],
                'name': 'partsCount',
                'domain': [5, 10, 20],
                'force': [10]
            }
        ]
    },
    {
        'name': 'agParser',
        'domain': [True],
        'subparams':
        [
            {
                'name': 'removeStopWords',
                'domain': [True, False],
                'force': [True]
            },
            {
                'name': 'removePunct',
                'domain': [True, False],
                'force': [True]
            },
            {
                'name': 'toLowerCase',
                'domain': [True, False],
                'force': [True]
            },
            {
                'name': 'lemma',
                'domain': [True, False],
                'force': [False]
            }
        ]
    },
    {
        'name': 'LengthFeature',
        'domain': [True, False],
        'disabled': False,
        'force': [True],
        'subparams':
        [
            {
                'name': 'string',
                'domain': [True, False],
                'force': [True]
            },
            {
                'name': 'tokens',
                'domain': [True, False],
                'force': [True]
            }
        ]
    },
    {
        'name': 'SultanAlignerFeature',
        'domain': [True, False],
        'disabled': True,
        'force': [True],
        'subparams':
        [
            {
                'constraints': [[True]],
                'name': 'similarity1',
                'domain': [True, False],
                'force': [True],
            },
            {
                'constraints': [[True]],
                'name': 'similarity2',
                'domain': [True, False],
                'force': [True],
            }
        ]
    },
    {
        'name': 'Word2VecFeature',
        'domain': [True, False],
        'disabled': True,
        'force': [True],
        'subparams':
        [
            {
                'constraints': [[True]],
                'name': 'homeMadeSimilarity',
                'domain': [True, False],
                'force': [True],
            },
            {
                'constraints': [[True]],
                'name': 'w2vNSimilarity',
                'domain': [True, False],
                'force': [True],
                'subparams':
                [
                    {
                        'constraints': [[True]],
                        'name': 'defaultSimilarity',
                        'domain': [0.0, 0.2, 0.5, 0.8, 1.0],
                        'force': [0.8]
                    }
                ]
            },
            {
                'constraints': [[True]],
                'name': 'vector',
                'domain': [True, False],
                'force': [False],
            },
            { 
                'constraints': [[True]],
                'name': 'data',
                'domain': ['STSAll', 'BaroniVectors', 'GoogleNews', 'STSTrain'],
                # 'force': ['STSAll', 'STSTrain']
            }
        ]
    },
    {
        'name': 'RandomFeature',
        'domain': [True, False],
        'disabled': True,
        'force': [False]
    },
    {
        'name': 'Doc2VecFeature',
        'domain': [True, False],
        'force': [True],
        'disabled': False,
        'subparams':
        [
            {
                'constraints': [[True]],
                'name': 'data',
                'domain': ['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac', 'brown', 'stsall'],
                # 'force': ['brown', 'stsall'],
                # 'force': ['ukwac'],
                'subparams':
                [
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac']],
                        'name': 'min_count',
                        'sorted': True,
                        'domain': [5, 7, 15, 20],
                        # 'force': [15]
                    },
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac']],
                        'name': 'window',
                        'sorted': True,
                        'domain': [5, 8, 15, 20],
                        # 'force': [8]
                    },
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac']],
                        'name': 'size',
                        'sorted': True,
                        'domain': [100, 300, 500, 1000, 3000],
                        # 'force': [500]
                    },
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac']],
                        'name': 'removeStopWords',
                        'domain': [True, False],
                        # 'force': [False]
                    },
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac']],
                        'name': 'removePunct',
                        'domain': [True, False],
                        # 'force': [True]
                    },
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac']],
                        'name': 'toLowerCase',
                        'domain': [True, False],
                        # 'force': [True]
                    },
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac']],
                        'name': 'lemma',
                        'domain': [True, False],
                        # 'force': [False]
                    },
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac']],
                        'name': 'dataPart',
                        'sorted': True,
                        'domain': [0.1, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
                        # 'force': [0.6]
                    },
                ]
            },
            {
                'constraints': [[True]],
                'name': 'd2vSimilarity',
                'domain': [True, False],
                'force': [True],
                'subparams':
                [
                    {
                        'constraints': [[True]],
                        'name': 'defaultSimilarity',
                        'domain': [0.0, 0.2, 0.5, 0.8, 1.0],
                        # 'force': [0.0, 0.5, 1.0],
                        'force': [0.0],
                        
                    }
                ]
            },
            {
                'constraints': [[True]],
                'name': 'vector',
                'domain': [True, False],
                'force': [False],
            }
        ]
    }
]



parameters50 = \
[
    {
        'name': 'Doc2VecFeature',
        'domain': [True, False],
        'force': [True],
        'disabled': False,
        'subparams':
        [
            {
                'constraints': [[True]],
                'name': 'data',
                'domain': ['ukwac', 'brown'],
                'subparams':
                [
                    {
                        'constraints': [['ukwac']],
                        'name': 'min_count',
                        'domain': [5, 7],
                    }
                ]
            }
        ]
    }
]



parameters60 = \
[
    {
        'name': 'score',
        'domain': ['MeanLeastSquares', 'MeanDifference', 'ScipyPearsonCorrelation', 'NumpyPearsonCorrelation', 'AgirrePearsonCorrelation'],
        'force': ['NumpyPearsonCorrelation']
    },
    {
        'name': 'regressor',
        'domain': ['Ridge', 'Lasso', 'ElasticNet', 'Linear'],
        'force': ['Ridge'],
        'subparams':
        [
            {
                'constraints': [['Ridge']],
                'name': 'alpha',
                'domain': np.arange(0.1, 3.0, 0.1),
                'force': [1.0]
            }
        ]
    },
    {
        'name': 'data',
        'domain': ['Normal2015', 'Normal2016', 'CrossValidation2015', 'CrossValidation2016', 'CrossValidation2017'],
        'disabled': False,
        'force': ['CrossValidation2017'],
        'subparams':
        [
            {
                'constraints': [['CrossValidation2015', 'CrossValidation2016', 'CrossValidation2017']],
                'name': 'partsCount',
                'domain': [5, 10, 20],
                'force': [10]
            }
        ]
    },
    {
        'name': 'agParser',
        'domain': [True],
        'subparams':
        [
            {
                'name': 'removeStopWords',
                'domain': [True, False],
                'force': [True]
            },
            {
                'name': 'removePunct',
                'domain': [True, False],
                'force': [True]
            },
            {
                'name': 'toLowerCase',
                'domain': [True, False],
                'force': [True]
            },
            {
                'name': 'lemma',
                'domain': [True, False],
                'force': [False]
            }
        ]
    },
    {
        'name': 'LengthFeature',
        'domain': [True, False],
        'disabled': False,
        'force': [True],
        'subparams':
        [
            {
                'name': 'string',
                'domain': [True, False],
                'force': [True]
            },
            {
                'name': 'tokens',
                'domain': [True, False],
                'force': [True]
            }
        ]
    },
    {
        'name': 'SultanAlignerFeature',
        'domain': [True, False],
        'disabled': True,
        'force': [True],
        'subparams':
        [
            {
                'constraints': [[True]],
                'name': 'similarity1',
                'domain': [True, False],
                'force': [True],
            },
            {
                'constraints': [[True]],
                'name': 'similarity2',
                'domain': [True, False],
                'force': [True],
            }
        ]
    },
    {
        'name': 'Word2VecFeature',
        'domain': [True, False],
        'disabled': True,
        'force': [True],
        'subparams':
        [
            {
                'constraints': [[True]],
                'name': 'homeMadeSimilarity',
                'domain': [True, False],
                'force': [True],
            },
            {
                'constraints': [[True]],
                'name': 'w2vNSimilarity',
                'domain': [True, False],
                'force': [True],
                'subparams':
                [
                    {
                        'constraints': [[True]],
                        'name': 'defaultSimilarity',
                        'domain': [0.0, 0.2, 0.5, 0.8, 1.0],
                        'force': [0.8]
                    }
                ]
            },
            {
                'constraints': [[True]],
                'name': 'vector',
                'domain': [True, False],
                'force': [False],
            },
            { 
                'constraints': [[True]],
                'name': 'data',
                'domain': ['brown_stsall', 'brown', 'stsall', 'ststrain', 'BaroniVectors', 'GoogleNews'],
                'force': ['GoogleNews']
            }
        ]
    },
    {
        'name': 'RandomFeature',
        'domain': [True, False],
        'disabled': True,
        'force': [False]
    },
    {
        'name': 'Doc2VecFeature',
        'domain': [True, False],
        'force': [True],
        'disabled': False,
        'subparams':
        [
            {
                'constraints': [[True]],
                'name': 'data',
                'domain': ['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac', 'brown', 'stsall'],
                # 'force': ['brown', 'stsall'],
                # 'force': ['ukwac'],
                'force': ['brown'],
                'subparams':
                [
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac', 'brown']],
                        'name': 'min_count',
                        'sorted': True,
                        'domain': [5, 7, 15, 20],
                        'force': [5, 15]
                    },
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac', 'brown']],
                        'name': 'window',
                        'sorted': True,
                        'domain': [5, 8, 15, 20],
                        'force': [5, 8]
                    },
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac', 'brown']],
                        'name': 'size',
                        'sorted': True,
                        'domain': [100, 300, 500, 1000, 3000],
                        'force': [100, 500, 3000]
                    },
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac', 'brown']],
                        'name': 'removeStopWords',
                        'domain': [True, False],
                        # 'force': [False]
                    },
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac', 'brown']],
                        'name': 'removePunct',
                        'domain': [True, False],
                        # 'force': [True]
                    },
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac', 'brown']],
                        'name': 'toLowerCase',
                        'domain': [True, False],
                        # 'force': [True]
                    },
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac', 'brown']],
                        'name': 'lemma',
                        'domain': [True, False],
                        # 'force': [False]
                    },
                    {
                        'constraints': [['ukwac', 'enwiki', 'ukwac_enwiki', 'halfukwac', 'brown']],
                        'name': 'dataPart',
                        'sorted': True,
                        'domain': [0.1, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
                        'force': [0.001]
                    },
                ]
            },
            {
                'constraints': [[True]],
                'name': 'd2vSimilarity',
                'domain': [True, False],
                'force': [True],
                'subparams':
                [
                    {
                        'constraints': [[True]],
                        'name': 'defaultSimilarity',
                        'domain': [0.0, 0.2, 0.5, 0.8, 1.0],
                        # 'force': [0.0, 0.5, 1.0],
                        'force': [0.0],
                        
                    }
                ]
            },
            {
                'constraints': [[True]],
                'name': 'vector',
                'domain': [True, False],
                'force': [False],
            }
        ]
    }
]


