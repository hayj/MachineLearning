# coding: utf-8

import os
import sys
sys.path.append('../')

import unittest
import doctest
from machinelearning import kerasutils
from machinelearning.kerasutils import *

# The level allow the unit test execution to choose only the top level test
mini = 0
maxi = 9
assert mini <= maxi

print("==============\nStarting unit tests...")

if mini <= 0 <= maxi:
    class DocTest(unittest.TestCase):
        def testDoctests(self):
            """Run doctests"""
            doctest.testmod(kerasutils)

if mini <= 1 <= maxi:
    class Test1(unittest.TestCase):
        def test1(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_loss': {'patience': 3, 'min_delta': 0.1},
                'val_acc': {'patience': 2},
                'val_top_k_categorical_accuracy': {'patience': 2, 'min_delta': 0.1},
            })
            history = \
            {
                'val_loss': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                'val_acc': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                'val_top_k_categorical_accuracy': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            }
            self.assertTrue(not hasToEarlyStop(history, esm))
        def test2(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_loss': {'patience': 3},
                'val_acc': {'patience': 2},
                'val_top_k_categorical_accuracy': {'patience': 2},
            })
            history = \
            {
                'val_loss': [0.1, 0.1, 0.1, 0.1, 0.1, 0.09],
                'val_acc': [0.1, 0.1, 0.09, 0.08, 0.07, 0.06],
                'val_top_k_categorical_accuracy': [0.1, 0.1, 0.1, 0.09, 0.08, 0.07],
            }
            self.assertTrue(not hasToEarlyStop(history, esm))
        def test3(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_loss': {'patience': 3, 'min_delta': 0.01},
                'val_acc': {'patience': 2, 'min_delta': 0.01},
                'val_top_k_categorical_accuracy': {'patience': 2, 'min_delta': 0.01},
            })
            history = \
            {
                'val_loss': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                'val_acc': [0.1, 0.1, 0.09, 0.08, 0.07, 0.06],
                'val_top_k_categorical_accuracy': [0.1, 0.1, 0.1, 0.09, 0.08, 0.07],
            }
            self.assertTrue(hasToEarlyStop(history, esm))
        def test3(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_acc': {'patience': 2},
            })
            history = \
            {
                'val_acc': [0.12, 0.13, 0.07, 0.06, 0.05],
            }
            self.assertTrue(hasToEarlyStop(history, esm))
        def test4(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_acc': {'patience': 3},
            })
            history = \
            {
                'val_acc': [0.1, 0.1, 0.12, 0.13, 0.07, 0.06, 0.05],
            }
            self.assertTrue(not hasToEarlyStop(history, esm))
        def test5(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_acc': {'patience': 1},
            })
            history = \
            {
                'val_acc': [0.1, 0.1, 0.12, 0.13, 0.07],
            }
            self.assertTrue(not hasToEarlyStop(history, esm))
        def test5(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_acc': {'patience': 0},
            })
            history = \
            {
                'val_acc': [0.1, 0.1, 0.12, 0.13, 0.07],
            }
            self.assertTrue(hasToEarlyStop(history, esm))
        def test7(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_acc': {'patience': 4},
            })
            history = \
            {
                'val_acc': [0.1, 0.14, 0.10, 0.11, 0.12, 0.13],
            }
            self.assertTrue(not hasToEarlyStop(history, esm))
        def test7(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_loss': {'patience': 1, 'min_delta': 0.5},
                'val_acc': {'patience': 3},
            })
            history = \
            {
                'val_loss': [10, 20, 9, 8, 8, 6, 5, 6],
                'val_acc': [0.1, 0.14, 0.10, 0.11, 0.12, 0.13],
            }
            self.assertTrue(not hasToEarlyStop(history, esm))
        def test8(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_loss': {'patience': 1, 'min_delta': 0.5},
                'val_acc': {'patience': 3},
            })
            history = \
            {
                'val_loss': [10, 20, 9, 8, 8, 6, 5, 6, 7],
                'val_acc': [0.1, 0.14, 0.10, 0.11, 0.12, 0.13],
            }
            self.assertTrue(hasToEarlyStop(history, esm))
        def test9(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_loss': {'patience': 3},
                'val_acc': {'patience': 3},
                'val_top_k_categorical_accuracy': {'patience': 3},
            })
            history = \
            {
                'val_loss': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                'val_acc': [0.1, 0.1, 0.09, 0.08, 0.07, 0.1, 0.1],
                'val_top_k_categorical_accuracy': [0.1, 0.1, 0.1, 0.09, 0.08, 0.1],
            }
            self.assertTrue(not hasToEarlyStop(history, esm))
        def test10(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_loss': {'patience': 3},
                'val_acc': {'patience': 3},
                'val_top_k_categorical_accuracy': {'patience': 3},
            })
            history = \
            {
                'val_loss': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                'val_acc': [0.1, 0.1, 0.09, 0.1, 0.07, 0.09],
                'val_top_k_categorical_accuracy': [0.1, 0.1, 0.1, 0.09, 0.08, 0.1],
            }
            self.assertTrue(not hasToEarlyStop(history, esm))
        def test11(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_loss': {'patience': 3},
                'val_acc': {'patience': 3},
                'val_top_k_categorical_accuracy': {'patience': 3},
            })
            history = \
            {
                'val_loss': [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3],
                'val_acc': [0.1, 0.1, 0.09, 0.1, 0.07, 0.12, 0.13],
                'val_top_k_categorical_accuracy': [0.1, 0.1, 0.1, 0.09, 0.08, 0.1, 0.12, 0.13],
            }
            self.assertTrue(not hasToEarlyStop(history, esm))
        def test12(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_loss': {'patience': 3},
                'val_acc': {'patience': 3},
                'val_top_k_categorical_accuracy': {'patience': 3},
            })
            history = \
            {
                'val_loss': [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3],
                'val_acc': [0.1, 0.1, 0.09, 0.1, 0.07, 0.07, 0.09, 0.09],
                'val_top_k_categorical_accuracy': [0.1, 0.1, 0.1, 0.09, 0.08, 0.1, 0.12, 0.13],
            }
            self.assertTrue(not hasToEarlyStop(history, esm))
        def test13(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_loss': {'patience': 2},
                'val_acc': {'patience': 3},
                'val_top_k_categorical_accuracy': {'patience': 3},
            })
            history = \
            {
                'val_loss': [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4],
                'val_acc': [0.1, 0.1, 0.09, 0.1, 0.07, 0.07, 0.09, 0.09],
                'val_top_k_categorical_accuracy': [0.1, 0.1, 0.1, 0.09, 0.08, 0.099, 0.095, 0.06],
            }
            self.assertTrue(hasToEarlyStop(history, esm))
        def test14(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_top_k_categorical_accuracy': {'patience': 1, 'min_delta': 0.03},
            })
            history = \
            {
                'val_top_k_categorical_accuracy': [0.13, 0.1, 0.1, 0.09, 0.08, 0.099, 0.095, 0.11, 0.12],
            }
            self.assertTrue(hasToEarlyStop(history, esm))
        def test15(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_top_k_categorical_accuracy': {'patience': 1, 'min_delta': 0.03},
            })
            history = \
            {
                'val_top_k_categorical_accuracy': [0.13, 0.1, 0.1, 0.09, 0.08, 0.099, 0.095, 0.15, 0.12],
            }
            self.assertTrue(hasToEarlyStop(history, esm))
        def test16(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_top_k_categorical_accuracy': {'patience': 1, 'min_delta': 0.019},
            })
            history = \
            {
                'val_top_k_categorical_accuracy': [0.13, 0.1, 0.1, 0.09, 0.08, 0.099, 0.095, 0.15, 0.12],
            }
            self.assertTrue(not hasToEarlyStop(history, esm))
        def test17(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_top_k_categorical_accuracy': {'patience': 2, 'min_delta': 0.1},
            })
            history = \
            {
                'val_top_k_categorical_accuracy': [0.1, 0.2, 0.3, 0.4],
            }
            self.assertTrue(not hasToEarlyStop(history, esm))
        def test18(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_top_k_categorical_accuracy': {'patience': 2, 'min_delta': 0.2},
            })
            history = \
            {
                'val_top_k_categorical_accuracy': [0.1, 0.2, 0.3, 0.4],
            }
            self.assertTrue(hasToEarlyStop(history, esm))
        def test19(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_top_k_categorical_accuracy': {'patience': 2, 'min_delta': 0.1},
            })
            history = \
            {
                'val_top_k_categorical_accuracy': [0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4],
            }
            self.assertTrue(hasToEarlyStop(history, esm))
        def test20(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_acc': {'patience': 3},
                'val_top_k_categorical_accuracy': {'patience': 3},
            })
            history = \
            {
                'val_acc': [0.1, 0.2, 0.3, 0.4, 0.3, 0.4, 0.3, 0.3],
                'val_top_k_categorical_accuracy': [0.1, 0.2, 0.3, 0.4, 0.3, 0.4, 0.3, 0.3],
            }
            self.assertTrue(not hasToEarlyStop(history, esm))
        def test20(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_acc': {'patience': 3},
                'val_top_k_categorical_accuracy': {'patience': 3},
            })
            history = \
            {
                'val_acc': [0.1, 0.2, 0.3, 0.4, 0.3, 0.3, 0.3, 0.3],
                'val_top_k_categorical_accuracy': [0.1, 0.2, 0.3, 0.4, 0.3, 0.4, 0.3, 0.3],
            }
            self.assertTrue(not hasToEarlyStop(history, esm))
        def test20(self):
            esm = normalizeEarlyStopMonitor(\
            {
                'val_acc': {'patience': 3},
                'val_top_k_categorical_accuracy': {'patience': 3},
            })
            history = \
            {
                'val_acc': [0.1, 0.2, 0.3, 0.4, 0.3, 0.3, 0.3, 0.3],
                'val_top_k_categorical_accuracy': [0.1, 0.2, 0.3, 0.4, 0.3, 0.3, 0.3, 0.3],
            }
            self.assertTrue(hasToEarlyStop(history, esm))

if mini <= 2 <= maxi:
    class Test2(unittest.TestCase):
        def test1(self):
            normalizeEarlyStopMonitor\
            (
                {
                    'val_loss': {'patience': 50, 'min_delta': 0.5555, 'mode': 'auto'},
                    'val_acc': {'patience': 50, 'mode': 'auto'},
                    'val_top_k_categorical_accuracy': {'patience': 50, 'min_delta': 0, 'mode': 'auto'},
                },
            )

if mini <= 12 <= maxi:
    class Test2(unittest.TestCase):
        def test1(self):
            x1 = xVal
            y1 = yVal
            x2 = iteratorToArray(asap.getTokensOnlyValidationInfiniteBatcher(), steps=asap.getValidationBatchsCount())
            y2 = iteratorToArray(asap.getLabelOnlyValidationInfiniteBatcher(), steps=asap.getValidationBatchsCount())
            for i in range(len(x1)):
                if i % 100 == 0:
                    print("--------a")
                    print(x1[i])
                    print(x2[i])
                    print("--------b")
                    print(y1[i])
                    print(y2[i])
                self.assertTrue(np.array_equal(x1[i], x2[i]))
                self.assertTrue(np.array_equal(y1[i], y2[i]))
                self.assertTrue(x1[i][1] == x2[i][1])
                self.assertTrue(y1[i][1] == y2[i][1])
            self.assertTrue(np.array_equal(x1, x2))
            self.assertTrue(np.array_equal(y1, y2))
            self.assertTrue(not np.array_equal(x1[2], x2[4]))
            self.assertTrue(not np.array_equal(y1[2], y2[4]))


if __name__ == '__main__':
    unittest.main() # Orb executes it as a Python unit-test in eclipse


print("Unit tests done.\n==============")