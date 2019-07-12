# coding: utf-8
# pew in systemtools-venv python ./test/basics.py

import os
import sys
sys.path.append('../')

import unittest
import doctest
from systemtools.basics import *
from machinelearning import iterator
from machinelearning.iterator import *
import numpy as np

# The level allow the unit test execution to choose only the top level test
mini = 0
maxi = 9
assert mini <= maxi

print("==============\nStarting unit tests...")

if mini <= 0 <= maxi:
    class DocTest(unittest.TestCase):
        def testDoctests(self):
            """Run doctests"""
            doctest.testmod(iterator)

def itemGen2():
    rdInt = getRandomInt(0, 10000000)
    for i in range(5):
        yield rdInt

def itemGen3():
    for i in range(3):
        rdInt = getRandomInt(0, 10000000)
        for u in range(3):
            yield rdInt




def itemGen1():
    for i in range(5):
        yield getRandomStr()


if mini <= 1 <= maxi:
    class Test1(unittest.TestCase):
        def test1(self):
            aaa = AgainAndAgain(itemGen1)
            infAAA = InfiniteBatcher(aaa, 3)
            data = []
            for i in range(10):
                data.append(next(infAAA))
            self.assertTrue(isinstance(data[0][0], str))
            self.assertTrue(len(data[0]) == 3)
            self.assertTrue(len(data[-1]) == 2)
        def test2(self):
            random.seed(0)
            aaa = AgainAndAgain(itemGen2)
            infAAA = InfiniteBatcher(aaa, 3, shuffle=0, toNumpyArray=False)
            data = []
            for i in range(10):
                data.append(next(infAAA))
            self.assertTrue(len(set(data[0])) == 1)
            self.assertTrue(len(data[0]) == 3)
            self.assertTrue(len(data[-1]) == 2)
            self.assertTrue(not isinstance(data[0], np.ndarray))
        def test3(self):
            random.seed(0)
            aaa = AgainAndAgain(itemGen2)
            infAAA = InfiniteBatcher(aaa, 3, shuffle=100, toNumpyArray=True)
            data = []
            for iteratori in range(10):
                data.append(next(infAAA))
            self.assertTrue(len(set(data[0])) == 1)
            self.assertTrue(len(data[0]) == 3)
            self.assertTrue(len(data[-1]) == 2)
            self.assertTrue(isinstance(data[0], np.ndarray))
        def test4(self):
            random.seed(0)
            aaa = AgainAndAgain(itemGen3)
            infAAA = InfiniteBatcher(aaa, 3, shuffle=100, toNumpyArray=False)
            data = []
            for i in range(10):
                data.append(next(infAAA))
            self.assertTrue(len(set(data[0])) > 1)
            self.assertTrue(len(data[0]) == 3)
            self.assertTrue(len(data[-1]) == 3)
            self.assertTrue(isinstance(data[0], list))

        def test5(self):
            def itemGen():
                for i in range(5):
                    yield (getRandomStr(), getRandomStr(), getRandomStr())
            random.seed(0)
            aaa = AgainAndAgain(itemGen)
            infAAA = InfiniteBatcher(aaa, 3, shuffle=0, toNumpyArray=True)
            data = []
            for i in range(10):
                current = next(infAAA)
                data.append(current)
                self.assertTrue(len(current) == 3)
                self.assertTrue(len(current[0]) == 3 or len(current[0]) == 2)
                self.assertTrue(len(current[0]) == len(current[2]))
            self.assertTrue(len(set(data[0][0])) == 3)
            self.assertTrue(len(data) == 10)
            self.assertTrue(len(data[0]) == 3)
            self.assertTrue(isinstance(data[0][0][0], str))
            self.assertTrue(isinstance(data[0][0], np.ndarray))
            self.assertTrue(isinstance(data[0], tuple))
            self.assertTrue(isinstance(data, list))
        def test6(self):
            def itemGen():
                for i in range(5):
                    yield (getRandomStr(), getRandomStr(), getRandomStr())
            random.seed(0)
            aaa = AgainAndAgain(itemGen)
            infAAA = InfiniteBatcher(aaa, 3, shuffle=0, toNumpyArray=True)
            count = 0
            for i in range(1000):
                count += 1
                current = next(infAAA)
            self.assertTrue(count >= 1000)
        def test7(self):
            def itemGen():
                for i in range(5):
                    yield (i, i + 1, i + 2)
            random.seed(0)
            aaa = AgainAndAgain(itemGen)
            infAAA = InfiniteBatcher(aaa, 3, queueSize=4, shuffle=0, toNumpyArray=True, skip=1)
            data = []
            for i in range(100):
                current = next(infAAA)
                data.append(current)
            self.assertTrue(data[0][1][0] == 4)
            self.assertTrue(data[1][1][0] == 1)
            self.assertTrue(data[1][2][0] == 2)
        def test8(self):
            def itemGen():
                for i in range(15):
                    yield ("a", "b", "c", "d", "e")
            random.seed(0)
            aaa = AgainAndAgain(itemGen)
            infAAA = InfiniteBatcher(aaa, 3, shuffle=0, toNumpyArray=False)
            count = 0
            data = []
            for i in range(1000):
                count += 1
                current = next(infAAA)
                data.append(current)
            self.assertTrue(count >= 1000)
            self.assertTrue(data[0][0] == ["a", "a", "a"])
            self.assertTrue(data[0][1] == ["b", "b", "b"])
            self.assertTrue(data[0][4] == ["e", "e", "e"])
            self.assertTrue(data[100][0] == ["a", "a", "a"])
            self.assertTrue(data[100][1] == ["b", "b", "b"])
            self.assertTrue(data[100][4] == ["e", "e", "e"])
        def test9(self):
            def itemGen():
                for i in range(15):
                    yield ("a", "b", "c", "d", "e")
            random.seed(0)
            aaa = AgainAndAgain(itemGen)
            infAAA = InfiniteBatcher(aaa, 3, shuffle=4, toNumpyArray=False)
            count = 0
            data = []
            for i in range(1000):
                count += 1
                current = next(infAAA)
                data.append(current)
            self.assertTrue(count >= 1000)
            self.assertTrue(data[0][1] == ["b", "b", "b"])
            self.assertTrue(data[0][0] == ["a", "a", "a"])
            self.assertTrue(data[0][4] == ["e", "e", "e"])
            self.assertTrue(data[100][0] == ["a", "a", "a"])
            self.assertTrue(data[100][1] == ["b", "b", "b"])
            self.assertTrue(data[100][4] == ["e", "e", "e"])
        def test10(self):
            def itemGen():
                for i in range(15):
                    yield ("a", "b", "c", "d", "e")
            random.seed(0)
            aaa = AgainAndAgain(itemGen)
            infAAA = InfiniteBatcher(aaa, 3, shuffle=4, toNumpyArray=True)
            count = 0
            data = []
            for i in range(1000):
                count += 1
                current = next(infAAA)
                data.append(current)
            self.assertTrue(count >= 1000)
            self.assertTrue(np.array_equal(data[0][1], np.array(["b", "b", "b"])))
                



if __name__ == '__main__':
    unittest.main() # Orb execute as Python unit-test in eclipse


print("Unit tests done.\n==============")