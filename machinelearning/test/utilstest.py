# coding: utf-8
# pew in st-venv python /home/hayj/Workspace/Python/Utils/NLPTools/nlptools/test/tokenizertest2.py

import os
import sys
sys.path.append('../')

import unittest
import doctest
from machinelearning import utils
from machinelearning.utils import *
from systemtools.location import *

# The level allow the unit test execution to choose only the top level test
mini = 0
maxi = 12
assert mini <= maxi

print("==============\nStarting unit tests...")


if mini <= 0 <= maxi:
	class DocTest(unittest.TestCase):
		def testDoctests(self):
			"""Run doctests"""
			doctest.testmod(utils)

if mini <= 1 <= maxi:
	class Test1(unittest.TestCase):
		def test1(self):
			self.assertTrue(padSequence([], 5) == [MASK_TOKEN] * 5)


		def test2(self):
			mask = MASK_TOKEN
			maxlen = 5
			data = [[], ["a", "b"], ["a", "b", "c", "d", "e", "f"]]
			result = padSequences(data, maxlen, value=mask, padding='pre', truncating='post')
			self.assertTrue(result[0] == [mask] * maxlen)
			self.assertTrue(result[1] == [mask] * 3 + ["a", "b"])
			self.assertTrue(result[2] == ["a", "b", "c", "d", "e"])

		def test3(self):
			mask = MASK_TOKEN
			maxlen = 5
			data = [[], ["a", "b"], ["a", "b", "c", "d", "e", "f"]]
			result = padSequences(data, maxlen, value=mask, padding='post', truncating='post')
			self.assertTrue(result[0] == [mask] * maxlen)
			self.assertTrue(result[1] == ["a", "b"] + [mask] * 3)
			self.assertTrue(result[2] == ["a", "b", "c", "d", "e"])

		def test4(self):
			mask = MASK_TOKEN
			maxlen = 5
			data = [[], ["a", "b"], ["a", "b", "c", "d", "e", "f"]]
			result = padSequences(data, maxlen, value=mask, padding='post', truncating='pre')
			self.assertTrue(result[0] == [mask] * maxlen)
			self.assertTrue(result[1] == ["a", "b"] + [mask] * 3)
			self.assertTrue(result[2] == ["b", "c", "d", "e", "f"])

		def test5(self):
			self.assertTrue(padSequence(["a", "b"], 1) == ["a"])
			
		def test6(self):
			self.assertTrue(padSequence(["a"], 1) == ["a"])

		def test7(self):
			self.assertTrue(padSequence(["a", "b"], 1, truncating='pre') == ["b"])

		def test8(self):
			mask = MASK_TOKEN
			maxlen = 10
			data = [[], ["a", "b"], ["a", "b", "c", "d", "e", "f"]]
			result = padSequence(data, maxlen, value=mask, padding='pre', truncating='pre')
			self.assertTrue(result == [[MASK_TOKEN, MASK_TOKEN], ["a", "b"], ["a", "b", "c", "d", "e", "f"]])
			result = padSequence(data, maxlen, value=mask, padding='post', truncating='pre')
			self.assertTrue(result == [["a", "b"], ["a", "b", "c", "d", "e", "f"], [MASK_TOKEN, MASK_TOKEN]])

		def test9(self):
			mask = MASK_TOKEN
			maxlen = 5
			data = [[], ["a", "b"], ["a", "b", "c", "d", "e", "f"]]
			result = padSequence(data, maxlen, value=mask, padding='pre', truncating='pre')
			self.assertTrue(result == [["b", "c", "d", "e", "f"]])
			result = padSequence(data, maxlen, value=mask, padding='post', truncating='post')
			self.assertTrue(result == [["a", "b"], ["a", "b", "c"]])

		def test10(self):
			mask = MASK_TOKEN
			maxlen = 8
			data = [[], ["a", "b"], ["a", "b", "c", "d", "e", "f"]]
			result = padSequence(data, maxlen, value=mask, padding='pre', truncating='pre')
			self.assertTrue(result == [["a", "b"], ["a", "b", "c", "d", "e", "f"]])

		def test11(self):
			mask = MASK_TOKEN
			maxlen = 5
			data = [["a"]]
			result = padSequence(data, maxlen, value=mask, padding='post', truncating='pre')
			self.assertTrue(result == [["a"], [mask] * 4])

		def test12(self):
			mask = MASK_TOKEN
			maxlen = 5
			data = [["a"]]
			result = padSequence(data, maxlen, value=mask, padding='pre', truncating='pre')
			self.assertTrue(result == [[mask] * 4, ["a"]])

		def test13(self):
			mask = MASK_TOKEN
			maxlen = 5
			data = [[]]
			result = padSequence(data, maxlen, value=mask, padding='post', truncating='pre')
			self.assertTrue(result == [[mask] * 5])
			result = padSequence(data, maxlen, value=mask, padding='pre', truncating='pre')
			self.assertTrue(result == [[mask] * 5])

		def test14(self):
			mask = MASK_TOKEN
			maxlen = 6
			data = [["a"], ["a", "b", "c", "d"]]
			result = padSequence(data, maxlen, value=mask, padding='post', truncating='pre')
			self.assertTrue(result == [["a"], ["a", "b", "c", "d"], [mask]])
			result = padSequence(data, maxlen, value=mask, padding='pre', truncating='post')
			self.assertTrue(result == [[mask], ["a"], ["a", "b", "c", "d"]])

		def test15(self):
			mask = MASK_TOKEN
			maxlen = 4
			data = [["a"], ["a", "b", "c", "d"]]
			result = padSequence(data, maxlen, value=mask, padding='post', truncating='post')
			self.assertTrue(result == [["a"], ["a", "b", "c"]])
			result = padSequence(data, maxlen, value=mask, padding='pre', truncating='pre')
			self.assertTrue(result == [["a", "b", "c", "d"]])

		def test16(self):
			mask = MASK_TOKEN
			maxlen = 1
			data = [["a"], ["a", "b", "c", "d"]]
			result = padSequence(data, maxlen, value=mask, padding='post', truncating='post')
			self.assertTrue(result == [["a"]])
			result = padSequence(data, maxlen, value=mask, padding='pre', truncating='pre')
			self.assertTrue(result == [["d"]])

		def test18(self):
			mask = MASK_TOKEN
			maxlen = 100
			data = [[]]
			result = padSequence(data, maxlen, value=mask, padding='post', truncating='post')
			self.assertTrue(result == [[mask] * 100])
			result = padSequence(data, maxlen, value=mask, padding='pre', truncating='pre')
			self.assertTrue(result == [[mask] * 100])

		def test19(self):
			mask = MASK_TOKEN
			maxlen = 2
			data = [["a"], ["b"], ["c"]]
			result = padSequence(data, maxlen, value=mask, padding='post', truncating='post')
			self.assertTrue(result == [["a"], ["b"]])
			result = padSequence(data, maxlen, value=mask, padding='pre', truncating='pre')
			self.assertTrue(result == [["b"], ["c"]])




if __name__ == '__main__':
	unittest.main() # Or execute as Python unit-test in eclipse

print("Unit tests done.\n==============")




