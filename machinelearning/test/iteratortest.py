from systemtools.basics import *
from systemtools.logger import *
from systemtools.location import *
from datastructuretools.processing import *
from datatools.jsonutils import *
import random
from multiprocessing import cpu_count, Process, Pipe, Queue, JoinableQueue
import queue
from machinelearning.iterator import *



def useIt(seed, containers):
	random.seed(seed)
	def itemGenerator(container, **kwargs):
		for a in NDJson(container):
			yield a
	def subProcessParseFunct(item, key=None, **kwargs):
		return str(item)[:40] + " " + key
	def mainProcessParseFunct(item, key=None, **kwargs):
		return item + " " + key
	cg = ConsistentIterator(containers, itemGenerator, subProcessParseFunct=subProcessParseFunct, mainProcessParseFunct=mainProcessParseFunct, subProcessParseFunctKwargs={"key": "aaa"}, mainProcessParseFunctKwargs={"key": "bbb"})
	allElements = []
	for a in cg:
		allElements.append(a)
	print(len(allElements))
	print(allElements[0])
	print(allElements[1000])
	if len(allElements) >= 59000:
		print(allElements[60000])
	print(allElements[-2])
	print(allElements[-1])
	return allElements

def test1():
	tt = TicToc()
	tt.tic(display=False)
	containers = sortedGlob("/home/hayj/tmp/Asa/asaminbis/asamin-train-2019.05.22-19.46/*.bz2")
	# containers = sortedGlob("/home/hayj/tmp/Asa/asaminbis/aaa/*.bz2")
	printLTS(containers)
	tt.tic(display=False)
	allElements1 = useIt(0, containers)
	tt.tic()
	allElements2 = useIt(1, containers)
	tt.tic()
	assert len(allElements1) == len(allElements2)
	print("ok1")
	for i in range(len(allElements1)):
		assert allElements1[i] == allElements2[i]
	print("ok2")
	tt.tic(display=False)
	count = 0
	for file in containers:
		for row in NDJson(file):
			count += 1
	tt.tic()
	print(count)
	assert count == len(allElements1)
	print("ok3")

def test2():
	containers = sortedGlob("/home/hayj/tmp/Asa/asaminbis/asamin-train-2019.05.22-19.46/*.bz2")
	def itemGenerator(container, *args, **kwargs):
		for a in NDJson(container):
			yield str(a)[:30]
	# gen = ConsistentIterator(containers, itemGenerator)
	gen = AgainAndAgain(ConsistentIterator, containers, itemGenerator)
	for i in range(3):
		count = 0
		for current in gen:
			count += 1
		print(count)

if __name__ == '__main__':
	test1()
	# test2()
