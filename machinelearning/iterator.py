from systemtools.basics import *
from systemtools.logger import *
from systemtools.location import *
from datastructuretools.processing import *
from datatools.jsonutils import *
import random
from multiprocessing import cpu_count, Process, Pipe, Queue, JoinableQueue
import queue
import numpy as np


TERMINATED_TOKEN = "__TERMINATED__"
NO_RESULT_TOKEN = "__NO_RESULT__"

def itemGeneratorWrapper(container, itemGenerator, subProcessParseFunct, subProcessParseFunctArgs, subProcessParseFunctKwargs, itemQueue, verbose=False, name=None):
    logger = None
    if verbose:
        if name is None:
            name = getRandomStr()
        logger = Logger(name + ".log")
    for current in itemGenerator(container, logger=logger, verbose=verbose):
        if subProcessParseFunct is not None:
            current = subProcessParseFunct(current, *subProcessParseFunctArgs, **subProcessParseFunctKwargs, logger=logger, verbose=verbose)
        itemQueue.put(current)
    itemQueue.put(TERMINATED_TOKEN)
    itemQueue.close()

class ConsistentIterator:
    """
        This class allow you to generate items from containers (list of files)
        and yield all elements using multiprocessing BUT in the same order
        so you can iterate and yiled labels in one generator and in an other
        yield data.
        You must give containers, basically list of files
        itemGenerator which is a generator which have to yield rows
        subProcessParseFunct which must do the maximum of data preprocessing because it works in sub processes
        (take care of data serialization (= duplication in each process) --> memory usage)
        it have to return values (not yield)
        mainProcessParseFunct take the output of subProcessParseFunct and return values, it works on the main unique python process, so it can be usefull when you cannot pass object across process (which mean the object is not serializable) or the object you want to pass is too big to be duplicated on each subprocess...
        take care of defining your callbacks with *args and **kwargs...
        See an example in machinelearning.test.iteratortest
        You can wrap an instance of ConsistentIterator in AgainAndAgain, so your iterator can be restarted again and again...
    """
    def __init__\
    (
        self,
        containers,
        itemGenerator,
        subProcessParseFunct=None,
        subProcessParseFunctArgs=None,
        subProcessParseFunctKwargs=None,
        mainProcessParseFunct=None,
        mainProcessParseFunctArgs=None,
        mainProcessParseFunctKwargs=None,
        logger=None,
        verbose=True,
        parallelProcesses=cpuCount(),
        queuesMaxSize=100000,
        subProcessesVerbose=False,
        seed=None,
        printRatio=0.1,
    ):
        self.containers = containers
        self.itemGenerator = itemGenerator
        self.subProcessParseFunct = subProcessParseFunct
        self.subProcessParseFunctArgs = subProcessParseFunctArgs or ()
        self.subProcessParseFunctKwargs = subProcessParseFunctKwargs or dict()
        self.mainProcessParseFunct = mainProcessParseFunct
        self.mainProcessParseFunctArgs = mainProcessParseFunctArgs or ()
        self.mainProcessParseFunctKwargs = mainProcessParseFunctKwargs or dict()
        self.logger = logger
        self.verbose = verbose
        self.parallelProcesses = parallelProcesses
        self.queuesMaxSize = queuesMaxSize
        self.subProcessesVerbose = subProcessesVerbose
        self.seed = seed
        self.printRatio = printRatio
        self.init()

    def init(self):
        if self.seed is not None:
            random.seed(self.seed)
        log(str(len(self.containers)) + " containers to process.", self)
        self.pbar = ProgressBar(len(self.containers), logger=self.logger, verbose=self.verbose, printRatio=self.printRatio)
        self.processes = [None] * self.parallelProcesses
        self.queues = [None] * self.parallelProcesses
        self.currentIndex = 0
        self.containersQueue = queue.Queue() # WARNING, don't use the processing Queue but queue.Queue because the procesing queue `put` method is async...
        for c in self.containers:
            self.containersQueue.put(c)

    def __iter__(self):
        return self

    def __next__(self):
        result = NO_RESULT_TOKEN
        while isinstance(result, type(NO_RESULT_TOKEN)) and result == NO_RESULT_TOKEN:
            if self.processes[self.currentIndex] is None and self.containersQueue.qsize() > 0:
                # First we find an element which is not None (if exists):
                self.currentIndex = None
                for i in range(len(self.processes)):
                    if self.processes[i] is not None:
                        self.currentIndex = i
                        break
                # Then we fill all None in processes:
                try:
                    for i in range(len(self.processes)):
                        # print(i)
                        if self.processes[i] is None:
                            # time.sleep(1)
                            container = self.containersQueue.get(block=False)
                            # We take the first we insert as the self.currentIndex:
                            if self.currentIndex is None:
                                self.currentIndex = i
                            self.queues[i] = Queue(self.queuesMaxSize)
                            self.processes[i] = Process\
                            (
                                target=itemGeneratorWrapper,
                                args=(container, self.itemGenerator, self.subProcessParseFunct, self.subProcessParseFunctArgs, self.subProcessParseFunctKwargs, self.queues[i],),
                                kwargs={"verbose": self.subProcessesVerbose, "name": None}
                            )
                            self.processes[i].start()
                except queue.Empty as e:
                    pass
            # If we have no more process, we can break:
            allAreNone = True
            for current in self.processes:
                if current is not None:
                    allAreNone = False
                    break
            if allAreNone:
                raise StopIteration
            # We get the current process and queue:
            currentProcess = self.processes[self.currentIndex]
            currentQueue = self.queues[self.currentIndex]
            if currentQueue is not None:
                # We get the next element:
                try:
                    current = currentQueue.get()
                    if isinstance(current, type(TERMINATED_TOKEN)) and current == TERMINATED_TOKEN:
                        raise queue.Empty
                    # if self.currentIndex == 1: print("a")
                    if self.mainProcessParseFunct is not None:
                        current = self.mainProcessParseFunct(current, *self.mainProcessParseFunctArgs, **self.mainProcessParseFunctKwargs, logger=self.logger, verbose=self.verbose)
                    result = current
                except queue.Empty:
                    # We remove the current process and queue if there are no more items:
                    currentProcess.join()
                    self.processes[self.currentIndex] = None
                    self.queues[self.currentIndex] = None
                    self.pbar.tic()
            # We go to the next process:
            self.currentIndex += 1
            if self.currentIndex == len(self.processes):
                self.currentIndex = 0
        # We return the result:
        return result


class InfiniteBatcher:
    """
        This class take an AgainAndAgain iterator and yield batch samples
        An AgainAndAgain iterator is an iterator that calling `iter(instance)` of `for i in instance`
        will produce a new fresh iterator to be able to iterate again and again...
        Each tuple which is yield by the iterator given will be transformed in batchs.
        It means instead of return (a, b) then (c, d) then (e, f) (from the AgainAndAgain instance)
        it wil return, for a bacth size of 2, ([a, c], [b, d]) then ([e], [f]) then ([a, c], [b, d]) then ([e], [f]) etc infinitely without StopIteration, which mean without the need to call `for i in instance` several time.

    """
    def __init__(self, againAndAgainIterator, batchSize, toNumpyArray=True, logger=None, verbose=True):
        assert isinstance(againAndAgainIterator, AgainAndAgain)
        self.againAndAgainIterator = againAndAgainIterator
        self.toNumpyArray = toNumpyArray
        self.batchSize = batchSize
        self.logger = logger
        self.verbose = verbose
        self.currentGenerator = None
    def __iter__(self):
        return self
    def __next__(self):
        if self.currentGenerator is None:
            self.currentGenerator = iter(self.againAndAgainIterator)
            log("Init of a new generator from the given AgainAndAgain instance...", self)
        data = None
        isTuple = False
        for i in range(self.batchSize):
            try:
                current = next(self.currentGenerator)
                if isinstance(current, tuple):
                    isTuple = True
                    if data is None:
                        data = [None] * len(current)
                        for u in range(len(data)):
                            data[u] = []
                    for tupleIndex in range(len(current)):
                        data[tupleIndex].append(current[tupleIndex])
                else:
                    if data is None:
                        data = []
                    data.append(current)
            except StopIteration:
                self.currentGenerator = None
                if data is None:
                    return next(self)
                else:
                    break
        assert data is not None
        assert (isTuple and len(data[0]) > 0) or (len(data) > 0)
        if self.toNumpyArray:
            if isTuple:
                for i in range(len(data)):
                    data[i] = np.array(data[i])
            else:
                data = np.array(data)
        if isTuple:
            return tuple(data)
        else:
            return data