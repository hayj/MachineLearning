from systemtools.basics import *
from systemtools.logger import *
from systemtools.location import *
from datastructuretools.processing import *
from datatools.jsonutils import *
import random
from multiprocessing import cpu_count, Process, Pipe, Queue, JoinableQueue
# from multiprocessing import Lock as MPLock
import queue
import numpy as np
from threading import Lock as TLock
from machinelearning.utils import *


TERMINATED_TOKEN = "__TERMINATED__"
NO_RESULT_TOKEN = "__NO_RESULT__"

class AgainAndAgain():
    # https://www.reddit.com/r/Python/comments/40idba/easy_way_to_make_an_iterator_from_a_generator_in/
    def __init__(self, generator_func, *args, **kwargs):
        self.generator_func = generator_func
        self.args = args
        self.kwargs = kwargs
    def __iter__(self):
        return self.generator_func(*self.args, **self.kwargs)

def iteratorToArray(it, steps=None):
    if it is None:
        return None
    newVal = None
    if isinstance(it, InfiniteBatcher):
        batchs = []
        for i in range(steps):
            current = next(it)
            batchs.append(current)
        if isListOrArray(batchs[0][0]):
            newVal = np.vstack(batchs)
        else:
            newVal = np.array(flattenLists(batchs))
    elif isinstance(it, list):
        newVal = np.array(it)
    elif isinstance(it, np.ndarray):
        newVal = it
    else:
        newVal = []
        for current in it:
            newVal.append(current)
        newVal = np.array(newVal)
    return newVal

def itemGeneratorWrapper(container, itemGenerator, itemGeneratorArgs, itemGeneratorKwargs, subProcessParseFunct, subProcessParseFunctArgs, subProcessParseFunctKwargs, itemQueue, verbose=False, name=None):
    logger = None
    if verbose:
        if name is None:
            name = getRandomStr()
        logger = Logger(name + ".log")
    for current in itemGenerator(container, *itemGeneratorArgs, **itemGeneratorKwargs, logger=logger, verbose=verbose):
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
        take care of defining your callbacks with *args and **kwargs to avoid errors...
        See an example in machinelearning.test.iteratortest
        You can wrap an instance of ConsistentIterator in AgainAndAgain, so your iterator can be restarted again and again...
        TODO merge itemGenerator and subProcessParseFunct
    """
    def __init__\
    (
        self,
        containers,
        itemGenerator,
        itemGeneratorArgs=None,
        itemGeneratorKwargs=None,
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
        self.itemGeneratorArgs = itemGeneratorArgs or ()
        self.itemGeneratorKwargs = itemGeneratorKwargs or dict()
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
        self.tlock = TLock()
        # self.mplock = MPLock()
        for c in self.containers:
            self.containersQueue.put(c)

    def __iter__(self):
        return self

    def __next__(self):
        result = NO_RESULT_TOKEN
        doRaise = False
        with self.tlock:
            # with self.mplock:
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
                                    args=(container, self.itemGenerator, self.itemGeneratorArgs, self.itemGeneratorKwargs, self.subProcessParseFunct, self.subProcessParseFunctArgs, self.subProcessParseFunctKwargs, self.queues[i],),
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
                    doRaise = True
                    break
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
        # We check if we have to raise:
        if doRaise:
            raise StopIteration
        # We return the result:
        return result


class InfiniteBatcher:
    """
        This class take an AgainAndAgain iterator and yield batch samples
        An AgainAndAgain iterator is an iterator that calling `iter(instance)` or `for i in instance`
        will produce a new fresh iterator to be able to iterate again and again...

        Each tuple which is yield by the iterator given will be transformed in batchs.
        It means instead of return (a, b) then (c, d) then (e, f) (from the AgainAndAgain instance)
        it wil return, for a bacth size of 2, ([a, c], [b, d]) then ([e], [f]) then ([a, c], [b, d]) then ([e], [f]) etc infinitely without StopIteration, which mean without the need to call `for i in instance` several time.

        It will automatically detect if yielded elements are tuples... This mean if your againAndAgainIterator doesn't yield tuples, the infinite batcher will simply yield batches of elements instead of batches of tuples.

        This class is usefull to create generators for keras `fit_generator`.

        To pass an InfiniteBatcher to keras `fit_generator` you must first count the number of samples and call `history = model.fit_generator(myInfiniteBatcher, steps_per_epoch=math.ceil(trainSamplesCount / myInfiniteBatcher.batchSize)` 
        
        Use shuffle=1 to shuffle each bacthes. If shuffle > 1, multiple batches will be flattened, shuffled, re-splitted and returned... If shuffle is None or 0, no shuffling will be applied.

        Use skip to skip n batches, typically usefull when you resume a deep learning training from a previous train (to do not start allways at the beggining of the dataset if you resume your training again and again...)

        Use queueSize > 1 to pre-load batches

        TODO param `fillBatches` which never let a batch had a len lower that the batch size (typically when the end of the dataset arrive with a StopIteration)

    """
    def __init__(self, againAndAgainIterator, batchSize=128, skip=0, shuffle=0, seed=0, toNumpyArray=True, queueSize=1, logger=None, verbose=True):
        assert isinstance(againAndAgainIterator, AgainAndAgain)
        self.logger = logger
        self.verbose = verbose
        self.skip = skip
        self.shuffle = shuffle
        if self.shuffle is None or self.shuffle < 0:
            self.shuffle = 0
        self.againAndAgainIterator = againAndAgainIterator
        self.toNumpyArray = toNumpyArray
        self.batchSize = batchSize
        self.currentGenerator = None
        self.tlock = TLock()
        self.queueSize = queueSize
        if self.queueSize < 1:
            self.queueSize = 1
        self.seed = seed
        self.rd = random.Random(self.seed)
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        with self.tlock:
            while self.queue.qsize() < self.queueSize:
                self.__enqueue()
            return self.queue.get()

    def __enqueue(self):
        batches = []
        if self.shuffle == 0 or self.shuffle == 1:
            nbBatches = 1
        else:
            nbBatches = self.shuffle
        gotStop = False
        while len(batches) < nbBatches and not gotStop:
            if self.currentGenerator is None:
                self.currentGenerator = iter(self.againAndAgainIterator)
            batch = None
            try:
                for i in range(self.batchSize):
                    current = next(self.currentGenerator)
                    if batch is None:
                        batch = []
                    batch.append(current)
            except StopIteration:
                self.currentGenerator = None
                self.rd = random.Random(self.seed)
                gotStop = True
            if batch is not None and len(batch) > 0:
                batches.append(batch)
        if self.shuffle > 0:
            tmpBatchesLen = len(batches)
            batches = flattenLists(batches)
            self.rd.shuffle(batches)
            batches = split(batches, tmpBatchesLen)
        isTuple = False
        try:
            isTuple = isinstance(batches[0][0], tuple)
        except: pass
        if isTuple:
            newBatches = []
            for u in range(len(batches)):
                batch = []
                for i in range(len(batches[0][0])):
                    batch.append([])
                newBatches.append(batch)
            batchIndex = 0
            for batch in batches:
                for currentTuple in batch:
                    tupleIndex = 0
                    for element in currentTuple:
                        newBatches[batchIndex][tupleIndex].append(element)
                        tupleIndex += 1
                batchIndex += 1
            batches = newBatches
        if self.toNumpyArray:
            for u in range(len(batches)):
                if isTuple:
                    for i in range(len(batches[u])):
                        batches[u][i] = np.array(batches[u][i])
                else:
                    batches[u] = np.array(batches[u])
        if isTuple:
            for i in range(len(batches)):
                batches[i] = tuple(batches[i])
        for batch in batches:
            if self.skip == 0:
                self.queue.put(batch)
            else:
                self.skip -= 1



class InfiniteBatcher_deprecated1:
    """
        This class take an AgainAndAgain iterator and yield batch samples
        An AgainAndAgain iterator is an iterator that calling `iter(instance)` or `for i in instance`
        will produce a new fresh iterator to be able to iterate again and again...
        Each tuple which is yield by the iterator given will be transformed in batchs.
        It means instead of return (a, b) then (c, d) then (e, f) (from the AgainAndAgain instance)
        it wil return, for a bacth size of 2, ([a, c], [b, d]) then ([e], [f]) then ([a, c], [b, d]) then ([e], [f]) etc infinitely without StopIteration, which mean without the need to call `for i in instance` several time.
        To pass an InfiniteBatcher to keras `fit_generator` you must first count the number of samples and call `history = model.fit_generator(myInfiniteBatcher, steps_per_epoch=math.ceil(trainSamplesCount / myInfiniteBatcher.batchSize)` 

    """
    def __init__(self, againAndAgainIterator, batchSize, toNumpyArray=True, logger=None, verbose=True):
        assert isinstance(againAndAgainIterator, AgainAndAgain)
        self.logger = logger
        self.verbose = verbose
        self.againAndAgainIterator = againAndAgainIterator
        self.toNumpyArray = toNumpyArray
        self.batchSize = batchSize
        self.currentGenerator = None
        self.tlock = TLock()

    def __iter__(self):
        return self
    def __next__(self):
        self.tlock.acquire()
        if self.currentGenerator is None:
            self.currentGenerator = iter(self.againAndAgainIterator)
            # log("Init of a new generator from the given AgainAndAgain instance...", self)
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
                    self.tlock.release()
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
        self.tlock.release()
        if isTuple:
            return tuple(data)
        else:
            return data

