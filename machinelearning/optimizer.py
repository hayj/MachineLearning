# coding: utf-8

import utils.text as text
import copy
from utils.text import listToStr
import collections
import random

# TODO Faire l'optimiseur hiérarchique avec un memory manager encore mieux qui libère la rap intelligemment + un PreprocessingDesc _al0.1_sw12 (liste numéro 12 de stop Word) attention mes stop Woods dépendent de la tokenisation donc faire une hierarchie dépendante de ça et une version...
# TODO Pour afficher les meilleurs résultats, voir quel paramètre fluctu dans la description et n'afficher que ceux là
# TODO Une fois sur 100 refaire le calcul dans le memory manager pour être sur. Et pouvoir le désactiver et changer la fréquence.
# TODO faire du map reduce sur plusieurs serveurs
# TODO Pour les params qui montrent le plus de variance dans les scores : on fait plus d'essais
# TODO paralleliser
# TODO contrainst at least one suparamTrue...
# TODO prendre une valeur, essayer plein au hazard sur d'autres, puis la fixer, puis faire les autres. Et faire plein de redemarrages aléatoires comme ça. En prennant une autrepremiere variable... Et garder les traces d'execution, exemple : alpha=3, beta=4 | beta=3, alpha=0.1...
# Conseil de ouassim : si on a un arbre profond : monte carlo tree search. Sinon i-racing ou f-racing optimisation, suppression des moins interessants à chaque iteration, comme aux courses, de facon hierarchique, essayer de voir si on peut convertir la structure stsparameters en une structure irace http://iridia.ulb.ac.be/irace/#examples et utiliser cette library
# Voir opencl ou autre pour faire du GPU...
# TODO pyspark et hadoop ou un map-reduce home made pour exectuer sur plusieurs machine (celle de massy, les serveur de la fac, d'Octopeek... ?
# TODO faire en sorte qu'on puisse acceder à la config en utilisant des attributs comme config.data <- "all2016" ou  encore config.learner <- "SVM"
class HierarchicalSearch(object):
    """
    Init Parameters
    ===============
    
     * maximize if you set this param to True, the algorithm will maximize, otherwise it will minimize the compute
     * compute is a function which return a value to maximize (or minimize TODO minimize), it take a dict in parameter
       which specify the current parameters to apply. This function must return None if the given
       parameters can't be computed. In case of machine learning task, you can average cross-predict on the
       training set to indirectly perform a crossvalidation, otherwise you can just use all the data to see the best
       performing set of parameters.
     * algorithm to choose an optimisation algorithm, TODO for now there is only ALL which take all combination of
       parameters
     * parameters is a specific structure which describe all parameters of the function to optimize. For example :
       parameters = \
       [
           {
                'name': 'alpha',
                    # The name of the parameter
                'domain': range(5),
                    # This is the domain of the param, the algorithm will search a "good" value.
                    # For example if you have something hard to compute for each value of this parameter in this domain
                    # you can compute all before the optimization (for example when you have to train a word2vec...)
                    # TODO take into account continuous function
                'sorted': False, # optional, default False
                    # This is an information on values in the domain, is sorted is set to True, the algorithm will
                    # search a good value and take into account that each closest value has chance to be as good
                    # TODO
                'disabled': False, # optional, default False
                    # Just for disable the parameter and all sub-parameters
                'weight': 0.0, # optional, default 0.0
                    # The weight is to set the importance of the current parameter. It takes values between 0 and 1.
                    # For example you can set the weight to 1.0 if this parameter involves a long time computation,
                    # so the paramter will be choose early. Or for example if you know that this param is very
                    # discriminating, you can set a significant weight.
                    # TODO not yet implemented
                'subparams': [...] # optional
                    # Here it's the same structure which is nested. It represent all sub-parameters which depends on
                    # the current parameter. For exemple you can specify a feature in a machine learning job (True or False)
                    # as parent and specify some related params in nested like the algorithm to extract the feature
                'constraints': [[True], [1, 2]],
                    # The domain constraint for parents, grand-parents... Here for example, the direct parent
                    # must be set to True to take into account this sub-parameter, and the grand-parent
                    # must be set to 1 or 2, otherwise this  parameter will not be used to search the
                    # optimum. Note that here it's not usefull because there are no parent and
                    # grand-parent for the current depth.
                    # TODO grand-parents...
                'force': [1, 2]
                    # Allows to force the value to a subset of the domain.
           },
           ...
       ]
    """
    class AlgorithmEnum():
        (
            ALL,
            SIMULATED_ANNEALING, # TODO not yet implemented
            TABU_SEARCH # TODO not yet implemented
        ) = list(range(3))
    
    def __init__(self, parameters, compute, algorithm=AlgorithmEnum.ALL, maximize=True, verbose=True, maxCombinason=1000000, validators=[], shuffle=False):
        self.parameters = copy.deepcopy(parameters)
        self.compute = compute
        self.shuffle = shuffle
        self.algorithm = algorithm
        self.maximize = maximize
        self.verbose = verbose
        self.maxCombinason = maxCombinason
        self.validators = validators
    
    
    def optimize(self):
        """
        This function call compute with differents specified parameters. It return a tuple (bestDict, allDicts)
        """
        if self.algorithm == HierarchicalSearch.AlgorithmEnum.ALL:
            return self.all()
        else:
            raise NotImplementedError("not yet implemented")
    
    def all(self):
        # Get all conbinason from the parameters :
        combinasons = self.getCombinasons()
        if self.shuffle:
            random.shuffle(combinasons)
        if self.verbose:
            print("Optimization for " + str(len(combinasons)) + " combinasons")
        results = []
        for combinason in combinasons:
            score = self.compute(combinason)
            if score is not None:
                results.append((combinason, score))
        
        if len(results) == 0:
            return None
        else:
            best = [results[0]]
            for i in range(1, len(results)):
                if results[i][1] == best[0][1]:
                    best.append(results[i])
                else:
                    if self.maximize:
                        if results[i][1] > best[0][1]:
                            best = [results[i]]
                    else:
                        if results[i][1] < best[0][1]:
                            best = [results[i]]
            return (best, results)
    
    def cleanCombinasons(self, combinasons):
        toDelete = []
        for i in range(len(combinasons)):
            config = combinasons[i]
            for validator in self.validators:
                if not validator(config):
                    toDelete.append(i)
        
        deletedCount = 0
        for i in toDelete:
            del combinasons[i - deletedCount]
            deletedCount += 1
        
        return combinasons
    
    def sortCombinasons(self, combinasons):
        sortedCombinasons = []
        for comb in combinasons:
            sortedComb = collections.OrderedDict(sorted(comb.items()))
            sortedCombinasons.append(sortedComb)
        return sortedCombinasons

    
    def getCombinasons(self):
        parameters = self.preprocParameters(self.parameters)
        combinasons = self.getCombinasonsOf(parameters)
        combinasons = self.cleanCombinasons(combinasons)
        combinasons = self.sortCombinasons(combinasons)
        return combinasons
    
    def preprocParameters(self, parameters):
        i = 0
        toDelete = []
        # For all params :
        for param in parameters:
            # We overwrite the domain :
            if 'force' in param:
                param['domain'] = param['force']
                del param['force']
            # We delete if it is disabled :
            if 'disabled' in param:
                if param['disabled']:
                    toDelete.append(i)
                del param['disabled']
            i += 1
        
        deletedCount = 0
        for i in toDelete:
            del parameters[i - deletedCount]
            deletedCount += 1
        
        # And we apply this function recursively :
        for param in parameters:
            if 'subparams' in param:
                self.preprocParameters(param['subparams'])
               
#         if parameters == self.parameters:
#             print text.listToStr(parameters)
        return parameters
    
    def getCombinasonsOf(self, parameters, constraint=None, parentId=""):
        """
        This func return a list of dict, each dict is a combinason of all parameters
        """
        # For each param, we combine it with all its subparams :
        allParamCombinasons = []
        for param in parameters:
            go = False
            if constraint is None:
                go = True
            elif ('constraints' not in param):
                go = True
            elif (constraint in param['constraints'][0]):
                go = True
            if go:
                currentCombinason = []
                # For all values :
                for value in param['domain']:
                    valueInserted = False
                    # We take all subparams :
                    if 'subparams' in param:
                        parentIdTmp = parentId
                        if parentIdTmp != "":
                            parentIdTmp += '.'
                        allSubCombinasons = self.getCombinasonsOf(param['subparams'], constraint=value, parentId=(parentIdTmp+param['name']))
                        # And for each combinason of all subparams (according to the current value,
                        # we add the value of the parent :
                        for combinason in allSubCombinasons:
                            combinason[parentIdTmp+param['name']] = value
                            valueInserted = True
                        currentCombinason += allSubCombinasons
                    # If this value has no subparam with right constraints,
                    # we had it alone :
                    if not valueInserted:
                        parentIdTmp = parentId
                        if parentIdTmp != "":
                            parentIdTmp += '.'
                        currentCombinason.append({(parentIdTmp + param['name']): value})
                        
                allParamCombinasons.append(currentCombinason)
                if len(allParamCombinasons) > self.maxCombinason:
                    raise OverflowError('Too much possibility')

        # Here we are all subparams combinason with all parent,
        # So now we have to combine all params together :
        allCombinason =  self.combineParams(allParamCombinasons)
        
        return allCombinason
    
    
    def combineParams(self, allParamCombinasons, index=0):
        """
        This function return the combinaison of all dict in the list,
        but not in a single element of the list 
        """
        # If there is no param :
        if (len(allParamCombinasons) == 0) :
            return []
        # Or we are at the end :
        elif (len(allParamCombinasons) == (index + 1)):
            return allParamCombinasons[index]
        # Else we must combine the current index with all next indexes :
        else:
            allCombinason = []
            nextComb = self.combineParams(allParamCombinasons, index + 1)
            for element in allParamCombinasons[index]:
                currentComb = []
                for element2 in nextComb:
                    currentComb.append(dict(list(element.items()) + list(element2.items())))
                allCombinason += currentComb
                if len(allCombinason) > self.maxCombinason:
                    raise OverflowError('Too much possibility')
            return allCombinason
        
        
    def getTop(self, l, n):
        sortedBySecond = sorted(l, key=lambda tup: tup[1])
        if self.maximize:
            sortedBySecond = list(reversed(sortedBySecond))
        return sortedBySecond[0:n]
        
        
        
