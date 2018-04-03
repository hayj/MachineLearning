# coding: utf-8

from fr.hayj.util.parser import SerializableParser
from fr.hayj.util.text import *

class Feature(SerializableParser):
    def __init__(self, *args, **kwargs):
        # We get some params :
        erase = kwargs.pop('erase', False)
        # We set the autoSerialize if not specified :
        autoSerialize = kwargs.pop('autoSerialize', False)
        kwargs['autoSerialize'] = autoSerialize
        
        # Get the file id :
        config = kwargs.pop("config", None)
        hashConfig = type(self).getFeatureHashConfig(config)
        fileId = kwargs.pop('fileId', "")
        if fileId != "":
            fileId = fileId + "_"
        fileId = fileId + hashConfig
        kwargs['fileId'] = fileId
        
        # We init the parser :
        super(Feature, self).__init__(*args, **kwargs)
        # We clean the parser if needed :
        self.erase = erase
        if self.erase:
            self.clean()
    
    @staticmethod
    def getFeatureHashConfig(config):
        return "no_hash"
    
    @staticmethod
    def getDataHashConfig(config):
        return "no_hash"
    
    def computeFeature(self):
        raise NotImplementedError("Please implement this method")
        
    def parseAll(self, sentences):
        super(Feature, self).parseAll(sentences, self.computeFeature)

    def parse(self, sentence, *args, **kwargs):
        return super(Feature, self).parse(sentence, self.computeFeature, *args, **kwargs)
    
    def initParsing(self, *args, **kwargs):
        return None
    
    @staticmethod
    def getNormalHashFeature(theClass, config):
        hashConfig = []
        for key, value in list(config.items()):
            if theClass.__name__ in key:
                hashConfig.append(key + str(value))
        hashConfig = strToHashCode(listToStr(hashConfig))
        return str(theClass.__name__.lower()) + "_" + hashConfig

    def needInit(self):
        return False


