
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from newssource.asattribution.asamin import *
from newssource.asa.asapreproc import *
from databasetools.mongo import mongoStorable
from systemtools.basics import *
from systemtools.file import *


"""
 * on a des id uniques dans row["id"], par contre ils ne suivent pas et provoquent un MemoryError
 * Si on utilise un count manuelle alors on aura pas le meme id pour chaque iteration sur le corpus
 * Du coup on peut faire consistentYields=True mais c'est plus lent
 * Donc on peut donner str(row["id"])
 * Ou alors on passe une premiere fois sur le dataset pour faire un mapping des items et ordonné...
 * Mais 1) ce sera pas le meme mapping entre différents run et 2) pas dans l'ordre
 * Sinon directement dans 10-asamin on met des ids dans l'ordre...
 * avec un script qui édite un asamin directory, mais pareil c'est pas dans l'ordre donc autant
 * faire un count avec consistentYields=True
 * Donc solution finale, un consistent count ou alors un str
"""


def checkModel(modelFilePath, logger=None):
    """
        This function will try to load a doc2vec model and try to infer a vector
    """
    try:
        log("Size of the model: " + getHumanSize(modelFilePath))
        log("Size of the model directory: " + getHumanSize(decomposePath(modelFilePath)[0]))
        loadedModel = Doc2Vec.load(modelFilePath)
        log(str(loadedModel.infer_vector(["system", "response"]))[:100] + "...", logger)
        log("Model ok", logger)
        return True
    except Exception as e:
        logException(e, logger)
        logError("Model NOK", logger)
        return False