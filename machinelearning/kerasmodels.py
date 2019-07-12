from keras.utils import multi_gpu_model
from keras import backend
from systemtools.logger import *
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.callbacks import Callback, History
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.utils import multi_gpu_model
from machinelearning.metrics import *
from systemtools.logger import *
from systemtools.duration import *
from systemtools.basics import *
from systemtools.file import *
from systemtools.location import *
from systemtools.printer import *
from systemtools.system import *
from systemtools.logger import *
from datatools.jsonutils import *
from machinelearning.utils import *
from machinelearning.iterator import *
import copy
import os


from keras.layers import LSTM, GRU, Dense, CuDNNLSTM, CuDNNGRU, Bidirectional
from keras.layers import BatchNormalization, Activation, SpatialDropout1D, InputSpec
from keras.layers import MaxPooling1D, TimeDistributed, Flatten, concatenate, Conv1D
from keras.utils import multi_gpu_model, plot_model
from keras.layers import concatenate, Input, Dropout
from keras.models import Model, load_model, Sequential
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback, History, ModelCheckpoint, EarlyStopping
from keras import optimizers
from keras import callbacks
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras import backend as K


class AttentionWeightedAverage(Layer):
    """
        From : https://github.com/tsterbak/keras_attention/blob/master/models.py
        Computes a weighted average attention mechanism from:
            Zhou, Peng, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao and Bo Xu.
            “Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification.”
            ACL (2016). http://www.aclweb.org/anthology/P16-2034
        How to use:
        see: [BLOGPOST]
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.w = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_w'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.w]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, h, mask=None):
        h_shape = K.shape(h)
        d_w, T = h_shape[0], h_shape[1]
        
        logits = K.dot(h, self.w)  # w^T h
        logits = K.reshape(logits, (d_w, T))
        alpha = K.exp(logits - K.max(logits, axis=-1, keepdims=True))  # exp
        
        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            alpha = alpha * mask
        alpha = alpha / K.sum(alpha, axis=1, keepdims=True) # softmax
        r = K.sum(h * K.expand_dims(alpha), axis=1)  # r = h*alpha^T
        h_star = K.tanh(r)  # h^* = tanh(r)
        if self.return_attention:
            return [h_star, alpha]
        return h_star

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None


def buildRNN\
(
    docLength=None,
    vocSize=None,
    embeddingMatrix=None,
    nbClasses=None, # units of the last layer
    isEmbeddingsTrainable=False,
    denseUnits=[],
    denseActivation='tanh', # tanh, sigmoid, relu
    rnnUnits=100,
    embSpacialDropout=0.0,
    firstDropout=0.0,
    recurrentDropout=0.0,
    attentionDropout=0.0,
    denseDropout=0.0,
    useRNNDropout=True,
    isBidirectional=False,
    isCuDNN=False,
    rnnType='LSTM', # GRU, LSTM
    addAttention=False,
    addConv1D=False,
    conv1dActivation='relu',
    filters=32,
    kernelSize=3,
    poolSize=3,
    bnAfterEmbedding=False,
    bnAfterRNN=False,
    bnAfterAttention=False,
    bnAfterDenses=False,
    bnAfterLast=False,
    bnBeforeActivation=True,
    lastActivation="softmax", # softmax, sigmoid
    logger=None,
    verbose=False,
    returnBuildInfos=True,
):
    """
        This function return a RNN Keras model (GRU or LSTM) which can, optionaly, be CuDNN optimized, bidirectional)
        After getting the model you can convert the model to a multi gpu model, compile the model, print the summary and plot the architecture
    """
    # We get build infos:
    currentKwargs = copy.copy(locals())
    del currentKwargs["logger"]
    del currentKwargs["verbose"]
    del currentKwargs["embeddingMatrix"]
    script = fileToStr(os.path.realpath(__file__))
    # We check values:
    assert denseActivation in [None, 'tanh', 'sigmoid', 'relu']
    assert rnnType in ['GRU', 'LSTM']
    assert docLength is not None
    assert vocSize is not None
    assert embeddingMatrix is not None
    assert nbClasses is not None
    assert denseUnits is not None
    if not isinstance(denseUnits, list):
        denseUnits = [denseUnits]
    # We remove dropouts:
    if bnAfterEmbedding and embSpacialDropout > 0.0:
        embSpacialDropout = 0.0
        logWarning("We remove embSpacialDropout because we use bnAfterEmbedding", logger=logger, verbose=verbose)
    if bnAfterRNN and firstDropout > 0.0:
        firstDropout = 0.0
        logWarning("We remove firstDropout because we use bnAfterRNN", logger=logger, verbose=verbose)
    if bnAfterRNN and recurrentDropout > 0.0:
        recurrentDropout = 0.0
        logWarning("We remove recurrentDropout because we use bnAfterRNN", logger=logger, verbose=verbose)
    if bnAfterAttention and attentionDropout > 0.0:
        attentionDropout = 0.0
        logWarning("We remove attentionDropout because we use bnAfterAttention", logger=logger, verbose=verbose)
    if bnAfterDenses and denseDropout > 0.0:
        denseDropout = 0.0
        logWarning("We remove denseDropout because we use bnAfterDenses", logger=logger, verbose=verbose)
    # We find some informations:
    embeddingsDimension = embeddingMatrix.shape[1]
    assert embeddingsDimension <= 1500
    # We define the model:
    input = Input(shape=(docLength,))
    # We add en embedding layer:
    current = Embedding(vocSize, embeddingsDimension,
                        input_length=docLength,
                        weights=[embeddingMatrix],
                        trainable=isEmbeddingsTrainable)(input)
    # We add a dropout to the emb layer:
    if embSpacialDropout is not None and embSpacialDropout > 0.0:
        current = SpatialDropout1D(embSpacialDropout)(current)
    # We add a batch normalization:
    if bnAfterEmbedding:
        current = BatchNormalization()(current)
    # Optionnaly we add a Conv1D layer:
    if addConv1D:
        current = Conv1D(filters=filters, kernel_size=kernelSize,
                         padding='same', activation=conv1dActivation)(current)
        current = MaxPooling1D(pool_size=poolSize)(current)
    # We find the RNN to add:
    if isCuDNN:
        if rnnType == 'GRU':
            TheLayerClass = CuDNNGRU
        else:
            TheLayerClass = CuDNNLSTM
        useRNNDropout = False
    else:
        if rnnType == 'GRU':
            TheLayerClass = GRU
        else:
            TheLayerClass = LSTM
    # We add a dropout and the RNN:
    if firstDropout is not None and firstDropout > 0.0 and not useRNNDropout:
        current = Dropout(firstDropout)(current)
    if useRNNDropout:
        if isBidirectional:
            current = Bidirectional(TheLayerClass(rnnUnits, dropout=firstDropout,
                                                  recurrent_dropout=recurrentDropout,
                                                  return_sequences=addAttention))(current)
        else:
            current = TheLayerClass(rnnUnits, dropout=firstDropout,
                                    recurrent_dropout=recurrentDropout,
                                    return_sequences=addAttention)(current)
    else:
        if isBidirectional:
            current = Bidirectional(TheLayerClass(rnnUnits, return_sequences=addAttention))(current)
        else:
            current = TheLayerClass(rnnUnits, return_sequences=addAttention)(current)
    if recurrentDropout is not None and recurrentDropout > 0.0 and not useRNNDropout:
        current = Dropout(recurrentDropout)(current)
    # We add a bn:
    if bnAfterRNN:
        current = BatchNormalization()(current)
    # We add an attention layer:
    if addAttention:
        current, attn = AttentionWeightedAverage(return_attention=True)(current)
        if attentionDropout is not None and attentionDropout > 0.0:
            current = Dropout(attentionDropout)(current)
        if bnAfterAttention:
            current = BatchNormalization()(current)
    # We add dense layers:
    for units in denseUnits:
        current = Dense(units)(current)
        if bnAfterDenses and bnBeforeActivation:
            current = BatchNormalization()(current)
        current = Activation(denseActivation)(current)
        if bnAfterDenses and not bnBeforeActivation:
            current = BatchNormalization()(current)
        if denseDropout is not None and denseDropout > 0.0:
            current = Dropout(denseDropout)(current)
    # We add the last layer:
    current = Dense(nbClasses)(current)
    if bnAfterLast and bnBeforeActivation:
        current = BatchNormalization()(current)
    # In case it is categorical_crossentropy:
    current = Activation(lastActivation)(current)
    # In case we do binary_crossentropy:
    # out = Dense(1, activation='sigmoid')(x)
    if bnAfterLast and not bnBeforeActivation:
        current = BatchNormalization()(current)
    # We build the model:
    model = Model(inputs=input, outputs=current)
    # We return the model and build infos:
    if returnBuildInfos:
        return (model, currentKwargs, script)
    # Else we just return the model:
    else:
        return model

def saveModel(model, directoryPath, kwargs=None, script=None,
                makeSubDir=True, extraInfos=None,
                logger=None, verbose=True):
    """
        This function save a model, its weights and other information like kwargs from the build...
    """
    assert isDir(directoryPath)
    if script is None:
        fileToStr(os.path.realpath(__file__))
    if kwargs is None:
        logWarning("You didn't provide kwargs", logger=logger, verbose=verbose)
    json = model.to_json()
    if makeSubDir:
        key = objectToHash((json, kwargs, script))
        name = "kerasmodel-" + key + "-" + getDateSec()
        directoryPath += "/" + name
        mkdir(directoryPath)
    log("Starting to save the model in " + directoryPath, logger=logger, verbose=verbose)
    if kwargs is not None:
        toJsonFile(kwargs, directoryPath + "/kwargs.json")
        log("We saved kwargs.json", logger=logger, verbose=verbose)
    if extraInfos is not None:
        toJsonFile(extraInfos, directoryPath + "/extraInfos.json")
        log("We saved extraInfos.json", logger=logger, verbose=verbose)
    strToFile(script, directoryPath + "/script.py")
    log("We saved script.py", logger=logger, verbose=verbose)
    strToFile(json, directoryPath + "/model.json")
    log("We saved model.json", logger=logger, verbose=verbose)
    try:
        plot_model(model, to_file=directoryPath + "/model.png")
        log("We saved model.png", logger=logger, verbose=verbose)
    except:
        logError("Failed to save model.png", logger=logger, verbose=verbose)
    model.save_weights(directoryPath + "/weights.h5")
    log("We saved weights.h5. Now you can re-build the model using the script and use model.load_weights(...)",
        logger=logger, verbose=verbose)
    return directoryPath



def getAttentions(model, inputs, attToken="Attention"):
    """
        This function will return activation maps.
        You give the original model, the function will take a part of it, from the input to the attention layer
        You give inpus which are same inputs as training or validation data (no labels)
        The function will return the activation map with the shape (<nb samples>, <length of each input>)
        Then you can use the `attentions` with showAttentionMap from machinelearning.attmap.builder
    """
    # We get string representation of all layers:
    layersStr = [str(l) for l in model.layers]
    # We check that we have only one attention layer:
    nbAttLayer = len([l for l in layersStr if attToken in l])
    assert nbAttLayer == 1
    # We get the index of the attention layer:
    attIndex = -1
    for i, l in enumerate(layersStr):
        if attToken in l:
            attIndex = i
            break
    assert attIndex > 0
    # We build the attModel:
    attModelLayers = model.layers[0:attIndex + 1]
    attModel = Model(attModelLayers[0].input, attModelLayers[-1].output)
    # We predict attentions:
    yAttn = attModel.predict(inputs)[1]
    # We reshape all:
    attentions = []
    for i in range(len(inputs)):
        currentAttention = yAttn[i]
        activationMap = np.expand_dims(currentAttention, axis=1)
        activationMap = [a[0] for a in activationMap]
        attentions.append(activationMap)
    # And finally we return the result:
    return attentions


def test1():
    # We get embeddigns:
    from nlptools.embedding import Embeddings
    wordEmbeddings = Embeddings("test").getVectors()
    # We get data:
    from newssource.asa.asapreproc import AsaPreproc
    bash("rsync -avhuP hayj@titanv.lri.fr:~/NoSave/Data/Asa/asaminbis/asamin-new*-maxads10-maxdocperad10 ~/tmp")
    asap = AsaPreproc\
    (
        sortedGlob(tmpDir() + "/asamin-newtrain-maxads10-maxdocperad10/*.bz2"),
        sortedGlob(tmpDir() + "/asamin-newval-maxads10-maxdocperad10/*.bz2"),
        dataCol="textSentences", doFlattenSentences=True, doLower=True, filterNonWordOrPunct=True,
        punct={".", ","}, minTokensLength=3, minVocDF=1, wordEmbeddings=wordEmbeddings, batchSize=32,
        docLength=300, encoding="onehot",
    )
    x, y = asap.getTrain()
    valData = asap.getValidation()
    # We build a model:
    (model, kwargs, script) = buildRNN\
    (
        docLength=asap.getDocLength(),
        vocSize=len(asap.getVocIndex()),
        embeddingMatrix=asap.getEmbeddingMatrix(),
        nbClasses=len(asap.getEncodedAds()),
        isEmbeddingsTrainable=False,
        denseUnits=100,
        rnnUnits=128,
        denseDropout=0.2,
        isBidirectional=True,
        isCuDNN=False,
        rnnType='LSTM',
        addAttention=True,
        bnAfterEmbedding=True,
    )
    # We compile it:
    opt = optimizers.Adam(clipnorm=1.0)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy', 'top_k_categorical_accuracy'])
    # We fit the model:
    history = model.fit(x, y, validation_data=valData, batch_size=32, epochs=1, verbose=1)
    # We save the model:
    directoryPath = saveModel(model, tmpDir(), kwargs=kwargs, script=script, makeSubDir=True, extraInfos={"embeddings": "test"})
    # We print the content of directoryPath:
    bash("lsa " + directoryPath + "/*")
    # We load the model:
    (model, kwargs, script) = buildRNN\
    (
        docLength=asap.getDocLength(),
        vocSize=len(asap.getVocIndex()),
        embeddingMatrix=asap.getEmbeddingMatrix(),
        nbClasses=len(asap.getEncodedAds()),
        isEmbeddingsTrainable=False,
        denseUnits=100,
        rnnUnits=128,
        denseDropout=0.2,
        isBidirectional=True,
        isCuDNN=False,
        rnnType='LSTM',
        addAttention=True,
        bnAfterEmbedding=True,
    )
    def valAcc():
        c = 0
        i = 0
        for p in model.predict(valData[0]):
            if np.argmax(p) == np.argmax(valData[1][i]):
                c += 1
            i += 1
        print("val acc: " + str(c / len(valData[0])))
    valAcc()
    model.load_weights(directoryPath + "/weights.h5")
    valAcc()

if __name__ == '__main__':
    test1()