import numpy as np

from tars.gradient.creator import *
from tars.layer.input import *
from tars.layer.convolution import *
from tars.layer.max_pooling import *
from tars.layer.flatten import *
from tars.layer.dense import *
from tars.layer.basicRNN import *
from tars.layer.lstm import *



def createLayers(layersTemplate, gradientTemplate):

    backward_layer = None
    head = None
    tail = None

    for layer in layersTemplate:

        parameter = layer['parameter'] if 'parameter' in layer else {}
        parameter['backward_layer'] = backward_layer

        layerClass = {'input':Input, 'convolution':Convolution, 'maxPooling':MaxPooling, 'flatten':Flatten, 'dense':Dense, 'basicRNN':BasicRNN, 'lstm':LSTM}
        gradientLayerClass = {'convolution':Convolution, 'dense':Dense, 'basicRNN':BasicRNN, 'lstm':LSTM}
        type = layer['type']

        if type in gradientLayerClass:
            gradient = createGradient(gradientTemplate)
            parameter['gradient'] = gradient

        backward_layer = layerClass[type](**parameter)

        if head == None:
            head = backward_layer

    tail = backward_layer

    return head, tail
