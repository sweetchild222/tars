from tars.gradient.adam import *
from tars.gradient.rms_prop import *
from tars.gradient.sgd import *


def createGradient(gradient):

    type = gradient['type']
    parameter = gradient['parameter']
    typeClass = {'rmsProp':RMSprop, 'adam':Adam, 'sgd':SGD}

    return typeClass[type](**parameter)
