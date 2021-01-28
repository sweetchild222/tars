from tars.activation.abs_activation import *
from tars.activation.leakyRelu import *
from tars.activation.relu import *
from tars.activation.elu import *
from tars.activation.leakyRelu import *
from tars.activation.sigmoid import *
from tars.activation.softmax import *
from tars.activation.tanh import *
from tars.activation.linear import *


def createActivation(activation):

    type = activation['type']
    parameter = activation['parameter'] if 'parameter' in activation else {}

    typeClass = {'softmax':Softmax, 'relu':Relu, 'tanh':Tanh, 'leakyRelu':LeakyRelu, 'sigmoid':Sigmoid, 'elu':ELU, 'linear':Linear}

    return typeClass[type](**parameter)
