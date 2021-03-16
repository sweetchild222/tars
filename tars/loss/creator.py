from tars.loss.softmax import *
from tars.loss.sigmoid import *


def createLoss(loss):

    type = loss['type']
    parameter = {}
    typeClass = {'softmax':Softmax, 'sigmoid':Sigmoid}

    return typeClass[type](**parameter)
