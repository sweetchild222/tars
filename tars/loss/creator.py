from tars.loss.binary import *
from tars.loss.categorical import *


def createLoss(loss):

    type = loss['type']
    parameter = {}
    typeClass = {'binary':Binary, 'categorical':Categorical}

    return typeClass[type](**parameter)
