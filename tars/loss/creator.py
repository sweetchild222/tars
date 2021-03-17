from tars.loss.binary import *
from tars.loss.categorical import *
from tars.loss.mean_square import *


def createLoss(loss):

    type = loss['type']
    parameter = {}
    typeClass = {'binary':Binary, 'categorical':Categorical, 'meansquare':MeanSquare}

    return typeClass[type](**parameter)
