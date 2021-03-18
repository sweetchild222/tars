import numpy as np
from tars.loss.abs_loss import *


class Categorical(ABSLoss):

    def __init__(self):
        super(Categorical, self).__init__()


    def forward(self, input):
        output = np.exp(input)
        sum = np.sum(output, axis=-1, keepdims = True)

        return output / sum


    def backward(self, y, target):

        return (y - target)


    def loss(self, y, target):

        loss = target * np.log(y)
        sum = np.sum(loss, axis= -1)

        return -np.mean(sum)
