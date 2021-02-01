import numpy as np
from abc import *
from tars.activation.abs_activation import *


class Softmax(ABSActivation):

    def __init__(self):
        super(Softmax, self).__init__()


    def layerName(self):

        return self.__class__.__name__


    def test(self, input):

        return self.forwardCore(input)


    def forward(self, input):

        return self.forwardCore(input)


    def forwardCore(self, input):

        output = np.exp(input)
        sum = np.sum(output, axis=output.ndim - 1, keepdims = True)

        return output / sum


    def backward(self, error, target):

        return (error - target)
