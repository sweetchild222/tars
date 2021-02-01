import numpy as np
from abc import *
from tars.activation.abs_activation import *


class Relu(ABSActivation):

    def __init__(self):
        super(Relu, self).__init__()


    def layerName(self):

        return self.__class__.__name__


    def test(self, input):

        return self.forwardCore(input)


    def forward(self, input):

        self.last_input = input

        return self.forwardCore(input)


    def forwardCore(self, input):

        return np.where(input > 0, input, 0)


    def backward(self, error, target):

        return np.where(self.last_input > 0, error, 0)
