import numpy as np
from abc import *
from tars.activation.abs_activation import *


class Tanh(ABSActivation):

    def __init__(self):
        super(Tanh, self).__init__()


    def layerName(self):

        return self.__class__.__name__


    def test(self, input):

        return self.forwardCore(input)


    def forward(self, input):

        output = self.forwardCore(input)
        self.last_output = output

        return output


    def forwardCore(self, input):

        return np.tanh(input)


    def backward(self, error):

        return error * (1 - (self.last_output)**2)
