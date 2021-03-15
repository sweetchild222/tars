import numpy as np
from abc import *
from tars.activation.abs_activation import *


class Sigmoid(ABSActivation):

    def __init__(self):
        super(Sigmoid, self).__init__()


    def layerName(self):

        return self.__class__.__name__


    def test(self, intput):

        return self.forwardCore(intput)


    def forward(self, intput):

        output = self.forwardCore(intput)
        self.last_output = output

        return output


    def forwardCore(self, input):

        return 1 / (1 + np.exp(-input))


    def backward(self, error):

        return error * self.last_output * (1.0 - self.last_output)
