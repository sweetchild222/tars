import numpy as np
from abc import *
from tars.activation.abs_activation import *


class ELU(ABSActivation):

    def __init__(self, alpha):
        super(ELU, self).__init__()
        self.alpha = alpha


    def layerName(self):

        return self.__class__.__name__


    def test(self, input):

        return self.forwardCore(input)


    def forward(self, input):

        self.last_input = input

        output = self.forwardCore(input)

        self.last_output = output

        return output


    def forwardCore(self, input):

        return np.where(input > 0, input, (np.exp(input) - 1) * self.alpha)


    def backward(self, error):

        return error * np.where(self.last_input > 0, 1, (self.last_output + self.alpha))
