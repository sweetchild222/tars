import numpy as np
from abc import *
from tars.activation.abs_activation import *


class LeakyRelu(ABSActivation):

    def __init__(self, alpha):
        super(LeakyRelu, self).__init__()
        self.alpha = alpha


    def layerName(self):

        return self.__class__.__name__


    def predict(self, input):

        return self.forwardCore(input)


    def forward(self, input):

        self.last_input = input

        return self.forwardCore(input)


    def forwardCore(self, input):

        return np.where(input > 0, input, input * self.alpha)


    def backward(self, error, target):

        return error * np.where(self.last_input > 0, 1, self.alpha)
