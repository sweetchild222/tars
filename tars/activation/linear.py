import numpy as np
from abc import *
from tars.activation.abs_activation import *


class Linear(ABSActivation):

    def __init__(self):
        super(Linear, self).__init__()


    def layerName(self):

        return self.__class__.__name__


    def predict(self, input):

        return self.forwardCore(input)


    def forward(self, input):

        return self.forwardCore(input)


    def forwardCore(self, input):

        return input


    def backward(self, error, target):

        return error
