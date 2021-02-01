import numpy as np
from tars.layer.abs_layer import *
import operator
from functools import reduce


class Flatten(ABSLayer):

    def __init__(self, backward_layer):
        super(Flatten, self).__init__(backward_layer)


    def resetState(self):
        pass


    def layerName(self):

        return self.__class__.__name__


    def test(self, input):

        return self.forwardCore(input)


    def forward(self, input):

        return self.forwardCore(input)


    def forwardCore(self, input):

        return input.reshape(input.shape[0], -1)


    def backward(self, error, target):

        batches = error.shape[0]
        return error.reshape((batches, ) + self.input_shape)


    def outputShape(self):
        return (reduce(operator.mul, self.input_shape), )
