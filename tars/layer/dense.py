import numpy as np
from tars.layer.abs_layer import *
from tars.activation.creator import *
from tars.weight_init.weight_init import *


class Dense(ABSLayer):

    def __init__(self, units, activation, weight_init, backward_layer, gradient):
        super(Dense, self).__init__(backward_layer)

        self.activation = createActivation(activation)

        self.weight = createWeight(weight_init, self.input_shape[-1], units, (self.input_shape[-1], units))
        self.bias = np.zeros((units))

        self.gradient = gradient
        self.gradient.bind([self.weight, self.bias])


    def resetState(self):
        pass


    def layerName(self):

        activationName = self.activation.layerName()
        layerName = self.__class__.__name__
        layerName += (' (' + activationName + ')')

        return layerName


    def test(self, input):

        output = self.forwardCore(input)

        return self.activation.test(output)


    def forward(self, input):

        self.last_input = input

        output = self.forwardCore(input)

        return self.activation.forward(output)


    def forwardCore(self, input):

        return np.matmul(input, self.weight) + self.bias


    def backward(self, error, target):

        error = self.activation.backward(error, target)

        i = np.expand_dims(self.last_input, axis=-1)
        err = error.reshape(error.shape[:-1] + (1, error.shape[-1]))

        w_delta = np.matmul(i, err)
        b_delta = error

        back_layer_error =  np.matmul(error, self.weight.T)

        w_delta = w_delta.reshape((-1, ) + w_delta.shape[-2:])
        b_delta = b_delta.reshape((-1, ) + b_delta.shape[-1:])

        self.gradient.update([w_delta, b_delta])

        return back_layer_error


    def outputShape(self):

        units = self.weight.shape[-1]
        return (units, )
