from tars.layer.abs_layer import *


class Input(ABSLayer):

    def __init__(self, input_shape, backward_layer=None):
        super(Input, self).__init__(backward_layer)
        self.input_shape = input_shape


    def resetState(self):
        pass


    def layerName(self):

        return self.__class__.__name__


    def test(self, input):

        return self.forwardCore(input)


    def forward(self, input):

        return self.forwardCore(input)


    def forwardCore(self, input):

        return input


    def backward(self, error, target):

        return error


    def outputShape(self):

        return self.input_shape
