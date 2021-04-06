import numpy as np
from tars.layer.abs_layer import *
from tars.weight_init.weight_init import *
from tars.activation.creator import *


class Convolution(ABSLayer):

    def __init__(self, filters, kernel_size, strides, padding, activation, weight_init, backward_layer, gradient):
        super(Convolution, self).__init__(backward_layer)

        self.strides = strides
        self.padding_size = self.paddingSize(kernel_size[0], kernel_size[1]) if padding else (0,0)

        self.weight = self.createWeight(weight_init, (kernel_size[0], kernel_size[1], self.input_shape[-1], filters))
        self.bias = np.zeros((filters))

        self.gradient = gradient
        self.gradient.bind([self.weight, self.bias])

        self.activation = createActivation(activation)


    def layerName(self):

        activationName = self.activation.layerName()
        layerName = self.__class__.__name__
        layerName += (' (' + activationName + ')')

        return layerName


    def resetState(self):
        pass


    def createWeight(self, weight_random, size):

        (kernel_height, kernel_width, colors, filters) = size

        receptive_field = kernel_height * kernel_width

        fab_in = receptive_field * colors
        fab_out = receptive_field * filters

        return createWeight(weight_random, fab_in, fab_out, size)


    def appendPadding(self, input):

        size = self.padding_size

        return np.pad(input, ((0, 0), (size[0], size[0]), (size[1], size[1]), (0, 0)), 'constant', constant_values=0)


    def test(self, input):

        input = self.appendPadding(input)

        output = self.forwardCore(input)

        return self.activation.test(output)


    def forward(self, input):

        input = self.appendPadding(input)

        self.last_input = input

        output = self.forwardCore(input)

        return self.activation.forward(output)


    def forwardCore(self, input):

        (kernel_height, kernel_width, colors, filters) = self.weight.shape
        (batches, input_height, input_width, input_colors) = input.shape
        (stride_y, stride_x) = self.strides

        output = np.zeros((batches, ) + self.outputShape())

        input_y = out_y = 0
        while (input_y + kernel_height) <= input_height:
            input_x = out_x = 0
            while (input_x + kernel_width) <= input_width:

                i = input[:, input_y:input_y + kernel_height, input_x:input_x + kernel_width, :]
                i = np.concatenate([i] * filters, axis=-1).reshape(i.shape + (filters, ))

                iw = i * self.weight
                iw = np.sum(iw.reshape((batches, -1, filters)), axis=1)

                output[:, out_y, out_x, :] =  iw + self.bias

                input_x += stride_x
                out_x += 1

            input_y += stride_y
            out_y += 1

        return output


    def backward(self, error):

        error = self.activation.backward(error)

        batches = error.shape[0]
        (kernel_height, kernel_width, colors, filters) = self.weight.shape
        (input_height, input_width, input_colors) = self.input_shape
        (stride_y, stride_x) = self.strides

        back_layer_error = np.zeros(((batches, ) + self.input_shape))
        batch_weight = np.array([self.weight] * batches)

        w_delta = np.zeros((batches, ) + self.weight.shape)
        b_delta = np.zeros((batches, ) + self.bias.shape)

        input_y = out_y = 0
        while (input_y + kernel_height) <= input_height:
            input_x = out_x = 0
            while (input_x + kernel_width) <= input_width:

                err = (error[:, out_y, out_x, :])[:, np.newaxis, np.newaxis, np.newaxis, :]
                i = self.last_input[:, input_y:input_y + kernel_height, input_x:input_x + kernel_width, :]

                w_delta += (err * np.expand_dims(i, axis=-1))
                b_delta += err.reshape((batches, filters))

                bw_err = batch_weight * err

                #shallow copy
                bl_err = back_layer_error[:, input_y:input_y + kernel_height, input_x:input_x + kernel_width, :]
                bl_err += np.sum(bw_err, axis=-1)

                input_x += stride_x
                out_x += 1

            input_y += stride_y
            out_y += 1

        self.gradient.update([w_delta, b_delta])

        return back_layer_error


    def paddingSize(self, kernel_height, kernel_width):

        return ((kernel_height - 1) // 2, (kernel_width - 1) // 2)


    def outputShape(self):

        (kernel_height, kernel_width, colors, filters) = self.weight.shape
        (stride_y, stride_x) = self.strides

        numerator_height = ((self.padding_size[0] * 2) - kernel_height) + self.input_shape[0]
        numerator_width = ((self.padding_size[1] * 2) - kernel_width) + self.input_shape[1]

        calc_shape = ((numerator_height // stride_y) + 1, (numerator_width // stride_x) + 1)

        return calc_shape + (filters,)
