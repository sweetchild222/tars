import numpy as np
from tars.layer.abs_layer import *


class MaxPooling(ABSLayer):

    def __init__(self, pool_size, strides, backward_layer):
        super(MaxPooling, self).__init__(backward_layer)

        self.pool_size = pool_size
        self.strides = pool_size if strides == None else strides


    def resetState(self):
        pass


    def layerName(self):

        return self.__class__.__name__


    def test(self, input):

        return self.forwardCore(input)


    def forward(self, input):

        self.last_input = input

        return self.forwardCore(input)


    def forwardCore(self, input):

        (batches, input_height, input_width, input_colors) = input.shape
        (pool_height, pool_width) = self.pool_size
        (stride_y, stride_x) = self.strides

        output = np.zeros((batches, ) + self.outputShape())

        input_y = out_y = 0
        while (input_y + pool_height) <= input_height:
            input_x = out_x = 0
            while (input_x + pool_width) <= input_width:

                i = input[:, input_y:input_y + pool_height, input_x:input_x + pool_width, :]
                max_pool = np.max(i.reshape((batches, -1, input_colors)), axis=1)
                output[:, out_y, out_x, :] = max_pool

                input_x += stride_x
                out_x += 1

            input_y += stride_y
            out_y += 1

        return output


    def unravel_indices(self, input):

        indices = np.nanargmax(input)
        indices = np.unravel_index(indices, input.shape)

        return indices


    def backward(self, error, y):

        batches = error.shape[0]
        (input_height, input_width, input_colors) = self.input_shape
        (pool_height, pool_width) = self.pool_size
        (stride_y, stride_x) = self.strides

        back_layer_error = np.zeros(((batches, ) + self.input_shape))

        input_y = out_y = 0
        while (input_y + pool_height) <= input_height:
            input_x = out_x = 0
            while (input_x + pool_width) <= input_width:

                input = self.last_input[:, input_y:input_y + pool_height, input_x:input_x + pool_width, :]
                input = input.reshape((batches, -1, input_colors))

                max_indices = np.nanargmax(input, axis=1)

                unravel_indices = np.unravel_index(max_indices.reshape(-1), self.pool_size)
                unravel_indices = np.swapaxes(np.array(unravel_indices), 0, 1)

                unravel_indices = unravel_indices.reshape((batches, input_colors, -1))

                for batche_index in range(len(unravel_indices)):
                    for color_index in range(len(unravel_indices[batche_index])):
                        indices = unravel_indices[batche_index][color_index]
                        back_layer_error[batche_index, input_y + unravel_indices[:][:][0], input_x + unravel_indices[:][:][1], color_index] = error[batche_index, out_y, out_x, color_index]

                input_x += stride_x
                out_x += 1

            input_y += stride_y
            out_y += 1

        return back_layer_error


    def outputShape(self):

        calc_shape = ((self.input_shape[0] // self.strides[0]), (self.input_shape[1] // self.strides[1]))

        colors = self.input_shape[2]

        return calc_shape + (colors,)
