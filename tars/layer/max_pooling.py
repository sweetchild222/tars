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

        output = self.forwardCore(input)

        return output[0]


    def forward(self, input):

        output, self.max_indices_list = self.forwardCore(input)

        return output


    def forwardCore(self, input):

        (batches, input_height, input_width, input_colors) = input.shape
        (pool_height, pool_width) = self.pool_size
        (stride_y, stride_x) = self.strides

        (output_x, output_y, output_color)= self.outputShape()

        output = np.zeros((batches, ) + self.outputShape())

        indices_batch = np.array([[i] * input_colors for i in range(batches)]).reshape(-1)
        indices_color = np.array([[i] for i in range(input_colors)] * batches).reshape(-1)

        max_indices_list = []

        input_y = 0
        for y in range(output_y):
            input_x = 0
            for x in range(output_x):
                i = input[:, input_y:input_y + pool_height, input_x:input_x + pool_width, :]
                i = i.reshape((batches, -1, input_colors))

                max_indices = np.nanargmax(i, axis=1)
                max_indices_list.append(max_indices)

                max_input = i[indices_batch, max_indices.reshape(-1), indices_color]

                output[:, y, x, :] = max_input.reshape((batches, input_colors))

                input_x += stride_x
            input_y += stride_y

        return output, max_indices_list


    def backward(self, error):

        batches = error.shape[0]
        (input_height, input_width, input_colors) = self.input_shape
        (pool_height, pool_width) = self.pool_size
        (stride_y, stride_x) = self.strides
        (output_x, output_y, output_color)= self.outputShape()

        indices_batch = np.array([[i] * input_colors for i in range(batches)]).reshape(-1)
        indices_color = np.array([[i] for i in range(input_colors)] * batches).reshape(-1)

        back_layer_error = np.zeros(((batches, ) + self.input_shape))

        indices_index = 0

        input_y = 0
        for y in range(output_y):
            input_x = 0
            for x in range(output_x):
                max_indices = self.max_indices_list[indices_index]

                indices_index += 1

                unravel_index = np.unravel_index(max_indices, self.pool_size)

                indices_y = np.array(unravel_index[0] + input_y).reshape(-1)
                indices_x = np.array(unravel_index[1] + input_x).reshape(-1)

                back_layer_error[indices_batch, indices_y, indices_x, indices_color] = (error[:, y, x,:].reshape(-1))

                input_x += stride_x

            input_y += stride_y

        return back_layer_error


    def outputShape(self):

        numerator_height = (self.input_shape[0] - self.pool_size[0])
        numerator_width = (self.input_shape[1] - self.pool_size[1])

        calc_shape = ((numerator_height // self.strides[0]) + 1, (numerator_width // self.strides[1]) + 1)

        colors = self.input_shape[2]

        return calc_shape + (colors,)
