import numpy as np
from tars.layer.abs_layer import *
from tars.weight_init.weight_init import *
from tars.activation.creator import *


class BasicRNN(ABSLayer):

    def __init__(self, units, activation, weight_init, backward_layer, gradient, unroll, stateful):
        super(BasicRNN, self).__init__(backward_layer)

        self.activation = activation
        self.unroll = unroll
        self.stateful = stateful

        kernel_count = 1 if self.unroll is False else self.input_shape[-2]

        self.weight_i_list = self.createWeightList(weight_init, (self.input_shape[-1], units), kernel_count)
        self.weight_h_list = self.createWeightList(weight_init, (units, units), kernel_count)
        self.bias_list = [np.zeros((units)) for i in range(kernel_count)]

        self.gradient = self.gradientBind(gradient, self.weight_i_list, self.weight_h_list, self.bias_list)

        self.h_test = None
        self.h_test_index = 0

        self.h_next = None


    def gradientBind(self, gradient, weight_i_list, weight_h_list, bias_list):

        bindList = [w_i for w_i in weight_i_list]
        bindList += [w_h for w_h in weight_h_list]
        bindList += [b for b in bias_list]

        gradient.bind(bindList)

        return gradient


    def createWeightList(self, weight_init, size, kernel_count):

        list = []

        for i in range(kernel_count):
            weight = createWeight(weight_init, size[0], size[1], size)
            list.append(weight)

        return list


    def layerName(self):

        activationName = createActivation(self.activation).layerName()
        layerName = self.__class__.__name__
        layerName += (' (' + activationName + ')')

        return layerName


    def resetState(self):

        self.h_next = None
        self.h_test = None
        self.h_test_index = 0


    def test(self, input):

        (sequence_length, vocab_size) = self.input_shape

        (batche, cur_sequence_length, cur_vocab_size) = input.shape

        h_test_list = []

        if (self.stateful is False and self.h_test_index == 0) or self.h_test is None:
            self.h_test = np.zeros((batche, self.getUnits()))

        activation = createActivation(self.activation)

        for s in range(cur_sequence_length):

            i = input[:,s,:]

            kernel_index = 0 if self.unroll is False else self.h_test_index

            matmul_i = np.matmul(i, self.weight_i_list[kernel_index])
            matmul_h = np.matmul(self.h_test, self.weight_h_list[kernel_index])

            self.h_test = activation.forward(matmul_i + matmul_h + self.bias_list[kernel_index])
            h_test_list.append(self.h_test)

            self.h_test_index = (self.h_test_index + 1) % sequence_length

            if self.stateful is False and self.h_test_index == 0:
                self.h_test = np.zeros((batche, self.getUnits()))

        return np.swapaxes(np.array(h_test_list), 1, 0)


    def forward(self, input):

        self.last_input = input

        return self.forwardCore(input)


    def forwardCore(self, input):

        (batche, sequence_length, vocab_size) = input.shape

        self.h_list = []

        if self.stateful is False or self.h_next is None:
            self.h_next = np.zeros((batche, self.getUnits()))

        h_init = self.h_next

        self.activationList = [createActivation(self.activation) for i in range(sequence_length)]

        for s in range(sequence_length):
            i = input[:,s,:]

            kernel_index = 0 if self.unroll is False else s

            matmul_i = np.matmul(i, self.weight_i_list[kernel_index])
            matmul_h = np.matmul(self.h_next, self.weight_h_list[kernel_index])

            activation = self.activationList[s]

            self.h_next = activation.forward(matmul_i + matmul_h + self.bias_list[kernel_index])

            self.h_list.append(self.h_next)

        output = np.swapaxes(np.array(self.h_list), 1, 0)

        self.activationList.insert(0, createActivation(self.activation))
        self.h_list.insert(0, h_init)

        return output


    def backward(self, error, target):

        (batche, sequence_length, units) = error.shape

        d_h_prev = np.zeros((batche, self.getUnits()))

        wi_delta_list = []
        wh_delta_list = []
        b_delta_list = []

        for s in range(sequence_length - 1, -1, -1):

            kernel_index = 0 if self.unroll is False else s

            activation = self.activationList[s + 1]

            h_prev = self.h_list[s]

            i = self.last_input[:, s,:]
            i = np.expand_dims(i, axis=-1)

            err = error[:, s,:]

            back_h_error = err + d_h_prev

            d_h_raw = activation.backward(back_h_error, target)
            d_h_prev = np.matmul(d_h_raw, self.weight_h_list[kernel_index].T)

            wi_delta_list.append(np.matmul(i, np.expand_dims(d_h_raw, axis=1)))
            wh_delta_list.append(np.matmul(np.expand_dims(h_prev, axis=-1), np.expand_dims(d_h_raw, axis=1)))
            b_delta_list.append(d_h_raw)

        self.gradientUpdate(np.array(wi_delta_list), np.array(wh_delta_list), np.array(b_delta_list))

        back_layer_error = np.matmul(error, self.weight_i_list[kernel_index].T)

        return back_layer_error


    def gradientUpdate(self, wi_delta_list, wh_delta_list, b_delta_list):

        updateList = []

        if self.unroll is True:
            updateList += [wi_delta for wi_delta in wi_delta_list]
            updateList += [wh_delta for wh_delta in wh_delta_list]
            updateList += [b_delta for b_delta in b_delta_list]
        else:
            updateList.append(wi_delta_list.reshape((-1, ) + wi_delta_list.shape[-2:]))
            updateList.append(wh_delta_list.reshape((-1, ) + wh_delta_list.shape[-2:]))
            updateList.append(b_delta_list.reshape((-1, ) + b_delta_list.shape[-1:]))

        self.gradient.update(updateList)


    def getUnits(self):

        return self.weight_i_list[0].shape[-1]


    def outputShape(self):

        (sequence_length, vocab_size) = self.input_shape

        return (sequence_length, self.getUnits())
