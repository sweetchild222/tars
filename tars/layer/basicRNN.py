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
        self.test_proceed = 0

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
        self.test_proceed = 0


    def test(self, input):

        (sequence_length, vocab_size) = self.input_shape

        (batche, cur_sequence_length, cur_vocab_size) = input.shape

        h_test_list = []

        if (self.stateful is False and self.test_proceed == 0) or self.h_test is None:
            self.h_test = np.zeros((batche, self.getUnits()))

        activation = createActivation(self.activation)

        for s in range(cur_sequence_length):

            kernel_index = 0 if self.unroll is False else self.test_proceed

            weight_i = self.weight_i_list[kernel_index]
            weight_h = self.weight_h_list[kernel_index]
            bias = self.bias_list[kernel_index]

            matmul_i = np.matmul(input[:,s,:], weight_i)
            matmul_h = np.matmul(self.h_test, weight_h)

            self.h_test = activation.forward(matmul_i + matmul_h + bias)
            h_test_list.append(self.h_test)

            self.test_proceed = (self.test_proceed + 1) % sequence_length

            if self.stateful is False and self.test_proceed == 0:
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

        self.act_func = [createActivation(self.activation) for i in range(sequence_length)]

        for s in range(sequence_length):

            kernel_index = 0 if self.unroll is False else s

            weight_i = self.weight_i_list[kernel_index]
            weight_h = self.weight_h_list[kernel_index]
            bias = self.bias_list[kernel_index]

            matmul_i = np.matmul(input[:,s,:], weight_i)
            matmul_h = np.matmul(self.h_next, weight_h)

            self.h_next = self.act_func[s].forward(matmul_i + matmul_h + bias)

            self.h_list.append(self.h_next)

        output = np.swapaxes(np.array(self.h_list), 1, 0)

        self.h_list.insert(0, h_init)

        return output


    def backward(self, error):

        (batche, sequence_length, units) = error.shape

        d_h_prev = np.zeros((batche, self.getUnits()))

        wi_delta_list = []
        wh_delta_list = []
        b_delta_list = []
        back_layer_error_list = []

        for s in range(sequence_length - 1, -1, -1):

            kernel_index = 0 if self.unroll is False else s

            weight_h = self.weight_h_list[kernel_index]
            weight_i = self.weight_i_list[kernel_index]

            h_prev = self.h_list[s]

            err = error[:, s,:] + d_h_prev

            d_h_raw = self.act_func[s].backward(err)
            d_h_prev = np.matmul(d_h_raw, weight_h.T)

            back_error = np.matmul(err, weight_i.T)
            back_layer_error_list.append(back_error)

            d_h_raw_expand = np.expand_dims(d_h_raw, axis=1)
            last_i = np.expand_dims(self.last_input[:, s,:], axis=-1)
            h_prev = np.expand_dims(h_prev, axis=-1)

            wi_delta = np.matmul(last_i, d_h_raw_expand)
            wh_delta = np.matmul(h_prev, d_h_raw_expand)

            wi_delta_list.append(wi_delta)
            wh_delta_list.append(wh_delta)
            b_delta_list.append(d_h_raw)

        self.gradientUpdate(np.array(wi_delta_list), np.array(wh_delta_list), np.array(b_delta_list))

        back_layer_error = np.array(back_layer_error_list)
        back_layer_error = back_layer_error.swapaxes(1, 0)

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
