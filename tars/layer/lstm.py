import numpy as np
from tars.layer.abs_layer import *
from tars.weight_init.weight_init import *
from tars.activation.creator import *

class LSTM(ABSLayer):

    def __init__(self, units, activation, recurrent_activation, weight_init, backward_layer, gradient, unroll, stateful):
        super(LSTM, self).__init__(backward_layer)

        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.unroll = unroll
        self.stateful = stateful

        kernel_count = 1 if self.unroll is False else self.input_shape[-2]

        self.units = units
        self.allUnits = units * 4

        self.weight_i_list = self.createWeightList(weight_init, (self.input_shape[-1], self.allUnits), kernel_count)
        self.weight_h_list = self.createWeightList(weight_init, (self.units, self.allUnits), kernel_count)
        self.bias_list = [np.zeros((self.allUnits)).copy()] * kernel_count

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
        pass


    def forward(self, input):

        self.last_input = input

        return self.forwardCore(input)


    def forwardCore(self, input):

        (batche, sequence_length, vocab_size) = input.shape

        self.h_list = []
        self.cs_list = []

        if self.stateful is False or self.h_next is None:
            self.h_next = np.zeros((batche, self.units))
            self.cs_next = np.zeros((batche, self.units))

        h_init = self.h_next
        cs_init = self.cs_next

        self.recur_act_func = [createActivation(self.recurrent_activation) for i in range(sequence_length)]
        self.g_act_func = [createActivation(self.activation) for i in range(sequence_length)]
        self.output_act_func = [createActivation(self.activation) for i in range(sequence_length)]

        for s in range(sequence_length):
            i = input[:,s,:]

            kernel_index = 0 if self.unroll is False else s

            matmul_i = np.matmul(i, self.weight_i_list[kernel_index])
            matmul_h = np.matmul(self.h_next, self.weight_h_list[kernel_index])

            matmul_calc = matmul_i + matmul_h + self.bias_list[kernel_index]

            recur_calc = self.recur_act_func[s].forward(matmul_calc[:,:self.units * 3]).reshape(matmul_calc.shape[:-1] + (3, -1))

            g_value = self.g_act_func[s].forward(matmul_calc[:,self.units * 3:])
            i_value = recur_calc[:,0]
            f_value = recur_calc[:,1]
            o_value = recur_calc[:,2]

            self.cs_next = (f_value * self.cs_next) + (i_value * g_value)
            self.cs_list.append(self.cs_next)

            self.h_next = o_value * self.output_act_func[s].forward(self.cs_next)
            self.h_list.append(self.h_next)

        output = np.swapaxes(np.array(self.h_list), 1, 0)

        self.recur_act_func.insert(0, createActivation(self.recurrent_activation))
        self.g_act_func.insert(0, createActivation(self.activation))
        self.output_act_func.insert(0, createActivation(self.activation))

        self.h_list.insert(0, h_init)
        self.cs_list.insert(0, cs_init)

        return output


    def backward(self, error):

        return error


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

        return self.units


    def outputShape(self):

        (sequence_length, vocab_size) = self.input_shape

        return (sequence_length, self.getUnits())
