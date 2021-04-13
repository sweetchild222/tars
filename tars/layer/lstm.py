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

        self.weight_i_list = self.createWeightList(weight_init, 4, (self.input_shape[-1], units), kernel_count)
        self.weight_h_list = self.createWeightList(weight_init, 4, (units, units), kernel_count)
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


    def createWeightList(self, weight_init, sets, size, kernel_count):

        list = []

        for i in range(kernel_count):
            weight_sets = []
            for s in range(sets):
                weight = createWeight(weight_init, size[0], size[1], size)
                weight_sets.append(weight)

            list.append(np.array(weight_sets))

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

        #aa = np.array([[[1,1,1],[1,1,1]], [[2,2,2],[2,2,2]], [[3,3,3],[3,3,3]]])
        #print(aa[-1].shape)

        (batche, sequence_length, vocab_size) = input.shape

        self.h_list = []
        self.c_list = []
        self.recur_sets_list = []
        self.g_list = []
        self.z_list = []

        if self.stateful is False or self.h_next is None:
            self.h_next = np.zeros((batche, self.getUnits()))
            self.c_next = np.zeros((batche, self.getUnits()))

        h_init = self.h_next
        c_init = self.c_next

        self.recur_act_func = [createActivation(self.recurrent_activation) for i in range(sequence_length)]
        self.g_act_func = [createActivation(self.activation) for i in range(sequence_length)]
        self.output_act_func = [createActivation(self.activation) for i in range(sequence_length)]

        for s in range(sequence_length):

            kernel_index = 0 if self.unroll is False else s

            weight_i = self.weight_i_list[kernel_index]
            weight_h = self.weight_h_list[kernel_index]
            bias = self.bias_list[kernel_index]

            matmul_i = np.matmul(input[:,s,:], weight_i)
            matmul_h = np.matmul(self.h_next, weight_h)

            matmul_calc = matmul_i + matmul_h + bias

            g_value = self.g_act_func[s].forward(matmul_calc[-1])
            self.g_list.append(g_value)

            recur_sets = self.recur_act_func[s].forward(matmul_calc[:-1])
            self.recur_sets_list.append(recur_sets)
            i_value = recur_sets[0]
            f_value = recur_sets[1]
            o_value = recur_sets[2]

            self.c_next = (f_value * self.c_next) + (i_value * g_value)
            self.c_list.append(self.c_next)

            z_value = self.output_act_func[s].forward(self.c_next)
            self.z_list.append(z_value)

            self.h_next = o_value * z_value
            self.h_list.append(self.h_next)

        output = np.swapaxes(np.array(self.h_list), 1, 0)

        self.h_list.insert(0, h_init)
        self.c_list.insert(0, c_init)

        return output


    def backward(self, error):

        (batche, sequence_length, units) = error.shape

        d_h_prev = np.zeros((batche, self.getUnits()))
        d_c_prev = np.zeros((batche, self.getUnits()))

        wi_delta_list = []
        wh_delta_list = []
        b_delta_list = []

        for s in range(sequence_length - 1, -1, -1):

            kernel_index = 0 if self.unroll is False else s

            g_value = self.g_list[s]

            recur_sets = self.recur_sets_list[s]
            i_value = recur_sets[0]
            f_value = recur_sets[1]
            o_value = recur_sets[2]

            err = error[:, s,:] + d_h_prev

            d_c = d_c_prev + self.output_act_func[s].backward(err) * o_value
            d_c_prev = d_c * f_value

            d_g = d_c * i_value
            d_i = d_c * g_value
            d_f = d_c * self.c_list[s]
            d_o = err * self.z_list[s]

            d_recur_sets = self.recur_act_func[s].backward(np.array([d_i, d_f, d_o]))
            d_d_g = self.g_act_func[s].backward(d_g)

            print(d_recur_sets.shape)
            print(d_d_g.shape)




            #d_d_g =










        a = a / 0

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

        return self.weight_i_list[0].shape[-1]


    def outputShape(self):

        (sequence_length, vocab_size) = self.input_shape

        return (sequence_length, self.getUnits())
