import numpy as np
from tars.layer.abs_layer import *
from tars.weight_init.weight_init import *
from tars.activation.creator import *


class GRU(ABSLayer):

    def __init__(self, units, activation, recurrent_activation, weight_init, backward_layer, gradient, unroll, stateful):
        super(GRU, self).__init__(backward_layer)

        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.unroll = unroll
        self.stateful = stateful

        self.sets_count = 3

        kernel_count = 1 if self.unroll is False else self.input_shape[-2]

        self.weight_i_list = self.createWeightList(weight_init, self.sets_count, (self.input_shape[-1], units), kernel_count)
        self.weight_h_list = self.createWeightList(weight_init, self.sets_count, (units, units), kernel_count)
        self.bias_list = self.createBiasList(self.sets_count, units, kernel_count)

        self.gradient = self.gradientBind(gradient, self.weight_i_list, self.weight_h_list, self.bias_list)

        self.h_test = None
        self.test_proceed = 0

        self.h_next = None


    def gradientBind(self, gradient, weight_i_list, weight_h_list, bias_list):

        bindList = []

        for w_i_sets in weight_i_list:
            for w_i in w_i_sets:
                bindList.append(w_i)

        for w_h_sets in weight_h_list:
            for w_h in w_h_sets:
                bindList.append(w_h)

        for bais_sets in bias_list:
            for b in bais_sets:
                bindList.append(w_h)

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


    def createBiasList(self, sets, units, kernel_count):

        list = []

        for i in range(kernel_count):
            bais_sets = []
            for s in range(sets):
                bias = np.zeros(units)
                bais_sets.append(bias)

            list.append(np.array(bais_sets))

        return list


    def layerName(self):

        activationName = createActivation(self.activation).layerName()
        layerName = self.__class__.__name__
        layerName += (' (' + activationName + ')')

        return layerName


    def resetState(self):

        self.h_test = None
        self.c_test = None
        self.test_proceed = 0

        self.h_next = None


    def test(self, input):

        (sequence_length, vocab_size) = self.input_shape

        (batche, cur_sequence_length, cur_vocab_size) = input.shape

        h_list = []

        if (self.stateful is False and self.test_proceed == 0) or self.h_test is None or self.c_test is None:
            self.h_test = np.zeros((batche, self.getUnits()))
            self.c_test = np.zeros((batche, self.getUnits()))

        recur_act_func = [createActivation(self.recurrent_activation) for i in range(cur_sequence_length)]
        g_act_func = [createActivation(self.activation) for i in range(cur_sequence_length)]
        output_act_func = [createActivation(self.activation) for i in range(cur_sequence_length)]

        for s in range(cur_sequence_length):

            kernel_index = 0 if self.unroll is False else self.test_proceed

            weight_i = self.weight_i_list[kernel_index]
            weight_h = self.weight_h_list[kernel_index]
            bias = self.bias_list[kernel_index]

            matmul_i = np.matmul(input[:,s,:], weight_i)
            matmul_h = np.matmul(self.h_test, weight_h)

            batch_bias = np.array([bias] * batche).reshape((self.sets_count, batche, -1))

            matmul_calc = matmul_i + matmul_h + batch_bias

            g_value = g_act_func[s].forward(matmul_calc[-1])

            recur_sets = recur_act_func[s].forward(matmul_calc[:-1])
            i_value = recur_sets[0]
            f_value = recur_sets[1]
            o_value = recur_sets[2]

            self.c_test = (f_value * self.c_test) + (i_value * g_value)

            z_value = output_act_func[s].forward(self.c_test)

            self.h_test = o_value * z_value
            h_list.append(self.h_test)

            self.test_proceed = (self.test_proceed + 1) % sequence_length

            if self.stateful is False and self.test_proceed == 0:
                self.h_test = np.zeros((batche, self.getUnits()))
                self.c_test = np.zeros((batche, self.getUnits()))

        return np.swapaxes(np.array(h_list), 1, 0)


    def forward(self, input):

        self.last_input = input

        return self.forwardCore(input)


    def forwardCore(self, input):

        (batche, sequence_length, vocab_size) = input.shape

        self.h_list = []
        self.recur_sets_list = []
        self.g_list = []

        if self.stateful is False or self.h_next is None:
            self.h_next = np.zeros((batche, self.getUnits()))

        h_init = self.h_next

        self.recur_act_func = [createActivation(self.recurrent_activation) for i in range(sequence_length)]
        self.g_act_func = [createActivation(self.activation) for i in range(sequence_length)]

        for s in range(sequence_length):

            kernel_index = 0 if self.unroll is False else s

            weight_i = self.weight_i_list[kernel_index]
            weight_h = self.weight_h_list[kernel_index]
            bias = self.bias_list[kernel_index]

            matmul_i = np.matmul(input[:,s,:], weight_i)
            matmul_h = np.matmul(self.h_next, weight_h)

            batch_bias = np.array([bias] * batche).reshape((self.sets_count, batche, -1))

            matmul_calc_rz = matmul_i[:-1] + matmul_h[:-1] + batch_bias[:-1]

            recur_sets = self.recur_act_func[s].forward(matmul_calc_rz)
            self.recur_sets_list.append(recur_sets)
            r_value = recur_sets[0]
            z_value = recur_sets[1]

            matmul_calc_g = matmul_i[-1] + (matmul_h[-1] * r_value) + batch_bias[-1]

            g_value = self.g_act_func[s].forward(matmul_calc_g)
            self.g_list.append(g_value)

            self.h_next = z_value * self.h_next + ((1-z_value) * g_value)
            self.h_list.append(self.h_next)

        output = np.swapaxes(np.array(self.h_list), 1, 0)

        self.h_list.insert(0, h_init)

        print(output.shape)

        return output


    def backward(self, error):

        (batche, sequence_length, units) = error.shape

        d_h_prev = np.zeros((batche, self.getUnits()))
        d_c_prev = np.zeros((batche, self.getUnits()))

        wi_delta_list = []
        wh_delta_list = []
        b_delta_list = []
        back_layer_error_list = []

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

            d_d_recur_sets = self.recur_act_func[s].backward(np.array([d_i, d_f, d_o]))
            d_d_g = self.g_act_func[s].backward(d_g)
            d_d_g = np.expand_dims(d_d_g, axis=0)

            d_h_raw = np.concatenate((d_d_recur_sets, d_d_g), axis=0)
            d_h_raw_expand = np.expand_dims(d_h_raw, axis=-2)

            last_i = np.expand_dims(self.last_input[:, s,:], axis=-1)
            h_prev = np.expand_dims(self.h_list[s], axis=-1)

            wi_delta = np.matmul(last_i, d_h_raw_expand)
            wh_delta = np.matmul(h_prev, d_h_raw_expand)

            wi_delta_list.append(wi_delta)
            wh_delta_list.append(wh_delta)
            b_delta_list.append(d_h_raw_expand)

            weight_h = self.weight_h_list[kernel_index]
            weight_i = self.weight_i_list[kernel_index]

            d_h_prev = np.matmul(d_h_raw, weight_h.swapaxes(-2, -1))
            d_h_prev = np.sum(d_h_prev, axis=0)

            back_error = np.matmul(d_h_raw, weight_i.swapaxes(-2, -1))
            back_error = np.sum(back_error, axis=0)

            back_layer_error_list.append(back_error)

        self.gradientUpdate(np.array(wi_delta_list), np.array(wh_delta_list), np.array(b_delta_list))

        back_layer_error = np.array(back_layer_error_list)
        back_layer_error = back_layer_error.swapaxes(1, 0)

        return back_layer_error


    def gradientUpdate(self, wi_delta_list, wh_delta_list, b_delta_list):

        wi_list = []
        wh_list = []
        b_list = []

        for i in range(self.sets_count):
            wi_delta = wi_delta_list[:, i]
            wh_delta = wh_delta_list[:, i]
            b_delta = b_delta_list[:, i]

            if self.unroll is True:
                wi_list += [wi for wi in wi_delta]
                wh_list += [wh for wh in wh_delta]
                b_list += [b for b in b_delta]
            else:
                wi_list.append(wi_delta.reshape((-1, ) + wi_delta.shape[-2:]))
                wh_list.append(wh_delta.reshape((-1, ) + wh_delta.shape[-2:]))
                b_list.append(b_delta.reshape((-1, ) + b_delta.shape[-2:]))

        self.gradient.update(wi_list + wh_list + b_list)


    def getUnits(self):

        return self.weight_i_list[0].shape[-1]


    def outputShape(self):

        (sequence_length, vocab_size) = self.input_shape

        return (sequence_length, self.getUnits())
