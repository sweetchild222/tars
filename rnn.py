
from rnn_lib.loader import *
from rnn_lib.model_templates import *
from tars.tars import *
from util.util import *
import argparse
import datetime as dt


def print_arg(model, activation, weightInit, gradient, loss, epochs, batches, train_x_states_shape, train_t_states_shape, unroll, stateful):

    classes = train_t_states_shape[-1]

    arg = ['model', 'activation', 'weightInit', 'gradient', 'loss', 'classes', 'epochs', 'batches', 'train x states shape', 'train t states shape', 'unroll', 'stateful']
    values = [model, activation, weightInit, gradient, loss, classes, epochs, batches, train_x_states_shape, train_t_states_shape, unroll, stateful]
    table = {'Argument':arg, 'Values':values}
    print_table(table)


def print_performance(accuracy, span):

    performance = ['accuracy', 'train minute span']

    min_span = '{:.2f}'.format(span.total_seconds() / 60)
    values = [str(accuracy) + ' %', min_span]
    table = {'Performance':performance, 'Values':values}
    print_table(table)


def make_persons_list(test_x_states, oneHotmap):

    person_delim = oneHotmap[':']

    persons_list = []

    for text_x in test_x_states:
        persons = []
        for x in text_x:
            i = 0
            for s in x:
                if person_delim == np.argmax(s):
                    person = x[0:i + 1, :]
                    persons.append(person)
                    break
                i += 1
        persons_list.append(persons)

    return persons_list


def testCore(tars, test_x_list, test_t_list):

    p = 0

    test_y_list = []

    for test_x in test_x_list:
        test_t = test_t_list[p]

        input_sequence = test_x.shape[-2]
        sequence_length = test_t.shape[-2]

        test_x = np.expand_dims(test_x, axis=0)

        test_y = tars.test(test_x)

        #last output will be next test input
        y = test_y[:, -1, :][:, np.newaxis, :]

        test_y_phase = []
        test_y_phase.append(y.reshape(-1))

        x = y

        for i in range(sequence_length - input_sequence):
            y = tars.test(x)
            test_y_phase.append(y.reshape(-1))

            mapIndex = np.argmax(y.reshape(-1))
            x = np.eye(y.shape[-1])[mapIndex].reshape((1, 1, y.shape[-1]))

        test_y_list.append(test_y_phase)

        p += 1

    tars.resetState()

    return test_y_list


def covertString(data, mapToChar):

    string = [mapToChar[np.argmax(d)] for d in data]
    return ''.join(string).rstrip()


def test(tars, test_x_states, test_t_states, mapToChar, oneHotmap, stateful):

    persons_list = make_persons_list(test_x_states, oneHotmap)

    p = 0
    correct_count = 0
    total_count = 0

    for persons in persons_list:
        test_t = test_t_states[p]
        test_y = testCore(tars, persons, test_t)
        i = 0
        for y in test_y:
            person_name = covertString(persons[i], mapToChar)
            test = covertString(y, mapToChar)
            target = covertString(test_t[i][len(person_name) - 1: ], mapToChar)

            correct = (test == target)

            correct_count += int(correct == True)

            correct_string = 'O' if correct else 'X'

            print_data = {'batch - stateful': str(p) + ' - ' + str(i)} if stateful is True else {}

            print_data.update({'correct':correct_string, 'input':person_name, 'test':test, 'target':target})

            print_list('test', print_data)

            total_count += 1
            i += 1

        p += 1

    accuracy = float(correct_count / total_count) * 100

    return accuracy


def train(model, train_x_states, train_t_states, epochs, batches):

    start_time = dt.datetime.now()

    for epoch in range(epochs):

        train_x_states, train_t_states = model.shuffle(train_x_states, train_t_states)
        length = train_x_states.shape[0]
        phase = train_x_states.shape[1]

        loss = 0
        i = 0

        for b in range(0, length, batches):
            for s in range(phase):
                batch_x = train_x_states[b : b + batches, s]
                batch_t = train_t_states[b : b + batches, s]

                loss += model.train(batch_x, batch_t)
                i += 1

            model.resetState()

        print_table({'Epochs':[str(epoch + 1) +'/' + str(epochs)], 'Loss':[loss / i]})

    return (dt.datetime.now() - start_time)


def print_layer(layerList):

    layerNames = [layer.layerName() for layer in layerList]
    outputShapes = [layer.outputShape() for layer in layerList]

    print_table({'Layer':layerNames, 'Output Shape':outputShapes})


def create(modelType, activationType, weightInitType, input_shape, classes, gradientType, lossType, unroll, stateful):

    layersTemplate = createLayersTemplate(modelType, activationType, weightInitType, input_shape, classes, unroll, stateful)
    gradientTemplate = createGradientTemplate(gradientType)
    lossTemplate = createLossTemplate(lossType)

    tars = Tars(layersTemplate, gradientTemplate, lossTemplate)
    layerList = tars.build()
    print_layer(layerList)

    return tars


def main(modelType, activationType, weightInitType, gradientType, lossType, epochs, batches, unroll, stateful):

    train_x_states, train_t_states, oneHotmap, mapToChar = loadDataSet(stateful)

    print_arg(modelType, activationType, weightInitType, gradientType, lossType, epochs, batches, train_x_states.shape, train_t_states.shape, unroll, stateful)

    tars = create(modelType, activationType, weightInitType, train_x_states.shape[-2:], train_t_states.shape[-1], gradientType, lossType, unroll, stateful)

    train_span = train(tars, train_x_states, train_t_states, epochs, batches)

    accuracy = test(tars, train_x_states, train_t_states, mapToChar, oneHotmap, stateful)

    print_performance(accuracy, train_span)


def parse_arg():

    parser = argparse.ArgumentParser(prog='RNN')
    parser.add_argument('-m', dest='modelType', type=str, default='light', choices=['light', 'complex'], help='sample model type (default:light)')
    parser.add_argument('-g', dest='gradientType', type=str, default='rmsProp', choices=['adam', 'sgd', 'rmsProp'], help='sample gradient type (default: rmsProp)')
    parser.add_argument('-l', dest='lossType', type=str, default='softmax', choices=['softmax', 'sigmoid'], help='loss type (default: softmax)')
    parser.add_argument('-a', dest='activationType', type=str, default='tanh', choices=['linear', 'relu', 'elu', 'leakyRelu', 'sigmoid', 'tanh'], help='sample activation type (default: relu)')
    parser.add_argument('-w', dest='weightInitType', type=str, default='he_normal', choices=['lecun_uniform', 'glorot_uniform', 'he_uniform', 'lecun_normal', 'glorot_normal', 'he_normal'], help='weight initial type (default: he_normal)')
    parser.add_argument('-e', dest='epochs', type=int, default=200, help='epochs (default: 200)')
    parser.add_argument('-b', dest='batches', type=int, default=30, help='batches (default: 30)')
    parser.add_argument('--f', dest='stateful', action='store_true', help='stateful (default: False)')
    parser.add_argument('--u', dest='unroll', action='store_true', help='unroll (default: False)')

    args = parser.parse_args()

    if args.epochs < 1:
        print('RNN: error: argument -e: invalid value: ', str(args.epochs), ' (value must be over 0)')
        return None

    return args


if __name__ == "__main__":

    args = parse_arg()

    if args != None:
        main(args.modelType, args.activationType, args.weightInitType, args.gradientType, args.lossType, args.epochs, args.batches, args.unroll, args.stateful)
