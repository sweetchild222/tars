from cnn_lib.loader import *
from cnn_lib.model_templates import *
from tars.tars import *
from util.util import *
import argparse
import datetime as dt


def print_oneHotMap(oneHotMap):

    oneHotList = []
    labelList = []

    classes = len(oneHotMap)

    for mapKey in oneHotMap:
        map = np.eye(classes)[oneHotMap[mapKey]].reshape(classes, 1)
        oneHotList.append(map.reshape(-1))
        labelList.append(mapKey)

    print_table({'Label':labelList, 'OneHot':oneHotList})


def print_performance(accuracy, span):

    performance = ['accuracy', 'train minute span']

    min_span = '{:.2f}'.format(span.total_seconds() / 60)
    values = [str(accuracy) + ' %', min_span]
    table = {'Performance':performance, 'Values':values}
    print_table(table)


def print_arg(model, activation, weightInit, gradient, classes, epochs, batches, train_x_shape, train_t_shape, test_x_shape, test_t_shape):

    trimed = batches > train_x_shape[0]

    batches = train_data_count if trimed else batches

    batch_str = str(batches) + (' (trimed)' if trimed else '')

    arg = ['model', 'activation', 'weight init', 'gradient', 'classes', 'epochs', 'batches', 'train x shape', 'train t shape', 'test x shape', 'test t shape']
    values = [model, activation, weightInit, gradient, classes, epochs, batch_str, train_x_shape, train_t_shape, test_x_shape, test_t_shape]
    table = {'Argument':arg, 'Values':values}
    print_table(table)


def print_layer(layerList):

    layerNames = [layer.layerName() for layer in layerList]
    outputShapes = [layer.outputShape() for layer in layerList]

    print_table({'Layer':layerNames, 'Output Shape':outputShapes})


def create(modelType, activationType, weightInitType, input_shape, classes, gradientType):

    layersTemplate = createLayersTemplate(modelType, activationType, weightInitType, input_shape, classes)
    gradientTemplate = createGradientTemplate(gradientType)

    tars = Tars(layersTemplate, gradientTemplate)
    layerList = tars.build()
    print_layer(layerList)

    return tars


def train(tars, train_x, train_t, epochs, batches):

    start_time = dt.datetime.now()

    for epoch in range(epochs):

        train_x, train_t = tars.shuffle(train_x, train_t)

        length = train_x.shape[0]
        loss = 0
        i = 0

        for b in range(0, length, batches):
            batch_x = train_x[b : b + batches]
            batch_t = train_t[b : b + batches]

            loss += tars.train(batch_x, batch_t)
            i += 1

        print_table({'Epochs':[str(epoch + 1) +'/' + str(epochs)], 'Loss':[loss / i]})

    return (dt.datetime.now() - start_time)


def test(tars, test_x, test_t):

    test_y = tars.test(test_x)

    count = test_y.shape[0]
    correct_count = 0

    np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})

    for i in range(count):

        y = test_y[i]
        t = test_t[i]

        correct = (np.argmax(y) == np.argmax(t))

        correct_count += (1 if correct else 0)
        correct_char = 'O' if correct else 'X'

        print_table({'Test':[y.round(decimals=2)], 'Label':[t.round(decimals=2)], 'Correct':[correct_char]})

    accuracy = float(correct_count / count) * 100

    return accuracy


def main(modelType, activationType, weightInitType, gradientType, classes, epochs, batches):

    train_x, train_t, test_x, test_t, oneHotMap = loadDataSet(classes)

    print_arg(modelType, activationType, weightInitType, gradientType, train_t.shape[-1], epochs, batches, train_x.shape, train_t.shape, test_x.shape, test_t.shape)

    print_oneHotMap(oneHotMap)

    tars = create(modelType, activationType, weightInitType, train_x.shape[1:], train_t.shape[-1], gradientType)

    train_span = train(tars, train_x, train_t, epochs, batches)

    accuracy = test(tars, test_x, test_t)

    print_performance(accuracy, train_span)


def parse_arg():

    parser = argparse.ArgumentParser(prog='CNN')
    parser.add_argument('-c', dest='classes', type=int, default='3', metavar="[1-10]", help='classes (default: 3)')
    parser.add_argument('-m', dest='modelType', type=str, default='light', choices=['light', 'complex'], help='sample model type (default:light)')
    parser.add_argument('-g', dest='gradientType', type=str, default='adam', choices=['adam', 'sgd', 'rmsProp'], help='sample gradient type (default: rmsProp)')
    parser.add_argument('-a', dest='activationType', type=str, default='leakyRelu', choices=['linear', 'relu', 'elu', 'leakyRelu', 'sigmoid', 'tanh'], help='sample activation type (default: relu)')
    parser.add_argument('-w', dest='weightInitType', type=str, default='he_normal', choices=['lecun_uniform', 'glorot_uniform', 'he_uniform', 'lecun_normal', 'glorot_normal', 'he_normal'], help='weight initial type (default: he_normal)')
    parser.add_argument('-e', dest='epochs', type=int, default=5, help='epochs (default: 5)')
    parser.add_argument('-b', dest='batches', type=int, help='batches (default: classes x 3)')
    args = parser.parse_args()

    if args.classes < 1 or args.classes > 10:
        print('CNN: error: argument -c: invalid value: ', str(args.classes), ' (value must be 1 from 10)')
        return None

    if args.batches == None:
        args.batches = args.classes * 3

    if args.batches < 1:
        print('CNN: error: argument -b: invalid value: ', str(args.batches), ' (value must be over 0)')
        return None

    return args

if __name__ == "__main__":

    args = parse_arg()

    if args != None:
        main(args.modelType, args.activationType, args.weightInitType, args.gradientType, args.classes, args.epochs, args.batches)
