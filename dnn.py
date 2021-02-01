from dnn_lib.drawer import *
from dnn_lib.loader import *
from dnn_lib.model_templates import *
from tars.tars import *
from util.util import *
import argparse
import datetime as dt


def drawTestY(test_y_on_epoch, feature_max):

    path = 'dnn_output/' + datetime.datetime.now().strftime("%m%d_%H%M")

    for test_y_value in reversed(test_y_on_epoch):

        epoch = test_y_value['epoch']
        test_y = test_y_value['test_y']

        index = 0

        height = feature_max[0]
        width = feature_max[1]
        matrix = np.zeros((height, width))

        for h in range(height):
            for w in range(width):
                matrix[h, w] = np.argmax(test_y[index])
                index += 1

        fileName = 'epoch_' + str(epoch) + '.png'

        print_table({'Write Path':[path + '/' + fileName]})

        matrixToImage(path, fileName, matrix)


def print_arg(model, activation, weightInit, gradient, classes, epochs, batches, train_x_shape, train_t_shape, test_x_shape):

    trimed = batches > train_x_shape[0]

    batches = train_data_count if trimed else batches

    batch_str = str(batches) + (' (trimed)' if trimed else '')

    arg = ['model', 'activation', 'weight init', 'gradient', 'classes', 'epochs', 'batches', 'train x shape', 'train t shape', 'test x shape']
    values = [model, activation, weightInit, gradient, classes, epochs, batch_str, train_x_shape, train_t_shape, test_x_shape]
    table = {'Argument':arg, 'Values':values}
    print_table(table)


def print_performance(span):

    performance = ['train minute span']

    min_span = '{:.2f}'.format(span.total_seconds() / 60)
    values = [min_span]
    table = {'Performance':performance, 'Values':values}
    print_table(table)


def print_layer(layerList):

    layerNames = [layer.layerName() for layer in layerList]
    outputShapes = [layer.outputShape() for layer in layerList]

    print_table({'Layer':layerNames, 'Output Shape':outputShapes})


def create(modelType, activationType, weightInitType, input_shape, classes, gradientType):

    layersTemplate = createLayersTemplate(modelType, activationType, weightInitType, input_shape, classes)
    gradientTemplate =  createGradientTemplate(gradientType)

    tars = Tars(layersTemplate, gradientTemplate)
    layerList = tars.build()
    print_layer(layerList)

    return tars


def train_test(tars, train_x, train_t, test_x, epochs, batches, draw_epoch_term):

    start_time = dt.datetime.now()

    test_y_on_epoch = []

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

        if ((epochs - epoch) % draw_epoch_term) == 1:
            test_y_on_epoch.append({'epoch': epoch + 1, 'test_y': tars.test(test_x)})

    train_span = (dt.datetime.now() - start_time)

    return test_y_on_epoch, train_span


def main(modelType, activationType, weightInitType, gradientType, epochs, batches, draw_epoch_term):

    train_x, train_t, test_x, feature_max = loadDataSet()

    print_arg(modelType, activationType, weightInitType, gradientType, train_t.shape[-1], epochs, batches, train_x.shape, train_t.shape, test_x.shape)

    tars = create(modelType, activationType, weightInitType, train_x.shape[1:], train_t.shape[-1], gradientType)

    test_y_on_epoch, train_span = train_test(tars, train_x, train_t, test_x, epochs, batches, draw_epoch_term)

    print_performance(train_span)

    drawTestY(test_y_on_epoch, feature_max)


def parse_arg():

    parser = argparse.ArgumentParser(prog='DNN')
    parser.add_argument('-m', dest='modelType', type=str, default='light', choices=['light', 'complex'], help='sample model type (default:light)')
    parser.add_argument('-g', dest='gradientType', type=str, default='rmsProp', choices=['adam', 'sgd', 'rmsProp'], help='sample gradient type (default: rmsProp)')
    parser.add_argument('-a', dest='activationType', type=str, default='elu', choices=['linear', 'relu', 'elu', 'leakyRelu', 'sigmoid', 'tanh'], help='sample activation type (default: relu)')
    parser.add_argument('-w', dest='weightInitType', type=str, default='glorot_uniform', choices=['lecun_uniform', 'glorot_uniform', 'he_uniform', 'lecun_normal', 'glorot_normal', 'he_normal'], help='weight initial type (default: glorot_uniform)')
    parser.add_argument('-e', dest='epochs', type=int, default=60, help='epochs (default: 60)')
    parser.add_argument('-b', dest='batches', type=int, default=100, help='batches (default: 100)')
    parser.add_argument('-d', dest='draw_epoch_term', type=int, default=20, help='draw epoch term (default: 20)')

    args = parser.parse_args()

    if args.epochs < 1:
        print('DNN: error: argument -e: invalid value: ', str(args.epochs), ' (value must be over 0)')
        return None

    if args.batches < 1:
        print('DNN: error: argument -b: invalid value: ', str(args.batches), ' (value must be over 0)')
        return None

    return args


if __name__ == "__main__":

    args = parse_arg()

    if args != None:
        main(args.modelType, args.activationType, args.weightInitType, args.gradientType, args.epochs, args.batches, args.draw_epoch_term)
