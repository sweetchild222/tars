from PIL import Image
import numpy as np
import os



def dataFileNames(stateful):

    files = []

    dir = 'rnn_lib/data'

    for file in os.listdir('rnn_lib/data'):

        if file.startswith('rnn') == False:
            continue

        if file.endswith('.data') == False:
            continue

        files.append(dir + '/' + file)

        if stateful is False:
            break

    files.sort()

    return files


def findSequenceLength(data_list):

    sequence_length = 0

    for data in data_list:
        string_list = data.split('\n')
        for string in string_list:
            if len(string) > sequence_length:
                sequence_length = len(string)

    return sequence_length


def makeOneHotMap(data):

    np_text = np.array(list(data))
    unique = np.unique(np_text, return_counts=False)

    oneHotMap = {key : index for index, key in enumerate(unique)}
    mapToChar = {index : key for index, key in enumerate(unique)}

    return oneHotMap, mapToChar


def dataToNumpy(data_list, oneHotMap, sequence_length):

    num_list_list = []

    for data in data_list:
        string_list = data.split('\n')
        num_list = []
        for string in string_list:
            string = string + ''.join([' '] * (sequence_length - len(string)))
            num_list.append([oneHotMap[char] for char in string])

        num_list_list.append(num_list)

    return np.array(num_list_list)


def extractData(files):

    data_list = [open(f, 'r').read().strip() for f in files]

    oneHotMap, mapToChar = makeOneHotMap(''.join(data_list))

    sequence_length = findSequenceLength(data_list)

    numpy_list_list = dataToNumpy(data_list, oneHotMap, sequence_length)

    return numpy_list_list, oneHotMap, mapToChar


def encodeOneHot(data, classes):

    oneHotEncode = [np.eye(classes)[i] for i in data]

    return np.array(oneHotEncode)


def loadDataSet(stateful):

    files = dataFileNames(stateful)

    numpy_list_list, oneHotmap, mapToChar = extractData(files)

    sequence_length = numpy_list_list.shape[2]

    train_x_states = []
    train_t_states = []

    for numpy_list in numpy_list_list:

        train_x = []
        train_t = []

        for np_data in numpy_list:
            x = encodeOneHot(np_data[0: sequence_length - 1], len(oneHotmap))
            t = encodeOneHot(np_data[1: sequence_length], len(oneHotmap))

            train_x.append(x)
            train_t.append(t)

        train_x_states.append(train_x)
        train_t_states.append(train_t)

    train_x_states = np.array(train_x_states)
    train_t_states = np.array(train_t_states)

    train_x_states = np.swapaxes(train_x_states, 1, 0)
    train_t_states = np.swapaxes(train_t_states, 1, 0)

    return train_x_states, train_t_states, oneHotmap, mapToChar
