from PIL import Image
import numpy as np


def encodeOneHot(train_t):

    unique = np.unique(train_t, return_counts=False)
    oneHotMap = {key : index for index, key in enumerate(unique)}

    classes = len(oneHotMap)
    oneHotEncode = [np.eye(classes)[oneHotMap[t]] for t in train_t]

    return np.array(oneHotEncode)


def loadTestDataSet(feature_max):

    test_x = []

    height = feature_max[0]
    width = feature_max[1]

    for h in range(height):
        for w in range(width):
            feature1 = float(w)
            feature2 = float(h)
            test_x.append([feature1, feature2])

    return np.array(test_x)


def loadDataSet():

    train_x, train_t, feature_max = extractData()

    train_t = encodeOneHot(train_t)

    test_x = loadTestDataSet(feature_max)

    all_x = np.vstack((train_x, test_x))
    all_x -= np.mean(all_x, axis=0)
    all_x /= np.std(all_x, axis=0)

    train_x = all_x[0:len(train_x)]
    test_x = all_x[-len(test_x):]

    return train_x, train_t, test_x, feature_max


def extractData():

    img = np.array(Image.open('dnn_lib/data/train.png')).astype(np.float32)

    (height, width) = img.shape

    x = []
    t = []

    for h in range(height):
        for w in range(width):

            value = img[h][w]

            if value == 0.0:
                continue

            feature1 = float(w)
            feature2 = float(h)

            x.append([feature1, feature2])
            t.append(value)

    return np.array(x), np.array(t), img.shape
