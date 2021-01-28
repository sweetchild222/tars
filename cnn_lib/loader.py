from PIL import Image
import numpy as np
import random
import os
from array import *
from random import shuffle
import operator

def randomLabels(classes, trainPath):

	labels = []

	for dirname in os.listdir(trainPath):
		labels.append(dirname)

	return random.sample(labels, k=classes)


def loadMNISTFiles(path, lables):

	colorDim = 1

	X = []
	T = []

	imgSize = 0

	for label in lables:

		subPath = path + '/' + label

		for fileName in os.listdir(subPath):

			if fileName.endswith(".png") == False:
				continue

			filePath = subPath + '/' + fileName

			img = np.array(Image.open(filePath)).astype(np.float32)

			imgSize = img.shape[0]

			colorImg = np.hstack([img.reshape((-1, 1)) for i in range(colorDim)])

			X.append(colorImg)

			T.append(label)


	#matrixToImage(np.array(X).reshape(len(X), 1, imgSize * imgSize))

	return np.array(X).reshape(len(X), imgSize, imgSize, colorDim), np.array(T)


def extractMNIST(classes, trainPath, testPath):

	lables = randomLabels(classes, trainPath)

	train_x, train_t = loadMNISTFiles(trainPath, lables)

	test_x, test_t = loadMNISTFiles(testPath, lables)

	return train_x, train_t, test_x, test_t


def makeOneHotMap(train_t, test_t):

    labels = np.hstack((train_t, test_t))

    unique = np.unique(labels, return_counts=False)

    return {key : index for index, key in enumerate(unique)}


def encodeOneHot(oneHotMap, train_t, test_t):

    labels = np.hstack((train_t, test_t))

    classes = len(oneHotMap)

    labels = [np.eye(classes)[oneHotMap[l]] for l in labels]
    labels = np.array(labels)

    train = labels[0:len(train_t)]
    test = labels[-len(test_t):]

    return train, test


def loadDataSet(classes):

    train_x, train_t, test_x, test_t = extractMNIST(classes, 'cnn_lib/mnist/train', 'cnn_lib/mnist/test')

    all_x = np.vstack((train_x, test_x))
    all_x -= np.mean(all_x)
    all_x /= np.std(all_x)

    train_x = all_x[0:len(train_x)]
    test_x = all_x[-len(test_x):]

    oneHotMap = makeOneHotMap(train_t, test_t)

    train_t, test_t = encodeOneHot(oneHotMap, train_t, test_t)

    return train_x, train_t, test_x, test_t, oneHotMap
