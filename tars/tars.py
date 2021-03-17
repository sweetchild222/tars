import numpy as np
from tars.gradient.creator import *
from tars.layer.creator import *
from tars.loss.creator import *
from tars.layer.input import *
from tars.layer.convolution import *
from tars.layer.max_pooling import *
from tars.layer.flatten import *
from tars.layer.dense import *
from tars.layer.basicRNN import *


class Tars:
    def __init__(self, layersTemplate, gradientTemplate, lossTemplate):
        self.layersTemplate = layersTemplate
        self.gradientTemplate = gradientTemplate
        self.lossTemplate = lossTemplate
        self.head = None
        self.tail = None
        self.loss = None


    def createModel(self, layersTemplate, gradientTemplate, lossTemplate):

        head, tail = createLayers(layersTemplate, gradientTemplate)
        loss = createLoss(lossTemplate)

        return head, tail, loss


    def builtLayerList(self, head):

        layer_list = []

        next_layer = head

        while True:

            layer_list.append(next_layer)

            next_layer = next_layer.forwardLayer()

            if next_layer is None:
                break

        return layer_list


    def build(self):

        head, tail, loss = self.createModel(self.layersTemplate, self.gradientTemplate, self.lossTemplate)

        self.head = head
        self.tail = tail
        self.loss = loss

        return self.builtLayerList(self.head)


    def resetState(self):
        next_layer = self.head

        while True:
            next_layer.resetState()

            next_layer = next_layer.forwardLayer()

            if next_layer is None:
                break


    def shuffle(self, train_x, train_t):

        shuffle_indices = np.arange(train_x.shape[0])
        np.random.shuffle(shuffle_indices)

        return train_x[shuffle_indices], train_t[shuffle_indices]


    def train(self, batch_x, batch_t):

        return self.trainCore(self.head, self.tail, batch_x, batch_t)


    def trainCore(self, head, tail, batch_x, batch_t):

        batch_y = self.forward(head, batch_x)

        batch_y = self.loss.forward(batch_y)

        loss = self.loss.loss(batch_y, batch_t)

        batch_e = self.loss.backward(batch_y, batch_t)

        batch_e = self.backward(tail, batch_e)

        return loss


    def forward(self, head, batch_x):

        next_layer = head

        while True:
            batch_y = next_layer.forward(batch_x)

            next_layer = next_layer.forwardLayer()

            if next_layer is None:
                break

            batch_x = batch_y

        return batch_y


    def backward(self, tail, batch_e):

        prev_layer = tail

        while True:
            batch_e = prev_layer.backward(batch_e)

            prev_layer = prev_layer.backwardLayer()

            if prev_layer is None:
                break

        return batch_e


    def testCore(self, head, batch_x):

        next_layer = head

        while True:
            batch_y = next_layer.test(batch_x)

            next_layer = next_layer.forwardLayer()

            if next_layer is None:
                break

            batch_x = batch_y

        return batch_y


    def test(self, batch_x):

        batch_y = self.testCore(self.head, batch_x)

        return self.loss.forward(batch_y)
