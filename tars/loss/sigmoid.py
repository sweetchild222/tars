import numpy as np
from tars.loss.abs_loss import *


class Sigmoid(ABSLoss):

    def __init__(self):
        super(Sigmoid, self).__init__()

        self.alpha = 0.0000000000001 #avoid divide zero

        #self.alpha = 0.0

    def forward(self, input):

        return 1 / (1 + np.exp(-input))


    def backward(self, y, target):

        negative = np.where(y == 0.0, self.alpha, y)
        positive = np.where(y == 1.0, y -  self.alpha, y)

        return np.where(target == 1, -1 / negative, 1 / (1 - positive))


    def loss(self, y, target):

        negative = np.where(y == 0.0, self.alpha, y)
        positive = np.where(y == 1.0, y -  self.alpha, y)

        loss = -target*np.log(negative) - (1 - target)* np.log(1 - positive)

        return np.mean(loss)
