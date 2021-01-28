import numpy as np
from tars.gradient.abs_gradient import *


class SGD(ABSGradient):

    def __init__(self, lr):
        super(SGD, self).__init__(lr)


    def bind(self, weights):
        self.weights = weights


    def update(self, deltas):

        for i in range(len(self.weights)):
            average = np.average(deltas[i], axis=0)
            diff = average * self.lr
            self.weights[i] -= diff
