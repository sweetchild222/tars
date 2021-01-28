import numpy as np
from tars.gradient.abs_gradient import *


class RMSprop(ABSGradient):

    def __init__(self, lr, beta, exp):
        super(RMSprop, self).__init__(lr)

        self.beta = beta
        self.exp = exp


    def bind(self, weights):

        self.weights = weights
        self.accels = [np.zeros(w.shape) for w in self.weights]


    def calcDiff(self, weight, delta, accel):

        average = np.average(delta, axis=0)
        accel = self.beta * accel + (1 - self.beta) * (average)**2

        return self.lr * (average) / (np.sqrt(accel + self.exp)), accel


    def update(self, deltas):

        for i in range(len(self.weights)):
            diff, accel = self.calcDiff(self.weights[i], deltas[i], self.accels[i])
            self.accels[i] = accel
            self.weights[i] -= diff
