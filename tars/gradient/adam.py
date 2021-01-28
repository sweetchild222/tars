import numpy as np
from tars.gradient.abs_gradient import *


class Adam(ABSGradient):

    def __init__(self, lr, beta1, beta2, exp):
        super(Adam, self).__init__(lr)

        self.beta1 = beta1
        self.beta2 = beta2
        self.exp = exp


    def bind(self, weights):
        self.weights = weights
        self.accels = [np.zeros(w.shape) for w in self.weights]


    def update(self, deltas):

        for i in range(len(self.weights)):
            diff, accel = self.calcDiff(self.weights[i], deltas[i], self.accels[i])
            self.accels[i] = accel
            self.weights[i] -= diff


    def calcDiff(self, weight, delta, accel):

        average = np.average(delta, axis=0)
        accel = self.beta1 * accel + (1 - self.beta2) * (average)**2

        return self.lr * (average)/(np.sqrt(accel) + self.exp), accel
