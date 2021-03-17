import numpy as np
import math
from tars.loss.abs_loss import *



class Sigmoid(ABSLoss):

    def __init__(self):
        super(Sigmoid, self).__init__()


    def forward(self, input):

        self.last_input = input

        return 1 / (1 + np.exp(-input))


    def backward(self, y, target):

        return (y - target)


    def loss(self, y, target):

        e = math.exp(1)

        input = self.last_input

        positive = input - input*y + np.log(1 + (e**(-input)))
        negative = -input*y + np.log((e**(input)) + 1)

        loss = np.where(input >= 0.0, positive, negative)

        return np.mean(loss)
