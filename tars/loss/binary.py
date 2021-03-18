import numpy as np
import math
from tars.loss.abs_loss import *



class Binary(ABSLoss):

    def __init__(self):
        super(Binary, self).__init__()


    def forward(self, input):

        return 1 / (1 + np.exp(-input))


    def backward(self, y, target):

        return (y - target)


    def loss(self, y, target):

        loss = -(target*np.log2(y) + (1-target)*np.log2(1-y))

        return np.mean(loss)
