import numpy as np
import math
from tars.loss.abs_loss import *



class MeanSquare(ABSLoss):

    def __init__(self):
        super(MeanSquare, self).__init__()


    def forward(self, input):

        return input


    def backward(self, y, target):

        return (y - target)


    def loss(self, y, target):

        return np.mean((y-target)**2)
