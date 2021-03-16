from abc import *


class ABSLoss(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass


    def forward(self, input):
        pass


    def backward(self, y, target):
        pass
