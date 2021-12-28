from abc import *


class ABSLoss(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass


    @abstractmethod
    def forward(self, input):
        pass


    @abstractmethod
    def backward(self, y, target):
        pass


    @abstractmethod
    def loss(self, y, target):
        pass
