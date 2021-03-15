from abc import *


class ABSActivation(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass


    @abstractmethod
    def layerName(self, output):
        pass


    @abstractmethod
    def test(self, output):
        pass


    @abstractmethod
    def forward(self, output):
        pass


    @abstractmethod
    def backward(self, error):
        pass
