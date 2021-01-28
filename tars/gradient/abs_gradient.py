from abc import *


class ABSGradient(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, lr):
        self.lr = lr

    @abstractmethod
    def bind(self, weights):
        pass

    @abstractmethod
    def update(self, deltas):
        pass
