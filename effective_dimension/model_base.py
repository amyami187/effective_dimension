from abc import ABC, abstractmethod


# This is the structure of a model class needed to compute the effective dimension. It is an abstract base class

class Model(ABC):
    """
    Abstract base class for classical/quantum models.
    """
    def __init__(self):
        """
        :param thetamin: int, minimum used in uniform sampling of the parameters
        :param thetamax: int,  minimum used in uniform sampling of the parameters
        """
        # Stack data together and combine parameter sets to make calcs more efficient
        self.thetamin = 0
        self.thetamax = 1

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_gradient(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_fisher(self, *args, **kwargs):
        raise NotImplementedError()