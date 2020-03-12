from abc import ABC, abstractmethod

import torch.nn as nn 

class BaseFlow(nn.Module, ABC):

    def __init__(self, config):

        nn.Module.__init__(self)

        self._construct_layers()

    @abstractmethod
    def _construct_layers(self):
        raise NotImplementedError("Base class method")