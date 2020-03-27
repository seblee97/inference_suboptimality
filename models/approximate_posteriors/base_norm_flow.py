from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseFlow(nn.Module, ABC):

    def __init__(self, config):

        nn.Module.__init__(self)

        self.initialisation = config.get(["flow", "weight_initialisation"])
        self._construct_layers()

        self.activation_function = config.get(["flow", "nonlinearity"])
        if self.activation_function == 'elu':
            self.activation = F.elu


    @abstractmethod
    def _construct_layers(self):
        raise NotImplementedError("Base class method")

    def _initialise_weights(self, layer) -> None:
        """
        Weight initialisation method for given layer
        """
        if self.initialisation == "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        elif self.initialisation == "xavier_normal":
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        elif self.initialisation == "default":
            pass
        else:
            raise ValueError("Initialisation {} not recognised".format(self.initialisation))

        return layer

    @abstractmethod
    def forward(self, z0: torch.Tensor):
        raise NotImplementedError("Base class method")
