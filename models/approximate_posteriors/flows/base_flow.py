from abc import ABC, abstractmethod

import torch.nn as nn 

class BaseFlow(nn.Module, ABC):

    def __init__(self, config):

        nn.Module.__init__(self)

        self.initialisation = config.get(["flow", "weight_initialisation"])
        self._construct_layers()

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