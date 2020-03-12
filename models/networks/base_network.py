from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Dict

class baseNetwork(nn.Module, ABC):

    def __init__(self, config: Dict):
        super(baseNetwork, self).__init__()

        self.nonlinearity_name = config.get(["model", "nonlinearity"])

        if self.nonlinearity_name == 'relu':
            self.nonlinear_function = F.relu
        elif self.nonlinearity_name == 'elu':
            self.nonlinear_function = F.elu 
        elif self.nonlinearity_name == 'sigmoid':
            self.nonlinear_function = torch.sigmoid
        else:
            raise ValueError("Invalid nonlinearity name")

        self.initialisation = config.get(["model", "initialisation"])
        self.initialisation_std = config.get(["model", "initialisation_std"])

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
        else:
            raise ValueError("Initialisation {} not recognised".format(self.initialisation))

        return layer

    def forward(self, x):   
        """
        Feeds the given input tensor through this network.  Note that the
        activation function is not applied to the output of the final layer.

        :param x: input tensor to network
        :return: output of network
        """
        for layer in self.layers[:-1]:
            x = self.nonlinear_function(layer(x))
        return self.layers[-1](x)
    
