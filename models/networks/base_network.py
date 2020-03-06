from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Dict

class baseNetwork(nn.Module, ABC):

    def __init__(self, config: Dict):
        super(baseNetwork, self).__init__()

        self.nonlinearity_name = config.get(["training", "nonlinearity"])

        if self.nonlinearity_name == 'relu':
            self.nonlinear_function = F.relu
        elif self.nonlinearity_name == 'elu':
            self.nonlinear_function = F.elu 
        elif self.nonlinearity_name == 'sigmoid':
            self.nonlinear_function = F.sigmoid
        else:
            raise ValueError("Invalid nonlinearity name")

        self.initialisation_std = config.get(["training", "initialisation_std"])

        self._construct_layers()

    @abstractmethod
    def _construct_layers(self):
        raise NotImplementedError("Base class method")

    def _initialise_weights(self, layer) -> None:
        """
        Weight initialisation method for given layer
        """
        # TODO: make initialisation part of config to allow for other methods
        torch.nn.init.normal_(layer.weight, std=self.initialisation_std)    
        torch.nn.init.normal_(layer.bias, std=self.initialisation_std)
        return layer

    def forward(self, x):   
        """
        Forward pass

        :param x: input tensor to network
        :return x: output of network
        """
        for layer in self.layers:
            x = self.nonlinear_function(layer(x))

        return x
    
