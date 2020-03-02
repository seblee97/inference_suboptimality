from abc import ABC

import torch
import torch.nn as nn

import numpy as np

from typing import Dict

class baseNetwork(nn.Module, ABC):

    def __init__(self, config: Dict):
        super(baseNetwork, self).__init__()

        self.nonlinear_function = config.get(["training", "nonlinearity"])
        self.initialisation_std = config.get(["training", "initialisation_std"])

        self._construct_layers()

    def _construct_layers(self):
        raise NotImplementedError("Base class method")

    def _initialise_weights(self, layer) -> None:
        """
        Weight initialisation method for given layer
        """
        if self.nonlinear_function == 'relu':
            # maybe change this to initialisation rather than dependent on nonlinearity?
            torch.nn.init.normal_(layer.weight, std=self.initialisation_std)    
            torch.nn.init.normal_(layer.bias, std=self.initialisation_std)
        else:
            raise ValueError("Non linearity type unknown")
        
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
    
