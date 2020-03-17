from typing import List, Dict


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


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





class NFlowfeedForwardNetwork(baseNetwork):

    def __init__(self):

        # half of the latent z size as input
        self.input_dimension = 392
        self.latent_dimension = 25
        self.hidden_dimensions = [50, 50]
        self.activation_function = F.elu
        # The activation function is defined at mother class level as well as forward
        baseNetwork.__init__(self, config=config)

    def _construct_layers(self):

        self.layers = nn.ModuleList([])

        input_layer = self._initialise_weights(nn.Linear(self.input_dimension, self.hidden_dimensions[0]))
        self.layers.append(input_layer)

        for h in range(len(self.hidden_dimensions[:-1])):
            hidden_layer = self._initialise_weights(nn.Linear(self.hidden_dimensions[h], self.hidden_dimensions[h + 1]))
            self.layers.append(hidden_layer)

        # final layer to latent dim
        hidden_to_latent_layer = self._initialise_weights(nn.Linear(self.hidden_dimensions[-1], self.latent_dimension))
        self.layers.append(hidden_to_latent_layer)

    def forward(self, x):
        for layer in self-layers[:-1]:
            x = self.activation_function(layer(x))
        return self.layers[-1](x)