from typing import List, Dict

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.networks.base_network import baseNetwork

class NFlowfeedForwardNetwork(baseNetwork):

    def __init__(self, config: Dict):

        # half of the latent z size as input
        self.input_dimension = config.get(["flow", "input_dim"])
        self.latent_dimension = config.get(["flow", "latent_dim"])
        self.hidden_dimensions = config.get(["flow", "flow_layers"])
        self.activation_function = config.get(["flow", "nonlinearity"])
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
        for layer in self.layers[:-1]:
            x = self.activation_function(layer(x))
        return self.layers[-1](x)