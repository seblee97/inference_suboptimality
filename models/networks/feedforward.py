from .base_network import baseNetwork

from typing import List, Dict

import torch.nn as nn

class feedForwardNetwork(baseNetwork):

    def __init__(self, config: Dict):

        self.input_dimension = config.get(["model", "input_dimension"])
        self.latent_dimension = config.get(["model", "latent_dimension"])
        self.hidden_dimensions = config.get(["model", "encoder", "hidden_dimensions"])
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
