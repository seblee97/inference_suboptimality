from .base_network import baseNetwork

from typing import List, Dict

import torch.nn as nn

class feedForwardNetwork(baseNetwork):

    def __init__(self, config: Dict):

        self.input_dimension = config.get(["training", "input_dimension"])
        self.latent_dimension = config.get(["training", "latent_dimension"])
        self.hidden_dimensions = config.get(["encoder", "hidden_dimensions"])

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
