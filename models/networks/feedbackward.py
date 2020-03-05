from .base_network import baseNetwork

from typing import List, Dict

import torch.nn as nn

class feedBackwardNetwork(baseNetwork):

    def __init__(self, config: Dict):

        self.input_dimension = config.get(["training", "input_dimension"])
        self.latent_dimension = config.get(["training", "latent_dimension"])
        self.hidden_dimensions = config.get(["decoder", "hidden_dimensions"])

        baseNetwork.__init__(self, config=config)

    def _construct_layers(self):
        
        self.layers = nn.ModuleList([])
        
        latent_to_hidden_layer = self._initialise_weights(nn.Linear(self.latent_dimension, self.hidden_dimensions[0]))
        self.layers.append(latent_to_hidden_layer)

        for h in range(len(self.hidden_dimensions[:-1])):
            hidden_layer = self._initialise_weights(nn.Linear(self.hidden_dimensions[h], self.hidden_dimensions[h + 1]))
            self.layers.append(hidden_layer)

        # final decoding layer (back to input dimension)
        output_layer = self._initialise_weights(nn.Linear(self.hidden_dimensions[-1], self.input_dimension))
        self.layers.append(output_layer)