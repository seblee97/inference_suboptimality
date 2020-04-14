from .base_network import baseNetwork

from typing import List, Dict

import torch.nn as nn

class feedBackwardNetwork(baseNetwork):

    def __init__(self, config: Dict):

        self.input_dimension = config.get(["model", "input_dimension"])
        self.latent_dimension = config.get(["model", "latent_dimension"])
        self.hidden_dimensions = config.get(["model", "decoder", "hidden_dimensions"])

        baseNetwork.__init__(self, config=config)

    def _construct_layers(self):
        
        self.layers = nn.ModuleList([])
        
        self.layer_config = [self.latent_dimension] + self.hidden_dimensions + [self.input_dimension]
        
        for h in range(len(self.layer_config)-1):
            self.layers.append(self._initialise_weights(nn.Linear(self.layer_config[h], self.layer_config[h + 1])))

