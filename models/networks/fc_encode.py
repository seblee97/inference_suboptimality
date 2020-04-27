from .base_network import _BaseNetwork

from typing import List, Dict

import torch.nn as nn

class FullyConnectedEncoderNetwork(_BaseNetwork):

    def __init__(self, config: Dict):

        self.input_dimension = config.get(["model", "input_dimension"])
        self.latent_dimension = config.get(["model", "latent_dimension"])
        self.hidden_dimensions = config.get(["model", "encoder", "hidden_dimensions"])
        self.factor = config.get(["model", "encoder", "output_dimension_factor"])
        
        _BaseNetwork.__init__(self, config=config)
        
    def _construct_layers(self):
        
        self.layers = nn.ModuleList([])
        self.layer_config = [self.input_dimension] + self.hidden_dimensions + [self.factor * self.latent_dimension]
        
        for h in range(len(self.layer_config)-1):
            self.layers.append(self._initialise_weights(nn.Linear(self.layer_config[h], self.layer_config[h + 1])))

