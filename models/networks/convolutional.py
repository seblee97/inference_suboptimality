from .base_network import _BaseNetwork
from typing import List, Dict

import torch
import torch.nn as nn

class ConvNetwork(_BaseNetwork):

    def __init__(self, config: Dict):
        """
        Config needs a list of parameters for each layer including the input layer structure.
        All of it should be stored in encoder hidden_dimensions (including input info!).
        
        Shape:
        For convolutional ones:
        param[0], param[1], param[2], param[3], param[4]
                    =
        [in_channels, out_channels, kernel_size, stride=1, padding=0]
    
        The last one linear:
        [input, output]
        
        They implement: (init_channel is 64 or 128 if wide)
        [3, init_channel, 4, 2, 0]
        [init_channel, init_channel*2, 4, 2, 0]
        [init_channel*2, init_channel*4, 4, 2, 0]
        [init_channel*4*2*2, self.z_size*2]
        """
        
        self.input_dimension = config.get(["model", "input_dimension"])
        self.latent_dimension = config.get(["model", "latent_dimension"])
        self.hidden_dimensions = config.get(["model", "encoder", "hidden_dimensions"])
        #super(convNetwork, self).__init__()
        _BaseNetwork.__init__(self, config=config)
    
    def _construct_layers(self) -> None:
        self.layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])
        
        # Initial layer
        param = self.hidden_dimensions[0]
        input_layer = self._initialise_weights(nn.Conv2d(param[0], param[1], param[2], param[3], padding=param[4]))
        bn_init = nn.BatchNorm2d(param[1])
        self.layers.append(input_layer)
        self.bn_layers.append(bn_init)
        
        for h in range(1, len(self.hidden_dimensions[:-1])):
            param = self.hidden_dimensions[h]
            hidden_layer = self._initialise_weights(nn.Conv2d(param[0], param[1], param[2], param[3], padding=param[4]))
            bn_layer = nn.BatchNorm2d(param[1])
            self.layers.append(hidden_layer)
            self.bn_layers.append(bn_layer)
        
        # final layer to latent dim is a fully connected one.
        param = self.hidden_dimensions[-1]
        hidden_to_latent_layer = self._initialise_weights(nn.Linear(param[0], param[1]))
        self.layers.append(hidden_to_latent_layer)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass
        
        :param x: input tensor to network
        :return x: output of network
        """
        for level, layer in enumerate(self.layers[:-1]):
            x = self.nonlinear_function(self.bn_layers[level](layer(x)))
        x = x.view(x.size(0), -1)
        x = self.layers[-1](x)  # the last fully connected layer has no activation function
        return x

