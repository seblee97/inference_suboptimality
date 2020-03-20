from .base_network import baseNetwork
from typing import List, Dict

import torch.nn as nn

class deconvNetwork(baseNetwork):

    def __init__(self, config: Dict):
        """
        Config needs a list of parameters for each layer including the input layer structure.
        All of it should be stored in decoder hidden_dimensions (including input info!).
        
        Shape:
        The first one is linear:
        [input, output]
        
        For convolutional ones:
        param[0], param[1], param[2], param[3], param[4]
        =
        [in_channels, out_channels, kernel_size, stride=1, padding=0]
        
        They implement: (init_channel is 64 or 128 if wide)
        [self.z_size, 256*2*2]
        [256, 128, 4, 2]
        [128, 64, 4, 2, output_padding=1]       ! Output padding for this one
        [64, 3, 4, 2]
        """
            
        self.input_dimension = config.get(["model", "input_dimension"])
        self.latent_dimension = config.get(["model", "latent_dimension"])
        self.hidden_dimensions = config.get(["model", "decoder", "hidden_dimensions"])
        #super(convNetwork, self).__init__()
        baseNetwork.__init__(self, config=config)
    
    def _construct_layers(self):
        self.layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])
        
        # Initial layer is fully connected
        param = self.hidden_dimensions[0]
        input_layer = self._initialise_weights(nn.Linear(param[0], param[1]))
        bn_init = nn.BatchNorm2d(param[1])
        self.layers.append(input_layer)
        self.bn_layers.append(bn_init)
        
        for h in range(1, len(self.hidden_dimensions[:-1])):
            param = self.hidden_dimensions[h]
            hidden_layer = self._initialise_weights(nn.ConvTranspose2d(param[0], param[1], param[2], param[3], param[4]))
            bn_layer = nn.BatchNorm2d(param[1])
            self.layers.append(hidden_layer)
            self.bn_layers.append(bn_layer)

        # final layer to latent dim is a convolutional one with no batch normalisation.
        param = self.hidden_dimensions[-1]
        hidden_to_latent_layer = self._initialise_weights(nn.ConvTranspose2d(param[0], param[1], param[2], param[3], param[4]))
        self.layers.append(hidden_to_latent_layer)

    def forward(self, x):
        """
        Forward pass
        
        :param x: input tensor to network
        :return x: output of network
        """
        x = self.nonlinear_function(self.layers[0](x))
        x = x.view(x.size(0), -1, 2, 2)
        for level, layer in enumerate(self.layers[1:-1]):
            x = self.nonlinear_function(self.bn_layers[level](layer(x)))
        x = self.layers[-1](x)  # the last fully connected layer has no activation function
        return x
