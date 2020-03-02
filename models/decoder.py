from abc import ABC

import torch
import torch.nn as nn

class Decoder(nn.Module, ABC):

    def __init__(self, network):
        #This must receive a network.
        super(Decoder, self).__init__()

        self.network = network 

    def forward(self, x):
        """Latent forward computation on the lattent z"""
        decoding = self.network.forward(x)

        return decoding
