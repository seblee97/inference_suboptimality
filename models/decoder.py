from abc import ABC

import torch
import torch.nn as nn

class Decoder(nn.Module, ABC):

    def __init__(self, network):
        super(Decoder, self).__init__()

        self.network = network 

    def forward(self, x):

        decoding = self.network(x)

        return decoding
