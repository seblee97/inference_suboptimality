from abc import ABC

import torch
import torch.nn as nn

class Encoder(nn.Module, ABC):

    def __init__(self, network, approximate_posterior):
        super(Encoder, self).__init__()

        self.network = network 
        self.approximate_posterior = approximate_posterior

    def forward(self, x):

        approximate_posterior_parameters = self.network(x)
        latent_vector = self.approximate_posterior.sample(approximate_posterior_parameters)

        return latent_vector
