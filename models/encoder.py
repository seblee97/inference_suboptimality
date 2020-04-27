from abc import ABC

import torch
import torch.nn as nn

class Encoder(nn.Module, ABC):

    def __init__(self, network, approximate_posterior):
        super(Encoder, self).__init__()

        self.network = network
        self.approximate_posterior = approximate_posterior

    def forward(self, x):
        """
        Defines a run through the network to get latent parameter,
        pass them through a flow/other if require in the sampling part to further approximate.
        """
        # should define a run through the network and return approximate parameters
        approximate_posterior_parameters = self.network(x)
        latent_vector, params = self.approximate_posterior.sample(approximate_posterior_parameters)

        return {'z': latent_vector, 'params': params}
