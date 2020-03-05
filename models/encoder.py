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
        latent_vector, params = self.approximate_posterior.sample(approximate_posterior_parameters)

        return {'z': latent_vector, 'params': params}

# # compute probability of sample under q
#         log_pqz = log_normal(latent_vector, mu, var)

#         import pdb; pdb.set_trace()

#         # compute prior probability p(z)
#         logpz = log_normal(latent_vector, torch.zeros(latent_vector.shape), torch.ones(latent_vector.shape[0]))