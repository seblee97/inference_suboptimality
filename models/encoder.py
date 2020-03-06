from abc import ABC

import torch
import torch.nn as nn

class Encoder(nn.Module, ABC):

    def __init__(self, network, approximate_posterior, using_flows):
        super(Encoder, self).__init__()

        self.network = network
        self.using_flows = using_flows
        self.approximate_posterior = approximate_posterior
    
    def sample(self, approximate_posterior_parameters):
        """
            Inputs are parameters as obtained by a pass through the network.
            Returns a sample of latent z as well as logpz (log of the latent distribution) and logqz (approximate posterior).
            Can call a method from the flow or other to approximate the posterior further (parameters are in approximate_posterior).
        """
        mu = approximate_posterior_parameters[0]
        logvar =approximate_posterior_parameters[1]
        
        z = ...
        log_qz = log_normal(z, mu, logvar)
        
        If using_flows:
            #Note: approximate_posterior.sample should return the latent and a log-probability!
            z, logp = self.approximate_posterior.sample(approximate_posterior_parameters)
            log_qz += logp
    
        zeros = Variable(torch.zeros(z.size()).type(self.dtype))
        logpz = log_normal(z, zeros, zeros)
        ...
        return z, log_pz, log_qz

    def forward(self, x):
        """
        Defines a run through the network to get latent parameter,
        pass them through a flow/other if require in the sampling part to further approximate.
        """
        # should define a run through the network and return approximate parameters
        approximate_posterior_parameters = self.network.forward(x)

        approximate_posterior_parameters = self.network(x)
        latent_vector, params = self.approximate_posterior.sample(approximate_posterior_parameters)

        return {'z': latent_vector, 'params': params}

# # compute probability of sample under q
#         log_pqz = log_normal(latent_vector, mu, var)

#         import pdb; pdb.set_trace()

#         # compute prior probability p(z)
#         logpz = log_normal(latent_vector, torch.zeros(latent_vector.shape), torch.ones(latent_vector.shape[0]))
