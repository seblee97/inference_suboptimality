from .base_approximate_posterior import approximatePosterior

import torch
import torch.distributions as tdist

from utils import log_normal

class gaussianPosterior(approximatePosterior):

    def __init__(self):
        super(gaussianPosterior, self).__init__()
        self.noise_distribution = tdist.Normal(torch.Tensor([0]), torch.Tensor([1]))

    def construct_posterior(self):
        pass

    def sample(self, parameters):

        mu = parameters[:0.5*len(parameters)]
        var = parameters[0.5*len(parameters):]
        
        noise = torch.flatten(self.noise_distribution.sample((var.shape)))
        
        latent_vector = mu + torch.sqrt(var) * noise

        # compute probability of sample under q
        log_pqz = log_normal(latent_vector, mu, var)

        # compute prior probability p(z)
        logpz = log_normal(latent_vector, torch.zeros(latent_vector.shape), torch.ones(latent_vector.shape[0]))
        
        return latent_vector, log_pqz, logpz