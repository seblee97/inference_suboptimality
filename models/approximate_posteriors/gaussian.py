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

        cutoff = int(0.5 * len(parameters))

        mu = parameters[:, :cutoff]
        var = parameters[:, cutoff:]
        
        # sample noise (reparameterisation trick), unsqueeze to match dimensions
        noise = self.noise_distribution.sample(var.shape).squeeze()
        
        latent_vector = mu + torch.sqrt(var) * noise
        
        return latent_vector, [mu, var]