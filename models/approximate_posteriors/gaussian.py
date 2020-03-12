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
        # There are two dimensions to |parameters|; the second one consists of two halves:
        #   1. The first half represents the multidimensional mean of the Gaussian posterior.
        #   2. The second half represents the multidimensional log variance of the Gaussian posterior.
        dimensions = parameters.shape[1] // 2
        mean = parameters[:, :dimensions]
        log_var = parameters[:, dimensions:]

        # Sample the standard deviation along each dimension of the factorized Gaussian posterior.
        noise = self.noise_distribution.sample(log_var.shape).squeeze()
        # Apply the reparameterization trick.
        latent_vector = mean + torch.sqrt(torch.exp(log_var)) * noise
        return latent_vector, [mean, log_var]