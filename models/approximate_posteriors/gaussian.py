from .base_approximate_posterior import approximatePosterior

import torch
import torch.distributions as tdist

class gaussianPosterior(approximatePosterior):

    def __init__(self):
        super(gaussianPosterior, self).__init__()
        self.noise_distribution = tdist.Normal(torch.Tensor([0]), torch.Tensor([1]))

    def construct_posterior(self):
        pass

    def sample(self, parameters):

        mu = parameters[:0.5*len(parameters)]
        sigma = parameters[0.5*len(parameters):]
        
        noise = torch.flatten(self.noise_distribution.sample((sigma.shape)))
        
        latent_vector = mu + sigma * noise
        
        return latent_vector

