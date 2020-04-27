from .base_approximate_posterior import _ApproximatePosterior

import torch
import torch.distributions as tdist

from typing import Dict, List

class GaussianPosterior(_ApproximatePosterior):

    def __init__(self, config: Dict) -> None:
        _ApproximatePosterior.__init__(self, config)
        self.noise_distribution = tdist.Normal(torch.Tensor([0]), torch.Tensor([1]))

    def sample(self, parameters: torch.Tensor) -> (torch.Tensor, List):
        """
        :param parameters: dimensions (B, X) where the first half of X represents the multidimensional 
                           mean of the Gaussian posterior. The second half of X represents the multidimensional
                           log variance of the Gaussian posterior.
        :return latent_vector: latent vector sample
        :return [mean, log_var]: mean and log variance used downstream to evaluate log-probabilities 
        """
        dimensions = parameters.shape[1] // 2
        mean = parameters[:, :dimensions]
        log_var = parameters[:, dimensions:]

        # Sample the standard deviation along each dimension of the factorized Gaussian posterior.
        noise = self.noise_distribution.sample(log_var.shape).squeeze()
        # Apply the reparameterization trick.
        latent_vector = mean + torch.sqrt(torch.exp(log_var)) * noise
        
        return latent_vector, [mean, log_var]