import torch
import torch.distributions as tdist

from .base_local_ammortisation import BaseLocalAmmortisation

class GaussianLocalAmmortisation(BaseLocalAmmortisation):
    """
    *FFG from paper
    """
    def __init__(self, config):
        BaseLocalAmmortisation.__init__(self, config)

        self.noise_distribution = tdist.Normal(torch.Tensor([0]), torch.Tensor([1]))

    def get_additional_parameters(self):
        # no additional parameters in FFG approximate posterior
        return []

    def sample_latent_vector(self, params: torch.Tensor):
        # reparameterise to get latent TODO: add reparameterise to utils?
        noise = self.noise_distribution.sample(params[1].shape).squeeze()
        z = params[0] + noise * torch.sqrt(torch.exp(params[1]))
        return z, params