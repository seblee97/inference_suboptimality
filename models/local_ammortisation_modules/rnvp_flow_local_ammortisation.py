import torch

from .base_local_ammortisation import BaseLocalAmmortisation

class RNVPLocalAmmortisation(BaseLocalAmmortisation):
    """
    *FFG from paper
    """
    def __init__(self, config):
        BaseLocalAmmortisation.__init__(self, config)

    def get_parameters(self):
        # start with unit normal prior
        mean = torch.zeros()
        logvar = torch.zeros()
        return mean, logvar

    def sample_latent_vector(self):
        pass
