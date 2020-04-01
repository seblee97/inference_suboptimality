import torch

from .base_local_ammortisation import BaseLocalAmmortisation
from models.approximate_posteriors import RNVPAux

class RNVPAuxLocalAmmortisation(BaseLocalAmmortisation):
    """
    *AF from paper
    """
    def __init__(self, config):
        BaseLocalAmmortisation.__init__(self, config)

        self._flow_module = None
        self._config = config

    def sample_latent_vector(self):
        pass
