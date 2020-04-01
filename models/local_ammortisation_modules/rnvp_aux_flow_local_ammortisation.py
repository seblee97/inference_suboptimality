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

    def get_additional_parameters(self):
        self._flow_module = RNVPAux(config=self._config)
        flow_parameters = self._flow_module.parameters()
        return flow_parameters

    def sample_latent_vector(self, params: torch.Tensor):
        params_concat = torch.cat([params[0], params[1]], dim=1)
        z, [mean, log_var, z0, log_det_jacobian, rv, rv_mean, rv_log_var] = self._flow_module.sample(params_concat)
        return z, [mean, log_var, z0, log_det_jacobian, rv, rv_mean, rv_log_var]
