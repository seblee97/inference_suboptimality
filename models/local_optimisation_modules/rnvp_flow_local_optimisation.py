import torch

from .base_local_optimisation import _BaseLocalOptimisation
from models.approximate_posteriors import RNVPPosterior

from typing import List, Iterable

class RNVPLocalOptimisation(_BaseLocalOptimisation):
    """
    *NFlow from paper
    """
    def __init__(self, config):
        _BaseLocalOptimisation.__init__(self, config)

        self._flow_module = None
        self._config = config

    def get_additional_parameters(self) -> Iterable:
        self._flow_module = RNVPPosterior(config=self._config)
        flow_parameters = self._flow_module.parameters()
        return flow_parameters

    def sample_latent_vector(self, params: torch.Tensor) -> (torch.Tensor, List):
        params_concat = torch.cat([params[0], params[1]], dim=1)
        z, [mean, log_var, z0, log_det_jacobian] = self._flow_module.sample(params_concat)
        return z, [mean, log_var, z0, log_det_jacobian]
