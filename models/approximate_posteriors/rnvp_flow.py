from .base_norm_flow import BaseFlow
from .base_approximate_posterior import approximatePosterior

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

class RNVPPosterior(approximatePosterior, BaseFlow):
    
    """Applied Real-NVP normalising flows (Dinh et al https://arxiv.org/abs/1605.08803)"""

    def __init__(self, config: Dict):

        # get architecture of flows from config
        self.num_flows = config.get(["flow", "num_flows"])
        # sigma and mu from eq 9, 10 in paper https://arxiv.org/pdf/1801.03558.pdf
        # also equivalent to s, t in https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models#realnvp
        self.sigma_flow_layers = config.get(["flow", "flow_layers"])
        self.mu_flow_layers = config.get(["flow", "flow_layers"])

        # input to flow maps will be half of latent dimension (i.e. output of first part of inference network)
        self.input_dimension = config.get(["model", "latent_dimension"]) // 2

        assert (self.num_flows == len(self.sigma_flow_layers)) and (self.num_flows == len(self.mu_flow_layers)), \
            "Number of flows (num_flows) does not match flow layers specified"

        approximatePosterior.__init__(self, config)
        BaseFlow.__init__(self, config)

    def _construct_layers(self):

        self.sigma_maps = nn.ModuleList([])
        self.mu_maps = nn.ModuleList([])

        # initiate mapping functions according to config spec
        self.sigma_maps.append(self._initialise_weights(nn.Linear(self.input_dimension, self.sigma_flow_layers[0])))
        self.mu_maps.append(self._initialise_weights(nn.Linear(self.input_dimension, self.mu_flow_layers[0])))

        for f in range(len(self.sigma_flow_layers[:-1])):
            self.sigma_maps.append(self._initialise_weights(nn.Linear(self.sigma_flow_layers[f], self.sigma_flow_layers[f + 1])))
            self.mu_maps.append(self._initialise_weights(nn.Linear(self.mu_flow_layers[f], self.mu_flow_layers[f + 1])))
        
        self.sigma_maps.append(self._initialise_weights(nn.Linear(self.sigma_flow_layers[-1], self.input_dimension)))
        self.mu_maps.append(self._initialise_weights(nn.Linear(self.mu_flow_layers[-1], self.input_dimension)))

    def forward(self, parameters):

        z = parameters

        z1 = z[:, :self.input_dimension]
        z2 = z[:, self.input_dimension:]

        def apply_flow(partitioned_parameter: torch.Tensor, map_index: int) -> torch.Tensor:
            return self.sigma_maps[map_index](partitioned_parameter) + self.mu_maps[map_index](partitioned_parameter)

        # ensure flow is applied to whole part of latent by alternating half to which flow is applied.
        # in each flow transformation jacobian remains triangular because half is unchanged.
        apply_flow_to_top_partition = True

        # this implements 9, 10 from https://arxiv.org/pdf/1801.03558.pdf
        for f in range(self.num_flows):
            if apply_flow_to_top_partition:
                z1 = z1 * (self.sigma_maps[f](z2) + self.mu_maps[f](z2))
            else:
                z2 = z2 * (self.sigma_maps[f](z1) + self.mu_maps[f](z1))

            apply_flow_to_top_partition = not apply_flow_to_top_partition

        z = torch.cat([z1, z2], dim=1)

        return z, None

        # # det(J) = exp(sigma) ** input dimension.det(AB) = det(A)det(B)
        # det_jacobian = (torch.exp(sigma_z2) ** cutoff) * (torch.exp(sigma_zp) ** cutoff)
        # log_det_jacobian = cutoff * (sigma_z2 + sigma_zp)

        # return z, log_det_jacobian

    def sample(self, parameters):
        # apply flow transformations
        z, log_det_jacobian = self.forward(parameters)

        #for k in range(self.num_flows):
        #    z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
        #    z_through_flow.append(z_k)
        #    log_det_jacobian_sum += log_det_jacobian

        #latent_vector = z_through_flow[-1]

        return z, [log_det_jacobian]
