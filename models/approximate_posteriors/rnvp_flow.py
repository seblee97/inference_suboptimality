from .base_norm_flow import BaseFlow
from .base_approximate_posterior import approximatePosterior

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

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

        self.noise_distribution = tdist.Normal(torch.Tensor([0]), torch.Tensor([1]))

        approximatePosterior.__init__(self, config)
        BaseFlow.__init__(self, config)

    def _construct_layers(self):

        self.flow_sigma_modules = nn.ModuleList([])
        self.flow_mu_modules = nn.ModuleList([])

        for f in range(self.num_flows):

            sigma_map = nn.ModuleList([])
            mu_map = nn.ModuleList([])

            # initiate mapping functions according to config spec
            sigma_map.append(self._initialise_weights(nn.Linear(self.input_dimension, self.sigma_flow_layers[0])))
            mu_map.append(self._initialise_weights(nn.Linear(self.input_dimension, self.mu_flow_layers[0])))

            for f in range(len(self.sigma_flow_layers[:-1])):
                sigma_map.append(self._initialise_weights(nn.Linear(self.sigma_flow_layers[f], self.sigma_flow_layers[f + 1])))
                mu_map.append(self._initialise_weights(nn.Linear(self.mu_flow_layers[f], self.mu_flow_layers[f + 1])))
            
            sigma_map.append(self._initialise_weights(nn.Linear(self.sigma_flow_layers[-1], self.input_dimension)))
            mu_map.append(self._initialise_weights(nn.Linear(self.mu_flow_layers[-1], self.input_dimension)))

            self.flow_sigma_modules.append(sigma_map)
            self.flow_mu_modules.append(mu_map)

        #     # initiate mapping functions according to config spec
        #     self.add_module("flow_{}_sigma_layer_0".format(f), self._initialise_weights(nn.Linear(self.input_dimension, self.sigma_flow_layers[0])))
        #     self.add_module("flow_{}_mu_layer_0".format(f), self._initialise_weights(nn.Linear(self.input_dimension, self.mu_flow_layers[0])))

        #     for l in range(len(self.sigma_flow_layers[:-1])):
        #         self.add_module("flow_{}_sigma_layer_{}".format(f, l + 1), self._initialise_weights(nn.Linear(self.sigma_flow_layers[l], self.sigma_flow_layers[l + 1])))
        #         self.add_module("flow_{}_mu_layer_{}".format(f, l + 1), self._initialise_weights(nn.Linear(self.mu_flow_layers[l], self.mu_flow_layers[l + 1])))
            
        #     self.add_module("flow_{}_sigma_output_layer".format(f), self._initialise_weights(nn.Linear(self.sigma_flow_layers[-1], self.input_dimension)))
        #     self.add_module("flow_{}_mu_output_layer".format(f), self._initialise_weights(nn.Linear(self.mu_flow_layers[-1], self.input_dimension)))

    def _mapping_forward(self, mapping_network: nn.ModuleList, z_partition: torch.Tensor) -> torch.Tensor:
        for layer in mapping_network:
            z_partition = self.activation(layer(z_partition))
        return z_partition

    def forward(self, z0: torch.Tensor):

        z1 = z0[:, :self.input_dimension]
        z2 = z0[:, self.input_dimension:]

        # ensure flow is applied to whole part of latent by alternating half to which flow is applied.
        # in each flow transformation jacobian remains triangular because half is unchanged.
        apply_flow_to_top_partition = True

        log_det_jacobian = torch.zeros(z0.shape[0])

        # this implements 9, 10 from https://arxiv.org/pdf/1801.03558.pdf
        for f in range(self.num_flows):
            if apply_flow_to_top_partition:
                sigma_map = torch.exp(self._mapping_forward(self.flow_sigma_modules[f], z2))
                z1 = z1 * sigma_map + self._mapping_forward(self.flow_mu_modules[f], z2)
            else:
                sigma_map = torch.exp(self._mapping_forward(self.flow_sigma_modules[f], z1))
                z2 = z2 * sigma_map + self._mapping_forward(self.flow_mu_modules[f], z1)
            
            # this computes the jacobian determinant (sec 6.4 in supplementary of paper)
            log_det_transformation = torch.sum(torch.log(sigma_map), axis=1)
            log_det_jacobian += log_det_transformation

            apply_flow_to_top_partition = not apply_flow_to_top_partition

        z = torch.cat([z1, z2], dim=1)

        return z, log_det_jacobian

        # # det(J) = exp(sigma) ** input dimension.det(AB) = det(A)det(B)
        # det_jacobian = (torch.exp(sigma_z2) ** cutoff) * (torch.exp(sigma_zp) ** cutoff)
        # log_det_jacobian = cutoff * (sigma_z2 + sigma_zp)

        # return z, log_det_jacobian

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
        z0 = mean + torch.sqrt(torch.exp(log_var)) * noise

        # The above is identical to gaussian case, now apply flow transformations
        z, log_det_jacobian = self.forward(z0)

        return z, [mean, log_var, z0, log_det_jacobian]
