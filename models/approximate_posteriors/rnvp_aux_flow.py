from .base_norm_flow import BaseFlow
from .base_approximate_posterior import approximatePosterior

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

from typing import Dict

class RNVPAux(approximatePosterior, BaseFlow):
    
    """Applied Real-NVP normalising flows (Dinh et al https://arxiv.org/abs/1605.08803)"""

    def __init__(self, config: Dict):

        # get architecture of flows from config
        self.num_flows = config.get(["flow", "num_flows"])
        # sigma and mu from eq 9, 10 in paper https://arxiv.org/pdf/1801.03558.pdf
        # also equivalent to s, t in https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models#realnvp
        self.sigma_flow_layers = config.get(["flow", "flow_layers"])
        self.mu_flow_layers = config.get(["flow", "flow_layers"])

        self.auxillary_forward_dimensions = config.get(["flow", "auxillary_forward_dimensions"])
        self.auxillary_reverse_dimensions = config.get(["flow", "auxillary_reverse_dimensions"])

        # input to flow maps will be latent dimension (i.e. output of first part of inference network)
        # this is in contrast to the non-auxillary case where the latent is split in two.
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

            # initialise forward model for auxillary variables - q(v|x,z)
            self.auxillary_forward_map = nn.ModuleList([])

            input_auxillary_forward_layer = self._initialise_weights(nn.Linear(self.input_dimension, self.auxillary_forward_dimensions[0]))
            self.auxillary_forward_map.append(input_auxillary_forward_layer)

            for h in range(len(self.auxillary_forward_dimensions[:-1])):
                hidden_layer = self._initialise_weights(nn.Linear(self.auxillary_forward_dimensions[h], self.auxillary_forward_dimensions[h + 1]))
                self.auxillary_forward_map.append(hidden_layer)

            # output of auxillary map needs to be twice input since it will be halved in reparamterisation
            output_auxillary_forward_layer = self._initialise_weights(nn.Linear(self.auxillary_forward_dimensions[-1], 2 * self.input_dimension))
            self.auxillary_forward_map.append(output_auxillary_forward_layer)

            # initialise reverse model for auxillary variables - r(v|x,z)
            self.auxillary_reverse_map = nn.ModuleList([])

            input_auxillary_reverse_layer = self._initialise_weights(nn.Linear(2 * self.input_dimension, self.auxillary_reverse_dimensions[0]))
            self.auxillary_reverse_map.append(input_auxillary_reverse_layer)

            for h in range(len(self.auxillary_reverse_dimensions[:-1])):
                hidden_layer = self._initialise_weights(nn.Linear(self.auxillary_reverse_dimensions[h], self.auxillary_reverse_dimensions[h + 1]))
                self.auxillary_reverse_map.append(hidden_layer)

            # output of auxillary map needs to be twice input since it will be halved in reparamterisation (same as for forward aux)
            output_auxillary_reverse_layer = self._initialise_weights(nn.Linear(self.auxillary_reverse_dimensions[-1], 2 * self.input_dimension))
            self.auxillary_reverse_map.append(output_auxillary_reverse_layer)

    def _mapping_forward(self, mapping_network: nn.ModuleList, z_partition: torch.Tensor) -> torch.Tensor:
        for layer in mapping_network:
            z_partition = self.activation(layer(z_partition))
        return z_partition

    def _reparameterise(self, raw_vector: torch.Tensor) -> torch.Tensor:
        dimensions = raw_vector.shape[1] // 2
        mean = raw_vector[:, :dimensions]
        log_var = raw_vector[:, dimensions:]

        # Sample the standard deviation along each dimension of the factorized Gaussian posterior.
        noise = self.noise_distribution.sample(log_var.shape).squeeze()
        # Apply the reparameterization trick.
        reparameterised_vector = mean + torch.sqrt(torch.exp(log_var)) * noise

        return reparameterised_vector, mean, log_var


    def forward(self, z0: torch.Tensor):

        auxillary_forward_output = self._mapping_forward(mapping_network=self.auxillary_forward_map, z_partition=z0)

        # reparameterise auxillary forward output
        v, _, _ = self._reparameterise(auxillary_forward_output)

        z1 = z0
        z2 = v

        # ensure flow is applied to whole part of latent by alternating half to which flow is applied.
        # in each flow transformation jacobian remains triangular because half is unchanged.
        apply_flow_to_top_partition = True

        log_det_jacobian = torch.zeros(z0.shape[0])

        # this implements 9, 10 from https://arxiv.org/pdf/1801.03558.pdf
        for f in range(self.num_flows):
            if apply_flow_to_top_partition:
                sigma_map = self._mapping_forward(self.flow_sigma_modules[f], z2)
                z1 = z1 * torch.exp(sigma_map) + self._mapping_forward(self.flow_mu_modules[f], z2)
            else:
                sigma_map = self._mapping_forward(self.flow_sigma_modules[f], z1)
                z2 = z2 * torch.exp(sigma_map) + self._mapping_forward(self.flow_mu_modules[f], z1)
            
            # this computes the jacobian determinant (sec 6.4 in supplementary of paper)
            log_det_transformation = torch.sum(sigma_map, axis=1)
            log_det_jacobian += log_det_transformation

            apply_flow_to_top_partition = not apply_flow_to_top_partition

        z = torch.cat([z1, z2], dim=1)

        # compute reverse model
        auxillary_reverse_output = self._mapping_forward(mapping_network=self.auxillary_reverse_map, z_partition=z)

        # reparamterise auxillary reverse output
        rv, rv_mean, rv_log_var = self._reparameterise(auxillary_reverse_output)

        return z, log_det_jacobian, rv, rv_mean, rv_log_var

    def sample(self, parameters):
        # There are two dimensions to |parameters|; the second one consists of two halves:
        #   1. The first half represents the multidimensional mean of the Gaussian posterior.
        #   2. The second half represents the multidimensional log variance of the Gaussian posterior.
        z0, mean, log_var = self._reparameterise(parameters)

        # The above is identical to gaussian case, now apply flow transformations
        z, log_det_jacobian, rv, rv_mean, rv_log_var = self.forward(z0)

        return z, [mean, log_var, z0, log_det_jacobian, rv, rv_mean, rv_log_var]
