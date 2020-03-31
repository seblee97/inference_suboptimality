from .base_norm_flow import BaseFlow
from .base_approximate_posterior import approximatePosterior

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

from typing import Dict

class PlanarPosterior(approximatePosterior, BaseFlow):

    """Applied Planar normalising flows (https://github.com/riannevdberg/sylvester-flows/blob/master/models/VAE.py)"""

    def __init__(self, config: Dict):

        # get architecture of flows from config
        self.num_flow_transformations = config.get(["flow", "num_flow_transformations"])
        self.num_flow_passes = config.get(["flow", "num_flow_passes"])
        # sigma and mu from eq 9, 10 in paper https://arxiv.org/pdf/1801.03558.pdf
        # also equivalent to s, t in https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models#realnvp
        self.u_flow_layers = config.get(["flow", "flow_layers"])
        self.w_flow_layers = config.get(["flow", "flow_layers"])
        self.b_flow_layers = config.get(["flow", "flow_layers"])

        # input to flow maps will be half of latent dimension (i.e. output of first part of inference network)
        self.input_dimension = config.get(["model", "latent_dimension"]) // 2

        #assert (self.num_flow_transformations == len(self.sigma_flow_layers)) and (self.num_flow_transformations == len(self.mu_flow_layers)), \
        #    "Number of flows (num_flow_transformations) does not match flow layers specified"

        self.noise_distribution = tdist.Normal(torch.Tensor([0]), torch.Tensor([1]))

        approximatePosterior.__init__(self, config)
        BaseFlow.__init__(self, config)

    def _construct_layers(self):

        self.flow_u_modules = nn.ModuleList([])
        self.flow_w_modules = nn.ModuleList([])
        self.flow_b_modules = nn.ModuleList([])

        # do we need weights here?
        self.flow_u_modules.append(self._initialise_weights(nn.Linear(2 * self.input_dimension, self.input_dimension)))
        self.flow_w_modules.append(self._initialise_weights(nn.Linear(2 * self.input_dimension, self.input_dimension)))
        self.flow_u_modules.append(self._initialise_weights(nn.Linear(2 * self.input_dimension, self.num_flow_transformations)))


    def _mapping_forward(self, mapping_network: nn.ModuleList, z_partition: torch.Tensor) -> torch.Tensor:
        for layer in mapping_network:
            z_partition = self.activation(layer(z_partition))
        return z_partition

    def deriv_tanh(self, x):
        return 1 - self.activation(x) ** 2

    def forward(self, z0: torch.Tensor, u, w, b):

        #z1 = z0[:, :self.input_dimension]
        #z2 = z0[:, self.input_dimension:]
        zk = zk.unsqueeze(2)

        uw = torch.bmmw(w, u)
        m_uw = -1. + self.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)

        # compute flow with u_hat
        wzb = torch.bmm(w, zk) + b
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)

        # this computes the jacobian determinant (sec 6.4 in supplementary of paper)
        psi = w * self.deriv_tanh(wzb)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        #log_det_jacobian += log_det_transformation


        #z = torch.cat([z1, z2], dim=1)

        return z, log_det_jacobian

    def sample(self, parameters):
        # There are two dimensions to |parameters|; the second one consists of two halves:
        #   1. The first half represents the multidimensional mean of the Gaussian posterior.
        #   2. The second half represents the multidimensional log variance of the Gaussian posterior.
        dimensions = parameters.shape[1] // 2
        mean = parameters[:, :dimensions]
        log_var = parameters[:, dimensions:]

        # calculate u , w, b before reparameterization trick
        h = parameters.view(-1, self.input_dimension * 2)
        u0 = self.flow_u_modules(h).view(self.input_dimension * 2, self.num_flow_transformations, self.input_dimension, 1)
        w0 = self.flow_w_modules(h).view(self.input_dimension * 2, self.num_flow_transformations, 1, self.input_dimension)
        b0 = self.flow_b_modules(h).view(self.input_dimension * 2, self.num_flow_transformations, 1, 1)


        # Sample the standard deviation along each dimension of the factorized Gaussian posterior.
        noise = self.noise_distribution.sample(log_var.shape).squeeze()
        # Apply the reparameterization trick.
        z0 = mean + torch.sqrt(torch.exp(log_var)) * noise

        # The above is essentially identical to gaussian case, now apply flow transformations
        log_det_jacobian = torch.zeros(z0.shape[0])
        z = z0
        u = w0
        w = w0
        b = b0

        # pass latent sample through same flow module multiple times
        # !note! distinction between num_flow_transformations and num_flow_passes
        for f in range(self.num_flow_passes):
            z_k, pass_log_det_jacobian = forward(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            z.append(z_k)
            log_det_jacobian += pass_log_det_jacobian

        print(z, [mean, log_var, z0, log_det_jacobian])

        return z, [mean, log_var, z0, log_det_jacobian]
