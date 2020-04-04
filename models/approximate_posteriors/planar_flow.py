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
        self.input_dimension = config.get(["model", "latent_dimension"])
        self.batch_size = config.get(["training", "batch_size"])

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
        self.flow_u_modules.append(self._initialise_weights(nn.Linear(self.batch_size, self.num_flow_transformations * self.input_dimension)))
        self.flow_w_modules.append(self._initialise_weights(nn.Linear(self.batch_size, self.num_flow_transformations * self.input_dimension)))
        self.flow_b_modules.append(self._initialise_weights(nn.Linear(self.batch_size, self.num_flow_transformations)))


    def _mapping_forward(self, mapping_network: nn.ModuleList, z_partition: torch.Tensor) -> torch.Tensor:
        for layer in mapping_network:
            z_partition = self.activation(layer(z_partition))
        return z_partition

    def deriv_tanh(self, x):
        return 1 - self.activation(x) ** 2

    def _reparameterise(self, raw_vector: torch.Tensor) -> torch.Tensor:
        dimensions = raw_vector.shape[1] // 2
        mean = raw_vector[:, :dimensions]
        log_var = raw_vector[:, dimensions:]

        # Sample the standard deviation along each dimension of the factorized Gaussian posterior.
        noise = self.noise_distribution.sample(log_var.shape).squeeze()
        # Apply the reparameterization trick.
        reparameterised_vector = mean + torch.sqrt(torch.exp(log_var)) * noise

        return reparameterised_vector, mean, log_var

    def forward(self, z0: torch.Tensor, u, w, b):

        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """

        #z1 = z0[:, :self.input_dimension]
        #z2 = z0[:, self.input_dimension:]
        #zk = zk.unsqueeze(2)
        print("z0: ", z0.shape)
        zk = z0.unsqueeze(1)
        zk = zk.unsqueeze(1)
        print("zk: ", zk.shape)
        #print(zk)


        uw = u * w
        m_uw = -1. + F.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)

        # compute flow with u_hat
        print("w: ", w.shape)
        print("zk: ", zk.shape)
        print("b: ", b.shape)
        print("u_hat: ", u_hat.shape)
        wz = (w * zk)
        wzb = wz + b
        z = zk + u_hat * self.activation(wzb)
        z = z.squeeze(2)

        # this computes the jacobian determinant (sec 6.4 in supplementary of paper)
        psi = w * self.deriv_tanh(wzb)
        log_det_jacobian = torch.log(torch.abs(1 + (psi * u_hat)))
        #print(log_det_jacobian)
        #log_det_jacobian += log_det_transformation


        #z = torch.cat([z1, z2], dim=1)

        return z, log_det_jacobian

    def sample(self, parameters):
        # There are two dimensions to |parameters|; the second one consists of two halves:
        #   1. The first half represents the multidimensional mean of the Gaussian posterior.
        #   2. The second half represents the multidimensional log variance of the Gaussian posterior.
        dimensions = parameters.shape[1] // 2
        h = parameters #.view(-1, self.batch_size)
        #mean = parameters[:, :dimensions]
        #log_var = parameters[:, dimensions:]
        #print(h)

        # calculate u , w, b before reparameterization trick
        #h = parameters.view(-1, self.input_dimension * 2)
        u0 = self.flow_u_modules[0](h).view(self.batch_size, self.num_flow_transformations, self.input_dimension, 1)
        w0 = self.flow_w_modules[0](h).view(self.batch_size, self.num_flow_transformations, 1, self.input_dimension)
        b0 = self.flow_b_modules[0](h).view(self.batch_size, self.num_flow_transformations, 1, 1)

        z0, mean, log_var = self._reparameterise(parameters)

        # The above is essentially identical to gaussian case, now apply flow transformations
        log_det_jacobian = torch.zeros(z0.shape[0])
        z = z0
        u = w0
        w = w0
        b = b0

        #print("u: ", u)

        # pass latent sample through same flow module multiple times
        # !note! distinction between num_flow_transformations and num_flow_passes
        for k in range(self.num_flow_passes):
            z_k, pass_log_det_jacobian = self.forward(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            z.append(z_k)
            log_det_jacobian += pass_log_det_jacobian

        #print(z, mean, log_var, z0, log_det_jacobian)

        return (z, [mean, log_var, z0, log_det_jacobian])
