from .base_norm_flow import BaseFlow
from .base_approximate_posterior import approximatePosterior

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

from typing import Dict

class PlanarPosterior(approximatePosterior, BaseFlow):

    """Applied Planar normalising flows (https://github.com/riannevdberg/sylvester-flows/blob/master/models/VAE.py) [repo 1]
    Varientional auto-encoders with normalizing flows, applied in https://github.com/e-hulten/planar-flows/blob/master/transform.py#L4
    both applying paper: https://arxiv.org/pdf/1803.05649.pdf

    https://arxiv.org/pdf/1505.05770.pdf"""

    def __init__(self, config: Dict):

        # get architecture of flows from config
        self.num_flow_transformations = config.get(["flow", "num_flow_transformations"])

        # input to flow maps will be latent dimension (i.e. output of first part of inference network)
        self.input_dimension = config.get(["model", "latent_dimension"])
        self.batch_size = config.get(["training", "batch_size"])

        self.noise_distribution = tdist.Normal(torch.Tensor([0]), torch.Tensor([1]))

        approximatePosterior.__init__(self, config)
        BaseFlow.__init__(self, config)

    def _construct_layers(self):

        self.flow_u_modules = self._initialise_weights(nn.Linear(2 * self.input_dimension, self.num_flow_transformations * self.input_dimension))
        self.flow_w_modules = self._initialise_weights(nn.Linear(2 * self.input_dimension, self.num_flow_transformations * self.input_dimension))
        self.flow_b_modules = self._initialise_weights(nn.Linear(2 * self.input_dimension, self.num_flow_transformations))

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

    def forward(self, zk: torch.Tensor, u, w, b):

        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b) (section 2.1, equation 10 from paper cited above)
        or actually
        z'^T = z^T + h(z^T w + b)u^T

        u, w, b are matrices parametrized by a linear map.
        z is the latent
        h() is a smooth activation function (here, tanh)

        Assumes the following input shapes: (from repo cited above)
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """      
        # following flow code adapted from [repo 1]

        uw = torch.sum(w * u, dim=1, keepdim=True)
        m_uw = -1. + F.softplus(uw)
        w_norm_sq = torch.sum(w * w, dim=1, keepdim=True)
        u_hat = u + ((m_uw - uw) * w / w_norm_sq) # (u^T)

        # compute flow with u_hat

        # multiply w^T and z
        wz = torch.sum(w * zk, dim=1, keepdim=True)
        wzb = wz + b
        #print("activation: ", self.activation(wzb).shape)
        uwzb = u_hat * self.activation(wzb)

        # f(z) =  z + u h (w^T z + b) = zk + u_hat * tanh(wzb) (10)
        z_out = zk + uwzb

        # this computes the jacobian determinant (sec 4.1, eq 11-12 of above paper )
        # psi(z) = w * h'(w^T z + b) (11)
        # from our variables:
        # psi = w * tanh'(wz +b) = w * tanh'(wzb) (w transposed)
        psi = w * self.deriv_tanh(wzb)
        # logDJ = |det (I + u psi z^T)| = |1 + u^T psi(z)| (12)
        uTpsi = torch.sum(psi * u_hat, dim=1)
        log_det_jacobian = torch.log(torch.abs(1 + uTpsi))

        return z_out, log_det_jacobian

    def sample(self, parameters):

        # reparameterization trick
        z0, mean, log_var = self._reparameterise(parameters)
        log_det_jacobian = torch.zeros(z0.shape[0])
        z = z0

        # pass resized parameters into u, w, b to parametrize the matrices
        u = self.flow_u_modules(parameters).view(-1, self.num_flow_transformations, self.input_dimension)
        w = self.flow_w_modules(parameters).view(-1, self.num_flow_transformations, self.input_dimension)
        b = self.flow_b_modules(parameters).view(-1, self.num_flow_transformations, 1)

        # run flow transformations using latent + u, w, b parameters

        for k in range(self.num_flow_transformations):
            z, pass_log_det_jacobian = self.forward(z, u[:, k, :], w[:, k, :], b[:, k])
            log_det_jacobian += pass_log_det_jacobian

        return z, [mean, log_var, z0, log_det_jacobian]
