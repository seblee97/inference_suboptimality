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
        self.latent_dimension = config.get(["model", "latent_dimension"])

        self.noise_distribution = tdist.Normal(torch.Tensor([0]), torch.Tensor([1]))

        approximatePosterior.__init__(self, config)
        BaseFlow.__init__(self, config)

        if self.activation_function == 'tanh':
            self._reparameterise_u = self._tanh_reparameterise_u
        else:
            raise ValueError("Enforcing invertibility condition by reparameterising u currently only implemented for tanh nonlinearity")

    def _construct_layers(self):
        # input to ammortization networks for u, w, b are 2 * latent_dimension, 
        # since this is output of first part of inference network (halved in reparamterisation)
        self.flow_u_modules = self._initialise_weights(nn.Linear(2 * self.latent_dimension, self.num_flow_transformations * self.latent_dimension))
        self.flow_w_modules = self._initialise_weights(nn.Linear(2 * self.latent_dimension, self.num_flow_transformations * self.latent_dimension))
        self.flow_b_modules = self._initialise_weights(nn.Linear(2 * self.latent_dimension, self.num_flow_transformations))

    def _reparameterise(self, raw_vector: torch.Tensor) -> torch.Tensor:
        dimensions = raw_vector.shape[1] // 2
        mean = raw_vector[:, :dimensions]
        log_var = raw_vector[:, dimensions:]

        # Sample the standard deviation along each dimension of the factorized Gaussian posterior.
        noise = self.noise_distribution.sample(log_var.shape).squeeze()

        # Apply the reparameterization trick.
        reparameterised_vector = mean + torch.sqrt(torch.exp(log_var)) * noise

        return reparameterised_vector, mean, log_var

    def _tanh_reparameterise_u(self, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        uw = torch.sum(w * u, dim=1, keepdim=True)
        m_uw = -1. + F.softplus(uw)
        w_norm_sq = torch.sum(w * w, dim=1, keepdim=True)
        u_hat = u + ((m_uw - uw) * w / w_norm_sq) # (u^T)
        return u_hat

    def forward(self, zk: torch.Tensor, u: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b) (section 2.1, equation 10 from paper cited above)
        or actually
        z'^T = z^T + h(z^T w + b)u^T

        u, w, b are matrices parametrized by a linear map.
        z is the latent
        h() is a smooth activation function

        Assumes the following input shapes:
        shape u = (batch_size, latent_dimension)
        shape w = (batch_size, latent_dimension)
        shape b = (batch_size, 1)
        shape z = (batch_size, latent_dimension).
        """      
        # reparameterise u to satisfy invertibility conditions
        u_hat = self._reparameterise_u(u, w)

        # compute flow with u_hat
        # f(z) =  z + \hat{u} (w^T\cdot z + b)
        wz = torch.sum(w * zk, dim=1, keepdim=True)
        wzb = wz + b
        z_out = zk + u_hat * self.activation(wzb)

        # this computes the jacobian determinant (sec 4.1, eq 11-12 of above paper)
        # psi(z) = w * h'(w^T z + b) (11)
        # from our variables:
        # psi = w * tanh'(wz +b) = w * tanh'(wzb) (w transposed)
        psi = w * self.activation_derivative(wzb)
        # logDJ = log(|det (I + u psi z^T)|) = log(|1 + u^T psi(z)|) (12)
        uTpsi = torch.sum(psi * u_hat, dim=1)
        log_det_jacobian = torch.log(torch.abs(1 + uTpsi))

        return z_out, log_det_jacobian

    def sample(self, parameters):

        # reparameterization trick
        z0, mean, log_var = self._reparameterise(parameters)
        log_det_jacobian = torch.zeros(z0.shape[0])
        z = z0

        # pass resized parameters into u, w, b to parametrize the matrices
        u = self.flow_u_modules(parameters).view(-1, self.num_flow_transformations, self.latent_dimension)
        w = self.flow_w_modules(parameters).view(-1, self.num_flow_transformations, self.latent_dimension)
        b = self.flow_b_modules(parameters).view(-1, self.num_flow_transformations, 1)

        # run flow transformations using latent + u, w, b parameters
        for k in range(self.num_flow_transformations):
            z, flow_log_det_jacobian = self.forward(z, u[:, k, :], w[:, k, :], b[:, k])
            log_det_jacobian += flow_log_det_jacobian

        return z, [mean, log_var, z0, log_det_jacobian]
