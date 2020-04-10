from .base_norm_flow import BaseFlow
from .base_approximate_posterior import approximatePosterior

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

from typing import Dict

class PlanarPosterior(approximatePosterior, BaseFlow):

    """Applied Planar normalising flows (https://github.com/riannevdberg/sylvester-flows/blob/master/models/VAE.py)
    Varientional auto-encoders with normalizing flows, applied in https://github.com/e-hulten/planar-flows/blob/master/transform.py#L4
    both applying paper: https://arxiv.org/pdf/1803.05649.pdf"""

    def __init__(self, config: Dict):

        # get architecture of flows from config
        self.num_flow_transformations = config.get(["flow", "num_flow_transformations"])
        self.num_flow_passes = config.get(["flow", "num_flow_passes"])

        # also equivalent to s, t in https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models#realnvp
        self.u_flow_layers = config.get(["flow", "flow_layers"])
        self.w_flow_layers = config.get(["flow", "flow_layers"])
        self.b_flow_layers = config.get(["flow", "flow_layers"])

        # input to flow maps will be latent dimension (i.e. output of first part of inference network)
        self.input_dimension = config.get(["model", "latent_dimension"])
        self.batch_size =  self.input_dimension * 2 #config.get(["training", "batch_size"])


        self.noise_distribution = tdist.Normal(torch.Tensor([0]), torch.Tensor([1]))

        approximatePosterior.__init__(self, config)
        BaseFlow.__init__(self, config)

    def _construct_layers(self):


        self.flow_u_modules = nn.ModuleList([])
        self.flow_w_modules = nn.ModuleList([])
        self.flow_b_modules = nn.ModuleList([])

        self.flow_u_modules.append(self._initialise_weights(nn.Linear(self.batch_size, self.num_flow_transformations * self.input_dimension)))
        self.flow_w_modules.append(self._initialise_weights(nn.Linear(self.batch_size, self.num_flow_transformations * self.input_dimension)))
        self.flow_b_modules.append(self._initialise_weights(nn.Linear(self.batch_size, self.num_flow_transformations)))

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


        #print("z0: ", z0.shape)
        zk = z0.unsqueeze(2)
        #print("zk: ", zk.shape)

        uw = torch.bmm(w, u)
        m_uw = -1. + F.softplus(uw)
        #print("m_uw: ", m_uw.shape)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)

        # compute flow with u_hat

        #print("w: ", w.shape)
        #print("zk: ", zk.shape)
        #print("b: ", b.shape)
        #print("u_hat: ", u_hat.shape)

        wz = torch.bmm(w, zk)
        wzb = wz + b
        #print("activation: ", self.activation(wzb).shape)
        uwzb = u_hat * self.activation(wzb)
        z = zk + uwzb
        #print("z 1: ", z.shape)
        z = z.squeeze(2)
        #print("z 2: ", z.shape)

        # this computes the jacobian determinant (sec 2.1 of above paper )
        psi = w * self.deriv_tanh(wzb)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)

        #print("log det pass: ", log_det_jacobian.shape)


        return z, log_det_jacobian

    def sample(self, parameters):

        #print("h.shape 1: ", parameters.shape)
        h = parameters.view(-1, self.input_dimension * 2)

        #print("h.shape: ", h.shape)

        # reparameterization trick
        z0, mean, log_var = self._reparameterise(h)
        log_det_jacobian = torch.zeros(z0.shape[0])
        z = z0

        # pass resized parameters into u, w, b to parametrize the matrices
        u0 = self.flow_u_modules[0](h)
        u = u0.view(-1, self.num_flow_transformations, self.input_dimension, 1)
        w0 = self.flow_w_modules[0](h)
        w = w0.view(-1, self.num_flow_transformations, 1, self.input_dimension)
        b0 = self.flow_b_modules[0](h)
        b = b0.view(-1, self.num_flow_transformations, 1, 1)

        # check matrix size, should be [100, 50] then [100,50,1]
        #print("u shape ", u0.shape)
        #print("u shape: ", u.shape)


        # run flow transformations using latent + u, w, b parameters

        for k in range(0, self.num_flow_transformations):
            z, pass_log_det_jacobian = self.forward(z, u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            log_det_jacobian += pass_log_det_jacobian

        return z, [mean, log_var, z0, log_det_jacobian]
