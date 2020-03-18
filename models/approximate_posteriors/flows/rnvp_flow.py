from .base_flow import BaseFlow
from .scale_map import ScaleMap
from .translate_map import TranslateMap

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

class RNVP(BaseFlow):
    # TODO: doc

    def __init__(self, config: Dict):

        # half of the latent z size as input
        self.input_dimension = config.get(["flow", "input_dim"])
        self.latent_dimension = config.get(["flow", "latent_dim"])
        self.hidden_dimensions = config.get(["flow", "flow_layers"])
        self.activation_function = config.get(["flow", "nonlinearity"])

        # The activation function is defined at mother class level as well as forward

        BaseFlow.__init__(self, config)

        # 4 mapping functions
        self.sigma1 = ScaleMap(config=config)
        self.mu1 = TranslateMap(config=config)
        self.sigma2 = ScaleMap(config=config)
        self.mu2 = TranslateMap(config=config)


    def _construct_layers(self):

        self.layers = nn.ModuleList([])

        input_layer = self._initialise_weights(nn.Linear(self.input_dimension, self.hidden_dimensions[0]))
        self.layers.append(input_layer)

        for h in range(len(self.hidden_dimensions[:-1])):
            hidden_layer = self._initialise_weights(nn.Linear(self.hidden_dimensions[h], self.hidden_dimensions[h + 1]))
            self.layers.append(hidden_layer)

        # final layer to latent dim
        hidden_to_latent_layer = self._initialise_weights(nn.Linear(self.hidden_dimensions[-1], self.latent_dimension))
        self.layers.append(hidden_to_latent_layer)

    def activation_derivative(self, x):
        """ Derivative of tanh """
        return 1 - self.activation(x) ** 2

    def forward(self, parameters):

        cutoff = int(parameters.shape[1] / 2)

        # separate latent vector into two variables
        z1 = parameters[:, :cutoff]
        z2 = parameters[:, cutoff:]
        #print(z2)

        # flow: za' = za * exp(sigma(zb)) + mu(zb)

        sigma_z2 = self.sigma1(self.activation(z2))
        mu_z2 = self.mu1(self.activation(z2))
        z_prime = z1 * torch.exp(sigma_z2) + mu_z2 # does not match comment below, no b

        sigma_zp = self.sigma2(self.activation(z_prime))
        mu_zp = self.mu2(self.activation(z_prime))
        z2_prime = z2 * torch.exp(sigma_zp) + mu_zp

        #now calculate Jacobian log det

        z = torch.cat([z_prime, z2_prime], dim=0)

        # det(J) = exp(sigma) ** input dimension.det(AB) = det(A)det(B)
        det_jacobian = (torch.exp(sigma_z2) ** cutoff) * (torch.exp(sigma_zp) ** cutoff)
        log_det_jacobian = cutoff * (sigma_z2 + sigma_zp)

        '''
        # BELOW HERE IS STUFF I HAVENT TOUCHED, just commented out

        # sample noise (reparameterisation trick), unsqueeze to match dimensions
        #noise = self.noise_distribution.sample(z2.shape).squeeze()
        #z0 = z1 + torch.sqrt(z2) * noise

        #z_through_flow = []

        #log_det_jacobian_sum = 0


        """
        params =zk, u, w, b

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

        zk = zk.unsqueeze(2)

        # reparameterize u such that the flow becomes invertible (see appendix paper)
        uw = torch.bmm(w, u)
        m_uw = -1. + self.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)

        # compute flow with u_hat
        wzb = torch.bmm(w, zk) + b
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)

        # compute logdetJ
        psi = w * self.der_h(wzb)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)
        '''

        return z, log_det_jacobian