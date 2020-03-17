from .base_flow import BaseFlow

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

        z1 = parameters[:, :cutoff]
        z2 = parameters[:, cutoff:]
        print(z2)
        z2.squeeze()
        print(z2)

        u1_in = self.flow_network(z2)
        u1 = self.F.sigmoid(u1_in)

        w1_in = self.flow_network(z2)
        w1 = self.activation(w1_in)

        z_prime = z1 * u1 + w1 #does not match comment below, no b

        u2_in = self.flow_network(z_prime)
        u2 = self.sigmap(u2_in)

        w2_in = self.flow_network(z_prime)
        w2 = self.activation(w2_in)

        z2_prime = z2 * u2 + w2

        #now calculate Jacobian


        # sample noise (reparameterisation trick), unsqueeze to match dimensions
        #noise = self.noise_distribution.sample(z2.shape).squeeze()
        #z0 = z1 + torch.sqrt(z2) * noise

        z_through_flow = []

        log_det_jacobian_sum = 0


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

        return z, log_det_jacobian
