from .base_flow import BaseFlow
from .nflow_feedforward import NFlowfeedForwardNetwork


import torch.nn as nn
import torch.nn.functional as F


class RNVP(BaseFlow):
    # TODO: doc

    def __init__(self, config):

        BaseFlow.__init__(self, config)

        self.activation = nn.ELU()
        self.softplus = nn.Softplus()
        self.flow_network = NFlowfeedForwardNetwork

    def _construct_layers(self):
        pass

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
