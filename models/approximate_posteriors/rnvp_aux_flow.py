from .base_norm_flow import BaseFlow

import torch
import torch.nn as nn

class RNVPAux(BaseFlow):
    # TODO: doc

    def __init__(self, config):

        BaseFlow.__init__(self, config)

        self.latent_dimension = config.get(["training", "latent_dimension"])
        self.auxillary_forward_dimensions = config.get(["flow", "auxillary_forward_dimensions"])
        self.auxillary_reverse_dimensions = config.get(["flow", "auxillary_reverse_dimensions"])

        self.activation = nn.Elu()
        self.softplus = nn.Softplus()

    def _construct_layers(self) -> None:

        # q(v|x,z)
        self.auxillary_forward_layers = nn.ModuleList([])

        input_auxillary_forward_layer = self._initialise_weights(nn.Linear(self.input_dimension, self.auxillary_forward_dimensions[0]))
        self.auxillary_forward_layers.append(input_auxillary_forward_layer)

        for h in range(len(self.auxillary_forward_dimensions[:-1])):
            hidden_layer = self._initialise_weights(nn.Linear(self.auxillary_forward_dimensions[h], self.auxillary_forward_dimensions[h + 1]))
            self.auxillary_forward_layers.append(hidden_layer)

        # final layer back to latent dim
        output_auxillary_forward_layer = self._initialise_weights(nn.Linear(self.auxillary_forward_dimensions[-1], self.latent_dimension))
        self.auxillary_forward_layers.append(output_auxillary_forward_layer)

        # r(v|x,z)
        self.auxillary_reverse_layers = nn.ModuleList([])

        input_auxillary_reverse_layer = self._initialise_weights(nn.Linear(self.input_dimension, self.auxillary_reverse_dimensions[0]))
        self.auxillary_reverse_layers.append(input_auxillary_reverse_layer)

        for h in range(len(self.reverse_auxillary_reverse_dimensions[:-1])):
            hidden_layer = self._initialise_weights(nn.Linear(self.auxillary_reverse_dimensions[h], self.auxillary_reverse_dimensions[h + 1]))
            self.auxillary_reverse_layers.append(hidden_layer)

        # final layer back to latent dim
        output_auxillary_reverse_layer = self._initialise_weights(nn.Linear(self.auxillary_reverse_dimensions[-1], self.latent_dimension))
        self.auxillary_reverse_layers.append(output_auxillary_reverse_layer)

    def activation_derivative(self, x):
        """ Derivative of tanh """
        return 1 - self.activation(x) ** 2

    def forward(self, z: torch.Tensor):
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

        auxillary_variable_vector_params = self._auxillary_forward(z)

        cutoff = int(0.5 * auxillary_variable_vector_params.shape[1])

        v0_mu = auxillary_variable_vector_params[:, :cutoff]
        v0_var = auxillary_variable_vector_params[:, cutoff:]

        # sample noise (reparameterisation trick), unsqueeze to match dimensions
        noise = self.noise_distribution.sample(v0_var.shape).squeeze()

        auxillary_variable_vector = v0_mu + torch.sqrt(v0_var) * noise





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

    def _auxillary_forward(self, z: torch.Tensor):

        v = z

        for layer in self.auxillary_forward_layers:
            v = self.activation(layer(v))

        return v


