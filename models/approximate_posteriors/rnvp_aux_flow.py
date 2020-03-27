from .base_norm_flow import BaseFlow
from .base_approximate_posterior import approximatePosterior

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

from typing import Dict

class RNVPAux(approximatePosterior, BaseFlow):
    # TODO: doc

    def __init__(self, config):


        # get architecture of flows from config
        self.num_flows = config.get(["flow", "num_flows"])
        # sigma and mu from eq 9, 10 in paper https://arxiv.org/pdf/1801.03558.pdf
        # also equivalent to s, t in https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models#realnvp
        self.sigma_flow_layers = config.get(["flow", "flow_layers"])
        self.mu_flow_layers = config.get(["flow", "flow_layers"])

        # input to flow maps will be full latent dimension (for both latent and auxiliary)
        self.input_dimension = config.get(["model", "latent_dimension"])
        self.auxillary_input = config.get(["training", "batch_size"])

        assert (self.num_flows == len(self.sigma_flow_layers)) and (self.num_flows == len(self.mu_flow_layers)), \
            "Number of flows (num_flows) does not match flow layers specified"

        self.noise_distribution = tdist.Normal(torch.Tensor([0]), torch.Tensor([1]))

        # aux q(v|z) and r(v|z) have the same dimensions as the sigma/mu mappings 6.1.2

        self.auxillary_forward_dimensions = config.get(["flow", "auxillary_forward_dimensions"])
        self.auxillary_reverse_dimensions = config.get(["flow", "auxillary_reverse_dimensions"])
        #self.auxillary_forward_dimensions = config.get(["flow", "flow_layers"])
        #self.auxillary_reverse_dimensions = config.get(["flow", "flow_layers"])

        approximatePosterior.__init__(self, config)
        BaseFlow.__init__(self, config)


    def _construct_layers(self) -> None:

        # construct sigma and mu flow maps ############

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

        # q(v|x,z) ###################################

        self.auxillary_forward_layers = nn.ModuleList([])

        input_auxillary_forward_layer = self._initialise_weights(nn.Linear(self.auxillary_input, self.auxillary_forward_dimensions[0]))
        self.auxillary_forward_layers.append(input_auxillary_forward_layer)

        for h in range(len(self.auxillary_forward_dimensions[:-1])):
            hidden_layer = self._initialise_weights(nn.Linear(self.auxillary_forward_dimensions[h], self.auxillary_forward_dimensions[h + 1]))
            self.auxillary_forward_layers.append(hidden_layer)

        # final layer back to latent dim
        output_auxillary_forward_layer = self._initialise_weights(nn.Linear(self.auxillary_forward_dimensions[-1], self.auxillary_input))
        self.auxillary_forward_layers.append(output_auxillary_forward_layer)


        # r(v|x,z) ###################################

        self.auxillary_reverse_layers = nn.ModuleList([])

        input_auxillary_reverse_layer = self._initialise_weights(nn.Linear(self.input_dimension, self.auxillary_reverse_dimensions[0]))
        self.auxillary_reverse_layers.append(input_auxillary_reverse_layer)

        for h in range(len(self.auxillary_reverse_dimensions[:-1])):
            hidden_layer = self._initialise_weights(nn.Linear(self.auxillary_reverse_dimensions[h], self.auxillary_reverse_dimensions[h + 1]))
            self.auxillary_reverse_layers.append(hidden_layer)

        # final layer back to latent dim
        output_auxillary_reverse_layer = self._initialise_weights(nn.Linear(self.auxillary_reverse_dimensions[-1], self.auxillary_input))
        self.auxillary_reverse_layers.append(output_auxillary_reverse_layer)

    def activation_derivative(self, x):
        """ Derivative of tanh """
        return 1 - self.activation(x) ** 2

    def _mapping_forward(self, mapping_network: nn.ModuleList, variable: torch.Tensor) -> torch.Tensor:
        """ passes variable through either sigma or mu """

        for layer in mapping_network:
            variable = self.activation(layer(variable))
        return variable

    def _auxillary_flow(self, z: torch.Tensor, direction):
        """ passes auxiliary variable through either forward or reverse model """

        v = z

        if direction == "forward":
            for layer in self.auxillary_forward_layers:
                v = self.activation(layer(v))

        if direction == "reverse":
            for layer in self.auxillary_reverse_layers:
                v = self.activation(layer(v))

        return v

    def aux_forward(self, params: torch.Tensor):

        dimensions = params.shape[1] // 2
        mean = params[:, :dimensions]
        log_var = params[:, dimensions:]

        # Sample the standard deviation along each dimension of the factorized Gaussian posterior.
        noise = self.noise_distribution.sample(log_var.shape).squeeze()
        # Apply the reparameterization trick.
        z0 = mean + torch.sqrt(torch.exp(log_var)) * noise
        log_qz0 = -0.5 * (log_var.sum(1) + ((z0 - mean).pow(2) / torch.exp(log_var)).sum(1))

        # auxiliary flow, forward psss ###################################

        auxillary_variable_vector_params = self._auxillary_flow(params, "forward")

        v0_mean = auxillary_variable_vector_params[:, :dimensions]
        v0_var = auxillary_variable_vector_params[:, dimensions:]

        # sample noise (reparameterisation trick), unsqueeze to match dimensions
        noise = self.noise_distribution.sample(v0_var.shape).squeeze()
        auxillary_variable_vector = v0_mean + torch.sqrt(v0_var) * noise

        log_qv0 = -0.5 * (v0_var.sum(1) + ((auxillary_variable_vector - v0_mean).pow(2) / torch.exp(v0_var)).sum(1))

        return z0, auxillary_variable_vector, log_qv0, log_qz0

    def forward(self, z0: torch.Tensor, v0: torch.Tensor):
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

        zT, vT = z0, v0

        # apply norm flows to zT and vT ###################################

        apply_flow_to_top_partition = True

        log_det_jacobian = torch.zeros(zT.shape[0])

        # this implements 9, 10 from https://arxiv.org/pdf/1801.03558.pdf
        for f in range(self.num_flows):
            if apply_flow_to_top_partition:
                sigma_map = self._mapping_forward(self.flow_sigma_modules[f], vT)
                zT = zT * torch.exp(sigma_map) + self._mapping_forward(self.flow_mu_modules[f], vT)
            else:
                sigma_map = self._mapping_forward(self.flow_sigma_modules[f], zT)
                vT = vT * torch.exp(sigma_map) + self._mapping_forward(self.flow_mu_modules[f], zT)

            # this computes the jacobian determinant (sec 6.4 in supplementary of paper)
            log_det_transformation = torch.sum(sigma_map, axis=1)
            log_det_jacobian += log_det_transformation

            apply_flow_to_top_partition = not apply_flow_to_top_partition

        return zT, vT, log_det_jacobian

    def aux_reverse(self, zT: torch.Tensor, vT: torch.Tensor):

        # auxiliary flow, reverse pass #####################################

        dimensions = self.auxillary_input // 2
        auxillary_variable_reverse = self._auxillary_flow(zT, "reverse")

        vT_mean = auxillary_variable_reverse[:, :dimensions]
        vT_var = auxillary_variable_reverse[:, dimensions:]

        #auxillary_reversee_vector = v0_mean + torch.sqrt(v0_var)
        log_rvT = -0.5 * (vT_var.sum(1) + ((vT - vT_mean).pow(2) / torch.exp(vT_var)).sum(1))

        return auxillary_variable_reverse, vT_mean, vT_var, log_rvT

        # log_prob = log_qv0 - log_det_jacobian - log_rvT

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
        """



    def sample(self, parameters):
        # There are two dimensions to |parameters|; the second one consists of two halves:
        #   1. The first half represents the multidimensional mean of the Gaussian posterior.
        #   2. The second half represents the multidimensional log variance of the Gaussian posterior.
        '''
        dimensions = parameters.shape[1] // 2
        mean = parameters[:, :dimensions]
        log_var = parameters[:, dimensions:]

        # Sample the standard deviation along each dimension of the factorized Gaussian posterior.
        noise = self.noise_distribution.sample(log_var.shape).squeeze()
        # Apply the reparameterization trick.
        z0 = mean + torch.sqrt(torch.exp(log_var)) * noise
        '''

        z0, auxillary_variable_vector, log_qv0, log_qz0 = self.aux_forward(parameters)

        # The above is identical to gaussian case, now apply flow transformations
        zT, vT, log_det_jacobian = self.forward(z0, auxillary_variable_vector)

        auxillary_variable_reverse, vT_mean, vT_var, log_rvT = self.aux_reverse(zT, vT)

        return zT, [z0, log_qv0, log_qz0, log_rvT, log_det_jacobian] # what to return?
