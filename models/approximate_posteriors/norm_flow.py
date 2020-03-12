from .base_approximate_posterior import approximatePosterior

import torch
import torch.nn as nn
import torch.distributions as tdist

from typing import Dict

from .flows import RNVP, RNVPAux

class NormFlowPosterior(approximatePosterior):

    def __init__(self, config: Dict):
        approximatePosterior.__init__(self, config)

        self.flow_type = config.get(["flow", "flow_type"])
        self.num_flows = config.get(["flow", "num_flows"])

        self.noise_distribution = tdist.Normal(torch.Tensor([0]), torch.Tensor([1]))

        # ammortisation networks for flow parameters
        if self.flow_type == 'rnvp':
            self.flow_module = RNVP(config=config)
        elif self.flow_type == 'rnvp_aux':
            self.flow_module = RNVPAux(config=config)        
        else:
            raise ValueError("flow_type {} not recognised".format(self.flow_type))

    def construct_posterior(self):
        pass

    def sample(self, parameters):

        # cutoff = int(0.5 * parameters.shape[1])

        # z1 = parameters[:, :cutoff]
        # z2 = parameters[:, cutoff:]

        # u = self.sigma_mapping(parameters)
        # w = self.mu_mapping(parameters)

        # sample noise (reparameterisation trick), unsqueeze to match dimensions
        # noise = self.noise_distribution.sample(z2.shape).squeeze()
        
        # z0 = z1 + torch.sqrt(z2) * noise

        # z_through_flow = []

        # log_det_jacobian_sum = 0

        # apply flow transformations
        self.flow_module(parameters)
        
        for k in range(self.num_flows):
            z_k, log_det_jacobian = flow_k(z[k], u[:, k, :, :], w[:, k, :, :], b[:, k, :, :])
            z_through_flow.append(z_k)
            log_det_jacobian_sum += log_det_jacobian

        latent_vector = z_through_flow[-1]
        
        return {'z': latent_vector, 'params': [z1, z2, z0, log_det_jacobian_sum]}






#from author
# from utils.math_ops import log_bernoulli, log_normal, log_mean_exp


class Flow(nn.Module):
    """A combination of R-NVP and auxiliary variables."""

    def __init__(self, model, n_flows=2):
        super(Flow, self).__init__()
        self.z_size = model.z_size
        self.n_flows = n_flows
        self._construct_weights()

    def forward(self, z, grad_fn=lambda x: 1, x_info=None):
        return self._sample(z, grad_fn, x_info)

    def _norm_flow(self, params, z, v, grad_fn, x_info):
        h = F.elu(params[0][0](torch.cat((z, x_info), dim=1)))
        mu = params[0][1](h)
        logit = params[0][2](h)
        sig = F.sigmoid(logit)

        # old CIFAR used the one below
        # v = v * sig + mu * grad_fn(z)

        # the more efficient one uses the one below
        v = v * sig - F.elu(mu) * grad_fn(z)
        logdet_v = torch.sum(logit - F.softplus(logit), 1)

        h = F.elu(params[1][0](torch.cat((v, x_info), dim=1)))
        mu = params[1][1](h)
        logit = params[1][2](h)
        sig = F.sigmoid(logit)

        z = z * sig + mu
        logdet_z = torch.sum(logit - F.softplus(logit), 1)
        logdet = logdet_v + logdet_z

        return z, v, logdet

    def _sample(self, z0, grad_fn, x_info):
        B = z0.size(0)
        z_size = self.z_size
        act_func = F.elu
        qv_weights, rv_weights, params = self.qv_weights, self.rv_weights, self.params

        out = torch.cat((z0, x_info), dim=1)
        for i in range(len(qv_weights)-1):
            out = act_func(qv_weights[i](out))
        out = qv_weights[-1](out)
        mean_v0, logvar_v0 = out[:, :z_size], out[:, z_size:]

        eps = Variable(torch.randn(B, z_size).type( type(out.data) ))
        v0 = eps.mul(logvar_v0.mul(0.5).exp_()) + mean_v0
        logqv0 = log_normal(v0, mean_v0, logvar_v0)

        zT, vT = z0, v0
        logdetsum = 0.
        for i in range(self.n_flows):
            zT, vT, logdet = self._norm_flow(params[i], zT, vT, grad_fn, x_info)
            logdetsum += logdet

        # reverse model, r(vT|x,zT)
        out = torch.cat((zT, x_info), dim=1)
        for i in range(len(rv_weights)-1):
            out = act_func(rv_weights[i](out))
        out = rv_weights[-1](out)
        mean_vT, logvar_vT = out[:, :z_size], out[:, z_size:]
        logrvT = log_normal(vT, mean_vT, logvar_vT)

        assert logqv0.size() == (B,)
        assert logdetsum.size() == (B,)
        assert logrvT.size() == (B,)

        logprob = logqv0 - logdetsum - logrvT

        return zT, logprob

    def _construct_weights(self):
        z_size = self.z_size
        n_flows = self.n_flows
        h_s = 200

        qv_arch = rv_arch = [z_size*2, h_s, h_s, z_size*2]
        qv_weights, rv_weights = [], []

        # q(v|x,z)
        id = 0
        for ins, outs in zip(qv_arch[:-1], qv_arch[1:]):
            cur_layer = nn.Linear(ins, outs)
            qv_weights.append(cur_layer)
            self.add_module('qz%d' % id, cur_layer)
            id += 1

        # r(v|x,z)
        id = 0
        for ins, outs in zip(rv_arch[:-1], rv_arch[1:]):
            cur_layer = nn.Linear(ins, outs)
            rv_weights.append(cur_layer)
            self.add_module('rv%d' % id, cur_layer)
            id += 1

        # nf
        params = []
        for i in range(n_flows):
            layer_grid = [
                [nn.Linear(z_size*2, h_s),
                 nn.Linear(h_s, z_size),
                 nn.Linear(h_s, z_size)],
                [nn.Linear(z_size*2, h_s),
                 nn.Linear(h_s, z_size),
                 nn.Linear(h_s, z_size)],
            ]

            params.append(layer_grid)

            id = 0
            for layer_list in layer_grid:
                for layer in layer_list:
                    self.add_module('flow%d_layer%d' % (i, id), layer)
                    id += 1

        self.qv_weights = qv_weights
        self.rv_weights = rv_weights
        self.params = params

        self.sanity_check_param = self.params[0][0][0]._parameters['weight']