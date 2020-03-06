from .base_loss import baseLoss

from typing import Dict

import torch.nn.functional as F
import torch

class gaussianLoss(baseLoss):

    def __init__(self):
        baseLoss.__init__(self)

    def compute_loss(self, x, vae_output):
        
        vae_reconstruction = vae_output['x_hat']

        params = vae_output['params']
        mu = params[0]
        var = params[1]

        reconstruction_loss = F.binary_cross_entropy(vae_reconstruction, x, size_average=False)

        kl_loss = 0.5 * torch.sum(torch.exp(var) + mu**2 - 1.0 - var)

        return reconstruction_loss + kl_loss

        # loss, rec, kl, bpd = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args, beta=beta)

        # loss, rec, kl = binary_loss_function(x_mean, x, z_mu, z_var, z_0, z_k, ldj, beta=beta)

        
        
        # elbo = reconstruction_loss + logpz - plogqz
        # loss = -elbo

        # raise NotImplementedError
