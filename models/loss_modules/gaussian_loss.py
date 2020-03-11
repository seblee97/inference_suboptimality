from .base_loss import baseLoss

from typing import Dict

import torch.nn.functional as F
import torch

class gaussianLoss(baseLoss):

    def __init__(self):
        baseLoss.__init__(self)

    def compute_loss(self, x, vae_output):
        vae_reconstruction = vae_output['x_hat']
        mean, log_var = vae_output['params']

        # Assuming a normal Gaussian prior and a fully-factorized Gaussian approximation,
        # the loss function is detailed in https://arxiv.org/pdf/1907.08956.pdf.
        reconstruction_loss = F.binary_cross_entropy(vae_reconstruction, x, reduction='mean')
        kl_loss = torch.sum(torch.exp(log_var) + mean**2 - 1 - log_var) / 2

        return reconstruction_loss + kl_loss

        # loss, rec, kl, bpd = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args, beta=beta)

        # loss, rec, kl = binary_loss_function(x_mean, x, z_mu, z_var, z_0, z_k, ldj, beta=beta)