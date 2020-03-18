from .base_loss import baseLoss

from typing import Dict

import torch.nn.functional as F
import torch
import numpy as np
from typing import Dict

class RNVPLoss(baseLoss):

    def __init__(self, input_dim):

        baseLoss.__init__(self)

        self.input_dim = input_dim

    """
    loss module :https://github.com/chrischute/real-nvp/blob/df51ad570baf681e77df4d2265c0f1eb1b5b646c/models/real_nvp/real_nvp_loss.py
    """

    def compute_loss(self, x, vae_output):
        vae_reconstruction = vae_output['x_hat']
        z1, z2, z0, log_det_jacobian = vae_output['params']

        prior = -0.5 * (vae_reconstruction ** 2 + np.log(2 * np.pi))
        prior_l = prior.view(vae_reconstruction.size(0), -1).sum(-1) / - np.log(self.input_dim) * np.prod(z.size()[1:])
        ll = prior_l + log_det_jacobian
        nll = -ll.mean()


        # Assuming a normal Gaussian prior and a fully-factorized Gaussian approximation,
        # the loss function is detailed in https://arxiv.org/pdf/1907.08956.pdf.
        #reconstruction_loss = F.binary_cross_entropy(vae_reconstruction, x, reduction='mean')
        #kl_loss = torch.sum(torch.exp(log_var) + mean**2 - 1 - log_var) / 2

        return nll

        # loss, rec, kl, bpd = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args, beta=beta)

        # loss, rec, kl = binary_loss_function(x_mean, x, z_mu, z_var, z_0, z_k, ldj, beta=beta)