from .base_loss import baseLoss

from typing import Dict

import torch.nn.functional as F
import torch
import numpy as np
from typing import Dict

class RNVPLoss(baseLoss):

    def __init__(self):

        baseLoss.__init__(self)

    """
    loss module :https://github.com/chrischute/real-nvp/blob/df51ad570baf681e77df4d2265c0f1eb1b5b646c/models/real_nvp/real_nvp_loss.py
    """

    def compute_loss(self, x, vae_output):

        vae_reconstruction = vae_output['x_hat']
        vae_latent = vae_output['z']
        mean, log_var, z0, log_det_jacobian = vae_output['params']

        # Calculate the logs in the ELBO with ONE sample from the expectation.
        #   ELBO = E[log(p(x,z) / q(z|x))]
        #        = E[log(p(x|z) * p(z) / q(z|x))]
        #        = E[log p(x|z) + log p(z) - log q(z|x)]
        log_p_xz = -F.binary_cross_entropy(vae_reconstruction, x, reduction='none').sum(-1)
        log_p_z = -0.5 * vae_latent.pow(2).sum(1)
        log_q_zx = -0.5 * (log_var.sum(1) + ((vae_latent - mean).pow(2) / torch.exp(log_var)).sum(1)) - log_det_jacobian
        # TODO: Add a warm-up constant to the last two terms.
        log_p_x = log_p_xz + log_p_z - log_q_zx

        # The ELBO is defined to be the mean of the logs in the batch.
        elbo = torch.mean(log_p_x)

        # Maximizing the ELBO is equivalent to minimizing the negative ELBO.
        loss = -elbo

        return loss, [], []