from .base_loss import baseLoss

from typing import Dict

import torch.nn.functional as F
import torch
import numpy as np
from typing import Dict

class RNVPAuxLoss(baseLoss):

    def __init__(self):

        baseLoss.__init__(self)

    """
    loss module :https://github.com/chrischute/real-nvp/blob/df51ad570baf681e77df4d2265c0f1eb1b5b646c/models/real_nvp/real_nvp_loss.py
    """

    def compute_loss(self, x, vae_output, warm_up=1):

        vae_reconstruction = vae_output['x_hat']
        vae_latent = vae_output['z']
        mean, log_var, z0, log_det_jacobian, rv, rv_mean, rv_log_var = vae_output['params']

        # Calculate the logs in the ELBO with ONE sample from the expectation.
        # Flow adds extra term - the sum of the logs of the determinants of the transformation Jacobians.
        #   ELBO = E[log(p(x,z) / q(z|x))]
        #        = E[log(p(x|z) * p(z) / q(z|x))]
        #        = E[log p(x|z) + log p(z) - log q(z|x)]
        #          explicit flow expression
        #        = E[log p(x|z) + log p(z) - log q_0(z_0|x) + sum(log_det_jacobian)]
        log_p_xz = -F.binary_cross_entropy_with_logits(vae_reconstruction, x, reduction='none').view(x.shape[0], -1).sum(-1)
        log_p_z = -0.5 * vae_latent.pow(2).sum(1)
        log_q_zx = -0.5 * (log_var.sum(1) + ((z0 - mean).pow(2) / torch.exp(log_var)).sum(1)) - log_det_jacobian

        # reverse model log
        log_r_vxz = -0.5 * (rv_log_var.sum(1) + ((rv - rv_mean).pow(2) / torch.exp(rv_log_var)).sum(1))

        # TODO: Add a warm-up constant to the last two terms.
        log_p_x = log_p_xz + log_p_z - warm_up * (log_q_zx - log_r_vxz)

        # The ELBO is defined to be the mean of the logs in the batch.
        elbo = torch.mean(log_p_x)

        # Maximizing the ELBO is equivalent to minimizing the negative ELBO.
        loss = -elbo

        loss_metrics = {}
        loss_metrics["elbo"] = float(elbo)
        loss_metrics["log p(x|z)"] = float(torch.mean(log_p_xz))
        loss_metrics["log p(z)"] = float(torch.mean(log_p_z))
        loss_metrics["log q(z|x)"] = float(torch.mean(log_q_zx))
        loss_metrics["log_det_jacobian"] = float(torch.mean(log_det_jacobian))
        loss_metrics["reverse: log r(v|x,z)"] = float(torch.mean(log_r_vxz))

        return loss, loss_metrics, log_p_x
