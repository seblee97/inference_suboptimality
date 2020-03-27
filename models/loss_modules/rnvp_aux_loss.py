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

    def compute_loss(self, x, vae_output, warm_up):

        vae_reconstruction = vae_output['x_hat']
        vae_latent = vae_output['z']
        z0, log_qv0, log_qz0, log_rvT, log_det_jacobian = vae_output['params']

        # Calculate the logs in the ELBO with ONE sample from the expectation.
        # Flow adds extra term - the sum of the logs of the determinants of the transformation Jacobians.
        #   ELBO = E[log(p(x,z) / q(z|x))]
        #        = E[log(p(x|z) * p(z) / q(z|x))]
        #        = E[log p(x|z) + log p(z) - log q(z|x)]
        #          explicit flow expression
        #        = E[log p(x|z) + log p(z) - log q_0(z_0|x) + sum(log_det_jacobian)]

         # Calculate the logs in the ELBO with ONE sample from the expectation.
        # Flow adds extra term - the sum of the logs of the determinants of the transformation Jacobians.
        #   ELBO = E[log(p(x,z) * r(v|x, z) / q(z, v|x))]
        #        = E[log(p(x|z) * p(z) * r(v|x, z) / q(z, v|x))]
        #        = E[log p(x|z) + log p(z) + log r(v|x, z) - log q(z, v|x)]
        #          explicit flow expression
        #        = E[log p(x|z) + log p(z) + log_rvT - log q_v0(z_0, v|x) + sum(log_det_jacobian)]


        log_p_xz = -F.binary_cross_entropy(vae_reconstruction, x, reduction='none').sum(-1)
        log_p_z = -0.5 * vae_latent.pow(2).sum(1)
        log_q_z_vx = log_qv0 - log_det_jacobian
        log_r_vx_z = log_rvT
        # TODO: Add a warm-up constant to the last two terms.
        log_p_x = log_qv0 - log_det_jacobian - log_rvT
        #og_p_x = log_p_xz * log_r_vx_z - warm_up * log_q_zx


        # The ELBO is defined to be the mean of the logs in the batch.
        elbo = torch.mean(log_p_x)

        # Maximizing the ELBO is equivalent to minimizing the negative ELBO.
        loss = -elbo

        loss_metrics = {}
        loss_metrics["elbo"] = float(elbo)
        loss_metrics["log p(x|z)"] = float(torch.mean(log_p_xz))
        loss_metrics["log p(z)"] = float(torch.mean(log_p_z))
        loss_metrics["log q(z|x)"] = float(torch.mean(log_q_z_vx))
        loss_metrics["log r(v|x, z)"] = float(torch.mean(log_r_vx_z))
        loss_metrics["log_det_jacobian"] = float(torch.mean(log_det_jacobian))

        return loss, loss_metrics, log_p_x