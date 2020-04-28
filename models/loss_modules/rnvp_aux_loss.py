from .base_loss import _BaseLoss

from typing import Dict

import torch.nn.functional as F
import torch
import numpy as np
from typing import Dict, Tuple

class RNVPAuxLoss(_BaseLoss):

    def __init__(self):

        _BaseLoss.__init__(self)

    def compute_loss(self, x, vae_output, warm_up=1) -> Tuple[torch.Tensor, Dict, torch.Tensor]:

        vae_reconstruction = vae_output['x_hat']
        vae_latent = vae_output['z']
        mean, logvar, z0, log_det_jacobian, rv, rv_mean, rv_log_var, v, mean_v, logvar_v = vae_output['params']

        # Calculate the logs in the ELBO with ONE sample from the expectation.
        # Flow adds extra term - the sum of the logs of the determinants of the transformation Jacobians.
        # Aux adds reverse log prob term.
        #   ELBO = E[log(p(x,z) * r(v|x,z) / (q(z,v|x) * sum(log_det_jacobian)))]
        #        = E[log(p(x|z) * p(z) * r(v|x,z) / (q(z,v|x)) * sum(det_jacobian))]
        #        = E[log p(x|z) + log p(z) + log r(v|x,z) - log q(z,v|x) - sum(log_det_jacobian)]
        #          explicit flow expression
        #        = E[log p(x|z) + log p(z) + log r(v|x,z) - log q_0(z_0, v_0|x) + sum(log_det_jacobian)]
        log_p_xz = -F.binary_cross_entropy_with_logits(vae_reconstruction, x, reduction='none').view(x.shape[0], -1).sum(-1)
        log_p_z = -0.5 * vae_latent.pow(2).sum(1)
        log_q_v = -0.5 * (logvar_v.sum(1) + ((v - mean_v).pow(2) / torch.exp(logvar_v)).sum(1)) # where v is the auxiliary variable
        log_q_zx = -0.5 * (logvar.sum(1) + ((z0 - mean).pow(2) / torch.exp(logvar)).sum(1))

        # reverse model log
        log_r_vxz = -0.5 * (rv_log_var.sum(1) + ((rv - rv_mean).pow(2) / torch.exp(rv_log_var)).sum(1))

        log_p_x = log_p_xz + log_p_z - warm_up * (log_q_zx + log_q_v - log_det_jacobian - log_r_vxz)

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
