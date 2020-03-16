from .base_loss import baseLoss

from typing import Dict

import torch.nn.functional as F
import torch

class gaussianLoss(baseLoss):

    def __init__(self):
        baseLoss.__init__(self)

    # def compute_loss2(self, x, vae_output):
    #     vae_reconstruction = vae_output['x_hat']
    #     mean, log_var = vae_output['params']

    #     # Assuming a normal Gaussian prior and a fully-factorized Gaussian approximation,
    #     # the loss function is detailed in https://arxiv.org/pdf/1907.08956.pdf.
    #     reconstruction_loss = F.binary_cross_entropy(vae_reconstruction, x, reduction='mean')
    #     kl_loss = torch.sum(torch.exp(log_var) + mean**2 - 1 - log_var) / 2
    #     return reconstruction_loss + kl_loss

        # loss, rec, kl, bpd = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args, beta=beta)

        # loss, rec, kl = binary_loss_function(x_mean, x, z_mu, z_var, z_0, z_k, ldj, beta=beta)

    def compute_loss(self, x, vae_output) -> (torch.Tensor, Dict):
        """
        This one mimicks what the paper does.
        """
        loss_metrics = {}

        vae_reconstruction = vae_output['x_hat']
        vae_latent = vae_output['z']
        mean, log_var = vae_output['params']

        logpz = -0.5 * (((vae_latent).pow(2)).sum(1))
        logqz = -0.5 * (log_var.sum(1) + ((vae_latent - mean).pow(2) / torch.exp(log_var)).sum(1))

        reconstruction_loss = F.binary_cross_entropy(vae_reconstruction, x, reduction='none').sum(-1) #this is the equivalent of log_bernouilli which is summed, not meaned.

        ELBO = torch.mean(-reconstruction_loss + logpz - logqz)  # the last term should have a warmup constant
        loss = -ELBO    

        loss_metrics["reconstruction_loss"] = float(torch.mean(reconstruction_loss))
        loss_metrics["elbo"] = float(ELBO)
        loss_metrics["log(p|z)"] = float(torch.mean(logpz))
        loss_metrics["log(q|z)"] = float(torch.mean(logqz))

        return loss, loss_metrics

