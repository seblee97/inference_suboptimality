from .base_loss import baseLoss

from typing import Dict

import torch.nn.functional as F
import torch

class gaussianLoss(baseLoss):

    def __init__(self):
        baseLoss.__init__(self)

    def compute_loss2(self, x, vae_output):
        vae_reconstruction = vae_output['x_hat']
        mean, log_var = vae_output['params']

        # Assuming a normal Gaussian prior and a fully-factorized Gaussian approximation,
        # the loss function is detailed in https://arxiv.org/pdf/1907.08956.pdf.
        reconstruction_loss = F.binary_cross_entropy(vae_reconstruction, x, reduction='mean')
        kl_loss = torch.sum(torch.exp(log_var) + mean**2 - 1 - log_var) / 2
        return reconstruction_loss + kl_loss

        # loss, rec, kl, bpd = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args, beta=beta)

        # loss, rec, kl = binary_loss_function(x_mean, x, z_mu, z_var, z_0, z_k, ldj, beta=beta)

    def compute_loss(self, x, vae_output):
        """
        This one mimicks what the paper does.
        """
        
        vae_reconstruction = vae_output['x_hat']
        vae_latent = vae_output['z']
        mean, log_var = vae_output['params']
        #print("size vae_latent", vae_latent.size() )
        #print("size mean", mean.size() )
        #print("size x", x.size() )
        #print("size vae_reconstruction", vae_reconstruction.size() )
        #log(p(z)): N(0, I) thus logvariance is 0 and mean 0.
        logpz = -0.5 * (((vae_latent).pow(2)).sum(1))
        #log(q(z)) : N(M, S)
        logqz = -0.5 * (log_var.sum(1) + ((vae_latent - mean).pow(2) / torch.exp(log_var)).sum(1))
        #print("size logpz", logpz.size() )
        #print("size logqz", logqz.size() )
        reconstruction_loss = F.binary_cross_entropy(vae_reconstruction, x, reduction='none').sum(-1) #this is the equivalent of log_bernouilli which is summed, not meaned.
        
        #print("size reconstruction_loss", reconstruction_loss.size() )
        #print("Automatic: ", reconstruction_loss_2)
        """
        reconstruction_loss = -F.relu(vae_reconstruction) + torch.mul(x, vae_reconstruction) - torch.log(1. + torch.exp( -vae_reconstruction.abs() ))
        while len(reconstruction_loss.size()) > 1:
            reconstruction_loss = reconstruction_loss.sum(-1)
            """
        #print("Manual: ", reconstruction_loss.sum(-1))
        ELBO = -reconstruction_loss + logpz - logqz  # the last term should have a warmup constant
        #print("logpz: ", logpz)
        #print("step 1", ELBO)
        ELBO = ELBO.view(1, -1).transpose(0, 1)
        #print("step 2", ELBO)
        max_, _ = torch.max(ELBO, 1, keepdim=True)
        #print("max_", max_)
        #ELBO = torch.log(torch.mean(torch.exp(ELBO - max_), 1)) + torch.squeeze(max_)
        #print("step 3", ELBO)
        ELBO = torch.mean(ELBO)
        #print("step 4", ELBO)
        ELBO = -ELBO
        return ELBO

        # loss, rec, kl, bpd = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args, beta=beta)

        # loss, rec, kl = binary_loss_function(x_mean, x, z_mu, z_var, z_0, z_k, ldj, beta=beta)

