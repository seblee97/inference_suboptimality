from .base_estimator import BaseEstimator
from models.loss_modules import baseLoss

import torch
import torch.nn as nn

class IWAEEstimator(BaseEstimator):

    def __init__(self, num_samples: float):
        """
        Constructs an IWAEEstimator with the given number of samples.

        :param num_samples: number of samples
        """
        BaseEstimator.__init__(self)
        self._num_samples = num_samples

    def estimate_log_likelihood_loss(self, batch_input: torch.Tensor, vae: nn.Module, loss_module: baseLoss) -> torch.Tensor:
        """
        Estimates the log-likelihood loss of the given input batch using the
        provided VAE and loss module.

        :param x: input batch
        :param vae: VAE model
        :param loss_module: loss module
        :return loss: negative IWAE ELBO
        """
        # Remember the state of the VAE and restore it after the routine.
        training = vae.training
        vae.eval()

        # Global warming is no joke, and neither are gradient calculations.
        with torch.no_grad():
            # Duplicate the input batch once for each sample in |self._samples|.
            inputs = batch_input.repeat(self._samples, 1)
            # Compute the log-likelihood of each input.
            outputs = vae(inputs)
            _, _, log_p_x = loss_module.compute_loss(x=inputs, vae_output=outputs)
            # Align the rows of the log-likelihoods with the inputs in the batch.
            log_p_x = log_p_x.view(-1, self._samples)
            # Find the maximum log-likelihood to avoid numeric overflow with the
            # LogSumExp trick.  For more details, see https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/.
            max_log_p_x, _ = torch.max(log_p_x, 1, keepdim=True)
            # Calculate the IWAE ELBO.
            elbo = torch.mean(torch.log(torch.mean(torch.exp(log_p_x - max_log_p_x), 1)) + torch.squeeze(max_log_p_x))
            # By definition, the loss is the negative ELBO.
            loss = -elbo

        if training:
            vae.train()

        return loss