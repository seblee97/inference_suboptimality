from .base_estimator import BaseEstimator
from models.loss_modules import baseLoss

import torch
import torch.nn as nn

class IWAEEstimator(BaseEstimator):

    def __init__(self, num_samples: float, batch_size: int):
        """
        Constructs an IWAEEstimator with the given number of samples and batch
        size.  Specifying a batch size is necessary to avoid running out of GPU
        memory for large input batches.

        :param num_samples: number of samples
        :param batch_size: batch size
        """
        BaseEstimator.__init__(self)
        self._num_samples = num_samples
        self._batch_size = batch_size

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
            elbo = 0
            # Cache the input batch size; it is used several times below.
            input_batch_size = len(batch_input)
            # Iterate over the start index of each IWAE batch.
            for beg in range(0, input_batch_size, self._batch_size):
                # Truncate the end index if it exceeds the size of the input batch.
                end = min(input_batch_size, beg + self._batch_size)
                # Duplicate the input batch once for each sample in |self._num_samples|.
                inputs = batch_input[beg:end].repeat(self._num_samples, 1)
                # Compute the log-likelihood of each input.
                outputs = vae(inputs)
                _, _, log_p_x = loss_module.compute_loss(x=inputs, vae_output=outputs)
                # Align the rows of the log-likelihoods with the inputs in the batch.
                log_p_x = log_p_x.view(self._num_samples, -1).transpose(0, 1)
                # Find the maximum log-likelihood to avoid numeric overflow with the
                # LogSumExp trick.  For more details, see https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/.
                max_log_p_x, _ = torch.max(log_p_x, 1, keepdim=True)
                # Calculate the IWAE ELBO.
                elbo += torch.sum(torch.log(torch.mean(torch.exp(log_p_x - max_log_p_x), 1)) + torch.squeeze(max_log_p_x))
            # By definition, the loss is the negative average ELBO.
            loss = -elbo / input_batch_size

        if training:
            vae.train()

        return loss