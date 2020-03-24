from .base_estimator import BaseEstimator
from models.loss_modules import baseLoss

import math
import torch
import torch.nn as nn


class AISEstimator(BaseEstimator):

    def __init__(self, num_samples: float, batch_size: int, latent_size: int, num_dists: int, leapfrog_steps: int):
        """
        Constructs a new AISEstimator with the given number of samples, batch size,
        latent vector size, distribution count, and leapfrog steps.

        :param num_samples: Number of samples
        :param batch_size: Size of the minibatches
        :param latent_size: Size of the latent vector in the VAE
        :param num_dists: Number of intermediate distributions
        :param leapfrog_steps: Number of leapfrog steps in the HMC simulation
        """
        BaseEstimator.__init__(self)
        self._num_samples = num_samples
        self._batch_size = batch_size
        self._latent_size = latent_size
        self._num_dists = num_dists
        self._leapfrog_steps = leapfrog_steps

    def estimate_log_likelihood_loss(self, batch_input: torch.Tensor, vae: nn.Module, loss_module: baseLoss) -> torch.Tensor:
        """
        Estimates the average log-likelihood loss of the given input batch using
        the provided VAE and loss module.

        :param batch_input: Input batch for the log-likelihood
        :param vae: VAE model (only the decoder is used)
        :param loss_module: Loss module
        :return: The log-likelihood of the given batch input predicted by AIS.
        """
        def log_likelihood(x: torch.Tensor, z: torch.Tensor, beta: float) -> torch.Tensor:
            """
            Calculates the log-likelihood of the distribution p(x, z, β) where

                p(x, z, β) = p(x, z)^β * p(z)^(1 - β)

            and

                log p(x, z, β) = log p(x, z)^β + log p(z)^(1 - β)
                               = β * log p(x, z) + (1 - β) * log p(z)
                               = β * log p(x|z) + β * log p(z) + log p(z) - β * log p(z)
                               = β * log p(x|z) + log p(z)

            :param x: Input minibatch for the log-likelihood
            :param z: Latent vector (i.e., the position in HMC)
            :param beta: Exponent of the annealed distribution
            :return: The log-likelihood of the intermediate distribution corresponding to
                     β with respect to the given input and latent vector
            """
            # The probability density of p(z) is the standard normal.
            log_p_z = -0.5 * (z.pow(2).sum(1) + z.size(1) * torch.log(2 * math.pi))
            # The probability density of p(x|z) is given by the loss module.
            _, _, log_p_xz = loss_module.compute_loss(x=x, vae_output=vae.decoder(z))
            # As described in the function description.
            return beta * log_p_xz + log_p_z

        def U(x: torch.Tensor, z: torch.Tensor, beta: float) -> torch.Tensor:
            """
            Calculates the potential energy of the latent vector for HMC, which,
            for convenience, is defined to be -log p(x, z, β).

            :param x: Input minibatch for the log-likelihood
            :param z: Latent vector (i.e., the position in HMC)
            :param beta: Exponent of the annealed distribution
            :return: The potential energy of the latent vector with respect to
                     the intermediate distribution exponent and input
            """
            return -log_likelihood(x, z, beta)

        def K(v: torch.Tensor) -> torch.Tensor:
            """
            Calculates the kinetic energy of the latent vector for HMC.  The
            one defined by Newtonian mechanics (assume the mass is 1) tends to
            be a reasonable choice.

            :param v: The velocity in HMC
            :return: The kinetic energy associated with the given velocity
            """
            # Note: the Inference Suboptimality authors use a "normalized" version
            #       which is the negative log-likelihood of the velocity.
            return 0.5 * (v * v).sum(1)

        # Remember the state of the VAE and restore it after the routine.
        training = vae.training
        vae.eval()

        # Construct a tuple with a 1 for each dimension of the input.
        batch_repeat_shape = tuple(1 for _ in batch_input.shape[1:])

        # Assume a linear AIS schedule for the values of β.
        schedule = torch.linspace(start=0.0, end=1.0, steps=self._num_dists)

        elbo = 0
        # Cache the input batch size; it is used several times below.
        input_batch_size = len(batch_input)
        for beg in range(0, input_batch_size, self._batch_size):
            # Truncate the end index if it exceeds the size of the input batch.
            end = min(input_batch_size, beg + self._batch_size)
            # Duplicate the input subbatch once for each sample in |self._num_samples|.
            minibatch = batch_input[beg:end].repeat(self._num_samples, *batch_repeat_shape)
            minibatch_size = minibatch.size(0)

            # These step sizes are used in the leapfrog algorithm to simulate
            # the path of the particle in HMC.
            steps = 0.01 * torch.ones(minibatch_size)
            # The history of accepted or rejected decisions drive step size
            # adjustments in the leapfrog algorithm.
            history = torch.zeros(minibatch_size)
            # The logarithm of the importance weights of the desired normalization factor.
            log_weights = torch.zeros(minibatch_size)
            # The prior distribution of the latent vector is always the standard normal.
            z = torch.randn(minibatch_size, self._latent_size)

            for rounds, betas in enumerate(iterable=zip(schedule[:-1], schedule[1:]), start=1):
                # Record the contribution of this fraction of likelihoods.
                log_p_prev = log_likelihood(x=minibatch, z=z, beta=betas[0])
                log_p_next = log_likelihood(x=minibatch, z=z, beta=betas[1])
                log_weights += log_p_next - log_p_prev

                # For simplicity, the prior of the velocities is assumed to be
                # the standard normal distribution.
                prev_v = torch.randn(z.size())
                prev_z = z

                next_v = prev_v.clone().detach()
                next_z = prev_z.clone().detach().requires_grad_(True)

                # Perform the leapfrog algorithm.  This implementation of HMC
                # follows from https://arxiv.org/pdf/1206.1901.pdf.
                for _ in range(self._leapfrog_steps):
                    # Update the "position".
                    next_z += next_v * steps
                    # Update the "velocity".
                    next_z.grad.data.zero_()
                    dz = torch.autograd.grad(U(minibatch, next_z, betas[0]), next_z)
                    # Prevent explosive gradients the easy way.
                    # TODO: Add constant or parameter for magic numbers.
                    dz = dz.clamp(-10000, 10000)
                    # Assuming that the energy of the system is constant, the
                    # gradient with respect to the velocity is the negative of
                    # the gradient with respect to the position:
                    #
                    #    ∇ E(z, v) = 0 = ∇ U(z) + ∇ K(v)
                    #           ∇ K(v) = - ∇ U(z)
                    #                v = - ∇ U(z)
                    dv = -dz
                    next_v += dv * steps

                # The exponential of the difference in the energy states is
                # the probability that the sample is accepted in HMC.
                prev_E = U(prev_z) + K(prev_v)
                next_E = U(next_z) + K(next_v)
                diff_E = next_E - prev_E
                accepted = torch.exp(diff_E) < torch.rand(diff_E.size()).view(-1, 1)
                # Update the accepted latent vectors.
                z = next_z * accepted + prev_z * (1 - accepted)

                # Adjust the step sizes according to the acceptance rate.
                history += accepted
                # TODO: Add constant or parameter for magic numbers.
                increase = history / rounds > 0.65
                steps *= 1.02 if increase else 1.98
                # TODO: Add constant or parameter for magic numbers.
                steps = steps.clamp(1E-4, 0.5)

            # Align the rows of the weights with the inputs in the minibatch.
            log_weights = log_weights.view(self.num_samples, -1).transpose(0, 1)
            # Perform the same aggregation as IWAE.
            max_log_weights, _ = torch.max(log_weights, 1, keepdim=True)
            elbo += torch.sum(torch.log(torch.mean(torch.exp(log_weights - max_log_weights), 1)) + torch.squeeze(max_log_weights))

        # By definition, the loss is the negative average ELBO.
        loss = -elbo / input_batch_size

        # ----------------------------------------------------------------------

        if training:
            vae.train()

        return loss