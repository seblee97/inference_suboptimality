from .base_estimator import BaseEstimator
from models.loss_modules import baseLoss

import torch


class AISEstimator(BaseEstimator):

    def __init__(self, num_chains: float, batch_size: int, latent_size: int, num_dists: int, num_leapfrog_steps: int):
        """
        Constructs a new AISEstimator with the given number of samples, batch size,
        latent vector size, distribution count, and leapfrog steps.

        :param num_chains: Number of simulation chains per input
        :param batch_size: Size of the minibatches
        :param latent_size: Size of the latent vector in the VAE
        :param num_dists: Number of intermediate distributions
        :param num_leapfrog_steps: Number of leapfrog steps in the HMC simulation
        """
        BaseEstimator.__init__(self)
        self._num_chains = num_chains
        self._batch_size = batch_size
        self._latent_size = latent_size
        self._num_dists = num_dists
        self._num_leapfrog_steps = num_leapfrog_steps

    def estimate_log_likelihood_loss(self, batch_input: torch.Tensor, vae: torch.nn.Module, _: baseLoss) -> torch.Tensor:
        """
        Estimates the average log-likelihood loss of the given input batch using
        the provided VAE and loss module.

        :param batch_input: Input batch for the log-likelihood
        :param vae: VAE model (only the decoder is used)
        :param loss_module: Not used.
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
                     β with respect to the given input and latent vector.
            """
            # The probability density of p(x|z) is the joint Bernoulli distribution.
            output = vae.decoder(z)
            log_p_xz = -torch.nn.functional.binary_cross_entropy_with_logits(output, x, reduction='none').view(x.shape[0], -1).sum(1)
            # The probability density of p(z) is the standard Gaussian distribution.
            # Note that the constant term is ignored since it will get cancelled anyway.
            log_p_z = -0.5 * z.pow(2).sum(1)
            # Compute log p(x, z, β) as derived in the function description.
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
            return 0.5 * (v * v).sum(1)

        def update_velocity(x: torch.Tensor, z: torch.Tensor, beta: float, v: torch.Tensor, step_sizes: torch.Tensor) -> torch.Tensor:
            """
            Calculates the new velocity of the latent vector for HMC.  This
            implementation assumes Hamiltonian dynamics, where ∂v/∂t = - ∂U/∂z.
            For more information, see https://arxiv.org/pdf/1206.1901.pdf.

            :param x: Input minibatch for the potential energy function
            :param z: Latent vector (i.e., the position in HMC)
            :param beta: Exponent of the annealed distribution
            :param v: The velocity in HMC
            :param step_sizes: Scaling factor applied to the velocity gradient
            :return: The new velocity.
            """
            # Calculate the current potential energy of the system; this dictates
            # the gradient of the velocity as per the function description.
            u = U(x, z, beta)
            # The automatic differentiation routine takes ∂U/∂U as an input.
            duu = torch.ones(u.size(0))
            # Automagically compute ∂U/∂z.
            duz = torch.autograd.grad(u, z, duu)[0]
            # The clamping constants below were adopted from the authors' code.
            duz = torch.clamp(duz, -1E4, 1E4)
            # Adjust the velocity by its scaled gradient.
            return v - duz * step_sizes

        # Remember the state of the VAE and restore it after the routine.
        training = vae.training
        vae.eval()

        # Construct a tuple with a 1 for each dimension of the input.
        batch_repeat_shape = tuple(1 for _ in batch_input.shape[1:])

        # Assume a linear AIS schedule for the values of β.
        schedule = torch.linspace(start=0.0, end=1.0, steps=self._num_dists)

        # Accumulate the ELBOs of each minibatch.
        elbo = 0

        # Cache the input batch size; it is used several times below.
        input_batch_size = len(batch_input)
        for beg in range(0, input_batch_size, self._batch_size):
            # Truncate the end index if it exceeds the size of the input batch.
            end = min(input_batch_size, beg + self._batch_size)
            # Duplicate the input subbatch once for each sample in |self._num_chains|.
            minibatch = batch_input[beg:end].repeat(self._num_chains, *batch_repeat_shape)
            minibatch_size = minibatch.size(0)
            minibatch_size_broadcast = (minibatch.size(0), 1)

            # Below, |step_sizes| and |history| control the simulation of the
            # latent vector particle in the leapfrog algorithm for HMC:
            #    1. |step_sizes| controls the time increments in the simulation.
            #    2. |history| is used to adjust the time increments.
            # The initial step size was adopted from the authors' code.
            step_sizes = 0.01 * torch.ones(minibatch_size_broadcast)
            history = torch.zeros(minibatch_size_broadcast)
            # The logarithm of the importance weights gives the desired normalization factor.
            log_weights = torch.zeros(minibatch_size)
            # The prior distribution of the latent vector is the standard Gaussian distribution.
            z = torch.randn((minibatch_size, self._latent_size))

            for rounds, (b0, b1) in enumerate(iterable=zip(schedule[:-1], schedule[1:]), start=1):
                # Record the contribution of this fraction of likelihoods.
                log_p_prev = log_likelihood(x=minibatch, z=z, beta=b0)
                log_p_next = log_likelihood(x=minibatch, z=z, beta=b1)
                log_weights += log_p_next - log_p_prev

                # For simplicity, the prior distribution of the velocity is also the standard Gaussian distribution.
                prev_v = torch.randn(z.size())
                prev_z = z

                # Automatic differentiation is used to compute the gradient of U(z).
                with torch.enable_grad():
                    # Cloning and detaching the tensors is the equivalent of a deep copy.
                    next_v = prev_v.clone().detach()
                    next_z = prev_z.clone().detach().requires_grad_(True)

                    # The details for this implementation of the leapfrog algorithm
                    # can be found in https://arxiv.org/pdf/1206.1901.pdf.
                    next_v = update_velocity(minibatch, next_z, b1, next_v, step_sizes / 2)
                    for i in range(self._num_leapfrog_steps):
                        next_z = next_z + next_v * step_sizes
                        if i < self._num_leapfrog_steps - 1:
                            next_v = update_velocity(minibatch, next_z, b1, next_v, step_sizes)
                    next_v = update_velocity(minibatch, next_z, b1, next_v, step_sizes / 2)

                # The probability of a latent vector is the inverse exponential of its energy.
                prev_E = U(minibatch, prev_z, b1) + K(prev_v)
                next_E = U(minibatch, next_z, b1) + K(next_v)
                diff_E = (prev_E - next_E).reshape(minibatch_size_broadcast)

                # Accept or reject the proposals using the Metropolis criterion.
                accepted = torch.rand(diff_E.size()) < torch.exp(diff_E)
                rejected = ~accepted
                z = next_z * accepted + prev_z * rejected

                # Adjust the step sizes according to the acceptance rate.
                history += accepted
                # The acceptance rate threshold originates from https://arxiv.org/pdf/1206.1901.pdf.
                increase = history / rounds > 0.65
                decrease = ~increase
                # The step size constants below were adopted from the authors' code.
                step_sizes *= 1.02 * increase + 0.98 * decrease
                step_sizes = step_sizes.clamp(1E-4, 5E-1)

            # Align the rows of the weights with the inputs in the minibatch.
            log_weights = log_weights.view(self._num_chains, -1).transpose(0, 1)
            # Perform the same aggregation as IWAE.
            max_log_weights, _ = torch.max(log_weights, 1, keepdim=True)
            elbo += torch.sum(torch.log(torch.mean(torch.exp(log_weights - max_log_weights), 1)) + torch.squeeze(max_log_weights))

        # By definition, the loss is the negative average ELBO.
        loss = -elbo / input_batch_size

        if training:
            vae.train()

        return loss
