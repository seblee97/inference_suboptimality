from .base_estimator import BaseEstimator
from .ais_estimator import AISEstimator
from .iwae_estimator import IWAEEstimator
from models.loss_modules import baseLoss

import torch


class MaxEstimator(BaseEstimator):

    def __init__(self, ais_estimator: AISEstimator, iwae_estimator: IWAEEstimator):
        """
        Constructs a new MaxEstimator with the given AIS and IWAE estimators.

        :param ais_estimator: AIS estimator
        :param iwae_estimator: IWAE estimator
        """
        BaseEstimator.__init__(self)
        self._ais_estimator = ais_estimator
        self._iwae_estimator = iwae_estimator

    def estimate_log_likelihood_loss(self, batch_input: torch.Tensor, vae: torch.nn.Module, loss_module: baseLoss) -> torch.Tensor:
        """
        Estimates the log-likelihood loss of the given input batch using the
        provided VAE and loss module.

        :param x: input batch
        :param vae: VAE model
        :param loss_module: loss module
        :return: negative maximum ELBO of the AIS and IWAE estimators
        """
        ais_elbo = -self._ais_estimator.estimate_log_likelihood_loss(batch_input, vae, loss_module)
        iwae_elbo = -self._iwae_estimator.estimate_log_likelihood_loss(batch_input, vae, loss_module)
        return -max(ais_elbo, iwae_elbo)
