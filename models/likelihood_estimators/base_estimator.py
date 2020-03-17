from abc import ABC, abstractmethod

class BaseEstimator(ABC):

    @abstractmethod
    def estimate_log_likelihood_loss(self, x, vae, loss_module):
        return
