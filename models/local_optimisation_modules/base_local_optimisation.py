import torch

from abc import ABC, abstractmethod

from typing import Union, List, Dict

class _BaseLocalOptimisation(ABC):

    def __init__(self, config):
        self.optimiser_type = config.get(["local_optimisation", "optimiser", "type"])
        self.optimiser_params = config.get(["local_optimisation", "optimiser", "params"])
        self.learning_rate = config.get(["local_optimisation", "optimiser", "learning_rate"])

    def get_local_optimiser(self, parameters: Union[List, Dict]):
        if self.optimiser_type == "adam":
            beta_1 = self.optimiser_params[0]
            beta_2 = self.optimiser_params[1]
            epsilon = self.optimiser_params[2]
            return torch.optim.Adam(
                parameters, lr=self.learning_rate, betas=(beta_1, beta_2), eps=epsilon
                )
        else:
            raise ValueError("Optimiser {} not recognised". format(self.optimiser_type))

    @abstractmethod
    def get_additional_parameters(self):
        raise NotImplementedError("Base class method")

    @abstractmethod
    def sample_latent_vector(self):
        raise NotImplementedError("Base class method")