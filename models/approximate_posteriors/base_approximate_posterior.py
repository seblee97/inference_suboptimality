from abc import ABC, abstractmethod

from typing import Dict, List

import torch

class _ApproximatePosterior(ABC):

    def __init__(self, config: Dict) -> None:
        pass

    @abstractmethod
    def sample(self) -> (torch.Tensor, List):
        """
        Should returns the latent vector as well as other values 
        necessary to compute the elbo e.g. the mean and variance to 
        compute log-probability. 
        """
        raise NotImplementedError("Base class method.")
