from abc import ABC, abstractmethod

from typing import Dict
import torch

class _BaseLoss(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def compute_loss(self, x: torch.Tensor, vae_output: Dict, warm_up: float):
        raise NotImplementedError("Base class method")