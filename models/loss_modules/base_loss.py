from abc import ABC, abstractmethod

from typing import Dict

class baseLoss(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def compute_loss(self, vae_output: Dict):
        raise NotImplementedError("Base class method")