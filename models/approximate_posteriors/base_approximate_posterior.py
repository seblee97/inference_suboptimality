from abc import ABC, abstractmethod

from typing import Dict

class approximatePosterior(ABC):

    def __init__(self, config: Dict):
        pass

    @abstractmethod
    def sample(self):
        #approximate_posterior.sample should return the latent and a log-probability

        #return z, logp
        pass
