from abc import ABC

from typing import Dict

class approximatePosterior(ABC):

    def __init__(self, config: Dict):
        pass

    def construct_posterior(self):
        raise NotImplementedError("Base class method")

    def sample(self):
        #approximate_posterior.sample should return the latent and a log-probability

        #return z, logp
        pass
