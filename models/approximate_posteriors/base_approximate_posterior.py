from abc import ABC

class approximatePosterior(ABC):

    def __init__(self):
        pass

    def construct_posterior(self):
        raise NotImplementedError("Base class method")

    def sample(self):
        #approximate_posterior.sample should return the latent and a log-probability
        
        #return z, logp
        pass
