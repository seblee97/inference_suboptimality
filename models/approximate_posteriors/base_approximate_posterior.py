from abc import ABC

class approximatePosterior(ABC):

    def __init__(self, config: Dict):
        pass

    def construct_posterior(self):
        raise NotImplementedError("Base class method")

    def sample(self):
        pass