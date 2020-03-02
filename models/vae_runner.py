from .vae import VAE

from .networks.convolutional import convNetwork
from .networks.feedforward import feedForwardNetwork
from .networks.deconvolutional import deconvNetwork
from .networks.feedbackward import feedBackwardNetwork

from .encoder import Encoder
from .decoder import Decoder

from .approximate_posteriors.gaussian import gaussianPosterior

from utils import mnist_dataloader

from typing import Dict
import os

class VAERunner():

    def __init__(self, config: Dict) -> None:
        """
        Class for constructing variational autoencoder (from encoder, approximate posterior family, and decoder)
        and orchestrating training.

        :param config: dictionary containing parameters to specify training etc.
        """
        # extract relevant parameters from config
        self._extract_parameters(config)

        # initialise encoder, decoder
        # This should also give activation functions !!
        encoder = self._setup_encoder(config)
        decoder = self._setup_decoder(config)

        # construct vae from encoder and decoder
        self.vae = VAE(encoder=encoder, decoder=decoder, param) # param should contain more parameters

        self.dataloader = self._setup_dataset(config)

    def _extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        """
        self.encoder_type = config.get(["encoder", "network_type"])
        self.decoder_type = config.get(["decoder", "network_type"])
        self.approximate_posterior_type = config.get(["model", "approximate_posterior"])

        self.relative_data_path = config.get(["relative_data_path"])
        self.dataset = config.get(["training", "dataset"])
        self.batch_size = config.get(["training", "batch_size"])

    def _setup_encoder(self, config: Dict):
        
        # network
        if self.encoder_type == "feedforward":
            network = feedForwardNetwork(config=config)
        else:
            raise ValueError("Encoder type {} not recognised".format(self.encoder_type))
        
        # approximate posterior family
        if self.approximate_posterior_type == "gaussian":
            approximate_posterior = gaussianPosterior()
        else:
            raise ValueError("Approximate posterior family {} not recognised".format(self.approximate_posterior_type))
        
        return Encoder(network=network, approximate_posterior=approximate_posterior)

    def _setup_decoder(self, config: Dict):
        if self.decoder_type == "feedbackward":
            network = feedBackwardNetwork(config=config)
        else:
            raise ValueError("Decoder type {} not recognised".format(self.decoder_type))

        return network

    def _setup_dataset(self, config: Dict):
        file_path = os.path.dirname(__file__)
        if self.dataset == "mnist":
            dataloader = mnist_dataloader(data_path=os.path.join(file_path, self.relative_data_path), batch_size=self.batch_size, train=True)
        else:
            raise ValueError("Dataset {} not recognised".format(self.dataset))
        return dataloader

    def train(self):
        raise NotImplementedError
