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
import copy

import torch

class VAERunner:

    def __init__(self, config: Dict) -> None:
        """
        Class for constructing variational autoencoder (from encoder, approximate posterior family, and decoder)
        and orchestrating training.

        :param config: dictionary containing parameters to specify training etc.
        """
        # extract relevant parameters from config
        self._extract_parameters(config)

        # initialise loss function and optimiser
        self.loss_function = torch.nn.MSELoss()

        # initialise encoder, decoder
        encoder = self._setup_encoder(config)
        decoder = self._setup_decoder(config)

        # construct vae from encoder and decoder
        self.vae = VAE(encoder=encoder, decoder=decoder)

        # setup loss
        self._setup_loss()

        self.dataloader = self._setup_dataset(config)

        self.optimiser = torch.optim.Adam(self.vae.parameters(), lr=self.learning_rate)

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

        self.num_epochs = config.get(["training", "num_epochs"])
        self.loss_type = config.get(["training", "loss_function"])
        self.learning_rate = config.get(["training", "learning_rate"])

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

    def _setup_loss(self):
        if self.loss_type == "bce":
            self.loss_function = nn.BCELoss()

    def _setup_dataset(self, config: Dict):
        file_path = os.path.dirname(__file__)
        if self.dataset == "mnist":
            dataloader = mnist_dataloader(data_path=os.path.join(file_path, self.relative_data_path), batch_size=self.batch_size, train=True)
        else:
            raise ValueError("Dataset {} not recognised".format(self.dataset))
        return dataloader

    def train(self):

        for e in range(self.num_epochs):

            for batch_input, batch_labels in self.dataloader:

                resized_input = batch_input.view((-1, 784))
                
                encoder_reconstruction, plogqz, logpz = self.vae(resized_input)

                reconstruction_loss = F.binary_cross_entropy(encoder_reconstruction, resized_input, size_average=False)
                
                elbo = reconstruction_loss + logpz - plogqz
                loss = -elbo

                self.optimiser.zero_grad()

                loss.backward()

                self.optimiser.step()

                print(float(loss))
