from .vae import VAE

from .networks import feedForwardNetwork
from .networks import feedBackwardNetwork

from .encoder import Encoder
from .decoder import Decoder

from .approximate_posteriors import gaussianPosterior

from .loss_modules import gaussianLoss

from utils import mnist_dataloader

from typing import Dict
import os
import copy
import pandas as pd

import matplotlib.pyplot as plt

import torch

from tensorboardX import SummaryWriter

class VAERunner():

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
        # This should also give activation functions !!
        encoder = self._setup_encoder(config)
        decoder = self._setup_decoder(config)

        # construct vae from encoder and decoder
        self.vae = VAE(encoder=encoder, decoder=decoder) # param should contain more parameters

        # setup loss
        self._setup_loss_module()

        self.dataloader, self.test_data = self._setup_dataset(config)

        self.optimiser = torch.optim.Adam(self.vae.parameters(), lr=self.learning_rate)

        # initialise general tensorboard writer
        self.writer = SummaryWriter(self.checkpoint_path)

    def _extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        """

        self.checkpoint_path = config.get("checkpoint_path")

        self.encoder_type = config.get(["encoder", "network_type"])
        self.decoder_type = config.get(["decoder", "network_type"])
        self.approximate_posterior_type = config.get(["model", "approximate_posterior"])

        self.relative_data_path = config.get(["relative_data_path"])
        self.dataset = config.get(["training", "dataset"])
        self.batch_size = config.get(["training", "batch_size"])
        self.latent_dimension = config.get(["training", "latent_dimension"])

        self.num_epochs = config.get(["training", "num_epochs"])
        self.loss_type = config.get(["training", "loss_function"])
        self.learning_rate = config.get(["training", "learning_rate"])

        self.test_frequency = config.get(["testing", "test_frequency"])
        self.visualise_test = config.get(["testing", "visualise"])

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

    def _setup_loss_module(self):
        if self.approximate_posterior_type == "gaussian":
            self.loss_module = gaussianLoss()
        else:
            raise ValueError("Loss module not correctly specified")

    def _setup_dataset(self, config: Dict):
        file_path = os.path.dirname(__file__)
        if self.dataset == "mnist":
            dataloader = mnist_dataloader(data_path=os.path.join(file_path, self.relative_data_path), batch_size=self.batch_size, train=True)
            test_data = iter(mnist_dataloader(data_path=os.path.join(file_path, self.relative_data_path), batch_size=10000, train=False)).next()[0]
        else:
            raise ValueError("Dataset {} not recognised".format(self.dataset))
        return dataloader, test_data

    def train(self):

        # explicitly set model to train mode
        self.vae.train()

        step_count = 0

        for e in range(self.num_epochs):

            for batch_input, batch_labels in self.dataloader:

                if step_count % self.test_frequency == 0:
                    self._perform_test_loop(step=step_count)
                
                step_count += 1

                vae_output = self.vae(batch_input)

                loss = self.loss_module.compute_loss(x=batch_input, vae_output=vae_output)

                self.optimiser.zero_grad()

                loss.backward()

                self.optimiser.step()

                self.writer.add_scalar("training_loss", float(loss) / self.batch_size, step_count)

    def _perform_test_loop(self, step:int):

        # explicitly set model to evaluation mode
        self.vae.eval()

        with torch.no_grad():

            vae_output = self.vae(self.test_data)

            overall_test_loss = self.loss_module.compute_loss(x=self.test_data, vae_output=vae_output)
    
            self.writer.add_scalar("test_loss", float(overall_test_loss) / 10000, step)

            if self.visualise_test:

                # sample latent variable
                z = torch.randn(1, int(0.5 * self.latent_dimension))

                # pass sample through decoder
                reconstructed_image = self.vae.decoder(z)

                numpy_image = reconstructed_image.detach().numpy().reshape((28, 28))

                fig = plt.figure()
                plt.imshow(numpy_image, cmap='gray')

                self.writer.add_figure("test_autoencoding", fig, step)

        # set model back to train mode
        self.vae.train()
