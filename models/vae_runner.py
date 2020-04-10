from .vae import VAE

from .networks import feedForwardNetwork
from .networks import feedBackwardNetwork
from .networks import convNetwork
from .networks import deconvNetwork

from .encoder import Encoder
from .decoder import Decoder

from .approximate_posteriors import gaussianPosterior, RNVPPosterior, RNVPAux

from .loss_modules import gaussianLoss, RNVPLoss, RNVPAuxLoss
from .likelihood_estimators import BaseEstimator, AISEstimator, IWAEEstimator, MaxEstimator
from .local_ammortisation_modules import GaussianLocalAmmortisation, RNVPAuxLocalAmmortisation, RNVPLocalAmmortisation

from utils import mnist_dataloader, binarised_mnist_dataloader, fashion_mnist_dataloader, cifar_dataloader, repeat_batch, partition_batch

from typing import Dict
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import hashlib

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

        # construct hash of current run
        self._construct_model_hash(config)

        # set path to save model weights for this run (make config path if necessary)
        config_save_path = os.path.join(self.saved_models_path, self.config_hash)
        self.save_model_path = os.path.join(config_save_path, self.experiment_timestamp)
        os.makedirs(config_save_path, exist_ok=True)

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

        # setup optimiser with parameters from config
        self.optimiser = self._setup_optimiser(config)

        # setup likelihood estimator with parameters from config
        if self.is_estimator:
            self.estimator = self._setup_estimator(config)
        
        if self.optimise_local:
            self.localised_ammortisation_network = self._setup_local_ammortisation(config)

        # initialise general tensorboard writer
        self.writer = SummaryWriter(self.log_path)

        # initialise dataframe to log metrics
        if self.log_to_df:
            self.logger_df = pd.DataFrame()

    def _extract_parameters(self, config: Dict) -> None:
        """
        Method to extract relevant parameters from config and make them attributes of this class
        """
        self.device = config.get("device")

        self.log_path = config.get("log_path")
        self.checkpoint_frequency = config.get("checkpoint_frequency")
        self.experiment_timestamp = config.get("experiment_timestamp")
        self.saved_models_path = config.get("saved_models_path")
        self.log_to_df = config.get("log_to_df")
        self.df_log_path = config.get("df_log_path")
        self.load_decoder_only = config.get("load_decoder_only")

        self.encoder_type = config.get(["model", "encoder", "network_type"])
        self.decoder_type = config.get(["model", "decoder", "network_type"])
        self.approximate_posterior_type = config.get(["model", "approximate_posterior"])
        self.latent_dimension = config.get(["model", "latent_dimension"])

        self.relative_data_path = config.get(["relative_data_path"])
        self.dataset = config.get(["training", "dataset"])
        self.batch_size = config.get(["training", "batch_size"])

        self.warm_up_program = config.get(["training", "warm_up_program"])
        self.num_epochs = config.get(["training", "num_epochs"])
        self.learning_rate = config.get(["training", "learning_rate"])
        self.lr_scheduler = config.get(["training", "lr_scheduler"])

        self.test_frequency = config.get(["testing", "test_frequency"])
        self.visualise_test = config.get(["testing", "visualise"])

        self.optimiser_type = config.get(["training", "optimiser", "type"])
        self.optimiser_params = config.get(["training", "optimiser", "params"])

        self.train_mc_samples = config.get(["training", "mc_samples"])
        self.test_mc_samples = config.get(["testing", "mc_samples"])

        self.is_estimator = config.get(["model", "is_estimator"])
        self.optimise_local = config.get(["model", "optimise_local"])

        if self.optimise_local:
            self.manual_saved_model_path = config.get(["local_ammortisation", "manual_saved_model_path"])

            self.max_num_epochs = config.get(["local_ammortisation", "max_num_epochs"])
            self.convergence_check_period = config.get(["local_ammortisation", "convergence_check_period"])
            self.cycles_until_convergence = config.get(["local_ammortisation", "cycles_until_convergence"])
            self.local_mc_samples = config.get(["local_ammortisation", "mc_samples"])
            self.local_approximate_posterior = config.get(["local_ammortisation", "approximate_posterior"])
            self.local_num_batches = config.get(["local_ammortisation", "num_batches"])

            # account for different inference net output : latent dimension ratio
            factors = {"gaussian": 2, "rnvp_norm_flow": 2, "rnvp_aux_flow": 1}
            self.sample_factor = factors[self.local_approximate_posterior] / 2
            # self.factor = config.get(["local_ammortisation", "encoder", "output_dimension_factor"]) // 2

    def checkpoint_df(self, step: int) -> None:
        """save dataframe"""
        print("Checkpointing Dataframe...")
        # check for existing dataframe
        if self.optimise_local:
            try:
                previous_df = pd.read_csv(self.df_log_path, index_col=0)
                merged_df = pd.concat([previous_df, self.logger_df])
            except:
                merged_df = self.logger_df
        else:
            if step > self.checkpoint_frequency:
                previous_df = pd.read_csv(self.df_log_path, index_col=0)
                merged_df = pd.concat([previous_df, self.logger_df])
            else:
                merged_df = self.logger_df

        merged_df.to_csv(self.df_log_path)
        if self.log_to_df:
            self.logger_df = pd.DataFrame()

    def _construct_model_hash(self, config: Dict):
        """
        Unique signature for current config architecture.
        Note 1. Fields such as learning rate and even non-linearities are not included as they do not 
        affect capacity/architecture of model and thus do not impact save/load compatibility of config.
        Note 2. The field optimise_local is not included in the config, when it is set to true a new hash
        is constructed for this run so that runs can be saved independently without conflict.
        """
        model_specification_components = [
                config.get(["model", "approximate_posterior"]),
                config.get(["model", "input_dimension"]),
                config.get(["model", "latent_dimension"]),
                config.get(["model", "encoder", "network_type"]),
                config.get(["model", "encoder", "hidden_dimensions"]),
                config.get(["model", "encoder", "output_dimension_factor"]),
                config.get(["model", "decoder", "network_type"]),
                config.get(["model", "decoder", "hidden_dimensions"]),
            ]

        if self.approximate_posterior_type == "rnvp_norm_flow":
            model_specification_components.extend([
                config.get(["flow", "flow_layers"])
            ])

        if self.approximate_posterior_type == "rnvp_aux_flow":
            model_specification_components.extend([
                config.get(["flow", "flow_layers"]),
                config.get(["flow", "auxillary_forward_dimensions"]),
                config.get(["flow", "auxillary_reverse_dimensions"])
            ])

        # hash relevant elements of current config to see if trained model exists
        self.config_hash = hashlib.md5(str(model_specification_components).encode('utf-8')).hexdigest()

    def _setup_encoder(self, config: Dict):

        # network
        if self.encoder_type == "feedforward":
            network = feedForwardNetwork(config=config)
        elif self.encoder_type == "convolutional":
            network = convNetwork(config=config)
        else:
            raise ValueError("Encoder type {} not recognised".format(self.encoder_type))

        # approximate posterior family
        if self.approximate_posterior_type == "gaussian":
            approximate_posterior = gaussianPosterior(config=config)
        elif self.approximate_posterior_type == "rnvp_norm_flow":
            approximate_posterior = RNVPPosterior(config=config)
        elif self.approximate_posterior_type == "rnvp_aux_flow":
            approximate_posterior = RNVPAux(config=config)
        else:
            raise ValueError("Approximate posterior family {} not recognised".format(self.approximate_posterior_type))

        # XXX: maybe return flow/approx parameters here and if not None they can be added to optimiser (rather than explicity have flow object be part of vae graph construction)

        return Encoder(network=network, approximate_posterior=approximate_posterior)

    def _setup_decoder(self, config: Dict):
        if self.decoder_type == "feedbackward":
            network = feedBackwardNetwork(config=config)
        elif self.decoder_type == "deconvolutional":
            network = deconvNetwork(config=config)
        else:
            raise ValueError("Decoder type {} not recognised".format(self.decoder_type))

        return Decoder(network=network)

    def _setup_loss_module(self):
        if self.optimise_local:
            approximate_posterior_type = self.local_approximate_posterior
        else:
            approximate_posterior_type = self.approximate_posterior_type

        if approximate_posterior_type == "gaussian":
            self.loss_module = gaussianLoss()
        elif approximate_posterior_type == "rnvp_norm_flow":
            self.loss_module = RNVPLoss()
        elif approximate_posterior_type == "rnvp_aux_flow":
            self.loss_module = RNVPAuxLoss()
        else:
            raise ValueError("Loss module not correctly specified")

    def _setup_dataset(self, config: Dict):
        file_path = os.path.dirname(__file__)
        if self.dataset == "mnist":
            dataloader = mnist_dataloader(data_path=os.path.join(file_path, self.relative_data_path), batch_size=self.batch_size, train=True)
            test_data = iter(mnist_dataloader(data_path=os.path.join(file_path, self.relative_data_path), batch_size=10000, train=False)).next()[0]
        elif self.dataset == "binarised_mnist":
            dataloader = binarised_mnist_dataloader(data_path=os.path.join(file_path, self.relative_data_path), batch_size=self.batch_size, train=True, local=self.optimise_local)
            test_data = iter(binarised_mnist_dataloader(data_path=os.path.join(file_path, self.relative_data_path), batch_size=10000, train=False)).next()[0]
        elif self.dataset == "fashion_mnist":
            dataloader = fashion_mnist_dataloader(data_path=os.path.join(file_path, self.relative_data_path), batch_size=self.batch_size, train=True)
            test_data = iter(fashion_mnist_dataloader(data_path=os.path.join(file_path, self.relative_data_path), batch_size=10000, train=False)).next()[0]
        elif self.dataset == "cifar":
            dataloader = cifar_dataloader(data_path=os.path.join(file_path, self.relative_data_path), batch_size=self.batch_size, train=True)
            test_data = iter(cifar_dataloader(data_path=os.path.join(file_path, self.relative_data_path), batch_size=10000, train=False)).next()[0]
        else:
            raise ValueError("Dataset {} not recognised".format(self.dataset))
        return dataloader, test_data.to(self.device)

    def _setup_optimiser(self, config: Dict):
        if self.optimise_local:
            # Setup new optimiser for each datapoint, no global optimiser
            return None
        if self.optimiser_type == "adam":
            beta_1 = self.optimiser_params[0]
            beta_2 = self.optimiser_params[1]
            epsilon = self.optimiser_params[2]
            return torch.optim.Adam(
                self.vae.parameters(), lr=self.learning_rate, betas=(beta_1, beta_2), eps=epsilon
                )
        else:
            raise ValueError("Optimiser {} not recognised". format(self.optimiser_type))

    def _setup_estimator(self, config: Dict) -> BaseEstimator:
        """
        Sets up the likelihood estimator for this VAERunner.

        :param config: parsed configuration file.
        """
        def construct_estimator(estimator_type: str) -> BaseEstimator:
            """
            Constructs a BaseEstimator of the given type.

            :param estimator_type: Type of the estimator (capitalized)
            :return: The BaseEstimator object.
            """
            if estimator_type == "IWAE":
                num_samples = config.get(['estimator', 'iwae', 'num_samples'])
                batch_size = config.get(['estimator', 'iwae', 'batch_size'])
                return IWAEEstimator(num_samples, batch_size)
            elif estimator_type == "AIS":
                num_chains = config.get(['estimator', 'ais', 'num_chains'])
                batch_size = config.get(['estimator', 'ais', 'batch_size'])
                num_dists = config.get(['estimator', 'ais', 'num_dists'])
                num_leapfrog_steps = config.get(['estimator', 'ais', 'num_leapfrog_steps'])
                return AISEstimator(num_chains, batch_size, self.latent_dimension, num_dists, num_leapfrog_steps)
            elif estimator_type == "MAX":
                ais_estimator = construct_estimator("AIS")
                iwae_estimator = construct_estimator("IWAE")
                return MaxEstimator(ais_estimator, iwae_estimator)
            else:
                raise ValueError("Estimator {} not recognised".format(estimator_type))

        self.estimator_frequency = config.get(['estimator', 'frequency'])
        self.estimator_type = config.get(["estimator", "type"])
        return construct_estimator(self.estimator_type.upper())

    def _setup_local_ammortisation(self, config: Dict):

        if self.local_approximate_posterior == "gaussian":
            local_ammortisation_module = GaussianLocalAmmortisation(config=config)
        elif self.local_approximate_posterior == "rnvp_norm_flow":
            local_ammortisation_module = RNVPLocalAmmortisation(config=config)
        elif self.local_approximate_posterior == "rnvp_aux_flow":
            local_ammortisation_module = RNVPAuxLocalAmmortisation(config=config)
        else:
            raise ValueError("Approximate posterior family {} not recognised for local ammortisation".format(local_approximate_posterior))
            
        return local_ammortisation_module

    def _load_checkpointed_model(self, model_path: str) -> None:
        """
        Load weights saved in specified path into vae model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError("Saved weights for specified config could not be found in specified path. \
                                    To train locally optimised ammortisation, pretrained model is required.")
        else:
           self.vae.load_weights(device=self.device, weights_path=model_path, load_decoder_only=self.load_decoder_only)

    def train_local_optimisation(self) -> None:
        """
        Local optimisation (per data batch) of ammortisation network. 

        Loads pretrained model and freezes weights.
        """
        if self.manual_saved_model_path:
            saved_model_path = self.manual_saved_model_path
        else:
            # check for existing model (with hash signature) and load
            correct_hash_saved_weights_path = os.path.join(self.saved_models_path, self.config_hash)

            # there may be multiple runs with consistent hash that are saved. Heuristic: choose most recent saved run
            consistent_saved_run_paths = sorted(Path(correct_hash_saved_weights_path).iterdir(), key=os.path.getmtime)
            if consistent_saved_run_paths:
                saved_model_path = os.path.join(consistent_saved_run_paths[-1], "saved_vae_weights.pt")
        
        self._load_checkpointed_model(saved_model_path)

        # set model to evaluation mode i.e. freeze weights
        self.vae.eval()

        # artificial step-count for logging purposes
        log_step_count = 0

        # Accumulate the total average loss after each optimisation.
        total_average_loss = 0.0

        # loop through every datapoint and optimise for each individually
        for b, (batch_input, _) in enumerate(self.dataloader, start=1): # target discarded

            print("Locally optimising ammortisation for data batch {}/{}".format(b, self.local_num_batches))

            # take multiple monte carlo samples to reduce variance
            copied_batch = repeat_batch(batch_input, self.local_mc_samples)

            current_average_loss = 0.0
            best_average_loss = np.inf
            num_cycles_without_improvement = 0
                
            # start with unit normal prior
            mean = torch.zeros((self.batch_size * self.local_mc_samples, int(self.sample_factor * self.latent_dimension)), requires_grad=True)
            logvar = torch.zeros((self.batch_size * self.local_mc_samples, int(self.sample_factor * self.latent_dimension)), requires_grad=True)

            # for gaussian case, mean and logvar are only encoder parameters. For flow etc. there are others
            additional_optimisation_parameters = self.localised_ammortisation_network.get_additional_parameters()
            parameters_to_optimise = [{'params': [mean, logvar]}, {'params': additional_optimisation_parameters}]
            
            local_optimiser = self.localised_ammortisation_network.get_local_optimiser(parameters=parameters_to_optimise)

            for e in range(1, self.max_num_epochs + 1):

                log_step_count += 1

                if self.log_to_df:
                    self.logger_df.append(pd.Series(name=log_step_count))
                
                z, params = self.localised_ammortisation_network.sample_latent_vector([mean, logvar])

                reconstruction = self.vae.decoder(z)
                vae_output = {'x_hat': reconstruction, 'z': z, 'params': params}

                loss, _, _ = self.loss_module.compute_loss(x=copied_batch, vae_output=vae_output)

                local_optimiser.zero_grad()
                loss.backward()
                local_optimiser.step()

                # Technically, the numeric precision can be improved by dividing
                # only once; however, with small convergence check periods, it
                # doesn't really matter.
                current_average_loss += float(loss) / self.convergence_check_period

                if self.log_to_df:
                    self.logger_df.at[log_step_count, "local_loss"] = float(loss)

                # check for threshold
                if e % self.convergence_check_period == 0:
                    # Useful message for debugging.
                    # print("Cycles = {}, Current = {}, Best = {}.".format(num_cycles_without_improvement, current_average_loss, best_average_loss))

                    if current_average_loss < best_average_loss:
                        best_average_loss = current_average_loss
                        num_cycles_without_improvement = 0
                    else:
                        num_cycles_without_improvement += 1
                        if num_cycles_without_improvement == self.cycles_until_convergence:
                            break

                    # Reset the current average loss.
                    current_average_loss = 0

            # evaluation
            test_z, test_params = self.localised_ammortisation_network.sample_latent_vector([mean, logvar])
            test_reconstruction = self.vae.decoder(z)
            vae_output = {'x_hat': test_reconstruction, 'z': test_z, 'params': test_params}
            loss, _, _ = self.loss_module.compute_loss(x=copied_batch, vae_output=vae_output)

            test_elbo = -loss
            total_average_loss += float(loss) / self.local_num_batches
            
            if self.log_to_df:
                self.logger_df.at[log_step_count, "test_loss"] = float(loss)

            self.checkpoint_df(log_step_count)

            # Stop after optimising |self.local_num_batches| batches.
            if b == self.local_num_batches:
                break

        print("Average loss after {} batches: {}.".format(self.local_num_batches, total_average_loss))

    def train(self):

        # explicitly set model to train mode
        self.vae.train()

        step_count = 0
        
        # For the LR scheduler.  See https://arxiv.org/abs/1509.00519 for more details.
        exponent_of_3 = 0
        epoch_elapsed = 0
        
        for epoch in range(1, self.num_epochs + 1):
            
            if self.lr_scheduler:
                if epoch_elapsed >= 3 ** exponent_of_3:
                    self.learning_rate *= 10 ** (-1 / 7)
                    exponent_of_3 += 1
                    epoch_elapsed = 0
                    # Update the learning rate of the optimiser.
                    for param_group in self.optimiser.param_groups:
                        param_group['lr'] = self.learning_rate
                epoch_elapsed += 1
            
            for batch_input, _ in self.dataloader: # target discarded

                step_count += 1

                if self.log_to_df:
                    self.logger_df.append(pd.Series(name=step_count))

                # Move the batch input onto the GPU if necessary.
                batch_input = batch_input.to(self.device)

                # Take multiple expectations of the ELBO to reduce variance.
                batch_input = repeat_batch(batch_input, self.train_mc_samples)

                vae_output = self.vae(batch_input)

                # Get entropy-annealing factor for linear program
                warm_up_factor = 1.0
                if self.warm_up_program != 0:
                    warm_up_factor = min(1.0, epoch / self.warm_up_program)

                loss, loss_metrics, _ = self.loss_module.compute_loss(x=batch_input, vae_output=vae_output, warm_up=warm_up_factor)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                self.writer.add_scalar("training_loss", float(loss), step_count)
                self.logger_df.at[step_count, "training_loss"] = float(loss)

                for metric in loss_metrics: # should include e.g. elbo
                    self.writer.add_scalar(metric, loss_metrics[metric], step_count)
                    self.logger_df.at[step_count, metric] = loss_metrics[metric]

                if step_count % self.checkpoint_frequency == 0:
                    self.vae.checkpoint_model_weights(path=self.save_model_path)
                    self.checkpoint_df(step_count)

                if step_count % self.test_frequency == 0:
                    self._perform_test_loop(step=step_count)

                if self.is_estimator and step_count % self.estimator_frequency == 0:
                    estimated_loss = self.estimator.estimate_log_likelihood_loss(self.test_data, self.vae, self.loss_module)
                    self.writer.add_scalar("estimated_loss", float(estimated_loss), step_count)
                    print("Estimated loss after {} steps: {}".format(step_count, float(estimated_loss)))

            print("Training loss after {} epochs ({} steps): {}".format(epoch, step_count, float(loss)))

    def _perform_test_loop(self, step:int):
        # explicitly set model to evaluation mode
        self.vae.eval()

        with torch.no_grad():
            # Take multiple expectations of the ELBO to reduce variance.
            test_batch = repeat_batch(self.test_data, self.test_mc_samples)

            vae_output = self.vae(test_batch)
            overall_test_loss, _, _ = self.loss_module.compute_loss(x=test_batch, vae_output=vae_output)
            self.writer.add_scalar("test_loss", float(overall_test_loss), step)
            self.logger_df.at[step, "test_loss"] = float(overall_test_loss)
            print("Testing loss after {} steps: {}".format(step, float(overall_test_loss)))

            if self.visualise_test:
                index = random.randint(0, self.test_data.size(0) - 1)
                if self.dataset == "cifar":
                    #Test 1: closeness output-input
                    reconstructed_image = torch.sigmoid(vae_output['x_hat'][index])
                    numpy_image = np.transpose(((reconstructed_image.detach().cpu()>= 0.5).float()).numpy(), (1, 2, 0))
                    numpy_input = np.transpose(self.test_data[index].detach().cpu().float().numpy(), (1, 2, 0))

                    fig, (ax0, ax1) = plt.subplots(ncols=2)
                    ax0.imshow(numpy_image)
                    ax1.imshow(numpy_input)
                    self.writer.add_figure("test_autoencoding", fig, step)

                    #Test 2: random latent variable sample (i.e. from prior)
                    z = torch.randn(1, self.latent_dimension)
                    reconstructed_image = torch.sigmoid(self.vae.decoder(z))[0]
                    numpy_image = np.transpose(((reconstructed_image.detach().cpu()>= 0.5).float()).numpy(), (1, 2, 0))

                    fig2 = plt.figure()
                    plt.imshow(numpy_image)
                    self.writer.add_figure("test_autoencoding_random_latent", fig2, step)

                else:
                    #Test 1: closeness output-input
                    reconstructed_image = torch.sigmoid(vae_output['x_hat'][index])
                    numpy_image = reconstructed_image.detach().cpu().numpy().reshape((28, 28))
                    numpy_input = self.test_data[index].detach().cpu().numpy().reshape((28, 28))

                    fig, (ax0, ax1) = plt.subplots(ncols=2)
                    ax0.imshow(numpy_image, cmap='gray')
                    ax1.imshow(numpy_input, cmap='gray')
                    self.writer.add_figure("test_autoencoding", fig, step)

                    #Test 2: random latent variable sample (i.e. from prior)
                    z = torch.randn(1, self.latent_dimension)
                    reconstructed_image = torch.sigmoid(self.vae.decoder(z))
                    numpy_image = reconstructed_image.detach().cpu().numpy().reshape((28, 28))
                    fig2 = plt.figure()
                    plt.imshow(numpy_image, cmap='gray')
                    self.writer.add_figure("test_autoencoding_random_latent", fig2, step)
        # set model back to train mode
        self.vae.train()

    def analyse(self, saved_model_path) -> None:
        """
        Calculates the average test loss and IWAE estimated loss of a saved VAE.

        :param saved_model_path: path to the saved VAE model (optional).
        """
        # Load the model to be analysed.  This code snippet was copied from
        # VAERunner.train_local_optimisation().
        if not saved_model_path:
            # check for existing model (with hash signature) and load
            correct_hash_saved_weights_path = os.path.join(self.saved_models_path, self.config_hash)

            # there may be multiple runs with consistent hash that are saved. Heuristic: choose most recent saved run
            consistent_saved_run_paths = sorted(Path(correct_hash_saved_weights_path).iterdir(), key=os.path.getmtime)
            if consistent_saved_run_paths:
                saved_model_path = os.path.join(consistent_saved_run_paths[-1], "saved_vae_weights.pt")
            else:
                raise Exception("Saved weights for specified config could not be found in specified path. \
                                To analyse the test loss, a pretrained model is required.")

        self._load_checkpointed_model(saved_model_path)

        self.vae.eval()

        with torch.no_grad():
            # Extract the training dataset for convenience.
            training_data = torch.cat([minibatch.to(self.device) for minibatch, _ in self.dataloader], dim=0)

            # Compute the average training loss using |self.train_mc_samples|
            # samples from each element of the training set.
            mean_loss = torch.Tensor([0.0])
            for minibatch in partition_batch(training_data, self.batch_size):
                minibatch = repeat_batch(minibatch, self.train_mc_samples)
                vae_output = self.vae(minibatch)
                loss, _, _ = self.loss_module.compute_loss(x=minibatch, vae_output=vae_output)
                mean_loss += loss * len(minibatch)
                del minibatch, vae_output
            mean_loss /= len(training_data) * self.train_mc_samples
            print("Training loss using {} MC samples: {}.".format(self.train_mc_samples, float(mean_loss)))

            # Ensure the GPU has enough memory to perform the estimation.
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Compute the estimated test loss (if applicable).
            if self.is_estimator:
                estimated_loss = self.estimator.estimate_log_likelihood_loss(training_data, self.vae, self.loss_module)
                print("Estimated loss using estimator '{}': {}.".format(self.estimator_type, float(estimated_loss)))
