
from context import models, utils

import argparse
import torch
import yaml
import time
import datetime
import os

parser = argparse.ArgumentParser()

parser.add_argument('-config', type=str, help='path to configuration file for student teacher experiment', default='base_config.yaml') # base_config.yaml or base_config_CIFAR.yaml
parser.add_argument('-additional_configs', '--ac', type=str, help='path to folder containing additional configuration files (e.g. for flow)', default='additional_configs/')

# General
parser.add_argument('--experiment_name', type=str, default=None)

# Training
parser.add_argument('-learning_rate', '--lr', type=float, default=None)
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--params', nargs='+', type=float, default=None)
parser.add_argument('-batchsize', '--bs', type=int, default=None)
parser.add_argument('-number_epochs', '--ne', type=int, default=None)
parser.add_argument('-warm_up','--wu', type=int, default=None)
parser.add_argument('-loss_function', '--lf', type=str, default=None)

# Architecture VAE
parser.add_argument('-latent_dimension', '--ld', type=float, default=None)
parser.add_argument('-approximate_posterior', '--ap', type=str, default=None)
parser.add_argument('-is_estimator', '--ie',  type=str, default=None)
parser.add_argument('-encoder_network_type', '--ent', type=str, default=None)
parser.add_argument('-encoder_hidden_dimensions', '--ehd', type=str, default=None)
parser.add_argument('-encoder_output_dimensions', '--eod', type=str, default=None)
parser.add_argument('-decoder_network_type', '--dnt', type=str, default=None)
parser.add_argument('-decoder_hidden_dimensions', '--dhd', type=str, default=None)

parser.add_argument('-decoder_unfreeze_ratio', '--dur', type=float, default=None)
parser.add_argument('-encoder_unfreeze_ratio', '--eur', type=float, default=None)

# Estimator likelihood
parser.add_argument('-estimator_type', '--et', type=str, default=None)
parser.add_argument('-IWAE_samples', '--IWAEs', type=float, default=None)
parser.add_argument('-IWAE_batch_size', '--IWAEbs', type=float, default=None)
parser.add_argument('-AIS_batch_size', '--AISbs', type=float, default=None)

# Local amortisation
parser.add_argument('-optimise_local', '--ol', type=str, default=None)
parser.add_argument('-local_ammortisation_posterior', '--lap', type=str, default=None)
parser.add_argument('-local_num_batches', '--lnb', type=int, default=None)

# Saved model for local optimisation (if none provided, will attempt to find one from hash of config)
parser.add_argument('-local_opt_saved_model', '--losm', type=str, help="path to saved model file for use in local optimisation", default=None)

# Flows

# Auxiliary Flows

# Analysis
parser.add_argument('-analyse', action='store_true', default=None)


args = parser.parse_args()

if __name__ == "__main__":

    main_file_path = os.path.dirname(os.path.realpath(__file__))

    # read base-parameters from base-config
    base_config_full_path = os.path.join(main_file_path, args.config)
    with open(base_config_full_path, 'r') as yaml_file:
        params = yaml.load(yaml_file, yaml.SafeLoader)

    # create object in which to store experiment parameters
    inference_gap_parameters = utils.parameters.InferenceGapParameters(params)

    supplementary_configs_path = args.ac
    additional_configurations = []

    # Update parameters with (optional) args given in command line

    # Experiment level
    if args.experiment_name:
        inference_gap_parameters._config["experiment_name"] = args.experiment_name
    
    # Training level
    if args.lr:
        inference_gap_parameters._config["training"]["learning_rate"] = args.lr
    if args.dataset:
        inference_gap_parameters._config["training"]["dataset"] = args.dataset
    if args.params:
        inference_gap_parameters._config["training"]["optimiser"]["params"] = args.params
    if args.bs:
        inference_gap_parameters._config["training"]["batch_size"] = args.bs
    if args.ne:
        inference_gap_parameters._config["training"]["num_epochs"] = args.ne
    if args.wu:
        inference_gap_parameters._config["training"]["warm_up_program"] = args.wu
    if args.lf:
        inference_gap_parameters._config["training"]["loss_function"] = args.lf
    if args.dur:
        inference_gap_parameters._config["training"]["decoder_unfreeze_ratio"] = args.dur
    if args.eur:
        inference_gap_parameters._config["training"]["encoder_unfreeze_ratio"] = args.eur

    # Model level
    if args.ld:
        inference_gap_parameters._config["model"]["latent_dimension"] = args.ld
    if args.ap:
        inference_gap_parameters._config["model"]["approximate_posterior"] = args.ap
    if args.ie:
        if args.ie == "True":
            args.ie = True
        else:
            args.ie = False
        inference_gap_parameters._config["model"]["is_estimator"] = args.ie
    if args.ent:
        inference_gap_parameters._config["model"]["encoder"]["network_type"] = args.ent
    if args.ehd:
        whole_structure = []
        for item in args.ehd.split(','):
            substructure = []
            for sub_item in item.split(' '):
                if(not(sub_item)):
                    continue
                substructure.append(int(sub_item))
            whole_structure.append(substructure)
        if (len(whole_structure) == 1):
            whole_structure = whole_structure[0]
        inference_gap_parameters._config["model"]["encoder"]["hidden_dimensions"] = whole_structure
    if args.eod:
        inference_gap_parameters._config["model"]["encoder"]["output_dimension_factor"] = args.eod
    if args.dnt:
        inference_gap_parameters._config["model"]["decoder"]["network_type"] = args.dnt
    if args.dhd:
        whole_structure = []
        for item in args.dhd.split(','):
            substructure = []
            for sub_item in item.split(' '):
                if(not(sub_item)):
                    continue
                substructure.append(int(sub_item))
            whole_structure.append(substructure)
        if (len(whole_structure) == 1):
            whole_structure = whole_structure[0]
        inference_gap_parameters._config["model"]["decoder"]["hidden_dimensions"] = whole_structure
    if args.dur:
        inference_gap_parameters._config["model"]["decoder"]["decoder_unfreeze_ratio"] = args.dur

    # Local amortisation level
    if args.ol:
        if args.ol == "True":
            args.ol = True
        else:
            args.ol = False
        inference_gap_parameters._config["model"]["optimise_local"] = args.ol

    approximate_posterior_configuration = inference_gap_parameters.get(["model", "approximate_posterior"])
    if approximate_posterior_configuration == 'gaussian':
        pass
    elif approximate_posterior_configuration == 'rnvp_norm_flow':
        additional_configurations.append(os.path.join(supplementary_configs_path, 'flow_config.yaml'))
    elif approximate_posterior_configuration == 'rnvp_aux_flow':
        additional_configurations.append(os.path.join(supplementary_configs_path, 'aux_flow_config.yaml'))
    else:
        raise ValueError("approximate_posterior_configuration {} not recognised. Please use 'gaussian', \
                             'rnvp_norm_flow', or 'rnvp_aux_flow'".format(approximate_posterior_configuration))

    is_estimator = inference_gap_parameters.get(["model", "is_estimator"])
    if is_estimator:
        additional_configurations.append(os.path.join(supplementary_configs_path, 'estimator_config.yaml'))

    optimise_local = inference_gap_parameters.get(["model", "optimise_local"])
    if optimise_local:
        additional_configurations.append(os.path.join(supplementary_configs_path, 'local_ammortisation_config.yaml'))

    # specific parameters
    for additional_configuration in additional_configurations:
        additional_configuration_full_path = os.path.join(main_file_path, additional_configuration)
        with open(additional_configuration_full_path, 'r') as yaml_file:
            specific_params = yaml.load(yaml_file, yaml.SafeLoader)

        # update base-parameters with specific parameters
        inference_gap_parameters.update(specific_params)

    # Estimator level
    if args.et:
        inference_gap_parameters._config["estimator"]["type"] = args.et
    if args.IWAEs:
        inference_gap_parameters._config["estimator"]["iwae"]["samples"] = args.IWAEs
    if args.IWAEbs:
        inference_gap_parameters._config["estimator"]["iwae"]["batch_size"] = args.IWAEbs
    if args.AISbs:
        inference_gap_parameters._config["estimator"]["ais"]["batch_size"] = args.AISbs

    if optimise_local:
        if args.lap:
            inference_gap_parameters._config["local_ammortisation"]["approximate_posterior"] = args.lap
                # Saved model path for local optimisation
        if args.losm:
            inference_gap_parameters._config["local_ammortisation"]["manual_saved_model_path"] = args.losm
        if args.lnb:
            inference_gap_parameters._config["local_ammortisation"]["num_batches"] = args.lnb

    # establish experiment name / log path etc.
    exp_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    experiment_name = inference_gap_parameters.get("experiment_name")
    log_path = os.path.join(main_file_path, 'results', exp_timestamp, experiment_name)

    inference_gap_parameters.set_property("log_path", log_path)
    inference_gap_parameters.set_property("experiment_timestamp", exp_timestamp)
    inference_gap_parameters.set_property("df_log_path", os.path.join(log_path, 'data_logger.csv'))

    # get specified random seed value from config
    seed_value = inference_gap_parameters.get("seed")

    # import packages with non-deterministic behaviour
    import random
    import numpy as np
    import torch
    # set random seeds for these packages
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # establish whether gpu is available
    if torch.cuda.is_available() and inference_gap_parameters.get('use_gpu'):
        print("Using the GPU")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        inference_gap_parameters.set_property("device", "cuda")
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        print("Using the CPU")
        inference_gap_parameters.set_property("device", "cpu")

    # write copy of config_yaml in model_checkpoint_folder (note this also creates results folder if necessary)
    inference_gap_parameters.save_configuration(log_path)

    # setup saved models path
    saved_models_path = inference_gap_parameters.get("saved_models_path")
    os.makedirs(saved_models_path, exist_ok=True)

    runner = models.VAERunner(config=inference_gap_parameters)

    if args.analyse:
        runner.analyse()
    elif optimise_local:
        runner.train_local_optimisation()
    else:
        runner.train()

