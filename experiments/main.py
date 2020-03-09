
from context import models, utils

import argparse
import torch
import yaml
import time
import datetime
import os

parser = argparse.ArgumentParser()

parser.add_argument('-config', type=str, help='path to configuration file for student teacher experiment', default='config.yaml')

args = parser.parse_args()

if __name__ == "__main__":

    main_file_path = os.path.dirname(os.path.realpath(__file__))

    # read base-parameters from base-config
    base_config_full_path = os.path.join(main_file_path, args.config)
    with open(base_config_full_path, 'r') as yaml_file:
        params = yaml.load(yaml_file, yaml.SafeLoader)

    # create object in which to store experiment parameters
    inference_gap_parameters = utils.parameters.InferenceGapParameters(params)

    # establish experiment name / log path etc.
    exp_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    experiment_name = inference_gap_parameters.get("experiment_name")
    checkpoint_path = os.path.join(main_file_path, 'results', exp_timestamp, experiment_name)

    inference_gap_parameters.set_property("checkpoint_path", checkpoint_path)
    inference_gap_parameters.set_property("experiment_timestamp", exp_timestamp)

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
        experiment_device = torch.device("cuda:{}".format(args.gpu_id))
    else:
        print("Using the CPU")
        inference_gap_parameters.set_property("device", "cpu")
        experiment_device = torch.device("cpu")

    # write copy of config_yaml in model_checkpoint_folder
    inference_gap_parameters.save_configuration(checkpoint_path)

    runner = models.VAERunner(config=inference_gap_parameters)
    
    runner.train()
    
