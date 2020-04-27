# TODO

Things we should polish on our code base before we submit. Most of this is cosmetic so I suggest we wait until at least the experiments are finished to absolutely minimise risk of breaking something.

- [ ] Spelling mistakes (e.g. ammortise -> amortise)
- [ ] Anonymise (e.g. authors below but let's make sure there are no other traces)
- [ ] Linked to the above: squash commits. I suggest we copy this code over to the anonymised repos and squash them there since the commit history is useful to have here.
- [ ] Renaming of stuff (e.g. feedforward and feedbackward)
- [ ] Make Experiment_List lower case? I've weirdly grown to like it, given it contains all the hard coded stuff
- [ ] Remove files that aren't used (math_operations? Sylv Flows if we don't get around to them? Do we use the other base configs in the experiment dir?)
- [ ] Linked to above I think early one some pycache folders etc. creeped in before we firmed up the gitignore, we should remove these
- [ ] Run another lint session and get all the type hinting and docstrings etc. in there 
- [ ] Unify the naming conventions in the [experiments/results](experiments/results) directory. 
- [ ] Clean up the script commands that run each experiment (e.g., adding the `--ldo True` flag for local optimisation).
- [ ] Actually run `mypy` over the codebase to make sure our types are consistent.
- [ ] Add tests to functions that are easy to test.
- [ ] ...

# Oxford2020AdvML
This repository contains code for a project undertaken as part of the Advanced Topics in Machine Learning course (HT 2020) at Oxford. The code here was written by Mikhail Andrenkov, Maxence Draguet, Sebastian Lee, and Diane Magnin.

It is a reproduction of the code for the paper [_Inference Suboptimality in Variational Autoencoders_](https://arxiv.org/pdf/1801.03558.pdf) by Cremer, Li & Duvenaud. 

Our code contains the following features relevant to replicating results from the paper:

* Flexible encoder/decoder architectures
* Various approximate posteriors inlcuding:
    * Factorised Gaussian
    * R-NVP flows
    * R-NVP flows with auxiliary variables
* Relevant binarised image datasets inlcuding:
    * MNIST
    * Fashion-MNIST
    * CIFAR-10
* Local optimisation training loop
* AIS and IWAE log-likelihood estimators

Additionally we implemented a planar flows approximate posterior.

## Quick Links
* [Spreadsheet](https://docs.google.com/spreadsheets/d/1y8K3G4ih2Ta9uB6wM7noJpNtmomSwDDZmKUIKUfGlTk/edit#gid=0)
* [Overleaf Report](https://www.overleaf.com/2537812191smnpkcprxdxs)
* [Experiment List](https://docs.google.com/document/d/1mjVGWMD_I13s5KsolYSpAij37CbMO58AiLj4kKjfKHA/edit?usp=sharing)
* [Research Paper](https://arxiv.org/abs/1801.03558)
* [Conference Poster](https://unioxfordnexus-my.sharepoint.com/:p:/g/personal/hert5869_ox_ac_uk/EeDaJHDlDnhNr4R5VdsCbaoBEgjwqa5hrsuq-I7QbCKh8Q?e=uGHYxV)

## Prerequisites

To run this code you will need the following:

* Python 3.7+

Our code uses PyTorch. We include a requirements file (requirements.txt). We recommend creating a virtual environment (using ```conda``` or ```virtualenv```) for this code base e.g.

```python3 -m venv aml; source aml/bin/activate```

From there, all Python prerequisites should be satisfied by running

```pip3 install -r requirements.txt```

To run experiments with a GPU, it is essential to use Python **3.7.5** (on Windows).  Our code is compatible with CUDA 10.1.

## Datasets

We do not provide the datasets directly in this repository. However we are using modifications of standard datasets (e.g. MNIST, CIFAR10) that can be loaded with the torchvision datasets module. To retrieve the datasets, and make the requisite modifications (the binarisation specified by [Larochelle et al](https://dl.acm.org/doi/abs/10.1145/1390156.1390224)) run:

```python data/get_datasets.py```

## Running Code

Standalone experiments can be run from the experiment folder using the main.py script. Configuration for such an experiment can be set using the base_config.yaml file for general attributes of the experiment as well as specific config files in the additional_configs/ folder (e.g. for setting parameters of a flow module).

Running a specific experiment from the paper can be done by accessing the relevant hard coded configuration files in the Experiment_List folder, which have been made to match the specifications of the paper. For example to reproduce the configuration of a fully-factorised gaussian approximate posterior with an amortised inference network (`𝓛(VAE[q]) | qFFG` from Table 2. in the paper), run from the experiments folder:

```python main.py -config Experiment_List/Exp2/base_config.yaml -additional_configs Experiment_List/Exp2/additional_configs/```

Alternatively, all results from a given experiment can be run at once in sequence using the bash script in the relevant experiment folder.

## Accessing Experimental Results

Results of an experiment are by default saved in experiments/results/X/ where X is a timestamp for the experiment. Here you will find a copy of the configuration used to run that experiment, a .csv file containing logging of relevant metrics (e.g. train/test loss), and tensorboard events files. To view the tensorboard logs navigate to this folder and run:

```tensorboard --logdir .```

Alternatively run the command from elsewhere and modify the path accordingly. Plots of an experiment run can also be made by running the plot_from_df.py script from the experiments/plotting folder and passing the path to the folder containing the csv file to the -save_path flag.

Weights of the models being trained in a given experiment are also saved by default in experiment/saved_models/Y/X/ where Y is a hash of the configuration file and X is a timestamp for the experiment. Saved models can be loaded (e.g. to run local optimisation) by specifying the saved model path in the base config (Note they are saved weights and not full checkpoints so cannot be used to resume training).

## Code Structure

Below is the structure of the relevant files in our repository. 

```
│
├── requirements.txt
├── README.md
│     
├── data
│     
├── experiments
│    │
│    │
│    ├── additional_configs
│    │   │
│    │   ├── aux_flow_config.yaml
│    │   ├── esimator_config.yaml
│    │   ├── flow_config.yaml
│    │   ├── local_amortisation_config.yaml
│    │   └── planar_config.yaml
│    │
│    ├── Experiment_List (bash scripts for paper experiments)
│    │   │
│    │   ├── Exp2
│    │   ├── Exp3
│    │   ├── Exp3B
│    │   ├── Exp6
│    │   └── ExpPlanar
│    │
│    ├── plotting
│    │   │
│    │   ├── plot_config.json
│    │   └── plot_from_df.py
│    │
│    ├── results
│    │   │
│    │   └── **result files (not tracked/commited)**
│    │
│    ├── saved_models
│    │   │
│    │   └── **saved_model files (not tracked/commited)**
│    │
│    ├── base_config.yaml
│    ├── context.py
│    └── main.py
│     
├── models
│    │
│    │
│    ├── approximate_posteriors
│    │   │
│    │   ├── __init__.py
│    │   │
│    │   ├── base_approximate_posterior.py
│    │   ├── base_norm_flow.py
│    │   ├── gaussian.py
│    │   ├── planar_flow.py
│    │   ├── rnvp_aux_flow.py
│    │   ├── rnvp_flow.py
│    │   └── sylv_flow.py
│    │
│    ├── likelihood_estimators
│    │   │
│    │   ├── __init__.py
│    │   │
│    │   ├── ais_estimator.yaml
│    │   ├── base_estimator.yaml
│    │   ├── iwae_estimator.yaml
│    │   └── max_estimator.yaml
│    │
│    ├── local_amortisation_modules
│    │   │
│    │   ├── __init__.py
│    │   │
│    │   ├── base_local_amortisation.py
│    │   ├── gaussian_local_amortisation.py
│    │   ├── rnvp_aux_flow_local_amortisation.py
│    │   └── rnvp_flow_local_amortisation.py
│    │
│    ├── loss_modules
│    │   │
│    │   ├── __init__.py
│    │   │
│    │   ├── base_loss.py
│    │   ├── gaussian_loss.py
│    │   ├── planar_loss.py
│    │   ├── rnvp_aux_loss.py
│    │   └── rnvp_loss.py
│    │
│    ├── networks
│    │   │
│    │   ├── __init__.py
│    │   │
│    │   ├── base_network.py
│    │   ├── convolutional.py
│    │   ├── deconvolutional.py
│    │   ├── fc_encoder.py
│    │   └── fc_decoder.py
│    │
│    ├── decoder.py
│    ├── encoder.py
│    ├── vae_runner.py
│    └── vae.py
│    
└── utils
     │
     │
     ├── __init__.py 
     │     
     ├── custom_torch_transforms.py
     ├── dataloaders.py
     ├── math_operations.py
     ├── parameters.py 
     ├── torch_operations_test.py
     └── torch_operations.py             
```
