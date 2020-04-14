# Oxford2020AdvML
This repository contains code for a project undertaken as part of the Advanced Topics in Machine Learning course (HT 2020) at Oxford. The code here was written by Mikhail Andrenkov, 

It is a reproduction of the code for the paper [_Inference Suboptimality in Variational Autoencoders_](https://arxiv.org/pdf/1801.03558.pdf) by Cremer, Li & Duvenaud. 

## Quick Links
* [Spreadsheet](https://docs.google.com/spreadsheets/d/1y8K3G4ih2Ta9uB6wM7noJpNtmomSwDDZmKUIKUfGlTk/edit#gid=0)
* [Overleaf Report](https://www.overleaf.com/2537812191smnpkcprxdxs)
* [Experiment List](https://docs.google.com/document/d/1mjVGWMD_I13s5KsolYSpAij37CbMO58AiLj4kKjfKHA/edit?usp=sharing)
* [Research Paper](https://arxiv.org/abs/1801.03558)
* [Conference Poster](https://docs.google.com/presentation/d/1sgkaef6lSHZtU6eBMmfr-AtU_yjEadXdqj8ZBqy02gM/edit?usp=sharing)

## Prerequisites

To run this code you will need the following:

* python 3+

Our code uses PyTorch. We include a requirements file (requirements.txt). We recommend creating a virtual environment (using ```conda``` or ```virtualenv```) for this code base e.g.

```python3 -m venv aml; source aml/bin/activate```

From then, all Python prerequisites should be satisfied by running

```pip3 install -r requirements.txt```

To run experiments with a GPU, it is essential to use Python **3.7.5** (on Windows).  Our code is compatible with CUDA 10.1.

## Datasets

We do not provide the datasets directly in this repository. However we are using standard datasets (e.g. MNIST, CIFAR10) that can be loaded with the torchvision datasets module. To retrive the datasets, run:

```python data/get_datasets.py```

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
│    │   ├── local_ammortisation_config.yaml
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
│    ├── local_ammortisation_modules
│    │   │
│    │   ├── __init__.py
│    │   │
│    │   ├── base_local_ammortisation.py
│    │   ├── gaussian_local_ammortisation.py
│    │   ├── rnvp_aux_flow_local_ammortisation.py
│    │   └── rnvp_flow_local_ammortisation.py
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
│    │   ├── feedbackward.py
│    │   └── feedforward.py
│    │
│    ├── decoder.py
│    ├── encoder.py
│    ├── vae_runner.py
│    └── vae.py
│    
├── utils
│    │
│    │
│    ├── __init__.py 
│    ├── model.py 
│    └── sinusoid.py
│     
├── tests
│    │
│    │
│    ├── test_configs
│    │   │
│    │   ├── test_base_config.yaml
│    │   └── test_maml_config.yaml
│    │
│    ├── __init__.py 
│    ├── context.py
│    ├── test_base_priority_queue.py
│    └── test_sin_priority_queue.py
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

## Running Code

Specify the experiment type using experiments/config.yaml. Then you can run an experiment using

```python main.py```
