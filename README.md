# Oxford2020AdvML
Advanced Machine Learning 2020 HT Oxford project

## Quick Links
* [Spreadsheet](https://docs.google.com/spreadsheets/d/1y8K3G4ih2Ta9uB6wM7noJpNtmomSwDDZmKUIKUfGlTk/edit#gid=0)
* [Overleaf Report](https://www.overleaf.com/2537812191smnpkcprxdxs)
* [Experiment List](https://docs.google.com/document/d/1mjVGWMD_I13s5KsolYSpAij37CbMO58AiLj4kKjfKHA/edit?usp=sharing)
* [Research Paper](https://arxiv.org/abs/1801.03558)
* [Conference Poster](https://docs.google.com/presentation/d/1sgkaef6lSHZtU6eBMmfr-AtU_yjEadXdqj8ZBqy02gM/edit?usp=sharing)

## Prerequisites

* python 3+

Our code uses PyTorch. We include a requirements file (requirements.txt). We recommend creating a virtual environment (using ```conda``` or ```virtualenv```) for this code base e.g.

```python3 -m venv aml; source aml/bin/activate```

From then, all python prerequisites should be satisfied by running

```pip3 install -r requirements.txt```

To run experiments with a GPU, it is essential to use Python **3.7.5** (on Windows).  Our code is compatible with CUDA 10.1.

## Datasets

We do not provide the datasets directly in this repository. However we are using standard datasets (e.g. MNIST, CIFAR10) that can be loaded with the torchvision datasets module. To retrive the datasets, run:

```python data/get_datasets.py```

## Running Code

Specify the experiment type using experiments/config.yaml. Then you can run an experiment using

```python main.py```
