# Oxford2020AdvML
Advanced Machine Learning 2020 HT Oxford project

## Quick Links
* [Research Notes](https://docs.google.com/spreadsheets/d/1y8K3G4ih2Ta9uB6wM7noJpNtmomSwDDZmKUIKUfGlTk/edit#gid=0)
* [Research Paper](https://arxiv.org/abs/1801.03558)

## Prerequisites

* python 3+

Our code uses PyTorch. We include a requirements file (requirements.txt). We recommend creating a virtual environment (using ```conda``` or ```virtualenv```) for this code base e.g.

```python3 -m venv aml; source aml/bin/activate```

From then, all python prerequisites should be satisfied by running

```pip3 install -r requirements.txt```

## Datasets

We do not provide the datasets directly in this repository. However we are using standard datasets that can be loaded with the torchvision datasets module. To retrive the datasets, run

```python data/get_datasets.py```
