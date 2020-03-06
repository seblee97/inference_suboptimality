import torch
import torchvision

import os

file_path = os.path.dirname(__file__)

torchvision.datasets.MNIST(os.path.join(file_path, '.'), download=True)
torchvision.datasets.FashionMNIST(os.path.join(file_path, '.'), download=True)
torchvision.datasets.CIFAR10(os.path.join(file_path, '.'), download=True)
