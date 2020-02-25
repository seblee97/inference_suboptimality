import torch
import torchvision

torchvision.datasets.MNIST('.', download=True)
torchvision.datasets.FashionMNIST('.', download=True)
torchvision.datasets.CIFAR10('.', download=True)
