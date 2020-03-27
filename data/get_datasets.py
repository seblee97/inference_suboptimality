import torch
import torchvision

import os
import urllib

file_path = os.path.dirname(__file__)


torchvision.datasets.MNIST(os.path.join(file_path, '.'), download=True)
torchvision.datasets.FashionMNIST(os.path.join(file_path, '.'), download=True)
torchvision.datasets.CIFAR10(os.path.join(file_path, '.'), download=True)

file_path = file_path + "binarizedMNIST/"
if not os.path.exists("binarizedMNIST/"):
    os.makedirs("binarizedMNIST/")
    print("Downloading train")
    urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat',os.path.join(file_path,'binarized_mnist_train.amat'))
    print("Downloading valid")
    urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat',os.path.join(file_path,'binarized_mnist_valid.amat'))
    print("Downloading test")
    urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat',os.path.join(file_path,'binarized_mnist_test.amat'))

