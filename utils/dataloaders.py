import torch
import torchvision

from .custom_torch_transforms import TensorFlatten
import os
import numpy as np
def mnist_dataloader(data_path: str, batch_size: int, train:bool=True):
    """
    Load mnist image data from specified, convert to grayscaled (then binarised) tensors, flatten, return dataloader

    :param data_path: full path to data directory
    :param batch_size: batch size for dataloader
    :param train: whether to load train or test data
    :return dataloader: pytorch dataloader for mnist training dataset
    """
    # transforms to add to data
    transform = torchvision.transforms.Compose([
                                                torchvision.transforms.Grayscale(),
                                                torchvision.transforms.ToTensor(),
                                                lambda x: x>=0.5,
                                                lambda x: x.float(),
                                                TensorFlatten()
                                                ])

    mnist_data = torchvision.datasets.MNIST(data_path, transform=transform, train=train)
    dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

    return dataloader

def binarised_mnist_dataloader(data_path: str, batch_size: int, train:bool=True):
    """
        Load binarised mnist image data from specified, convert to  tensors, flatten, return dataloader
        
        :param data_path: full path to data directory
        :param batch_size: batch size for dataloader
        :param train: whether to load train or test data
        :return dataloader: pytorch dataloader for mnist training dataset
    """
    def read_file_to_numpy(file_name):
        mynumbers = []
        with open(file_name) as f:
            for line in f:
                mynumbers.append(np.array([float(n) for n in line.strip().split()]))
        return mynumbers

    train_file,valid_file,test_file = [os.path.join(data_path, 'binarizedMNIST/binarized_mnist_' + ds + '.amat') for ds in ['train','valid','test']]

    if train:
        train_ar,valid_ar = [read_file_to_numpy(f) for f in [train_file,valid_file]]
        train_ar = np.concatenate((train_ar, valid_ar), axis=0) #discard the valid set and put into the train one.
        tensorData = torch.Tensor(train_ar)
    else:
        test_ar = read_file_to_numpy(test_file)
        tensorData = torch.Tensor(test_ar)

    tensorDatabaset = torch.utils.data.TensorDataset(tensorData)
    dataloader = torch.utils.data.DataLoader(tensorDatabaset, batch_size=batch_size, shuffle=True)
    return dataloader


def fashion_mnist_dataloader(data_path: str, batch_size: int, train:bool=True):
    """
        Load fashion_mnist image data from specified location, convert to tensors, binarise, flatten, and return dataloader
        Note: fashion_mnist already grayscaled in each channel.
        :param data_path: full path to data directory
        :param batch_size: batch size for dataloader
        :param train: whether to load train or test data
        :return dataloader: pytorch dataloader for mnist training dataset
        """
    # transforms to add to data
    transform = torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                lambda x: x>=0.5,
                                                lambda x: x.float(),
                                                TensorFlatten()
                                                ])
        
    fashion_mnist_data = torchvision.datasets.FashionMNIST(data_path, transform=transform, train=train)
    dataloader = torch.utils.data.DataLoader(fashion_mnist_data, batch_size=batch_size, shuffle=True)
                                                
    return dataloader

def cifar_dataloader(data_path: str, batch_size: int, train:bool=True):
    """
        Load cifar image data from specified location, convert to  tensors, binarise, and return dataloader
        Note: Cifar already grayscaled in each channel.
        
        :param data_path: full path to data directory
        :param batch_size: batch size for dataloader
        :param train: whether to load train or test data
        :return dataloader: pytorch dataloader for mnist training dataset
        """
    # transforms to add to data
    transform = torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                lambda x: x>=0.5,
                                                lambda x: x.float(),
                                                ])
        
    CIFAR10_data = torchvision.datasets.CIFAR10(data_path, transform=transform, train=train)
    dataloader = torch.utils.data.DataLoader(CIFAR10_data, batch_size=batch_size, shuffle=True)
                                                
    return dataloader
