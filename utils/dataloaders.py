import torch
import torchvision

def mnist_dataloader(data_path: str, batch_size: int, train:bool=True):
    """
    Load mnist image data from specified, convert to grayscaled tensors, flatten, return dataloader

    :param data_path: full path to data directory
    :param batch_size: batch size for dataloader
    :param train: whether to load train or test data
    :return dataloader: pytorch dataloader for mnist training dataset
    """
    # transforms to add to data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()
    ])

    mnist_data = torchvision.datasets.MNIST(data_path, transform=transform, train=train)
    dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

    return dataloader