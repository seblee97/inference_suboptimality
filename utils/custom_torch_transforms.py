import torch

class TensorFlatten(object):
    """Flatten the image"""

    def __call__(self, tensor):

        tensor = torch.flatten(tensor)

        return tensor

class ImageFlatten(object):
    """Flatten the image"""

    def __call__(self, image):

        img = torch.flatten(image)

        return img