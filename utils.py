import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


from torchvision.datasets import FashionMNIST

## Utils to handle newer PyTorch Lightning changes from version 0.6
## ==================================================================================================== ##


def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try: # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except: # Works for version > 0.6.0
            return fn(self)

    return func_wrapper


class subMNIST(FashionMNIST):
    def __init__(self, root, train=True, target_transform=None, download=False, k=3000):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        transforms.Resize(32)
                                        ])
        super(subMNIST, self).__init__(root, train, transform, target_transform, download)
        self.k = k

    def __len__(self):
        if self.train:
            return self.k
        else:
            return 10000