from sklearn import svm
from sklearn.metrics import log_loss
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from torchvision.datasets import FashionMNIST


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

