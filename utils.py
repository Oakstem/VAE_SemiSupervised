import torch
import yaml
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms


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


def get_config(filename: str):
    with open(f"configs/{filename}", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def parse_args():
    # Command line arguments parser. Described as in their 'help' sections.
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vae.yaml')
    parser.add_argument('--limit', '-l',
                        type=float,
                        help='limit dataset length',
                        default='0.01')
    parser.add_argument('--model', '-m',
                        type=str,
                        help='model name',
                        default='m1')
    parser.add_argument('--train', '-t',
                        type=int,
                        help="set '1' for training or '0' testing mode",
                        default='0')
    return parser.parse_args()
