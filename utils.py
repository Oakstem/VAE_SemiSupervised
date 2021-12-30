import torch
import yaml
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms


def get_config(filename: str):
    with open(f"configs/{filename}.yaml", 'r') as file:
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
                        default='vae')
    parser.add_argument('--limit', '-l',
                        type=float,
                        help='limit dataset length',
                        default='0.05')
    parser.add_argument('--model', '-m',
                        type=str,
                        help='model name',
                        default='vanilla')
    return parser.parse_args()
