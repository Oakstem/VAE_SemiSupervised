import math
import torch
import torchvision
import numpy as np
import sklearn.utils.validation as check
from torch import optim
from models import BaseVAE
from models.types_ import *
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback


## Utils to handle newer PyTorch Lightning changes from version 0.6
## ==================================================================================================== ##

def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try:  # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except:  # Works for version > 0.6.0
            return fn(self)

    return func_wrapper


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict, log_params: dict, model_params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.log_params = log_params
        self.curr_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.hold_graph = False
        self.datasets = []  # train&val datasets
        self.test_dataset = []  # test dataset
        self.num_train_imgs = 0
        self.num_test_imgs = 0
        self.epoch_loss = dict.fromkeys(('loss', 'Reconstruction_Loss', 'KLD', 'SVM_Accuracy'), 0)
        self.val_sz = 0.1
        self.run = f"-latent_sz:{model_params['latent_dim']}"
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="checkpoints/",
            filename="vae-{epoch:02d}-{val_loss:.2f}",
            mode="min"
        )
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        self.load_datasets()

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        # print('Forward Pass')
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)
        return train_loss

    def training_epoch_end(self, outputs: list):
        self.logger.experiment.add_scalar("Loss/Train", outputs[0]['loss'].item(), self.current_epoch + 1)

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)

        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['kld_weight'],
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)
        if self.current_epoch == 0:
            self.logger.experiment.add_graph(self.model, real_img)

        return val_loss

    def validation_epoch_end(self, outputs):
        # logging for checkpoint monitoring
        self.log('val_loss', outputs[0]['loss'].item(), on_epoch=True, on_step=False)
        self.logger.experiment.add_scalar("Loss/Val", outputs[0]['loss'].item(), self.current_epoch)
        if self.current_epoch % 5 == 0:
            self.sample_images()

        return {'val_loss': outputs[0]['loss'].item()}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=8)
        img_grid = torchvision.utils.make_grid(recons)
        self.logger.experiment.add_image('Encoder generated images', img_grid)

        del test_input, recons

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        dataset = self.datasets[0]
        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          drop_last=True)

    @data_loader
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          drop_last=True)

    @data_loader
    def val_dataloader(self):
        self.sample_dataloader = DataLoader(self.datasets[1],
                                            batch_size=self.params['batch_size'],
                                            shuffle=False,
                                            drop_last=True)
        self.num_val_imgs = len(self.datasets[1])

        return self.sample_dataloader

    def load_datasets(self):
        transform = self.data_transforms()
        if self.params['dataset'] == 'fmnist':
            mnist_dataset = FashionMNIST(root=self.params['data_path'],
                                         train=True,
                                         transform=transform,
                                         download=True)
            val_samples = np.round(len(mnist_dataset) * self.val_sz).astype(int)
            train_samples = len(mnist_dataset) - val_samples
            self.datasets = random_split(mnist_dataset, [train_samples, val_samples],
                                         generator=torch.Generator().manual_seed(42))
            self.test_dataset = FashionMNIST(root=self.params['data_path'],
                                             train=False,
                                             transform=transform,
                                             download=True)
        elif self.params['dataset'] == 'mnist':
            mnist_dataset = MNIST(root=self.params['data_path'],
                                  train=True,
                                  transform=transform,
                                  download=True)
            val_samples = np.round(len(mnist_dataset) * self.val_sz).astype(int)
            train_samples = len(mnist_dataset) - val_samples
            self.datasets = random_split(mnist_dataset, [train_samples, val_samples],
                                         generator=torch.Generator().manual_seed(42))
            self.test_dataset = MNIST(root=self.params['data_path'],
                                      train=False,
                                      transform=transform,
                                      download=True)

    def data_transforms(self):
        # SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        # SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'fmnist':
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.3,), (0.3,)),
                                            transforms.Resize(self.params['img_size'])
                                            ])
        elif self.params['dataset'] == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5,), (0.5,)),
                                            transforms.Resize(self.params['img_size'])
                                            ])
        else:
            raise ValueError('Undefined dataset type')
        return transform

    def load_checkpoint(self, path=None):
        from collections import OrderedDict
        if path is None:
            checkpoint_path = f"{self.log_params['save_dir']}" \
                              f"{self.log_params['name']}"
            list_of_files = glob.glob(f"{checkpoint_path}/*")
            latest_file = max(list_of_files, key=os.path.getctime)
            checkpoint_path = glob.glob(f"{latest_file}/checkpoints/*")[0]
        else:
            checkpoint_path = path

        checkpoint = torch.load(checkpoint_path, map_location=self.curr_device)
        new_state_dict = OrderedDict()
        if "model" in next(iter(checkpoint['state_dict'])):
            for k, v in checkpoint["state_dict"].items():
                name = k[6:]  # remove "module"
                new_state_dict[name] = v
        else:
            new_state_dict = checkpoint['state_dict']
        self.model.load_state_dict(new_state_dict)
        return self.model


class SaveCallback(Callback):
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="best-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    checkpoint_callback2 = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="sample-{epoch:02d}-{val_loss:.2f}",
    )

    def on_train_end(self, trainer, experiment):
        print("Training is done.")
        # Saving the model with best results as "best.model"
        best = experiment.load_checkpoint(path=trainer.checkpoint_callback.best_model_path)
        torch.save(best, experiment.log_params['best_model_dir'])


def get_model_version(params: dict):
    path = f"run-latent_sz_{params['model_params']['latent_dim']}"
    # path = "logs/"
    return path
