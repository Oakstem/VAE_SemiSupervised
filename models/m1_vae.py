import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class M1_VAE(BaseVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(M1_VAE, self).__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.img_sz = kwargs['img_size']

        modules = []
        # if hidden_dims is None:
        hidden_dims = [600, 600]
        dec_hidden_dims = [500]

        # Build Encoder

        self.encoder = nn.Sequential(
                nn.Linear(self.img_sz**2, hidden_dims[0]),
                nn.LeakyReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.LeakyReLU(),
            )


            # in_channels = h_dim

        # self.encoder = nn.Sequential()
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, dec_hidden_dims[-1])

        hidden_dims.reverse()
        self.final_layer = nn.Sequential(
                            # nn.ConvTranspose2d(dec_hidden_dims[-1],
                            #                    dec_hidden_dims[-1],
                            #                    kernel_size=3,
                            #                    stride=2,
                            #                    padding=1,
                            #                    output_padding=1),
                            # nn.BatchNorm2d(dec_hidden_dims[-1]),
                            # nn.LeakyReLU(),
                            # nn.Conv2d(dec_hidden_dims[-1], out_channels=1,
                            #           kernel_size=3, stride=1, padding=1),
                            # nn.Tanh())
                            nn.Linear(dec_hidden_dims[0], dec_hidden_dims[0]),
                            nn.LeakyReLU(),
                            nn.Linear(dec_hidden_dims[0], self.img_sz**2),
                            nn.LeakyReLU())


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = torch.flatten(input, start_dim=1)
        result = self.encoder(result)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # result = result.view(-1, 500, 16, 16)
        result = self.final_layer(result)
        result = result.view(-1, 1, self.img_sz, self.img_sz)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var, z]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        result = {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

        return result

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]