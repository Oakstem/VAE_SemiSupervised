from .base import *
from .vanilla_vae import *
from .m1_vae import *

# Aliases
VAE = VanillaVAE

vae_models = {'vanilla':VanillaVAE, 'm1':M1_VAE}
