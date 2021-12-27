import argparse
import numpy as np
import torch
from utils import get_config
from utils import parse_args
from models import vae_models
from experiment import VAEXperiment
from experiment import SaveCallback
from experiment import get_model_version
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from svm.svm_model import run_svm_tests
from svm.svm_model import run_svm_train


def train_vae():
    args = parse_args()
    config = get_config(args)

    path = get_model_version(config)
    tb_logger = TensorBoardLogger(config['logging_params']['save_dir'], name=path)
    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    model = vae_models[args.model](**config['model_params'])
    experiment = VAEXperiment(model,
                              config['exp_params'], config['logging_params'], config['model_params'])
    gpus = 1 if torch.cuda.is_available() else []
    config['trainer_params']['gpus'] = gpus

    runner = Trainer(default_root_dir=f"{tb_logger.save_dir}",
                     min_epochs=1,
                     logger=tb_logger,
                     limit_train_batches=args.limit,
                     limit_val_batches=args.limit,
                     num_sanity_val_steps=1,
                     callbacks=[SaveCallback.checkpoint_callback, SaveCallback.checkpoint_callback2, SaveCallback()],
                     **config['trainer_params'])

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)
    print(f"======= Finished Training =======")