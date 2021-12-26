import yaml
import argparse
import numpy as np
import torch
from models import vae_models
from experiment import VAEXperiment
from experiment import SaveCallback
from experiment import get_model_version
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from svm.svm_model import SVMClass
import common


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')
parser.add_argument('--limit',  '-l',
                    type=float,
                    help = 'limit dataset length',
                    default='0.01')
parser.add_argument('--model',  '-m',
                    type=str,
                    help = 'model name',
                    default='m1')
parser.add_argument('--train',  '-t',
                    type=int,
                    help = "set '1' for training or '0' testing mode",
                    default='0')


args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

if args.train:
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
else:
    if not common.IN_COLAB:
        #   Test for 3000 labeled samples
        import pandas as pd
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        num_samples = [10, 60, 1000, 3000]
        df = pd.DataFrame(columns=['Samples_num', 'Accuracy', 'Loss'])
        model = vae_models[args.model](**config['model_params'])
        experiment = VAEXperiment(model, config['exp_params'], config['logging_params'], config['model_params'])
        model = experiment.load_checkpoint(path="trained_models/sample-epoch=29-val_loss=0.03-v1.ckpt")
        experiment = VAEXperiment(model, config['exp_params'], config['logging_params'], config['model_params'])

        for samples in num_samples:
            classifier = SVMClass(model, config['exp_params'])
            train_dataset = experiment.datasets[0]
            test_dataset = experiment.test_dataset
            latent, labels = classifier.gen_latent(train_dataset, samples)
            classifier.train(latent, labels)
            classifier.test(test_dataset)
            print(f"Classifier trained with:{samples} samples, Resulted Accuracy:{100*classifier.accuracy:.0f}%,"
                f" Loss:{classifier.loss:.2f}")
            dd = {'Samples_num': samples, 'Accuracy': 100 * classifier.accuracy, 'Loss': classifier.loss.item()}
            df = df.append(dd, ignore_index=True)
            df.to_csv('results.csv', index=False)

