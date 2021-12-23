import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
from experiment import SaveCallback
from experiment import get_model_version
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from svm.utils import SVMClass


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


path = get_model_version(config)
tb_logger = TensorBoardLogger(config['logging_params']['save_dir'], name=path)
# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'], config['logging_params'], config['model_params'])
gpus = 1 if torch.cuda.is_available() else []
config['trainer_params']['gpus'] = gpus
runner = Trainer(default_root_dir=f"{tb_logger.save_dir}",
                 min_epochs=1,
                 logger=tb_logger,
                 limit_train_batches=0.01,
                 limit_val_batches=0.1,
                 num_sanity_val_steps=1,
                 callbacks=[SaveCallback().checkpoint_callback, SaveCallback()],
                 **config['trainer_params'])
#
print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)

print(f"======= Finished Training =======")

#   Test for 3000 labeled samples
import pandas as pd
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_samples = [10, 60, 1000, 3000]
df = pd.DataFrame(columns=['Samples_num', 'Accuracy', 'Loss'])
model = torch.load(config['logging_params']['best_model_dir'], map_location=device)
experiment = VAEXperiment(model, config['exp_params'], config['logging_params'])
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

