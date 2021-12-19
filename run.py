import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.loggers import TensorBoardLogger
import os
import glob

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


# tt_logger = TestTubeLogger(
#     save_dir=config['logging_params']['save_dir'],
#     name=config['logging_params']['name'],
#     debug=False,
#     create_git_tag=False,
# )
tb_logger = TensorBoardLogger(config['logging_params']['save_dir'], name=config['logging_params']['name'])
# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])
gpus = 1 if torch.cuda.is_available() else []
config['trainer_params']['gpus'] = gpus
runner = Trainer(default_root_dir=f"{tb_logger.save_dir}",
                 min_epochs=1,  
                 logger=tb_logger,
                 log_every_n_steps=100,
                 limit_train_batches=1.,
                 limit_val_batches=1.,
                 num_sanity_val_steps=1,
                 **config['trainer_params'])
#
print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)

################## Loading from Checkpoint Func ##################
checkpoint_path = f"{config['logging_params']['save_dir']}" \
                  f"{config['logging_params']['name']}"
list_of_files = glob.glob(f"{checkpoint_path}/*")
latest_file = max(list_of_files, key=os.path.getctime)
checkpoint_path = glob.glob(f"{latest_file}/checkpoints/*")[0]
from collections import OrderedDict
checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
new_state_dict = OrderedDict()
for k, v in checkpoint["state_dict"].items():
    name = k[6:] # remove module.
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
########################################################################
################## Generating Latent Vectors ###########################
model.eval()
experiment.val_dataloader()
latent_vec = torch.empty(0)
labels = torch.empty(0)
max_samples = 10
for idx, batch in enumerate(experiment.train_dataloader()):
    if idx < max_samples:
        mu, log_var = model.encode(batch[0])
        latent_vec = torch.cat((latent_vec, model.reparameterize(mu, log_var)))
        labels = torch.cat((labels, batch[1]))
    else:
        break
################## Training SVM ###########################
from sklearn import svm
x = latent_vec.tolist()
y = labels.tolist()
clf = svm.SVC()
clf.fit(x, y)
################## Testing with SVM ###########################
latent_vec = torch.empty(0)
labels = torch.empty(0)
preds = torch.empty(0)
model.eval()
for idx, batch in enumerate(experiment.test_dataloader()):
    if idx < max_samples:
        mu, log_var = model.encode(batch[0])
        latent_vec = torch.cat((latent_vec, model.reparameterize(mu, log_var)))
        labels = torch.cat((labels, batch[1]))
    else:
        break
preds = clf.predict(latent_vec.detach())
loss = clf.score(latent_vec.detach(), labels.numpy())
tst = 1
