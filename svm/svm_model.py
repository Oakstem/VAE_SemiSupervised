import torch
import numpy as np
import pickle
from utils import subMNIST
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from sklearn import svm
from sklearn.metrics import log_loss
from torch.utils.data import DataLoader
from models import vae_models
from experiment import VAEXperiment
import pandas as pd
from utils import get_config
from utils import parse_args


class SVMClass():
    def __init__(self,
                 vae_model: object,
                 params: dict,
                 svm_model: object = None) -> None:

        self.model = vae_model
        if svm_model is None:
            self.svm = svm.SVC(kernel='rbf', probability=True)
        else:
            self.svm = svm_model
        self.accuracy = 0
        self.loss = 0
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.params = params

    def gen_latent(self, dataset: object, num_samples: int):
        # Generating Latent Vectors
        self.model.eval()
        latent_vec = torch.empty(0).to(self.device)
        labels = torch.empty(0).to(self.device)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

        train_label_index = []
        samples_per_class = np.int(num_samples/10)
        for i in range(10):
            train_label_list = dataset.dataset.targets.numpy()
            label_index = np.where(train_label_list == i)[0]
            label_subindex = list(label_index[:samples_per_class])
            train_label_index += label_subindex

        trainset_np = dataset.dataset.data.numpy()
        trainset_label_np = dataset.dataset.targets.numpy()
        train_data_sub = torch.from_numpy(trainset_np[train_label_index])
        train_labels_sub = torch.from_numpy(trainset_label_np[train_label_index])

        trainset_new = subMNIST(root=self.params['exp_params']['data_path'],
                                train=True, download=True, k=num_samples)
        trainset_new.data = train_data_sub.clone()
        trainset_new.targets = train_labels_sub.clone()

        dataloader = DataLoader(trainset_new, batch_size=10, shuffle=True, drop_last=False)

        for idx, sample in enumerate(dataloader):
            x = sample[0].to(self.device)
            y = sample[1].to(self.device)
            mu, log_var = self.model.encode(x)
            latent_vec = torch.cat((latent_vec, self.model.reparameterize(mu, log_var)))
            labels = torch.cat((labels, y))

        return latent_vec, labels

    def train(self, latent_vec: torch.tensor, labels: torch.tensor):
        # Training SVM
        x = latent_vec.detach()
        y = labels.detach()
        self.svm = self.svm.fit(x, y)

    def test(self, test_dataset: object):
        # Testing with SVM

        self.model.eval()
        dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True, drop_last=True)

        for idx, batch in enumerate(dataloader):
            batch[0] = batch[0].to(self.device)
            batch[1] = batch[1].to(self.device)
            mu, log_var = self.model.encode(batch[0])

            latent_vec = self.model.reparameterize(mu, log_var).detach().tolist()
            labels = batch[1].tolist()

            self.accuracy += self.svm.score(latent_vec, labels)
            y_pred = self.svm.predict_proba(latent_vec)
            self.loss += log_loss(labels, y_pred)

        self.accuracy = self.accuracy / len(dataloader)
        self.loss = self.loss / len(dataloader)


def run_svm_tests(model_name: str, config_file: str):
    # args = parse_args()
    num_samples = [100, 600, 1000, 3000]
    config = get_config(config_file)
    df = pd.DataFrame(columns=['Samples_num', 'Accuracy', 'Loss'])
    model = torch.load(f"trained_models/{model_name}.model")
    experiment = VAEXperiment(model, config['exp_params'], config['logging_params'], config['model_params'])

    for samples in num_samples:
        svm_model = pickle.load(open(f"trained_models/{model_name}_svm_{samples}_samples", 'rb'))
        classifier = SVMClass(model, config, svm_model)
        test_dataset = experiment.test_dataset
        classifier.test(test_dataset)
        print(f"Classifier test result with:{samples} samples, Accuracy:{100*classifier.accuracy:.0f}%,"
            f" Loss:{classifier.loss:.2f}")
        dd = {'Samples_num': samples, 'Accuracy': 100 * classifier.accuracy, 'Loss': classifier.loss.item()}
        df = df.append(dd, ignore_index=True)
        df.to_csv(f'logs/classifiers/class_result_{model_name}_model.csv', index=False)


def run_svm_train(model_name: str, config_file: str):
    # args = parse_args()
    num_samples = [100, 600, 1000, 3000]
    config = get_config(config_file)
    model = torch.load(f"trained_models/{model_name}.model")
    experiment = VAEXperiment(model, config['exp_params'], config['logging_params'], config['model_params'])

    for samples in num_samples:
        classifier = SVMClass(model, config)
        train_dataset = experiment.datasets[0]
        latent, labels = classifier.gen_latent(train_dataset, samples)
        classifier.train(latent, labels)
        path = f"trained_models/{model_name}_svm_{samples}_samples"
        pickle.dump(classifier.svm, open(path, 'wb'))
        print(f"Classifier trained with:{samples} samples, saved at {path}")