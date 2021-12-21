from sklearn import svm
import torch
from torch.utils.data import DataLoader


class SVMClass():
    def __init__(self,
                 vae_model: object,
                 params: dict) -> None:

        self.model = vae_model
        self.svm = svm.SVC(probability=True)
        self.params = params
        self.accuracy = 0
        self.loss = 100

    def gen_latent(self, dataset, num_samples):
        # Generating Latent Vectors
        self.model.eval()
        latent_vec = torch.empty(0)
        labels = torch.empty(0)
        dataloader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle = False, drop_last=True)
        for idx, batch in enumerate(dataloader):
            if idx < num_samples:
                mu, log_var = self.model.encode(batch[0])
                latent_vec = torch.cat((latent_vec, self.model.reparameterize(mu, log_var)))
                labels = torch.cat((labels, batch[1]))
            else:
                break
        return latent_vec, labels

    def train(self, latent_vec, labels):
        # Training SVM
        x = latent_vec.tolist()
        y = labels.tolist()
        self.svm = self.svm.fit(x, y)

    def test(self, test_dataset, num_samples=10):
        # Testing with SVM
        latent_vec = torch.empty(0)
        labels = torch.empty(0)

        self.model.eval()
        dataloader = DataLoader(test_dataset, batch_size=self.params['batch_size'], shuffle=False, drop_last=True)
        for idx, batch in enumerate(dataloader):
            if idx < num_samples:
                mu, log_var = self.model.encode(batch[0])
                latent_vec = torch.cat((latent_vec, self.model.reparameterize(mu, log_var)))
                labels = torch.cat((labels, batch[1]))
            else:
                break
        preds = torch.tensor(self.svm.predict_proba(latent_vec.detach()))
        self.accuracy = self.svm.score(latent_vec.detach(), labels)
        loss = torch.nn.CrossEntropyLoss()
        self.loss = loss(preds, torch.tensor(labels, dtype=torch.int64))
