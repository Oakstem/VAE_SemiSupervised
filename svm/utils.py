from sklearn import svm
from sklearn.metrics import log_loss
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
        self.loss = 0
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def gen_latent(self, dataset: object, num_samples: int):
        # Generating Latent Vectors
        self.model.eval()
        latent_vec = torch.empty(0).to(self.device)
        labels = torch.empty(0).to(self.device)
        dataloader = DataLoader(dataset, batch_size=self.params['batch_size'], shuffle = False, drop_last=True)
        for idx, batch in enumerate(dataloader):
            batch[0] = batch[0].to(self.device)
            batch[1] = batch[1].to(self.device)
            if idx < num_samples:
                mu, log_var = self.model.encode(batch[0])
                latent_vec = torch.cat((latent_vec, self.model.reparameterize(mu, log_var)))
                labels = torch.cat((labels, batch[1]))
            else:
                break
        return latent_vec, labels

    def train(self, latent_vec, labels):
        # Training SVM
        x = latent_vec.detach()
        y = labels.detach()
        self.svm = self.svm.fit(x, y)

    def test(self, test_dataset):
        # Testing with SVM
        latent_vec = torch.empty(0).to(self.device)
        labels = torch.empty(0).to(self.device)

        self.model.eval()
        dataloader = DataLoader(test_dataset, batch_size=self.params['batch_size'], shuffle=False, drop_last=True)
        loss = torch.nn.CrossEntropyLoss()

        for idx, batch in enumerate(dataloader):
            batch[0] = batch[0].to(self.device)
            batch[1] = batch[1].to(self.device)
            mu, log_var = self.model.encode(batch[0])

            latent_vec = self.model.reparameterize(mu, log_var).detach().tolist()
            labels = batch[1].tolist()

            # preds = torch.as_tensor(self.svm.predict_proba(latent_vec))
            self.accuracy += self.svm.score(latent_vec, labels)
            # self.loss += loss(preds, batch[1])
            self.loss += log_loss(labels, self.svm.predict_proba(latent_vec))

        self.accuracy = self.accuracy / len(dataloader)
        self.loss = self.loss / len(dataloader)
