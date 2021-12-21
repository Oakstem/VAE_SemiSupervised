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
        self.loss = 0
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def gen_latent(self, dataset, num_samples):
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
                latent_vec = self.model.reparameterize(mu, log_var).tolist()
                labels = batch[1].tolist()
                self.svm = self.svm.fit(latent_vec, labels)
            else:
                break
        return latent_vec, labels

    def train(self, latent_vec, labels):
        # Training SVM
        x = latent_vec.tolist()
        y = labels.tolist()
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

            preds = torch.tensor(self.svm.predict_proba(latent_vec))
            print(f"latent type:{type(latent_vec)}, labels type:{type(labels)}")
            self.accuracy += self.svm.score(latent_vec, labels)
            self.loss += loss(preds, torch.tensor(labels, dtype=torch.int64))

        self.accuracy = self.accuracy / len(dataloader)
        self.loss = self.loss / len(dataloader)
