from sklearn import svm
from sklearn.metrics import log_loss
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np


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
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

        train_label_index = []
        valid_label_index = []
        samples_per_class = np.int(num_samples/10)
        for i in range(10):
            train_label_list = dataset.dataset.train_labels.numpy()
            label_index = np.where(train_label_list == i)[0]
            label_subindex = list(label_index[:samples_per_class])
            train_label_index += label_subindex

        trainset_np = dataset.dataset.train_data.numpy()
        trainset_label_np = dataset.dataset.train_labels.numpy()
        train_data_sub = torch.from_numpy(trainset_np[train_label_index])
        train_labels_sub = torch.from_numpy(trainset_label_np[train_label_index])

        trainset_new = subMNIST(root='data', train=True, download=True, k=num_samples)
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
        dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=True)
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
            y_pred = self.svm.predict_proba(latent_vec)
            print(idx)
            if idx == 38:
                stop = 1
            self.loss += log_loss(labels, y_pred)

        self.accuracy = self.accuracy / len(dataloader)
        self.loss = self.loss / len(dataloader)


from torchvision.datasets import FashionMNIST


class subMNIST(FashionMNIST):
    def __init__(self, root, train=True, target_transform=None, download=False, k=3000):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        transforms.Resize(32)
                                        ])
        super(subMNIST, self).__init__(root, train, transform, target_transform, download)
        self.k = k

    def __len__(self):
        if self.train:
            return self.k
        else:
            return 10000

