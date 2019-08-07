"""
Linear autoencoder based on 
MNIST dataset
"""
import torch
import numpy as np 
import argparse
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn 
import torch.nn.functional as F

def build_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args

class DataLoaderAutoencoder():

    def __init__(self, root='data', num_workers=0, batch_size=20):
        self.root = root
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.transforms = self.transform()
        self.train_data, self.test_data = self.load_data()
        self.train_loader, self.test_loader = self.data_loader()

    def transform(self):
        transforms=transforms.ToTensor()
        return transforms

    def load_data(self):
        train_data = datasets.MNIST(root=self.root, train=True, download=True, transform=self.transforms)
        test_data = datasets.MNIST(root=self.root, train=False, download=True, transform=self.transforms)
        return train_data, test_data
    
    def data_loader(self):
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size = self.batch_size, num_workers = self.num_workers)
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size = self.batch_size, num_workers = self.num_workers)
        return train_loader, test_loader 


class Autoencoder(nn.Module):
    def __init__(self, encoding_dim=32):
        super(Autoencoder, self).__init__()
        self.fully_con1 = nn.Linear(28*28, encoding_dim)
        self.fully_con2 = nn.Linear(encoding_dim, 28*28)

    def forward(self, x):
        x = F.relu(self.fully_con1(x))
        x = F.sigmoid(self.fully_con2(x))
        return x

class TrainEncoder():
    def __init__(self, n_epochs=20, model=Autoencoder(32) ):
        super(DataLoaderAutoencoder, self).__init__()
        self.n_epochs = n_epochs
        self.criterion = nn.MSELoss()
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        
    def training_phase(self):
        for epoch in range(1, self.n_epochs):
            train_loss = 0.0
            for data in self.train_loader:

                images, _ = data
                images = images.view(images.size(0), -1)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, images)

                loss.backward()
                self.optimizer.step()
                train_loss += self.loss.item()*images.size(0)

        train_loss = train_loss/len(self.train_loader)
        print('Epoch : {} \t Training Loss: {}'.format(epoch, train_loss))

def main():

    pass 


if __name__ == "__main__":
    main()
