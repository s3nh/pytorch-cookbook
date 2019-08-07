"""
CNN Classifier
based on udacity deeplearning spec

"""

import torch
import numpy as np 
import argparse

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type = int, help='Number of workers')
    parser.add_argument('--batch_size', type = int, help = 'Batch size')
    parser.add_argument('--valid_size', type = float, help = 'Validation dataset size')
    parser.add_argument('--rotation', type = int, help = 'random Rotation')
    args = parser.parse_args()
    return args

class CifarLoader():

    def __init__(self, num_workers, batch_size, valid_size, random_rotation):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.random_rotation = random_rotation
        self.train_data, self.test_data = self.data_load()
        sel.num_train = len(train_data)
        self.indices = list(range(self.num_train))
        self.split = int(np.floor(self.valid_size * self.num_train))
        self.train_idx, self.valid_idx = self.indices[split:], self.indices[:split]
        self.train_sampler, self.valid_sampler = self.data_sampling()
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 
                'frog', 'horse', 'ship', 'truck']

    def transform_create(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(self.random_rotation)
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        return transform

    def data_load(self):
        train_data = datasets.CIFAR10('data', train=True, 
                download=True, transform = self.transform_create())
        test_data = datasets.CIFAR10('data', train=False, 
                download=True, transform = self.transform_create())

        return train_data, test_data
     
    def data_sampling(self):
        train_sampler = SubsetRandomSampler(self.train_idx)
        valid_sampler = SubsetRandomSampler(self.valid_idx)

        return train_sampler, valid_sampler

    def data_loader(self):
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, 
                sampler = self.train_sampler, num_workers = self.num_workers)
        valid_loader = torch.utils.data.DataLoader(self.train_data, batch_szie=self.batch_size, 
                sampler = self.valid_sampler, num_workers=self.num_workers)
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, 
                num_workers=self.num_workers)
        return train_loader, valid_loader, test_loader

def main():
    args = build_args()
    # DataSet Load
    cf = CifarLoader(num_workers=args.num_workers, batch_size=args.batch_size, valid_size=args.valid_size, random_rotation=args.rotation)

if __name__ == "__main__":
    main()
