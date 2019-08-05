"""
CNN Classifier
based on udacity deeplearning spec

"""

import torch
import numpy as np 

## More imports

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


class DataTransformer():

    def __init__(self, num_workers, batch_size, valid_size, random_rotation):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.valid_size = valid_size
        self.random_rotation = random_rotation


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




def main():

    pass


if __name__ == "__main__":
    main()
